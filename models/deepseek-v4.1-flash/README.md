# DeepSeek V4.1 Flash on six CMP 170HX GPUs

Reproducible minimal patches and a Podman Compose deployment for
`deepseek-ai/DeepSeek-V4.1-Flash` on six SM80 CMP 170HX GPUs.

Remaining investigation candidates and delivery rules:
[optimisation backlog](docs/OPTIMIZATION-BACKLOG.md). These are not measured gains.

## Source and image

- Author source: `https://github.com/344303947/dsv41-flash-pp5-170hx.git`
- Pinned source revision: `d63af5a472dc76b12d7a73d50a5af142844c15d1`
- Pinned SM80 base image:
  `docker.io/lazymio/vllm-backport@sha256:8094fcbab905a04a480b327f2761255e3d17cd8d39470cac9d76450bbb567f7e`
- Default output image: `localhost/vllm-backport:deepseek-v4.1-flash`

### Image naming (fixed convention - do not invent tags)

| tag | meaning |
|---|---|
| `localhost/vllm-backport:deepseek-v4.1-flash` | moving non-debug deployment tag |
| `localhost/vllm-backport:deepseek-v4.1-flash-debug` | moving debug deployment tag |
| `localhost/vllm-backport:deepseek-v4.1-flash-<shortsha>` | versioned non-debug client |
| `localhost/vllm-backport:deepseek-v4.1-flash-<shortsha>-debug` | versioned debug client |
| `localhost/lmcache-server:latest` | shared LMCache server and payload source for both clients |

The unversioned model tags above are the only moving client tags. Experimental
variants get their tag only for the lifetime of the experiment and are deleted afterwards;
a result worth keeping is described by a commit sha, not by an adjective.

The model checkpoint is mounted read-only and is not modified. The two 94.4-GiB
Engram tables use the author's exact-size pinned CPU offload path. The pinned
author revision includes native PP6+DSpark support. DeepSeek-specific patches
prime persistent PP communicators before model and KV allocation and reuse a
fixed sparse-indexer logits workspace on SM80. The shared vLLM series fixes
asynchronous-PP Mamba reclamation and resumed state geometry. Performance-debug
runtime code is optional and is not included in default images.

## Runtime configuration

```text
TP1 x PP6, partition 7,7,7,7,7,5
DSpark 5, local argmax reduction
max_model_len=1,048,576
max_num_batched_tokens=4096
max_num_seqs=32
KV=fp8_ds_mla, fixed at 6 GiB per rank
CPU Engram offload, prefix caching
NCCL Ring/Simple, P2P enabled, IB disabled
```

No `--compilation-config` is supplied. vLLM automatically derives CUDA Graph
capture sizes from the six-token DSpark target verification width,
`max_num_seqs`, and the platform ceiling. Do not substitute request-count powers
of two: `cudagraph_capture_sizes` is measured in expanded forward tokens, not
HTTP concurrency.

GPU selection is done only through numeric NVIDIA CDI devices. The redundant
`NVIDIA_VISIBLE_DEVICES` variable is intentionally absent;
`CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA's selected-device ordering stable.

## Opt-in verification widths and load-aware PP cohorts

`VLLM_DSV41_OPT_PROFILE=verification` extends the deployed `default` profile;
`default` itself is unchanged. The extension enables:

- `VLLM_DSV41_VERIFICATION=1`: exact small-cohort uniform graphs for query widths
  1–6, least-padding dispatch, and DSpark request-local verification depth;
- `VLLM_DSV41_BALANCED_COHORTS=1`: the established-decode cap uses live load
  rather than the configured capacity. Prefill is uncapped and the PP feedback
  ring retains its original `pp_size` cadence and collective order;
- `VLLM_DSV41_FAST_METADATA=1`: compute a persistent device token-owner map once
  per small decode batch and share it across KV groups (not enabled with DBO);
- `VLLM_DSV41_METADATA_GRAPHS=1`: on SM80, replay metadata preparation for exact
  CPU-known uniform batches, including CPU-selected adaptive widths. Prefill,
  ragged, padded, dummy and GPU-allocated adaptive batches keep the original builder. No new PP messages or communicators are introduced.

Automatic depth is controlled only by
`--speculative-config '{"method":"dspark", "num_speculative_tokens":5,
"enable_adaptive_verification":true}'` (in Compose set
`VLLM_ADAPTIVE_VERIFICATION=true` in `.env`). On DeepSeek V4.1 this selects the
fixed request-local measured policy: discounted prefix-acceptance estimates,
calibrated output/cost predictions blended with attributed real feedback, and
uncertainty-aware gain comparisons. It adds no cohort-coupled decisions or
fixed observation window. `false` keeps fixed width. Manual `spec_k` always wins.
Production has no external algorithm loader. `ENABLE_PERF_DEBUG=1` builds the
trusted hot-policy experiment interface separately; see
[fixed and debug CPU policies](docs/HOT-CPU-POLICY.md).
History recency is configured in the same JSON, e.g.
`"adaptive_verification_decay":0.95` (the default; Compose `.env`:
`VLLM_ADAPTIVE_VERIFICATION_DECAY=0.95`). Each observed verification block
multiplies previous per-position risk/success weights by this value. Valid
finite numbers are `[0,1)`; 0 keeps only the newest block. Values near 1 react
more slowly. Decay is a per-block retention factor, not an acceptance-rate
threshold or a guaranteed number of independent samples.
There is no separate `VLLM_DSV41_HISTORY_POLICY` user switch; the `verification`
profile only enables the four execution optimisations above, never auto-k.

All four optimisation switches default off independently; explicit environment values override
profile defaults. Configure the profile in `.env`, then use the regular build and
Compose deployment. Do not enable a new performance profile without checking the
workload's k5 regression gate.

With the verification profile enabled, OpenAI completions/chat requests may set
`"vllm_xargs": {"spec_k": 2}`. `spec_k` is an integer in `[0, 5]` and limits the
**target-verified draft prefix**, not the target's sampling policy. Every emitted
draft is still accepted/rejected by the target. The configured DSpark drafter
still proposes its full block; this is not a drafter-compute shortcut. Requests
can have different limits; nonuniform mixtures fall back to a compatible graph
or eager execution rather than being falsely labelled uniform. Omission uses
the CPU history policy when `enable_adaptive_verification=true`, otherwise
retains configured full width. Manual `spec_k` is fixed, not an adaptive upper
bound. The field is validated before numeric coercion and
rejected on unsupported models or when the feature is disabled. It is a
request-creation setting, not a hot update of a running request.

The cost table is a local PP6/CMP170HX approximation from uniform-width
measurements, not a general hardware model or oracle.
No mode/prompt labels select k. Short requests may finish before enough feedback;
near ties keep the current width. Structured-output and stale/preempted blocks
are excluded from history updates. **Follow-up:** mixed widths can fall back to
PIECEWISE rather than FULL and lose static metadata replay even at equal rows;
the policy observes real recurrence but does not fix mixed FULL graph capture.
Workload-level measurements can follow different output trajectories across k;
historical replay is not independent throughput or correctness validation.

Diagnostics: `VLLM_DSV41_DECODE_TRACE=1` logs sampled shape histograms. For a torch
profile, set `VLLM_PROFILER=torch` and
`VLLM_TORCH_PROFILE_DIR=/root/.cache/profile`, restart, and call `/start_profile`
then `/stop_profile` once. Restart before a second CUPTI session. These options
are off by default and no measurement RPC is shipped.

## Build

```bash
CONTAINER_ENGINE=podman \
OUTPUT_IMAGE=localhost/vllm-backport:deepseek-v4.1-flash \
bash scripts/build-deepseek-v41-image.sh
```

The default build applies `models/deepseek-v4.1-flash/patches/series` followed by
`patches/vllm/series`, and does **not** include the hot performance-debug
runtime. Build a separate diagnostic image explicitly:

```bash
ENABLE_PERF_DEBUG=1 \
OUTPUT_IMAGE=localhost/vllm-backport:deepseek-v4.1-flash-<sha>-debug \
bash scripts/build-deepseek-v41-image.sh
```

This one build checks out the pinned author revision, applies and verifies the
DeepSeek-specific and shared vLLM series and, only when requested, the optional
debug series. It reuses `lmcache-server:latest` (building it if absent or when
`REBUILD_LMCACHE_IMAGE=1`) and copies its patched LMCache payload alongside the
complete patched `vllm/` tree into a final client image. Build-time checks
validate the Python/native imports and LMCache patches; there is no stage1
client image or separate client-payload step.

To inspect only the source result:

```bash
git clone https://github.com/344303947/dsv41-flash-pp5-170hx.git /tmp/dsv41
cd /tmp/dsv41
git checkout d63af5a472dc76b12d7a73d50a5af142844c15d1
/path/to/this/repo/scripts/apply-deepseek-v41-patches.sh /tmp/dsv41
```

## Deploy

```bash
cd models/deepseek-v4.1-flash
cp .env.example .env
# Set VLLM_API_KEY and adjust model/cache paths if needed.

# One-time cleanup of stale local services from the previous deployment:
sudo bash scripts/disable-legacy-services.sh

podman compose -f compose.yml config
podman compose -f compose.yml up -d
podman logs -f deepseek-v4.1-flash
```

The DAX-backed 286-GiB VM took about 53 minutes to become healthy in one clean
load; plan accordingly before replacing a running container. The container does not
auto-restart: a failed load is expensive and GPU passthrough health must be
verified before trying again.

## Stop safely

Do not reset the VM or kill the container. A hard stop can leave CMP GSP/ACR
state set and make the next guest driver probe fail.

```bash
podman stop deepseek-v4.1-flash
podman rm deepseek-v4.1-flash
```

Nothing in this repository grants a stop timeout any more; Compose carries no
`stop_grace_period`, so an unattended stop uses the engine default. The stack
needs longer than that to unwind CUDA IPC and pinned Engram tables across the API
server, the engine core and six PP workers, so give the stop an explicit timeout
rather than letting it end in a kill.

The reason is driver state, not tidiness: a hard stop can leave CMP GSP/ACR state
set and make the next guest driver probe fail. During the 2026-09-26 scheduler
experiments the container was SIGKILLed several times to skip the wait; the machine
came out clean (no Xid in dmesg, all eight CMP cards enumerable, `RestartCount=0`
on both deployments), but that is not a licence. If restart cycles are too slow,
batch the measurements per boot instead of stopping harder.

## Optional LMCache deployment

`compose.yml` is the deployment: PP6 on six CMP 170HX with the LMCache KV tier (engine + `lmcache` server in one file, connector wired by default). LMCache is not an optional overlay any more; there is no `compose.lmcache.yml`. Add `compose.debug.yml` only with a debug-tagged image.

LMCache no longer comes from the third-party DeepSeek image. The client build
uses the shared LMCache server image as the source of the patched official
v0.5.5 CUDA 13.0 payload:

```bash
ENABLE_PERF_DEBUG=1 \
OUTPUT_IMAGE=localhost/vllm-backport:deepseek-v4.1-flash-<sha>-debug \
  scripts/build-deepseek-v41-image.sh

# After verification, set VLLM_IMAGE in models/deepseek-v4.1-flash/.env to this tag.
# LMCACHE_IMAGE=localhost/lmcache-server:latest selects the shared server.
podman compose --podman-run-args=--ipc=host \
  -f models/deepseek-v4.1-flash/compose.yml \
  -f models/deepseek-v4.1-flash/compose.debug.yml up -d
```

The explicit Podman argument is required: the automatic pod path can ignore
Compose `ipc: host` and expose only 63 MiB `/dev/shm`. The complete official
LMCache manifest, patch rationale, and build details are in
`patches/lmcache/README.md`. Use `models/deepseek-v4.1-flash/compose.debug.yml` only with a debug-enabled
DeepSeek base. Do not promote a build without real store/evict/L2-restore tests.

## Combined optional debug package: DSpark compute toggle

```bash
ENABLE_PERF_DEBUG=1 OUTPUT_IMAGE=localhost/vllm-backport:deepseek-v4.1-flash-<sha>-debug \
  bash scripts/build-deepseek-v41-image.sh
```

Use that image with `models/deepseek-v4.1-flash/compose.yml` plus
`models/deepseek-v4.1-flash/compose.debug.yml`. Both performance
tracing and the DSpark compute toggle are included by this one build flag;
default builds include neither. The control-file environment must be present
at startup to capture both K=0 and K=5 target graphs. Performance tracing stays
disabled until explicitly enabled. DSpark initially stays on.

Write `0` (off) or `1` (on) atomically to `dspark-enabled` in the host cache
mount. The scheduler polls at most once per second; in-flight batches finish
under their original settings. Missing/invalid files retain the current mode
(initially on). No signal or restart is needed after startup.

```bash
printf '0\n' > "$VLLM_CACHE/dspark-enabled.tmp"
mv "$VLLM_CACHE/dspark-enabled.tmp" "$VLLM_CACHE/dspark-enabled"
# Repeat with 1 to enable.
```

Off skips draft backbone/Markov sampling/graph replay, but continues draft
context-KV maintenance so ongoing requests can safely resume drafting. Weights,
aux outputs, fixed-shape PP feedback and draft caches stay resident. This is
not a zero-overhead non-speculative baseline. Requires MRV2 DSpark, DP=1 and
adaptive verification disabled. The combined image has been live-validated with ON→OFF→ON transitions; see
`models/deepseek-v4.1-flash/docs/INVESTIGATION-20260921.md` for throughput ranges and limits.

## Optional hot performance diagnostics

Safety update: `detail` and hot `profile` are disabled. The tracer rejects
graph-changing/module-hook options and ignores `torch_profile_steps` even when
written directly to the control file. Runtime sampling never changes
target/draft graph dispatch. Use `enable` for asynchronous stage timing;
external process sampling can capture CPU stacks. Trace records retain their
original output path/session across delayed flushes.


Default images contain no diagnostic runtime code or hot-path hooks. Build and
deploy a separate image with `ENABLE_PERF_DEBUG=1` when diagnosis is needed.
Within that diagnostic image the tracer is still disabled by default: disabled
workers do not create CUDA events, synchronize streams, copy timing tensors,
read control files, or write logs. Runtime control uses a shared JSON file plus
`SIGUSR2`, so subsequent enable/disable cycles do not require another restart.

Sample asynchronous timings on every tenth **scheduled** step, up to 256
samples on all PP ranks (the real-step filter is in the next image build, not
the older running `73d0be8-debug-eventfix` instance):

```bash
bash models/deepseek-v4.1-flash/scripts/perf-debug-control.sh enable decode-ab 10 256 all
# Run the benchmark, then inspect or explicitly stop early:
bash models/deepseek-v4.1-flash/scripts/perf-debug-control.sh status
bash models/deepseek-v4.1-flash/scripts/perf-debug-control.sh disable
python3 models/deepseek-v4.1-flash/scripts/summarize-perf-debug.py \
  /root/app/deepseek-v41/cache/vllm-perf-debug/steps-decode-ab-rank*.jsonl
```

Captured JSONL includes PP rank, real/padded tokens, graph dispatch mode,
request/cohort sizes, CPU PP waits/enqueues, asynchronous GPU target/sampler/
draft/postprocess timings, feedback-broadcast timings, and per-request accepted
and rejected token counts.

CMP 170HX does not expose CUPTI CUDA kernel activities. `detail`/`profile`
commands are refused: detailed eager tracing can desynchronize PP graph shapes,
and hot CPU profiling reproduced worker stalls. For lower overhead, increase
`sample_every` and compare uninstrumented-before → sparse trace →
uninstrumented-after under the **same actual cohort**. Even asynchronous
CUDA Events can perturb high-concurrency throughput. The control directory is
inside the existing cache mount at `/root/.cache/vllm-perf-debug`.

## Validation gates

After startup, verify:

```bash
podman inspect deepseek-v4.1-flash --format '{{.State.Healthcheck.Status}}'
curl -fsS http://127.0.0.1:8000/health
podman exec deepseek-v4.1-flash nvidia-smi -L
swapon --show
```

KV cache memory is fixed at 6 GiB per rank. `max_num_seqs=32` is an admission
limit, not capacity for 32 one-million-token requests. Record the reported token
capacity and PP2 free HBM for every new image because Graph coverage and model
allocations can change the runtime headroom.

## Decode optimisation: single-kernel deterministic MoE align (patch 0003)

Profiling one DSpark request (PP6, CUDA graphs, node1) showed every MoE layer
spending ~100 us on `VLLM_DETERMINISTIC_MOE_ALIGN`'s torch implementation
(stable argsort, scatter_add, two cumsums, searchsorted, ~25 small kernels),
about 12% of each stage's per-step GPU time. Patch 0003 computes the same
`sorted_ids` / `expert_ids` / `num_tokens_post_pad` in one Triton program
(`vllm/model_executor/layers/fused_moe/fast_det_align.py`).

- Switch `VLLM_DSV4_FAST_DET_MOE_ALIGN` (patch default 0; `compose.yml` sets 1).
- Only without an expert map and for at most 512 routed entries (decode); larger
  batches keep the torch path, which is faster there.
- Correctness: `tests/test_fast_det_align_gpu.py` compares all three outputs with
  the torch path bit for bit (3,368 cases: 384/128/257/8 experts, block 8-64,
  invalid ids, int32/int64, plus 200 CUDA-graph replays with changing ids). The
  MoE GEMM therefore sees the same layout and produces the same numbers.
- Micro-benchmark (graph replay, 384 experts, top-6, block 16): 6 tokens
  103 -> 21 us, 48 tokens 119 -> 66 us.

```bash
podman run --rm --device nvidia.com/gpu=0 --security-opt label=disable \
  -v "$PWD/models/deepseek-v4.1-flash/tests:/t:ro" -w /t --entrypoint python3 \
  localhost/vllm-backport:deepseek-v4.1-flash test_fast_det_align_gpu.py
```

Measured on node1 (PP6, 2 warm-up + 5 timed rounds, medians, 500 tokens).
"before" is the deployed debug image without this patch; "after" is the
rebuilt default image with `compose.yml` as committed, nothing mounted:

| load | before tok/s (steps/s) | after tok/s (steps/s) | accepted/step |
|---|---:|---:|---:|
| c1 counting | 150.2 (25.28) | 164.5 (27.69) | 5.88 / 5.88 |
| c1 code | 127.3 (23.32) | 138.3 (25.35) | 5.41 / 5.41 |
| c1 prose | 58.4 (25.38) | 63.6 (27.65) | 2.29 / 2.29 |
| c8 counting | 719.4 | 819.2 | 5.88 / 5.88 |
| c8 code | 587.7 | 665.7 | 5.37 / 5.38 |
| c8 prose | 190.7 | 267.8 | 2.25 / 2.25 |

- Only the c1 step rate (+8.7-9.5%) is a firm result: c8 runs have wide ranges
  on this deployment (e.g. counting 525-890).
- Cold prefill is unchanged by the patch: on the same rebuilt image, switch
  off vs on (warm, 7k / 27k / 107k tokens) 2033-2053 / 3841-3868 / 4212-4409
  vs 2044-2054 / 3381-3848 / 4072-4304 tok/s. The first requests after a
  restart of a new image are slower (JIT); the long-running debug image
  measured 2150 / 4235 / 4956 in a single run.
- KV pool unchanged: 4,142,306 tokens.
- The engine is not deterministic at T=0 even without this patch (two
  identical greedy requests diverge after a few tokens), so correctness rests
  on the bitwise layout test, not on text comparison.
- `podman-compose` 1.3 does not apply `${VAR:-default}` for variables missing
  from `.env` inside `environment:`, so the switch is written as a literal.

## Decode optimisation: weight-streaming MXFP4 MoE kernels (patch 0004)

Hardware facts that shape every kernel on this box (CMP 170HX, measured):

| | measured |
|---|---|
| HBM pure read | 1,612 GB/s (copy r+w 1,594, write 1,356) |
| FP32 FFMA / FMUL / FADD / FMNMX, bf16 HFMA2, integer LOP3/SHF/PRMT/IMAD | ~64 thread-instr / SM / clk (A100 rate, not throttled) |
| f16 HMUL2 | ~122 / SM / clk |
| bf16 tensor-core mma | A100 rate |
| L2 | 32 MiB (CUDA device attribute) |
| board power limit | 180 W on every card (default 250 W, max 300 W) |

Correction (later measurements): an earlier version of this table listed
FP32/bf16 FMA-class instructions at ~42 / SM / clk and called them
throttled, and the L2 as 40 MB. Clean microbenchmarks on node1/node2 (O2,
P4 lab runs) measure ~64 / SM / clk for FP32, bf16 HFMA2 and integer ops
alike, i.e. no ALU throttling, and the device reports 32 MiB of L2. The real
limit on sustained decode is the 180 W power cap: under load every card sits
at 179-181 W with the SW power-cap throttle active and SM clocks of
~800-1150 MHz, so energy per byte (bytes read + instructions issued per
byte), not instruction rate, decides kernel speed. With the clock allowed
to recover (~1400 MHz) the deployed MXFP4 MoE kernel already reaches
1.17-1.46 TB/s. The earlier ~42 figure was taken under that clock
throttling.

Marlin's MXFP4 MoE GEMM reached only 740-880 GB/s at decode sizes. Patch 0004
(`VLLM_DSV4_MXFP4_DECODE`, patch default 0, `compose.yml` sets 1) runs the two
expert GEMMs of `fused_marlin_moe` for batches of at most 64 tokens with
`native/dsv4_moe/mxfp4_decode.cu`, compiled into the image as
`vllm/_dsv4_moe_C.abi3.so`:

- adapted from the GLM W4A16 weight-streaming decode kernel: tokens are the
  mma N dimension, each warp pulls (expert block, 64-column tile, K slice)
  items from an integer work queue and streams the Marlin-packed weights
  through a per-lane `cp.async` ring; w13 is split four ways over K into
  fixed-order fp32 partials (deterministic, no atomics on data);
- reads the deployed Marlin MXFP4 layout and e8m0 scales directly (no repack,
  prefill keeps Marlin); dequantisation is exact (folded scale 2^(S-1), valid
  for every e8m0 scale <= 128; the checkpoint's range is 115-128, and a layer
  with a larger scale keeps Marlin);
- the activation is the caller's own `activation_func` on the bf16 w13 result
  and the slot sum stays `moe_sum`; only the fp32 accumulation order differs.

Kernel results (`tests/test_mxfp4_decode_gpu.py --bench`, 384 experts, top-6):

| tokens / distinct experts | Marlin | 0004 |
|---|---:|---:|
| 1 / 6 | 196 us | 183 us |
| 6 / 12 | 303 us | 291 us |
| 6 / 18 | 430 us | 384 us |
| 6 / 36 | 797 us | 687 us |
| 48 / 150 | 3,188 us | 2,822 us |

Error against an fp64 reference equals Marlin's (mean 4.3-4.7e-3 relative in
every case); CUDA-graph replay equals eager; repeated calls are bitwise
identical.

Engine (node1 PP6, rebuilt image + 0004 mounted for development, 2 warm-up +
5 timed rounds, medians):

| load | 0003 tok/s (steps/s) | 0003+0004 tok/s (steps/s) | accepted/step |
|---|---:|---:|---:|
| c1 counting | 164.5 (27.69) | 171.6 (28.89) | 5.88 / 5.88 |
| c1 code | 138.3 (25.35) | 144.3 (26.56) | 5.41 / 5.41 |
| c1 prose | 63.6 (27.65) | 65.9 (28.27) | 2.29 / 2.33 |
| c8 counting | 819.2 | 859.7 | 5.88 / 5.88 |
| c8 code | 665.7 | 632.1 | 5.38 / 5.41 |
| c8 prose | 267.8 | 311.8 | 2.25 / 2.29 |

Rebuilt image with 0004, `compose.yml` as committed, nothing mounted (same
method): c1 counting/code/prose 169.0/143.2/66.0 tok/s (steps/s
28.44/26.34/28.30), c8 863.4/636.9/270.8, acceptance unchanged.

A weight-streaming FP8 kernel for the dense (MXFP8 Marlin) projections was
also written and verified (error equal to Marlin's) but was 15-40 % slower
than Marlin, which already reaches 930-1,140 GB/s on those shapes at M=6; it
is not included.

Greedy self-consistency (3 cases x 4 runs x 128 tokens, scored with
`prompt_logprobs` under the same engine): every non-top-1 token is a near tie
(logprob gap <= 0.5).

## Decode optimisation: faster sm80 sparse-attention decode (patch 0005)

The sm80 split-K sparse-attention decode (`_sparse_attn_decode_partial_kernel`,
shared with ROCm) spent ~60 us per layer at c1. It decoded every fp8 K byte
with a 256-entry LUT gather plus an `exp2` per element (FMA-class ALU is
limited on this card), and with 16-head blocks each K tile was decoded four
times (MQA: one K for all 64 heads). Patch 0005
(`VLLM_DSV4_SPARSE_DECODE_FAST`, patch default 0, `compose.yml` sets 1):

- integer fp8 placement (e4m3 byte as bf16 v * 2^-120, times 2^120, times the
  e8m0 scale; NaN codes -> 0; scale byte 0 -> 2^-127): exactly the LUT values;
- 8 warps; 32-head blocks for more than 16 queries;
- the split count is still derived from 16-head blocks, so every head sees
  the same K slices and merge order.

`tests/test_sparse_decode_fast_gpu.py` checks bitwise equality with the switch
off (1-48 queries, padded top-k, NaN K codes, scale byte 0). Kernel time
(synthetic cache, SWA 128 + top-512): 6 queries 127 -> 88 us, 16 queries
307 -> 213 us, 48 queries 854 -> 347 us.

Engine (node1 PP6, 0003+0004 image + 0005 mounted for development, 2 warm-up
+ 5 timed rounds): c1 counting/code/prose 178.0/150.9/69.8 tok/s (steps/s
29.96/28.12/29.93, from 28.44/26.34/28.30), prose acceptance unchanged
(2.33). Greedy self-scoring: only near ties (gap <= 0.5).

Rebuilt image with 0003-0005, `compose.yml` as committed, nothing mounted:
c1 counting/code/prose 179.0/148.4/69.9 tok/s (steps/s 30.13/27.65/30.00),
c8 897.4/710.1/348.5 tok/s, acceptance unchanged. Against the deployed debug
image before this work (150.2/127.3/58.4, steps/s 25.28/23.32/25.38; c8
719.4/587.7/190.7): c1 steps/s +18-19 %.

Evaluated and not included: a tensor-core router gate GEMV (19.4 -> 9.3 us per
call, error at the scalar GEMV's level) changed engine step rate by only
+0.5 % while changing fp32 summation order in routing.

## Decode optimisation round 2 (patches 0006-0008)

Written in parallel by subagents from a shared profile/hardware brief and
verified on node1 by the main session.

| patch | switches (compose sets 1) | what | numerics |
|---|---|---|---|
| 0006 | `VLLM_DSV4_ATTN_DIRECT_OUT`, `VLLM_DSV41_TOPK_RAGGED_FUSED`, `VLLM_DSV41_SWA_RAGGED_INPLACE`, `VLLM_DSV41_INDEXER_Q_LUT_FUSED` (`VLLM_DSV41_TOPK_RAGGED_REUSE` included, not enabled) | small decode kernels around sparse attention / indexer fused, per-layer DtoD copies removed | bitwise (`tests/test_dsv41_small_kernels_gpu.py`) |
| 0007 | `VLLM_DSV4_MHC_V2` | Gluon mHC decode v2 generalised to hidden 5120, M 1-64, warmed before capture | residual bitwise; mixes / layer input fp64 error within 4x of TileLang's 1e-7 level at M <= 16, ~100x lower at M = 48 (`tests/test_mhc_v2_gpu.py`); 24-47 % faster per call |
| 0008 | `VLLM_DSV4_MXFP4_FUSED_ACT`, `VLLM_DSV4_MXFP4_FUSED_SUM` (+`_ORDER=1`), `VLLM_DSV4_CUDA_DET_ALIGN` | split-K sum + activation inside w13, top-k sum inside w2 (<= 16 tokens), single-CTA CUDA align (3-6 us vs 17-89 us) | bitwise (`tests/test_mxfp4_decode_v2_gpu.py`) |

`VLLM_DSV4_MXFP4_DQ=2` (cheaper dequant, exact weights, different fp32 order)
passes its tests but measured slower end to end (c1 counting steps/s 31.45 vs
31.96 with DQ=0) with lower prose acceptance in that run, so it stays off.

Development measurements (node1 PP6, 2 warm-up + 5 timed rounds, medians;
0001-0005 image + 0006-0008 mounted):

| load | 0001-0005 image tok/s (steps/s) | + 0006-0008 tok/s (steps/s) |
|---|---:|---:|
| c1 counting | 179.0 (30.13) | 189.9 (31.96) |
| c1 code | 148.4 (27.65) | 161.0 (29.15) |
| c1 prose | 69.9 (30.00) | 73.9 (31.70) |
| c8 counting / code / prose | 897.4 / 710.1 / 348.5 | 937.0 / 708.6 / 329.8 |

Rebuilt image with 0001-0008, `compose.yml` as committed, nothing mounted:
c1 counting/code/prose 188.9/157.8/73.7 tok/s (steps/s 31.79/29.08/31.61),
c8 878.7/743.3/324.5, KV pool 4,142,306 tokens (unchanged), all in-image GPU
tests pass. Against the deployed debug image before this work (steps/s
25.28/23.32/25.38): c1 steps/s +24-26 %.

Acceptance unchanged (counting 5.88, prose 2.33). Greedy self-scoring under
the engine: only near ties (gap <= 0.5). c8 ranges stay wide on this box.

## fp8 draft head (patch 0009)

`VLLM_DSPARK_DRAFT_HEAD_FP8` (patch default 0, `compose.yml` sets 1): the
DSpark drafter computes its base logits from a per-row fp8 copy of lm_head
(FP8 Marlin, 0.5 ms instead of 1.0 ms at 6 rows); target verification keeps
the bf16 head, so the output distribution is unchanged. Last-stage GPU time
per step 7.74 -> 7.03 ms; acceptance unchanged on the benchmark prompts
(counting 5.88, prose 2.33). Development run: c1 steps/s 31.90/29.16/31.72,
c8 937.1/745.9/379.1 tok/s (c8 ranges wide). `tests/test_draft_head_fp8_gpu.py`.

Rebuilt image with 0001-0009 (compose as committed, nothing mounted): c1
counting/code/prose 191.6/157.3/75.0 tok/s (steps/s 32.26/29.62/32.14),
c8 925.6/750.4/347.6; cold prefill (same warm instance, median of 2)
8k/32k/64k/107k 1919/2874/4000/4206 tok/s.

## Long-context prefill (patches 0011-0012) and thin wo_a (0010)

| patch | switch (compose sets 1) | what |
|---|---|---|
| 0010 | `VLLM_DSV41_THIN_WOA` | grouped wo_a GEMM for small M, short-chain tensor-core Triton (45-47 vs 50 us at M=6), fp64 error <= cuBLAS |
| 0011 | `VLLM_DSV41_MOE_PREFILL_SPLIT` (`VLLM_DSV41_SPARSE_PREFILL_64H` included, off) | MXFP4 MoE prefill split expert lists (27.4 -> 23.0 ms per 4096-token chunk) |
| 0012 | `VLLM_DSV41_INDEXER_PREFILL_FAST` | prefill indexer logits + top-512 as two CUDA kernels (`native/dsv41_indexer`, built into the image): logits bitwise, same top-512 set (sorted by column); 80.2 -> 38.6 ms per 4096-row chunk at 114K KV |

Cold prefill on node1 PP6 (180 W cap, warm instance, unique prompts, actual
prompt length shown, tok/s):

| prompt tokens | 0001-0009 image | + 0011/0012 (development tree) |
|---:|---:|---:|
| 106k | 4572 | 5326 (+16 %) |
| 261k | 3978 | 5114 (+29 %) |
| 523-541k | 3021 | 4432 (+47 %) |
| 719k | 2549 | 4078 (+60 %) |
| 989k | 2013 | 3555 (+77 %; 491 s -> 278 s) |

The tail drop from 106k to 989k goes from -56 % to -33 %. Runs within a pair
agree within ~1 %. Earlier jumpy long-prompt numbers came from the first
requests on a fresh instance (one-time JIT/warm-up), not from steady state.

## Dense MXFP8 decode GEMM (patch 0013)

`VLLM_DSV4_O2_DENSE` (patch default 0, `compose.yml` sets 1): the dense MXFP8
projections (q_b, o, qkv_a, shared expert w13/w2) use our kernels from
`native/dsv4_o2` (built into the image as `vllm/_o2.so`) where a measured
per-shape dispatch table shows them faster than Marlin, Marlin elsewhere:

- M <= 8: warp-level stream-K; M 9-64: CTA-shared activations (cp.async once
  per stage, ldmatrix by all warps), static split-K + fixed-order sum;
  M 65-192: two warps per tile, each dequantising half the weights;
- 236 of 250 scanned (shape, M) points use our kernel, each >= 1.04x Marlin
  (median 1.11-1.54x per shape); layer sequence 1.08-1.29x for M 1-192;
- fp64 error at Marlin's level (mean ratio 1.00000-1.00001, max <= 1.0024),
  not bitwise; deterministic, CUDA-graph safe; `tests/test_o2_dense_gpu.py`.

Engine (node1 PP6, development tree): c1 steps/s +2.6 % on all three loads
(32.18/29.56/32.13 -> 33.01/30.34/32.96 with the M <= 8 kernel), acceptance
and greedy self-scoring unchanged. c8 shows no measurable change: PP6
schedules the c8 requests in micro-batches of 1-2 requests per stage step
(M 6-12), so the larger-M kernels matter only at higher concurrency.

## Optimisation switches (patch 0014 profile)

`compose.yml` enables all optimisation patches with one variable,
`VLLM_DSV41_OPT_PROFILE: "default"` (patch 0014). The profile sets each switch
below to the listed value **only if that variable is not already set**, so a
single optimisation can be turned off by adding it to `environment:` (for
example `VLLM_DSV41_INDEXER_PREFILL_FAST: "0"`). Without the profile every
switch defaults to off (original code path). The profile is applied from
`vllm.env_override`, before any other vLLM import, in every process.

| patch | switch | profile value | what |
|---|---|---|---|
| 0003 | `VLLM_DSV4_FAST_DET_MOE_ALIGN` | 1 | single-kernel deterministic MoE align (bitwise) |
| 0004 | `VLLM_DSV4_MXFP4_DECODE` | 1 | weight-streaming MXFP4 MoE decode (<= 64 tokens) |
| 0005 | `VLLM_DSV4_SPARSE_DECODE_FAST` | 1 | faster sm80 sparse-attention decode (bitwise) |
| 0006 | `VLLM_DSV4_ATTN_DIRECT_OUT` | 1 | attention writes straight into the output buffer |
| 0006 | `VLLM_DSV41_TOPK_RAGGED_FUSED` | 1 | top-k ragged metadata in one kernel |
| 0006 | `VLLM_DSV41_SWA_RAGGED_INPLACE` | 1 | SWA ragged metadata built in place |
| 0006 | `VLLM_DSV41_INDEXER_Q_LUT_FUSED` | 1 | indexer q LUT decode in one kernel |
| 0006 | `VLLM_DSV41_TOPK_RAGGED_REUSE` | (off) | reuse top-k metadata across layers; not enabled |
| 0007 | `VLLM_DSV4_MHC_V2` | 1 | Gluon mHC decode v2 for hidden 5120 |
| 0008 | `VLLM_DSV4_MXFP4_FUSED_ACT` | 1 | split-K sum + activation inside w13 |
| 0008 | `VLLM_DSV4_MXFP4_FUSED_SUM` | 1 | top-k sum inside w2 (<= 16 tokens) |
| 0008 | `VLLM_DSV4_MXFP4_FUSED_SUM_ORDER` | 1 | sum order bitwise equal to `_moe_C.moe_sum` |
| 0008 | `VLLM_DSV4_CUDA_DET_ALIGN` | 1 | single-CTA CUDA deterministic align |
| 0008 | `VLLM_DSV4_MXFP4_DQ` | (0) | cheaper-dequant variants; slower end to end, not enabled |
| 0009 | `VLLM_DSPARK_DRAFT_HEAD_FP8` | 1 | fp8 copy of lm_head for draft logits (target stays bf16) |
| 0010 | `VLLM_DSV41_THIN_WOA` | 1 | thin-M grouped wo_a GEMM (decode) |
| 0011 | `VLLM_DSV41_MOE_PREFILL_SPLIT` | 1 | MXFP4 MoE prefill split-list |
| 0011 | `VLLM_DSV41_SPARSE_PREFILL_64H` | (off) | 64-head sparse prefill tile; no stable gain, not enabled |
| 0012 | `VLLM_DSV41_INDEXER_PREFILL_FAST` | 1 | fast prefill indexer (logits bitwise, same top-512 set) |
| 0013 | `VLLM_DSV4_O2_DENSE` | 1 | dense MXFP8 decode GEMM for M 1-192 (Marlin elsewhere) |
| 0013 | `VLLM_DSV4_O2_LIB` | `vllm/_o2.so` | path of the 0013 kernel library (built into the image) |

Requirements: patch 0003/0008 align switches act only with
`VLLM_DETERMINISTIC_MOE_ALIGN` (on by default in this tree).
