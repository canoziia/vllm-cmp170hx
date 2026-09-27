# DeepSeek V4.1 Flash on six CMP 170HX GPUs

Reproducible minimal patches and a Podman Compose deployment for
`deepseek-ai/DeepSeek-V4.1-Flash` on six SM80 CMP 170HX GPUs.

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
