# Qwen3.8 Flash Next NVFP4 on two CMP 170HX GPUs

Reproducible PP2 deployment for `nvidia/Qwen3.8-Flash-Next-NVFP4`, including
process-isolated NVMe PLE offload and the shared patched official LMCache
v0.5.5 payload.

## Immutable inputs

- Author runtime source:
  `https://github.com/344303947/dsv41-flash-pp5-170hx.git`
- Source commit: `d63af5a472dc76b12d7a73d50a5af142844c15d1`
- Base image:
  `docker.io/lazymio/vllm-backport@sha256:0fc33cd7d2be777bf6faaee9b02adf781dc6d0bc44b9a09ce4865089f67e9cba`
- Model revision: `fc694b54fb0174e0913e6adf86691ef85a4ead47`
- Shared LMCache source and image digests: `manifests/lmcache.env`

## Patch layout

Applied in this order:

1. `models/qwen3.8-flash-next-nvfp4/patches/0001-process-isolated-nvme-ple.patch`
   - keeps the 47.68-GiB PLE table disk-backed;
   - performs native parallel `pread` row gathering;
   - owns Python I/O and dynamic H2D scheduling in a dedicated process so CUDA
     Graph capture sees only fixed IPC buffers and stream semaphores.
2. `patches/vllm/0001-async-pp-mamba-state-reclaim.patch`
   - prevents lone-request self-preemption while PP work remains in flight;
   - queues old Mamba source-state indices until safe reclamation.
3. `patches/vllm/0002-mamba-resolved-cache-geometry.patch`
   - restores resumed Mamba state from the resolved per-group block geometry.
4. `models/qwen3.8-flash-next-nvfp4/patches/0002-generated-history-checkpoints-pp2.patch`
   - stores finalized generated-history checkpoints every 128 tokens;
   - preserves uniform MTP verification shapes and caps only accepted output;
   - fences optimistic PP2 work at checkpoint boundaries;
   - performs copy-on-write only in the owning KV cache group;
   - identifies Qwen's dedicated `mtp.layers.*` full-attention group without
     incorrectly applying the speculative trailing-block rule to Mamba groups.
5. Upstream vLLM PR [#57105](https://github.com/vllm-project/vllm/pull/57105),
   applied as `0003-qsa-fixed-logits-workspace.patch`, which reserves the
   worst-case 512 MiB QSA prefill logits workspace once per call and reuses it
   for every internal chunk instead of retaining every increasing allocation
   size in the CUDA caching allocator.
6. The complete shared `patches/lmcache/series`, including MTP align-mode
   speculative-block relocation tracking.
7. Decode optimisation patches `0005`-`0013` (after the QSA patches in
   `series.post-common`), each behind its own default-off switch and enabled
   together by `VLLM_QWEN38_OPT_PROFILE=default`; see
   [Decode optimisation](#decode-optimisation).

The two vLLM Mamba fixes are shared with DeepSeek and therefore live in root
`patches/vllm/`; Qwen's PLE, generated-history, and QSA workspace changes remain
model-specific. The complete runtime tree is reproducible from the pinned clean
source and these ordered patch series.

## Decode optimisation

Every switch defaults to off (switch off = the code path before the patch).
`compose.yml` sets `VLLM_QWEN38_OPT_PROFILE: "default"`, which sets each switch
below to `1` unless that variable is already present, so a single switch can be
overridden by listing it in the compose environment (`"0"`), and
`VLLM_QWEN38_OPT_PROFILE: "off"` disables all of them.

| patch | switch | effect | numerics |
|---|---|---|---|
| `0005-single-kernel-moe-align` | `VLLM_QWEN38_FAST_MOE_ALIGN` | one Triton program replaces the ~45-kernel torch deterministic MoE align (~80-95 us -> 7-22 us per MoE layer at decode, <= 4096 routed entries) | bitwise identical |
| `0006-mtp-draft-head-fp8` | `VLLM_QWEN38_DRAFT_HEAD_FP8` | MTP draft logits from a per-row FP8 copy of lm_head via Marlin W8A16 (0.77 -> 0.45 ms per draft step; +0.59 GiB on PP1) | target unchanged; draft logits FP8 (acceptance only) |
| `0007-thin-m-bf16-gemm` | `VLLM_QWEN38_THIN_GEMM` | split-K thin-M Triton GEMM for the measured BF16 projection cells (<= 32 rows), cuBLAS elsewhere | FP32 accumulate, different summation order, error <= cuBLAS vs FP64 |
| `0008-ple-native-ngram-hash` | `VLLM_QWEN38_PLE_NATIVE_HASH` | PLE offload worker hashes n-grams in `native/ple_pread.c` (~0.5 ms -> ~0.07 ms per step on the PP0 critical path) | bitwise identical |
| `0009-mtp-draft-widths` | `VLLM_QWEN38_VERIFICATION_WIDTHS` | draft k, verify exactly those k: the scheduler decides per scheduled batch how many drafts the MTP drafter generates (`vllm_xargs.spec_k` 0..K pins a request), the drafter runs only k MTP steps (fused draft graph per step count) and the next step verifies exactly the generated drafts (exact uniform target graphs per width) | target unchanged |
| `0009-mtp-draft-widths` | `VLLM_QWEN38_ADAPTIVE_WIDTH` | DeepSeek V4.1 measured verification policy ported to MTP k<=K chooses k per request (at-risk acceptance, 0.95 decay, measured/predicted blend, 3 % gates, +1 width growth, manual first); a batch drafts the width (between its members' smallest and largest choice) with the highest summed relative policy score | target unchanged |
| `0010-ple-no-spec-blocks` | `VLLM_QWEN38_PLE_NO_SPEC_BLOCKS` | the PLE short conv keeps its speculative window in the running block, so it reserves no speculative state blocks (align mode); MTP5 holds 21 instead of 26 blocks per request | none (byte-identical outputs) |
| `0011-gdn-recoverssm` | `VLLM_QWEN38_GDN_RECOVER` | MTP verification reads the committed state and records per-token update operands; after sampling every PP rank replays the accepted prefix (bit-identical state). GDN and PLE reserve no speculative state blocks: MTP5 6 instead of 21 blocks per request, c32 runs without preemption (KV 27 %). Verification uses `native/gdn_mtp_record` (a copy of the fused CUDA MTP kernel with a recording mode, built into the image); without it the Triton path is used (slower) | none (bit-identical greedy output) |
| `0012-fused-shared-gate` | `VLLM_QWEN38_FUSED_SHARED_GATE` | shared-expert `sigmoid(gate) * out` in one Triton kernel | none (bit-identical) |
| `0013-qwen38-opt-profile` | `VLLM_QWEN38_OPT_PROFILE` | the profile switch; registers the switches in `vllm.envs` | - |

`vllm_xargs.spec_k` (0..K) is handled only in Qwen3.8 code
(`vllm/v1/core/sched/qwen38_spec_k.py`), reached from two call sites gated on
the Qwen3.8 architecture (input validation -> HTTP 400, scheduler admission);
the generic request model and `SamplingParams` are unchanged and other models
ignore the key. JSON `true`/`false` arrive as 1/0 after the generic coercion.

GPU checks (run in the image before deployment):
`python3 /opt/qwen-cache-tests/test_decode_opt_kernels_gpu.py [--bench]`;
the width policy CPU check `tests/test_mtp_width_policy.py` runs during the build.
The compose default is MTP5 (`QWEN_MTP_TOKENS=5`, LMCache chunk and prefix
retention 1632, the policy chooses 1..5 drafts). Without `0010`/`0011` MTP5
fits only ~27 concurrent requests in the fixed 16 GiB KV budget (26 blocks per
request: 4 Mamba groups x (1+K) + 2; MTP3: 18 blocks, ~39 requests);
with the default profile (`0010` + `0011`) a request holds 6 blocks and c32
runs without preemption at unchanged speed.
GPU checks for `0011`: `python3 /opt/qwen-cache-tests/test_gdn_recover_gpu.py`
and `python3 /opt/qwen-cache-tests/test_gdn_mtp_record_native.py` (native kernel
vs the image's fused CUDA kernel, bit-identical).
Measured on host-a (2x CMP 170HX, 180 W stock, PP2, same 16 GiB/GPU KV budget
and 1M context), full-batch decode tok/s, mean of the measured rounds, 500
output tokens, fresh cache salt, thinking off. Baseline = previous image
(profile off, MTP3); optimised = default profile with MTP5
(`QWEN_MTP_TOKENS=5`, `LMCACHE_CHUNK_SIZE=1632`,
`QWEN_PREFIX_RETENTION_INTERVAL=1632`, a separate `LMCACHE_L2_PATH`):

| prompt | c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|---|
| counting | 143.9 -> 229.0 | 269.9 -> 446.4 | 462.6 -> 729.0 | 766.4 -> 1156 | 1247 -> 1644 | 2171 -> 2590 |
| code | 135.6 -> 220.3 | 252.3 -> 379.3 | 432.4 -> 633.1 | 733.5 -> 980.3 | 1135 -> 1403 | 1952 -> 2320 |
| prose | 82.3 -> 107.4 | 150.3 -> 199.1 | 251.9 -> 314.3 | 429.1 -> 538.2 | 681 -> 792 | 1081 -> 1189 |

c24: counting 2241, code 1911, prose 999. MTP5 + RecoverSSM fits 32 concurrent
requests with no preemption (6 KV blocks per request). With MTP3
(`QWEN_MTP_TOKENS=3`, chunk/retention 1600) the profile still applies
(c1 counting 181.8, code 174.5, prose 108.3). Acceptance with MTP5: counting 5.4-5.8, code 5.1,
prose 2.0-2.2 tokens/step (the measured width policy verifies ~2.4-3.3 drafts
for prose).

Cold prefill (c1, default profile, MTP5): 7k 1.46 s, 28k 4.24 s, 114k 16.3 s,
456k 78.7 s, 912k 200-286 s (run-to-run).

## Build

From the repository root:

```bash
CONTAINER_ENGINE=podman \
OUTPUT_IMAGE=localhost/vllm-backport:qwen3.8-flash-next-nvfp4 \
bash scripts/build-qwen38-image.sh
```

The build checks out the pinned source, applies the model and shared patch
series, takes the patched LMCache payload from the shared server image (building
that image first if this revision's is not present - see
`patches/lmcache/README.md`), builds the native PLE `pread` helper, and runs the
LMCache, generated-history, MTP-group, and QSA-workspace regression gates. Run the
GPU tests in `/opt/qwen-cache-tests/` before deployment.

## Runtime geometry

```text
GPU devices: 6,7 (`QWEN_GPU_0`, `QWEN_GPU_1` in `.env`)
TP1 x PP2, partition 26,22
max_model_len=1,000,000
max_num_batched_tokens=4,096
max_num_seqs=32
KV=bfloat16, 16 GiB per GPU
MTP=5 by default (QWEN_MTP_TOKENS; 3 restores the previous window)
prefix_match_unit=32
mamba_cache_mode=align
prefix_cache_retention_interval=1,632 (MTP5; 1,600 for MTP3)
CUDA Graph=FULL_AND_PIECEWISE
PLE backend=pread, 48 workers, next-chunk prefetch
LMCache chunk=1,600, L1=64 GiB by default, buffered native-FS L2
```

A 1M configured maximum is not by itself proof that a 1M request is stable.
Retain explicit long-context acceptance tests before advertising that limit.
The QSA workspace fix keeps the prefill logits allocation at a fixed 512 MiB
(`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`) instead of allowing the allocator's
reserved high-water mark to grow with every new context width; it does not
reduce that per-call cap or change indexer results.

## Deploy

```bash
cd models/qwen3.8-flash-next-nvfp4
cp .env.example .env
chmod 600 .env
# Set a private API key and host paths. Never commit .env.

podman compose --podman-run-args=--ipc=host \
  -f compose.yml up -d
podman logs -f qwen3.8-flash-next-nvfp4
```

`QWEN_MTP_TOKENS` changes the model command and may change the resolved KV
block size: QSA's ring is rounded from `indexer_compress_ratio + MTP depth`.
Set `LMCACHE_CHUNK_SIZE` to a multiple of the **resolved** block size for that
depth, and choose a distinct `LMCACHE_L2_PATH` for incompatible cache geometry.
`QWEN_PREFIX_RETENTION_INTERVAL` can track that chunk size. An optional
`LMCACHE_L2_CAPACITY_GB` bounds a new test cache. Defaults are the MTP5 /
1632-token configuration with its own L2 directory (`...-mtp5`); data written
with the MTP3 / 1600-token geometry is not reused. `compose.pd.yml` keeps MTP3 /
1600 (not validated with MTP5). The PP2 generated-history snapshot
retention and scheduler lookahead follow the configured window; verify both
GPU decode state and multi-turn cache hits when raising it. Before switching
an existing service, keep the original image and Compose/.env for rollback.

The default Compose assigns only GPUs 6 and 7 and uses loopback ports:

```text
vLLM API:    8001
LMCache RPC: 5557
LMCache HTTP:18557
```

After `/health` succeeds, issue a real completion request and verify both PP
registrations, actual L2 files, nonzero cached tokens on a repeat, and no
`incomplete read`, degraded transition, or CUDA error.

## Prefill/decode disaggregation (`compose.pd.yml`)

Four GPUs, two roles, and **one LMCache server per role** over a **shared L2
filesystem**:

| role | GPUs | PP2 partition | speculative | port | LMCache server |
|---|---|---|---|---|---|
| prefill | 0,1 | `26,22` | `mtp` (geometry) | 8101 | A `127.0.0.1:5557` |
| decode | 2,3 | `26,22` | `mtp` | 8102 | B `127.0.0.1:5567` |

Both servers use `base_path=/lmcache-l2` -> `$LMCACHE_L2_PATH`, so prefill stores
the prompt KV there and decode finds the same chunk keys and loads them instead
of recomputing. Verified: prefill logs `Stored 1600 tokens`, decode logs
`Retrieved 1600 tokens`, and the decode response reports
`prompt_tokens_details.cached_tokens = 1600`.

### Why the topology looks like this

- **One server per role.** The LMCache MP server binds a single engine's KV
  layout; when a second engine registers against the same server it segfaults in
  `cuda`/`torch` (exit 139). Each role therefore gets its own server, addressed
  through `lmcache.mp.server_urls` (the patched connector prefers it over
  `lmcache.mp.host`/`lmcache.mp.port`). Sharing happens through the L2 tier.
- **Both roles must resolve the same block geometry.** vLLM derives the block
  size from `num_speculative_tokens` (the Qwen QSA ring term), so the prefill
  role declares the same `--speculative-config` as decode. Otherwise the roles
  resolve different block sizes (1568 vs 1600) and the shared L2 cannot hold both
  layouts. Similarly the prefill role uses the same `VLLM_PP_LAYER_PARTITION`
  (`26,22`), because the mamba/attention page geometry depends on it.
- **The decode role keeps host-a's production values** (`--max-num-seqs=32`,
  `--max-num-batched-tokens=4096`, `FULL_AND_PIECEWISE`). What has to agree
  between the two roles is the *block geometry* (previous bullet), not the step
  size. Cold cache misses on the decode role — the decode engine prefilling on
  its own — were exercised at 886 / 1642 / 3370 prompt tokens, and as the very
  first request of a freshly started engine (1500 tokens): every one was served
  normally with `cached_tokens = 0`.
- **Unreproduced incident, recorded without a cause.** During bring-up one cold
  request against a decode engine running `--max-num-batched-tokens=4096` and
  *no* `--max-num-seqs` flag died with `PP intermediate tensor 'hidden_states'
  has 40 rows but this step expects 1536`. It has not reproduced with the
  committed configuration, and no mechanism has been established. An earlier
  revision of this file claimed the step size "must equal the block size" and
  justified it with a guard that the LMCache patch series actually *removes* —
  that was an unjustified generalisation and is not a requirement of this stack.
- **`--ipc=host` is required.** The connector moves KV through CUDA IPC;
  `podman-compose` 1.3.0 silently ignores the `ipc` key, so start the stack with
  `podman compose -f compose.pd.yml --podman-run-args=--ipc=host up -d`, the same
  flag host-a uses. Without it the server never answers `register_kv_caches` and
  the engines time out after 300 s.

### Verified behaviour

| step | request | result |
|---|---|---|
| prefill role, first time | prompt (2263 tok) | server A logs `Stored 1600 tokens`; 4 objects (161 MB) land in the L2 directory |
| decode role, first time | same prompt, after prefill | server B logs `Retrieved 1600 tokens`; `prompt_tokens_details.cached_tokens = 1600` |
| decode role, cold | unseen prompt, no prefill | `cached_tokens = 0` and the request is served (no crash) |

A cross-role check with a nonce-bearing prompt (created seconds before, delivered
only to the prefill role) showed the same result, and the decode answer echoed the
nonce, so the context came from the transferred KV. Evidence logs:
`/srv/pd-validation/`.

### Run

```bash
# from this directory
cp .env.example .env               # set VLLM_API_KEY / machine paths, merge .env.pd.example
podman compose -f compose.pd.yml --podman-run-args=--ipc=host up -d
podman compose -f compose.pd.yml ps
podman compose -f compose.pd.yml down
```

`--podman-run-args=--ipc=host` is required: the connector moves KV over CUDA IPC,
and podman-compose 1.3.0 ignores the compose `ipc` key. host-a starts its
deployments with the same flag.

### Two-step PD request

```bash
KEY=$(grep '^VLLM_API_KEY=' .env | cut -d= -f2)
PF=http://127.0.0.1:8101   # prefill role
DC=http://127.0.0.1:8102   # decode role
BODY='{"model":"nvidia/Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"your prompt"}],"max_tokens":64}'
# 1) prefill the prompt so its KV lands in the shared L2 tier
curl -sS -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "${BODY/\"max_tokens\":64/\"max_tokens\":1}" $PF/v1/chat/completions
# 2) decode the same prompt; the response reports prompt_tokens_details.cached_tokens
curl -sS -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d "$BODY" $DC/v1/chat/completions
```

### Single endpoint in front of the pair (`-pd-router`)

Each pair is fronted by the **vLLM production-stack router** - community-maintained,
actively released (pinned here to `vllm-stack-0.1.13`) - running as an ordinary compose
service on a stock python image, so there is no custom image to build and no Kubernetes:

| model | single endpoint | prefill | decode |
|---|---|---|---|
| qwen | `:9101` | `:8101` | `:8102` |
| deepseek | `:9102` | `:8201` | `:8202` |

`--routing-logic disaggregated_prefill` makes the router perform both hops itself
(request -> prefill with `max_tokens=1` -> decode with the original request) and stream
the decode response back to the caller.

Measured through the router: a 5119-token prompt returned
`prompt_tokens_details.cached_tokens = 4800` while the prefill server logged
`Stored 1600 tokens` twice and the decode server `Retrieved 4800 tokens`.

Things worth knowing:

* the router does **not** forward the caller's `Authorization` header - it builds its
  own from `OPENAI_API_KEY`, which the compose service wires to `VLLM_API_KEY`;
* use `disaggregated_prefill`, **not** `disaggregated_prefill_orchestrated`: the
  orchestrated variant injects `kv_transfer_params` but sends **no** Authorization
  header to the backends (`request.py:833` in 0.1.13 and in main), so it returns 401
  against engines started with `--api-key`. We do not need `kv_transfer_params` because
  our KV moves through the shared LMCache tier;
* other policies shipped by the router: `roundrobin`, `session` (add `--session-key`
  for session affinity), `prefixaware`, `kvaware`, `loadaware`, `priority`;
* `--static-backend-health-checks` exists but is currently incompatible with
  authenticated backends (upstream vllm-project/production-stack issue #631).

The earlier vendored copy of vLLM's example `disagg_proxy_server.py` was removed in
favour of this router.

### One LMCache server shared by both PD roles

The pair runs one LMCache server (chunk 1600, `--l1-size-gb=64`, `fs_native` L2)
instead of one per role, so both engines register against the same layout
registry (keyed by `(model_name, world_size)`; the roles are homogeneous by
construction). The server container therefore has to see **all four** GPUs of
the pair, not just its own role's two.

`patches/lmcache/0008-shared-server-mp-worker-rank.patch` is what makes this
work: the server prefers the client's `mp_worker_rank` layout hint and otherwise
falls back to its own device index, which is only a rank when the server sees a
single engine (indices `0..world_size-1`). With two PP2 engines the second
engine's GPUs sit at indices 2 and 3 for `world_size=2` and registration aborted
with `Cannot derive MP worker rank from registered cache device cuda:2`. The
patch folds the index into `[0, world_size)` and logs the resolution:

```
device=cuda:0 hint=None -> worker_rank=0     engine A rank 0
device=cuda:1 hint=None -> worker_rank=1     engine A rank 1
device=cuda:2 hint=None -> worker_rank=0     engine B rank 0  (2 % 2)
device=cuda:3 hint=None -> worker_rank=1     engine B rank 1  (3 % 2)
```

Note the `hint=None`: the vLLM connector does build the hint
(`vllm_multi_process_adapter._send_register_kv_caches_request` sets
`layout_hints["mp_worker_rank"] = self.worker_id`), but it does not survive to
the server in this build, so the rank fold is the path that actually runs.

Both engine services are deliberately **not** `depends_on` the LMCache service:
restarting LMCache must not restart vLLM.

### Role configuration relative to host-a

Both roles mirror the host-a production instance (PP2/TP1, modelopt, MTP-3,
`--max-num-seqs=32`, `--max-num-batched-tokens=4096`, `kv-cache-memory=16 GiB`,
cudagraph `FULL_AND_PIECEWISE`, prefix cache `--prefix-match-unit=32`,
`--prefix-cache-retention-interval=1600`, `--enable-auto-tool-choice` with the
qwen3_xml/qwen3 parsers, the NCCL/PLE/TORCH_CUDA_ARCH_LIST environment). The
differences that must remain: two engines rather than one, their ports
8101/8102, the engine-local `kv_ip`/`kv_port` (14579/14580 - vLLM defaults both
engines to 14579), `HF_HUB_OFFLINE=1`, and the GPU set.

### Billing: the router merges both hops' usage

The router (`localhost/vllm-router`, built from `patches/router/` - see
`scripts/build-router-image.sh`) hands the client a standard OpenAI body, so the
accounting is computed in the router and the raw per-hop numbers ride along in
the same JSON under a namespaced key.

| field | meaning after the merge |
|---|---|
| `prompt_tokens` | the prompt; both hops must agree (a mismatch is logged and decode reported) |
| `prompt_tokens_details.cached_tokens` | `min(prefill cached, decode cached)` - what was cached **before** the request; the min removes the intra-request prefill->decode handoff |
| `completion_tokens` | the decode hop's generation, untouched |
| `total_tokens` | standard `prompt_tokens + completion_tokens` |
| `router_hops` | custom block: `input`, `cached_read`, `cache_write_prefill_compute` = `(p_prompt - p_cached) + (d_prompt - d_cached)`, plus both hops' raw usage |

Measured on a 36054-token prompt (chunk 1600):

```
cold: P(prompt 36054, cached 0)      D(prompt 36054, cached 33600)
      -> cached_tokens=0      router_hops.cache_write_prefill_compute=38508 (= 36054 + 2454)
warm: P(prompt 36054, cached 33600)  D(prompt 36054, cached 35200)
      -> cached_tokens=33600  router_hops.cache_write_prefill_compute=3308  (= 2454 + 854)
```

### Both response families

The router handles `/v1/chat/completions` and `/v1/responses` (the Responses API
is what pi speaks). The usage is merged in both, in the family's own field names
(`prompt_tokens`/`completion_tokens`/`prompt_tokens_details` vs
`input_tokens`/`output_tokens`/`input_tokens_details`, the latter nested under
`data["response"]["usage"]` in the SSE stream).

The prefill hop must be capped in both families. The Responses API ignores
`max_tokens`, so without an explicit `max_output_tokens=1` the prefill engine
generated the whole answer before the decode hop was asked to start, and the
client waited for both generations. Measured on a 300-token answer:

```
before: prefill engine +297 generated tokens, router prefill TTFT 4.69s,
        client first byte 4.77s / total 9.43s
after:  prefill engine +1 token,             router prefill TTFT 0.11s,
        client first byte 0.19s / total 5.00s
```

The standard fields stay standard on purpose:

* this fork's engines report a non-standard `created_cache_tokens`, and it is
  stripped from `prompt_tokens_details`;
* no `cache_write_tokens` alias is emitted either - a client that derives
  `input = prompt - cached - cache_write` (pi does) would then report
  `input = 0` for every request, because the prefill compute is always >=
  `prompt - cached` (chunked PD recomputes the sub-chunk tail on decode).

So a stock OpenAI client sees `in = prompt - cached`, `cache R = cached`,
`out = completion`, and the prefill-compute line is available only to a billing
layer that reads `router_hops`. `scripts/test-router-patches.py` (8 tests, run
during the image build) pins all of this down.

Bill from the explicit fields: with `prompt_tokens` priced at 0 and
`router_hops.cache_write_prefill_compute` priced non-zero, the bill is
proportional to the prefill compute.
