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

The two vLLM Mamba fixes are shared with DeepSeek and therefore live in root
`patches/vllm/`; Qwen's PLE, generated-history, and QSA workspace changes remain
model-specific. The complete runtime tree is reproducible from the pinned clean
source and these ordered patch series.

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
GPU devices: 6,7
TP1 x PP2, partition 26,22
max_model_len=1,000,000
max_num_batched_tokens=4,096
max_num_seqs=32
KV=bfloat16, 16 GiB per GPU
MTP=3 by default; set QWEN_MTP_TOKENS=5 to test a longer draft window
prefix_match_unit=32
mamba_cache_mode=align
prefix_cache_retention_interval=1,600
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
`LMCACHE_L2_CAPACITY_GB` bounds a new test cache. Defaults preserve the
MTP3/1600-token production configuration. The PP2 generated-history snapshot
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
- **The decode role keeps node1's production values** (`--max-num-seqs=32`,
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
  flag node1 uses. Without it the server never answers `register_kv_caches` and
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
`/root/app/pd-validation/`.

### Run

```bash
# from this directory
cp .env.example .env               # set VLLM_API_KEY / machine paths, merge .env.pd.example
podman compose -f compose.pd.yml --podman-run-args=--ipc=host up -d
podman compose -f compose.pd.yml ps
podman compose -f compose.pd.yml down
```

`--podman-run-args=--ipc=host` is required: the connector moves KV over CUDA IPC,
and podman-compose 1.3.0 ignores the compose `ipc` key. node1 starts its
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
