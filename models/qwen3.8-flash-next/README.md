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

1. `models/qwen3.8-flash-next/patches/0001-process-isolated-nvme-ple.patch`
   - keeps the 47.68-GiB PLE table disk-backed;
   - performs native parallel `pread` row gathering;
   - owns Python I/O and dynamic H2D scheduling in a dedicated process so CUDA
     Graph capture sees only fixed IPC buffers and stream semaphores.
2. `patches/vllm/0001-async-pp-mamba-state-reclaim.patch`
   - prevents lone-request self-preemption while PP work remains in flight;
   - queues old Mamba source-state indices until safe reclamation.
3. `patches/vllm/0002-mamba-resolved-cache-geometry.patch`
   - restores resumed Mamba state from the resolved per-group block geometry.
4. `models/qwen3.8-flash-next/patches/0002-generated-history-checkpoints-pp2.patch`
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
OUTPUT_IMAGE=localhost/vllm-backport:qwen38-flash-next-nvfp4 \
bash scripts/build-qwen38-image.sh
```

The build checks out the pinned source, applies the model and shared patch
series, compiles Python files, extracts the official LMCache payload by digest,
applies the shared LMCache series, builds the native PLE `pread` helper, and
runs the LMCache, generated-history, MTP-group, and QSA-workspace regression
gates. Run the GPU tests in `/opt/qwen-cache-tests/` before deployment.

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
cd models/qwen3.8-flash-next
cp .env.example .env
chmod 600 .env
# Set a private API key and host paths. Never commit .env.

podman compose --podman-run-args=--ipc=host \
  -f compose.yml up -d
podman logs -f qwen3.8-flash-next
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
