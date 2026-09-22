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
4. The complete shared `patches/lmcache/series`, including MTP align-mode
   speculative-block relocation tracking.

The two vLLM Mamba fixes are shared with DeepSeek and therefore live in root
`patches/vllm/`; Qwen's PLE implementation remains model-specific. Applying
these series to the pinned clean source reproduces all 18 `vllm/` files changed
by the authoritative `qwen38-cmp170hx-final` tree byte-for-byte.

## Build

From the repository root:

```bash
CONTAINER_ENGINE=podman \
OUTPUT_IMAGE=localhost/vllm-backport:qwen38-cmp170hx-final \
bash scripts/build-qwen38-image.sh
```

The build performs a fresh checkout of the pinned source, applies the model and
shared series, compiles Python files, extracts the official LMCache payload by
immutable digest, applies the shared LMCache series, builds the native PLE
`pread` helper, and runs LMCache regression gates.

## Runtime geometry

```text
GPU devices: 6,7
TP1 x PP2, partition 26,22
max_model_len=1,000,000
max_num_batched_tokens=4,096
max_num_seqs=64
KV=bfloat16, 16.5 GiB per GPU
MTP=3
mamba_cache_mode=align
prefix_cache_retention_interval=1,600
CUDA Graph=FULL_AND_PIECEWISE
PLE backend=pread, 48 workers, next-chunk prefetch
LMCache chunk=1,600, L1=4 GiB, buffered native-FS L2
```

A 1M configured maximum is not by itself proof that a 1M request is stable.
Retain explicit long-context acceptance tests before advertising that limit.

## Deploy

```bash
cd models/qwen3.8-flash-next
cp .env.example .env
chmod 600 .env
# Set a private API key and host paths. Never commit .env.

podman compose --podman-run-args=--ipc=host \
  -f compose.yml up -d
podman logs -f qwen38-author-nvfp4
```

The default Compose assigns only GPUs 6 and 7 and uses loopback ports:

```text
vLLM API:    8001
LMCache RPC: 5557
LMCache HTTP:18557
```

After `/health` succeeds, issue a real completion request and verify both PP
registrations, actual L2 files, nonzero cached tokens on a repeat, and no
`incomplete read`, degraded transition, or CUDA error.
