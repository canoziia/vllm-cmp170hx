# Shared vLLM patches

These patches target the pinned author source revision
`d63af5a472dc76b12d7a73d50a5af142844c15d1` and are applied to every model
build that uses that runtime.

1. `0001-async-pp-mamba-state-reclaim.patch`
   - avoids self-preempting a lone request while asynchronous pipeline work is
     still in flight;
   - retains every pending Mamba source-state index until its processed token
     boundary makes reclamation safe.

2. `0002-mamba-resolved-cache-geometry.patch`
   - binds the resolved `KVCacheConfig` before requests are admitted;
   - derives align-mode resumed-state columns from `MambaSpec.block_size`
     instead of the smallest global cache group.

Both patches were validated in the DeepSeek default/debug series and in the
Qwen PP2/MTP3 series. Model-specific code such as Qwen's process-isolated PLE
NVMe backend remains under `models/qwen3.8-flash-next/patches/`.
