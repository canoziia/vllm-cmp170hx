# DeepSeek V4.1 patch series

Base source: `344303947/dsv41-flash-pp5-170hx` at
`d63af5a472dc76b12d7a73d50a5af142844c15d1` (the author's current HEAD).

## Default active series

`models/deepseek-v41/patches/series` contains exactly one patch:

1. `0001-prime-pp-communicators-before-model-load.patch`
   - Initializes persistent PP communication resources immediately after
     `GPUModelRunner` and `PPHandler` construction, before model, CPU Engram,
     and KV allocation.
   - Primes the vLLM/PyNCCL point-to-point communicator, torch
     ProcessGroupNCCL edge communicators, and the sampled/draft-token feedback
     broadcast group.
   - Uses an explicit idempotency flag, so the existing pre-warmup ordering
     check becomes a no-op after successful early initialization.
   - Makes later memory accounting include communicator resources and avoids
     first-use NCCL allocation failures under tight HBM budgets.

Default builds include no performance-debug runtime code.

## Optional diagnostic series

`ENABLE_PERF_DEBUG=1` additionally applies:

1. `optional/0002-hot-perf-debug.patch`
   - Includes both the DSpark compute toggle and a disabled-by-default,
     SIGUSR2-controlled diagnostic tracer. No separate hot-DSpark build flag.
   - The toggle uses a scheduler control file, per-batch K, K0/K5 target graphs,
     and context-only draft updates while off to preserve restart-free re-enable.
   - Sampled mode uses asynchronous CUDA Events and deferred pinned D2H copies
     without stream synchronization.
   - Detailed eager mode and hot torch-profiler mode are disabled: the former
     desynchronized PP shapes, and the latter reproduced PP2 worker stalls
     under the small-expert fixture. External process sampling is available;
     CMP 170HX does not expose CUPTI CUDA kernel activities.
   - Runtime sampling stays on the production Graph dispatch/padding path.

The legacy debug image `73d0be8-debug-eventfix` used the **image-layer-only**
`models/deepseek-v41/patches/optional/0003-lmcache-event-capability-cache.patch` against the third-party
base image's bundled LMCache. Keep it only to reproduce that historical
DeepSeek base. New LMCache deployments use the digest-pinned official package
and the auditable common series under `patches/lmcache/`; its canonical equivalent is
`patches/lmcache/0003-event-capability-cache.patch`. Do not apply both to the
same package.

The pinned author revision already contains native PP6+DSpark support, including
auxiliary hidden-state relay capability and last-rank target-embedding sharing.
No superseded PP+DSpark patches are retained or applied.

The author's CPU Engram offload, exact-size pinned allocation, staged Marlin,
ShadowSource, Ampere sparse-attention fixes, streamed Engram loading, and PP6
rank-2 memory fix are part of the pinned base. No disk Engram, Zero Engram,
GPU-scale, or lazy-trigger patches are included.
