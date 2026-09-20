# DeepSeek V4.1 patch series

Base source: `344303947/dsv41-flash-pp5-170hx` at
`d63af5a472dc76b12d7a73d50a5af142844c15d1` (the author's current HEAD).

## Active series

1. `0001-prime-pp-communicators-and-hot-perf-debug.patch`
   - Initializes all persistent PP communication resources immediately after
     `GPUModelRunner` and `PPHandler` construction, before model, CPU Engram,
     and KV allocation.
   - Primes the vLLM/PyNCCL point-to-point communicator, torch
     ProcessGroupNCCL edge communicators, and the sampled/draft-token feedback
     broadcast group.
   - Uses an explicit idempotency flag, so the existing pre-warmup ordering
     check becomes a no-op after successful early initialization.
   - Makes later memory accounting include communicator resources and avoids
     first-use NCCL allocation failures under tight HBM budgets.
   - Adds a disabled-by-default, SIGUSR2-controlled diagnostic tracer. Sampled
     mode uses asynchronous CUDA Events and deferred pinned D2H copies without
     stream synchronization. A bounded detailed mode temporarily selects eager
     dispatch and dynamically installs per-module hooks, then removes them and
     restores normal Graph dispatch without restart. An optional torch-profiler
     window records CPU/dispatch activity (CMP 170HX does not expose CUPTI CUDA
     kernel activities).

The pinned author revision already contains native PP6+DSpark support, including
auxiliary hidden-state relay capability and last-rank target-embedding sharing.
No superseded PP+DSpark patches are retained or applied.

The author's CPU Engram offload, exact-size pinned allocation, staged Marlin,
ShadowSource, Ampere sparse-attention fixes, streamed Engram loading, and PP6
rank-2 memory fix are part of the pinned base. No disk Engram, Zero Engram,
debug timing, GPU-scale, or lazy-trigger patches are included.
