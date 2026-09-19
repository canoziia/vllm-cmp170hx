# DeepSeek V4.1 patch series

Base source: `344303947/dsv41-flash-pp5-170hx` at
`d63af5a472dc76b12d7a73d50a5af142844c15d1`.

## Active series

1. `0001-prime-pp-communicators-before-kv-profile.patch`
   - Initializes both lazy two-rank PP communicators (vLLM PyNCCL and torch ProcessGroupNCCL) before KV memory
     profiling, so their persistent buffers reduce the calculated KV budget.
   - Retains the existing pre-warmup call as an idempotent ordering check.
   - Fixes a reproduced startup OOM where communicator initialization happened
     after KV allocation at both 0.94 and 0.90 utilization.

The pinned author revision already contains the other two PP+DSpark features:

- `DeepseekV4Model.supports_aux_hidden_states_over_pp = True`;
- `spec_decode_needs_target_embed(vllm_config)`, which makes the last PP rank
  own and load the target embedding used by DSpark.

## Upstreamed history

`patches/upstreamed/` retains the old independent patches for audit only:

1. `0001-deepseek-v41-aux-hidden-state-pp-relay.patch`
2. `0002-dspark-pp-last-rank-embedding.patch`

They are not applied to the current source. The author's implementation loads
the last-rank embedding through the normal model loader, superseding the older
out-of-tree `safe_open` helper.

The author's CPU Engram offload, exact-size pinned allocation, staged Marlin,
ShadowSource, Ampere sparse-attention fixes, streamed Engram loading and PP6
rank-2 memory fix are also part of the pinned base. No disk Engram, Zero Engram,
debug timing, GPU-scale or lazy-trigger patches are included.
