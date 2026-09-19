# DeepSeek V4.1 patch series

Base source: `344303947/dsv41-flash-pp5-170hx` at
`9d0f9181756ec3100e302bbf04e313baa3f54bbf`.

The author's CPU Engram offload, exact-size pinned allocation, staged Marlin,
DeepSeek V4.1 model implementation, ShadowSource and SM80 support are already in
that base and are not duplicated here.

1. `0001-deepseek-v41-aux-hidden-state-pp-relay.patch`
   - Declares that `DeepseekV4Model`'s auxiliary hidden-state layout is safe for
     the model runner's existing PP relay.
   - This is the model capability gate required by DSpark with PP > 1.

2. `0002-dspark-pp-last-rank-embedding.patch`
   - Keeps the drafter on the last PP rank.
   - Shares a materialized target embedding/lm_head when available.
   - When the target embedding is a `PPMissingLayer` on the last rank, loads the
     draft embedding tensor directly from the indexed safetensors checkpoint.
   - Fails loudly if the last rank does not own a real target lm_head.
   - Preserves dummy-load behavior without reading a checkpoint.

Each patch applies and compiles independently in series. No disk Engram, Zero
Engram, debug timing, GPU-resident scale, lazy-trigger, or Marlin-option patches
are included.
