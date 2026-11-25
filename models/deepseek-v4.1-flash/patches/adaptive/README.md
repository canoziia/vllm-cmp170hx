# Adaptive Verification & Topology-Aware Fusion Series

Base source: official `vllm-project/vllm` (pinned in `manifests/vllm.env`),
already integrated into this repository's `vllm` branch for default
`SOURCE_MODE=vllm-branch` builds.

When rebuilding from upstream via `SOURCE_MODE=patch-upstream`,
`patches/adaptive/series` is applied by `scripts/apply-deepseek-v41-patches.sh` by
default (`ENABLE_ADAPTIVE_VERIFICATION=1`). All runtime features stay inert unless
explicitly enabled via `--speculative-config` and `VLLM_DSV41_OPT_PROFILE`.

## Patches

1. `0001-dsv41-adaptive-history-verification-and-subwidth-cudagraphs.patch`
   - Request-local measured verification width selection (`dsv41_measured_policy.py`),
     enabled per deployment via:
     ```json
     {"method":"dspark","num_speculative_tokens":5,"use_local_argmax_reduction":true,"enable_adaptive_verification":true,"adaptive_verification_decay":0.95}
     ```
   - Per-request manual override via OpenAI `"vllm_xargs": {"spec_k": k}` (`k` in `0..5`),
     validated in `dsv41_spec_k.py` (DeepSeek V4.1 DSpark only; ignored on other models).
   - Uniform sub-width `FULL` CUDA graph capture (`VLLM_DSV41_SUBWIDTH_GRAPHS`) alongside
     varlen buckets, persistent device token-owner metadata sharing (`VLLM_DSV41_FAST_METADATA`),
     and metadata graph replay (`VLLM_DSV41_METADATA_GRAPHS`).

2. `0002-dsv41-topology-fusions-and-tep4-fused-profile.patch`
   - **Automatic NVLink vs PCIe P2P Topology Detection (`ampere_peer_topology()`)**:
     Queries `cudaDevP2PAttrPerformanceRank` / NVML NVLink status at worker startup:
     - On **NVLink** (`"nvlink"`): automatically unlocks full `MAX_ROWS = 192` two-shot
       CuTe Lamport AllReduce and column AllGather across all batch sizes.
     - On **PCIe Gen4** (`"pcie"`): bounds two-shot Lamport AllReduce to `<= 60` rows
       (handoff to full-duplex Ring NCCL above 60 rows) and uses single-kernel Lamport
       symmetric-memory exchange (`markov_symm.py`) for the 5 sequential DSpark Markov steps.
     - Zero `.py` edits are required when migrating between PCIe and NVLink SM80 hosts.
   - **Native `dsv41_r2` CUDA Fusions (`VLLM_DSV41_OPT_PROFILE=tep4_fused`)**:
     - `fused_gate_topk_align` & `dsv4_topk_v2`: fuses `gate_gemv + sqrtsoftplus top-k +
       deterministic EP block-8 align` (including `bias_vl` image sentinel routing and
       `VLLM_MOE_SKIP_PADDING` `is_padding` masks) into one CUDA launch across both
       backbone (`E=384, topk=6`) and DSpark (`E=128, topk=3`) layers.
     - `w2_splitk_ep`: split-K MXFP4 W2 decode GEMM with 256-col small-`M` reduction and
       fused shared-expert `__hadd2` epilogue (`VLLM_DSV41_FUSED_SHARED_ADD=1`).
     - `woa_mxfp8_splitk`: single-kernel C++ Tensor-Core MXFP8 `wo_a` batched GEMV.
     - `tp_greedy_local_argmax` / `tp_greedy_spec_verify`: TP-local greedy target
       verification (`32 B/row` stats exchange instead of `258 KB/row` full-vocab AllGather).
     - `dequant_marlin_mxfp8`: exact E4M3 + E8M0 (`0..254`) -> BF16 fast dequant into a
       40 MiB reusable scratch buffer feeding cuBLAS `torch.mm` for `M >= 768` prefill.
     - `VLLM_DSV41_IN_GRAPH_TEXT_EMBED=1`: keeps pure-text `embed_input_ids` inside
       `FULL_DECODE_ONLY` CUDA graphs on multimodal `DeepseekV41ForCausalLM` while
       preserving CPU-offloaded vision execution when multimodal embeddings are present.

3. `0003-dsv41-auto-profile-by-model-and-parallelism.patch`
   - Adds `VLLM_DSV41_OPT_PROFILE=auto` (auto-routes DeepSeek-V4.1 by parallelism
     while leaving non-DeepSeek models untouched).

4. `0004-dsv41-tp-local-greedy-masked-token-ids-compat.patch`
   - Guards `masked_token_ids` access in `_try_tp_local_greedy_sample` across upstream builds.

5. `0005-dsv41-require-pcie-switch-topology-for-lamport.patch`
   - Requires NVML switch-connected PCIe (`PIX`/`PXB`) before auto-enabling CuTe Lamport
     P2P and `NCCL_PROTO=Simple`, avoiding cross-Root-Complex BAR1 congestion.

6. `0006-dsv41-pcie-dma-ring-and-row-shard-prefill.patch`
   - **Native Copy-Engine Ring DMA (`dsv41_pcie_dma` / `pcie_dma_ring.py`)**:
     Dedicated 0-offset `cudaMalloc` IPC ring (`ce_reduce_scatter`, `ce_all_gather`,
     in-place `ce_all_reduce_inplace`) running lossless BF16 transfers on Copy Engines
     (zero SM contention, +20–32% vs PyNCCL on PCIe Gen4 x16, true overlap with GEMM/MoE).
   - **Lossless BF16 Row-Sharded Prefill (`_run_layers_prefill_row_shard`)**:
     Keeps mHC residual streams `[S, 4, H]`, mixes, and Engram post-updates sharded across
     TP ranks (`1/W` memory & compute) with zero `torch.cat` scratch spikes; eliminates
     long-context prefill OOM on 4×80G and conditionally enables 2-microbatch MoE when
     `M >= 4096` or `EP >= 6`.
   - Tightens `try_gate_gemv` upper bound from `M <= 8` to `M <= 6` (cuBLAS wins at `M >= 8`).

7. `0007-dsv41-pcie-dma-k3-triple-buffer.patch`
   - Upgrades the Copy-Engine Ring DMA slab from `2 * (W - 1)` slots (`45 MiB/rank` on TP4,
     `100–140 MiB` on TP6/8) to a constant `K = 3` triple-buffered circular ring (`22.5 MiB/rank`
     on TP4, `30 MiB` on TP6/8) using 64-bit bidirectional ACK flow control (zero pipeline bubbles,
     reuses the open peer IPC handle on `world == 2`).

8. `0008-dsv41-pure-tp4-default-and-mxfp4-ni576.patch`
   - Makes **Pure TP4 (`TP=4, EP=1`, `NI=576`, `E=384`)** the default 4×80G profile (`tp4_fused`,
     auto-selected by `VLLM_DSV41_OPT_PROFILE=auto` when `-tp 4` without `--enable-expert-parallel`;
     legacy `TEP4` remains supported as `tep4_fused` when `--enable-expert-parallel` is passed).
   - Extends `dsv4_mxfp4_decode.py`, `marlin_moe.py`, and `moe_prefill.py` for `NI=576` (paired with
     `mxfp4_decode_v2.cu` `NI=576` W13/W2 + unweighted W2 for `DeferringMarlinExperts`, and
     `dsv41_r2.cu` `w2_splitk_ep` `K=576` `KS=1/2/3` + `fused_gate_topk_align` `local_e <= 384`).
   - Adds `VLLM_DSV41_DEFER_FINALIZE_MIN_TOKENS=3` (`config.py`) so `M=1..2` keeps 1-hop OneShot
     PCIe AllReduce + `w2_splitk_ep` `KS=3`, while `M=3..60` fuses unweighted W2 into
     `TwoShotAllReduceFinalizePostCollapse`.
   - Adds Pure-TP4 calibrated adaptive step-cost curve `COST_MS_TP4` (`VLLM_DSV41_COST_MS_FUSED=tp4`)
     in `dsv41_measured_policy.py` from full 30-cell 4×80G measurements while preserving strict
     `[5, 1, 5]` trim-and-recovery invariants across `c=1..32`.
