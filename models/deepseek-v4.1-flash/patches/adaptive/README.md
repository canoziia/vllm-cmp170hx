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

9. `0009-dsv41-3regime-ring-lamport-engram-dot-and-moe-dynks.patch`
   - **3-Regime Push-Lamport AllReduce (`OneShot 1..2` → `TwoShot 3..15` → `Ring 16..192`)**:
     Adds `OneShotAllReducePostCollapse` / `OneShotAllReduceFinalizePostCollapse` (`M=1..2`,
     `VLLM_DSV41_DEFER_FINALIZE_MIN_TOKENS=1`, `VLLM_DSV41_FUSED_W2_AR=0`) and
     `RingAllReduce` / `RingAllReducePostCollapse` / `RingAllReduceFinalizePostCollapse`
     (`M=16..192`, deadlock-safe `ctas = min(rows, self.ctas)`, Hop-0 PCIe overlap with
     `PostCollapse.load_row`, `st.relaxed.sys.global.v4.u32` + deferred sentinel reset,
     `9.55 MiB/GPU` total mailbox). Raises PCIe `default_ampere_ar_capacity("pcie") = 192`
     and `default_ampere_ar_ring_min_rows("pcie") = 16` (`0` on NVLink).
   - **Engram Communication Overhaul (100% lossless)**:
     Replaces `[M, 25600]` `AllGather + slice` in prefill `_engram_local` with
     `dist.all_to_all_single` (`1/4` bytes, `-118 MiB` peak VRAM) + Copy-Engine `dma.all_gather`
     for `embed`; adds 5×-bandwidth-reduced Engram Local Partial-Dot reduction (`engram.py`,
     `M >= 160`, `VLLM_DSV41_ENGRAM_LOCAL_DOT=1`, all-reducing only `[M, 8]` FP32 scalars and
     broadcasting only `V` `[M, 5120]`). Also retains Copy-Engine `dma.all_reduce_inplace` +
     `VLLM_DSV41_PREFILL_AR_SPLIT_MOE` inside `_run_layers_prefill_ar_overlap` for future
     `M >= 4096` / `TP >= 6,8` scaling.
   - **Pure TP4 `NI=576` MoE + Router Tuning**:
     Pairs with `mxfp4_dec2_gemm_dynks` (`c10`, GPU-side CUDA-Graph-safe dynamic `nblk` Split-K
     switching `KS=4` when `nblk < 130` vs `KS=2` when `nblk >= 130` for `M >= 49`), routes
     `M = 4..6` to occ=5 `c0`, fixes `W2` cfg (`c4/c5/c2` across `M=1/2/>=3`), bypasses
     `w2_splitk_ep` for `NI=576 M > 8`, and disables `use_sentinel` on `E == 384`.

10. `0010-dsv41-online-self-calibrating-cost-model.patch`
    - **Online Self-Calibrating Step-Cost Model (`OnlineCostModel`)**:
      Adds a two-level CPU-only (`~5 us/step`, `VLLM_DSV41_ONLINE_COST_MODEL=1`) self-calibrating
      step-latency model in `dsv41_measured_policy.py` wired into `FixedHistoryPolicy`
      (`dsv41_fixed_policy.py`) with zero changes to `scheduler.py`:
      - Level 1: non-collinear 3-parameter physics-informed Bayesian WLS
        `T(M_eff) = theta_0 + theta_1 * M_eff + theta_2 * sqrt(M_eff)` with CUDA-graph-aware
        effective token count `M_eff = 0.75 * M + 0.25 * M_cg(M)`, bounded per-cell weights
        (eliminating covariance windup), Huber outlier clipping, and gentle aging to track
        live power-limit (`180W <-> 250W`), thermal, interconnect, and context-length shifts.
      - Level 2: local Empirical-Bayes `(B, M)` cell refinement over contiguous pure-decode
        steps (natively supporting mixed-`k` heterogeneous batches via `M = sum(1 + k_r)`),
        clamped to provable monotonic `[5, 1, 5]` trim-and-recovery ratio bounds.
    - Updates `COST_MS_TP4` prior table from full 30-cell 4×80G measurements under the new
      3-regime Ring Lamport + dynks + Engram Local Partial-Dot stack.

15. `0015-dsv41-shared-engram-and-yoco-prefill.patch`
    - **Cross-Container Zero-Copy Shared Engram (`VLLM_DSV41_ENGRAM_SHM_DIR`,
      `VLLM_DSV41_ENGRAM_SHM_REGISTER`)**:
      - TP4 Decode (`REGISTER=1`) maps disjoint 4KiB-aligned host `tmpfs` shards via
        `cudaHostRegister(cudaHostRegisterMapped)` for native GPU UVA lookup (zero extra RAM).
      - TP2 Prefill (`REGISTER=0`) maps the same host `tmpfs` shards `PROT_READ` without CUDA
        registration and runs multi-shard C++ CPU gather (`engram_cpu_gather_bf16_multi` /
        `engram_cpu_gather_fp8_multi`) into tiny pinned staging buffers + async H2D DMA.
    - **YOCO Encoder-Only Prefill Mode (`VLLM_DSV41_YOCO_PREFILL_ONLY=1`)**:
      Loads `layers 0..19` full + `layer 20` (`kv_write_only=True`, skipping `layer 20` MoE +
      `layers 21..39` + `norm` + `lm_head`, ~70 GiB/GPU weights on TP2), writes the exact 8
      global MLA+Indexer KV layers to shared LMCache MP Server (`chunk_size=1024`), and returns
      dummy logits (`1 MB`).

16. `0016-dsv41-fast-streaming-weight-load-and-engram-mlock.patch`
    - **Streaming Safetensors Load & $O(1)$ Expert Mapping**:
      Streams `language_model.model.*` shard-by-shard in disk order in `vl_model.py` (replacing
      full-checkpoint lexicographical `sorted(...)`), skips unused shards via
      `is_unused_checkpoint_weight` (`vl_model.py` & `dspark.py`), and indexes `expert_mapping`
      by `weight_name` for $O(1)$ lookup in `model.py` and `dspark.py`.
    - **`mlock()` for GUP-Pinned Engram SHM**:
      Moves `cudaHostRegister`-pinned `tmpfs` pages out of `Active/Inactive(anon)` into
      `Unevictable` LRU so Linux `kswapd`/direct-reclaim never spins scanning 188.8 GiB of
      un-evictable Engram pages during model loading.
