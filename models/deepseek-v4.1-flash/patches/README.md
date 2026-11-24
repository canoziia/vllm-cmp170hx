# DeepSeek V4.1 Flash SM80 Patch Series

Base source: official `vllm-project/vllm` (pinned in `manifests/source.env`).

> [!NOTE]
> The default image build (`SOURCE_MODE=vllm-branch` in
> `scripts/build-deepseek-v41-image.sh`) exports this repository's `vllm` branch
> directly. This directory preserves the auditable patch series used by
> `SOURCE_MODE=patch-upstream` and `scripts/apply-deepseek-v41-patches.sh`.

## Base Series (`patches/series`)

Applied in order by `scripts/apply-deepseek-v41-patches.sh`:

1. `0001-upstream-0e3c8db-compat-and-entrypoints.patch`
   - Preserves SM80 / DeepSeek-V4.1 entrypoints, `vllm.env_override` profile bootstrap,
     and `--sse-keep-alive-interval` CLI compatibility on upstream `vllm`.
2. `0002-common-async-pp-mamba-and-structured-spec-fixes.patch`
   - Ports all 4 shared runtime invariants verified by `scripts/test-vllm-common-patches.py`:
     async-PP Mamba state reclamation, resumed hybrid cache geometry restoration,
     structured-output + speculative batch alignment, and NVFP4 empty-amax safety.
3. `0003-openai-thinking-sse-keepalive-and-1017-encoding.patch`
   - Ports top-level `"thinking": {"type": "enabled" | "disabled"}` OpenAI chat mapping
     and exact checkpoint-`1017` prompt/tool-namespace encoding (`deepseek_v41_encoding.py`).
4. `0004-dsv41-memory-engram-cpu-offload-and-vision-offload.patch`
   - Ports post-KV `expandable_segments` allocator activation (`vllm/dsv41_memory.py`),
     NUMA-aware 264 B packed host Engram offload with OpenMP+AVX2 CPU gather + compact
     H2D DMA (`native/dsv41_engram_cpu`), early PP communicator priming, and zero-GPU-HBM
     CPU-offloaded multimodal vision tower execution (`vl_model.py`).
5. `0005-dsv41-sm80-moe-decode-and-prefill-split.patch`
   - Ports SM80 weight-streaming MXFP4 MoE decode kernels (`native/dsv4_moe`), fused
     SwiGLU activation + top-k sum, single-CTA deterministic MoE block alignment, and
     split-list MXFP4 MoE prefill (`moe_prefill.py`).
6. `0006-dsv41-sm80-dense-mxfp8-thin-woa-and-fast-router.patch`
   - Ports 1x-HBM dense MXFP8 decode GEMMs (`native/dsv41_dense_mxfp8`), post-repack
     zero-overshoot `wq_b + indexer.wq_b` fusion, thin-M grouped `wo_a` GEMM, and
     bitwise SM80 MoE router (`gate_gemv + sqrtsoftplus top-k`).
7. `0007-dsv41-sm80-attention-indexer-swa-yoco-and-dspark.patch`
   - Ports SM80 `nvfp4_ds_mla` sparse-attention decode & 32-head prefill optimizations,
     bounded 64/128 MiB sequence-chunked decode indexer workspace, fast CUDA prefill
     indexer (`native/dsv41_indexer`), ragged SWA C++ metadata (`native/dsv41_swa`),
     prefill AllReduce overlap, and FP8 DSpark draft head / Markov sampling hooks.
8. `0008-dsv41-opt-profiles-and-prefill-sched-helper.patch`
   - Adds `vllm/dsv41_opt_profile.py` (`default`, `verification`, `tep4`,
     `tep4_verification`, `tep4_fused`) and `dsv41_prefill_sched.py` quadratic cost-chunk
     helper.

## Adaptive Verification & Topology-Aware Fusion Series (`patches/adaptive/series`)

Compiled in by default (`ENABLE_ADAPTIVE_VERIFICATION=1`) and inert unless enabled via
runtime config / `VLLM_DSV41_OPT_PROFILE`:

1. `adaptive/0001-dsv41-adaptive-history-verification-and-subwidth-cudagraphs.patch`
   - Request-local measured adaptive verification policy (`dsv41_measured_policy.py`),
     manual `vllm_xargs.spec_k` (`[0, 5]`), live-load cohort packing, and uniform
     sub-width (`k=1..5`) + mixed-`spec_k` `FULL` CUDA graphs with metadata replay.
2. `adaptive/0002-dsv41-topology-fusions-and-tep4-fused-profile.patch`
   - Topology-aware SM80 collective dispatch (`ampere_peer_topology()` auto-detecting
     `"nvlink"` vs `"pcie"` with zero code changes), single-kernel Lamport Markov
     exchange (`markov_symm.py`), in-graph text embedding (`vl_model.py`), TP-local
     greedy speculative verification (`tp_greedy_*` in `native/dsv41_r2`), single-kernel
     `fused_gate_topk_align` (with `is_padding` + `bias_vl` + EP block-8 alignment),
     `w2_splitk_ep` with 256-col low-M sum + fused shared-expert `__hadd2`, C++ tensor-core
     `woa_mxfp8_splitk`, and zero-HBM `dequant_marlin_mxfp8 + cuBLAS` prefill.
