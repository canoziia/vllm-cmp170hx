# GLM-5.3-Flash W4A16 + DFlash2 patch series

Base: the source of `localhost/vllm-backport:deepseek-v4.1-flash`, i.e.
`344303947/dsv41-flash-pp5-170hx@d63af5a472dc76b12d7a73d50a5af142844c15d1`
plus `models/deepseek-v4.1-flash/patches/series`, `patches/vllm/series` and
`models/deepseek-v4.1-flash/patches/adaptive/series`, in that order
(`scripts/apply-deepseek-v41-patches.sh`). Each patch here is a git diff
against the result of the previous one and is applied with
`git apply --check` + `git apply` by `scripts/apply-glm53-dflash2-patches.sh`.

Port source: Morrowmake/vllm-cmp170hx `ampere-glm53` @
`3a2bf16dae8b97f5ff2c7e9bc5809d24545e6340` and
Morrowmake/glm53-flash-cmp170hx-recipe @
`a242b4fc53724bc5216f416fbbfaf7bf318b9f2c`.

Every switch keeps the default it had in the development series; only
`VLLM_GLM5_INDEXER_JIT_WARMUP` (0015) defaults to on, and it is compile-only.
`compose.w4a16-dflash2.yml` turns the rest on. With no GLM env set the image
behaves as the DeepSeek image for every other model.

| # | patch | switch (default) | development no. |
|---|---|---|---|
| 0001 | aux hidden states over PP (`SupportsEagle3`, `stream_mean`) | `VLLM_GLM5_AUX_HIDDEN_TENSOR` (`stream_mean`); active only with a drafter | 0001 |
| 0002 | kpool tail ring sized for the draft depth (correctness) | always | 0002 |
| 0003 | drafter KV groups ride the MLA tensors; needs `--block-size=4608` | always (only with a DFlash drafter) | 0003 |
| 0004 | adaptive DFlash verification depth | `VLLM_GLM5_DFLASH_ADAPTIVE_K` (0) | 0004 |
| 0005 | sm_80 thin-M BF16 GEMM | `VLLM_GLM5_THIN_GEMM` (0) | 0005 |
| 0006 | compiled Marlin MoE decode (`vllm._ampere_marlin_C`) | `VLLM_GLM5_MARLIN_DECODE_CUDA` (0) | 0006 |
| 0007 | DFlash2 fused grouped conv | `VLLM_DFLASH2_FUSED_GROUPED_CONV` (0) | 0008 |
| 0008 | deterministic small-M MoE alignment | `VLLM_GLM5_ROUTER_ALIGN_DECODE` (0) | 00010 |
| 0009 | sparse MLA decode schedule | `VLLM_GLM5_SPARSE_MLA_MM_EXPERIMENTAL` (0) | 0012 |
| 0010 | MoE route v2 (fused top-k + alignment) | `VLLM_GLM5_ROUTE_V2` (0), `_GEMV` (`gate`) | 0013 |
| 0011 | kpool indexer gather-workspace clamp | `VLLM_GLM5_INDEXER_GATHER_CLAMP` (0) | 0016 |
| 0012 | PP 64-head Gluon sparse MLA prefill | `VLLM_GLM5_PP_SPARSE_MLA_PREFILL` (0) | 0017 |
| 0013 | PP split-block Marlin MoE prefill | `VLLM_GLM5_PP_MARLIN_PREFILL` (0) | 0019 |
| 0014 | PP 64-head KDA chunk prefill | `VLLM_GLM5_PP_KDA_PREFILL` (0) | 0020 |
| 0015 | kpool indexer Triton JIT warmup | `VLLM_GLM5_INDEXER_JIT_WARMUP` (**1**) | 0022 |
| 0016 | drafter fc folded into the PP stages | `VLLM_GLM5_PP_FOLD_DRAFT_FC` (0) | 0023 |

Left out of the development series:

| dev no. | patch | why |
|---|---|---|
| 0007 | DFlash context-KV graph | no measured gain; nothing later depends on it |
| 0009 | target mHC v2 decode | no end-to-end gain; only 0018 built on it |
| 0011 | KDA dual projection | no end-to-end gain |
| 0014 | adaptive-k force file | diagnostic only |
| 0015 | draft trace | diagnostic only |
| 0018 | mHC v1 numerics | never validated on a GPU (and needs 0009) |
| 0021 | PP pipeline trace | diagnostic only |

Dropping them needed one regeneration: `0011` (dev 0016) had an `envs.py`
hunk next to the force-file entry of dev 0014; it was rebased without that
entry. Every other patch is the development patch rebased unchanged, except
`0006`, whose loader error text and docstring now point at
`models/glm-5.3-flash/native/ampere_marlin` and the in-image install. The final
tree equals the development tree minus the seven patches above (checked file by
file).

Static checks: `scripts/test-glm53-dflash2-patches.py` (run by the apply
script). CPU check of the whole stack:
`models/glm-5.3-flash/tests/check-dflash2-series.sh`.
