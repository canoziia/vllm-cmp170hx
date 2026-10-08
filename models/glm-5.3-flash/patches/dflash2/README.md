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

Every switch keeps the default it had in the development series, with one
exception: `VLLM_GLM5_ROUTE_V2_FIRST` (0023) defaulted to 1 under route v2 in
development and defaults to 0 here (whole-machine runs were slower with 1).
Only `VLLM_GLM5_INDEXER_JIT_WARMUP` (0015) defaults to on, and it is
compile-only.
`VLLM_GLM53_OPT_PROFILE=default` (0034) turns the rest on (0010 in `tc` mode,
see below); both W4A16 composes set it. With no GLM env set the image behaves
as the DeepSeek image for every other model.

| # | patch | switch (default) | development no. |
|---|---|---|---|
| 0001 | aux hidden states over PP (`SupportsEagle3`, `stream_mean`) | `VLLM_GLM5_AUX_HIDDEN_TENSOR` (`stream_mean`); active only with a drafter | 0001 |
| 0002 | kpool tail ring sized for the draft depth (correctness) | always | 0002 |
| 0003 | drafter KV rides the MLA pages (1024-token blocks, page padded) and Mamba checkpoints follow the target state page | always (only with a DFlash drafter); `--block-size=5120`, `--max-num-batched-tokens=5184` | 0003, revised |
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
| 0017 | sm_80 Gluon mHC decode v2 (full port) | `VLLM_GLM5_DECODE_MHC_V2` (0), `_MAX_TOKENS` (32) | 0017 (`patches/`) |
| 0018 | fused GDN/KDA spec-decode metadata build | `VLLM_GLM5_PROLOGUE_FUSE` (0), `_GDN` (1, bisect only) | 0018 (`patches-host/`) |
| 0019 | one HtoD for the prepare_inputs index arrays | `VLLM_GLM5_PROLOGUE_PACKED_H2D` (0) | 0019 (`patches-host/`) |
| 0020 | device-side sparse MLA `req_id_per_token` | `VLLM_GLM5_SPARSE_MLA_DEVICE_REQ_IDS` (0) | 0020 (`patches-host/`) |
| 0021 | DFlash2 draft tail on an earlier PP stage (audited) | `VLLM_PP_DRAFT_TAIL_STAGE` (-1), `_VERIFY` (0) | 0021 (`patches-tail/`) |
| 0022 | sparse-MLA profile placeholder clamp | `VLLM_GLM5_MLA_PROFILE_WS_CLAMP` (0) | 0022 (`patches-kv/`) |
| 0023 | route v2 tile load order; optional route-before-shared | `VLLM_GLM5_ROUTE_V2_FIRST` (**0**; dev default 1) | 0023 (`patches-route/`) |
| 0024 | indexer weight scale folded into the decode FWHT quant | `VLLM_GLM5_DECODE_IDX_GLUE` (0) | 0024 (`patches-idx/`, fixed) |
| 0025 | fp32 TileLang mHC prenorm GEMM at every T | `VLLM_GLM5_TARGET_PRENORM_FP32` (0) | 0026 (`patches/`, was "diagnostic") |
| 0026 | routed moe_sum fused with the shared-expert add | `VLLM_GLM5_MOE_SUM_ADD` (0) | 0027 (`patches-step/`) |
| 0027 | shared experts enqueued after the routed experts | `VLLM_GLM5_SHARED_EXPERT_REORDER` (0) | 0028 (`patches-step/`) |
| 0028 | bf16x3 tensor-core prenorm GEMM for T >= 384 (vendored Morrowmake kernel, Apache-2.0) | `VLLM_GLM5_PRENORM_BF16X3_MIN_TOKENS` (384; only inside 0025's `=1` branch) | 0026b (`patches/`) |
| 0029 | indexer wk+weights as one dual-store thin GEMM (k bf16 bitwise, head weights fp32; replaces cast + fp32 sgemm) | `VLLM_GLM5_IDX_DUAL_GEMM` (0; needs `VLLM_GLM5_THIN_GEMM=1`; **on in the profile**) | - (STEP-GAP-2) |
| 0030 | PP hop metadata cache (32-byte header; payload only when the pickled bytes change) | `VLLM_PP_METADATA_CACHE` (0; **on in the profile**; same on every PP rank) | 0025 (`patches-pp/`) |
| 0031 | PP hop tensors packed into one NCCL P2P | `VLLM_PP_PACK_TENSORS` (0; **on in the profile**; checked equal on every PP rank) | 0029 (`patches-pp/`) |
| 0032 | PP sparse-MLA / KDA prefill above 2312 rows (the old limit was the validation chunk, not a kernel limit) | with 0012/0014; `VLLM_GLM5_PP_KDA_PREFILL_MAX_TOKENS` (16384) | - |
| 0033 | parser splits on `</think>` with the GLM-5.3 template even when thinking is "off" | `VLLM_GLM53_FORCE_THINK_BOUNDARIES` (0) | - |
| 0034 | one profile switch for the series | `VLLM_GLM53_OPT_PROFILE` (unset) | - |

0003: each drafter block holds 1024 tokens in one MLA page (4 of 5 MiB at
target block 5120); padded pages are never split into kernel blocks. With
LMCache both the 5120 chunk and the 2048-token drafter window must divide by
the drafter block. The KDA state page at the deepest DFlash verification (7)
is 4,685,824 bytes, so the target block cannot be below 4608. The scheduler
used to align Mamba checkpoints to the smallest group block (the drafter's
1024); it now follows the homogeneous `MambaSpec.block_size`, so local prefill
and an LMCache restore snapshot the KDA state at the same positions. Mixed
Mamba block sizes keep the old behaviour. A 5120 block needs
`--max-num-batched-tokens=5184` to fit next to the draft input slots.

0032: sparse MLA runs one independent CTA per query row, so the row count only
changes the grid; the int32 offset guard remains the real bound (32768 rows).
KDA sizes its chunk workspaces dynamically under a 32768-token ceiling; the
default dispatch limit is 16384. 2312/5120/8192/10240-row calls match chunked
calls bitwise (output and final KDA state). Marlin MoE never had a row limit.

0034: `VLLM_GLM53_OPT_PROFILE=default` sets each switch to its deployed value
unless that variable is already set; the table is in the model README.

Left out of the development series:

| dev no. | patch | why |
|---|---|---|
| 0007 | DFlash context-KV graph | no measured gain; nothing later depends on it |
| 0009 | target mHC v2 decode | no end-to-end gain; only 0018 built on it |
| 0011 | KDA dual projection | no end-to-end gain |
| 0014 | adaptive-k force file | diagnostic only |
| 0015 | draft trace | diagnostic only |
| 0018 | mHC v1 numerics | never validated on a GPU (and needs 0009) |
| 0021 | PP pipeline trace (`patches/0021-pp-pipeline-trace-diagnostic`) | diagnostic only |
| - | accept-same-history overlay | diagnostic only |

Dropping them needed one regeneration: `0011` (dev 0016) had an `envs.py`
hunk next to the force-file entry of dev 0014; it was rebased without that
entry. Every other patch is the development patch rebased unchanged, except
`0006`, whose loader error text and docstring now point at
`models/glm-5.3-flash/native/ampere_marlin` and the in-image install. The final
tree equals the development tree minus the seven patches above (checked file by
file).

0017-0027 (second round). The development patches were made against
slightly different trees, so they were re-applied in the order above on top of
0001-0016 and each production patch is the git diff of one step:
- dev 0018 was made beside 0017 (same `envs.py` base); applied with a 3-way
  merge, no conflict;
- dev 0021 is the audited version; the old version + `0021-audit-upgrade-delta`
  gives the identical tree (checked);
- dev 0024 declared its new file as a modification of `/dev/null` content
  (`--- a/...`); rewritten as a new file and the trailing blank lines dropped
  (`git diff --check`);
- 0023: `VLLM_GLM5_ROUTE_V2_FIRST` default changed to 0 (opt-in with `1`);
- 0025: only the code comment changed (no longer called a diagnostic).
0028 (dev 0026b) was a `diff -ruN` patch; converted to git format (new file
mode), a provenance comment added to the vendored file and the code comment
renumbered. Apart from these comment/default edits the final tree equals the development
tree (`work-step/cand` + the 0021 audit delta) file by file.

0029 (third round, `/tmp/dcp/dflash-port/STEP-GAP-2.md`) was written directly
against the tree after 0028; it has no development predecessor. GPU: 15.7 us
saved per MLA layer, c1 counting 28.87 -> 29.05 steps/s, c8 727.8 -> 742.7,
acceptance unchanged; `compose.w4a16-dflash2.yml` turns it on. The compose
also switches 0010 to `VLLM_GLM5_ROUTE_V2_GEMV=tc`: its router logits are not
bitwise our old gate path but are bitwise Morrowmake's `_moe_route_kernel`
(prose acceptance 1.232/round, same as MM; gate 1.218), and its single 80-CTA
launch leaves SMs to the shared experts (c1 29.05 -> 29.21 steps/s).

0030 and 0031 are the development PP patches 0025
(`patches-pp/0025-pp-metadata-cache-experimental.patch`, `PP-HOP-FULL.md`) and
0029 (`patches-pp/0029-pp-pack-hop-tensors.patch`, `PP-PACK.md`). Both applied
on node2 with `patch -p1` (offsets only) on the image tree (0001-0028) + 0029;
here they were regenerated as clean git diffs against the tree that
`check-dflash2-series.sh` builds (0030 on 0001-0029, 0031 on 0001-0030). The
env names keep their development suffixes (`VLLM_PP_METADATA_CACHE`,
`VLLM_PP_PACK_TENSORS`); both default to 0 and change the PP wire
protocol, so every PP rank must use the same value (0031 checks this at
start-up). They were earlier listed as "no end-to-end gain"; that was on an
older baseline. On the full 0001-0029 stack + tc routing, both on: c1 counting
29.32 -> 29.44 steps/s, c8 counting/code/prose 739.5/589.9/324.0 ->
752.4/609.2/335.4 (Morrowmake 765.7/608.5/341.1); prose acceptance 1.232
unchanged. `compose.w4a16-dflash2.yml` turns both on.

Evaluated in the same round and dropped (no patch file; their evaluation
numbers 0030/0031 predate and are unrelated to the patches above):

| eval no. | patch | why |
|---|---|---|
| (0030) | thin GEMM schedule selected exactly as Morrowmake's (`VLLM_GLM5_THIN_GEMM_MM_SELECT`) | no end-to-end gain (c8 729.2) |
| (0031) | shared-experts side stream with a CUDA stream priority (`VLLM_GLM5_SHARED_STREAM_PRIORITY`) | slower in the MoE-layer bench (tc+high 381.3 vs tc+default 375.9 us/layer) |

Static checks: `scripts/test-glm53-dflash2-patches.py` (run by the apply
script). CPU check of the whole stack:
`models/glm-5.3-flash/tests/check-dflash2-series.sh`.
