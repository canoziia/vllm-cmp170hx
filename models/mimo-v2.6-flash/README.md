# MiMo-V2.6-Flash (MOPD) on CMP 170HX

`XiaomiMiMo/MiMo-V2.6-Flash-MOPD` @ `b12f5573350bb1704513fcf119fca48e498b0423`
(MIT; 309B total / 15B active MoE, 48 layers = 39 SWA-128 + 9 global GQA,
3 MTP layers, vision + audio towers). Use the `-MOPD` checkpoint, not `-RL`
(the RL release has the tool-call repetition problem the MOPD distillation
mitigates).

Single engine, PP4 on four GPUs, container `mimo-v2.6-flash`, port **8401**,
served name `XiaomiMiMo/MiMo-V2.6-Flash-MOPD`.

```bash
CONTAINER_ENGINE=podman bash scripts/build-mimo-v26-image.sh   # localhost/vllm-backport:mimo-v2.6-flash
cd models/mimo-v2.6-flash
cp .env.example .env && chmod 600 .env                          # VLLM_API_KEY, GPUs, paths
podman compose -f compose.yml --podman-run-args=--ipc=host up -d
```

`--podman-run-args=--ipc=host` is required: podman-compose otherwise starts the
container with `IpcMode=shareable` even though the file says `ipc: host`.

## Source stack

Same pinned vLLM source and base image as DeepSeek V4.1
(`models/deepseek-v4.1-flash/manifests/source.env`), with the DeepSeek V4.1,
shared and adaptive series applied exactly as for that image, then
`patches/series`:

| patch | what |
|---|---|
| 0001-mimo-v2-fp8-qkv-checkpoint-shards | fused FP8 `qkv_proj` loader for the main model and the MTP head (below) |
| 0002-mimo-v2-multi-layer-mtp | the three checkpoint MTP layers as parallel draft heads (below) |
| 0003-mimo26-opt-profile | `VLLM_MIMO26_OPT_PROFILE`: one switch for the series |
| 0004-moe-align-bitonic-sort | single-sort deterministic MoE align up to 4096 entries |
| 0005-mimo-v2-fp8-draft-head | fp8 copy of the shared lm_head for draft logits |
| 0006-mimo-v2-dflash-drafter | the bundled DFlash drafter under PP (experimental, not deployed; only active with `method: dflash`) |
| 0007-mimo-v2-mtp-draft-widths | draft k, verify those k; measured per-request width policy; `vllm_xargs.spec_k` in MiMo-only code |
| 0008-pp-deferred-feedback-recv | PP sampled/draft feedback receive posted when consumed (generic PP, switch) |
| 0009-mimo-v2-mxfp4-moe-decode | weight-streaming MXFP4 MoE decode kernels for MiMo's experts (`native/mimo_moe`, built into the image) |
| 0010-diffkv-prefill-block-m | TRITON_ATTN_DIFFKV prefill query tiles and qk head-dim split |

The pinned tree already has everything else: `MiMoV2OmniForCausalLM` (the
checkpoint has `vision_config`, so vLLM loads text + vision + audio), the
MiMo MTP head, the `mimo` reasoning and tool parsers, FP8 block-128 linears
through Marlin W8A16 and MXFP4 experts through the staged Marlin W4A16 repack
on sm_80, and `TRITON_ATTN_DIFFKV` for qk 192 / v 128 with attention sinks
(FA-DiffKV needs FA3).

### 0001: fused FP8 qkv_proj layout

The checkpoint stores `qkv_proj` pre-sharded for its export TP (4 for
V2.6-Flash, 8 for Pro): 4 consecutive row shards, each `[Q_s | K_s | V_s]`
for 16 query heads and `num_kv_heads/4` KV heads, and each with its own
128x128 block-scale rows padded to whole blocks. That is why a global layer
(13568 rows = 106 blocks) has 108 scale rows (4 x 27). Upstream assumed one
`[Q_i|K_i|V_i]` group per KV head with `scale_rows // num_kv_heads` scale
rows per group, which only holds when export TP == KV heads (Pro). On
V2.6-Flash it fails at load for SWA layers, and once the loads were made to
succeed the model produced unrelated multilingual tokens. The MTP loader
used a plain `chunk(tp_size)`, which is only right at TP 4.

The patch derives the shard count from the config (smallest KV-head count
over the layer types), dequantizes each checkpoint shard with its own scale
grid, reassembles Q/K/V in head order, and re-quantizes this TP rank's
`[Q|K|V]`. When TP equals the export TP it is a plain slice with no
re-quantization.

Evidence: a streaming torch reference that implements the official
`modeling_mimo_v2.py` semantics on the raw checkpoint predicts ` Paris`
(logit 17.62, next ` love` 15.12, ` **` 14.50) for "The capital of France is
the city of" only with this layout. The served model gives the same top-5 in
the same order (logprobs -0.27 / -2.46 / -3.27). MXFP4 nibbles are
low-nibble-first.

Tests:
- `tests/test_mimo_patches.py SOURCE_TREE`: static, run by
  `scripts/apply-mimo-v26-patches.sh`
- `tests/test_fp8_qkv_checkpoint_shards.py`: numeric, needs one GPU (vLLM
  import). Synthetic export-TP-4 matrices for both geometries; TP 1/2/4
  against the exact dequantized reference.

## Configuration (.env)

| key | value | note |
|---|---|---|
| `MIMO_GPU_0..3` | 5,6,7,8 | PP ranks 0–3 |
| `MIMO_PP_LAYER_PARTITION` | 13,13,13,9 | model loading 44.2 / 46.0 / 46.0 / 36.8 GiB per rank (rank 3 also holds lm_head, MTP, fp8 draft head) |
| `MIMO_MAX_MODEL_LEN` | 1048576 | 1M |
| `MIMO_KV_CACHE_MEMORY` | 12 GiB | 1,401,773 tokens, 1.34× at 1M; one 1M request needs 8.98 GiB/rank (5 padding layers in the hybrid groups, ≤12.5% waste) |
| `MIMO_MAX_NUM_SEQS` / `MIMO_MAX_NUM_BATCHED_TOKENS` | 32 / 4096 | 32 like DeepSeek/Qwen; 4096 covers 32 x (1 + 3 drafts) |
| `MIMO_SPECULATIVE_CONFIG` | `{"method":"mtp","num_speculative_tokens":3}` | MTP3, see below |
| `VLLM_MIMO26_OPT_PROFILE` | default | switch table below |

Peak GPU memory after capture: 59.2 / 61.2 / 61.2 / 47.8 GiB of 64 GiB.

Parsers follow the model card: `--reasoning-parser mimo --tool-call-parser mimo
--enable-auto-tool-choice --generation-config vllm`. Thinking is on by
default; `chat_template_kwargs: {"enable_thinking": false}` emits an empty
`<think></think>`. Recommended sampling is temperature 1.0, top_p 0.95.

## MTP: three parallel heads (0002)

The checkpoint has three MTP layers (`model.mtp.layers.0-2`). Upstream vLLM
loads only layer 0 and reuses it for every draft step. With
`VLLM_MIMO26_MTP_LAYERS=3` (in the profile) draft step k runs layer k
through vLLM's multi-module MTP speculator.

The layers are *parallel heads*, not a chain: layer k predicts token t+k+2
from the target model's hidden state at t and token t+k+1. Feeding layer k
the output of layer k-1 (normed or pre-norm), as for DeepSeek MTP, leaves
layers 1-2 at ~0% acceptance. Per-position acceptance, c1 greedy, 300
tokens:

| | layer-0 reuse, k=3 | 3 heads, k=3 | 3 heads, k=6 (heads reused) |
|---|---|---|---|
| counting | 1.00 / 0.91 / 0.25 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 0.85 / 0.51 / 0.34 |
| code | 0.96 / 0.61 / 0.18 | 0.90 / 0.78 / 0.63 | 0.91 / 0.84 / 0.71 / 0.28 / 0.13 / 0.07 |
| prose | 0.67 / 0.20 / 0.03 | 0.69 / 0.35 / 0.15 | 0.66 / 0.35 / 0.14 / 0.03 / 0 / 0 |

vLLM requires the width to be a multiple of the head count. k=6 is faster
only on counting and slower on code and prose, so the width is 3.

## Optimisation switches

`VLLM_MIMO26_OPT_PROFILE=default` (patch 0003) sets every switch below
unless it is already set; unset/off leaves every patch at its own default
(off, i.e. upstream behaviour).

| patch | switch | profile value | effect |
|---|---|---|---|
| 0002 | `VLLM_MIMO26_MTP_LAYERS` | 3 | MTP3 with parallel heads (needs `num_speculative_tokens` 3) |
| DeepSeek 0003 | `VLLM_DSV4_FAST_DET_MOE_ALIGN` | 1 | single-kernel deterministic MoE align, bitwise equal, <= 512 routed entries (c <= 16 at MTP3); c1 +13-14 %, c8 +5-11 % |
| 0004 | `VLLM_MIMO26_SORT_MOE_ALIGN` | 1 | single-sort align for every size up to 4096 entries (4-8 us vs 11-59 us; c32 1024 entries no longer on the torch path), bitwise equal; c32 +4-6 % |
| 0005 | `VLLM_MIMO26_DRAFT_HEAD_FP8` | 1 | fp8 copy of lm_head for draft logits (target keeps bf16); c1 +4 % counting/code, +12 % prose; +0.6 GiB on the last rank |
| 0007 | `VLLM_MIMO26_DRAFT_WIDTHS` + `VLLM_MIMO26_ADAPTIVE_WIDTH` | 1 | draft k, verify those k; measured per-request width: prose at c1/c8/c32 reaches 99 / 106 / 102 % of the best fixed width with the full profile, counting/code stay k3; manual `vllm_xargs.spec_k` 0-3 (MiMo only; other models ignore it) |
| 0008 | `VLLM_PP_DEFERRED_FEEDBACK_RECV` | 1 | PP feedback receive posted when consumed: no NCCL spin under Marlin MoE on ranks 0-2 (fixed k1 prose c32 31.8 -> 35.4, k3 26.1 -> 29.7) |
| 0009 | `VLLM_MIMO26_MXFP4_DECODE` + `VLLM_MIMO26_MXFP4_V2` + `VLLM_MIMO26_MXFP4_DQ=2` | 1 / 1 / 2 | weight-streaming MXFP4 MoE decode (DeepSeek kernels for MiMo's shape, fused act/sum, cheaper dequant; error = Marlin's, greedy divergences only at near-ties): c1 +9 %, c8 +5-18 %, c32 +25-40 % vs Marlin |
| 0010 | `VLLM_MIMO26_DIFFKV_PREFILL_BM` + `VLLM_MIMO26_DIFFKV_QK_SPLIT` | 128 / 1 | prefill attention: 8 query tokens per program instead of 1 (error vs fp64 equal, greedy divergences only at near-ties; 64 = bitwise alternative) and qk 192 = 128 + 64 instead of padding to 256 (bitwise): see the prefill table below |

Not in the profile but set in `.env` (measured on top of the profile):

| setting | value | effect |
|---|---|---|
| `MIMO_PP_LAYER_PARTITION` | 13,13,13,9 | the last rank also drafts; c32 counting +5 %, frees 14 GiB there |
| `VLLM_PP_DECODE_COHORT_BALANCE` | 1 | spreads decodes over the PP in-flight batches; c32 counting/code/prose +57 / +63 / +36 %, c1 bit-identical |

## Results (profile default, `.env.example`)

Decode, per-request tok/s, `scripts/benchmark-vllm.mjs`, 500 tokens,
`enable_thinking` false, 2 warm-up + 3 measured rounds:

| | c1 | c8 | c32 |
|---|---|---|---|
| counting | 92 -> 180 | 53 -> 122 | 28 -> 84 |
| code | 92 -> 175 | 44 -> 93 | 27 -> 75 |
| prose | 81 -> 112 | 36 -> 56 | 21 -> 44 |

(left: MTP1, upstream defaults, 12,12,12,12; right: everything above). c8
code varies between runs (PP phase), 87-104.

Cold prefill (unique random-token prompts, `max_tokens` 1, chunk 4096):

| prompt | upstream kernel | profile |
|---|---|---|
| 8k | 2.46 s | 2.18 s |
| 32k | 8.18 s | 5.61 s |
| 131k | 63.8 s | 26.6 s |
| 512k | 884 s | 229 s |
| 1M | not run (~3000 s est.) | 785 s |

Long-prompt prefill is bounded by the 9 global-attention layers (rank 0
holds three of them; 9 over 4 ranks always leaves one rank with three). The
prefill attention kernel runs at ~94 TFLOP/s against a measured bf16 GEMM
peak of ~145 on this card.

## DFlash drafter (0006, experimental)

`--speculative-config={"method":"dflash","model":"/models/MiMo-V2.6-Flash-MOPD/dflash","num_speculative_tokens":7}`
runs the bundled 5-layer drafter (block 8, attention sinks, SWA 1024, value
scale 0.612, separate mask embedding). It reads aux hidden states of target
layers 0/11/23/35/47 across all PP ranks and needs the target embedding on the
last rank (+1.25 GiB). Everything it adds is inert unless the speculative
method is `dflash`. Measured before 0007-0010, against MTP3 at that time
(c1/c8/c32 per-request tok/s):
counting +54 / +12 / +15 %, code +26 / +12 / 0 %, prose -43 / -46 / -50 %
(prose acceptance 10.5 %). At k=3 its first three positions are below the MTP
heads' (code 0.84/0.73/0.58 vs 0.90/0.80/0.67). Not deployed:

- KV: with the drafter loaded the KV pool drops to 294k tokens and 1M needs
  29.45 GiB/rank instead of 8.98. The +20.5 GiB equals the drafter's 5
  layers x 8 KV heads x (128+128) x 2 B per token over 1M tokens, i.e. its
  SWA layers are budgeted as full attention (11 hybrid groups). Until that is
  fixed it needs max-model-len <= 262144.
- `VLLM_MIMO26_DFLASH_DRAFT_HEAD_FP8` (fp8 draft head for DFlash, default 0,
  not in the profile) lowers DFlash's deep-position acceptance (code 4.68 ->
  4.21 tok/step).

## Not enabled yet

- LMCache: not connected. The hybrid SWA/global grouping with sinks has not
  been validated against the shared LMCache series.
- Multimodal inputs are loaded but have not been smoke-tested.
