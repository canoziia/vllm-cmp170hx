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
| `MIMO_KV_CACHE_MEMORY` | 12 GiB | 1,401,951 tokens, 1.34× at 1M; one 1M request needs 8.98 GiB/rank (5 padding layers in the hybrid groups, ≤12.5% waste) |
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

Not in the profile but set in `.env` (measured on top of the profile):

| setting | value | effect |
|---|---|---|
| `MIMO_PP_LAYER_PARTITION` | 13,13,13,9 | the last rank also drafts; c32 counting +5 %, frees 14 GiB there |
| `VLLM_PP_DECODE_COHORT_BALANCE` | 1 | spreads decodes over the PP in-flight batches; c32 counting/code/prose +57 / +63 / +36 %, c1 bit-identical |

## Not enabled yet

- The bundled `dflash/` drafter (5-layer DFlash, block 8) is not used. Only
  the MTP heads are.
- LMCache: not connected. The hybrid SWA/global grouping with sinks has not
  been validated against the shared LMCache series.
- Multimodal inputs are loaded but have not been smoke-tested.
