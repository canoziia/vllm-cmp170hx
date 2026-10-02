# DeepSeek-V4-Flash PD (prefill/decode disaggregated) on six CMP 170HX GPUs

Model: `deepseek-ai/DeepSeek-V4-Flash` - `model_type: deepseek_v4`,
`DeepseekV4ForCausalLM`, 43 transformer layers, `num_nextn_predict_layers: 1`,
fp8 e4m3 weights (`weight_block_size [128,128]`) with fp4 experts,
`vocab_size 129280`, `hidden_size 4096`, 1M context.

This is **not** DeepSeek-V4.1 (`deepseek_v41`, 40 layers, block size [32,32]).
Non-model settings follow the V4.1 deployment; the model-specific runtime names
(tokenizer mode, tool-call/reasoning parser, KV cache dtype, speculative method)
are exposed as `.env` overrides because they must be confirmed against the image
actually used.

## Roles

| role | GPUs | PP3 partition | speculative |
|---|---|---|---|
| prefill | 4,5,6 | `15,14,14` (even) | none |
| decode | 7,8,9 | `15,15,13` (even layers, nextn on the last rank) | `mtp` |
| lmcache | sees 4..9 | - | - |

Disaggregation uses the shared LMCache server as the KV medium: prefill stores,
decode retrieves, then decodes with the speculative decoder.

## Verified behaviour

| step | request | result |
|---|---|---|
| prefill role (GPU 4,5,6), first time | prompt (2409 tok) | server A logs `Stored 2048 tokens` on each of the 3 ranks; 12 objects land in the L2 directory |
| decode role (GPU 7,8,9), first time | same prompt, after prefill | server B logs `Retrieved 2048 tokens`; `prompt_tokens_details.cached_tokens = 2048` |
| decode role, larger budget | same prompt | answer correctly summarises the transferred passage and echoes the nonce that was only ever sent to the prefill role |

Evidence: `/root/app/pd-validation/deepseek-*/`.

## Fitting PP3 on 63 GiB cards

This model is 148.7 GiB, so PP3 puts ~50 GiB of weights on each card. Measured on
node2 (CMP 170HX, 63.4 GiB usable per card):

- weights: 49.3-51.5 GiB per rank;
- KV budget is therefore **4 GiB**, not the 6 GiB used by node1's PP6 V4.1
  deployment: at 6 GiB the engine OOM'd during kernel warmup, and at 4 GiB the
  KV cache still holds ~1.24M tokens (enough for one 1M-token request);
- **CUDA graphs must be PIECEWISE**: with FULL graphs the capture OOM'd in the
  ~8 GiB that remains;
- `--max-num-seqs` is reduced (4 prefill / 8 decode) to keep graph buffers small;
- the two LMCache servers must use a small L1 (`LMCACHE_L1_SIZE_GB=8`): the L1
  tier is host RAM, and 64 GiB per server got both servers OOM-killed while this
  model was loading on a 251 GiB host.

## Run

```bash
cp .env.example .env      # set VLLM_API_KEY and machine paths
podman-compose -f compose.pd.yml --env-file .env up -d
VLLM_API_KEY=... ./scripts/pd-request.sh "Explain pipeline parallelism in one paragraph." 64
```
