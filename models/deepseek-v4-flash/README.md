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

## Run

```bash
cp .env.example .env      # set VLLM_API_KEY and machine paths
podman-compose -f compose.pd.yml --env-file .env up -d
VLLM_API_KEY=... ./scripts/pd-request.sh "Explain pipeline parallelism in one paragraph." 64
```
