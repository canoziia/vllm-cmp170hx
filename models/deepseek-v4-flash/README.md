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
| prefill role (GPU 4,5,6), first time | prompt (1890 tok) | server A logs `Stored 1024 tokens` per rank; objects land in the L2 directory |
| decode role (GPU 7,8,9), first time | same prompt, after prefill | server B logs `Retrieved 1024 tokens`; `cached_tokens = 1024` (the remaining 512 tokens were computed and stored) |
| decode role output | same prompt | answer echoes the injected marker, so the context came from the transferred KV |

Evidence: `/root/app/pd-validation/`.

## Both roles must declare the same speculative depth

The L2 key is `(model, rank, object group)` — it does **not** include the size of
the object. The size follows from the KV geometry, and that geometry is resolved
from `num_speculative_tokens`. So if one role declares `--speculative-config` and
the other does not, both agree on the key but disagree on the byte count, and
every lookup fails with:

```
[LMCache GET] key /models/DeepSeek-V4-Flash@03020302@0@<hash> failed: incomplete read for
  <hash>.data: expected 4765696, got 4690944
```

The visible symptom on the engine is `cached_tokens = 0` together with
`created_cache_tokens > 0`, even though the other role clearly stored the chunk.
A prefill role therefore declares the same `--speculative-config` as the decode
role purely so both resolve one geometry; it does not sample draft tokens.

## Fitting PP3 on 63 GiB cards

This model is 148.7 GiB, so PP3 puts ~50 GiB of weights on each card. Measured on
node2 (CMP 170HX, 63.4 GiB usable per card):

- weights: 49.3-51.5 GiB per rank;
- KV budget is therefore **4 GiB**, not the 6 GiB used by node1's PP6 V4.1
  deployment: at 6 GiB the engine OOM'd during kernel warmup, and at 4 GiB the
  KV cache still holds ~1.24M tokens (enough for one 1M-token request);
- **CUDA graphs are PIECEWISE**: with FULL graphs the capture OOM'd in the
  ~8 GiB that remained at the 6 GiB KV setting (the committed 4 GiB setting was
  not re-tested with FULL graphs);
- `--max-num-seqs` is reduced (4 prefill / 8 decode) to keep graph buffers small;
- the two LMCache servers must use a small L1 (`LMCACHE_L1_SIZE_GB=8`): the L1
  tier is host RAM, and 64 GiB per server got both servers OOM-killed while this
  model was loading on a 251 GiB host.

## Run

```bash
# from this directory
cp .env.example .env               # set VLLM_API_KEY / machine paths, merge .env.pd.example
podman compose -f compose.pd.yml --podman-run-args=--ipc=host up -d
podman compose -f compose.pd.yml ps
podman compose -f compose.pd.yml down
```

`--podman-run-args=--ipc=host` is required: the connector moves KV over CUDA IPC,
and podman-compose 1.3.0 ignores the compose `ipc` key. node1 starts its
deployments with the same flag.

### Two-step PD request

```bash
KEY=$(grep '^VLLM_API_KEY=' .env | cut -d= -f2)
PF=http://127.0.0.1:8201   # prefill role
DC=http://127.0.0.1:8202   # decode role
BODY='{"model":"deepseek-ai/DeepSeek-V4-Flash","messages":[{"role":"user","content":"your prompt"}],"max_tokens":64}'
# 1) prefill the prompt so its KV lands in the shared L2 tier
curl -sS -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "${BODY/\"max_tokens\":64/\"max_tokens\":1}" $PF/v1/chat/completions
# 2) decode the same prompt; the response reports prompt_tokens_details.cached_tokens
curl -sS -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d "$BODY" $DC/v1/chat/completions
```

### Single endpoint in front of the pair (`-pd-router`)

Each pair is fronted by the **vLLM production-stack router** - community-maintained,
actively released (pinned here to `vllm-stack-0.1.13`) - running as an ordinary compose
service on a stock python image, so there is no custom image to build and no Kubernetes:

| model | single endpoint | prefill | decode |
|---|---|---|---|
| qwen | `:9101` | `:8101` | `:8102` |
| deepseek | `:9102` | `:8201` | `:8202` |

`--routing-logic disaggregated_prefill` makes the router perform both hops itself
(request -> prefill with `max_tokens=1` -> decode with the original request) and stream
the decode response back to the caller.

Measured through the router: a 5119-token prompt returned
`prompt_tokens_details.cached_tokens = 4800` while the prefill server logged
`Stored 1600 tokens` twice and the decode server `Retrieved 4800 tokens`.

Things worth knowing:

* the router does **not** forward the caller's `Authorization` header - it builds its
  own from `OPENAI_API_KEY`, which the compose service wires to `VLLM_API_KEY`;
* use `disaggregated_prefill`, **not** `disaggregated_prefill_orchestrated`: the
  orchestrated variant injects `kv_transfer_params` but sends **no** Authorization
  header to the backends (`request.py:833` in 0.1.13 and in main), so it returns 401
  against engines started with `--api-key`. We do not need `kv_transfer_params` because
  our KV moves through the shared LMCache tier;
* other policies shipped by the router: `roundrobin`, `session` (add `--session-key`
  for session affinity), `prefixaware`, `kvaware`, `loadaware`, `priority`;
* `--static-backend-health-checks` exists but is currently incompatible with
  authenticated backends (upstream vllm-project/production-stack issue #631).

The earlier vendored copy of vLLM's example `disagg_proxy_server.py` was removed in
favour of this router.

## DSpark speculative decoding (opt-in)

The image built by `scripts/build-deepseek-v41-image.sh` also carries the
V4-specific series `patches/series` (applied after the V4.1 adaptive series,
checked against `manifests/patches.sha256`, tested by
`tests/test_dsv4_verification.py`). Nothing changes unless enabled:

- `0001` native DSpark block: V4 drafts `dspark_block_size` = 5 tokens per
  parallel pass. Upstream treated the 3 MTP stages as `n_predict`, which only
  allowed K <= 3 or multiples of 3 (K=6 then drafts one block of 6, one slot
  past the trained block). Now K = 1..5 (or multiples of 5). It also extends
  the V4.1 verification-width machinery (`spec_k`, exact small-cohort graphs,
  decode metadata, measured history policy) to V4, with a PP3 cost table and
  one shared width per scheduled cohort (mixed widths lose the FULL graph).
- `0002` fp8 draft head (`VLLM_DSPARK_DRAFT_HEAD_FP8=1`): draft logits only;
  target verification keeps the bf16 head. 823 -> 413 us per step at 5 rows.

Recommended decode settings (single engine, PP3 `15,16,12`, max-num-seqs 8):

```text
VLLM_DSV41_OPT_PROFILE=v4-spec
--speculative-config={"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true}
```

`v4-spec` sets only `VLLM_DSV41_VERIFICATION`, `VLLM_DSV41_FAST_METADATA`,
`VLLM_DSV41_METADATA_GRAPHS` and `VLLM_DSPARK_DRAFT_HEAD_FP8`; target kernel
switches are independent. Adaptive history verification requires exactly
K=5, PP3, greedy drafting and CUDA graphs. Requests may pin a width with
`"vllm_xargs":{"spec_k":N}` (0..5). For PD, both roles must declare the same
`num_speculative_tokens` (see above).

Measured on node2 (CMP 170HX, tok/s; c1 per request, c4/c8 full batch; image
`deepseek-v4-flash-87b407e`, target kernels at their defaults):

| load | fixed K3 (old default) | fixed K5 | fixed K6 | K5 + v4-spec + adaptive |
|---|---|---|---|---|
| counting c1/c4/c8 | 121.5 / 265 / 446 | 119.9 / 318 / 502 | 109.9 / 275 / 460 | 127.1 / 328 / 573 |
| code c1/c4/c8 | 125.2 / 274 / 525 | 144.4 / 370 / 663 | 133.4 / 367 / - | 146.7 / 378 / 662 |
| prose c1/c4/c8 | 73.5 / 216* / 296 | 68.3 / 175 / 300 | 68.2 / 181 / 272 | 79.9 / 204 / 342 |

`*` unstable (round spread 15 tok/s). K6 drafts one slot past the trained
block and is worse than K5 everywhere. Prose acceptance is low (~2.1 tokens
per step); an offline per-step oracle over the measured acceptance stream
leaves no width headroom at c1, so the remaining prose cost is the target
step. Prose c1 also moves +-8 % with the output trajectory (a near-tie at the
first token changes the essay). `VLLM_PP_DECODE_COHORT_LIVE_LOAD=1` (a PP
cohort switch, not part of `v4-spec`) added +5-16 % at c4 on top.
Greedy outputs can differ between widths at near-ties (batch-shape numerics,
see `../deepseek-v4.1-flash/docs/GREEDY-OUTPUT-ARGMAX-AUDIT.md`); identical
counting outputs were observed at every width and concurrency.
