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

The image built by `scripts/build-deepseek-v4-flash-image.sh` carries
`patches/series` (applied after the V4.1 adaptive series, checked against
`manifests/patches.sha256`, tested by `tests/test_dsv4_verification.py`).
Nothing changes unless enabled:

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
VLLM_DSV41_OPT_PROFILE=dsv4-flash
--speculative-config={"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true}
```

`dsv4-flash` includes `VLLM_DSV41_VERIFICATION`, `VLLM_DSV41_FAST_METADATA`,
`VLLM_DSV41_METADATA_GRAPHS` and `VLLM_DSPARK_DRAFT_HEAD_FP8` together with the
target-kernel and PP switches below (the speculative-only `v4-spec` profile of
the first delivery was merged into it). Adaptive history verification requires exactly
K=5, PP3, greedy drafting and CUDA graphs. Requests may pin a width with
`"vllm_xargs":{"spec_k":N}` (0..5). For PD, both roles must declare the same
`num_speculative_tokens` (see above).

Measured on node2 (CMP 170HX, tok/s; c1 per request, c4/c8 full batch; image
`deepseek-v4-flash-87b407e`, target kernels at their defaults):

| load | fixed K3 (old default) | fixed K5 | fixed K6 | K5 + spec switches + adaptive |
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
cohort switch, then not in the speculative profile) added +5-16 % at c4 on top.
Greedy outputs can differ between widths at near-ties (batch-shape numerics,
see `../deepseek-v4.1-flash/docs/GREEDY-OUTPUT-ARGMAX-AUDIT.md`); identical
counting outputs were observed at every width and concurrency.

## Single engine (PP3, no PD) and the `dsv4-flash` profile

`compose.yml` is the decode role of `compose.pd.yml` as one engine without
LMCache (GPUs, port and container name from `.env.single`, see
`.env.single.example`). The image is built by
`scripts/build-deepseek-v4-flash-image.sh` (the DeepSeek V4.1 image plus
`patches/series`). `DSFLASH_OPT_PROFILE=dsv4-flash` sets
`VLLM_DSV41_OPT_PROFILE=dsv4-flash`; `off` (default) is the unmodified baseline.

The checkpoint's `mtp.*` layers are a 3-layer DSpark drafter (`main_proj`,
`markov_head`, `confidence_head`), so `method=mtp` does not load it;
`dspark` requires `num_speculative_tokens` divisible by 3.

Measured on node2 GPUs 1-3 (CMP 170HX), `benchmark-vllm.mjs`, 500 tokens, 2
warm-up + 3 rounds (median), max-num-seqs 8 unless noted; restart-to-restart
noise is about +-3 %.

| tok/s (per request at c1, full batch at c8) | baseline (off) | dsv4-flash |
|---|---|---|
| code c1 / c8 | 123.8 / 485.1 | 166.5 / 647.2 |
| counting c1 / c8 | 114.6 / 436.2 | 150.6 / 573.3 |
| prose c1 / c8 | 73.1 / 291.6 | 105.0 / 376.4 |

With `DSFLASH_MAX_NUM_SEQS=32` (no OOM at startup or in decode; 1M-token
prefill not re-tested): c16 code/counting/prose 1198 / 1012 / 691, c32 1588 /
1382 / 896 tok/s.

Contributions, each measured on top of the previous ones:

| switch | effect |
|---|---|
| `VLLM_PP_DECODE_COHORT_LIVE_LOAD=1` | counting c4 +33 %, others +-2 % |
| MXFP4 MoE decode v2 for 4096/2048 experts (0001; DQ2, fused act/sum) | c8 +15-17 %, c1 +2-4 % |
| `VLLM_PP_DEFERRED_FEEDBACK_RECV=1` (0003) | c8 +3.5 %, c4 +1-2 % |
| `VLLM_DSV4_FAST_DET_MOE_ALIGN` + `VLLM_DSV4_CUDA_DET_ALIGN` | c1 +15 %, c8 +6 % (the torch deterministic align cost ~78 us per MoE call) |
| sparse decode fast + attention direct out + thin wo_a + indexer q LUT | c1 +6-8 %, c8 +5-7 % |
| small-M dense FP8 GEMMs (0004) | c1 +5 % per step, c8 +-2 % |
| mHC v2 for hidden 4096 (0002, **not** in the profile) | +1-2 % per step; 2 of 191 stress cases exceed V4.1's 4x fp64 tolerance |

Tried and not kept: V4.1 prefill switches (indexer prefill fast, MoE prefill
split: neutral), per-M MoE kernel variants (within noise), w2 split-K for the
MoE kernel (faster only with <= 4 distinct experts), fused next_n paged indexer
logits (1.3x on the logits kernel only at >= 40k compressed tokens, slower
below).

Decode time split (sampled CUDA events + kernel micro-benchmarks, profile on):
c1 (4 tokens/step) target forward 85 %, DSpark draft 10 %, sampling 3 %; per
layer MoE ~55 %, dense FP8 ~29 %, mHC ~7 %, attention/indexer/small ops ~10 %.
c8 (PP cohorts of 8-12 tokens) per layer MoE ~69 %, dense ~14 %, attention and
small ops ~14 %. Long context (c1, prose): 48 / 41.5 / 38.8 steps/s at short /
50k / 159k tokens; the growth is mostly the indexer (logits + top-k over the
compressed keys).

### Merged stack (dsv4-flash profile, dspark K=5 + adaptive verification)

Image `deepseek-v4-flash-0e1b0ad` (+ re-swept cost table), max-num-seqs 32,
node2 GPUs 1-3, tok/s (c1 per request, others full batch):

| load | c1 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|
| code | 197.0 | 555 | 848 | 1339 | 1515 |
| counting | 168.1 | 497 | 745 | 1113 | 1189 |
| prose | 102.3 | 283 | 447 | 687 | 717 |

Against the same target kernels with fixed K=3: c1-c16 equal or better (c8
+19-31 %), c32 worse (K=3: 1588 / 1382 / 896). On the K=5 engine a c32 cohort
at spec_k=3 costs 101.8 ms/step against ~78 ms with a K=3 engine, and spec_k=4
144 ms; see the c=32 row of `COST_MS_DSV4_PP3`.

`COST_MS_DSV4_PP3` (patch 0001) was re-swept with the target kernels on
(`spec_k` 1-5, counting, c = 1-32; the old table was measured with the default
kernels and is 6-8 ms/step higher at c1).

### Final stack (image `deepseek-v4-flash-3418e57`, patches 0001-0009, `dsv4-flash`)

dspark K=5 + adaptive verification, max-num-seqs 32, KV 2.5 GiB; node2 GPUs
1-3 while another engine ran on GPUs 5-8 (c32 rounds vary by up to ~10 %).

| tok/s | c1 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|
| code | 197.1 | 563 | 847 | 1356 | 1482 (1377-1598 over runs) |
| counting | 167.5 | 494 | 754 | 1145 | 1288 (1288-1350) |
| prose | 101.3 | 284 | 439 | 642 | 919 (894-919) |

Same image and profile with fixed dspark K=3, measured right after: c16
1246 / 1065 / 687, c32 1646 / 1384 / 892 (code / counting / prose). K=5 wins
at c1-c16 except prose c16 (-6.5 %); at c32 K=3 is still ahead on code and
counting. With 0008 and 0009 the c32 gap is no longer cohort padding or the
draft block (both now match K=3 shapes: 44 rows -> 48-row graph, 4-position
drafts); at the same verification width a K=5 engine still steps ~4-6 ms
slower at c32 (code 73.5 vs 70 ms, counting 84 vs 78 ms), which points at the
K=5 decode metadata bounds (`decode_query_len` 6) inside the graphs.

## Prefill switches in `dsv4-flash` (patches `0011`, `0012`)

- `VLLM_DSV41_MOE_PREFILL_SPLIT=1`: MXFP4 MoE prefill split lists, extended to
  V4's 256x4096x2048 experts.
- `VLLM_DSV4_SPARSE_PREFILL_H32=1`: 32-head sparse prefill attention tile,
  bitwise identical to the 64-head tile and ~27 % faster per kernel.
- `VLLM_DSV4_SPARSE_PREFILL_REQS=1`: the sparse prefill gathers one request's
  compressed + SWA KV at a time; the warmup-sized bf16 workspace shrinks from
  1028.5 to 293 MiB (-0.71 GiB per card at 1M context).

None adds resident memory. PP3, MBT 1024, cold prefill: 6.7k 1.91 -> 1.72 s,
26.7k 6.71 -> 6.11 s, 107k 28.97 -> 26.28 s, 859k 438 -> 397 s.

**Warm up after start.** The first long prompt after an engine start pays the
Triton JIT for new prefill shapes (107k: ~33 s first vs ~27 s after; 6.7k:
~5 s vs 1.8 s). Once `/health` returns 200, send one ~32k-token request
(unique `cache_salt`, `max_tokens=1`) before taking traffic:

```bash
python3 - <<'PY'
import json, os, urllib.request, uuid
body = {"model": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", "max_tokens": 1,
        "prompt": "warm up " * 24000, "cache_salt": str(uuid.uuid4())}
req = urllib.request.Request(f"http://127.0.0.1:{os.environ.get('DSFLASH_PORT', '8501')}/v1/completions",
      json.dumps(body).encode(), {"Content-Type": "application/json",
      "Authorization": "Bearer " + os.environ["VLLM_API_KEY"]})
urllib.request.urlopen(req, timeout=600).read()
PY
```
