# DeepSeek-V4-Flash-Vision-Exp on CMP 170HX: PD pair (`compose.pd.yml`) and single engine (`compose.yml`)

Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` - `model_type: deepseek_v4`,
43 transformer layers plus a 3-layer DSpark drafter (`mtp.*`), fp8 e4m3
weights with fp4 experts, `hidden_size 4096`, 1M context. This is **not**
DeepSeek-V4.1 (`deepseek_v41`, 40 layers).

## PD (prefill/decode disaggregated), host-b

| role | GPUs | port | PP3 partition | MBT | max-num-seqs | KV (`--kv-cache-memory`) | KV tokens |
|---|---|---|---|---|---|---|---|
| router (public) | - | 8401 | - | - | - | - | - |
| prefill | 5,6,7 | 8412 | `15,16,12` | 2048 (long-prefill threshold 2048) | 4 | 3.5 GiB | 1,382,994 |
| decode | 1,2,3 | 8413 | `15,16,12` | 512 | 32 | 3.75 GiB | 2,228,171 |
| LMCache MP server | sees 1-3, 5-7 | 5571 / http 18571 | - | chunk 1024 | - | L1 12 GiB host RAM, L2 200 GiB fs_native | - |

Both engines: image `localhost/vllm-backport:deepseek-v4-flash-f5c9c8d`
(`scripts/build-deepseek-v4-flash-image.sh`; carries the patched LMCache
payload of `localhost/lmcache-server:latest`, incl. LMCache 0008
`mp_worker_rank`), profile `dsv4-flash` (prefill switches 0011/0012, live-load
PP cohorts), DSpark K=5 + adaptive verification, FULL_AND_PIECEWISE graphs,
`--prefix-cache-retention-interval` = LMCache chunk (SWA groups use the
default window path; LMCache 0010 is gone), `--shutdown-timeout=30`,
`--sse-keep-alive-interval=${VLLM_SSE_KEEP_ALIVE_INTERVAL:-15}`.
Router: `localhost/vllm-router:latest` (patches/router 0001-0004: usage merge,
client API key, cancel on disconnect, streamed prefill + keep-alive),
`--routing-logic=disaggregated_prefill`.

```bash
cp .env.pd.example .env.pd && chmod 600 .env.pd   # set VLLM_API_KEY
podman compose -f compose.pd.yml --env-file .env.pd \
  --podman-run-args="--ipc=host --pid=host" up -d
podman compose -f compose.pd.yml --env-file .env.pd down   # before every restart
```

Then warm up once (see "Warm up after start" below, port 8401): the first
107k prompt after start takes ~39 s instead of ~25 s.

Rules that matter:

- **Same partition and same speculative config on both roles.** The drafter's
  three SWA layers (global 43-45) are packed into the target's SWA KV groups
  on rank 2, so a prefill engine without DSpark registers 6-layer instead of
  7/8-layer objects under the same LMCache key (the key has no size): every
  rank-2 lookup then fails or misreads. The prefill engine computes the
  drafter's prompt KV, which the decode drafter reads back.
- **`pid: host` on every container** (plus `--ipc=host --pid=host` on the
  command line; podman-compose 1.3 ignores `ipc`): CUDA IPC event pools live in
  `/dev/shm/cuda.shm.0.<pid>`; private PID namespaces reuse PIDs and the
  LMCache server waits on the other engine's events (commit 901b42d).
- **Small L1.** host-b's Qwen LMCache already holds 64 GiB; 64 GiB here got the
  host OOM-killed while this 149 GiB model loaded. A ~960k-token context is
  ~21 GiB of LMCache objects, so L1 holds about half of one; L2 (200 GiB, ~9
  such contexts) serves the rest.
- The router builds the backend `Authorization` from `OPENAI_API_KEY` and
  requires the same key from clients (0002); `/health` stays open.
  `prompt_tokens_details.cached_tokens` in a PD response is the prefill hop's
  count (0 on a cold prompt); the decode hop's LMCache hit is in
  `usage.router_hops.decode_usage`.

Measured on host-b (`/srv/dsv4-work/PD-REPORT.md`), through the router:
decode tok/s (c1 per request, c8/c32 full batch; 2 warm-up + 3 rounds) and
cold TTFT, against the single engine (same image, MBT 2048, max-num-seqs 32):

| | PD | single engine |
|---|---|---|
| counting c1 / c8 / c32 | 174.1 / 788 / 1317 | 173.4 / 753 / 1408 |
| code c1 / c8 / c32 | 207.7 / 858 / 1456 | 206.6 / 852 / 1495 |
| prose c1 / c8 / c32 | 100.6 / 452 / 851 | 100.8 / 444 / 933 |
| TTFT 6.7k / 26.7k / 107k (s) | 2.23 / 6.24 / 25.48 | 1.81 / 5.99 / 24.71 |
| TTFT 107k, second turn (cached 107,008) | 1.40 s | - |
| c4 decode during a 107k cold prefill | 476 tok/s (501 before) | not measured |
| two ~960k requests (A decodes while B prefills) | A 95.2 tok/s during B, 94.9 after; TTFT 462 / 465 s | - |

Peak GPU memory: decode 59.1 / 61.9 / 61.0 GiB, prefill 58.5 / 61.3 / 60.2 GiB
(GPUs 1-3 / 5-7, after the 960k runs); GPU 2 (decode rank 1) has ~1 GiB left,
so do not raise the decode KV further.

Greedy counting outputs are token-identical to the single engine at c1/c8/c32;
code and prose differ only at the known near-ties (code token 3, prose token 0,
which also flip between single-engine runs).

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

Measured on host-b (CMP 170HX, tok/s; c1 per request, c4/c8 full batch; image
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

Measured on host-b GPUs 1-3 (CMP 170HX), `benchmark-vllm.mjs`, 500 tokens, 2
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
host-b GPUs 1-3, tok/s (c1 per request, others full batch):

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

dspark K=5 + adaptive verification, max-num-seqs 32, KV 2.5 GiB; host-b GPUs
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
