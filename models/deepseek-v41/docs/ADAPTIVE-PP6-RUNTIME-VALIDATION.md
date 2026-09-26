# PP6 clean adaptive runtime validation (in progress)

Status: **not performance-optimized / not production approved**. Candidate
`dspark-adaptive-clean-20260924`, from local main `b9bbba3`, clean pinned
source `d63af5a`. Production original image retained but stopped during
candidate runs. Qwen remains untouched. All candidate images are test-only.

## Feature OFF gate

Image `f94fce7a0f7eba3c8463ea6c5e7091d46f484041d134599a04565f3e0e0d686c`,
standard DSpark5, adaptive false, actual PP6 model and LMCache. Exact prior
Tonyd benchmark prompts, temperature 0, cold prefixes. Warm three-repeat
mean decode tok/s vs original (historical same-host remeasurement):

| Prompt | Original | Clean OFF |
|---|---:|---:|
| ceiling count | 139.41 | 139.76 |
| coding | 115.63 | 114.91 |
| prose | 53.13 | 52.73 |

The first cold JIT count in the initial clean-OFF boot was 78.53 tok/s. It was
kept in `exact-baseline-clean-off.json`; the separate warmed run above is
`exact-baseline-clean-off-warm.json`. Do not silently average cold JIT into
steady-state numbers. Feature OFF is a separate arm from adaptive verify-all.

## Two genuine PP lifecycle defects found and corrected

1. `warmup_kernels` temporarily sets manager None to run fixed JIT steps. The
   candidate PP confidence relay remained enabled and required a confidence
   tensor, but the runner supplied None. Diagnostic replay showed
   `confidence=None, expected=(32,5)`; no private values logged. Patch 0005
   disables/re-enables relay alongside manager, restoring both on error. Eight
   success/failure/PP/on-off test cases PASS. New image reached API 200.
2. At high concurrency, the runner inserted budget/lengths into the reused
   FULL-graph `IntermediateTensors.tensors` dict. A later no-draft batch reused
   the stale keys, creating missing/unexpected budget errors and peer stalls.
   Patch 0004 was revised to construct a new send-only wrapper dict.
   `test_pp_persistent_output.py` executes the actual return block over
   budget/no-budget/budget steps; original persistent dict stays unchanged.
   Same test image repeated C12/C16/C32 workloads with zero observed PP errors.

These are fixes of protocol defects, not demonstrated speed improvements.

## Adaptive real-model results so far

Image `7a4675240149e14f46d24fd5cf760d6da4db24125a2eee1de8108e5d46a26633`
with warmup fix booted PP6, completed FULL graph capture and real requests.
Then image `ae8d79bb70b0e8148dd7efb655b2c2d6fe4a0cbbbc8989c5044bfe80240ba5f6`
(persistent dict fix) passed 4 concurrent chat requests, a 1024-token output
and two LMCache hits (3072/3712 cached), plus C12/C16/C32 Tonyd corpora.
These do not prove per-request trimming or quality equivalence.

Same exact C1 prompts, adaptive hot three-repeat decode tok/s:
- ceiling count 137.30 / 140.12 / 141.64, original ~139.41;
- coding 112.30 / 116.59 / 115.29, original ~115.63;
- prose 45.84 / 47.91 / 47.44, original ~53.13 (**regression**).

Earlier v1 C2/C4 pooled eight-category output over wall time: original
115.21/177.79, adaptive 99.59/161.96 tok/s. C6 original 225.99,
adaptive 226.36 tok/s. These were not sufficient to show net benefit.

Two fresh-prefix, full-corpus A/B pairs (`v2-clean-ab-20260925` and
`v3-clean-ab-20260925`) used the same script, deterministic tag and output
budgets on the same host, with reverse startup ordering for the second pair.
Pooled tok/s is sum of each category's completed tokens divided by sum of
its wall times; count ceiling excluded. No prefix-cache hit was allowed.

| C | v2 original | v2 adaptive | v3 original | v3 adaptive |
|---|---:|---:|---:|---:|
| 1 | 69.21 | 67.55 | 68.30 | 65.84 |
| 2 | 112.63 | 100.97 | 108.04 | 105.18 |
| 4 | 168.26 | 169.04 | 164.86 | 164.81 |
| 6 | 222.87 | 218.06 | 229.68 | 213.76 |
| 16 | 356.26 | 380.09 | 344.69 | 384.69 |
| 32 | 364.48 | 434.24 | 370.39 | 435.95 |

C16/C32 improve in both pairs; C1/C2/C6 regress or are noisy. This is
**not an across-load speedup**. Independent max-stage cost experiment did
not remove low-concurrency regression and had a large C16 outlier; it is
excluded from final patch series. Earlier source-line diagnostic probes and
startup profiling logs are not part of the shipped path.

### Historical same-prompt C1 step decomposition

No new model boot is needed to localize the C1 regression. From three warmed
repeats of the same `v1` exact prompt and metrics deltas, `request steps =
drafted/5`, `decode time = total - TTFT`, and `step/s = request steps / decode
time`. These are **request-step estimates**, not measured GPU engine cycles; C1
has one request and fixed five-token proposals, so the count is meaningful.

| C1 prompt/arm | Steps | Accepted/step | Step/s (mean) | Decode tok/s (mean) |
|---|---:|---:|---:|---:|
| count original | 41 | 4.829 | 24.02 | 139.41 |
| count adaptive final hot | 41 | 4.829 | 24.18 | 140.36 |
| code original | 41 | 3.878 | 23.82 | 115.63 |
| code adaptive final hot | 41 | 3.878 | 23.70 | 115.05 |
| prose original | 55 | 1.200 | 24.35 | 53.13 |
| prose adaptive final hot | 59 | 1.051 | 23.53 | 47.85 |
| prose adaptive forced-full diagnostic | 55 | 1.200 | 24.10 | 52.59 |

Prose's ~9.9% tok/s loss is chiefly an accepted-output-per-step loss:
59 instead of 55 steps for the same 121 output tokens (~6.8% fewer output
tokens/step), plus ~3.4% fewer request steps/s. This is direct evidence that
saving target rows alone did not compensate for shortened speculative chains
and control/dispatch overhead. The public `draft` counter is proposals, not
admitted target rows. For concurrent batches `draft/5` counts summed
request-steps, **not** global engine steps; do not report it as an engine
step/s without batch-level instrumentation.

### Historical concurrent request-step decomposition

The same v2/v3 eight-category A/B results also contain server draft and
accepted deltas. The ratios below are **summed request-steps**, not engine
batch-step/s: `sum(draft)/5 / sum(batch wall seconds)`. Parallel requests
make request-step/s potentially much larger than a single engine cycle rate.
`accepted/step` is accepted drafts per request-step; `output/step` includes
approximately one target bonus token. Output totals/usage can differ slightly
across arms and prompts near ties. Thus this table isolates the acceptance
versus throughput trade-off but does not measure actual pipeline step/s.

| Pair/C | Original accepted/step | Adaptive accepted/step | Original request-steps/s | Adaptive request-steps/s | Original tok/s | Adaptive tok/s |
|---|---:|---:|---:|---:|---:|---:|
| v2/C1 | 2.214 | 2.236 | 21.53 | 20.84 | 69.21 | 67.55 |
| v2/C2 | 2.218 | 2.118 | 35.00 | 32.37 | 112.63 | 100.97 |
| v2/C4 | 2.259 | 2.075 | 51.58 | 55.10 | 168.26 | 169.04 |
| v2/C6 | 2.244 | 2.077 | 68.66 | 70.85 | 222.87 | 218.06 |
| v2/C16 | 2.245 | 1.968 | 109.66 | 128.09 | 356.26 | 380.09 |
| v2/C32 | 2.235 | 1.769 | 112.70 | 156.94 | 364.48 | 434.24 |
| v3/C1 | 2.152 | 2.138 | 21.63 | 20.98 | 68.30 | 65.84 |
| v3/C2 | 2.223 | 2.150 | 33.49 | 33.42 | 108.04 | 105.18 |
| v3/C4 | 2.186 | 2.017 | 51.69 | 54.63 | 164.86 | 164.81 |
| v3/C6 | 2.242 | 1.960 | 70.77 | 72.24 | 229.68 | 213.76 |
| v3/C16 | 2.228 | 1.959 | 106.81 | 130.30 | 344.69 | 384.69 |
| v3/C32 | 2.226 | 1.750 | 114.91 | 158.77 | 370.39 | 435.95 |

At C32 the lower acceptance is offset by ~39% more *request-steps per
second*; at C6 only ~2–3% more request-steps/s cannot offset the reduced
output/step. This explains why the same implementation helps C32 yet hurts
C6 without blaming graph dispatch alone. An actual engine batch-step count
needs batch-level instrumentation; cannot be reconstructed from these
aggregate Prometheus counters.

A separate diagnosis forced full draft budget for batches with <=4 requests.
Exact C1 prose recovered from ~47.85 to 52.6 tok/s (original ~53.13),
and all six deterministic 160-token/stop test outputs matched original.
But fresh-prefix C6 eight-category throughput was 186.74 (v3) and 214.55
(v4), so this is **not a safe final policy**. A further <=8 test-only gate
showed volatile C16 results: pooled 219.12 in v5 versus 356.30 on the v6
repeat, with no PP error; this is not an acceptable speed claim or a
production policy. Neither heuristic is in the formal patches. Exact C1
forced-full outputs matched the original on all six sampled deterministic
prompts, whereas the adaptive-trimmed arm differed on code/prose/JSON;
the latter was repeatable but the teacher-forced oracle near those choices
was sensitive to numerical/context differences. Strict greedy equivalence
is still unverified.

Server `draft` counter counts *proposed* tokens, not admitted target rows;
acceptance counters alone cannot establish that verification physically shrank.
A separate diagnostic image `e59ffe823a39ed1de4fa691a532d06b198020d008a27f0bcd7be47fdd070db6a`
counts scalar budget decisions without printing private tensors or changing
normal candidate hot path. On the first 400 exact-benchmark steps, 130 were
trimmed; 1,731 of 2,000 scheduled drafts were admitted (269 rows removed).
After a mixed C4 corpus and subsequent exact run, 1,600 accumulated steps
showed 664 trimmed and 9,431 of 12,540 scheduled drafts admitted (3,109
removed). These counters are first-rank budget decisions; the V2 runner
constructs an actual `num_tokens` and compacted input from those decisions,
but a stage-by-stage physical-row trace is still pending. No win follows
from trimming alone.

## Pending performance and correctness gates

- Confirm actual budget/admitted distribution and graph descriptor buckets
  through non-perturbing counters, without treating acceptance as admission.
- Verify PP cost model under concurrency; startup sum-of-six-stage GPU event
  timings (C1 ~20–27 ms) do not represent pipeline cycle occupancy (per-stage
  ~3–6 ms for 1–6 graph rows). Test a separate max-stage-cost candidate
  against unchanged original and sum-cost candidate; do not equate a better
  scalar proxy with a proven speedup. No arbitrary threshold tuning.
- Matched fixed/adaptive full benchmark across C1, C2, C4, C6 and C8–C32,
  with step/s, accepted/step, physical rows, output tok/s and no errors.
- Heterogeneous partial budgets on model, rejection KV rollback, logprobs,
  prefill/mixed batches, forced graph bucket changes. Component tests alone
  are insufficient. The candidate did pass 9 cancellations + 9 slot-reused
  replacement requests, 1024-token output + two LMCache hits (3072/3712
  tokens), C12/C16/C32 runs and logprob API smoke, but no strict model-wide
  KV/logprob equivalence claim follows.
> **Superseded 2026-09-26.** Greedy token-ID equality is not a decidable gate on this
> deployment: two runs of the *same* configuration disagree at concurrency > 1, and
> plain decode without any speculative decoding shows the same per-position
> below-argmax rate as DSpark (1.56% both). See
> `docs/GREEDY-OUTPUT-ARGMAX-AUDIT.md`; the live gate is `tests/argmax-audit.py`.
> The bullet below is kept as written for the record.

- Six temperature-0, seed-0, fresh-prefix original/adaptive token-ID pairs:
  count, math and table matched; code diverged at token 23, prose at 5,
  JSON at 9. Repeating the adaptive arm yielded identical IDs. A
  teacher-forced one-token oracle at the first divergent prefix was itself
  unstable for code (7/8 chose original, 1/8 adaptive), and prose picked a
  third token on all 8 trials; candidate logprobs ranked its selected tokens
  highest but by small margins. This does **not** establish that candidate
  outputs are incorrect, nor prove strict greedy equivalence. Investigate
  numeric/context sensitivity before production acceptance.
- If no measured improvement above original without quality regression, leave
  adaptive disabled. Do not call a no-regression fast path a performance win.
