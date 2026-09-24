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

C2/C4 pooled eight-category output over wall time: original 115.21/177.79,
adaptive 99.59/161.96 tok/s. C6 original 225.99, adaptive 226.36 tok/s;
no credible net gain. C8/C12/C16/C32 candidate-only tests completed; no
matched original high-concurrency run yet. Do not compare nonidentical prompts,
seeds or phase windows as strict A/B.

Server `draft` counter counts *proposed* tokens, not admitted target rows;
acceptance counters alone cannot establish that verification physically shrank.
Budget trace was added to a separate diagnostic image only to count scalar
budget decisions without printing private tensors. It has not been read yet.

## Pending performance and correctness gates

- Confirm actual budget/admitted distribution and graph descriptor buckets
  through non-perturbing counters, without treating acceptance as admission.
- Verify PP cost model under concurrency; startup sum-of-six-stage latency may
  misprice pipeline steady-state throughput. No arbitrary threshold tuning.
- Matched fixed/adaptive full benchmark across C1, C2, C4, C6 and C8–C32,
  with step/s, accepted/step, physical rows, output tok/s and no errors.
- Heterogeneous partial budgets on model, rejection KV rollback, logprobs,
  prefill/mixed batches, cancellation/slot reuse, cache cold/hit, forced graph
  bucket changes. Component tests alone are insufficient.
- If no measured improvement above original without quality regression, leave
  adaptive disabled. Do not call a no-regression fast path a performance win.
