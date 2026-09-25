# PP decode-cohort balancing (shared patch `patches/vllm/0003`)

## What it changes

With the V2 model runner, pipeline parallelism and async scheduling, a decode
request may only advance once every `pp_size` scheduler steps: the worker-side
sampled-token broadcast ring consumes a step-`T` result at `T + pp_size`
(vllm-project/vllm#42187). Nothing in the scheduler spreads the running set
over those in-flight batches, so requests admitted together advance together
and then all wait together.

`patches/vllm/0003-balance-pp-decode-cohorts.patch` adds an opt-in cap: one
scheduler step takes at most `ceil(max_num_seqs / pp_size)` **established
decode** requests (`num_computed_tokens >= num_prompt_tokens`, evaluated before
the request is scheduled). Prefill requests, prefill chunks and the chunk that
completes a prompt are never capped; a deferred request stays eligible for the
next step; the count is restored if a selected request is preempted. The
cadence, the ring and the collective order are untouched.

Enable with `VLLM_PP_DECODE_COHORT_BALANCE=1` (default `0` = upstream
behaviour, so one image serves both sides of an A/B).

## A/B result — six-GPU SM80, DeepSeek-V4.1-Flash, PP=6, DSpark K=5, max_num_seqs=32

Same image (`bal-debug-7eade20`), only the env var differing, same prompts,
same output budget, 3 runs each, tracer off, measured back to back.

| Concurrency | off (tokens/s, 3 runs) | off median | on (tokens/s, 3 runs) | on median | delta |
|---|---|---|---|---|---|
| 16 | 270 / 356 / 361 | 356 | 351 / 340 / 347 | 347 | −2.3% |
| 32 | 254 / 255 / 288 | 255 | 615 / 614 / 603 | 614 | **+141.1%** |

Output rate including prefill (second metric, same cells): c16 325 → 320
(−1.5%), c32 241 → 545 (+126.1%). Accepted tokens per request-step are
unchanged (2.35 → 2.32 at c16, 2.32 → 2.34 at c32), so this is scheduling
shape, not acceptance.

Note the run-to-run spread also tightens at c16 (270–361 → 340–351), which is
the bimodality previously seen on this box.

## Batch composition (worker step tracer, `sample_every=1`, concurrency 32)

| Metric | off | on |
|---|---|---|
| sampled steps | 1088 | 1366 |
| requests/step, median | 2 | **6** (= `ceil(32/6)`) |
| requests/step, max | 26 | 20 |
| steps with ≤ 2 requests | 79.1% | 20.5% |
| per-request decode gap (mode) | 5 | 6 (= `pp_size`) |

Histogram of requests per step:

* off: `{1: 432, 2: 429, 26: 203, …}` — burst-then-starve.
* on: `{6: 1010, 2: 216, 1: 64, …}` — pinned at the cap.

## Scope and open items

* Model-agnostic: it lives in the shared series, and Qwen builds the same pinned
  source (`d63af5a`) through `apply-vllm-common-patches.sh`. The cost is
  proportional to `pp_size`, so Qwen at PP=2 is expected to be marginal; it has
  not been A/B'd here.
* Correctness is argued structurally (the cap reuses the existing
  `continue`-style deferral and mutates nothing but a per-step set). A strict
  per-request token-ID oracle across the two arms has **not** been run yet: the
  bench harness only records token counts, so it needs a dedicated probe.
* Upstream: the gap is listed in vllm#50853; vllm#50410 caps *all* scheduled
  requests, which its own thread measured as −8.21% throughput and +11.31% mean
  TTFT versus doing nothing. This patch implements the decode-only form
  (+22.95% in that thread's measurement, on a different model/topology).
