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
## Correctness

Structurally the cap only defers: it is the same `continue` the existing cadence
gate already uses, it mutates nothing but a per-step set of request ids, the
count is returned when a selected request is preempted, and prefill/admission
paths are untouched. Cadence, ring and collective order are unchanged.

An exact per-request token-ID oracle turned out to be **undecidable on this
deployment**, and that is worth recording on its own. `tests/token-id-probe.mjs`
captures full streamed `token_ids` (`temperature=0`, `top_p=1`, `ignore_eos`,
random `cache_salt`) and repeats runs:

| Comparison | identical outputs |
|---|---|
| c1 run vs c1 run (batch shape fixed) | **4/4 byte-identical** |
| c4 run vs c4 run (same arm, same concurrency) | 1/4 |
| c32 run vs c32 run (same arm, same concurrency) | **3/32**, divergence from token ~27 |
| c1 vs c4 (different batch shape) | 3/4, 1/4 |

So greedy output here is already non-reproducible between two runs of the *same*
configuration once more than one request is in flight; batch shape changes
logistics enough to flip argmax early in the sequence. Any OFF/ON token-equality
test would be dominated by that noise and could neither clear nor condemn the
patch. (This also means the "token IDs diverged" observations from the earlier
adaptive-verification work were not by themselves evidence about trimming.)

What is decidable is content correctness under a binding cap.
`tests/semantic-check.mjs` runs six prompts with objectively checkable answers
(count to 60, 17x23, alphabet backwards, planet order, 2048 B to KiB, geometric
continuation), 400 tokens each, answered inside the reasoning block:

| Configuration | result |
|---|---|
| concurrency 1 (cap provably inert: 1 < target 6) | 12/12 |
| concurrency 30, **cap binding**, pass 1 | **30/30** |
| concurrency 30, **cap binding**, pass 2 | **30/30** |

with the composition trace confirming the cap was in force (median requests per
step exactly `ceil(32/6) = 6`). Accepted tokens per request-step are unchanged
between arms (2.32 to 2.35), so verification semantics are intact.

Still open: this is a correctness *proxy*, not a proof of equivalence with the
unpatched scheduler for the same request stream. A strict differential test
needs a batch-invariant baseline, which this stack does not currently provide.

## Upstream references

* vllm-project/vllm#42187 introduced the `pp_size`-step decode cadence
  ("Avoid pipeline parallel bubbles").
* vllm-project/vllm#50853 lists the resulting cohort imbalance as an open gap.
* vllm-project/vllm#50410 attempts the same rebalancing but caps *all* scheduled
  requests; per that thread the global cap measured -8.21% throughput and
  +11.31% mean TTFT versus doing nothing, while the decode-only form used here
  measured +22.95% on Kimi-K3 + DSpark, TP8xPP4, max_num_seqs=32.
* vllm-project/vllm#53810 / #53948 are adjacent work on the same broadcast ring
  (deferring receiver-side NCCL kernels), worth watching when the pin moves.
* vllm-project/vllm#55145 added `VLLM_XPU_PP_MICROBATCH`, letting XPU set the
  stagger to 1. There is no CUDA equivalent, which is why this is a local patch.
