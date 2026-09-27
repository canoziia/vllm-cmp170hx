# PP decode-cohort balancing (shared patch `patches/vllm/0003`)

## What it changes

With the V2 model runner, pipeline parallelism and async scheduling, a decode
request may only advance once every `pp_size` scheduler steps: the worker-side
sampled-token broadcast ring consumes a step-`T` result at `T + pp_size`
(vllm-project/vllm#42187). Nothing in the scheduler spread the running set over
those in-flight batches, so requests whose prefills complete together lock into
one cadence group, advance together, then starve together while the other batches
run nearly empty. #50853 lists this as an open gap.

`patches/vllm/0003-balance-async-pp-decode-batches.patch` is a **port of
vllm-project/vllm#57433**, which caps one scheduler step to
`ceil(max_num_seqs / pp_size)` *established decode* requests
(`num_computed_tokens >= num_prompt_tokens`, evaluated before scheduling).
Prefill, chunked prefill and the chunk that completes a prompt are never capped,
a deferred request simply stays eligible, established decodes resumed from
`WAITING` obey the same share, a preempted request gives its cohort slot back, and
the step ends with `assert len(scheduled_decode_req_ids) <= max_num_scheduled_decodes`.
The cadence, the ring and the collective order are untouched.

One deviation from upstream, recorded in the patch header: #57433 applies this
unconditionally for MRV2 + AsyncScheduler + PP>1, while here it is gated behind
`VLLM_PP_DECODE_COHORT_BALANCE`. The switch now defaults to `1`, on the strength
of the A/B below, and `=0` restores upstream behaviour, so rollback is one env var
plus a restart.

Recorded honestly: the Qwen PP2 deployment runs with the switch enabled and has not
been measured against it switched off, so its numbers are not evidence about this
patch - on PP2 the cap is about running/2 rather than running/6, and the effect
there is unquantified. The default rests entirely on the PP6 measurement.

## A/B result on the port — same image, only the flag differing

Six-GPU SM80, DeepSeek-V4.1-Flash, PP=6, DSpark K=5, `max_num_seqs=32`,
image `upp-debug-b6b6ea2`, 3 runs per cell, medians, tracer off, random
`cache_salt` (cold prefill), prompts and output budget identical between arms.

| Workload | concurrency | off tok/s | on tok/s | delta |
|---|---|---|---|---|
| prose (`Write a 500-word article`) | 16 | 294.4 | **351.2** | **+19.3%** |
| prose | 32 | 325.2 | **578.3** | **+77.8%** |
| counting (near-full acceptance) | 16 | 721.2 | **1073.9** | **+48.9%** |
| counting | 32 | 951.6 | **1700.2** | **+78.7%** |

Accepted tokens per request-step are unchanged by the flag (prose 2.30-2.37 both
arms; counting 5.952 both arms), so this is scheduling shape, not acceptance.
Note the off arm is not even monotonic on counting (c8 694 -> c16 721 -> c32 632 in
the sweep): past a point, adding concurrency used to *reduce* throughput.

## Does the port match what we had written before?

An earlier local variant of the same idea was measured on `hyg-debug-52db801` /
`bal-debug-7eade20`. Both medians, same protocol:

| metric | our variant (on) | port (on) | port vs ours |
|---|---|---|---|
| prose c16 | 347.5 | 351.2 | +1.1% |
| prose c32 | 614.0 | 578.3 | −5.8% |
| counting c16 | 1107.2 | 1073.9 | −3.0% |
| counting c32 | 1743.8 | 1700.2 | −2.5% |

Same direction and same magnitude band; the port sits 2-6% below our variant on the
two c32 cells. Candidate causes are the two things the port does and we did not:
capping established decodes resumed from `WAITING`, and evaluating the cap after
the other RUNNING-loop skips rather than immediately after the cadence gate. The
comparison is between different image builds, so part of that spread may be
build/session drift rather than the patch; it is reported as "equivalent within a
few percent", not as a measured penalty.

## Batch composition (worker step tracer, `sample_every=1`, concurrency 32, prose)

| | sampled steps | avg req/step | median | max | steps with <=2 | median step ms |
|---|---|---|---|---|---|---|
| off | 902 | 7.74 | 3 | 16 | **49.8%** | **33.8** |
| our variant on | 1366 | 5.06 | 6 | 20 | 20.5% | 21.2 |
| **port on** | 1360 | 5.11 | **6** | 26 | **19.3%** | **21.6** |

Dominant histogram bucket: off `{1: 235, 2: 214, 13: 205, 16: 201}` (burst then
starve) versus port `{6: 1017}` and ours `{6: 1010}`.

The mechanism is visible in the last column: an engine step costs 21-34 ms
regardless of how many requests it carries, because the pipeline round trip
dominates. Starved steps therefore threw the work away for free; pinning every
step to its share is nearly free throughput.

## Correctness

Structurally the cap only defers: prefill and admission are untouched, a deferred
request keeps its state and remains eligible, the count returns on preemption, and
the ring, cadence and collective order are unchanged.

Measured, on the port build with the flag on:

* content fixture (`tests/semantic-check.mjs`, six objectively checkable answers,
  400 tokens each): **30/30 twice at concurrency 30** with the cap binding,
  **12/12 at concurrency 1** where the cap is provably inert; the off arm also
  passes 30/30 twice;
* per-position argmax audit (`tests/argmax-audit.py`): speculative decoding shows
  the **same** below-argmax rate as plain decode (**64 of 4096 positions, 1.56%,
  in both arms**), so the cap on top of DSpark is not introducing wrong tokens.
  See `docs/GREEDY-OUTPUT-ARGMAX-AUDIT.md` for why exact token equality is not a
  usable gate on this stack and what that audit does and does not prove.

Remaining: the Qwen PP2 build of this shared series is unmeasured (the flag stays
off there), and the c32 argmax audit was stopped part-way rather than reported.

## Upstream references

* #42187 introduced the `pp_size`-step cadence; #50853 lists the imbalance as an
  open gap (its Workstream A).
* **#57433 is the fix, and what we now port.**
* #50410 caps *all* scheduled requests, which its own thread measured as −8.21%
  throughput and +11.31% mean TTFT against doing nothing; the decode-only form
  measured +22.95% there, which is the same sign and direction as our +19% to +79%.
* #53810 / #53948 defer receiver-side NCCL kernels on the same ring (+18% on a
  prefill-heavy workload); relevant when the pin moves.
* #55145 added `VLLM_XPU_PP_MICROBATCH` (XPU may set the stagger to 1); there is no
  CUDA equivalent, which is why upstream chose balancing over disabling.
