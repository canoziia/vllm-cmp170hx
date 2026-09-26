# Adaptive verification series (compiled into the default build, off at runtime)

`patches/adaptive/series` is applied by `scripts/build-deepseek-v41-image.sh` by
default. It stays inert because `enable_adaptive_verification` defaults to `false`
in the speculative config and every hook the series installs is behind an adaptive
check, which the build now asserts. Opt out with `ENABLE_ADAPTIVE_VERIFICATION=0`.

Enable per deployment:

    --speculative-config={"method":"dspark","num_speculative_tokens":5,\
      "use_local_argmax_reduction":true,"enable_adaptive_verification":true}

### Why shipping it by default is inert (asserted at build time)

* `AttentionBackend.supports_device_cpu_query_lens_mismatch()` answers `True` on
  the SM80 family, but is only consulted under `use_adaptive_verification` in
  `vllm/v1/attention/backend.py` and from the single caller in
  `adaptive_verification.maybe_create_...`;
* `PPHandler.relay_draft_confidences` is set only under
  `if self.use_pp and self.adaptive_verification is not None:`;
* the PP receive path unwraps `__verification_budget`/`__verification_lengths`
  only when they were published, which only the adaptive branch does;
* `enable_adaptive_verification: bool = False` in `vllm/config/speculative.py`.

`scripts/apply-deepseek-v41-patches.sh` greps for each of these and fails the
build if one disappears, so the inertness cannot rot silently.

### Ordering and the tracing image

The adaptive series is applied before `optional/series.perf-debug`, and the
tracing package is maintained against the adaptive-modified `PPHandler.receive()`
and draft-propose code (they previously collided in either order at
`pp_utils.py:58-67` and `model_runner.py:1573`). Consequence: the tracing image
carries adaptive too, and `ENABLE_PERF_DEBUG=1` with
`ENABLE_ADAPTIVE_VERIFICATION=0` is refused up front instead of failing
mid-patch.

Merged semantics worth knowing when reading either layer: a hot K=0 step skips the
draft tokens *and* the confidence columns of the same broadcast, publishes no
drafts, records no confidence, and the verification budget is clamped to the
drafts the scheduler actually scheduled - so the scheduler, not the manager, is
the single source of truth for K. That clamp is `0006` in this series, not part
of the tracing layer: it is a bug in the adaptive integration itself (the last
rank published `batch_draft_budget` without reconciling it against
`scheduled_drafts`, so a zero-draft step was read as a partial allocation
demanding device lengths for drafts that do not exist) and the production build
must carry it. The build now greps for it.

### What is upstream's and what is ours (attribution, since PP+adaptive is not an upstream feature)

`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` is upstream code (Apache-2.0,
"contributors to the vLLM project") and is already present in the pinned source. The
scoring and allocation policy - rank every (request, step) slot by survival
probability and admit the global top-k - and the cost curves built from replaying
captured CUDA graphs at startup are theirs, and so is the budget objective:

    draft_budget = argmax estimated_accepted_tokens / (draft_cost_ms + verify_cost_ms)

Upstream, however, refuses to run any of it under pipeline parallelism. The pinned
source says so directly:

    if self.parallel_config.pipeline_parallel_size > 1:
        # Cost curves and confidences currently only exist on the last PP rank;
        # earlier ranks would diverge on the trimmed batch shape.
        # TODO: we should be able to support adaptive verification with PP by
        # broadcasting the cost curves and confidences to all ranks.
        raise ValueError("Adaptive verification is not currently compatible with
            pipeline parallelism")

Everything that makes it run here is ours: 0001 (SM80 device-ragged backends), 0002
(authoritative per-step budget input), 0003 (confidence relay packed into the existing
draft feedback collective), 0004 (budget relay integration on the V2 runner), 0005
(relay/warmup pairing), and the replacement of that raise with a narrow allowlist
(SM80 + DeepSeek-V4.1 + DSpark greedy drafting + TP1/DP1/PCP1/DCP1 + no ubatching).
In other words, this branch implements the TODO in that comment, so every adaptive
performance number measured on PP6 - both the prose wins and the counting loss -
characterises our port, not an upstream deployment.

That distinction also assigns the two defects found so far:

* **0007 (graph padding) is entirely ours.** `varlen_decode` is set by our PP path,
  and its interaction with the upstream bucket list is what suppressed the uniform
  decode graphs. Upstream never reaches this combination because it never runs
  adaptive with PP.
* **The missing fixed-cost term is ours to fix, not an upstream bug.** The objective
  prices only drafting and target verification, which is a good approximation of a
  step on the single-stage configuration upstream targets: there the step is
  essentially draft plus verify. Under PP6 the unpriced part - inter-stage transfer
  and pipeline wait - is about 70% of a c1 step (42.6 ms period against 12-15 ms of
  priced work) and still about a third at c32. Moving the algorithm outside the
  regime its cost model assumed, without extending that model, is what produces the
  systematic under-verification at low concurrency. 0008 (price the fixed per-step
  cost) is the completion of our port.

### 0007: uniform decode graphs alongside varlen (why "on but trimmed nothing" was slow)

`CudaGraphManager._init_candidates` treated uniform-decode and varlen-decode
capture as mutually exclusive. Because the manager sets `varlen_decode=True`
whenever adaptive verification is enabled, the uniform branch was skipped and the
only decode graphs left were the varlen buckets (`[1,2,4]` + multiples of 8) -
which do not contain 6x6=36. So a step that the manager decided not to trim at all
still ran the 40-row graph: 11% extra verification work, purely from enabling the
feature. That is the mechanism behind the -12..-15% counting regression measured
below, not manager CPU cost and not the V1 cudagraph downgrade (#49986/#49548),
which cannot fire here because we run the V2 model runner.

0007 captures the uniform decode graphs alongside the varlen set for the row
counts no varlen bucket covers exactly (17 new graphs here, 36 among them). They
are prepended only to the row counts they can serve and kept out of the
range-partitioning, because inserting them there would let a uniform graph claim
the counts under its bucket and starve ragged steps of their varlen graph,
dropping those steps to eager. `_is_compatible()` still refuses a uniform graph
unless the step is uniform at that width, so ragged steps are unchanged.

### Measured cost of shipping it (runtime flag off)

Combined tracing image `comb-debug-ce386ad` (adaptive + tracer + cohort
balancing) versus the port-only image `upp-debug-b6b6ea2`, same protocol
(`balance=1`, DSpark on, 3 runs per cell, medians):

| cell | port only | combined | delta |
|---|---|---|---|
| prose c16 | 351.2 | 351.6 | +0.1% |
| prose c32 | 578.3 | 609.4 | +5.4% |
| counting c16 | 1073.9 | 1093.4 | +1.8% |
| counting c32 | 1700.2 | 1695.0 | -0.3% |
| counting sweep c1..c32 | 150/255/363/603/1074/1733 | 149/256/412/788/1093/1724 | mid-range better |

Acceptance is unchanged (prose 2.31-2.35, counting 5.952) and the composition
cell keeps the policy shape (avg 5.11 requests/step, median 6, 21.6 vs 21.7 ms),
with residual starved steps down from 19.3% to 4.4%. The mid-concurrency counting
increase here, versus our own variant's numbers earlier, confirms those single-run
sweep points were noise rather than a build property.

The rebased hot-K path was re-checked end to end on the combined image: 500
tokens at concurrency 1 gives 62.0 (K=5) / 41.3 (K=0) / 61.7 (K=5) tok/s,
matching the pre-rebase 62.2 / 41.9 / 62.4.


This directory is a replacement design, not an extension of the previous
experimental stack. The checkpoint below is candidate V2 PP integration; it is
compiled in but not yet a validated serving feature (see the gates at the end).

`0001-sm80-device-ragged-backends.patch`:
- Scoped V4.1 exact-SM80 adaptive metadata builders reuse existing MLA/SWA code.
- Advertise the tested persistent device-ragged FULL capability only under
  the standard adaptive flag; fixed/other hardware retains original support.
- V4.1 indexer accepts device/CPU length mismatch on SM80, reusing its existing
  flattened preparation. Depth=1 also explicitly chooses flattened metadata
  under adaptive (native fixed next_n=2 cannot represent arbitrary boundaries).
- No new kernel, no generic platform capability modification, no PP controller,
  no custom plan envelope, no environment runtime switches.

Validation (isolated GPU, no production patch):
- Existing MLA/SWA builders: 40 eager+graph numerical cases across compression
  ratios 1/2, padded requests, layout and boundary changes.
- Actual complete indexer builder: 10 layouts across ratio1/2; real MQA logits
  and top-k eager/graph checks.
- Same tests on scoped adaptive backend overlay plus policy checks passed.
- Existing compressor/insert: 96 speculative/rejected-prefix cycles, exact
  valid cache bytes vs accepted-prefix-only stream; no rollback cleanup.
- Two-GPU inline metadata transport: four real roundtrips; host budget readable
  before waiting on CUDA receive, no additional CPU tensor send/handle.

`0002` extends the existing manager (no new plan/buffer class) with optional
CPU authority budget and device capacities; preserves its full/zero paths,
logit-size limit and stale confidence buffers. Adds row-wise confidence publish
for the PP feedback callback and stable prefix tie ordering.

`0003` packs FP32 confidence bits into additional int64 columns of the existing
draft collective. No extra collective. Existing FIFO liveness filters both;
callback directly feeds manager state. Twelve real 2-GPU FIFO cycles PASS.

`0004` threads budget/lengths through GPUWorker/V2 gather/prepare. CPU integer
budget rides existing object metadata; partial budgets carry a device vector.
The send-side FULL graph output is persistent: wrap it in a fresh send-only
`IntermediateTensors` instead of mutating its reused `.tensors` dict. The
first candidate failed under C12 with stale budget presence; this fix passed
budget/no-budget/reuse runner-block test and subsequent C12/C16/C32 model runs.
Original adaptive config/sampler semantics stay enabled. Startup local stage
profiles are gathered into sum-of-stages cost, with last-rank drafter cost;
this is a low-concurrency estimate, not a validated concurrency cost model.
SM80 V4.1 DSpark greedy/TP1/DP1/CP1/no-ubatching is the candidate config scope.
GPU input tests (8), logprob boundaries (4), manager budgets (8), Worker control
boundary tests (8 CPU fake-transport cases), and real inline transfer + manager
layout graph checks (4 roundtrips) PASS. Full execute_model is not yet verified.

Transport-only 2-GPU microbenchmark (median of 4 blocks): six-row hidden-state
roundtrip .7815 ms, inline budget .7800, inline budget+GPU lengths .8314, old JSON
tensor+lengths .9302. At 96 rows all are ~7.34–7.38 ms. This is not PP6/model
throughput evidence and excludes confidence/allocator/model work.

`0005` pairs the existing fixed-width JIT warmup's temporary adaptive-manager
suspension with an equally temporary PP confidence-relay suspension. The
previous candidate failed in warmup with `confidence=None` while PPHandler
still demanded a `[32,5]` confidence payload. Actual warmup function control
flow was exercised for success/exception, PP on/off and relay on/off (8 cases).
PP6 model startup revalidation is pending; this is not yet a resolved E2E gate.

This does NOT establish whole-model FULL capture, KV equivalence at model level,
PP6 multi-step adaptive protocol, or mixed prefill support under adaptive FULL;
those remain enablement gates before the feature is turned on for a deployment.
Feature-off performance is now measured (table above): shipping the code costs
nothing measurable. What has NOT been done in this session is running a serving
test with `enable_adaptive_verification: true`, so the series is compiled in but
still unvalidated in production traffic.
