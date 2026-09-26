# Adaptive verification on PP6 — first live run

2026-09-26. Image `localhost/deepseek-v41-cmp170hx:latest-debug` (built from
main `8fddd4c`: model series + shared series incl. the #57433 cohort port + the
adaptive series + the tracing package), PP6, DSpark K=5, `max_num_seqs=32`,
`VLLM_PP_DECODE_COHORT_BALANCE=1`, engine spec config
`{"method":"dspark","num_speculative_tokens":5,"use_local_argmax_reduction":true,"enable_adaptive_verification":true}`.

This is the first time the adaptive series has run end to end on this PP6
deployment. Previously the branch record said "PP6 model startup revalidation is
pending; this is not yet a resolved E2E gate" - startup is now resolved: the
engine reached `health=200` through TileLang JIT, KV profiling, adaptive cost
profiling and FULL graph capture with zero exceptions, and served every cell
below with `num_preemptions_total = 0` and no `finished_reason="error"`.

## Result: it is workload-dependent, in both directions

Same image, same protocol (3 runs per cell for the repeat rows, medians),
comparing `enable_adaptive_verification` false (recorded in
`PP-DECODE-COHORT-BALANCE.md` / the `main-comb-on` battery) against true:

| workload | conc | adaptive off | adaptive on | delta |
|---|---|---|---|---|
| prose (low acceptance) | 16 | 351.6 | **420.4** (392/426/420) | **+19.6%** |
| prose | 32 | 609.4 | **710.0** (710/710/707) | **+16.5%** |
| counting (near-full acceptance) | 1 | 149 | 150.3 (150.6/149.7/150.3) | +0.9% |
| counting | 8 | 788 | 672 (672/697/666) | **-14.7%** |
| counting | 16 | 1093 | 941 (1046/909/941) | -13.9% |
| counting | 32 | 1724 (repeats 1706) | 1517 (1464/1519/1517) | -12.0% |
| prose | 1 | not measured in the off arm on this image | 55.9 (52.3/56.1/55.9) | see note |

Two readings, and the sign is not a coincidence:

* **prose gets faster even though acceptance drops** (2.35 -> 2.05-2.15 tokens per
  request-step). That is the intended trade: on low-acceptance content most of
  the five draft rows are rejected anyway, so not verifying them removes work
  without giving up emitted tokens. It also refutes the earlier assumption that
  verification-row count is irrelevant on this box - it is irrelevant when the
  rows would have been cut short anyway, and it is not irrelevant when a step
  carries six requests times six rows.
* **counting gets slower by a flat ~12-15%** at every concurrency from c8 to c32,
  with acceptance pinned at 5.952 of 6 in all of them. So the loss is manager
  overhead on a workload that has nothing to trim, not trimmed acceptances.

### Correction: the first sweep's low-concurrency numbers were cold-start artifacts

My first adaptive-on pass was an inline ascending sweep (`c1,2,4,8,16,32`) with no
discarded warm-up cell, and it gave counting c1 = 124.5 (-16.4%), c8 = 511.9
(-35%) and acceptance 5.708 at c8. Re-measured with an explicit warm-up cell and
three repeats per concurrency, those became 150.3, 672 and 5.952. The first cell
after a boot is still paying cold costs, and in an ascending sweep that penalty
lands on the smallest - and therefore most per-step-sensitive - cell, which is
where it looks worst. The A/B batteries in this repo always ran a discarded
warm-up first for exactly this reason; skipping it here was my error, and it
produced three numbers that were wrong enough to change the story (it looked like
adaptive was cutting accepted rows, and like the curve was badly non-monotonic).

Rule recorded: **never report the first cell after a boot.** Every adaptive-on
number in the table above comes from a warmed run with 3 repeats.

Still open on the c1 prose row: the off-arm reference for `prose c1` on this image
was never measured (the batteries only covered c16/c32), so 55.9 tok/s can only be
compared against the older non-debug main figure of ~57 - within noise, but not a
controlled comparison.

## Full prose sweep with the feature on (c1..c32)

Warmed, 3 repeats per concurrency, same instance as above:

| conc | aggregate tok/s (median) | runs | per-request tok/s | step/s | ms/step | accepted/step |
|---|---|---|---|---|---|---|
| 1 | 52.3 | 56/52/52 | 52.30 | 23.48 | 42.6 | 2.222 |
| 2 | 94.5 | 95/94/94 | 47.17 | 21.17 | 47.2 | 2.222 |
| 4 | 156.6 | 155/157/160 | 38.83 | 17.61 | 56.8 | 2.202 |
| 8 | 259.7 | 225/264/260 | 32.47 | 14.80 | 67.6 | 2.171 |
| 16 | 416.3 | 433/416/412 | 25.56 | 11.92 | 83.9 | 2.153 |
| 32 | 712.1 | 708/712/717 | 21.94 | 10.55 | 94.8 | 2.074 |

Against the adaptive-off arm on this image (measured only at c16/c32): **+18.4%**
and **+16.9%**, matching the +19.6%/+16.5% of the earlier warmed run, so the prose
gain is reproducible. Against the older non-debug, unpatched-main curve
(1:57, 2:96, 4:150, 8:188, 16:257, 32:260) the shape is what matters: the curve is
now monotonic to c32 (+174% at c32) where it used to flatten at c16.

Open question, deliberately unresolved: prose at c1 measured 55.9 and then 52.3 on
two independent warmed triples, i.e. 3-8% under the historical 57, while counting
at c1 was neutral (150.3 vs 149). There is no adaptive-off prose c1 cell on this
image, so this cannot be attributed between the debug image, manager overhead on a
single stream, or noise. Resolving it needs one boot with the feature off to fill
c1/c2/c4/c8 for prose.

## Recovered 09-24 raw data (cohort balancing did not exist yet)

Raw per-category results are on the server under
`/root/app/dspark-adaptive-dev/results/bench-{original,adaptive}-v{2,3}-c12461632.{json,md}`
(concurrencies 1/2/4/6/16/32, nine categories, `temperature=0`, thinking off,
identical prompt set across boots, `cached_tokens=0`). The pooled `C1..C32` table
quoted in `ADAPTIVE-PP6-RUNTIME-VALIDATION.md` averages eight of them and excludes
`ceiling_count`, which is why per-category reading matters here.

Aggregate tok/s for `ceiling_count` (1..80, 239 completion tokens per request):

| arm | C1 | C2 | C4 | C6 | C16 | C32 |
|---|---|---|---|---|---|---|
| original v2 | 131.0 | 226.5 | 361.5 | 404.6 | 827.3 | 697.1 |
| adaptive v2 | 130.2 | 225.8 | 288.2 | 412.8 | 695.2 | 856.6 |
| original v3 | 131.7 | 225.9 | 342.1 | 525.4 | 872.5 | 691.0 |
| adaptive v3 | 129.9 | 223.9 | 341.8 | 418.6 | 710.1 | 628.0 |

Round-to-round deltas: v2 `-0.6 / -0.3 / -20.3 / +2.0 / -16.0 / +22.9 %`,
v3 `-1.4 / -0.9 / -0.1 / -20.3 / -18.6 / -9.1 %`. But the two **original** arms
disagree with each other by up to 30% at C6 (404.6 vs 525.4) and 5.5% at C16, so
most of those deltas sit inside the same-arm cross-run spread. Honest reading for
counting with the balancing off: **unresolved - the instrument is too noisy at
these cell lengths**, and the sign flips between rounds. This is a short
completion (239 tokens) dominated by queueing and TTFT, unlike this session's
`counting` cell (1000 tokens per request, three warmed repeats, clean -12 to -15%).

Aggregate tok/s for `prose`:

| arm | C1 | C2 | C4 | C6 | C16 | C32 |
|---|---|---|---|---|---|---|
| original v2 | 49.9 | 79.6 | 114.8 | 162.3 | 292.7 | 285.9 |
| adaptive v2 | 50.0 | 74.7 | 128.8 | 156.3 | 287.3 | 391.7 |
| original v3 | 48.4 | 73.1 | 121.6 | 173.0 | 268.8 | 243.5 |
| adaptive v3 | 48.3 | 74.6 | 134.5 | 171.7 | 368.0 | 356.8 |

Deltas: v2 `+0.2 / -6.2 / +12.2 / -3.7 / -1.8 / +37.0 %`, v3 `-0.2 / +2.1 / +10.6 /
-0.8 / +36.9 / +46.5 %`. Here the two independent rounds **agree** at C4 (+10.6 to
+12.2%) and C32 (+37 to +46.5%), and those exceed the original-vs-original spread
at the same cells. So adaptive helping prose at higher concurrency is corroborated
across two separate weeks and two different builds, and it does not depend on the
cohort balancing.

Correction to what this document said before: it claimed the 09-24 data had no
counting measurement above C1. That was wrong - I had only looked at the pooled
table and the C1 step-decomposition in the markdown, not at the raw JSON, which
carries all nine categories at all six concurrencies. An earlier in-chat claim
that the old `+19% at C32` was counting-driven was also wrong: the pooled headline
excludes counting entirely.

## Correctness with the feature on

`tests/semantic-check.mjs` (objectively checkable answers, 400 tokens each,
temperature 0): **30/30 at concurrency 30** and **12/12 at concurrency 1**.
No errors, no preemptions, engine healthy throughout.

The per-position argmax audit (`tests/argmax-audit.py`) has **not** been run on
adaptive-on sequences yet. Until it is, this doc does not claim the trimming is
token-equivalent; it claims the content fixture passes. The reference floor for
that audit is 1.56% below-argmax positions (64 of 4096) for plain decode, so an
adaptive arm materially above that would be a real fault - see
`GREEDY-OUTPUT-ARGMAX-AUDIT.md`.

## Consequence for the default

The feature stays **compiled in but runtime-off**. It is not a candidate for a
deployment-wide default: the same flag that gives +16-20% on prose costs -12 to
-35% on high-acceptance traffic, and a single switch cannot serve both. Any
rollout has to be per workload (or driven by the confidence signal it already
has), and it needs the argmax audit plus the c4/c8 instability resolved first.

## How this run was made reproducible

The switch lives in the engine command. At the time of the run it was set by a
one-line edit to `compose.lmcache.yml` (then the file defining the LMCache
deployment; `compose.yml` was a different, LMCache-less definition and editing it
did nothing). That overlay has since been merged into `compose.yml` and
`enable_adaptive_verification` is now interpolated from
`${VLLM_ADAPTIVE_VERIFICATION:-false}`, so switching it is a `.env` edit.

## Other people hit exactly this (upstream corroboration)

The workload-split result is not specific to our box; it is the reported behaviour
of dynamic/adaptive speculation in vLLM generally:

* **#49986 - "DSD arms pay a large baseline tax vs no-spec under production
  defaults"** (H100 NVL, gemma-4-31B FP8): every configuration carrying a
  `speculative_config` is slower than no-spec at short context, `-21%` for static
  K=3 and `-27..-31%` for dynamic schedules at ctx 400, crossing over to `+9..+10%`
  only by ctx 4000. They attribute part of the tax to a
  `FULL_AND_PIECEWISE -> PIECEWISE` CUDA-graph downgrade.
* **#49548 - "Dynamic speculative decoding causes catastrophic aggregate-throughput
  collapse under concurrency at the batch-size threshold"** - on a **DGX Spark
  (GB10)**: an expected ~14% single-stream cost from the graph downgrade, plus an
  8-concurrent workload going from 232 tok/s (static K=2) to 24-157 tok/s with a
  schedule that disables speculation above batch 4. Their point matches ours: the
  intuitive assumption "a zero-K step just falls back to non-spec speed" is wrong.
* **#49369 - "DSpark much slower than no-spec on single B300 (DeepSeek-V4-Flash)"**:
  DSpark on roughly **halves** aggregate throughput versus no speculation **with
  healthy acceptance** - the same signature as our counting arm (acceptance 5.952
  unchanged, throughput -12..-15%). That issue also notes prefix caching had to be
  disabled there because acceptance collapsed with it on (#47930); our deployment
  runs prefix caching + LMCache and did not see that collapse.
* **#51303 - "Adaptive DSpark Bring-Up Tracker"** (open): the official tracker
  states adaptive verification needs backends that treat GPU tensors, not CPU
  query lengths, as the source of truth, and only lists FLASH_ATTN and DSV4
  attention as supporting per-request variable decode lengths. It also documents
  the startup cost profiling as "known to have some drift" and calls out the
  dead-spot / exploration-vs-exploitation failure mode - which is the most likely
  explanation for our non-monotonic counting curve (+15.5% at c4, -35% cold at c8,
  settling at -12..-15% warm).
* Related prior art in the same direction: #44336 (Adaptive K* from per-position
  acceptance rates, closed), #35301 (dynamic speculation length with
  confidence-threshold early exit), #53987 (entropy-gated deferred verification),
  and on the SGLang side #28045 (throughput-aware policy for cost-guided adaptive
  steps) and #39231 (stabilising the adaptive tier vote with a shared EMA and a
  minimum dwell) - the latter two exist precisely because tier selection is noisy.

Decision recorded: `enable_adaptive_verification` stays **false by default** in
`compose.yml` (and the deployment `.env` now states it explicitly). It is enabled
deliberately per deployment, only where the content is acceptance-poor, because on
this box the measured trade is prose +16-19% against counting -12-15%.

## Resolved: the graph-mode downgrade does not apply to this deployment

Upstream's mechanism is a deliberate rule, `VllmConfig._maybe_override_dynamic_sd_cudagraph_mode`
(`vllm/config/vllm.py`): if the config uses dynamic speculative decoding, the
cudagraph mode has full graphs, and **the V1 model runner is in use**, force
PIECEWISE, logging "Dynamic speculative decoding changes the target verification
length at runtime. Overriding cudagraph_mode ... for reliability. Use
VLLM_USE_V2_MODEL_RUNNER=1 if you want to use full CUDA graphs." The reason is
that a FULL replay bakes per-shape metadata, so a verification length that
changes at runtime would be checked against the wrong widths - upstream picks
reliability over graph coverage there, and names the V2 runner as the way out.

That rule cannot fire on this box, and three independent checks agree:

* we already run the V2 model runner, which is the early-out in the rule itself
  (the whole `v1/worker/gpu/` stack, and the step tracer, are V2-only);
* the resolved config keeps `cudagraph_mode=FULL_AND_PIECEWISE` with
  `mode=CompilationMode.NONE` (breakable CUDA graphs force NONE, so Inductor is
  not in play at all here);
* the boot log contains **zero** "Overriding cudagraph_mode" lines, and
  `LMCacheMPConnector` does not declare `requires_piecewise_for_cudagraph`, so the
  other PIECEWISE rule (KV-connector layerwise async ops) did not fire either;
* capture actually ran both phases: `Capturing CUDA graphs (FULL)` x17 and
  `(PIECEWISE)` x32.

So the -12..-15% on counting is **not** lost graph coverage. Remaining candidates,
in the order I would test them: per-step manager work on the critical path
(budget publication, `partial_capacities`, the compact/lengths handling), the
doubled draft-broadcast payload when confidence relaying is on, and cost-profile
drift from startup replay (#51303, #52057). Distinguishing them needs the tracer
across an adaptive-off boot, which is the natural next measurement.

## 0007 measured on PP6 (adaptive ON, image `cbf8c4e-debug`, 3 warm repeats)

Dispatch confirmation from the step tracer, before any throughput claim:

| step kind | descriptor (before fix) | descriptor (after fix) | wasted rows |
|---|---|---|---|
| full-budget counting step, 6 reqs x 6 rows | `num_tokens=40, uniform=None`, real 36 / padded 40 | `num_tokens=36, uniform=6`, real 36 / padded 36 | 4/step -> **0** |
| trimmed prose step | `num_tokens=16`, real 16 / padded 16 | unchanged | 0 |

Across 1619 traced steps the total padded-but-unneeded verification rows went from
~2712 to **0**, with no step falling back to eager and no ragged step changing
graph (as the offline dispatch simulation over all 192 reachable row counts
predicted: 17 counts improved, 0 regressed).

Throughput, `full_batch_tok_s` medians of 3 warm repeats, same image and same
deployment, compared against the adaptive-on run before 0007 and against the
session's adaptive-off baseline:

| cell | after 0007 | vs adaptive ON before 0007 | vs adaptive OFF | acceptance |
|---|---:|---:|---:|---:|
| counting c32 | **1639.7** | **+8.1%** | -4.9% (was -12.0%) | 5.930 of 6 |
| counting c16 | 979.9 | +4.1% | -10.4% (was -13.9%) | 5.939 |
| counting c8 | 452.2 | see note | see note | 5.952 |
| prose c32 | 690.6 | -2.3% | +14.2% | 2.065 |
| prose c16 | 411.7 | -1.1% | +17.0% | 2.118 |

So the padding was a real component of the counting regression: at c32 it accounts
for roughly half of it, and the residual -4.9% is where the remaining candidates
live (2x wide draft broadcast while confidence relaying is on, the per-step varlen
index build, and cost-profile drift).

Honest limits on this table:

* **counting c8 is not usable as evidence.** Within this single boot the three
  repeats were 447/452/552 (23% spread), and c4/c8 counting has been unstable in
  every previous boot too (the 09-24 raw data swings -20%..+23% at those cells).
  I am not claiming the fix helped or hurt c8.
* The adaptive-off column comes from a different boot of a different image, so its
  uncertainty is not zero. A same-image off baseline (the deployment flag is now
  back to false, so the next start provides it) is the clean comparison.
* prose is within noise of the pre-fix value (repeats were tight, +-0.7%): the fix
  removes padding on steps that were not trimmed, and prose mostly trims, so there
  was nothing for it to recover - which is the expected result, not a coincidence.

## Same-image A/B against the 0007 build, and three questions closed at once

Restarted the identical image (`cbf8c4e-debug`, `b4f5839d6445`) with
`VLLM_ADAPTIVE_VERIFICATION=false` to get the baseline that previous tables had to
borrow from other boots. Cold cell discarded, tracer disabled for all throughput
cells, 3 warm repeats per cell (1 for the extra prose points), `full_batch_tok_s`.

| cell | adaptive OFF | adaptive ON + 0007 | ON vs OFF | OFF acceptance |
|---|---:|---:|---:|---:|
| counting c32 | 1722.1 (spread 2.4%) | 1639.7 | **-4.8%** | 5.952 |
| counting c16 | 1064.2 (5.0%) | 979.9 | -7.9% | 5.952 |
| counting c8 | 667.2 (**25.9%**, reps 595/667/768) | 452.2 (447/452/552) | -32.2%, see note | 5.952 |
| prose c32 | 609.0 (2.2%) | 712.1 | **+16.9%** | 2.347 |
| prose c16 | 349.4 (0.7%) | 416.3 | **+19.1%** | 2.327 |
| prose c8 | 213.7 | 259.7 | +21.5% | 2.300 |
| prose c4 | 163.3 | 156.6 | -4.1% | 2.341 |
| prose c2 | 102.1 | 94.5 | -7.5% | 2.283 |
| prose c1 | 56.3 | 52.3 | -7.0% | 2.315 |

**Closed: the counting c4/c8 instability is not caused by adaptive verification.**
The OFF arm - no adaptive manager constructed at all - swings 595/667/768 (26%) on
counting c8 while reporting exactly 5.952 accepted tokens per step every time. A
pure acceptance-stable, throughput-unstable cell in the base deployment, at a
cohort size of ceil(8/6)=2 requests per step. It had been on the open list as
"suspect startup cost profiling drift (#52057)"; that hypothesis is wrong for this
cell, since the profiler does not exist in this arm. What is still true: at c8 every
adaptive repeat sits below every OFF repeat, so low concurrency pays the manager's
per-step fixed cost against a very small GPU payload.

**Closed: the ~12 ms in `state_updates` is relocated waiting, not work.** Comparing
counting@32 traces at matched content, matched request count and matched real rows
(36 both sides), OFF vs ON:

| span, median ms | OFF | ON | delta |
|---|---:|---:|---:|
| `state_updates` | 0.37 | 12.22 | **+11.85** |
| `execute_model` | 5.64 | 18.83 | +13.20 (contains the above) |
| `sample_tokens` | 6.91 | 1.79 | **-5.13** |
| GPU `feedback_receive` | 21.58 | 24.26 | +2.67 |
| GPU `target_forward` | 16.63 | 14.39 | -2.25 |
| **step period** | **21.55** | **21.86** | **+0.31 ms (+1.4%)** |

A span can grow by 11.85 ms while the step it lives in takes 1.4% longer, because
the span is where the CPU parks while waiting for the pipeline. In the OFF arm the
parking happens in `sample_tokens` instead, which is 5.13 ms cheaper in the ON arm.
So the number I first reported was measured correctly and interpreted wrongly both
times I discussed it: first as manager bookkeeping cost, then as a prose-vs-counting
artifact. The same trace also confirms 0007's premise from the other side: the OFF
arm dispatches every counting step as `num_tokens=36, uniform_token_count=6`,
padded 36 - the uniform graph was always there without the feature, which is
exactly what the varlen branch had been suppressing.

**Closed: the adaptive trade has a concurrency threshold near c8 on prose, and the
residual counting loss is small at high concurrency.** Prose goes -7.0/-7.5/-4.1%
at c1/c2/c4 and +21.5/+19.1/+16.9% at c8/c16/c32; counting is -4.8% at c32 versus
-12.0% before 0007, so the padding was about six tenths of the original regression
and the rest is small fixed cost. This is the same shape upstream argues for
scheduling by (batch x ctx) rather than batch alone (#48627, PR #48944), with our
coefficients.

Caveats kept on purpose: the prose c1-c8 ON numbers come from an earlier boot of the
previous image (0007 was dead in it, but those steps are trimmed, so 0007 should not
have touched them) and each OFF point is a single repeat; the c16/c32 rows are 3
repeats with spreads under 2.5% and are the ones worth quoting. Deployment is left at
the intended state: `.env` and the live engine both `false`, same image, health 200.

## 0008 tested and it did not work: the c1 loss is not over-trimming

I predicted that pricing the unpriced part of a step would remove the single-stream
prose deficit while keeping the high-concurrency gains. Image `8b2828a-debug`
(`7eba2f6c3d2b`), adaptive on, cold cell discarded, 3 warm repeats, and the same
build measured again with the tracer to check what the estimator actually did.

| cell | adaptive OFF | ON + 0007 | ON + 0008 | 0008 vs OFF | 0008 vs 0007 |
|---|---:|---:|---:|---:|---:|
| prose c1 | 56.3 | 52.3 | 52.5 | **-6.7%** | +0.5% |
| prose c2 | 102.1 | 94.5 | 93.3 | -8.6% | -1.3% |
| prose c4 | 163.3 | 156.6 | 153.6 | -5.9% | -1.9% |
| prose c8 | 213.7 | 259.7 | 242.9 | +13.6% | **-6.5%** |
| prose c16 | 349.4 | 411.7 | 401.5 | +14.9% | -2.5% |
| prose c32 | 609.0 | 690.6 | 638.4 | +4.8% | **-7.6%** |
| counting c16 | 1064.2 | 979.9 | 821.5 | -22.8% | -16.2% |
| counting c32 | 1722.1 | 1639.7 | 1597.4 | -7.2% | -2.6% |

The estimator was not broken, which is what makes this a clean negative result.
Traced prose c1 before and after:

| | full-width (6 rows) steps | trimmed to 4 rows | step period | throughput |
|---|---:|---:|---:|---:|
| ON + 0007 | 30.7% | 69% | 43.4 ms | 52.3 |
| ON + 0008 | **48.3%** | ~50% | 42.2 ms | 52.5 |

So it moved the decision in the predicted direction and the throughput did not
move. **The hypothesis is falsified: over-trimming is not what costs single-stream
prose.** Decomposing the c1 gap with the tracer explains why the prediction could
not work: OFF is 41.1 ms/step at 2.315 accepted, ON+0008 is 42.2 ms/step at 2.273 -
about +1.1 ms/step of machinery and -1.8% acceptance, roughly -4.5%, and no amount
of choosing-more-rows recovers either term, because at one request the row count
barely affects the step at all (4 rows and 6 rows both sit in the same ~42 ms
pipeline period). The saving 0008 was chasing did not exist.

Worse, adding a constant to the denominator degrades the cases where trimming is
genuinely profitable: prose c32 fell from +13.4% to +4.8% and counting c16 from
-7.9% to -22.8%, because the constant is largest exactly where cohort ramp and
drain dominate the sample set. Reported per-bucket estimates at first trigger:
1 req 107.3 ms, 2 req 7.3, 3 req 36.1, 4 req 11.1, 5 req 40.7, 6 req 10.1 - against
a tracer-measured 38.4 ms at 1 request and 6.8 ms at 6. Non-monotonic across
buckets, i.e. the estimator is measuring step *composition* (prefill mixed into
small-cohort steps), not a per-request-count constant, which is a second reason its
corrections land in the wrong places.

Disposition: 0008 stays in the series as code but **defaults to off**
(`VLLM_ADAPTIVE_VERIFICATION_PRICED_FIXED_COST=0`, also exposed in `compose.yml` so
a future experiment can enable it without a rebuild). 0007 is kept and stays on: it
was a measured +8.1% on counting c32 with an independently confirmed mechanism
(padded-but-unneeded verification rows 2712 -> 0). The deployment goes back to
adaptive off, which is the state that should be shipped.

What the corrected picture of the c1 deficit looks like, and what would be worth
trying next: the objective prices a row by the *standalone* forward delta at the
last rank, while what matters is a row's *marginal effect on the pipeline period* -
flat at one request, real at six. That is a different correction (reprice the verify
curve from observed step periods rather than add a constant), and it is the part of
#52057's "online profiling" that would actually apply here. I am not shipping a guess
at it: after 0008 the bar is a prediction that survives measurement, not a plausible
model.
