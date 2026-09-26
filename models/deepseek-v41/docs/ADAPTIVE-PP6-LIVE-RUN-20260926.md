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

The switch lives in the engine command, which for the LMCache deployment comes
from `compose.lmcache.yml` (NOT `compose.yml`, which the LMCache overlay
replaces). The working tree of `/root/app/src/vllm-cmp170hx-main` carries that
one-line edit for the duration of the test; revert with
`git checkout -- models/deepseek-v41/compose.lmcache.yml` to go back to
adaptive-off.
