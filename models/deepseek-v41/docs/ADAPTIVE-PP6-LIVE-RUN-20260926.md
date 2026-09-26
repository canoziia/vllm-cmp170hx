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
