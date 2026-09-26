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
| counting (near-full acceptance) | 1 | 149 | 124.5 | **-16.4%** |
| counting | 2 | 256 | 256.5 | +0.2% |
| counting | 4 | 412 | 475.8 | +15.5% |
| counting | 8 | 788 | 511.9 | **-35.0%** |
| counting | 16 | 1093 | 947.3 | -13.3% |
| counting | 32 | 1724 (repeats 1706) | **1513** (1513/1453/1517) | **-12.2%** |
| prose (low acceptance) | 16 | 351.6 | **420.4** (392/426/420) | **+19.6%** |
| prose | 32 | 609.4 | **710.0** (710/710/707) | **+16.5%** |

Two readings, and the sign is not a coincidence:

* **prose gets faster even though acceptance drops** (2.35 -> 2.05-2.15 tokens per
  request-step). That is the intended trade: on low-acceptance content most of
  the five draft rows are rejected anyway, so not verifying them removes work
  without giving up emitted tokens. It also refutes the earlier assumption that
  verification-row count is irrelevant on this box - it is irrelevant when the
  rows would have been cut short anyway, and it is not irrelevant when a step
  carries six requests times six rows.
* **counting gets slower** because there is nothing to trim (acceptance stays
  5.952 of 6) while the manager still costs work per step. Two cells reached
  `acceptance 5.708 < 5.952`, i.e. the trimming cut rows that would actually
  have been accepted - a real loss, not just overhead.

The counting curve is also non-monotonic (`+15.5%` at c4, `-35%` at c8), which is
the cost-estimate instability upstream describes in #52057 ("profiling at startup
by replaying recorded CUDA graphs is known to have some drift, especially at
higher batch sizes"). Our numbers are one sweep each at those points, so treat the
c4/c8 spread as "unstable", not as a measured effect.

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
