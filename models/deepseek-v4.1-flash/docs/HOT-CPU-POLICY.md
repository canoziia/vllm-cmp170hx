# Request-pinned CPU policy experiments

Default traffic remains on the existing history policy. Requires public
`enable_adaptive_verification=true` and DeepSeek MRV2 history routing. Manual
`vllm_xargs.spec_k` wins. This is not a second public adaptive mode switch.

Trusted host root writes `/root/app/deepseek-v41/cache/dsv41-policy` (already
mounted as `/root/.cache/dsv41-policy`). No restart is required after the initial
scheduler hook deployment. Keep directory0700, control/source0600. Run:

```
python3 scripts/publish-policy.py /root/app/deepseek-v41/cache/dsv41-policy scripts/policies/measured_v1.py
python3 scripts/publish-policy.py /root/app/deepseek-v41/cache/dsv41-policy --disable
```

The root-only control file contains a capability token. Local test clients read
it and submit `vllm_xargs.spec_policy_token`; never print it, put it on command
lines, commit it or expose it to ordinary clients. Unauthenticated requests do
not read/load source and retain baseline. Control has enabled, token, sha256,
config; modules live in versions/<sha256>.py. Publisher atomically replaces
control. Wrong hash, bad syntax/API/self-test/factory retains baseline.

API_VERSION=1; self_test(); Policy(config); observe(event); choose(context)->int
1..5. Optional JSON diagnostics, max16KB. observe receives actual scheduled k,
accepted drafts, output tokens, timestamps/seconds, validity, live-load epoch.
Choose runs only after fresh feedback, before NEXT eligible scheduling. Multiple
choose calls with no fresh feedback do not accumulate persistence or change k.
Requests are pinned to a source hash and copied config at admission, up to32
loaded versions per process. Active requests never migrate state. Disable only
stops NEW experiment admissions; request cancellation is separate.

Fallback: exceptions, illegal results or a completed call taking >20ms disable
that revision, falling back to warmed baseline state. Python is NOT sandboxed;
an infinite loop, native crash, memory exhaustion or blocking import cannot be
interrupted by the post-call time guard. Only vetted trusted CPU-only code may
be published. Do not import torch/CUDA, do IO or hold model/GPU references.

Timing is CPU-visible request feedback recurrence. It includes pipeline cadence,
queueing and scheduler/host overhead, NOT CUDA-only kernel execution. Exact
SchedulerOutput stores request generation ID and dispatched width, so in-flight
results cannot be credited to a newly selected k or a recycled request. First
feedback, width transition, load membership change, prefill, stale/invalid or
max-token terminal samples are excluded from measured reward. Dispatch latency
is logged separately for diagnosis, not mixed with recurrence. No synchronization
or PP protocol changes. Candidate resets measured N/Y/T on load epoch change.

Trace is private append-only JSONL at policy root, capped64MiB per process
lifetime; no prompt/output token IDs. Trace includes versions, actual widths,
proposed widths, timing semantics, N, real/predicted/blended scores. Trace IO is
synchronous CPU work for experiments and can affect their speed; baseline has no
per-step trace IO. Never call traced performance an uninstrumented production
speed. Source code storage is writable only by trusted host admin.

Candidate measured_v1 discounts per-k N/Y/T at.95, mixes one current predicted
pseudoblock, compares strict score>, max+1 upward and arbitrary downward jumps.
It is experimental, not an asserted optimum; no cohort coupling or fixed warmup.

## Later stability research (not a new production default)

`measured_v15.py` retains per-arm measured evidence and calibrated predictions,
adds measured-ratio uncertainty and paired-prefix gain uncertainty, and uses an
explicit3% improvement tolerance (cold choice retains2% wider near-tie rule).
All real evidence decays.95; old speeds themselves do not shrink. Up max+1,
down may jump; stale-width/invalid-timing feedback cannot immediately switch.
The uncertainty gate is a heuristic, not an iid confidence guarantee. No task
names/labels or cohort-coupled decision is used. CPU tests include six
concurrencies/all starting widths and high-low-high recovery. Candidate is still
experimental: mixed-load execution exposed a fixed-interface issue below.

## Retained width fix: adaptive/0014 (requires deploying scheduler code)

Before0014, only residents eligible for a fresh `choose()` got history limits.
A stale-output or capacity-excluded experiment could consequently executefull5
although its pinned selected width was2/3, causing spurious one-step5 jumps.
0014 seeds limits from `HotHistoryPolicy.retained_limits()` before refreshing
eligible choices. Only active authenticated experimental entries participate;
manual overrides, failed revisions and ordinary traffic keep their prior path.
No GPU/PP protocol or new feedback advances are introduced.

Adaptive series reproducibility: 0013 was originally generated against a
one-off snapshot that also carried the DISABLED hot-perf-debug scheduler code,
so the series did not replay from the pinned source commit (0013 failed at
`@@ -1,7 +1,6 @@`). It was regenerated from the real 0001..0012 output; the
replay now completes and reproduces the deployed container bytes exactly at
0013 and the fixed bytes at 0014. Any new patch must be validated by replaying
the WHOLE series from SOURCE_COMMIT, not by apply-check against whichever
snapshot happened to generate it.

This is NOT fixable by publishing a CPU policy alone. Source/CPU integration
checks pass; do not claim online fix until a separately approved scheduler
maintenance deployment and repeat mixed-load GPU validation. Existing image
8fc270533891 does NOT contain0014. Keep experimental admissions disabled until
that validation; historical single-load results do not certify mixed-load
stability.
