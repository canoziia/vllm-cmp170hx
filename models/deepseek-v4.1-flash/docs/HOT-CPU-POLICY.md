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
