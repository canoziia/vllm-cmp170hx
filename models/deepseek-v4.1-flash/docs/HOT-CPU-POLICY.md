# Fixed measured policy and debug-only hot experiments

With `enable_adaptive_verification=true`, DeepSeek uses the compiled request-local
measured policy. Other DSpark models keep the GPU confidence allocator; manual
`vllm_xargs.spec_k` takes priority. Disabling the public switch retains the fixed
full-width path. The verification execution profile does not select the algorithm.

## Production build

`ENABLE_PERF_DEBUG=0` (default) applies `adaptive/0013-fixed-measured-policy.patch`.
The scheduler constructs `FixedHistoryPolicy`. There is no external Python
loader, capability admission, file polling or per-step trace IO. Selected widths
are retained even when a request is ineligible for a fresh decision.

Statistics discount evidence at the configured decay (default0.95); empirical
speeds do not decay. Candidate predictions share current output/time calibration.
Per-width measured evidence blends with one predicted pseudoblock. Measured-ratio
and paired-prefix uncertainty gates plus a3% improvement tolerance reduce marginal
reversals. Cold selection prefers the widest candidate within2% of the maximum,
provided it beats full width by3%. Upward changes are at most one width; downward
changes may jump. These tolerances and uncertainty estimates are engineering
heuristics, not formal confidence guarantees. No prompt labels or new cohort
coordination are used.

Feedback carries the actual scheduled width and request generation. First,
transition, prefill, load-change, invalid and terminal periods do not train
measured timing. CPU-visible recurrence includes pipeline cadence and queueing,
NOT GPU kernel time. Invalid policy decisions fail closed to full5 for that
request. Finishing/aborting removes its state.

## Debug build

`ENABLE_PERF_DEBUG=1` additionally applies `optional/0002-hot-perf-debug.patch` and
`optional/0003-hot-cpu-policy-debug.patch`. The latter adds `HotHistoryPolicy`
in `dsv41_hot_policy_debug.py`, extending the same fixed adapter. Ordinary
requests still use the compiled measured policy.

Only trusted host administrators publish source under
`/root/app/deepseek-v41/cache/dsv41-policy` (container cache mount). Keep root
0700 and source/control0600. Publisher:

```
python3 scripts/publish-policy.py /root/app/deepseek-v41/cache/dsv41-policy trusted-policy.py
python3 scripts/publish-policy.py /root/app/deepseek-v41/cache/dsv41-policy --disable
```

Debug admission requires enabled control and the matching private capability in
`vllm_xargs.spec_policy_token`; manual widths never admit. Never print, commit or
expose the token. API_VERSION=1, self_test(), Policy(config), observe(event),
choose(context)->integer1..5. Requests pin hash/config at admission; new publishes
affect only new requests. Upward decisions must still obey max+1. Disable stops
new admission, not existing requests.

Source SHA is verified, symlinks/oversized source rejected, code version count
bounded32. Plugins are trusted executable Python, NOT sandboxed: the20ms completed
call guard cannot interrupt infinite loops, native faults or blocking imports.
Test CPU-only plugins independently before publication. No torch/CUDA imports,
GPU objects or IO in policy calls. Bad plugins disable their revision and use
fresh compiled fixed state with a new generation, avoiding old-feedback reuse.

Debug-only private JSONL records attributed feedback and decisions, bounded64MiB
per process. It excludes prompt/output IDs and credentials. Synchronous trace
IO can affect performance; do not call traced results uninstrumented throughput.
Production has no trace writer or external-source execution.

All source changes must replay from SOURCE_COMMIT through the COMPLETE main,
common, adaptive and (when requested) optional series. Self-generated snapshot
apply-check alone is insufficient. Production and debug configurations are both
validated before deployment. Legacy experiments remain local research artifacts,
not numbered algorithm variants in production source.
