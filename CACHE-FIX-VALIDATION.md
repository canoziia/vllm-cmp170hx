# Generated-history cache validation — 2026-09-06

## Deployed artifact

Code commit: `c051ee91a31f2a2e0d4915aa6a9506b02adab691` (plus documentation-only follow-up).
Image: `localhost/vllm-qwen38-nvfp4-spark:cachefix-c051ee9`.
ID: `f4d1a3cfb741880c0c9add721d684fd1d3ceb9a02e0bcb56d89acf558ba31959`.
Built natively on ARM64 from official base, all22 final-manifest files checked inside
the deployed container. Original NVFP4/BF16 weights unchanged. MTP3, batch8192,
seqs32, native262144, util.90, BF16 KV/Mamba, prefix-match-unit32 unchanged.
Restart policy unless-stopped; `.env` points to this version, not an experimental tag.

## Root cause and fix

Previous fine-grained caching only materialized the input prompt tail. Decode
outputs could leave no reusable recurrent snapshot near the end of an answer.
The next request then replayed much of the prior answer despite identical token
history. Tests verified the generated token IDs remain an exact prefix of the
next rendered input in64/256/1024-token examples.

0018 adds accepted decode checkpoints, protected by CoW and async fences. 0019
keeps uniform MTP target verification, limits accepted output at128-token snapshot
boundaries, accounts for the CPU's optimistic1–3-token advancement, retains
append-only attention hash aliases, and scopes batched CoW to the owning group.
Only three recent decode snapshots per producer are retained. EOS/length-trimmed
last steps are not registered as snapshots. Fine prefix matching remains32; the
128 snapshot interval is a latency/memory tradeoff, not a larger physical KV block.

## Final-image API tests

`~/app/tmp/cachefinal/verify-release.exit=0`, log ends `VERIFIED_RELEASE_COMPLETE`.

| Prior response length | Next input | Cache hit | Shared history replay |
| ---: | ---: | ---: | ---: |
| 64 | 1829 | 1696 | 116 |
| 256 | 2021 | 1920 | 84 |
| 1024 | 2786 | 2688 | 84 |
| 1500 | 3262 | 3200 | 48 |

All32 concurrent multi-turn continuations passed, shared replay121–122tokens.
These are real token counts, not rewritten accounting. New messages/template
are additional uncached tokens. Modified prefixes, evicted snapshots and branches
outside retained history can still miss normally; this is not a universal hit guarantee.

- Responses function-call/result round trip and Kyoto/Lisbon branch-specific facts
  passed (cached and cold answers correct).
- Repeated prompt regression at2459/4949/8909tokens: positive output IDs, exact
  text/token IDs/first-token logprobs across cold and two cached requests.
- Scheduler unit768cases, accepted-prefix cap82cases, normalization16 dtype/layout/
  acceptance combinations, group-copy3page sizes and Mamba memcpy27cases passed.
  CUDA unit tests ran while service stopped to avoid a second context OOM.
- Instrumented precursor: all73 GDN/PLE accepted, continued and restored state
  tensors byte-equal; same-shape cache-save on/off/on512 generated IDs identical.
  This evidence concerns lossless snapshot handling, not cold-prefill equivalence.

## Numerical scope

Long-prefill and stepwise decode states are not bitwise identical in this backend.
Greedy continuations from decode-derived snapshots can differ from long-prefill
recomputation. We observed and documented this, rather than treating a failed
cold-equality assertion as a pass. Existing prompt-only cache tests remain exact.
No lossy state/weight conversion was introduced. Model-wide batch invariance or
quality parity on all prompts is not claimed by these targeted regression tests.

## Performance and runtime

Original pre-fix32-concurrency500-token saturation test:417.38 tok/s full-batch.
Every32-token prototype:~295–303 vs same-instance disabled413.03; rejected as too
expensive. Efficient128-token version:392.44 in candidate test. Final deployed
smoke: single29.31,32-way368.79 tok/s full-batch (end-to-end28.90/343.54).
These are single measurements and still show overhead; do not advertise no-regression
or use them as a multi-run median. The cache repair targets repeated-history replay,
not a new decode throughput record.

Final loading473.98s, KV24.66GiB/900779tokens, health200, restart0, running/waiting0.
Startup had driver allocation stalls as before; no EngineCore failure, worker death
or illegal-access traceback in the deployed service log. Debug sentinels absent.
Experimental cache images removed; official base and pre-fix rollback image retained.
Raw logs, JSON and diagnostic state comparisons remain on Spark under
`~/app/tmp/cachefinal/`, `~/app/tmp/cachefix/` and local `/tmp/spark-audit/`.
