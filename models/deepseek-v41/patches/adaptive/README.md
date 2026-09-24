# Clean adaptive implementation (not in default patch series)

This directory is a replacement design, not an extension of the previous
experimental stack. Current checkpoint includes candidate V2 PP integration, not validated adaptive
serving or a throughput result. It is not in the default build chain.

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
PP6 multi-step adaptive protocol, mixed prefill support under adaptive FULL,
feature-OFF performance, or end-to-end speedup. These remain enablement gates.
No default build/configuration was changed. Apply only in an isolated test tree.
