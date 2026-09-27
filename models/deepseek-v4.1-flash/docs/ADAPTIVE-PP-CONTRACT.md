# Minimal PP contract under review (not implemented / not enabled)

## Keep from existing vLLM

- `AdaptiveVerificationManager` owns budget/cost policy and GPU allocation.
- V2 input/sampler layout contract: scheduled capacity stays CPU metadata;
  actual offsets live on device. Use the manager's sampler guard/logit limits.
- `PPHandler` owns delayed feedback, slot generations and stream/event liveness.
- `GroupCoordinator` owns forward dictionary metadata and CUDA tensor lifetime.
- Current SM80 builders already form ragged metadata; no new row-map abstraction.

## Observations that permit a smaller protocol

`GroupCoordinator.irecv_tensor_dict` receives its object metadata before returning
GPU receive handles. Python/non-tensor values ride in that already-existing
metadata message; they do not require additional Gloo tensor sends. GPU
`Work.wait()` normally inserts current-stream ordering rather than a host-side
readback. Calling `.item()` after that wait is what forces completion through
the dependency chain.

Therefore the prototype's CPU uint8 JSON header tensor, separate CPU-length
payload, runtime `.item()` validation and new persistent offset-buffer owner are
not justified as the final architecture. It is also wrong to equate all
`Work.wait()` calls with a host synchronize.

## Candidate design to test before adoption

1. Publish current per-request confidence with existing deferred draft feedback.
   Reuse that FIFO's request ownership; do not introduce a second generation map.
2. First rank chooses one scalar total budget from a previously completed CPU
   confidence snapshot using official cost semantics. No current-step D2H read.
3. First rank performs the existing deterministic device allocation and writes
   the manager's normal query/logit offsets.
4. Add a small non-tensor metadata item carrying total budget/shape on the same
   forward dictionary message. Consumers can parse that item before accessing
   CUDA hidden-state buffers. No per-step JSON, Gloo tensor payload or file read.
5. For authoritative per-request lengths, compare two options in the timing /
   correctness harness, then choose ONE:
   - carry the compact GPU length vector in forward CUDA payload, installed
     under existing stream dependencies; or
   - identical deterministic GPU allocation from the same deferred confidence,
     slot mapping and first-rank budget on all stages (must prove last-rank and
     non-last-rank confidence/draft generation agreement, no local decisions).
   The first is simpler to reason about; the second saves one CUDA transfer but
   creates a stronger cross-rank state invariant. Do not choose on aesthetics.
6. Downstream stages consume budget, not independently select it. Reuse manager
   layout outputs and input buffers; no parallel plan adapter phases.

This is a review proposal, not a promise that the above is sufficient. Need a
multi-step scheduling/ownership test and a small-tensor PP latency measurement
before choosing protocol. No same-step last-rank-to-first-rank dependency.

## Invariants / rejection gates

- Feature OFF: no confidence allocation/compute/transfer or new forward field.
- FULL admission: do not rebuild original layout unnecessarily, use appropriate
  existing fixed graph when compatible; distinguish from feature OFF overhead.
- Empty/prefill-only step: all ranks agree whether budget field exists.
- Reuse/cancel/preempt: confidence and draft token rows share the same generation
  validity; producer stream snapshots survive next proposal overwrite.
- Per-request actual length bounded by scheduled capacity by construction;
  CPU total chosen first; GPU offsets have the same total, including zero budget.
- CPU metadata is not a peer-untrusted network API; static/schema/order checks
  should not force GPU readback. If external injection is in scope, distributed
  error termination must be explicit, not a local ValueError before peer sends.
- Sampler sees correct adaptive state (including logprobs and chunk-size cap).
- Metadata must be removed before generic model IntermediateTensors slicing.
- Timing must account for draft, target, sampling, CPU and PP cycle, with graph
  mode/padding/context/concurrency provenance. No universal C1 SSE cost table.
