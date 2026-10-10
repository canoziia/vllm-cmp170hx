# Remaining DeepSeek V4.1 optimisation candidates

These are investigation candidates, **not demonstrated gains**. The historical
profile in `/tmp/ds-src/CONTEXT.md` predates small-kernel fusion, mHC v2,
MXFP4 activation/sum fusion, FP8 draft head, thin wo_a, prefill kernels, O2
dense kernels and the verification/cohort/metadata optimisations. Do not use
its timings as the current bottleneck ranking.

## First: obtain a current profile

Use the current production-equivalent image and settings, at normal 180 W,
with c1 and c8 initially. Separate CPU preparation, GPU compute, copies,
collectives and waiting; record actual cohort sizes and graph shapes. Warm
kernels before measuring. Profiling/maintenance needs an explicit window;
do not restart production just to investigate. Pick one or two candidates
from actual remaining time rather than implementing every item below.

## Candidates

| Priority | Candidate | What to inspect / potentially fuse | Acceptance requirements |
|---|---|---|---|
| High | Indexer / top-k metadata chain | Remaining top-k postprocessing, pool-position mapping, local-window indices and slot mapping. Look for duplicate scans, temporary arrays and DtoD copies after existing fusion. | Integer outputs bitwise identical; stable buffers, no capture-time allocation or host sync; end-to-end gain. |
| High | Mixed-width FULL graphs and metadata | CPU-selected heterogeneous `spec_k` batches may lose the exact uniform metadata replay path. Investigate compact ragged FULL descriptors plus graph-safe dynamic request boundaries / token ownership. | Correct real/padded rows and ownership, no PP feedback-order change; compare mixed throughput to uniform k5. Padding everything to k5 may erase pruning gains. |
| Medium | Sparse-attention output postprocessing | Remaining output copies, layout conversions, scaling or residual operations not already covered by direct-output fusion. | Preserve rounding and buffer alias/lifetime semantics; measure removed memory traffic and actual stage time. |
| Medium | Drafter / sampling tail chain | Argmax/top-k, candidate arrangement, acceptance metadata and feedback packing after logits. Identify avoidable serial launches or synchronization. | Target sampling, rejection, seeds and constraints unchanged; preserve PP collective ordering. Do not assume lm_head can be fused cheaply. |
| Medium | Dense GEMM epilogues | Scale, bias, residual addition or dtype/layout conversion immediately after projections; fold into existing output writeback where appropriate. | Preserve incumbent cast/add order or explicitly evaluate numerical and acceptance changes; no decode/prefill regression. |
| Lower | Remaining mHC pre/post work | Statistics, normalisation, stream mixing and residual chains left after mHC v2. | Profile first: much is already fused. Preserve reductions and multi-stream safety; numerical improvements are not automatically throughput gains. |

## Algorithm / execution follow-ups

- The request-local measured verification policy is already implemented; do not
  reintroduce its superseded cohort-cost policy as an additional algorithm.
- Investigate residual marginal k misselection / oscillation only if it causes
  meaningful workload regression. Users have accepted small differences and
  output-trajectory changes; avoid endless repeated benchmarks.
- Long-context decode may change the best width and cost balance. Small targeted
  checks can establish whether current selection remains useful; configured 1M
  context support is not itself a performance validation.
- Shared scheduler/metadata improvements may benefit GLM or Qwen, but their
  images/configurations require separate compatibility review. Do not enable
  DeepSeek-specific paths on other models without evidence.

## Completed or rejected directions (avoid duplicate work)

Already deployed: deterministic MoE alignment; MXFP4 MoE decode; sparse decode;
small-kernel fusion; mHC v2; MoE activation/sum fusion; FP8 draft head; thin wo_a;
MoE prefill split; fast prefill indexer; dense MXFP8 kernels; verification graph
selection, shared metadata / metadata graphs, live-load cohorts, request `spec_k`
and the measured automatic width policy. See the patch series and model README
for exact switches and scopes.

Previous engine-level negative/risky experiments include O1 MoE, alternative
MoE dequantisation paths, router fusion/reordering, drafter tail relocation,
64-head sparse-prefill variants and alternate PP partitions. Revisit only with
new evidence or a materially different design, not a repeated microbenchmark.

## Delivery rules

- New optimisation switches default off; compose enables through the model's
  optimisation profile, with explicit per-switch overrides respected.
- Prefer bitwise-identical metadata/fusion. If arithmetic order changes, report
  reference error, routing/sampling/acceptance effects and output differences.
- CUDA-graph safe, deterministic where required, no unowned persistent scratch
  or cross-stream races. Do not modify NCCL order casually.
- Compare baseline first with a fixed benchmark snapshot and warm-up protocol;
  report tok/s, actual steps/acceptance and variability, not kernel speed alone.
- Keep k5 regression checks across c1/2/4/8/16/32 where changes affect verification.
  Prefill optimisation decisions should include long-context measurements.
- One canonical patch per topic; consolidate follow-up fixes, descriptive names
  without experiment/version suffixes. Apply the full stack to pinned source.
- Use normal build scripts, fixed production image tags and repository compose;
  diagnostic hooks belong in debug builds, not ordinary production.
- Keep tests focused and reusable; preserve research evidence outside the
  production patch series. No development RPC or hot loader in production.
