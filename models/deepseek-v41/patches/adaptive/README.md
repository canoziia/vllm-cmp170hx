# Clean adaptive implementation (not in default patch series)

This directory is a replacement design, not an extension of the previous
experimental stack. Current checkpoint is backend-only, not PP adaptive serving.

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

This does NOT establish whole-model FULL capture, KV equivalence at model level,
PP6 multi-step adaptive protocol, mixed prefill support under adaptive FULL,
feature-OFF performance, or end-to-end speedup. These remain enablement gates.
No default build/configuration was changed. Apply only in an isolated test tree.
