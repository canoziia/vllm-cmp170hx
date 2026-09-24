# Clean SM80 adaptive verification implementation

Branch: `dspark-adaptive-clean-20260924`, from local main `b9bbba3`.
The previous `dspark-varlen-20260924` prototype remains intact as investigation
history. No experimental patch stack is inherited into this branch.

## Objective and constraints

Simple, correct, genuinely per-request variable-prefix DSpark target
verification that improves measured end-to-end throughput over original PP6.
Do not implement batch-wide K or acceptance-only caps. Preserve feature-OFF
runtime and original fixed execution optimizations. Reuse official budget /
allocation semantics and existing attention/PP mechanisms; new code needs a
specific demonstrated gap. No production restarts for this capability stage.

## First result: existing kernels are more capable than the old prototype assumed

Executed on GPU5 / SM80 in an isolated container using the original image
`6c1b2134517814938a23136790f32177659f23a8fec3545f3a56ea76928f8239`.
No model weights loaded. No source bind overlays, monkeypatches, AST-extracted
methods, capability overrides or new row-map/flatten kernels are used.

### MLA/SWA existing builders: 20 eager + captured-attention cases PASS

`tests/test_sm80_existing_ragged_graph.py` instantiates the actual Ampere MLA
builder and inherited ROCm SWA builder, builds metadata outside capture, then
runs the actual Triton sparse-attention kernel in a graph recorded once per
compression ratio. Rebuild metadata in place, replay the old graph, compare to
independent FP32 attention over decoded packed cache bytes.

Coverage:
- ratios 1/2; per-request lengths [6,6,6], [2,6,4], [6,2,4], [1,1,1], [0,6,0];
- page/compression boundary contexts [125,255,129] and [126,256,130];
- nonmonotonic physical block tables; stable request-owner/slot/ragged pointers;
- exact ownership, compressed slot values, padding validity and zero outputs;
- FP8 NoPE + BF16 RoPE + e8m0 scales; sink and extra compressed cache;
- output tolerance rtol=.02, atol=.003; NaN-poisoned output before replay.

Initial fixture incorrectly kept CPU query total at 18 while GPU total shrank
to 12, triggering PyTorch repeat_interleave's output_size assertion. This was
NOT a production defect: the existing contract permits CPU/GPU per-request
boundaries to differ but requires equal totals. Corrected fixture balances CPU
lengths to the same total and all cases pass. No code was changed to conceal
this fixture error. The isolated failed process exited; both production APIs
remained healthy.

### Existing indexer preparation + paged MQA: five layouts PASS

`tests/test_sm80_existing_indexer.py` calls the original unbound
`DeepseekV32IndexerMetadataBuilder._prepare_decode_tensors` with persistent
buffers. CPU boundaries are balanced approximations with the correct total;
GPU boundaries determine ownership and per-token causal context length.

Checks [1,6,2,4], [6,1,4,2], [0,6,0,1], [1,1,1,1], all-zero padding, strided
block tables, unchanged pointers and pad masks. Runs real
`fp8_paged_mqa_logits_triton` followed by Torch top-k, both eager and captured.
Full logits including -inf mask agree with a Torch reference at rtol=.02,
atol=.03; graph top-k must exactly equal eager top-k. Does not claim the fused
production top-k path has been tested.

### Provenance

The following installed files were SHA256-checked against clean-main source
replay `/tmp/deepseek-main-final-replay`; all eight match exactly:

```
d2422c85fe51b8df5041862fddbe47481887f661c7886eb7722a1659439410dc v1/attention/backend.py
bc5ce8034b5d43d32fbb058e9e61bdd85c7ae028553e415945861556dff0bf8a v1/attention/backends/mla/indexer.py
15db794abea3be63f610ad210f2fd215d8dee4d596cd47027b7edf698439c206 v1/attention/backends/mla/sparse_swa.py
3339bf1923e4a9a55816ee84d4b2135bbd6c9f2bf85677674417a8a0f41c1f10 v1/attention/ops/mqa_logits_triton.py
72ecfe633dd70ce7ea0f75bef45a78aa618a9d79194a9c3b58040fc2eebf9711 v1/attention/ops/rocm_aiter_mla_sparse.py
27b17dbeee8916f93849c0ddcd417cdc1650be27b40742fe3ca81ba4ac85cd40 models/deepseek_v4_1/sparse_mla.py
d17e971a926511815cc7737643ba9bedc1db0b230e6bd75a06a73d9ee3232fe8 models/deepseek_v4_1/amd/rocm.py
906cc7847192a42ad5ea0f644118eaed894ed758ce9b619781d76e9c79c12162 models/deepseek_v4_1/ampere/ampere_sparse.py
```

The image as a whole has old diagnostic files; these are not a claim of a clean
whole image. These tests exercise the exact unchanged attention files above.

## Design consequence

Do not carry the prototype's separate verification-row-map module or new
indexer-flatten kernel forward just because they exist. The tested contracts
already work through current builders and kernels. Any later replacement must
prove a missing capability or measurable bottleneck.

No runtime modifications or capability enables in this checkpoint. Existing
UNIFORM_BATCH guards remain untouched. A captured component working is NOT
proof that the runner dispatch/capture path can use a variable-length FULL
model graph safely.

## Next gates

1. Exercise real runner graph metadata construction, including padded request
   capacity and all persistent capture pointer identities. Audit preparation
   costs separately from captured attention time.
2. Add real KV insertion/compression/rollback numerical tests, plus mixed
   prefill/decode and production indexer top-k coverage.
3. Only then propose narrowly scoped SM80 backend capability changes using
   existing metadata, with no new scheduling layer.
4. Specify minimal PP budget/length authority and feedback lifecycle BEFORE
   coding it. Preserve stale CPU budget / current GPU allocation separation;
   don't wait for full hidden-state receive just to learn CPU shape.
5. Feature OFF, verify-all, forced uneven, then adaptive performance/correctness
   gates vs original. Success is net throughput gain, not fewer rows or a unit
   PASS. No claimed performance result yet.
