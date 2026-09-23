#!/usr/bin/env python3
"""Fixed-workspace regression for the SM80 sparse indexer."""

import ast
import sys
from pathlib import Path

root = Path(sys.argv[1])
indexer_file = root / "vllm/model_executor/layers/sparse_attn_indexer.py"
triton_file = root / "vllm/v1/attention/ops/mqa_logits_triton.py"
indexer = ast.parse(indexer_file.read_text())
triton = ast.parse(triton_file.read_text())

calls = [
    node
    for node in ast.walk(indexer)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "fp8_mqa_logits_triton"
]
assert len(calls) == 2
assert all(
    any(
        kw.arg == "out"
        and isinstance(kw.value, ast.Name)
        and kw.value.id == "logits_workspace"
        for kw in call.keywords
    )
    for call in calls
)

functions = {
    node.name: node for node in triton.body if isinstance(node, ast.FunctionDef)
}
impl = functions["_fp8_mqa_logits_triton_impl"]
assert "out" in [arg.arg for arg in impl.args.args]
assert any(
    isinstance(node, ast.If)
    and isinstance(node.test, ast.Compare)
    and isinstance(node.test.left, ast.Name)
    and node.test.left.id == "out"
    for node in ast.walk(impl)
)
assert "out.view(-1)[: M * N].view(M, N)" in triton_file.read_text()

if len(sys.argv) == 3 and sys.argv[2] == "--gpu":
    sys.path.insert(0, str(root))
    import torch
    import vllm.v1.attention.ops.mqa_logits_triton as op

    device = torch.device("cuda")
    rows, heads, dim = 4, 16, 128
    workspace = torch.empty(rows * 509, dtype=torch.float32, device=device)
    for width in (257, 509):
        q = torch.randn(rows, heads, dim, device=device).to(torch.float8_e4m3fn)
        k = torch.randn(width, dim, device=device).to(torch.float8_e4m3fn)
        scales = torch.ones(width, device=device)
        weights = torch.randn(rows, heads, device=device)
        starts = torch.zeros(rows, dtype=torch.int32, device=device)
        ends = torch.full((rows,), width, dtype=torch.int32, device=device)
        args = (q, (k, scales), weights, starts, ends)
        expected = op.fp8_mqa_logits_triton(*args, clean_logits=False)
        actual = op.fp8_mqa_logits_triton(
            *args, clean_logits=False, out=workspace
        )
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)
        assert actual.data_ptr() == workspace.data_ptr()

print("DEEPSEEK_FIXED_TRITON_LOGITS_WORKSPACE PASS")
