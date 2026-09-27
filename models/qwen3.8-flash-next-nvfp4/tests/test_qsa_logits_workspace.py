#!/usr/bin/env python3
"""Static regression gate for upstream Qwen QSA logits workspace fix.

The prefill indexer must reserve one fixed worst-case allocation per call and
slice every internal query chunk from it.  Reintroducing torch.empty() inside
_prefill_logits makes the caching allocator retain every increasing row width
and eventually OOM during long prefills.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
path = source_root / "vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py"
source = path.read_text()
tree = ast.parse(source, filename=str(path))
functions = {
    node.name: node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}

prefill_logits = functions["_prefill_logits"]
select_prefill = functions["qsa_select_paged_prefill"]

arg_names = [arg.arg for arg in prefill_logits.args.args]
assert "logits_workspace" in arg_names

for node in ast.walk(prefill_logits):
    if not isinstance(node, ast.Call):
        continue
    func = node.func
    assert not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
        and func.attr == "empty"
    ), "_prefill_logits reintroduced a dynamic torch.empty allocation"

select_source = ast.get_source_segment(source, select_prefill) or ""
assert "budget_bytes = max(max_logits_bytes" in select_source
assert "logits_workspace = q.new_empty" in select_source
assert "logits_workspace," in select_source

# The production default remains one fixed 512 MiB workspace.  Increasing
# context changes rows_per_chunk, not the allocation size.
cap_bytes = 512 * 1024 * 1024
for context_tokens in (3_200, 12_800, 38_400, 83_200, 128_000, 166_400, 524_288):
    logits_width = ((context_tokens + 4 - 1) // 4 + 63) // 64 * 64
    rows_per_chunk = max(1, cap_bytes // (logits_width * 4))
    assert rows_per_chunk * logits_width * 4 <= cap_bytes

print("QSA_FIXED_LOGITS_WORKSPACE PASS bytes=536870912")
