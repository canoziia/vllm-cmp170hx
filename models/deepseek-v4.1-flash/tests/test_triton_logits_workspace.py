#!/usr/bin/env python3
"""Fixed-workspace regression for the SM80 Ampere sparse indexer."""

import ast
import sys
from pathlib import Path

root = Path(sys.argv[1])
indexer_file = root / "vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/indexer.py"
logits_file = root / "vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/mqa_logits.py"
indexer = ast.parse(indexer_file.read_text())
logits_mod = ast.parse(logits_file.read_text())

functions = {
    node.name: node for node in logits_mod.body if isinstance(node, ast.FunctionDef)
}
for fn_name in ("mqa_logits", "paged_mqa_logits", "candidate_mqa_logits"):
    assert fn_name in functions, f"Missing {fn_name} in ampere/mqa_logits.py"
    arg_names = [arg.arg for arg in functions[fn_name].args.args]
    assert "out_workspace" in arg_names, f"Missing out_workspace in {fn_name}"

calls = [
    node
    for node in ast.walk(indexer)
    if isinstance(node, ast.Call)
    and (
        (isinstance(node.func, ast.Name) and node.func.id in ("mqa_logits", "paged_mqa_logits", "candidate_mqa_logits"))
        or (isinstance(node.func, ast.Attribute) and node.func.attr in ("mqa_logits", "paged_mqa_logits", "candidate_mqa_logits"))
    )
]
assert len(calls) >= 3
assert all(
    any(kw.arg == "out_workspace" for kw in call.keywords)
    for call in calls
), "All ampere indexer MQA logits calls must wire out_workspace"

print("DEEPSEEK_FIXED_TRITON_LOGITS_WORKSPACE PASS")
