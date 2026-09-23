#!/usr/bin/env python3
"""Static regression for uniform PP2 decode checkpoint scheduling."""

import ast
import sys
import types
from pathlib import Path

source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/usr/local/lib/python3.12/dist-packages")
path = source_root / "vllm/v1/core/sched/scheduler.py"
tree = ast.parse(path.read_text())
func = next(
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef) and node.name == "_mamba_block_aligned_split"
)
func.decorator_list = []
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        func,
    ],
    type_ignores=[],
)
namespace: dict[str, object] = {}
exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
split = namespace["_mamba_block_aligned_split"]

scheduler = types.SimpleNamespace(
    mamba_partial_cache_hit=True,
    use_v2_model_runner=True,
    scheduler_config=types.SimpleNamespace(async_scheduling=True),
    num_spec_tokens=3,
)
cases = 0
for start in range(128, 224):
    for count in range(1, 5):
        request = types.SimpleNamespace(
            num_computed_tokens=start,
            num_prompt_tokens=100,
            num_tokens=start + 1,
            num_in_flight_tokens=0,
        )
        assert split(scheduler, request, count) == count
        request.num_in_flight_tokens = 4
        assert split(scheduler, request, count) == (
            0 if start % 128 <= scheduler.num_spec_tokens else count
        )
        cases += 2
print(f"PP2_DECODE_CHECKPOINT_SCHEDULER cases={cases} PASS")
