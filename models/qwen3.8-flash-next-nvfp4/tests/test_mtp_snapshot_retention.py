#!/usr/bin/env python3
"""Check that generated-history snapshots scale with the configured MTP window."""

import ast
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/usr/local/lib/python3.12/dist-packages')
tree = ast.parse((root / 'vllm/v1/core/single_type_kv_cache_manager.py').read_text())
manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MambaManager')
loops = [node for node in ast.walk(manager) if isinstance(node, ast.While) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Call) and any(isinstance(arg, ast.Name) and arg.id == 'snapshots' for arg in node.test.left.args)]
assert len(loops) == 1, f'expected one snapshot eviction loop, got {len(loops)}'
limit = loops[0].test.comparators[0]
compiled = compile(ast.Expression(limit), '<snapshot-retention>', 'eval')
for depth in (1, 3, 5, 6):
    from types import SimpleNamespace
    bound = eval(compiled, {'max': max}, {'self': SimpleNamespace(num_speculative_blocks=depth)})
    assert bound == max(3, depth), (depth, bound)
print('MTP_SNAPSHOT_RETENTION depths=1,3,5,6 PASS')
