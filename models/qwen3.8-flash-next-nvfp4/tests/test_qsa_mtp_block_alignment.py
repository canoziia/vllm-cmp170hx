#!/usr/bin/env python3
"""Exercise hybrid QSA ring alignment for variable MTP draft depth."""
import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/usr/local/lib/python3.12/dist-packages')
tree = ast.parse((root / 'vllm/platforms/interface.py').read_text())
method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_align_hybrid_block_size')
# Extract the actual geometry code in the non-`all` Mamba branch.
branches = [n for n in ast.walk(method) if isinstance(n, ast.If) and ast.unparse(n.test) == "cache_config.mamba_cache_mode == 'all'"]
assert len(branches) == 1
branch = branches[0].orelse
assert 'qsa_ring' in ast.unparse(ast.Module(body=branch, type_ignores=[]))
code = compile(ast.fix_missing_locations(ast.Module(body=branch, type_ignores=[])), str(root), 'exec')
for depth in range(1, 17):
    for indexer_align in (None, 128):
        ratio = 4
        cache = SimpleNamespace(block_size=64, prefix_match_unit=32)
        cfg = SimpleNamespace(num_speculative_tokens=depth)
        model = SimpleNamespace(hf_text_config=SimpleNamespace(indexer_compress_ratio=ratio))
        cls = SimpleNamespace(_get_indexer_block_alignment=lambda _: indexer_align)
        scope = {'cache_config': cache, 'vllm_config': cfg, 'model_config': model,
                 'cls': cls, 'attn_block_size': 1616, 'kernel_block_alignment_size': 16,
                 'mamba_page_size': 1616, 'attn_page_size_1_token': 1,
                 'lcm': math.lcm, 'cdiv': lambda a, b: (a + b - 1) // b, 'getattr': getattr,
                 'max': max}
        exec(code, scope)
        size = scope['attn_block_size']
        ring = ratio * math.ceil((ratio + depth) / ratio)
        assert size >= 1616
        assert size % ring == size % (indexer_align or 1) == size % 16 == size % 32 == 0
        if depth == 4 and indexer_align is None:
            assert ring == 8 and size == 1632, (ring, size)
        if depth == 5 and indexer_align is None:
            assert ring == 12 and size == 1632, (ring, size)
print('QSA_MTP_BLOCK_ALIGNMENT depths=1..16 PASS')
