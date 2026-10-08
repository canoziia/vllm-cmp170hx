#!/usr/bin/env python3
"""CPU test of actual LMCache grouping methods (no torch/CUDA imports).

Usage: python3 scripts/test-lmcache-bounded-pd-groups.py /path/to/lmcache/v1/kv_layer_groups.py
AST loading strips unrelated imports only; the production method bodies run.
GPU transfer and draft acceptance still require the deployment acceptance suite.
"""
import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple
import sys
import unittest

source = ast.parse(Path(sys.argv.pop(1)).read_text())
keep = []
for node in source.body:
    if isinstance(node, ast.ClassDef) and node.name in ('_ObjectBucket', 'ObjectGroupInfo'):
        keep.append(node)
    if isinstance(node, ast.ClassDef) and node.name == 'KVLayerGroupsManager':
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in
                     ('_detect_object_groups', 'enable_full_sw_kv', '_validate_block_chunk_size_config')]
        keep.append(node)
ns = dict(defaultdict=defaultdict, dataclass=dataclass, NamedTuple=NamedTuple,
          logger=NS(info=lambda *args: None))
exec(compile(ast.Module(body=keep, type_ignores=[]), '<actual-lmcache-methods>', 'exec'), ns)
Manager = ns['KVLayerGroupsManager']

def group(window, recurrent=False, tag=0):
    return NS(sw_size_tokens=window, recurrent_state=recurrent, extra_object_group_tag=tag)

def manager(groups, enabled=True):
    obj = Manager()
    obj._kernel_groups = groups
    obj._separate_object_groups = True
    obj._lmcache_tokens_per_chunk = 8192
    obj._merge_bounded_pd = enabled
    return obj

def desc(obj):
    return [(g.sw_size_chunks, g.recurrent, g.aux) for g in obj._detect_object_groups(())]

class Tests(unittest.TestCase):
    def test_reproduce_old_mismatch(self):
        self.assertEqual(len(desc(manager([group(-1), group(8192, True)], False))), 2)
        self.assertEqual(len(desc(manager([group(-1), group(8192, True), group(2048)], False))), 3)

    def test_all_pp_ranks(self):
        regular = [group(-1), group(-1), group(8192, True), group(8192, True)]
        for rank in range(4):
            groups = regular + ([group(2048), group(2048)] if rank == 3 else [])
            obj = manager(groups)
            self.assertEqual(desc(obj), [(-1, False, False), (1, True, False)])
            buckets = obj._detect_object_groups(())
            self.assertEqual(sorted(i for b in buckets for i in b.kernel_group_indices), list(range(len(groups))))
            self.assertEqual([g.sw_size_tokens for g in groups], [-1, -1, 8192, 8192] + ([2048, 2048] if rank == 3 else []))

    def test_order_independent(self):
        self.assertEqual(desc(manager([group(2048), group(-1), group(8192, True)])),
                         desc(manager([group(-1), group(8192, True), group(2048)])))

    def test_aux_isolated(self):
        self.assertEqual(desc(manager([group(2048, tag=1), group(-1), group(8192, True)])),
                         [(-1, False, False), (1, True, False), (1, False, True)])

    def test_different_windows_not_merged(self):
        self.assertEqual(len(desc(manager([group(-1), group(8192, True), group(16384)]))), 3)

    def test_full_sw_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            manager([]).enable_full_sw_kv()
        obj = manager([], False)
        obj.enable_full_sw_kv()
        self.assertTrue(obj._full_sw_kv)

    def test_geometry(self):
        Manager._validate_block_chunk_size_config(0, 64, 2048, 8192, 2048)
        Manager._validate_block_chunk_size_config(1, 1, 8192, 8192, 8192)
        with self.assertRaisesRegex(ValueError, 'sliding window'):
            Manager._validate_block_chunk_size_config(0, 64, 1152, 4608, 2048)

if __name__ == '__main__':
    unittest.main()
