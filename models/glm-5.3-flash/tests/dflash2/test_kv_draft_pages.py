#!/usr/bin/env python3
"""CPU geometry/stride regression; run in the vLLM candidate image."""
import math
import unittest
from types import SimpleNamespace as NS
import torch
from vllm.v1.core.kv_cache_utils import _glm5_next_draft_groups
from vllm.v1.kv_cache_interface import (SlidingWindowSpec, MLAAttentionSpec,
    KVCacheTensor, KVCacheLayout, create_kv_cache_views)

class Tests(unittest.TestCase):
    def test_geometry(self):
        for target, expected in [(5120,1024),(6144,1024),(8192,1024)]:
            # Real GLM MLA 512 bf16 elements/token; draft 4 KV heads x128xK/V.
            mla=MLAAttentionSpec(block_size=target,num_kv_heads=1,head_size=512,dtype=torch.bfloat16)
            draft=SlidingWindowSpec(block_size=target,num_kv_heads=8,head_size=128,
                                    dtype=torch.bfloat16,sliding_window=2048)
            cfg=NS(parallel_config=NS(pipeline_parallel_size=1),speculative_config=None)
            groups=_glm5_next_draft_groups(cfg,{'draft':draft},{'mla':mla},['mla'],mla.page_size_bytes)
            self.assertEqual(len(groups),1)
            spec=groups[0].kv_cache_spec.kv_cache_specs['draft']
            self.assertEqual(spec.block_size,expected)
            self.assertEqual(math.lcm(target,expected),target)
            self.assertEqual(2048 % expected,0)
            self.assertEqual(spec.page_size_bytes,mla.page_size_bytes)
            self.assertLessEqual(spec.unpadded_page_size_bytes,spec.page_size_bytes)
            # Allocation offsets for logical slots: no slot reaches padding or
            # another page, including positions straddling the target boundary.
            for p in range(target-2048,target+505):
                bid,off=divmod(p,expected)
                addr=bid*spec.page_size_bytes+off*4096
                self.assertLess(addr+4095,(bid+1)*spec.page_size_bytes)

if __name__=='__main__':unittest.main()
