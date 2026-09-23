#!/usr/bin/env python3
"""GPU regression for group-scoped CoW with unrelated pages unchanged."""

import torch

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

for page in (64, 1032, 16384):
    first = torch.arange(16 * page, device="cuda", dtype=torch.int64).view(16, page)
    second = (first + 100000).clone()
    expected_first = first.clone()
    expected_second = second.clone()
    copies = [
        KVCacheBlockCopy(1, 10, 0),
        KVCacheBlockCopy(2, 11, 0),
        KVCacheBlockCopy(3, 12, 1),
    ]
    copy_kv_cache_blocks_inplace(
        [first, second], 16, copies, kv_caches_by_group=[[first, first], [second]]
    )
    expected_first[10] = expected_first[1]
    expected_first[11] = expected_first[2]
    expected_second[12] = expected_second[3]
    torch.cuda.synchronize()
    assert torch.equal(first, expected_first)
    assert torch.equal(second, expected_second)
print("GROUP_COW_BYTE_EQUAL_AND_UNRELATED_PAGES_UNCHANGED PASS")
