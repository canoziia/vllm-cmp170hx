#!/usr/bin/env python3
"""patch 0004: VLLM_MIMO26_SORT_MOE_ALIGN, bitwise vs the torch deterministic align.

MiMo routing (256 experts, top-8), 1..512 tokens (8..4096 entries), several
block sizes, int32 and int64 ids, invalid ids; plus timing at the c32 MTP3
decode size (128 tokens = 1024 entries). Needs one GPU:

    python3 test_sort_moe_align_gpu.py
"""
import os

os.environ["VLLM_DSV4_FAST_DET_MOE_ALIGN"] = "1"
os.environ["VLLM_MIMO26_SORT_MOE_ALIGN"] = "1"

import torch

import vllm.model_executor.layers.fused_moe  # noqa: F401  (import order)
from vllm.model_executor.layers.fused_moe import fast_det_align as F
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    _moe_align_block_size_deterministic,
)
from vllm.triton_utils import triton

E, TOPK = 256, 8
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)


def bufs(S, BS):
    cap = S + E * (BS - 1)
    if S < E:
        cap = min(S * BS, cap)
    return (torch.full((cap,), -7, dtype=torch.int32, device=dev),
            torch.full((triton.cdiv(cap, BS),), -7, dtype=torch.int32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev))


def ids(M, dtype):
    s = torch.rand(M, E, generator=g, device=dev) ** 3
    t = torch.topk(s, TOPK, dim=1).indices.to(dtype).contiguous()
    return t


checked = 0
for BS in (8, 16, 32, 64):
    for M in (1, 3, 8, 32, 63, 64, 65, 100, 128, 200, 256, 384, 511, 512):
        for dtype in (torch.int32, torch.int64):
            for trial in range(3):
                t = ids(M, dtype)
                if trial == 2:
                    t.view(-1)[:: 7] = -1  # invalid entries are dropped
                a, b = bufs(t.numel(), BS), bufs(t.numel(), BS)
                _moe_align_block_size_deterministic(t, E, BS, *a, None)
                used = F.try_align(t, E, BS, *b, None)
                assert used, (M, used)
                for x, y in zip(a, b):
                    assert torch.equal(x, y), (BS, M, dtype, trial)
                checked += 1
print(f"PASS {checked} cases bitwise equal (8..4096 entries, sort kernel)")

for M in (16, 64, 128, 256):
    t = ids(M, torch.int32)
    for name, fn in (("torch", lambda b: _moe_align_block_size_deterministic(t, E, 16, *b, None)),
                     ("fast", lambda b: F.try_align(t, E, 16, *b, None))):
        b = bufs(t.numel(), 16)
        for _ in range(5):
            fn(b)
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(200):
            fn(b)
        en.record(); torch.cuda.synchronize()
        print(f"  {M * TOPK:5d} entries {name:5s} {st.elapsed_time(en) / 200 * 1000:7.1f} us/call (eager)")
