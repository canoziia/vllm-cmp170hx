"""GPU test: fast deterministic MoE align == torch deterministic align, bitwise.

Run inside the DeepSeek V4.1 Flash image (single sm_80 GPU):
  VLLM_DSV4_FAST_DET_MOE_ALIGN=1 python3 test_fast_det_align_gpu.py
"""
import os
import sys
import time

import torch

os.environ["VLLM_DETERMINISTIC_MOE_ALIGN"] = "1"
import vllm.model_executor.layers.fused_moe.moe_align_block_size as M  # noqa
from vllm.model_executor.layers.fused_moe import fast_det_align as F  # noqa

dev = torch.device("cuda")
torch.manual_seed(0)
fails = 0
cases = 0


def run(ids, E, bs, fast):
    os.environ["VLLM_DSV4_FAST_DET_MOE_ALIGN"] = "1" if fast else "0"
    return M.moe_align_block_size(ids, bs, E)


def check(ids, E, bs, tag):
    global fails, cases
    cases += 1
    a = run(ids, E, bs, False)
    b = run(ids, E, bs, True)
    for x, y, n in zip(a, b, ("sorted", "experts", "ntpp")):
        if x.shape != y.shape or not torch.equal(x, y):
            fails += 1
            print("FAIL", tag, n, ids.shape, E, bs)
            return


for E in (384, 128, 257, 8):
    for bs in (8, 16, 32, 64):
        for m in (1, 2, 3, 6, 12, 30, 48, 64, 96, 192, 341):
            for topk in (6, 3, 1):
                if m * topk > F.MAX_ENTRIES:
                    continue
                for trial in range(3):
                    if trial == 0:
                        ids = torch.randint(0, E, (m, topk), device=dev)
                    elif trial == 1:  # heavy collisions
                        ids = torch.randint(0, min(E, 5), (m, topk), device=dev)
                    else:  # invalid / padded rows
                        ids = torch.randint(-1, E + 2, (m, topk), device=dev)
                    for dt in (torch.int32, torch.int64):
                        check(ids.to(dt).contiguous(), E, bs, f"t{trial}")

# CUDA graph capture/replay with changing contents
os.environ["VLLM_DSV4_FAST_DET_MOE_ALIGN"] = "1"
ids = torch.randint(0, 384, (6, 6), device=dev, dtype=torch.int32)
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    M.moe_align_block_size(ids, 16, 384)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = M.moe_align_block_size(ids, 16, 384)
for _ in range(200):
    ids.copy_(torch.randint(0, 384, (6, 6), device=dev, dtype=torch.int32))
    g.replay()
    ref = run(ids, 384, 16, False)
    cases += 1
    if not all(torch.equal(x, y) for x, y in zip(out, ref)):
        fails += 1
        print("FAIL graph")
        break

# micro-benchmark, graphs (the deployed path)
for m in (6, 48, 192):
    ids = torch.randint(0, 384, (m, 6), device=dev, dtype=torch.int32)
    res = {}
    for fast in (False, True):
        os.environ["VLLM_DSV4_FAST_DET_MOE_ALIGN"] = "1" if fast else "0"
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            M.moe_align_block_size(ids, 16, 384)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for _ in range(20):
                M.moe_align_block_size(ids, 16, 384)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(50):
            g.replay()
        torch.cuda.synchronize()
        res["fast" if fast else "torch"] = (time.perf_counter() - t) / 1000 * 1e6
    print(f"M={m} us/call torch {res['torch']:.1f} fast {res['fast']:.1f}")

print(f"{cases} cases, {fails} failures")
sys.exit(1 if fails else 0)
