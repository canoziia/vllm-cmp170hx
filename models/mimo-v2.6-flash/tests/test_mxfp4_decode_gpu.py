"""MiMo MXFP4 decode MoE kernel (_mimo26_moe_C) vs the deployed Marlin path.

    python3 test_mxfp4_decode_gpu.py [--lib PATH] [--bench]

--lib defaults to the library installed in the image (vllm/_mimo26_moe_C.abi3.so).
MiMo experts: hidden 4096, intermediate 2048, 256 experts, top-8, plain SiLU.

Checks (single sm_80 GPU):
  1. full MoE vs fused_marlin_moe: same exact weights, only the fp32 summation
     order differs -> error vs an fp64 reference no worse than Marlin's;
  2. CUDA-graph replay with changing routing equals eager; determinism;
  3. --bench: graph-replay timings at decode sizes, with the weight bytes read.
"""
import argparse
import os
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default=None)
ap.add_argument("--bench", action="store_true")
args = ap.parse_args()
if args.lib is None:
    import vllm
    args.lib = os.path.join(os.path.dirname(vllm.__file__), "_mimo26_moe_C.abi3.so")
torch.ops.load_library(args.lib)
OPS = torch.ops._mimo26_moe_C

from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe  # noqa
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa
    moe_align_block_size)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa
    rand_marlin_weight_mxfp4_like)
from vllm.scalar_type import scalar_types  # noqa

dev = torch.device("cuda")
K, NI, TOPK = 4096, 2048, 8
QT = scalar_types.float4_e2m1f.id
fails = []


def make_experts(E, seed):
    torch.manual_seed(seed)
    w1, w1s, w1r, w2, w2s, w2r = [], [], [], [], [], []
    for _ in range(E):
        ref, q, s = rand_marlin_weight_mxfp4_like(
            torch.empty(2 * NI, K, dtype=torch.bfloat16, device=dev), 32)[:3]
        w1r.append(ref); w1.append(q); w1s.append(s)
        ref, q, s = rand_marlin_weight_mxfp4_like(
            torch.empty(K, NI, dtype=torch.bfloat16, device=dev), 32)[:3]
        w2r.append(ref); w2.append(q); w2s.append(s)
    st = lambda xs: torch.stack(xs).contiguous()
    return st(w1), st(w1s), st(w1r), st(w2), st(w2s), st(w2r)


_CTR = {}


def run(x, w1, w1s, w2, w2s, tw, ids, cfg13=0, cfg2=0):
    M = x.size(0)
    E = w1.size(0)
    slots = M * TOPK
    key = torch.cuda.current_stream().cuda_stream
    ctr = _CTR.setdefault(key, torch.zeros(2, dtype=torch.int32, device=dev))
    sorted_ids, eids, ntpp = moe_align_block_size(ids, 8, E, None, ignore_invalid_experts=True)
    part = torch.empty(4 * slots * 2 * NI, dtype=torch.float32, device=dev)
    y = torch.empty(slots, 2 * NI, dtype=torch.bfloat16, device=dev)
    h = torch.empty(slots, NI, dtype=torch.bfloat16, device=dev)
    c3 = torch.empty(slots, K, dtype=torch.bfloat16, device=dev)
    twf = tw.view(-1)
    ks = OPS.gemm(x, w1, w1s.view(torch.uint8), sorted_ids, eids, ntpp, twf, TOPK, slots,
                  K, 2 * NI, True, cfg13, ctr, part)
    OPS.sum(part, ks, slots, 2 * NI, y)
    torch.ops._C.silu_and_mul(h, y)
    OPS.gemm(h, w2, w2s.view(torch.uint8), sorted_ids, eids, ntpp, twf, TOPK, slots,
             NI, K, False, cfg2, ctr, c3)
    return c3.view(M, TOPK, K).sum(1)


def marlin(x, w1, w1s, w2, w2s, tw, ids):
    return fused_marlin_moe(x, w1, w2, None, None, w1s, w2s, tw, ids, QT,
                            activation=MoEActivation.SILU)


def ref64(x, w1r, w2r, tw, ids):
    out = torch.zeros(x.size(0), K, dtype=torch.float64, device=dev)
    for m in range(x.size(0)):
        for k in range(TOPK):
            e = int(ids[m, k])
            a = x[m].double() @ w1r[e].double()  # ref is [K, 2*NI]
            g, u = a[:NI], a[NI:]
            out[m] += float(tw[m, k]) * ((g * torch.sigmoid(g) * u) @ w2r[e].double())
    return out


E = 16
w1, w1s, w1r, w2, w2s, w2r = make_experts(E, 0)
print("w1", tuple(w1.shape), "scales", tuple(w1s.shape), w1s.dtype)

for M in (1, 2, 4, 8, 16, 32):
    for trial in range(3):
        x = (torch.randn(M, K, device=dev) * (0.5 + trial)).to(torch.bfloat16)
        if trial == 2:
            ids = torch.randint(0, 3, (M, TOPK), device=dev, dtype=torch.int32)  # collisions
        else:
            ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
        tw = torch.rand(M, TOPK, device=dev).softmax(-1)
        ours = run(x, w1, w1s, w2, w2s, tw, ids).float()
        mar = marlin(x, w1, w1s, w2, w2s, tw, ids).float()
        r = ref64(x, w1r, w2r, tw, ids)
        eo, em = (ours.double() - r).abs(), (mar.double() - r).abs()
        sc = r.abs().mean()
        print(f"M={M:2d} t{trial} |ours-ref| mean {eo.mean()/sc:.2e} max {eo.max()/sc:.2e}"
              f"  |marlin-ref| mean {em.mean()/sc:.2e} max {em.max()/sc:.2e}"
              f"  bitwise==marlin {torch.equal(ours, mar)}")
        if eo.mean() > 1.2 * em.mean() + 1e-12 or eo.max() > 2.0 * em.max() + 1e-9:
            fails.append(f"accuracy M={M} t{trial}")

M = 4
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
tw = torch.rand(M, TOPK, device=dev).softmax(-1)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    run(x, w1, w1s, w2, w2s, tw, ids)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out_g = run(x, w1, w1s, w2, w2s, tw, ids)
for i in range(30):
    x.copy_(torch.randn(M, K, dtype=torch.bfloat16, device=dev))
    ids.copy_(torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32))
    g.replay()
    if not torch.equal(out_g, run(x, w1, w1s, w2, w2s, tw, ids)):
        fails.append("graph"); break
a = run(x, w1, w1s, w2, w2s, tw, ids)
b = run(x, w1, w1s, w2, w2s, tw, ids)
print("graph == eager:", "graph" not in fails, " deterministic:", torch.equal(a, b))
if not torch.equal(a, b):
    fails.append("determinism")

if args.bench:
    del w1r, w2r
    E = 256
    w1 = w1[:1].repeat(E, 1, 1).contiguous(); w1s = w1s[:1].repeat(E, 1, 1).contiguous()
    w2 = w2[:1].repeat(E, 1, 1).contiguous(); w2s = w2s[:1].repeat(E, 1, 1).contiguous()
    per = (w1[0].numel() * w1.element_size() + w2[0].numel() * w2.element_size()
           + w1s[0].numel() + w2s[0].numel())
    print(f"bytes per expert {per / 1e6:.2f} MB")

    def timed(fn):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(10):
                fn()
        g.replay(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / 200 * 1e6

    for M, distinct in ((1, 8), (2, 14), (4, 24), (4, 31), (8, 48), (16, 90), (32, 150), (64, 210)):
        pool = torch.randperm(E, device=dev)[:distinct]
        ids = torch.stack([pool[torch.randperm(distinct, device=dev)[:TOPK]] for _ in range(M)]).to(torch.int32)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        tw = torch.rand(M, TOPK, device=dev).softmax(-1)
        u = ids.unique().numel()
        tm = timed(lambda: marlin(x, w1, w1s, w2, w2s, tw, ids))
        res = []
        for c in range(6):
            to = timed(lambda: run(x, w1, w1s, w2, w2s, tw, ids, c, c))
            res.append(f"{c}:{to:.0f}")
        print(f"M={M:2d} distinct={u:3d} marlin {tm:6.1f} us ({u * per / tm / 1e3:4.0f} GB/s)  ours "
              + " ".join(res))

print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
