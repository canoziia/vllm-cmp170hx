"""MiMo MXFP4 decode v2 (_mimo26_moe2): dequant variants DQ0-3, fused act (plain
SiLU via limit=+inf) and fused top-k sum, vs v1 and Marlin.

    python3 test_mxfp4_decode_v2_gpu.py [--lib PATH] [--bench]

Checks: DQ3 == DQ0 bitwise (every e8m0 scale <= 128); fused act with limit=inf
== sum + _C.silu_and_mul bitwise; fused sum == moe_sum (reports per order);
error vs fp64 <= Marlin's; graph replay == eager. --bench: timings.
"""
import argparse
import math
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
V1 = torch.ops._mimo26_moe_C
V2 = torch.ops._mimo26_moe2

from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe  # noqa
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size  # noqa
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa
    rand_marlin_weight_mxfp4_like)
from vllm.scalar_type import scalar_types  # noqa

dev = torch.device("cuda")
K, NI, TOPK = 4096, 2048, 8
QT = scalar_types.float4_e2m1f.id
INF = math.inf
fails = []
_ST = {}


def state():
    key = torch.cuda.current_stream().cuda_stream
    if key not in _ST:
        _ST[key] = (torch.zeros(2, dtype=torch.int32, device=dev),
                    torch.zeros(32768, dtype=torch.int32, device=dev))
    return _ST[key]


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


def run(x, w1, w1s, w2, w2s, tw, ids, cfg=2, dq=0, fact=False, fsum=False, order=1):
    """Full MoE output [M, K] via v2 gemm (v1 sum + silu_and_mul when not fused)."""
    M = x.size(0); E = w1.size(0); slots = M * TOPK
    ctr, sync = state()
    sid, eids, ntpp = moe_align_block_size(ids, 8, E, None, ignore_invalid_experts=True)
    part = torch.empty(4 * slots * 2 * NI, dtype=torch.float32, device=dev)
    h = torch.empty(slots, NI, dtype=torch.bfloat16, device=dev)
    twf = tw.view(-1)
    w1b, w2b = w1s.view(torch.uint8), w2s.view(torch.uint8)
    if fact:
        V2.gemm(x, w1, w1b, sid, eids, ntpp, twf, TOPK, slots, K, 2 * NI, True, cfg, dq,
                True, INF, 0, ctr, sync, part, h)
    else:
        dummy = torch.empty(0, dtype=torch.bfloat16, device=dev)
        ks = V2.gemm(x, w1, w1b, sid, eids, ntpp, twf, TOPK, slots, K, 2 * NI, True, cfg, dq,
                     False, 0.0, 0, ctr, sync, part, dummy)
        y = torch.empty(slots, 2 * NI, dtype=torch.bfloat16, device=dev)
        V1.sum(part, ks, slots, 2 * NI, y)
        torch.ops._C.silu_and_mul(h, y)
    c3 = torch.empty(slots, K, dtype=torch.bfloat16, device=dev)
    if fsum:
        out = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
        V2.gemm(h, w2, w2b, sid, eids, ntpp, twf, TOPK, slots, NI, K, False, cfg, dq,
                True, 0.0, order, ctr, sync, c3, out)
        return out, h
    dummy = torch.empty(0, dtype=torch.bfloat16, device=dev)
    V2.gemm(h, w2, w2b, sid, eids, ntpp, twf, TOPK, slots, NI, K, False, cfg, dq,
            False, 0.0, 0, ctr, sync, c3, dummy)
    out = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
    torch.ops._moe_C.moe_sum(c3.view(M, TOPK, K), out)
    return out, h


def marlin(x, w1, w1s, w2, w2s, tw, ids):
    return fused_marlin_moe(x, w1, w2, None, None, w1s, w2s, tw, ids, QT,
                            activation=MoEActivation.SILU)


def ref64(x, w1r, w2r, tw, ids):
    out = torch.zeros(x.size(0), K, dtype=torch.float64, device=dev)
    for m in range(x.size(0)):
        for k in range(TOPK):
            e = int(ids[m, k])
            a = x[m].double() @ w1r[e].double()
            g, u = a[:NI], a[NI:]
            out[m] += float(tw[m, k]) * ((g * torch.sigmoid(g) * u) @ w2r[e].double())
    return out


def bits(t):
    return t.contiguous().view(torch.uint8)


E = 16
w1, w1s, w1r, w2, w2s, w2r = make_experts(E, 0)
print("max e8m0 scale byte:", int(w1s.view(torch.uint8).max()), int(w2s.view(torch.uint8).max()))
for M in (1, 2, 4, 8, 16, 32):
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.rand(M, TOPK, device=dev).softmax(-1)
    base, hb = run(x, w1, w1s, w2, w2s, tw, ids, dq=0)
    r = ref64(x, w1r, w2r, tw, ids)
    mar = marlin(x, w1, w1s, w2, w2s, tw, ids)
    sc = r.abs().mean()
    em = (mar.double() - r).abs().mean() / sc
    line = [f"M={M:2d} marlin err {em:.2e}"]
    for dq in (1, 2, 3):
        o, _ = run(x, w1, w1s, w2, w2s, tw, ids, dq=dq)
        e = (o.double() - r).abs().mean() / sc
        same = torch.equal(bits(o), bits(base))
        line.append(f"dq{dq} err {e:.2e} {'==dq0' if same else '!=dq0'}")
        if dq == 3 and not same:
            fails.append(f"dq3!=dq0 M={M}")
        if e > 1.2 * em + 1e-12:
            fails.append(f"dq{dq} accuracy M={M}")
    o, hf = run(x, w1, w1s, w2, w2s, tw, ids, fact=True)
    act_eq = torch.equal(bits(hf), bits(hb))
    line.append(f"fused act==sum+silu_and_mul {act_eq}")
    if not act_eq:
        fails.append(f"fused act M={M}")
    if M <= 16:
        for order in (0, 1, 2):
            o, _ = run(x, w1, w1s, w2, w2s, tw, ids, fsum=True, order=order)
            line.append(f"fsum{order}=={torch.equal(bits(o), bits(base))}")
    print("  ".join(line))

M = 4
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
tw = torch.rand(M, TOPK, device=dev).softmax(-1)
for dq in (0, 3):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run(x, w1, w1s, w2, w2s, tw, ids, dq=dq, fact=True, fsum=True)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        og = run(x, w1, w1s, w2, w2s, tw, ids, dq=dq, fact=True, fsum=True)[0]
    ok = True
    for i in range(20):
        x.copy_(torch.randn(M, K, dtype=torch.bfloat16, device=dev))
        ids.copy_(torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32))
        g.replay()
        ok &= torch.equal(og, run(x, w1, w1s, w2, w2s, tw, ids, dq=dq, fact=True, fsum=True)[0])
    print(f"dq{dq} graph==eager {ok}")
    if not ok:
        fails.append(f"graph dq{dq}")

if args.bench:
    del w1r, w2r
    E = 256
    w1 = w1[:1].repeat(E, 1, 1).contiguous(); w1s = w1s[:1].repeat(E, 1, 1).contiguous()
    w2 = w2[:1].repeat(E, 1, 1).contiguous(); w2s = w2s[:1].repeat(E, 1, 1).contiguous()
    per = (w1[0].numel() * w1.element_size() + w2[0].numel() * w2.element_size()
           + w1s[0].numel() + w2s[0].numel())

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

    for M, distinct in ((1, 8), (4, 24), (8, 48), (16, 90), (32, 150)):
        pool = torch.randperm(E, device=dev)[:distinct]
        ids = torch.stack([pool[torch.randperm(distinct, device=dev)[:TOPK]] for _ in range(M)]).to(torch.int32)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        tw = torch.rand(M, TOPK, device=dev).softmax(-1)
        u = ids.unique().numel()
        tm = timed(lambda: marlin(x, w1, w1s, w2, w2s, tw, ids))
        res = []
        for dq in (0, 2, 3):
            best = min((timed(lambda: run(x, w1, w1s, w2, w2s, tw, ids, cfg=c, dq=dq,
                                          fact=True, fsum=M <= 16)), c) for c in range(6))
            res.append(f"dq{dq}:{best[0]:.0f}(cfg{best[1]}, {u * per / best[0] / 1e3:.0f} GB/s)")
        print(f"M={M:2d} distinct={u:3d} marlin {tm:6.1f} us ({u * per / tm / 1e3:4.0f} GB/s)  "
              + "  ".join(res))

print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
