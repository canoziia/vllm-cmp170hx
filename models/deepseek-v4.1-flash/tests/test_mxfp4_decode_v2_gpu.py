"""MXFP4 decode MoE v2 (_dsv4_moe2) vs v1 (_dsv4_moe_C) vs Marlin.

    python3 test_mxfp4_decode_v2_gpu.py --lib /path/_dsv4_moe_C.abi3.so [--bench]

Checks (single sm_80 GPU):
  0. dequant probe: one-hot activations -> w13 output must equal the
     reference weights bit for bit (dq 0/1/2; dq 1/2 also prove the tensor
     cores take the bf16 subnormal +-0.5*2^-126 exactly);
  1. v2 dq=0 partials / h / w2 out == v1 bitwise (switch-off equivalence);
     fused act h == sum + _C.silu_and_mul_with_clamp bitwise (every dq);
  2. fused moe_sum vs torch.sum and _moe_C.moe_sum for sum orders 0/1/2
     (reports which order is bitwise equal);
  3. accuracy vs fp64 reference: v1, v2 dq 0/1/2 (fused + unfused) vs Marlin;
  4. CUDA-graph replay == eager with changing routing; determinism;
  5. CUDA det align == fast_det_align (Triton) == torch deterministic align;
  6. --bench: graph-replay timing at 6/12/18/36/150 distinct experts.
"""
import argparse
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default=None)
ap.add_argument("--bench", action="store_true")
ap.add_argument("--iters", type=int, default=20)
args = ap.parse_args()
if args.lib is None:
    import os as _os
    import vllm as _v
    args.lib = _os.path.join(_os.path.dirname(_v.__file__), "_dsv4_moe_C.abi3.so")
torch.ops.load_library(args.lib)
OPS = torch.ops._dsv4_moe_C
OPS2 = torch.ops._dsv4_moe2
assert int(OPS2.abi()) == 1

from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe  # noqa
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa
    moe_align_block_size)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa
    rand_marlin_weight_mxfp4_like)
from vllm.scalar_type import scalar_types  # noqa

dev = torch.device("cuda")
K, NI, TOPK, LIMIT = 5120, 2304, 6, 10.0
QT = scalar_types.float4_e2m1f.id
fails = []


def check(cond, name):
    if not cond:
        fails.append(name)
        print("  FAIL:", name)


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


_CTR, _SYNC = {}, {}


def bufs():
    key = torch.cuda.current_stream().cuda_stream
    ctr = _CTR.setdefault(key, torch.zeros(2, dtype=torch.int32, device=dev))
    sync = _SYNC.setdefault(key, torch.zeros(32768, dtype=torch.int32, device=dev))
    return ctr, sync


def align(ids, E):
    return moe_align_block_size(ids, 8, E, None, ignore_invalid_experts=True)


def run_v1(x, w1, w1s, w2, w2s, tw, ids, cfg13=2, cfg2=2):
    M, E, slots = x.size(0), w1.size(0), x.size(0) * TOPK
    ctr, _ = bufs()
    sorted_ids, eids, ntpp = align(ids, E)
    part = torch.empty(4 * slots * 2 * NI, dtype=torch.float32, device=dev)
    y = torch.empty(slots, 2 * NI, dtype=torch.bfloat16, device=dev)
    h = torch.empty(slots, NI, dtype=torch.bfloat16, device=dev)
    c3 = torch.empty(slots, K, dtype=torch.bfloat16, device=dev)
    twf = tw.view(-1)
    ks = OPS.gemm(x, w1, w1s.view(torch.uint8), sorted_ids, eids, ntpp, twf, TOPK, slots,
                  K, 2 * NI, True, cfg13, ctr, part)
    OPS.sum(part, ks, slots, 2 * NI, y)
    torch.ops._C.silu_and_mul_with_clamp(h, y, LIMIT, 1.0, 0.0)
    OPS.gemm(h, w2, w2s.view(torch.uint8), sorted_ids, eids, ntpp, twf, TOPK, slots,
             NI, K, False, cfg2, ctr, c3)
    return dict(out=c3.view(M, TOPK, K).sum(1), part=part, h=h, y=y, c3=c3)


def run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=1, cfg13=2, cfg2=2, fact=False, fsum=False,
           order=1, routing=None):
    M, E, slots = x.size(0), w1.size(0), x.size(0) * TOPK
    ctr, sync = bufs()
    sorted_ids, eids, ntpp = routing if routing is not None else align(ids, E)
    part = torch.empty(4 * slots * 2 * NI, dtype=torch.float32, device=dev)
    h = torch.empty(slots, NI, dtype=torch.bfloat16, device=dev)
    c3 = torch.empty(slots, K, dtype=torch.bfloat16, device=dev)
    dummy = torch.empty(0, dtype=torch.bfloat16, device=dev)
    twf = tw.view(-1)
    y = None
    ks = OPS2.gemm(x, w1, w1s.view(torch.uint8), sorted_ids, eids, ntpp, twf, TOPK, slots,
                   K, 2 * NI, True, cfg13, dq, fact, LIMIT, 0, ctr, sync, part,
                   h if fact else dummy)
    if not fact:
        y = torch.empty(slots, 2 * NI, dtype=torch.bfloat16, device=dev)
        OPS.sum(part, ks, slots, 2 * NI, y)
        torch.ops._C.silu_and_mul_with_clamp(h, y, LIMIT, 1.0, 0.0)
    if fsum:
        out = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
        OPS2.gemm(h, w2, w2s.view(torch.uint8), sorted_ids, eids, ntpp, twf, TOPK, slots,
                  NI, K, False, cfg2, dq, True, 0.0, order, ctr, sync, c3, out)
    else:
        OPS2.gemm(h, w2, w2s.view(torch.uint8), sorted_ids, eids, ntpp, twf, TOPK, slots,
                  NI, K, False, cfg2, dq, False, 0.0, 0, ctr, sync, c3, dummy)
        out = c3.view(M, TOPK, K).sum(1)
    return dict(out=out, part=part, h=h, y=y, c3=c3)


def marlin(x, w1, w1s, w2, w2s, tw, ids):
    from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
    cfg = ApplyMoEActivationConfig(clamp_limit=LIMIT)
    return fused_marlin_moe(x, w1, w2, None, None, w1s, w2s, tw, ids, QT,
                            activation=MoEActivation.SILU, activation_config=cfg)


def ref64(x, w1r, w2r, tw, ids):
    out = torch.zeros(x.size(0), K, dtype=torch.float64, device=dev)
    for m in range(x.size(0)):
        for k in range(TOPK):
            e = int(ids[m, k])
            if e < 0:
                continue
            a = x[m].double() @ w1r[e].double()
            g, u = a[:NI].clamp(max=LIMIT), a[NI:].clamp(-LIMIT, LIMIT)
            hh = g * torch.sigmoid(g) * u
            out[m] += float(tw[m, k]) * (hh @ w2r[e].double())
    return out


def rand_ids(M, E):
    return torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)


E = 16
w1, w1s, w1r, w2, w2s, w2r = make_experts(E, 0)
smax = max(int(w1s.view(torch.uint8).max()), int(w2s.view(torch.uint8).max()))
smin = min(int(w1s.view(torch.uint8).min()), int(w2s.view(torch.uint8).min()))
print(f"scales e8m0 range [{smin}, {smax}]  w1 {tuple(w1.shape)}")

# ---------------------------------------------------------------- 0. dequant probe
print("[0] one-hot dequant probe (w13 output == reference weights, bitwise)")
M = 48
kk = torch.randint(0, K, (M,), device=dev)
x = torch.zeros(M, K, dtype=torch.bfloat16, device=dev)
x[torch.arange(M, device=dev), kk] = 1.0
ids = torch.randint(0, E, (M, TOPK), device=dev, dtype=torch.int32)
tw = torch.rand(M, TOPK, device=dev).softmax(-1)
expect = torch.stack([w1r[ids[m].long(), kk[m]] for m in range(M)])  # [M, TOPK, 2NI]
for dq in (0, 1, 2):
    r = run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=dq)
    got = r["y"].view(M, TOPK, 2 * NI)
    ok = torch.equal(got, expect.to(torch.bfloat16))
    print(f"  dq={dq}: exact={ok}  ({expect.numel()} weights)")
    check(ok, f"probe dq{dq}")

# ------------------------------------------- 1. v2 dq0 == v1, fused act exactness
print("[1] switch-off equivalence and fused activation")
M = 6
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
ids = rand_ids(M, E)
tw = torch.rand(M, TOPK, device=dev).softmax(-1)
for cfg in (0, 1, 2, 3, 4, 5):
    a = run_v1(x, w1, w1s, w2, w2s, tw, ids, cfg, cfg)
    b = run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=0, cfg13=cfg, cfg2=cfg)
    slots = M * TOPK
    eq = (torch.equal(a["part"], b["part"]) and torch.equal(a["h"], b["h"])
          and torch.equal(a["c3"], b["c3"]))
    print(f"  cfg {cfg}: v2 dq0 == v1 (part, h, c3): {eq}")
    check(eq, f"dq0==v1 cfg{cfg}")
for dq in (0, 1, 2):
    for cfg in (1, 2, 5):
        a = run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=dq, cfg13=cfg, cfg2=cfg, fact=False)
        b = run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=dq, cfg13=cfg, cfg2=cfg, fact=True)
        eq = torch.equal(a["h"], b["h"]) and torch.equal(a["out"], b["out"])
        print(f"  dq={dq} cfg {cfg}: fused act == sum + _C op: {eq}")
        check(eq, f"fused act dq{dq} cfg{cfg}")
a1 = run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=1)
a2 = run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=2)
eq = torch.equal(a1["part"], a2["part"]) and torch.equal(a1["c3"], a2["c3"])
print(f"  dq1 == dq2 bitwise: {eq}")
check(eq, "dq1==dq2")

# ------------------------------------------------------------ 2. fused moe_sum
print("[2] fused moe_sum vs torch.sum / _moe_C.moe_sum")
order_ok = {}
for M in (1, 6, 16):
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    ids = rand_ids(M, E)
    tw = torch.rand(M, TOPK, device=dev).softmax(-1)
    base = run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=1)
    c3 = base["c3"].view(M, TOPK, K)
    ts = torch.sum(c3, dim=1)
    ms = torch.empty_like(ts)
    torch.ops._moe_C.moe_sum(c3, ms, None, None)
    for order in (0, 1, 2):
        f = run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=1, fsum=True, order=order)["out"]
        e1, e2 = torch.equal(f, ts), torch.equal(f, ms)
        order_ok.setdefault(order, [True, True])
        order_ok[order][0] &= e1
        order_ok[order][1] &= e2
        print(f"  M={M:2d} order {order}: == torch.sum {e1}   == _moe_C.moe_sum {e2}"
              f"   (max |diff| vs moe_sum {(f.float() - ms.float()).abs().max().item():.3e})")
good = [o for o, (a, b) in order_ok.items() if b]
print("  orders bitwise equal to _moe_C.moe_sum (deployed MarlinExperts path):", good)
print("  orders bitwise equal to torch.sum (fused_marlin_moe moe_sum=None):",
      [o for o, (a, b) in order_ok.items() if a])
check(bool(good), "no fused sum order matches _moe_C.moe_sum")

# --------------------------------------------------------------- 3. accuracy
print("[3] accuracy vs fp64 (relative to mean |ref|)")
variants = [("v1", lambda *a: run_v1(*a)["out"])]
for dq in (0, 1, 2):
    variants.append((f"dq{dq}", lambda *a, dq=dq: run_v2(*a, dq=dq)["out"]))
    variants.append((f"dq{dq}F", lambda *a, dq=dq: run_v2(*a, dq=dq, fact=True, fsum=True,
                                                           order=good[0] if good else 1)["out"]))
agg = {n: [0.0, 0.0, 0] for n, _ in variants}
aggm = [0.0, 0.0, 0]
for M in (1, 2, 6, 12, 48):
    for trial in range(3):
        x = (torch.randn(M, K, device=dev) * (0.5 + trial)).to(torch.bfloat16)
        ids = (torch.randint(0, 3, (M, TOPK), device=dev, dtype=torch.int32) if trial == 2
               else rand_ids(M, E))
        tw = torch.rand(M, TOPK, device=dev).softmax(-1)
        r = ref64(x, w1r, w2r, tw, ids)
        scale = r.abs().mean()
        mar = marlin(x, w1, w1s, w2, w2s, tw, ids).double()
        em = (mar - r).abs()
        aggm[0] += (em.mean() / scale).item(); aggm[1] = max(aggm[1], (em.max() / scale).item())
        aggm[2] += 1
        line = f"  M={M:2d} t{trial} marlin {em.mean()/scale:.2e}/{em.max()/scale:.2e}"
        for name, fn in variants:
            if M > 16 and name.endswith("F"):
                continue
            o = fn(x, w1, w1s, w2, w2s, tw, ids).double()
            eo = (o - r).abs()
            agg[name][0] += (eo.mean() / scale).item()
            agg[name][1] = max(agg[name][1], (eo.max() / scale).item())
            agg[name][2] += 1
            line += f" | {name} {eo.mean()/scale:.2e}/{eo.max()/scale:.2e}"
            if eo.mean() > 1.2 * em.mean() + 1e-12 or eo.max() > 2.0 * em.max() + 1e-9:
                fails.append(f"accuracy {name} M={M} t{trial}")
        print(line)
print(f"  summary mean-of-mean/max  marlin {aggm[0]/aggm[2]:.3e}/{aggm[1]:.3e}")
for n, (s, mx, c) in agg.items():
    print(f"  summary {n:5s} {s/c:.3e}/{mx:.3e}")

# ------------------------------------------------- 4. graph replay + determinism
print("[4] CUDA graph replay and determinism")
for dq, fact, fsum in ((1, True, True), (2, True, True), (0, True, False), (1, False, False)):
    M = 6
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    ids = rand_ids(M, E)
    tw = torch.rand(M, TOPK, device=dev).softmax(-1)
    fn = lambda: run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=dq, fact=fact, fsum=fsum,
                        order=good[0] if good else 1)["out"]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = fn()
    bad = False
    for i in range(30):
        x.copy_(torch.randn(M, K, dtype=torch.bfloat16, device=dev))
        ids.copy_(rand_ids(M, E))
        g.replay()
        e = fn()
        if not torch.equal(out_g, e):
            bad = True
            break
    det = all(torch.equal(fn(), e) for _ in range(5))
    print(f"  dq={dq} fact={fact} fsum={fsum}: graph==eager {not bad}  deterministic {det}")
    check(not bad, f"graph dq{dq} {fact} {fsum}")
    check(det, f"determinism dq{dq} {fact} {fsum}")

# ------------------------------------------------------------------- 5. align
print("[5] CUDA det align vs Triton fast_det_align vs torch deterministic")
try:
    from vllm.model_executor.layers.fused_moe import fast_det_align
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        _moe_align_block_size_deterministic)
    from vllm.triton_utils import triton
    from vllm.utils.math_utils import round_up

    def outs(ids, E, BS=8):
        cap = ids.numel() + E * (BS - 1)
        cap = round_up(cap, BS)
        if ids.numel() < E:
            cap = min(ids.numel() * BS, cap)
        nb = triton.cdiv(cap, BS)
        return (torch.full((cap,), -7, dtype=torch.int32, device=dev),
                torch.full((nb,), -7, dtype=torch.int32, device=dev),
                torch.full((1,), -7, dtype=torch.int32, device=dev))

    ok_all = True
    for E_ in (16, 384):
        for S_tok in (1, 6, 8, 48, 64, 85):
            for dt in (torch.int32, torch.int64):
                ids_ = torch.randint(0, E_, (S_tok, TOPK), device=dev, dtype=dt)
                if S_tok == 8:
                    ids_[0, 0] = -1
                a = outs(ids_, E_); b = outs(ids_, E_); c = outs(ids_, E_)
                OPS2.align(ids_.view(-1), E_, 8, *a)
                fast_det_align.try_align(ids_, E_, 8, *b, None)
                _moe_align_block_size_deterministic(ids_, E_, 8, *c, None)
                ab = all(torch.equal(p, q) for p, q in zip(a, b))
                ac = all(torch.equal(p, q) for p, q in zip(a, c))
                ok_all &= ab and ac
                if not (ab and ac):
                    print(f"  mismatch E={E_} S={S_tok} {dt}: ==triton {ab} ==torch {ac}")
    print("  align identical:", ok_all)
    check(ok_all, "align")
except ImportError as e:
    print("  skipped (", e, ")")

# ------------------------------------------------------------------- 6. bench
if args.bench:
    del w1r, w2r
    E = 384
    w1 = w1[:1].repeat(E, 1, 1).contiguous(); w1s = w1s[:1].repeat(E, 1, 1).contiguous()
    w2 = w2[:1].repeat(E, 1, 1).contiguous(); w2s = w2s[:1].repeat(E, 1, 1).contiguous()
    per = (w1[0].numel() * w1.element_size() + w2[0].numel() * w2.element_size()
           + w1s[0].numel() + w2s[0].numel())
    per13 = w1[0].numel() * w1.element_size() + w1s[0].numel()
    per2 = w2[0].numel() * w2.element_size() + w2s[0].numel()
    order = good[0] if good else 1

    def timed(fn, reps=10):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        g.replay(); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.iters):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / (args.iters * reps) * 1e6

    print("[6] graph-replay us per MoE call (lower is better); GB/s on expert bytes")
    for M, distinct in ((1, 6), (6, 12), (6, 18), (6, 36), (48, 150)):
        pool = torch.randperm(E, device=dev)[:distinct]
        ids = pool[torch.arange(M * TOPK, device=dev) % distinct].view(M, TOPK).to(torch.int32)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        tw = torch.rand(M, TOPK, device=dev).softmax(-1)
        u = ids.unique().numel()
        routing = align(ids, E)
        slots = M * TOPK
        ctr, sync = bufs()
        part = torch.empty(4 * slots * 2 * NI, dtype=torch.float32, device=dev)
        h = torch.empty(slots, NI, dtype=torch.bfloat16, device=dev)
        c3 = torch.empty(slots, K, dtype=torch.bfloat16, device=dev)
        dummy = torch.empty(0, dtype=torch.bfloat16, device=dev)
        twf = tw.view(-1)
        sid, eid, ntpp = routing
        tm = timed(lambda: marlin(x, w1, w1s, w2, w2s, tw, ids))
        line = f"M={M:2d} experts={u:3d}  marlin {tm:6.1f} ({u*per/tm/1e3:4.0f} GB/s)"
        for cfg in (2, 5):
            t1 = timed(lambda: run_v1(x, w1, w1s, w2, w2s, tw, ids, cfg, cfg))
            line += f" | v1 c{cfg} {t1:6.1f}"
            for dq in (0, 1, 2):
                t2 = timed(lambda: run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=dq, cfg13=cfg,
                                          cfg2=cfg))
                t3 = timed(lambda: run_v2(x, w1, w1s, w2, w2s, tw, ids, dq=dq, cfg13=cfg,
                                          cfg2=cfg, fact=True, fsum=M <= 16, order=order))
                line += f" dq{dq} {t2:6.1f}/F {t3:6.1f}"
        print(line)
        # GEMM-only bandwidth per kernel (no align / act / sum)
        gl = "    gemm-only GB/s:"
        for cfg in (2, 5):
            for dq in (0, 1, 2):
                a13 = timed(lambda: OPS2.gemm(x, w1, w1s.view(torch.uint8), sid, eid, ntpp,
                                              twf, TOPK, slots, K, 2 * NI, True, cfg, dq,
                                              False, LIMIT, 0, ctr, sync, part, dummy))
                a2 = timed(lambda: OPS2.gemm(h, w2, w2s.view(torch.uint8), sid, eid, ntpp,
                                             twf, TOPK, slots, NI, K, False, cfg, dq, False,
                                             0.0, 0, ctr, sync, c3, dummy))
                gl += (f" c{cfg}dq{dq} w13 {u*per13/a13/1e3:4.0f} ({a13:5.1f}us)"
                       f" w2 {u*per2/a2/1e3:4.0f} ({a2:5.1f}us)")
        print(gl)
        try:
            from vllm.model_executor.layers.fused_moe import fast_det_align
            ids_f = ids.contiguous()
            o_t = [t.clone() for t in routing]
            o_c = [t.clone() for t in routing]
            tt = timed(lambda: fast_det_align.try_align(ids_f, E, 8, *o_t, None))
            tc = timed(lambda: OPS2.align(ids_f.view(-1), E, 8, *o_c))
            print(f"    align: triton {tt:5.1f} us   cuda {tc:5.1f} us")
        except ImportError:
            pass

print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
