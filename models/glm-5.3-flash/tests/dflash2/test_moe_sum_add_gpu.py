# SPDX-License-Identifier: Apache-2.0
"""Patch 0026 (dev 0027, VLLM_GLM5_MOE_SUM_ADD): bitwise + micro-benchmark, one GPU.

    # in the GLM DFlash2 image (or PYTHONPATH=<patched tree>)
    python3 test_moe_sum_add_gpu.py \
        [--iters 200] [--json out.jsonl]

1. Bitwise: ``moe_sum_add(c3, shared)`` vs the incumbent
   ``ops.moe_sum(c3, routed); shared + routed`` for M in 1..32, topk 8,
   H 4096 (+ odd H and non-contiguous shared), several value regimes
   (normal, large dynamic range with cancellation, subnormals, ±0, inf).
   NaN positions must match too. Any mismatch -> exit 1.
2. CUDA graph: both paths captured and replayed; still bitwise equal.
3. End-to-end hand-off: compiled Marlin decode ``run`` with the deferral armed
   + ``moe_sum_add`` vs ``run`` unarmed + add (needs _ampere_marlin_C; skipped
   if missing).
4. Micro-benchmark: graph of 11 layers' (sum + add) vs 11 fused launches,
   us per layer, M in {4, 8}.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ["VLLM_GLM5_MOE_SUM_ADD"] = "1"

import torch  # noqa: E402

from vllm import _custom_ops as ops  # noqa: E402
from vllm.models.glm5next.nvidia.ops import moe_sum_add as msa  # noqa: E402


def incumbent(c3, shared):
    routed = torch.empty_like(shared, memory_format=torch.contiguous_format)
    ops.moe_sum(c3, routed)
    return shared + routed


def gen(M, topk, H, regime, g, dev):
    if regime == "normal":
        c3 = torch.randn(M, topk, H, generator=g) * 0.05
        sh = torch.randn(M, H, generator=g) * 0.5
    elif regime == "cancel":  # large dynamic range, heavy cancellation
        c3 = torch.randn(M, topk, H, generator=g) * torch.pow(
            10.0, torch.randint(-6, 4, (M, topk, H), generator=g).float())
        sh = -c3.sum(1) + torch.randn(M, H, generator=g) * 1e-3
    elif regime == "tiny":    # subnormals and signed zeros
        c3 = torch.randn(M, topk, H, generator=g) * 1e-39
        c3[..., ::7] = -0.0
        sh = torch.randn(M, H, generator=g) * 1e-39
        sh[:, ::5] = -0.0
    elif regime == "special":
        c3 = torch.randn(M, topk, H, generator=g)
        c3[0, 0, 0] = float("inf")
        c3[0, 1, 1] = float("-inf")
        c3[0, 2, 1] = float("inf")
        c3[-1, 3, 5] = float("nan")
        sh = torch.randn(M, H, generator=g)
        sh[0, 2] = float("inf")
        c3[:, :, 3] = 3.0e38     # overflow of the fp32 sum -> inf in bf16
    else:
        raise ValueError(regime)
    return c3.to(torch.bfloat16).to(dev), sh.to(torch.bfloat16).to(dev)


def same_bits(a, b):
    """Bit-identical, except NaN payload/sign (both NaN at the same spot)."""
    both_nan = a.isnan() & b.isnan()
    eq = (a.view(torch.int16) == b.view(torch.int16)) | both_nan
    return bool(eq.all()) and torch.equal(a.isnan(), b.isnan())


def check_bitwise(dev):
    g = torch.Generator().manual_seed(0)
    bad = 0
    cases = 0
    for H in (4096, 4100, 96):
        for M in list(range(1, 33)):
            for regime in ("normal", "cancel", "tiny", "special"):
                c3, sh = gen(M, 8, H, regime, g, dev)
                shared_variants = [sh]
                if H == 4096:
                    wide = torch.zeros(M, H + 64, dtype=sh.dtype, device=dev)
                    wide[:, :H] = sh
                    shared_variants.append(wide[:, :H])   # row stride != H
                for s in shared_variants:
                    cases += 1
                    ref = incumbent(c3, s)
                    out = msa.moe_sum_add(c3, s)
                    if not same_bits(out, ref):
                        nan_ok = torch.equal(out.isnan(), ref.isnan())
                        diff = (out.float() - ref.float()).abs().nan_to_num(0)
                        print(f"MISMATCH H={H} M={M} {regime} stride={s.stride()} "
                              f"nan_pos_equal={nan_ok} max|d|={diff.max().item():.3e} "
                              f"n={(~((out == ref) | (out.isnan() & ref.isnan()))).sum().item()}")
                        bad += 1
    print(f"bitwise: {cases - bad}/{cases} cases identical")
    return bad == 0


def check_graph(dev):
    g = torch.Generator().manual_seed(1)
    ok = True
    for M in (4, 8, 32):
        c3, sh = gen(M, 8, 4096, "normal", g, dev)
        o_ref = torch.empty_like(sh)
        o_new = torch.empty_like(sh)
        incumbent(c3, sh), msa.moe_sum_add(c3, sh)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            o_ref.copy_(incumbent(c3, sh))
            msa.moe_sum_add(c3, sh, out=o_new)
        c3.copy_(gen(M, 8, 4096, "cancel", g, dev)[0])
        gr.replay()
        torch.cuda.synchronize()
        ok &= same_bits(o_ref, o_new) and same_bits(o_new, incumbent(c3, sh))
    print(f"cuda graph: {'identical' if ok else 'MISMATCH'}")
    return ok


def check_handoff(dev):
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from marlin_decode_common import build_layer, routing
        from vllm.models.glm5next.nvidia.ops import marlin_decode as md
        md.require_extension()
    except Exception as exc:  # noqa: BLE001
        print(f"handoff: SKIPPED ({type(exc).__name__}: {exc})")
        return True
    layer = build_layer(E=64, device=dev)
    ok = True
    for M in (4, 8):
        x, ids, w = routing(M, layer, seed=M, device=dev)
        sh = (torch.randn(M, layer.K, device=dev) * 0.5).to(torch.bfloat16)
        out = torch.empty_like(x)
        md.run(layer.stub, out, x, layer.w1, layer.w2, w, ids)
        ref = sh + out
        out2 = torch.empty_like(x)
        msa.DEFERRAL.arm()
        md.run(layer.stub, out2, x, layer.w1, layer.w2, w, ids)
        p = msa.DEFERRAL.take()
        assert p is not None and p[1] is out2, "run() did not offer its c3"
        new = msa.moe_sum_add(p[0], sh)
        ok &= same_bits(new, ref)
        # not armed -> unchanged behaviour
        out3 = torch.empty_like(x)
        md.run(layer.stub, out3, x, layer.w1, layer.w2, w, ids)
        ok &= msa.DEFERRAL.take() is None and same_bits(out3, out)
    print(f"handoff via marlin_decode.run: {'identical' if ok else 'MISMATCH'}")
    return ok


def bench(dev, iters, layers=11):
    rows = []
    g = torch.Generator().manual_seed(2)
    for M in (4, 8):
        data = [gen(M, 8, 4096, "normal", g, dev) for _ in range(layers)]
        outs = [torch.empty_like(s) for _, s in data]
        res = {}
        for name in ("incumbent", "fused"):
            for (c3, s), o in zip(data, outs):
                o.copy_(incumbent(c3, s)) if name == "incumbent" else msa.moe_sum_add(c3, s, out=o)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for (c3, s), o in zip(data, outs):
                    if name == "incumbent":
                        r = torch.empty_like(s)
                        ops.moe_sum(c3, r)
                        torch.add(s, r, out=o)
                    else:
                        msa.moe_sum_add(c3, s, out=o)
            ts = []
            for _ in range(5):
                st, en = torch.cuda.Event(True), torch.cuda.Event(True)
                st.record()
                for _ in range(iters):
                    gr.replay()
                en.record()
                torch.cuda.synchronize()
                ts.append(st.elapsed_time(en) * 1e3 / iters / layers)
            res[name] = statistics.median(ts)
        row = {"M": M, "us_per_layer_incumbent": round(res["incumbent"], 3),
               "us_per_layer_fused": round(res["fused"], 3),
               "saved_us_per_layer": round(res["incumbent"] - res["fused"], 3)}
        print(row)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    ok = check_bitwise(dev) & check_graph(dev) & check_handoff(dev)
    rows = bench(dev, a.iters)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"test": "0027", "ok": bool(ok), "bench": rows}) + "\n")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
