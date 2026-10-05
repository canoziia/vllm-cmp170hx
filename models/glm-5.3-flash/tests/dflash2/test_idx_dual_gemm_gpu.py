# SPDX-License-Identifier: Apache-2.0
"""Patch 0029 (VLLM_GLM5_IDX_DUAL_GEMM): bitwise / error / graph / streams +
micro-benchmark, one sm_80 GPU.

    # in the GLM DFlash2 image (or PYTHONPATH=<patched tree>)
    python3 test_idx_dual_gemm_gpu.py [--ms 1-32] [--iters 200] [--json out.jsonl]

The indexer projection is W [128 + 32, 4096] bf16 (k rows, then head-weight
rows). Old path: ``thin_gemm(x, W)`` (k = first 128 columns) + ``torch.mm(
x.float(), W[128:].T.float())``. New path: one dual launch.

1. k columns: bitwise equal to the thin GEMM's first 128 columns, every M in
   --ms, several input regimes. Any mismatch -> exit 1.
2. Weight columns (fp32): error against an fp64 product of the same bf16
   inputs, compared with the old fp32 sgemm's error on the same inputs.
   Gate: mean |err| <= 1.5x and max |err| <= 1.5x the sgemm's, per regime
   (MM measured 0.62x / 0.51x; on outlier/cancel inputs we measured
   1.07x / 1.24x mean with a lower max -- rounding-level, hence 1.5x). Also printed: max relative error.
3. Custom op: equals ``thin_gemm_dual`` bitwise; rows outside the bound
   (M=33, 64) and an un-warmed M inside a capture give exactly the old
   two-op result (thin-GEMM gate + sgemm).
4. Determinism: 5 runs bitwise identical (split-K last-arriver reduce).
5. CUDA graph: captured op replayed on new inputs == eager, bitwise.
6. Two streams: the same shape on the main stream and, inside
   ``thin_gemm.private_workspace("pp_draft_tail")``, on a side stream,
   interleaved 200x without synchronisation -> each result bitwise its
   eager reference (split-K counters are per scope).
7. Micro-benchmark (CUDA graph of 3 layers, i.e. one PP stage's MLA layers):
   old (thin GEMM + cast + sgemm) vs dual, us per layer, M in {4, 8, 16}.
"""

import argparse
import json
import os
import statistics
import sys

os.environ["VLLM_GLM5_THIN_GEMM"] = "1"
os.environ["VLLM_GLM5_IDX_DUAL_GEMM"] = "1"

import torch  # noqa: E402

from vllm.models.glm5next.nvidia.ops import idx_dual_gemm as dg  # noqa: E402
from vllm.models.glm5next.nvidia.ops import thin_gemm as tg  # noqa: E402

N_LO, N_HEAD, K = 128, 32, 4096
N = N_LO + N_HEAD
FAILS: list[str] = []


def fail(msg):
    FAILS.append(msg)
    print("FAIL", msg)


def parse_ms(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def gen_x(M, regime, g, dev):
    x = torch.randn(M, K, generator=g)
    if regime == "outlier":  # a few massive channels, like post-norm hidden
        idx = torch.randint(0, K, (8,), generator=g)
        x[:, idx] *= 60.0
    elif regime == "cancel":  # rows that nearly cancel against W
        x = x * torch.pow(10.0, torch.randint(-3, 3, (M, K), generator=g).float())
    return x.to(torch.bfloat16).to(dev)


def gen_w(g, dev):
    w = torch.randn(N, K, generator=g) * 0.02
    w[N_LO:] *= 3.0  # head-weight rows: different scale than wk
    return w.to(torch.bfloat16).to(dev)


def old_path(x, w, w_hi_t):
    kw = torch.ops.vllm.glm5_thin_linear(x, w)
    return kw[:, :N_LO], torch.mm(x.float(), w_hi_t)


def errors(hi, ref):
    e = (hi.double() - ref).abs()
    return e.mean().item(), e.max().item(), (e / ref.abs().clamp_min(1e-30)).max().item()


def check_numerics(ms, dev, report):
    g = torch.Generator().manual_seed(0)
    w = gen_w(g, dev)
    w_hi_t = w[N_LO:].t().contiguous().float()
    w64 = w[N_LO:].double()
    for regime in ("normal", "outlier", "cancel"):
        agg = {"dual_mean": [], "sgemm_mean": [], "dual_max": 0.0, "sgemm_max": 0.0,
               "dual_rel": 0.0}
        for M in ms:
            if not dg.kernel_supported(M, N, K, N_LO):
                fail(f"gate refuses M={M}")
                continue
            x = gen_x(M, regime, g, dev)
            lo_full, hi = dg.thin_gemm_dual(x, w, N_LO)
            thin = tg.thin_gemm(x, w)
            if not torch.equal(lo_full[:, :N_LO], thin[:, :N_LO]):
                fail(f"k columns not bitwise, M={M} {regime}")
            ref = x.double() @ w64.t()
            dm, dx, dr = errors(hi, ref)
            sm, sx, _ = errors(torch.mm(x.float(), w_hi_t), ref)
            agg["dual_mean"].append(dm)
            agg["sgemm_mean"].append(sm)
            agg["dual_max"] = max(agg["dual_max"], dx)
            agg["sgemm_max"] = max(agg["sgemm_max"], sx)
            agg["dual_rel"] = max(agg["dual_rel"], dr)
            runs = [dg.thin_gemm_dual(x, w, N_LO) for _ in range(5)]
            if not all(torch.equal(r[1], hi) and torch.equal(r[0][:, :N_LO], lo_full[:, :N_LO])
                       for r in runs):
                fail(f"not deterministic, M={M} {regime}")
        dmean = statistics.fmean(agg["dual_mean"])
        smean = statistics.fmean(agg["sgemm_mean"])
        row = {"test": "error", "regime": regime, "dual_mean": dmean, "sgemm_mean": smean,
               "mean_ratio": dmean / smean, "dual_max": agg["dual_max"],
               "sgemm_max": agg["sgemm_max"], "max_ratio": agg["dual_max"] / agg["sgemm_max"],
               "dual_max_rel": agg["dual_rel"]}
        report(row)
        print(f"  {regime:8s} weights |err| mean {dmean:.3e} vs sgemm {smean:.3e} "
              f"({row['mean_ratio']:.2f}x), max {agg['dual_max']:.3e} vs "
              f"{agg['sgemm_max']:.3e} ({row['max_ratio']:.2f}x)")
        # Same order of magnitude as the incumbent fp32 sgemm vs fp64. On the
        # adversarial outlier/cancellation inputs the mean error measured
        # 1.07x/1.24x the sgemm's with a lower max (node2, CMP 170HX); this is
        # a different-summation-order rounding difference, not a regression.
        if row["mean_ratio"] > 1.5 or row["max_ratio"] > 1.5:
            fail(f"weights error above the sgemm bound ({regime})")


def check_op(dev):
    g = torch.Generator().manual_seed(1)
    w = gen_w(g, dev)
    w_hi_t = w[N_LO:].t().contiguous().float()
    for M in (1, 4, 8, 16, 32):
        x = gen_x(M, "normal", g, dev)
        k, wt = dg.idx_dual_linear(x, w, w_hi_t, N_LO)
        lo_full, hi = dg.thin_gemm_dual(x, w, N_LO)
        if not (torch.equal(k, lo_full[:, :N_LO]) and torch.equal(wt, hi)):
            fail(f"op != thin_gemm_dual, M={M}")
        if k.stride() != (N, 1):
            fail(f"k view strides {k.stride()} != {(N, 1)}")
    for M in (33, 64):  # outside the thin-GEMM bound: the old ops, exactly
        x = gen_x(M, "normal", g, dev)
        k, wt = dg.idx_dual_linear(x, w, w_hi_t, N_LO)
        k0, wt0 = old_path(x, w, w_hi_t)
        if not (torch.equal(k, k0) and torch.equal(wt, wt0)):
            fail(f"fallback M={M} not bitwise the old path")
    # un-warmed M inside a capture -> old path, bitwise
    M = 7
    dg._READY.discard((torch.device(dev).index, M, N, K))
    tg._READY.add((torch.device(dev).index, M, N, K))
    tg.thin_gemm(gen_x(M, "normal", g, dev), w)  # thin binary exists
    x = gen_x(M, "normal", g, dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
        k_g, wt_g = dg.idx_dual_linear(x, w, w_hi_t, N_LO)
    graph.replay()
    torch.cuda.synchronize()
    k0, wt0 = old_path(x, w, w_hi_t)
    if not (torch.equal(k_g, k0) and torch.equal(wt_g, wt0)):
        fail("capture fallback not bitwise the old path")
    print("  op / fallback checks done")


def check_graph(dev):
    g = torch.Generator().manual_seed(2)
    w = gen_w(g, dev)
    w_hi_t = w[N_LO:].t().contiguous().float()
    for M in (4, 8, 16):
        x = gen_x(M, "normal", g, dev)
        dg._glm5_idx_dual_linear_impl(x, w, w_hi_t, N_LO)  # warm
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
            k_g, wt_g = dg.idx_dual_linear(x, w, w_hi_t, N_LO)
        for _ in range(3):
            x.copy_(gen_x(M, "outlier", g, dev))
            graph.replay()
            torch.cuda.synchronize()
            k_e, wt_e = dg.idx_dual_linear(x, w, w_hi_t, N_LO)
            if not (torch.equal(k_g, k_e) and torch.equal(wt_g, wt_e)):
                fail(f"graph replay != eager, M={M}")
    print("  graph replay checks done")


def check_streams(dev, reps=200):
    g = torch.Generator().manual_seed(3)
    w = gen_w(g, dev)
    w_hi_t = w[N_LO:].t().contiguous().float()
    M = 8
    xs = [gen_x(M, "normal", g, dev) for _ in range(4)]
    refs = [dg.thin_gemm_dual(x, w, N_LO) for x in xs]
    with tg.private_workspace("pp_draft_tail"):
        dg.thin_gemm_dual(xs[0], w, N_LO)  # allocate the private workspace
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    outs_main, outs_side = [], []
    for i in range(reps):
        a, b = i % 4, (i + 1) % 4
        outs_main.append((a, dg.thin_gemm_dual(xs[a], w, N_LO)))
        with torch.cuda.stream(side), tg.private_workspace("pp_draft_tail"):
            outs_side.append((b, dg.thin_gemm_dual(xs[b], w, N_LO)))
    torch.cuda.synchronize()
    bad = 0
    for idx, (lo, hi) in outs_main + outs_side:
        if not (torch.equal(lo[:, :N_LO], refs[idx][0][:, :N_LO])
                and torch.equal(hi, refs[idx][1])):
            bad += 1
    if bad:
        fail(f"two-stream interleave: {bad} of {2 * reps} results differ")
    print(f"  two-stream interleave: {2 * reps} results checked")


def bench(dev, iters, report, layers=3):
    g = torch.Generator().manual_seed(4)
    ws = [gen_w(g, dev) for _ in range(layers)]
    wts = [w[N_LO:].t().contiguous().float() for w in ws]
    for M in (4, 8, 16):
        x = gen_x(M, "normal", g, dev)
        for w, wt in zip(ws, wts):
            old_path(x, w, wt)
            dg._glm5_idx_dual_linear_impl(x, w, wt, N_LO)
        res = {}
        for name, fn in (("old", old_path), ("dual", dg.idx_dual_linear)):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
                for w, wt in zip(ws, wts):
                    fn(x, w, wt) if name == "old" else fn(x, w, wt, N_LO)
            torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                graph.replay()
                b.record()
                b.synchronize()
                times.append(a.elapsed_time(b) * 1e3 / layers)
            res[name] = statistics.median(times)
        row = {"test": "bench", "M": M, "old_us": res["old"], "dual_us": res["dual"],
               "saved_us": res["old"] - res["dual"]}
        report(row)
        print(f"  M={M:2d}: old {res['old']:.2f} us/layer, dual {res['dual']:.2f} "
              f"us/layer, saved {row['saved_us']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1-32")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--json")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    if torch.cuda.get_device_capability() != (8, 0):
        print("needs sm_80")
        return 2
    dev = "cuda"
    jf = open(args.json, "a") if args.json else None

    def report(row):
        if jf:
            jf.write(json.dumps(row) + "\n")

    print(f"torch {torch.__version__}, triton "
          f"{__import__('triton').__version__}, {torch.cuda.get_device_name()}")
    print("1-2/4. k bitwise, weights error, determinism")
    check_numerics(parse_ms(args.ms), dev, report)
    print("3. custom op")
    check_op(dev)
    print("5. CUDA graph")
    check_graph(dev)
    print("6. two streams")
    check_streams(dev)
    if not args.no_bench:
        print("7. micro-benchmark (3 MLA layers in one graph)")
        bench(dev, args.iters, report)
    print("FAILED: " + "; ".join(FAILS) if FAILS else "all passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
