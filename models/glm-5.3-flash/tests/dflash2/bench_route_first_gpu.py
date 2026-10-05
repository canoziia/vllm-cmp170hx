"""0023 route v2: GPU numerics + microbenchmark, ours (orig / 0023) vs MM.

NOT RUN LOCALLY (no GPU).  Needs torch + triton on an sm_80 device; vllm is
NOT imported: all three kernels are loaded by FILE PATH (the two vllm imports
in our module are stubbed), so it runs in either image.

    python3 bench_route_first_gpu.py \
        [--orig fixtures/route_v2_decode_pre0023.py]       # route v2 before 0023
        [--new  $GLM_DFLASH2_TREE/vllm/models/glm5next/nvidia/ops/route_v2_decode.py]
        [--mm   <Morrowmake tree>/vllm/ampere_decode/moe_route.py] \
        [--layers 45] [--iters 200] [--ms 1,4,8] [--json out.json]

Timing: one CUDA graph per pipeline with --layers routers back to back, each
with its own [288,4096] weight + bias (L2 rotation like the model), CUDA
events over replays, us per layer.  Three scenarios:
  alone       router launches only
  co_before   0013 order: a side-stream "shared expert" (2 bf16 GEMMs
              [M,4096]x[4096,SH] ...) forked BEFORE each router -> co-runs
  co_after    0023 order: fork recorded AFTER the router -> no co-run
The co_* numbers are the whole fork/router/join sequence per layer; compare
co_before vs co_after for the in-model effect of VLLM_GLM5_ROUTE_V2_FIRST.

Numerics (exit code 1 on failure):
  * new vs orig, gate and tc mode: every output BITWISE equal (0023 only
    reorders loads); this is the "gate mode unchanged" check.
  * tc vs MM: logits bitwise (same split-K order); ids equal; weights within
    1 ulp-ish (MM divides before multiplying).
  * tc vs an fp64 reference (torch, x@W.T in float64): expert SET may differ only where the reference gap
    between the 8th and 9th key is < TIE_TOL; weights rel err < W_TOL where
    the sets agree.
"""
import argparse
import importlib.util
import json
import os
import pathlib
import sys
import tempfile

import torch

E, K, RSF, TOPK = 288, 4096, 2.5, 8
TIE_TOL = 2e-6     # key units; tc logit err ~1e-6 * sigmoid' <= 0.25
W_TOL = 1e-5       # relative, weights of identical sets


def load(path, name):
    src = pathlib.Path(path).read_text()
    src = src.replace("from vllm.logger import init_logger",
                      "import logging\ninit_logger = logging.getLogger")
    src = src.replace("from vllm.triton_utils import tl, triton",
                      "import triton\nimport triton.language as tl")
    tmp = pathlib.Path(tempfile.mkdtemp()) / f"{name}.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location(name, tmp)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ref_route(logits64, bias):
    s = torch.sigmoid(logits64)
    key = s + bias.double()
    top = key.topk(TOPK + 1, dim=1)
    ids = top.indices[:, :TOPK]
    gap = top.values[:, TOPK - 1] - top.values[:, TOPK]
    w = s.gather(1, ids)
    w = w / w.sum(1, keepdim=True) * RSF
    return ids, w, gap


def check_numerics(orig, new, mm, M, dev, fails, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed + M)
    x = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
    w = (torch.randn(E, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    b = torch.randn(E, device=dev, generator=g) * 0.01
    try:
        lg = torch.mm(x, w.T, out_dtype=torch.float32)
    except TypeError:
        lg = (x.float() @ w.float().T)
    res = {}
    for mode in ("gate", "tc"):
        kw = dict(logits=lg.clone()) if mode == "gate" else dict(x=x, weight=w)
        o = orig.route_v2(b, renormalize=True, routed_scaling_factor=RSF, **kw)
        kw = dict(logits=lg.clone()) if mode == "gate" else dict(x=x, weight=w)
        n = new.route_v2(b, renormalize=True, routed_scaling_factor=RSF, **kw)
        torch.cuda.synchronize()
        same = all(torch.equal(a, c) for a, c in zip(o, n))
        res[f"{mode}_new_vs_orig_bitwise"] = same
        if not same:
            fails.append(f"M={M} {mode}: 0023 not bitwise equal to 0013")
        res[mode] = n
    # tc vs MM
    if mm is not None:
        m_out = mm.moe_route(x, w, b, topk=8, block_size=8, num_experts=E,
                             renormalize=True, routed_scaling_factor=RSF)
        torch.cuda.synchronize()
        tc = res["tc"]
        res["mm_logits_bitwise"] = bool(torch.equal(tc[0], m_out[0])) if M > 1 else None
        res["mm_ids_equal"] = bool(torch.equal(tc[2], m_out[2]))
        res["mm_w_max_rel"] = float(((tc[1] - m_out[1]).abs() / m_out[1].abs()
                                     .clamp_min(1e-30)).max())
    # tc vs reference
    x64, w64 = x.double(), w.double()
    ids_r, w_r, gap = ref_route(x64 @ w64.T, b)
    tc = res["tc"]
    ids_t = tc[2].long()
    set_eq = (ids_t.sort(1).values == ids_r.sort(1).values).all(1)
    flips = (~set_eq).nonzero().flatten().tolist()
    bad = [r for r in flips if gap[r] >= TIE_TOL]
    res["tc_set_flips"] = len(flips)
    res["tc_flip_gaps"] = [float(gap[r]) for r in flips]
    if bad:
        fails.append(f"M={M} tc: set flip at non-tie rows {bad}")
    if set_eq.any():
        # compare per expert (order may differ on exact-ish key ties)
        wt = torch.zeros(M, E, device=dev, dtype=torch.float64)
        wr = torch.zeros_like(wt)
        wt.scatter_(1, ids_t, tc[1].double())
        wr.scatter_(1, ids_r, w_r)
        rel = ((wt - wr).abs() / wr.abs().clamp_min(1e-30))[set_eq]
        rel = rel[wr[set_eq] != 0]
        res["tc_w_max_rel_vs_fp64"] = float(rel.max())
        if float(rel.max()) > W_TOL:
            fails.append(f"M={M} tc: weight rel err {float(rel.max()):.2e}")
    del res["gate"], res["tc"]
    return res


def time_graph(fn, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    for _ in range(iters):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / iters


def bench(orig, new, mm, M, dev, layers, iters, shared_inter=2048):
    x = torch.randn(M, K, device=dev).to(torch.bfloat16)
    lw = [((torch.randn(E, K, device=dev) * 0.02).to(torch.bfloat16),
           torch.randn(E, device=dev) * 0.01) for _ in range(layers)]
    # stand-in shared expert: up [2*I,K] then down [K,I] bf16 (cuBLAS)
    sw = [((torch.randn(2 * shared_inter, K, device=dev) * 0.02).to(torch.bfloat16),
           (torch.randn(K, shared_inter, device=dev) * 0.02).to(torch.bfloat16))
          for _ in range(layers)]
    aux = torch.cuda.Stream()

    def r_orig(i):
        orig.route_v2(lw[i][1], x=x, weight=lw[i][0], routed_scaling_factor=RSF)

    def r_new(i):
        new.route_v2(lw[i][1], x=x, weight=lw[i][0], routed_scaling_factor=RSF)

    def r_mm(i):
        mm.moe_route(x, lw[i][0], lw[i][1], routed_scaling_factor=RSF)

    def shared(i):
        h = x @ sw[i][0].T
        a, c = h.chunk(2, 1)
        return (torch.nn.functional.silu(a) * c) @ sw[i][1].T

    def seq(router, order):
        def f():
            for i in range(layers):
                if order == "alone":
                    router(i)
                    continue
                cur = torch.cuda.current_stream()
                if order == "co_after":
                    router(i)
                aux.wait_stream(cur)
                with torch.cuda.stream(aux):
                    shared(i)
                if order == "co_before":
                    router(i)
                cur.wait_stream(aux)
        return f

    out = {}
    routers = {"ours_tc_0013": r_orig, "ours_tc_0023": r_new}
    if mm is not None:
        routers["mm"] = r_mm
    for name, r in routers.items():
        for order in ("alone", "co_before", "co_after"):
            out[f"{name}/{order}"] = time_graph(seq(r, order), iters) / layers
    out["shared_only"] = time_graph(
        lambda: [shared(i) for i in range(layers)], iters) / layers
    return out


def main():
    ap = argparse.ArgumentParser()
    here = pathlib.Path(__file__).resolve().parent
    tree = pathlib.Path(os.environ.get("GLM_DFLASH2_TREE", "/usr/local/lib/python3.12/dist-packages"))
    ap.add_argument("--orig", default=str(here / "fixtures/route_v2_decode_pre0023.py"))
    ap.add_argument("--new", default=str(tree / "vllm/models/glm5next/nvidia/ops/route_v2_decode.py"))
    ap.add_argument("--mm", default=None)
    ap.add_argument("--layers", type=int, default=45)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ms", default="1,4,8")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda", torch.cuda.current_device())
    orig, new = load(a.orig, "rv_orig"), load(a.new, "rv_new")
    mm = load(a.mm, "mm_moe_route") if a.mm else None
    fails, report = [], {"device": torch.cuda.get_device_name(dev)}
    for M in [int(v) for v in a.ms.split(",")]:
        num = {}
        for seed in range(20):
            r = check_numerics(orig, new, mm, M, dev, fails, seed * 101)
            for k, v in r.items():
                num.setdefault(k, []).append(v)
        t = bench(orig, new, mm, M, dev, a.layers, a.iters)
        report[f"M={M}"] = {"numerics": num, "us_per_layer": t}
        print(f"M={M}", json.dumps({k: round(v, 2) for k, v in t.items()}))
        print("   flips", sum(num["tc_set_flips"]), "w_rel",
              max(num.get("tc_w_max_rel_vs_fp64", [0])),
              "mm_ids_eq", all(num.get("mm_ids_equal", [True])),
              "mm_logits_bitwise", num.get("mm_logits_bitwise", [None])[0])
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1))
    print("FAIL" if fails else "PASS", *fails[:20], sep="\n  ")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
