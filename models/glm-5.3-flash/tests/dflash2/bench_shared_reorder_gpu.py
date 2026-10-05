# SPDX-License-Identifier: Apache-2.0
"""Patch 0027 (dev 0028, VLLM_GLM5_SHARED_EXPERT_REORDER) hypothesis test, one GPU.

Question: does the compiled Marlin decode kernel get slower when the
shared-expert GEMMs on the aux stream are enqueued (graph nodes created)
*before* it rather than after it?  This reproduces one PP4 rank's MoE layer
in a CUDA graph, both enqueue orders, with the same fork/join edges:

    upstream : mark(main) ; aux: wait, shared ; main: marlin ; join
    reorder  : mark(main) ; main: marlin ; aux: wait, shared ; join
    serial   : main: marlin ; main: shared              (no overlap)
    marlin   : main: marlin only                       (lower bound)

Shared expert = GLM-5.3-Flash n_shared_experts=1 MLP: thin_gemm 4096->4096
(gate_up), silu*mul, thin_gemm 2048->4096 (down); our thin_gemm kernel when
available (VLLM_GLM5_THIN_GEMM path), else torch.mm.

    python3 bench_shared_reorder_gpu.py \
        [--tokens 4,8] [--layers 11] [--iters 100] [--json out.jsonl]

Reports us per layer (graph replay) and, from torch.profiler, the mean
duration of the Marlin decode GEMM kernels and of _thin_gemm_kernel per order.
Bitwise: outputs of all orders must be identical (same kernels, same inputs).
"""

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marlin_decode_common import build_layer, routing  # noqa: E402


def shared_mlp(x, wgu, wd):
    try:
        from vllm.models.glm5next.nvidia.ops.thin_gemm import thin_gemm
        gu = thin_gemm(x, wgu)
        g, u = gu.chunk(2, dim=-1)
        return thin_gemm((torch.nn.functional.silu(g) * u).contiguous(), wd)
    except Exception:  # noqa: BLE001
        gu = x @ wgu.t()
        g, u = gu.chunk(2, dim=-1)
        return (torch.nn.functional.silu(g) * u) @ wd.t()


def build_graph(order, layer, inputs, wgu, wd, aux):
    from vllm.models.glm5next.nvidia.ops.marlin_decode import run

    outs = []
    def body():
        outs.clear()
        for x, ids, w in inputs:
            o = torch.empty_like(x)
            main = torch.cuda.current_stream()
            if order in ("upstream", "reorder"):
                ev_in, ev_out = torch.cuda.Event(), torch.cuda.Event()
                ev_in.record(main)
                if order == "reorder":
                    run(layer.stub, o, x, layer.w1, layer.w2, w, ids)
                with torch.cuda.stream(aux):
                    ev_in.wait(aux)
                    s = shared_mlp(x, wgu, wd)
                    ev_out.record(aux)
                if order == "upstream":
                    run(layer.stub, o, x, layer.w1, layer.w2, w, ids)
                ev_out.wait(main)
                outs.append(s + o)
            elif order == "serial":
                run(layer.stub, o, x, layer.w1, layer.w2, w, ids)
                outs.append(shared_mlp(x, wgu, wd) + o)
            else:
                run(layer.stub, o, x, layer.w1, layer.w2, w, ids)
                outs.append(o)
    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    return g, list(outs)


def time_graph(g, iters):
    ts = []
    for _ in range(5):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(iters):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1e3 / iters)
    return statistics.median(ts)


def kernel_means(g, iters):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
    acc = collections.defaultdict(list)
    for e in p.events():
        if e.device_type.name != "CUDA":
            continue
        n = e.name
        key = ("marlin_gemm" if "decode_gemm" in n or "moe_dec_gemm" in n
               else "marlin_act" if "decode_act" in n or "moe_dec_act" in n
               else "thin_gemm" if "thin_gemm" in n else None)
        if key:
            acc[key].append(e.device_time if hasattr(e, "device_time") else e.cuda_time)
    return {k: round(statistics.mean(v), 2) for k, v in acc.items()}


def _safe_kernel_means(g, iters):
    try:
        return kernel_means(g, iters)
    except Exception as exc:  # noqa: BLE001 - profiler API differences
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4,8")
    ap.add_argument("--layers", type=int, default=11)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    from vllm.models.glm5next.nvidia.ops import marlin_decode as md
    md.require_extension()
    md._counters(dev)
    layer = build_layer(device=dev)
    wgu = (torch.randn(4096, 4096, device=dev) * 0.02).to(torch.bfloat16)
    wd = (torch.randn(4096, 2048, device=dev) * 0.02).to(torch.bfloat16)
    aux = torch.cuda.Stream()
    rows, ok = [], True
    for M in (int(t) for t in a.tokens.split(",")):
        inputs = [routing(M, layer, seed=100 * M + i, device=dev, pool=24)
                  for i in range(a.layers)]
        ref = None
        for order in ("marlin", "serial", "upstream", "reorder"):
            g, outs = build_graph(order, layer, inputs, wgu, wd, aux)
            g.replay()
            torch.cuda.synchronize()
            if order != "marlin":
                cur = torch.stack(outs)
                if ref is None:
                    ref = cur.clone()
                elif not torch.equal(ref.view(torch.int16), cur.view(torch.int16)):
                    ok = False
                    print(f"M={M} {order}: output differs from serial")
            row = {"M": M, "order": order,
                   "us_per_layer": round(time_graph(g, a.iters) / a.layers, 2),
                   "kernel_mean_us": _safe_kernel_means(g, max(10, a.iters // 5))}
            print(row)
            rows.append(row)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"test": "0028-bench", "ok": ok, "rows": rows}) + "\n")
    print("outputs identical across orders" if ok else "FAIL: outputs differ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
