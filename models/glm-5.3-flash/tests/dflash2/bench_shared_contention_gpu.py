# SPDX-License-Identifier: Apache-2.0
"""Shared experts (aux stream) vs routed Marlin contention, one PP4 rank's MoE
layers in a CUDA graph (STEP-GAP-2 section 7). Diagnostic tool; needs no
patch switch (it builds its own streams). One sm_80 GPU with
vllm._ampere_marlin_C.

    VLLM_GLM5_THIN_GEMM=1 python3 bench_shared_contention_gpu.py \
        [--tokens 8] [--layers 11] [--iters 100] [--trace-dir /tmp/sc] [--json out.jsonl]

Each layer, as the runner enqueues it with VLLM_GLM5_SHARED_EXPERT_REORDER=1:

    main: mark ; <router prelude> ; marlin w13, act, w2 ; join
    aux : wait(mark) ; gate_up thin ; silu*mul ; down thin ; record

Variants (all with the same fork/join edges):
  prelude  none   no router kernels (node2: 363.8 us/layer at M=8)
           gate   our gate mode: _bf16_gemv_kernel (288 CTAs x 8 warps) +
                  _route_v2_kernel  (old compose; node2 385.5 us/layer)
           tc     route v2 tc mode: one launch, 72 GEMV CTAs + M routing CTAs
                  (MM's _moe_route_kernel structure; compose now; node2
                  375.9 us/layer)
  aux      default  torch.cuda.Stream()            (both trees today)
           high     greatest stream priority (evaluated as 0031, dropped:
                    tc+high 381.3 vs tc+default 375.9 us/layer on node2)
  serial   reference: everything on main (no overlap)

Reports per variant: us per layer (graph replay), and from torch.profiler the
mean us and other-stream overlap fraction of moe_dec_gemm and of the aux
_thin_gemm_kernel (the same split trace_overlap_split.py does on the real
model). Also checks that the outputs of every variant are bitwise equal
(scheduling must not change values) and whether a priority captured into a
graph has any effect at all (high vs default replay times and kernel order).
--trace-dir writes one chrome trace per variant for trace_overlap_split.py.
"""

import argparse
import collections
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("VLLM_GLM5_THIN_GEMM", "1")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marlin_decode_common import build_layer, routing  # noqa: E402

E, K = 288, 4096


def shared_mlp(x, wgu, wd):
    from vllm.models.glm5next.nvidia.ops.thin_gemm import thin_gemm

    gu = thin_gemm(x, wgu)
    g, u = gu.chunk(2, dim=-1)
    return thin_gemm((torch.nn.functional.silu(g) * u).contiguous(), wd)


def prelude(kind, x, gate_w, bias):
    if kind == "none":
        return
    from vllm.models.glm5next.nvidia.ops.route_v2_decode import route_v2

    if kind == "gate":
        from vllm.model_executor.kernels.linear.gemv_triton import bf16_gemv

        logits = bf16_gemv(x, gate_w, out_dtype=torch.float32)
        route_v2(bias, logits=logits, renormalize=True, routed_scaling_factor=2.5)
    else:
        route_v2(bias, x=x, weight=gate_w, renormalize=True, routed_scaling_factor=2.5)


def build(variant, layer, inputs, wgu, wd, gate_w, bias, aux):
    from vllm.models.glm5next.nvidia.ops.marlin_decode import run

    kind = variant[0]
    outs = []

    def body():
        outs.clear()
        for x, ids, w in inputs:
            o = torch.empty_like(x)
            main = torch.cuda.current_stream()
            if aux is None:
                prelude(kind, x, gate_w, bias)
                run(layer.stub, o, x, layer.w1, layer.w2, w, ids)
                outs.append(shared_mlp(x, wgu, wd) + o)
                continue
            ev_in, ev_out = torch.cuda.Event(), torch.cuda.Event()
            ev_in.record(main)
            prelude(kind, x, gate_w, bias)
            run(layer.stub, o, x, layer.w1, layer.w2, w, ids)
            with torch.cuda.stream(aux):
                ev_in.wait(aux)
                s = shared_mlp(x, wgu, wd)
                ev_out.record(aux)
            ev_out.wait(main)
            outs.append(s + o)

    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    return g, outs


def time_graph(g, iters):
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / iters)
    return statistics.median(ts)


def profile_split(g, iters, trace_path=None):
    """mean us and other-stream overlap fraction per kernel family."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
    if trace_path:
        p.export_chrome_trace(trace_path)
    ks = []
    for e in p.events():
        if e.device_type.name != "CUDA":
            continue
        st = e.time_range.start
        en = e.time_range.end
        ks.append((getattr(e, "device_resource_id", None) or e.thread, st, en, e.name))
    ks.sort(key=lambda k: k[1])
    fam = {"marlin_w13": lambda n: "moe_dec_gemm" in n or "decode_gemm" in n,
           "thin": lambda n: "_thin_gemm_kernel" in n}
    res = {}
    for name, pred in fam.items():
        durs, ov, tot = [], 0.0, 0.0
        for k in ks:
            if not pred(k[3]):
                continue
            d = k[2] - k[1]
            o = 0.0
            for q in ks:
                if q[1] >= k[2]:
                    break
                if q[0] != k[0] and q[2] > k[1]:
                    o += min(k[2], q[2]) - max(k[1], q[1])
            durs.append(d)
            ov += min(o, d)
            tot += d
        if durs:
            res[name] = {"mean_us": round(statistics.fmean(durs), 2),
                         "overlap": round(ov / tot, 3), "n": len(durs)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="8")
    ap.add_argument("--layers", type=int, default=11)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--trace-dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    if torch.cuda.get_device_capability() != (8, 0):
        print("needs sm_80")
        return 2
    dev = torch.device("cuda")
    from vllm.models.glm5next.nvidia.ops import marlin_decode as md

    md.require_extension()
    md._counters(dev)
    layer = build_layer(device=dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    wgu = (torch.randn(4096, 4096, device=dev, generator=gen) * 0.02).to(torch.bfloat16)
    wd = (torch.randn(4096, 2048, device=dev, generator=gen) * 0.02).to(torch.bfloat16)
    gate_w = (torch.randn(E, K, device=dev, generator=gen) * 0.02).to(torch.bfloat16)
    bias = torch.randn(E, device=dev, generator=gen) * 0.01
    least, greatest = torch.cuda.Stream.priority_range()
    streams = {"default": torch.cuda.Stream(), "high": torch.cuda.Stream(priority=greatest)}
    print(f"torch {torch.__version__}, priority range {least}..{greatest}")
    if a.trace_dir:
        os.makedirs(a.trace_dir, exist_ok=True)
    rows, ok = [], True
    for M in (int(t) for t in a.tokens.split(",")):
        inputs = [routing(M, layer, seed=100 * M + i, device=dev, pool=24)
                  for i in range(a.layers)]
        ref = None
        variants = [("none", "serial")] + [
            (kind, s) for kind in ("none", "gate", "tc") for s in ("default", "high")]
        for kind, sname in variants:
            aux = None if sname == "serial" else streams[sname]
            g, outs = build((kind,), layer, inputs, wgu, wd, gate_w, bias, aux)
            g.replay()
            torch.cuda.synchronize()
            cur = torch.stack(outs).view(torch.int16).clone()
            if ref is None:
                ref = cur
            elif not torch.equal(ref, cur):
                ok = False
                print(f"M={M} {kind}/{sname}: output differs")
            tp = (f"{a.trace_dir}/M{M}_{kind}_{sname}.json" if a.trace_dir else None)
            row = {"M": M, "prelude": kind, "aux": sname,
                   "us_per_layer": round(time_graph(g, a.iters) / a.layers, 2)}
            try:
                row["kernels"] = profile_split(g, max(10, a.iters // 5), tp)
            except Exception as exc:  # noqa: BLE001
                row["kernels"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(row)
            rows.append(row)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"test": "shared-contention", "ok": ok, "rows": rows}) + "\n")
    print("outputs identical across variants" if ok else "FAIL: outputs differ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
