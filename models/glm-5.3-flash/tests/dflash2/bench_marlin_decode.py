# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark for patch 0006: one GLM-5.3-Flash routed-MoE call,
fused_marlin_moe (incumbent) vs the compiled sm_80 decode, per token count.

    VLLM_GLM5_MARLIN_DECODE_LIB=/work/out/_ampere_marlin_C.abi3.so \
        python3 tests/bench_marlin_decode.py [--tokens 1,2,4,5,6,7,8,16] \
        [--N 2048] [--layers 8] [--iters 50] [--json out.jsonl]

Each path is captured in one CUDA graph that runs ``--layers`` consecutive MoE
calls with *different* routing (like consecutive decoder layers; the weights
touched per call are far larger than L2), and the graph is replayed
``--iters`` times. Reported: microseconds per MoE call (median of 5 timed
batches), whole call = alignment + 2 GEMMs + activation + slot sum, so the
numbers compare directly with what one layer costs inside a decode graph.
"exp" is the mean number of distinct experts per call and "TB/s" the
effective weight bandwidth (distinct expert bytes / time). Also printed: the number of marlin_moe_wna16 GEMMs it replaces (2 per call)
and the expected saving per PP4 step for the 45-layer model (×MoE layers).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marlin_decode_common import build_layer, routing, run_compiled, run_marlin  # noqa: E402


def time_graph(fn, layers_inputs, iters):
    outs = [torch.empty_like(x) for x, _, _ in layers_inputs]
    for (x, ids, w), o in zip(layers_inputs, outs):  # eager warm-up
        o.copy_(fn(x, ids, w))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.graph(g, stream=s):
        for (x, ids, w), o in zip(layers_inputs, outs):
            o.copy_(fn(x, ids, w))
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) * 1e3 / (iters * len(layers_inputs)))
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,2,3,4,5,6,7,8,16")
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--E", type=int, default=288)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--variants", default="orig,exact")
    ap.add_argument("--pool", type=int, default=0,
                    help="draw each call's experts from a random pool of this "
                         "size (0 = independent tokens, worst case)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from vllm.models.glm5next.nvidia.ops.marlin_decode import require_extension

    require_extension()
    dev = torch.cuda.get_device_name()
    print(f"device: {dev}, torch {torch.__version__}, E={args.E} K=4096 N={args.N}")
    layer = build_layer(E=args.E, N=args.N)
    rows = []
    hdr = f"{'M':>3} {'exp':>5} {'marlin us':>10} {'TB/s':>6}" + "".join(
        f" {v + ' us':>10} {'TB/s':>6} {'speedup':>8}" for v in args.variants.split(","))
    print(hdr)
    for M in [int(t) for t in args.tokens.split(",")]:
        inputs = [routing(M, layer, 1000 * M + i, pool=args.pool)
                  for i in range(args.layers)]
        # bytes of distinct expert weights+scales one call must stream
        per_expert = (layer.w1[0].numel() + layer.w2[0].numel()) * 4 + (
            layer.w1_scale[0].numel() + layer.w2_scale[0].numel()) * 2
        distinct = statistics.mean(len(set(i.flatten().tolist())) for _, i, _ in inputs)
        mbytes = distinct * per_expert / 1e6
        t_ref = time_graph(lambda x, i, w: run_marlin(layer, x, i, w), inputs, args.iters)
        line = f"{M:>3} {distinct:>5.1f} {t_ref:>10.1f} {mbytes / t_ref / 1e3:>6.2f}"
        row = {"M": M, "N": args.N, "E": args.E, "pool": args.pool,
               "distinct_experts": distinct, "marlin_us": t_ref, "device": dev}
        for v in args.variants.split(","):
            t = time_graph(lambda x, i, w, v=v: run_compiled(layer, x, i, w, v),
                           inputs, args.iters)
            line += f" {t:>10.1f} {mbytes / t / 1e3:>6.2f} {t_ref / t:>7.2f}x"
            row[f"{v}_us"] = t
        print(line)
        rows.append(row)
    print("\nPP4 step estimate: saving per step ~= (marlin_us - orig_us) x "
          "(MoE layers in the model, summed over the 4 ranks), for M = k+1.")
    if args.json:
        with args.json.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
