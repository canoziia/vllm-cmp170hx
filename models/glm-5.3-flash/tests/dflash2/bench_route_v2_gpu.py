"""0013 route v2 microbenchmark (sm_80, patched vllm installed).

    python3 tests/bench_route_v2_gpu.py [--layers 11] [--iters 200]

Each pipeline is captured in ONE CUDA graph that runs `--layers` MoE routers
back to back, each with its own [288, 4096] gate weight and bias (so the
gate weights rotate through L2 the way a PP rank's layers do).  Timed with
CUDA events over graph replays; reported per layer in microseconds.

  incumbent_00010  gate (bf16_gemv / cuBLAS) + fused_grouped_topk + 00010 _align
  incumbent_cuda   gate + fused_grouped_topk + moe_align_block_size
  v2_gate          gate + route v2 (top-k + alignment, one launch)
  v2_tc            route v2 with the tensor-core GEMV (one launch total)

This is a local kernel-sequence measurement, not end-to-end ITL; the
counting/c8 A/B in the real server is what decides adoption.
"""
import argparse
import json
import os

os.environ["VLLM_GLM5_ROUTER_ALIGN_DECODE"] = "1"

import torch  # noqa: E402

from vllm.model_executor.kernels.linear.gemv_triton import (  # noqa: E402
    bf16_gemv, should_use_triton_gemv)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (  # noqa: E402
    fused_grouped_topk)
from vllm.models.glm5next.nvidia.ops import route_v2_decode as rv  # noqa: E402
from vllm.models.glm5next.nvidia.ops.router_align_decode import maybe_align  # noqa: E402

E, K, RSF = 288, 4096, 2.5


def gate(x, w):
    if should_use_triton_gemv(x, w):
        return bf16_gemv(x, w, torch.float32)
    return torch.mm(x, w.T, out_dtype=torch.float32)


def pipelines(x, layers):
    def inc_00010():
        for w, b in layers:
            lg = gate(x, w)
            tw, ti = fused_grouped_topk(x, lg, 8, True, b, 1, 1, "sigmoid", RSF)
            a = maybe_align(ti, E)
            if a is None:
                moe_align_block_size(ti, 8, E, None, ignore_invalid_experts=True)

    def inc_cuda():
        for w, b in layers:
            lg = gate(x, w)
            tw, ti = fused_grouped_topk(x, lg, 8, True, b, 1, 1, "sigmoid", RSF)
            moe_align_block_size(ti, 8, E, None, ignore_invalid_experts=True)

    def v2_gate():
        for w, b in layers:
            rv.route_v2(b, logits=gate(x, w), renormalize=True,
                        routed_scaling_factor=RSF)

    def v2_tc():
        for w, b in layers:
            rv.route_v2(b, x=x, weight=w, renormalize=True,
                        routed_scaling_factor=RSF)

    return dict(incumbent_00010=inc_00010, incumbent_cuda=inc_cuda,
                v2_gate=v2_gate, v2_tc=v2_tc)


def time_graph(fn, iters):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(5):
        st.record()
        for _ in range(iters):
            g.replay()
        en.record()
        en.synchronize()
        samples.append(st.elapsed_time(en) * 1000 / iters)
    return sorted(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=11)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ms", default="1,4,8")
    a = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)
    layers = [((torch.randn(E, K, generator=gen) * 0.02).to(torch.bfloat16).to(dev),
               (torch.randn(E, generator=gen) * 0.05).float().to(dev))
              for _ in range(a.layers)]
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), layers=a.layers)))
    for M in (int(m) for m in a.ms.split(",")):
        x = torch.randn(M, K, generator=gen).to(torch.bfloat16).to(dev)
        for name, fn in pipelines(x, layers).items():
            s = time_graph(fn, a.iters)
            print(json.dumps(dict(M=M, pipeline=name,
                                  us_per_layer_median=round(s[2] / a.layers, 2),
                                  us_per_layer_min=round(s[0] / a.layers, 2))),
                  flush=True)


if __name__ == "__main__":
    main()
