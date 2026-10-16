"""VLLM_DSV4_DENSE_FP8_DECODE: small-M dense FP8 (128x128 e8m0 blocks) vs Marlin.

Run inside the image on one sm_80 GPU:
    python3 test_dense_fp8_decode_gpu.py [--bench]

Checks per V4-Flash dense shape and M in 1..64: error against an fp64 reference
at Marlin's level, bitwise determinism, CUDA-graph replay == eager, the load-time
fallback for non-power-of-two scales; --bench: cold-L2 graph timing vs Marlin.
"""
import argparse
import os
import time

os.environ["VLLM_DSV4_DENSE_FP8_DECODE"] = "1"
import torch  # noqa: E402

import vllm.model_executor.layers.fused_moe  # noqa: E402,F401  (import order)
from vllm.model_executor.kernels.linear.mxfp8 import dsv4_dense_fp8 as D  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (  # noqa: E402
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)

ap = argparse.ArgumentParser()
ap.add_argument("--bench", action="store_true")
args = ap.parse_args()
dev = torch.device("cuda")
SHAPES = [(4096, 1536, "fused_wqa_wkv"), (1024, 32768, "wq_b"), (8192, 4096, "wo_b"),
          (4096, 4096, "shared_w13"), (2048, 4096, "shared_w2"), (1024, 8192, "indexer_wq_b")]
fails = []


def make(K, N, seed, pow2=True):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (N, K), device=dev, dtype=torch.uint8, generator=g)
    w[(w & 127) == 127] = 60
    e = torch.randint(-12, -4, (N // 128, K // 128), device=dev, generator=g).float()
    s = torch.exp2(e) if pow2 else torch.exp2(e) * 1.25
    ref = (w.view(torch.float8_e4m3fn).double()
           * s.double().repeat_interleave(128, 0).repeat_interleave(128, 1)).T.contiguous()
    L = torch.nn.Module()
    L.weight = torch.nn.Parameter(w.view(torch.float8_e4m3fn), requires_grad=False)
    L.weight_scale_inv = torch.nn.Parameter(s.clone(), requires_grad=False)
    L.weight_block_size = [128, 128]
    L.input_size_per_partition, L.output_size_per_partition = K, N
    L.orig_dtype = torch.bfloat16
    L.input_scale = None
    prepare_fp8_layer_for_marlin(L, size_k_first=False)
    D.prepare(L, s, [128, 128])
    return L, ref


def marlin(x, L):
    return apply_fp8_marlin_linear(x, L.weight, L.weight_scale_inv, L.workspace,
                                   L.output_size_per_partition, L.input_size_per_partition, None)


def ours(x, L):
    o = D.try_apply(x, L, None)
    return o if o is not None else marlin(x, L)


MS = list(range(1, 25)) + [28, 32, 33, 40, 47, 48, 56, 64]
for K, N, name in SHAPES:
    L, ref = make(K, N, K + N)
    if L._dsv4_dense_fp8 is None:
        fails.append((name, "not prepared"))
        continue
    worst = 0.0
    for M in MS:
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        r = x.double() @ ref
        sc = r.abs().mean().item()
        o, m = ours(x, L), marlin(x, L)
        eo = ((o.double() - r).abs().mean() / sc).item()
        em = ((m.double() - r).abs().mean() / sc).item()
        worst = max(worst, eo / em)
        if eo > 1.05 * em or not torch.equal(o, ours(x, L)) or torch.isnan(o).any():
            fails.append((name, M, eo, em))
    for M in (4, 20, 40):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            out.copy_(ours(x, L))
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out.copy_(ours(x, L))
        for _ in range(3):
            x.copy_(torch.randn(M, K, device=dev).to(torch.bfloat16))
            g.replay()
            torch.cuda.synchronize()
            if not torch.equal(out, ours(x, L)):
                fails.append((name, "graph", M))
    print(f"{name:14s} K={K:5d} N={N:5d} worst mean-error ratio vs Marlin {worst:.5f}")
    L2, _ = make(K, N, 7, pow2=False)
    if L2._dsv4_dense_fp8 is not None:
        fails.append((name, "non-pow2 scales accepted"))

if args.bench:
    def timed(fns, it=10):
        g = torch.cuda.CUDAGraph()
        for f in fns:
            f()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for f in fns:
                f()
        g.replay()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(it):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / (it * len(fns)) * 1e6
    for M in (1, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64):
        tm = to = 0.0
        for K, N, name in SHAPES:
            Ls = [make(K, N, i)[0] for i in range(max(2, int(160e6 // (N * K))))]
            x = torch.randn(M, K, device=dev).to(torch.bfloat16)
            tm += timed([lambda L=L: marlin(x, L) for L in Ls])
            to += timed([lambda L=L: ours(x, L) for L in Ls])
            del Ls
        print(f"M={M:3d} layer dense marlin {tm:7.1f} us  ours {to:7.1f} us  x{tm / to:.3f}")
print("FAILURES:", fails if fails else "none")
