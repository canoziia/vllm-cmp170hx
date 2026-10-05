#!/usr/bin/env python
"""Patch 0028 (dev 0026b): prenorm GEMM micro-bench + bitwise check, one GPU.

Run in the GLM DFlash2 image (or with a patched tree on PYTHONPATH; patches
0025 = dev 0026 and 0028 = dev 0026b). The reference files are arguments
(a Morrowmake/vllm-cmp170hx checkout, ampere-glm53 @ 3a2bf16dae):
  python3 bench_prenorm_bf16x3_gpu.py \
      --mm-file <MM>/vllm/ampere_prefill/mhc_prenorm.py \
      [--mm-tl-file <MM>/vllm/model_executor/kernels/mhc/tilelang_kernels.py]

Variants (all on the same x / fn):
  old0026  : our TileLang dispatch (T<128 (1024,4); T<1024 (512,12); else block_m=2)
  new0026b : our _tilelang_hc_prenorm_gemm with both env switches on (full route)
  mm       : MM bf16x3 Triton kernel, loaded by FILE PATH from mm-vllm
  mm_tl    : MM's TileLang dispatch (what MM runs below 384 tokens), file path
  cublas   : our route with 0026 off (bf16 cuBLAS)
Bitwise: new0026b == mm for T>=384, new0026b == old0026 == mm_tl for T<384.
"""
import argparse
import importlib.util
import os
import sys

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bench(fn, iters=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm-file", required=True)
    ap.add_argument("--mm-tl-file", default=None)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--hc-mult", type=int, default=4)
    ap.add_argument("--tokens", default="2312,4096,1024,512,32")
    ap.add_argument("--layers", type=int, default=8,
                    help="distinct fn tensors cycled (catches pack-cache bugs)")
    a = ap.parse_args()

    mm = load(a.mm_file, "mm_prenorm_isolated")
    from vllm.model_executor.kernels.mhc import tilelang as ours
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        hc_prenorm_gemm_block_m_tilelang as bm_tl,
        hc_prenorm_gemm_tilelang as tl1,
    )
    mm_tl = None
    if a.mm_tl_file:
        try:  # MM's file imports MM-only vllm helpers; may fail in our env
            mm_tl = load(a.mm_tl_file, "mm_tl_isolated")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] mm TileLang file not loadable in this env: {e!r}")

    H, hc = a.hidden, a.hc_mult
    K, N = H * hc, hc * (hc + 2)
    dev = "cuda"
    torch.manual_seed(0)
    fns = [torch.randn(N, K, device=dev) * 0.02 for _ in range(a.layers)]

    def run_old(x, fn, out, sq):
        T = x.shape[0]
        if T >= 1024:
            bm_tl(x, fn, out, sq, H, hc, N, 512, 12, 2)
        elif T < 128 and K % 1024 == 0:
            tl1(x, fn, out, sq, H, hc, N, 1024, 4, 1)
        else:
            tl1(x, fn, out, sq, H, hc, N, 512, 12, 1)

    def run_route(env26, env26b):
        def f(x, fn, out, sq):
            os.environ["VLLM_GLM5_TARGET_PRENORM_FP32_0026"] = env26
            os.environ["VLLM_GLM5_TARGET_PRENORM_FP32_0026B_MIN_TOKENS"] = env26b
            ours._tilelang_hc_prenorm_gemm(x, fn, out, sq, H, hc)
        return f

    def run_mm(x, fn, out, sq):
        mm.hc_prenorm_gemm(x, fn, out=out, sqrsum=sq)

    def run_mm_tl(x, fn, out, sq):
        mm_tl._HC_PRENORM_GEMM_TILELANG_KERNEL(x, fn, out, sq, H, hc)

    variants = {
        "old0026": run_old,
        "new0026b": run_route("1", "384"),
        "mm": run_mm,
        "cublas": run_route("0", "0"),
    }
    if mm_tl is not None:
        variants["mm_tl"] = run_mm_tl

    ok_all = True
    for T in [int(t) for t in a.tokens.split(",")]:
        x = (torch.randn(T, K, device=dev) * 3).to(torch.bfloat16)
        res = {}
        for name, f in variants.items():
            outs = []
            for fn in fns:
                out = torch.empty(1, T, N, device=dev)
                sq = torch.empty(1, T, device=dev)
                f(x, fn, out, sq)
                outs.append((out.clone(), sq.clone()))
            res[name] = outs
        ref = [(x.double() @ fn.double().t(), x.double().square().sum(-1)) for fn in fns]

        def same(p, q):
            return all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                       for a, b in zip(res[p], res[q]))

        def err(p):
            return max((o[0][0].double() - r[0]).abs().max().item()
                       for o, r in zip(res[p], ref))

        want = "mm" if T >= 384 else ("mm_tl" if mm_tl else "old0026")
        bit = same("new0026b", want)
        if T < 384:
            bit = bit and same("new0026b", "old0026")
        ok_all &= bit
        line = [f"T={T:5d} bitwise new0026b=={want}: {bit}"]
        for name, f in variants.items():
            out = torch.empty(1, T, N, device=dev)
            sq = torch.empty(1, T, device=dev)
            i = [0]

            def call():
                f(x, fns[i[0] % len(fns)], out, sq)
                i[0] += 1
            line.append(f"{name}={bench(call):8.1f}us err={err(name):.2e}")
        print("  ".join(line))
    os.environ.pop("VLLM_GLM5_TARGET_PRENORM_FP32_0026B_MIN_TOKENS", None)
    print("ALL BITWISE OK" if ok_all else "BITWISE MISMATCH")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
