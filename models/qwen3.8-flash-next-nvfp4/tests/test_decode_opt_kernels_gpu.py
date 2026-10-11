#!/usr/bin/env python3
"""GPU checks for the Qwen3.8 decode optimisation kernels (run inside the image).

  python3 /opt/qwen-cache-tests/test_decode_opt_kernels_gpu.py [--bench]

1. VLLM_QWEN38_FAST_MOE_ALIGN: the single-kernel align is bitwise equal to the
   torch deterministic path (512 experts, Marlin decode block 8, int32/int64 ids,
   invalid ids, 1..4096 routed entries) and CUDA-graph capturable.
2. VLLM_QWEN38_THIN_GEMM: every table cell runs, is run-to-run deterministic and
   is no further from an FP64 reference than cuBLAS (max abs error).
3. VLLM_QWEN38_DRAFT_HEAD_FP8: the Marlin W8A16 draft head equals its own
   reference dequantisation (fp8 * row scale) within BF16 GEMM error.
"""
import sys

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.fused_moe  # noqa: F401  (import order for marlin utils)
from vllm.model_executor.layers.fused_moe import fast_moe_align as A
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    _moe_align_block_size_deterministic,
)
from vllm.models.qwen4_exp.nvidia.mtp_fp8_head import Fp8DraftHead, quantize_rows_fp8
from vllm.models.qwen4_exp.nvidia.ops import thin_gemm as T
from vllm.triton_utils import triton

BENCH = "--bench" in sys.argv
dev = "cuda"
failures = []


def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def graph_time_us(fn, n=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1000


# ---------------------------------------------------------------- 1. align
E, TOPK, BS = 512, 10, 8
gen = torch.Generator(device=dev).manual_seed(0)


def align_bufs(s):
    cap = s + E * (BS - 1)
    if s < E:
        cap = min(s * BS, cap)
    return (
        torch.empty(cap, dtype=torch.int32, device=dev),
        torch.empty(triton.cdiv(cap, BS), dtype=torch.int32, device=dev),
        torch.empty(1, dtype=torch.int32, device=dev),
    )


def routed(m, dtype):
    sc = torch.rand(m, E, generator=gen, device=dev) ** 3
    return torch.topk(sc, TOPK, dim=1).indices.to(dtype).contiguous()


ok = True
for m in (1, 2, 3, 4, 8, 16, 32, 51, 64, 128, 256, 409):
    for dtype in (torch.int32, torch.int64):
        ids = routed(m, dtype)
        if m % 3 == 0:
            ids.view(-1)[0] = -1
            ids.view(-1)[-1] = E + 7
        ref, new = align_bufs(ids.numel()), align_bufs(ids.numel())
        _moe_align_block_size_deterministic(ids, E, BS, *ref, None)
        used = A.try_align(ids, E, BS, *new, None)
        ok &= used and all(torch.equal(x, y) for x, y in zip(ref, new))
check(ok, "fast MoE align bitwise == torch deterministic align (1..4090 entries)")
check(not A.try_align(routed(410, torch.int32), E, BS, *align_bufs(4100), None),
      "fast MoE align declines > MAX_ENTRIES (prefill keeps torch path)")
ids = routed(32, torch.int32)
out = align_bufs(ids.numel())
t_new = graph_time_us(lambda: A.try_align(ids, E, BS, *out, None))
check(True, f"fast MoE align graph-capturable ({t_new:.1f} us at 320 entries)")
if BENCH:
    for m in (4, 32, 128):
        ids = routed(m, torch.int32)
        out = align_bufs(ids.numel())
        t0 = graph_time_us(lambda: _moe_align_block_size_deterministic(ids, E, BS, *out, None))
        t1 = graph_time_us(lambda: A.try_align(ids, E, BS, *out, None))
        print(f"  align entries={ids.numel()}: torch {t0:.1f} us  fast {t1:.1f} us")

# ---------------------------------------------------------------- 2. thin GEMM
torch.manual_seed(0)
ok_err = ok_det = True
for (n, k), cells in sorted(T._QWEN_TABLE.items()):
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
    for m in sorted(cells):
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        ref = x.double() @ w.double().t()
        y1 = T.thin_gemm(x, w)
        y2 = T.thin_gemm(x, w)
        yc = F.linear(x, w)
        e_t = (y1.double() - ref).abs().max().item()
        e_c = (yc.double() - ref).abs().max().item()
        # one BF16 rounding of the output dominates both; allow 1 ulp slack
        ulp = ref.abs().max().item() * 2.0**-7
        ok_err &= e_t <= e_c + ulp
        ok_det &= torch.equal(y1, y2)
        if BENCH:
            tt = graph_time_us(lambda: T.thin_gemm(x, w))
            tc = graph_time_us(lambda: F.linear(x, w))
            print(f"  thin N={n} K={k} M={m}: cuBLAS {tc:.1f} us thin {tt:.1f} us")
check(ok_err, "thin GEMM error vs FP64 <= cuBLAS error (+1 output ulp) on every table cell")
check(ok_det, "thin GEMM run-to-run bitwise deterministic")
check(T.table_config(33, 512, 2560) is None and T.table_config(4, 7, 2560) is None,
      "thin GEMM keeps cuBLAS above the last bucket and for unmeasured shapes")

# ---------------------------------------------------------------- 3. fp8 draft head
w = torch.randn(4096, 2560, device=dev, dtype=torch.bfloat16) * 0.02
head = Fp8DraftHead(w)
q, sc = quantize_rows_fp8(w)
deq = (q.float() * sc.float()[:, None]).to(torch.bfloat16)
for m in (1, 4, 8, 32):
    x = torch.randn(m, 2560, device=dev, dtype=torch.bfloat16)
    y = head(x).float()
    r = F.linear(x, deq).float()
    rel = ((y - r).norm() / r.norm()).item()
    check(rel < 1e-2, f"fp8 draft head M={m} matches fp8 reference dequant (rel {rel:.2e})")
check(abs(head.nbytes() - 4096 * 2560) < 4096 * 8, "fp8 draft head holds N*K bytes (+scales)")

print("ALL PASS" if not failures else f"{len(failures)} FAILURES")
sys.exit(1 if failures else 0)
