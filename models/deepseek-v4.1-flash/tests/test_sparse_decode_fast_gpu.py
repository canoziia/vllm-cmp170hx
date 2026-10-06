"""Patch 0005: sm80 sparse-attention decode with VLLM_DSV4_SPARSE_DECODE_FAST
must be bitwise identical to the switch off; reports time per call.

    python3 test_sparse_decode_fast_gpu.py
"""
import os
import sys
import time

import torch

import vllm.model_executor.layers.fused_moe  # noqa: F401  (import order)
import vllm.v1.attention.ops.rocm_aiter_mla_sparse as ORIG

VAR = ORIG

dev = torch.device("cuda")
torch.manual_seed(0)
H, NOPE, ROPE = 64, 448, 64
BS = 64


def make_cache(nblocks, nan_rate=0.0):
    c = torch.randint(0, 256, (nblocks, BS, 584), dtype=torch.uint8, device=dev)
    flat = c.view(nblocks, -1)
    data = flat[:, : BS * 576].view(nblocks, BS, 576)
    # rope part: valid bf16 values
    rope = (torch.randn(nblocks, BS, ROPE, device=dev) * 0.5).to(torch.bfloat16)
    data[:, :, NOPE:].copy_(rope.view(torch.uint8).view(nblocks, BS, ROPE * 2))
    nope = data[:, :, :NOPE]
    if nan_rate == 0.0:
        nope[(nope & 0x7F) == 0x7F] = 0x3C
    sc = flat[:, BS * 576:].view(nblocks, BS, 8)
    sc.copy_(torch.randint(110, 135, sc.shape, dtype=torch.uint8, device=dev))
    sc[0, 0, 0] = 0  # exercise scale byte 0
    return c


def run(mod, q, main, mi, mp, extra, ei, ep, sink):
    return mod._rocm_sparse_attn_decode_ragged_triton(
        q, main, mi, mp, 0.0441941738, sink, NOPE, ROPE, extra, ei, ep)


def timed(fn):
    for _ in range(3):
        fn()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 200 * 1e6


fails = []
for (nq, main_len, extra_len, nan) in ((6, 128, 512, 0.0), (6, 128, 512, 1.0), (48, 128, 512, 0.0), (48, 128, 512, 1.0), (16, 128, 512, 0.0), (17, 128, 512, 0.0),
                                        (1, 128, 512, 0.0), (6, 100, 300, 0.0)):
    main = make_cache(64, nan)
    extra = make_cache(256, nan)
    q = (torch.randn(nq, H, NOPE + ROPE, device=dev) * 0.3).to(torch.bfloat16)
    mi = torch.randint(0, 64 * BS, (nq * main_len,), dtype=torch.int32, device=dev)
    ei = torch.randint(0, 256 * BS, (nq * extra_len,), dtype=torch.int32, device=dev)
    ei[::17] = -1  # padded top-k entries
    mp = torch.arange(0, nq + 1, dtype=torch.int32, device=dev) * main_len
    ep = torch.arange(0, nq + 1, dtype=torch.int32, device=dev) * extra_len
    sink = torch.randn(H, device=dev)
    args = (q, main, mi, mp, extra, ei, ep, sink)
    os.environ.pop("VLLM_DSV4_SPARSE_DECODE_FAST", None)
    ref = run(ORIG, *args)
    t0 = timed(lambda: run(ORIG, *args))
    line = f"nq={nq:2d} main={main_len} extra={extra_len} nan={nan}: deployed {t0:6.1f} us |"
    for fast in ("1",):
        os.environ["VLLM_DSV4_SPARSE_DECODE_FAST"] = fast
        out = run(VAR, *args)
        same = torch.equal(out.view(torch.int16), ref.view(torch.int16))
        if not same:
            fails.append(f"nq={nq} nan={nan}: not bitwise")
        t = timed(lambda: run(VAR, *args))
        line += f" fast:{t:6.1f}{'=' if same else '!'}"
        os.environ.pop("VLLM_DSV4_SPARSE_DECODE_FAST")
        off = run(VAR, *args)
        if not torch.equal(off.view(torch.int16), ref.view(torch.int16)):
            fails.append(f"nq={nq}: switch off not bitwise")
    print(line)
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
