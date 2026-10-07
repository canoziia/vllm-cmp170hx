"""Shared helpers for O2 dense MXFP8 experiments (run inside the vLLM image)."""
import json
import statistics
import time

import torch

import vllm.model_executor.layers.fused_moe  # noqa: F401  (import order: avoid circular import)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (  # noqa: E402
    apply_mxfp8_marlin_linear, prepare_mxfp8_layer_for_marlin)

DEV = torch.device("cuda")
# (K, N, name) of one DeepSeek V4.1 layer, in engine call order
SHAPES = [(5120, 1792, "qkv_a"), (1280, 32768, "q_b"), (8192, 5120, "o"),
          (5120, 4608, "sh_w13"), (2304, 5120, "sh_w2")]


def nbytes(K, N):
    return K * N * (1 + 1 / 32)


def make(K, N, seed=1, ref=False, exp_lo=115, exp_hi=123):
    """Random finite e4m3 weights (incl. subnormals) + e8m0 group-32 scales,
    prepared exactly like the engine (prepare_mxfp8_layer_for_marlin)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randint(0, 256, (N, K), device=DEV, dtype=torch.uint8, generator=g)
    w[(w & 127) == 127] = 60  # no NaN encodings
    e = torch.randint(exp_lo, exp_hi, (N, K // 32), device=DEV, dtype=torch.int32, generator=g)
    r = None
    if ref:
        r = (w.view(torch.float8_e4m3fn).double() *
             torch.exp2(e.double() - 127).repeat_interleave(32, 1)).T.contiguous()  # [K,N] fp64
    L = torch.nn.Module()
    L.weight = torch.nn.Parameter(w.view(torch.float8_e4m3fn), requires_grad=False)
    L.weight_scale = torch.nn.Parameter(e.to(torch.uint8), requires_grad=False)
    L.input_size_per_partition = K
    L.output_size_per_partition = N
    prepare_mxfp8_layer_for_marlin(L)
    L.K, L.N = K, N
    return L, r


def clone(L):
    C = torch.nn.Module()
    for a in ("weight", "weight_scale", "workspace"):
        setattr(C, a, getattr(L, a).clone())
    C.K, C.N = L.K, L.N
    return C


def marlin(x, L):
    return apply_mxfp8_marlin_linear(x, L.weight, L.weight_scale, L.workspace, L.N, L.K, None)


def graph_of(fns, repeats=1):
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for fn in fns:
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repeats):
            for fn in fns:
                fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    return g


def time_graph(g, ncalls, replays=20, trials=7):
    """Median per-call time (us) over trials: events around `replays` back-to-back replays."""
    ts, walls = [], []
    for _ in range(trials):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        a.record()
        for _ in range(replays):
            g.replay()
        b.record()
        b.synchronize()
        walls.append((time.perf_counter() - t0) * 1e6 / (replays * ncalls))
        ts.append(a.elapsed_time(b) * 1e3 / (replays * ncalls))
    return dict(us=statistics.median(ts), min_us=min(ts), max_us=max(ts),
                wall_us=statistics.median(walls), samples=[round(t, 3) for t in ts])


def bench_fns(fns, repeats=1, replays=20, trials=7):
    g = graph_of(fns, repeats)
    r = time_graph(g, len(fns) * repeats, replays, trials)
    del g
    return r


def kernel_trace(g, replays=3, match=None):
    """CUPTI kernel durations (us) of a graph replay via torch.profiler."""
    import os
    import tempfile
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    p.export_chrome_trace(path)
    ev = json.load(open(path))["traceEvents"]
    os.unlink(path)
    k = [e for e in ev if e.get("cat") == "kernel" and (match is None or match in e["name"])]
    k.sort(key=lambda e: e["ts"])
    return [(e["name"][:60], e["dur"], e["ts"]) for e in k]


def emit(f=None, **kw):
    s = json.dumps(kw)
    print(s, flush=True)
    if f is not None:
        f.write(s + "\n")
        f.flush()
