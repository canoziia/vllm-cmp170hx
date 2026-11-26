# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""stream-K/split-K/wide split-K dense MXFP8 decode GEMMs (Marlin 8-bit layout, e8m0 group-32), M <= 192.

Default OFF.  Enable with VLLM_DSV41_DENSE_MXFP8_DECODE=1 (or VLLM_DSV41_DENSE_MXFP8=1;
and VLLM_DSV41_DENSE_MXFP8_LIB=/path/_dsv41_dense_mxfp8.so if the op is not in an
already-loaded extension).

Dispatch (tables in dispatch_table.py, generated from the final per-M cold-L2
A/B scan on one GPU by gen_dispatch.py):
  * M <= 8      : stream-K  (warp stream-K, per-warp activations)  -- unchanged since v1
  * 9 .. 64     : split-K  (CTA-shared activations, static split-K + fixed-order reduce)
  * 65 .. 192   : wide split-K  (two warps per 64-col tile -> one dequantisation serves up
                       to 128 tokens; balanced row chunks; split-K + reduce)
  Our kernel is used only for M inside RANGES[(K, N)] -- intervals whose scanned
  end points were both >= 1.03x Marlin; every other M (and M > 192) uses Marlin.
Common preconditions: bf16 input (row stride % 8 == 0, 16 B aligned), no bias,
a tuned (K, N), and the load-time check (all e8m0 scale bytes in [1, 134] ->
folded scale 2^(S-7) is a normal bf16, dequantised weights bit-identical to
Marlin's).

CUDA-graph & multi-stream safety: scratch is allocated per `(device.index, cuda_stream)`
OUTSIDE capture (in `prepare`, including `aux_stream()` so SharedExperts on `aux_stream()`
never shares scratch with concurrent default-stream kernels).  stream-K's scratch is
sentinel-filled and re-armed by the kernel; the split-K/wide split-K partial buffer is
fully rewritten every call.  Deterministic: all reductions run in a fixed order.
"""

import os

import torch

try:
    from . import dispatch_table as _T
except ImportError:  # standalone use
    import dispatch_table as _T  # type: ignore[no-redef]

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_LIBS = (
    os.path.join(_HERE, "..", "..", "..", "..", "_dsv41_dense_mxfp8.so"),
    os.path.join(_HERE, "..", "..", "..", "..", "..", "_dsv41_dense_mxfp8.so"),
    os.path.join(_HERE, "_dsv41_dense_mxfp8.so"),
    "/workspace/_dsv41_dense_mxfp8.so",
)
_loaded = False

# ---- M <= 8: stream-K (K, N) -> (cfg, ctas)
TUNED = {
    (5120, 1792): (3, 108),   # qkv_a (replicated across TP; tuned for 108-SM GA100)
    (1280, 32768): (3, 64),   # q_b
    (1280, 4096): (3, 64),    # indexer.wq_b (ShadowSource / unfused)
    (1280, 36864): (3, 72),   # fused wq_b + indexer.wq_b
    (8192, 5120): (3, 70),    # o
    (5120, 4608): (3, 72),    # shared w13
    (2304, 5120): (3, 70),    # shared w2
    # TP=4 shapes (tuned for 108-SM bottleneck rank GPU 3)
    (1280, 8192): (3, 96),    # q_b (TP=4)
    (1280, 12288): (3, 96),   # fused wq_b + indexer.wq_b (TP=4; streamk_split takes cfg 3 only)
    (2048, 5120): (3, 108),   # o (TP=4)
    (5120, 1152): (2, 72),    # shared w13 (TP=4; N/64=18 requires W=2)
    (576, 5120): (2, 64),     # shared w2 (TP=4)
    # DSpark draft projections (replicated on TP=4)
    (15360, 5120): (1, 108),  # DSpark main_proj
    (5120, 1536): (3, 72),    # DSpark context_wkv_proj
    (6144, 6400): (1, 100),   # Engram wkv (TP=4)
}
# Shapes where static split-K beats stream-K even at M <= 8 (cold-L2 on 108-SM GA100):
SPLITK_M8 = {
    (5120, 1792): (21, 10),   # 12.24 vs 13.23 us (-1.0 us x 43 layers!)
    (15360, 5120): (22, 10),  # 47.13 vs 50.10 us
    (5120, 1536): (21, 10),   # 11.85 vs 12.28 us
}
# Round 8 no-barrier warp K-split (M <= 8, 16-layer cold-L2 CUDA Graph on 108-SM GA100):
#   qkv_a 15.26->10.11us, wo_b 13.75->9.05us, wq_b+idx 14.75->12.10us,
#   shared_w13 12.43->7.39us, shared_w2 7.05->4.80us (-0.63ms/step, 0 extra HBM!)
KSPLIT_NOBAR_M8 = {
    (5120, 1792): 0,   # qkv_a: W8 TP1 ST3 GPU1 S=2 (56 CTAs)
    (5120, 1536): 8,   # context_wkv_proj: W8 TP1 ST3 GPU1 S=5 (120 CTAs on 126 SMs)
    (2048, 5120): 5,   # wo_b: W8 TP1 ST3 GPU1 S=1 (80 CTAs, 0-part, -0.60us vs c1)
    (1280, 12288): 2,  # wq_b+idx: W4 TP2 ST3 GPU2 S=1 (96 CTAs, 0-part, supports SPLIT)
    (1280, 8192): 1,   # wq_b: W4 TP1 ST4 GPU1 S=1 (128 CTAs on 126 SMs, 0-part; M<=4 uses cfg=9)
    (5120, 1152): 7,   # shared_w13: W4 TP1 ST4 GPU1 S=2 (36 CTAs, low W13 contention)
    (576, 5120): 1,    # shared_w2: W4 TP1 ST4 GPU1 S=1 (80 CTAs, 0-part)
    (6144, 6400): 5,   # engram_wkv: W8 TP1 ST3 GPU1 S=1 (100 CTAs, 0-part, 1685 GB/s!)
    (15360, 5120): 5,  # main_proj: W8 TP1 ST3 GPU1 S=1 (80 CTAs, 0-part)
}
# Round 12 TB=2 no-barrier warp K-split (M = 9..16, cold-L2 A/B on 126-SM GA100):
#   qkv_a 16.3->11.6us, wo_b 15.5->11.2us, wq_b+idx 19.5->14.7us,
#   wq_b 16.8->13.3us, shared_w2 9.9->6.6us, shared_w13 15.6->12.6us (-0.95ms/step!)
KSPLIT_NOBAR_M16 = {
    (5120, 1792): 14,   # qkv_a: W8 TP1 ST3 GPU1 S=4 TB=2 (112 CTAs)
    (5120, 1536): 15,   # context_wkv_proj: W8 TP1 ST3 GPU1 S=5 TB=2 (120 CTAs)
    (2048, 5120): 13,   # wo_b: W8 TP1 ST3 GPU1 S=1 TB=2 (80 CTAs, 0-part)
    (1280, 12288): 12,  # wq_b+idx: W4 TP2 ST3 GPU2 S=1 TB=2 (96 CTAs, 0-part, supports SPLIT)
    (1280, 8192): 17,   # wq_b: W4 TP1 ST3 GPU1 S=1 TB=2 (128 CTAs, 0-part)
    (576, 5120): 17,    # shared_w2: W4 TP1 ST3 GPU1 S=1 TB=2 (80 CTAs, 0-part)
    (6144, 6400): 13,   # engram_wkv: W8 TP1 ST3 GPU1 S=1 TB=2 (100 CTAs, 0-part)
    (15360, 5120): 13,  # main_proj: W8 TP1 ST3 GPU1 S=1 TB=2 (80 CTAs, 0-part)
}
MAX_WARPS = 128 * 2 * 4
_SCRATCH: dict[tuple[int | None, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_CAPTURE_ROOT_SCRATCH: dict[int | None, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def enabled() -> bool:
    if "VLLM_DSV41_DENSE_MXFP8_DECODE" in os.environ:
        return os.environ["VLLM_DSV41_DENSE_MXFP8_DECODE"] == "1"
    return os.environ.get("VLLM_DSV41_DENSE_MXFP8", "0") == "1"


def _ops():
    global _loaded
    if not _loaded:
        env_lib = os.environ.get("VLLM_DSV41_DENSE_MXFP8_LIB", "")
        candidates = (env_lib,) + _CANDIDATE_LIBS if env_lib else _CANDIDATE_LIBS
        for lib in candidates:
            if lib and os.path.exists(lib):
                torch.ops.load_library(lib)
                break
        _loaded = True
    return torch.ops._dsv41_dense_mxfp8


def layer_ok(weight_scale: torch.Tensor) -> bool:
    """Load-time check (host sync allowed here, never during capture)."""
    s = weight_scale.view(torch.uint8)
    return bool(((s >= 1) & (s <= 134)).all().item())


def _make_scratch_buffers(device: torch.device, stream: torch.cuda.Stream):
    with torch.cuda.stream(stream):
        part = torch.empty(MAX_WARPS * 32 * 16, dtype=torch.float32, device=device)
        part.view(torch.int32).fill_(-1)  # stream-K sentinel; re-armed by the kernel after use
        cnt = torch.zeros(1024, dtype=torch.int32, device=device)  # reserved
        part4 = torch.empty(max(_T.SCRATCH_FLOATS, 1), dtype=torch.float32, device=device)
    return (part, cnt, part4)


def _alloc_scratch_on_stream(device: torch.device, stream: torch.cuda.Stream):
    key = (device.index, stream.cuda_stream)
    t = _SCRATCH.get(key)
    if t is None:
        assert not torch.cuda.is_current_stream_capturing(), (
            "allocate dense_mxfp8 scratch before CUDA graph capture"
        )
        t = _SCRATCH[key] = _make_scratch_buffers(device, stream)
    return t


def scratch(device: torch.device):
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream)
    t = _SCRATCH.get(key)
    if t is not None:
        return t
    if torch.cuda.is_current_stream_capturing():
        # Unseen private capture stream (e.g. `with torch.cuda.graph(g):`);
        # aux_stream() already has its own disjoint entry in _SCRATCH.
        return _CAPTURE_ROOT_SCRATCH.get(device.index)
    return _alloc_scratch_on_stream(device, stream)


def select(M: int, K: int, N: int):
    """-> ("ksplit_nobar"|"streamk"|"splitk"|"splitk_wide", cfg, arg) or None (use Marlin)."""
    if not any(lo <= M <= hi for lo, hi in _T.RANGES.get((K, N), ())):
        return None
    if M <= 8:
        kn = KSPLIT_NOBAR_M8.get((K, N))
        if kn is not None and os.environ.get("VLLM_DSV41_DENSE_KSPLIT_NOBAR", "0") == "1":
            if K == 1280 and N == 8192 and M <= 4:
                kn = 9
            return ("ksplit_nobar", kn, 0)
        sp8 = SPLITK_M8.get((K, N))
        if sp8 is not None and os.environ.get("VLLM_DSV41_DENSE_SPLITK_M8", "1") == "1":
            return ("splitk",) + sp8
        e = TUNED.get((K, N))
        return ("streamk",) + e if e else None
    if M <= 16:
        kn16 = KSPLIT_NOBAR_M16.get((K, N))
        if kn16 is not None and os.environ.get("VLLM_DSV41_DENSE_KSPLIT_NOBAR", "0") == "1":
            return ("ksplit_nobar", kn16, 0)
    tab, kind = (_T.SPLITK_TIERS, "splitk") if M <= 64 else (_T.SPLITK_WIDE_TIERS, "splitk_wide")
    for t in sorted(tab):
        if M <= t:
            e = tab[t].get((K, N))
            if not e:
                return None
            return tuple(e) if len(e) == 3 else (kind,) + tuple(e)
    return None


def prepare(layer) -> None:
    """Call once after prepare_mxfp8_layer_for_marlin(layer)."""
    K, N = layer.input_size_per_partition, layer.output_size_per_partition
    try:
        from vllm.model_executor.layers.fused_moe import dsv4_mxfp4_decode

        dsv4_mxfp4_decode.warmup_device(layer.weight.device)
    except Exception:
        pass
    layer._dense_mxfp8_ok = enabled() and (K, N) in _T.RANGES and layer_ok(layer.weight_scale)
    if layer._dense_mxfp8_ok:
        dev = layer.weight.device
        cur = torch.cuda.current_stream(dev)
        _alloc_scratch_on_stream(dev, cur)
        if dev.index not in _CAPTURE_ROOT_SCRATCH:
            _CAPTURE_ROOT_SCRATCH[dev.index] = _make_scratch_buffers(dev, cur)
        # Pre-allocate on vLLM's global aux_stream() as well so SharedExperts
        # (MULTI_STREAM_OVERLAPPED) has its own race-free scratch before graph capture.
        try:
            from vllm.utils.torch_utils import aux_stream

            aux = aux_stream()
            if aux is not None and aux.device == dev:
                _alloc_scratch_on_stream(dev, aux)
        except Exception:
            pass


def try_apply(x: torch.Tensor, layer, size_n: int, size_k: int, bias):
    """Returns the output or None (caller falls back to Marlin)."""
    if not getattr(layer, "_dense_mxfp8_ok", False) or bias is not None:
        return None
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.size(0)
    if (
        M < 1
        or x2.dtype != torch.bfloat16
        or x2.stride(1) != 1
        or x2.stride(0) % 8
        or x2.data_ptr() % 16
    ):
        return None
    sel = select(M, size_k, size_n)
    if sel is None:
        return None
    kind, cfg, arg = sel
    scr = scratch(x2.device)
    if scr is None:
        return None
    part, cnt, part4 = scr
    if kind != "streamk" and arg > 1 and part4.numel() < arg * M * size_n:
        return None
    ops = _ops()
    if not hasattr(ops, "streamk"):
        return None
    out = torch.empty(M, size_n, dtype=torch.bfloat16, device=x2.device)
    w, s = layer.weight, layer.weight_scale.view(torch.uint8)
    if kind == "ksplit_nobar" and hasattr(ops, "ksplit_nobarrier"):
        ops.ksplit_nobarrier(x2, w, s, out, part, cfg)
    elif kind == "streamk":
        ops.streamk(x2, w, s, out, part, cnt, cfg, arg)
    elif kind == "splitk":
        ops.splitk(x2, w, s, out, part4, cfg, arg)
    else:
        ops.splitk_wide(x2, w, s, out, part4, cfg, arg)
    return out.reshape(*x.shape[:-1], size_n)




def try_apply_swiglu(x: torch.Tensor, layer, limit: float = 0.0) -> torch.Tensor | None:
    """Fused gate_up_proj + SiluAndMul(WithClamp) writing [..., N//2] directly."""
    if (
        not getattr(layer, "_dense_mxfp8_ok", False)
        or getattr(layer, "bias", None) is not None
        or getattr(layer, "gather_output", False)
    ):
        return None
    size_k = layer.input_size_per_partition
    size_n = layer.output_size_per_partition
    if size_n % 128 != 0:
        return None
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.size(0)
    if (
        M < 1
        or M > 192
        or x2.dtype != torch.bfloat16
        or x2.stride(1) != 1
        or x2.stride(0) % 8
        or x2.data_ptr() % 16
    ):
        return None
    ops = _ops()
    scr = scratch(x2.device)
    if scr is None:
        return None
    part, _cnt, part4 = scr
    w, s = layer.weight, layer.weight_scale.view(torch.uint8)
    half_n = size_n // 2
    if M <= 16:
        if size_k == 5120 and size_n == 1152 and hasattr(ops, "ksplit_nobarrier_swiglu"):
            out = torch.empty(M, half_n, dtype=torch.bfloat16, device=x2.device)
            cfg_sw = 0 if M <= 8 else 13  # 36 CTAs (low W13 contention)
            ops.ksplit_nobarrier_swiglu(x2, w, s, out, part, cfg_sw, float(limit))
            return out.reshape(*x.shape[:-1], half_n)
        if M <= 8:
            return None
    sel = select(M, size_k, size_n)
    if sel is None:
        return None
    kind, cfg, arg = sel
    if arg < 2 or part4.numel() < arg * M * size_n:
        return None
    if kind == "splitk" and hasattr(ops, "splitk_swiglu"):
        out = torch.empty(M, half_n, dtype=torch.bfloat16, device=x2.device)
        ops.splitk_swiglu(x2, w, s, out, part4, cfg, arg, float(limit))
        return out.reshape(*x.shape[:-1], half_n)
    if kind == "splitk_wide" and hasattr(ops, "splitk_wide_swiglu"):
        out = torch.empty(M, half_n, dtype=torch.bfloat16, device=x2.device)
        ops.splitk_wide_swiglu(x2, w, s, out, part4, cfg, arg, float(limit))
        return out.reshape(*x.shape[:-1], half_n)
    return None


def try_apply_split(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    size_n0: int,
    size_n1: int,
    size_k: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused GEMM writing two contiguous outputs ([..., size_n0], [..., size_n1]) directly."""
    size_n = size_n0 + size_n1
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.size(0)
    if (
        M < 1
        or x2.dtype != torch.bfloat16
        or x2.stride(1) != 1
        or x2.stride(0) % 8
        or x2.data_ptr() % 16
        or size_n0 % 64 != 0
        or size_n1 % 64 != 0
    ):
        return None
    sel = select(M, size_k, size_n)
    if sel is None:
        return None
    kind, cfg, arg = sel
    scr = scratch(x2.device)
    if scr is None:
        return None
    part, cnt, part4 = scr
    if kind != "streamk" and arg > 1 and part4.numel() < arg * M * size_n:
        return None
    ops = _ops()
    if not hasattr(ops, "streamk"):
        return None
    s = weight_scale.view(torch.uint8)
    lead = x.shape[:-1]
    if M == 1 or not hasattr(ops, "streamk_split"):
        out = torch.empty(M, size_n, dtype=torch.bfloat16, device=x2.device)
        if kind == "ksplit_nobar" and hasattr(ops, "ksplit_nobarrier"):
            ops.ksplit_nobarrier(x2, weight, s, out, part, cfg)
        elif kind == "streamk":
            ops.streamk(x2, weight, s, out, part, cnt, cfg, arg)
        elif kind == "splitk":
            ops.splitk(x2, weight, s, out, part4, cfg, arg)
        else:
            ops.splitk_wide(x2, weight, s, out, part4, cfg, arg)
        o0 = out[:, :size_n0].contiguous().reshape(*lead, size_n0)
        o1 = out[:, size_n0:].contiguous().reshape(*lead, size_n1)
        return o0, o1
    out0 = torch.empty(M, size_n0, dtype=torch.bfloat16, device=x2.device)
    out1 = torch.empty(M, size_n1, dtype=torch.bfloat16, device=x2.device)
    if kind == "ksplit_nobar" and hasattr(ops, "ksplit_nobarrier_split"):
        ops.ksplit_nobarrier_split(x2, weight, s, out0, out1, part, cfg)
    elif kind == "streamk":
        ops.streamk_split(x2, weight, s, out0, out1, part, cnt, cfg, arg)
    elif kind == "splitk":
        ops.splitk_split(x2, weight, s, out0, out1, part4, cfg, arg)
    else:
        ops.splitk_wide_split(x2, weight, s, out0, out1, part4, cfg, arg)
    return out0.reshape(*lead, size_n0), out1.reshape(*lead, size_n1)


# ---- Optional bitwise-identical MoE router fast path (VLLM_DSV41_FAST_GATE_TOPK=1, default off)
def fast_gate_topk_enabled() -> bool:
    return os.environ.get("VLLM_DSV41_FAST_GATE_TOPK", "0") == "1"


def try_gate_gemv(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype | None):
    """Bitwise-identical CUDA replacement for Triton bf16_gemv on M=1..6."""
    if not fast_gate_topk_enabled() or out_dtype != torch.float32:
        return None
    if x.ndim != 2 or weight.ndim != 2 or not x.is_cuda or not weight.is_cuda:
        return None
    M, K = x.shape
    E, Kw = weight.shape
    if (
        M < 1
        or M > 6
        or K < 8
        or K % 8 != 0
        or Kw != K
        or E < 1
        or x.device != weight.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.data_ptr() % 16 != 0
        or weight.data_ptr() % 16 != 0
    ):
        return None
    ops = _ops()
    if not hasattr(ops, "gate_gemv"):
        return None
    out = torch.empty(M, E, dtype=torch.float32, device=x.device)
    ops.gate_gemv(x, weight, out)
    return out


def try_dsv4_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    indices_dtype: torch.dtype,
    routed_scaling_factor: float,
    input_ids: torch.Tensor | None = None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
):
    """Bitwise-identical CUDA replacement for Triton dsv4_topk."""
    if not fast_gate_topk_enabled():
        return None
    if (
        gating_output.ndim != 2
        or not gating_output.is_cuda
        or not correction_bias.is_cuda
        or gating_output.device != correction_bias.device
        or gating_output.dtype != torch.float32
        or correction_bias.dtype != torch.float32
        or not gating_output.is_contiguous()
        or not correction_bias.is_contiguous()
        or indices_dtype not in (torch.int32, torch.uint32, torch.int64)
    ):
        return None
    M, E = gating_output.shape
    if (
        M < 1
        or E not in (256, 384)
        or correction_bias.shape != (E,)
        or gating_output.data_ptr() % 16 != 0
        or correction_bias.data_ptr() % 16 != 0
    ):
        return None
    if bias_vl is not None and image_sentinel_lo > 0:
        if (
            input_ids is None
            or not input_ids.is_cuda
            or not bias_vl.is_cuda
            or input_ids.device != gating_output.device
            or bias_vl.device != gating_output.device
            or input_ids.dtype not in (torch.int32, torch.int64)
            or not input_ids.is_contiguous()
            or input_ids.numel() < M
            or bias_vl.dtype != torch.float32
            or not bias_vl.is_contiguous()
            or bias_vl.shape != (E,)
            or bias_vl.data_ptr() % 16 != 0
        ):
            return None
    ops = _ops()
    if not hasattr(ops, "dsv4_topk"):
        return None
    topk_weights = gating_output.new_empty((M, 6), dtype=torch.float32)
    topk_ids = gating_output.new_empty((M, 6), dtype=indices_dtype)
    ops.dsv4_topk(
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        float(routed_scaling_factor),
        input_ids,
        bias_vl,
        int(image_sentinel_lo),
    )
    return topk_weights, topk_ids
