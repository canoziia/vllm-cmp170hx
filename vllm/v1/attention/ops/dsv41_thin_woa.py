# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 grouped thin GEMM for DeepSeek-V4.1 wo_a (BF16 and register-dequant MXFP8).
No persistent scratch, no atomics; warm JIT specializations before capture.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def wo_kernel(
    X,
    W,
    Y,
    M: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX: tl.constexpr,
    SG: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    g = tl.program_id(1)
    m = tl.arange(0, BM)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(tl.cdiv(K, BK)):
        kk = i * BK + k
        a = tl.load(
            X + m[:, None] * SX + g * SG + kk[None, :],
            (m[:, None] < M) & (kk[None, :] < K),
            0,
        )
        b = tl.load(
            W + g * N * K + n[None, :] * K + kk[:, None],
            (n[None, :] < N) & (kk[:, None] < K),
            0,
        )
        # Short independent dot then FP32 add (not long-chain MMA).
        tmp = tl.dot(a, b)
        # Explicit asm blocks Triton's combine-dot-add pass, which otherwise
        # folds even `acc + dot(a,b)` into a long MMA accumulator chain.
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[acc, tmp],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(
        Y + (m[:, None] * G + g) * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


def wo(x, w, bn=32, bk=128, warps=4, stages=4, out=None):
    m, g, k = x.shape
    n = w.shape[1]
    if out is None:
        out = torch.empty((m, g, n), device=x.device, dtype=x.dtype)
    return wo_launch(x, w, out, bn, bk, warps, stages)


def wo_launch(x, w, out, bn, bk, warps, stages):
    m, g, k = x.shape
    n = w.shape[1]
    wo_kernel[(triton.cdiv(n, bn), g)](
        x,
        w,
        out,
        m,
        g,
        n,
        k,
        x.stride(0),
        x.stride(1),
        max(16, triton.next_power_of_2(m)),
        bn,
        bk,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
    return out


# No persistent allocation or cross-stream scratch; graph-private outputs only.
# An unwarmed specialization falls back during capture rather than JIT there.
_READY: set = set()
# Per-G (bn, bk, warps, stages, max_m) tuned on sm_80 CMP 170HX for TP=8/4/2/1.
_WOA_CFGS = {
    1: (32, 256, 4, 3, 48),
    2: (32, 256, 4, 3, 48),
    4: (64, 256, 4, 3, 32),
    8: (128, 128, 4, 3, 48),
    16: (128, 128, 4, 3, 48),
}


def thin_woa(x, w):
    if not (
        x.is_cuda
        and x.dtype == w.dtype == torch.bfloat16
        and x.ndim == w.ndim == 3
        and x.shape[1] in _WOA_CFGS
        and x.shape[2] == 4096
        and w.shape == (x.shape[1], 1024, 4096)
        and x.stride(-1) == 1
        and w.is_contiguous()
        and x.device == w.device
        and torch.cuda.get_device_capability(x.device) == (8, 0)
    ):
        return torch.einsum("tgd,grd->tgr", x, w)
    bn, bk, warps, stages, max_m = _WOA_CFGS[x.shape[1]]
    if not (1 <= x.shape[0] <= max_m):
        return torch.einsum("tgd,grd->tgr", x, w)
    key = (x.device.index, x.shape[0], x.shape[1], x.stride(0), x.stride(1))
    if key not in _READY and torch.cuda.is_current_stream_capturing():
        return torch.einsum("tgd,grd->tgr", x, w)
    out = wo(x, w, bn, bk, warps, stages)
    _READY.add(key)
    return out


_FP8_LUT_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_READY_FP8: set = set()


def _get_fp8_bf16_lut(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = device.index or 0
    luts = _FP8_LUT_CACHE.get(idx)
    if luts is None:
        raw = torch.arange(256, dtype=torch.uint8, device=device)
        fp8 = raw.view(torch.float8_e4m3fn)
        f32 = fp8.to(torch.float32)
        v_lut = torch.where(torch.isnan(f32), torch.zeros_like(f32), f32).contiguous()
        s_lut = torch.exp2(
            torch.arange(256, dtype=torch.float32, device=device) - 127.0
        ).contiguous()
        luts = (v_lut, s_lut)
        _FP8_LUT_CACHE[idx] = luts
    return luts


@triton.jit
def wo_mxfp8_kernel(
    X,
    W_u8,
    S_u8,
    V_LUT_f32,
    S_LUT_f32,
    Y,
    M,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX,
    SG,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    g = tl.program_id(1)
    m = tl.arange(0, BM)
    k = tl.arange(0, BK)
    KS: tl.constexpr = K // 32
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(tl.cdiv(K, BK)):
        kk = i * BK + k
        a = tl.load(
            X + m[:, None] * SX + g * SG + kk[None, :],
            (m[:, None] < M) & (kk[None, :] < K),
            0,
        )
        w_byte = tl.load(
            W_u8 + g * N * K + n[None, :] * K + kk[:, None],
            (n[None, :] < N) & (kk[:, None] < K),
            0,
        ).to(tl.int32)
        w_val = tl.load(V_LUT_f32 + w_byte)
        s_byte = tl.load(
            S_u8 + g * N * KS + n[None, :] * KS + (kk[:, None] // 32),
            (n[None, :] < N) & (kk[:, None] < K),
            127,
        ).to(tl.int32)
        scale_f32 = tl.load(S_LUT_f32 + s_byte)
        prod_f32 = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[w_val, scale_f32],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        b = tl.inline_asm_elementwise(
            "cvt.rn.bf16.f32 $0, $1;",
            constraints="=h,f",
            args=[prod_f32],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=1,
        )
        tmp = tl.dot(a, b)
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[acc, tmp],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(
        Y + (m[:, None] * G + g) * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


_WARMED_DEVS: set[tuple[int | None, int]] = set()
_CPP_WOA_SCRATCH: dict[tuple[int | None, int], tuple[torch.Tensor, torch.Tensor]] = {}
_CPP_WOA_CAPTURE: dict[int | None, tuple[torch.Tensor, torch.Tensor]] = {}
_WOA_MARLIN_SENT: dict[tuple[int | None, int], torch.Tensor] = {}
_WOA_MARLIN_CAPTURE: dict[int | None, torch.Tensor] = {}


def _alloc_woa_marlin_sent(device: torch.device, stream: torch.cuda.Stream, g: int) -> torch.Tensor:
    key = (device.index, stream.cuda_stream)
    need = max(g, 2) * 16 * 2 * 16 * 64
    buf = _WOA_MARLIN_SENT.get(key)
    if buf is None or buf.numel() < need:
        with torch.cuda.stream(stream):
            buf = torch.empty(need, dtype=torch.float32, device=device)
            buf.view(torch.int32).fill_(-1)
        _WOA_MARLIN_SENT[key] = buf
    return buf


def _get_woa_marlin_sent(device: torch.device, g: int) -> torch.Tensor | None:
    stream = torch.cuda.current_stream(device)
    buf = _WOA_MARLIN_SENT.get((device.index, stream.cuda_stream))
    if buf is not None:
        return buf
    if torch.cuda.is_current_stream_capturing():
        return _WOA_MARLIN_CAPTURE.get(device.index)
    return _alloc_woa_marlin_sent(device, stream, g)


def prepare_woa_mxfp8_marlin(wo_a_layer, n_local_groups: int) -> bool:
    """Repack wo_a MXFP8 weight/scale in-place to per-group Marlin-8bit (0 extra HBM)."""
    import os

    if os.environ.get("VLLM_DSV41_WOA_MARLIN", "0") != "1":
        return False
    w = wo_a_layer.weight
    s = getattr(wo_a_layer, "weight_scale", None)
    if not (
        isinstance(w, torch.Tensor)
        and isinstance(s, torch.Tensor)
        and w.is_cuda
        and w.element_size() == 1
        and w.is_contiguous()
        and s.is_contiguous()
        and w.numel() == n_local_groups * 1024 * 4096
        and s.numel() == n_local_groups * 1024 * 128
        and torch.cuda.get_device_capability(w.device) == (8, 0)
    ):
        return False
    su = s.view(torch.uint8)
    if not bool(((su >= 1) & (su <= 134)).all().item()):
        return False
    from vllm.model_executor.layers.fused_moe.dsv4_mxfp4_decode import _ops_r2

    r2 = _ops_r2()
    if r2 is None or not hasattr(r2, "woa_marlin_ksplit") or not hasattr(r2, "dequant_marlin_mxfp8"):
        return False

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        mxfp8_marlin_process_scales,
        pack_fp8_to_int32,
    )
    from vllm.model_executor.utils import replace_parameter

    param_dtype = torch.get_default_dtype()
    w2d = w.data.view(n_local_groups, 1024, 4096)
    s2d = su.view(n_local_groups, 1024, 128)
    mw_list = []
    ms_list = []
    for gi in range(n_local_groups):
        qw = pack_fp8_to_int32(w2d[gi].contiguous(), size_k_first=False).T.contiguous()
        mw = ops.gptq_marlin_repack(b_q_weight=qw, size_k=4096, size_n=1024, num_bits=8)
        mw_list.append(mw)
        sg = s2d[gi].contiguous().view(torch.float8_e8m0fnu).to(param_dtype).T.contiguous()
        ms = mxfp8_marlin_process_scales(
            marlin_permute_scales(s=sg, size_k=4096, size_n=1024, group_size=32)
        )
        ms_list.append(ms)
    mw_all = torch.stack(mw_list, dim=0).contiguous()
    ms_all = torch.stack(ms_list, dim=0).contiguous()
    replace_parameter(wo_a_layer, "weight", mw_all)
    replace_parameter(wo_a_layer, "weight_scale", ms_all)
    wo_a_layer._woa_marlin = True
    dev = mw_all.device
    cur = torch.cuda.current_stream(dev)
    _alloc_woa_marlin_sent(dev, cur, n_local_groups)
    if dev.index not in _WOA_MARLIN_CAPTURE:
        buf = torch.empty(max(n_local_groups, 2) * 16 * 2 * 16 * 64, dtype=torch.float32, device=dev)
        buf.view(torch.int32).fill_(-1)
        _WOA_MARLIN_CAPTURE[dev.index] = buf
    try:
        from vllm.utils.torch_utils import aux_stream

        aux = aux_stream()
        if aux is not None and aux.device == dev:
            _alloc_woa_marlin_sent(dev, aux, n_local_groups)
    except Exception:
        pass
    return True


def thin_woa_marlin(
    x: torch.Tensor, w_marlin: torch.Tensor, s_marlin: torch.Tensor
) -> torch.Tensor:
    m, g, k = x.shape
    if m == 0:
        return torch.empty((0, g, 1024), device=x.device, dtype=x.dtype)
    from vllm.model_executor.layers.fused_moe.dsv4_mxfp4_decode import _ops_r2

    r2 = _ops_r2()
    assert r2 is not None
    if (
        1 <= m <= 192
        and k == 4096
        and x.dtype == torch.bfloat16
        and x.stride(2) == 1
        and x.stride(0) % 8 == 0
        and x.stride(1) % 8 == 0
        and x.data_ptr() % 16 == 0
    ):
        part = _get_woa_marlin_sent(x.device, g)
        if part is not None:
            out = torch.empty((m, g, 1024), device=x.device, dtype=x.dtype)
            r2.woa_marlin_ksplit(x, w_marlin, s_marlin.view(torch.uint8), part, out)
            return out
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        get_dequant_mxfp8_scratch,
    )

    w_bf16_2d, cached = get_dequant_mxfp8_scratch(
        x.device, g * 1024, 4096, w_marlin, s_marlin
    )
    if not cached:
        for gi in range(g):
            r2.dequant_marlin_mxfp8(
                w_marlin[gi],
                s_marlin[gi],
                w_bf16_2d[gi * 1024 : (gi + 1) * 1024],
                1024,
                4096,
            )
    return torch.einsum("tgd,grd->tgr", x, w_bf16_2d.view(g, 1024, 4096)).to(x.dtype)


def _alloc_cpp_woa_scratch(
    device: torch.device, stream: torch.cuda.Stream, g: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.index, stream.cuda_stream)
    buf = _CPP_WOA_SCRATCH.get(key)
    need_part = 256 * max(g, 2) * 1024
    need_cnt = 16 * max(g, 2) * 16
    if buf is None or buf[0].numel() < need_part or buf[1].numel() < need_cnt:
        with torch.cuda.stream(stream):
            part = torch.empty(need_part, dtype=torch.float32, device=device)
            cnt = torch.zeros(need_cnt, dtype=torch.int32, device=device)
        buf = _CPP_WOA_SCRATCH[key] = (part, cnt)
    return buf


def _get_cpp_woa_scratch(
    device: torch.device, g: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    stream = torch.cuda.current_stream(device)
    buf = _CPP_WOA_SCRATCH.get((device.index, stream.cuda_stream))
    if buf is not None:
        return buf
    if torch.cuda.is_current_stream_capturing():
        return _CPP_WOA_CAPTURE.get(device.index)
    return _alloc_cpp_woa_scratch(device, stream, g)


def _cpp_woa_split(m: int) -> int:
    if m <= 6:
        return 16
    if m <= 32:
        return 8
    if m <= 64:
        return 4
    return 2


def _try_launch_wo_mxfp8_cpp(
    x: torch.Tensor, w_u8: torch.Tensor, s_u8: torch.Tensor, bf16_mul: bool
) -> torch.Tensor | None:
    import os

    if os.environ.get("VLLM_DSV41_WOA_CPP", "1") != "1":
        return None
    m, g, _ = x.shape
    if not (
        1 <= m <= 128
        and x.stride(0) % 8 == 0
        and x.stride(1) % 8 == 0
        and x.data_ptr() % 16 == 0
    ):
        return None
    from vllm.model_executor.layers.fused_moe.dsv4_mxfp4_decode import _ops_r2

    r2 = _ops_r2()
    if r2 is None or not hasattr(r2, "woa_mxfp8_splitk"):
        return None
    scr = _get_cpp_woa_scratch(x.device, g)
    if scr is None:
        return None
    part, cnt = scr
    sp = _cpp_woa_split(m)
    if part.numel() < sp * m * g * 1024 or cnt.numel() < ((m + 15) >> 4) * g * 16:
        return None
    out = torch.empty((m, g, 1024), device=x.device, dtype=x.dtype)
    r2.woa_mxfp8_splitk(x, w_u8, s_u8, bf16_mul, sp, part, cnt, out)
    return out


def warmup_thin_woa_mxfp8(device: torch.device, g: int) -> None:
    if g not in _WOA_CFGS:
        return
    wkey = (device.index, g)
    if wkey in _WARMED_DEVS:
        return
    cur = torch.cuda.current_stream(device)
    _alloc_cpp_woa_scratch(device, cur, g)
    if device.index not in _CPP_WOA_CAPTURE:
        _CPP_WOA_CAPTURE[device.index] = (
            torch.empty(256 * max(g, 2) * 1024, dtype=torch.float32, device=device),
            torch.zeros(16 * max(g, 2) * 16, dtype=torch.int32, device=device),
        )
    try:
        from vllm.utils.torch_utils import aux_stream

        aux = aux_stream()
        if aux is not None and aux.device == device:
            _alloc_cpp_woa_scratch(device, aux, g)
    except Exception:
        pass
    w_u8 = torch.zeros((g * 1024, 4096), dtype=torch.uint8, device=device)
    s_u8 = torch.full((g * 1024, 128), 127, dtype=torch.uint8, device=device)
    for m_max, _cfg in _WOA_FAST_CFGS:
        m_try = min(m_max, 513)
        x = torch.zeros((m_try, g, 4096), dtype=torch.bfloat16, device=device)
        for bf16_mul in (False, True):
            _launch_wo_mxfp8_fast(x, w_u8, s_u8, bf16_mul)
            if m_try <= 128:
                _try_launch_wo_mxfp8_cpp(x, w_u8, s_u8, bf16_mul)
            split = _woa_splitk()
            if split > 1 and m_try <= _WOA_SPLITK_MAX_M:
                _launch_wo_mxfp8_splitk(x, w_u8, s_u8, bf16_mul, split)
    for m_cpp in (6, 16, 64, 128):
        x = torch.zeros((m_cpp, g, 4096), dtype=torch.bfloat16, device=device)
        _try_launch_wo_mxfp8_cpp(x, w_u8, s_u8, True)
    _WARMED_DEVS.add(wkey)


# Register-dequant MXFP8 thin GEMM (no LUT gathers). Per M bucket:
# (max_m, (BM, BN, BK)); tuned on sm_80, G=2 (TP4). Above 512 tokens (prefill),
# dequant_mxfp8_to_bf16 + cuBLAS einsum runs at ~260 TFLOPS regardless of e8m0.
_WOA_FAST_CFGS = (
    (16, (16, 32, 256)),
    (32, (32, 32, 128)),
    (48, (16, 64, 128)),
    (96, (32, 32, 128)),
    (192, (64, 32, 128)),
    (512, (64, 64, 128)),
)
_READY_FAST: set = set()
# data_ptr(w_scale) -> every E8M0 scale <= 134 (exact bf16 power-of-two path).
_SCALE_BF16_OK: dict[int, bool] = {}


@triton.jit(do_not_specialize=["M"])
def wo_mxfp8_fast_kernel(
    X,
    W_u8,
    S_u8,
    Y,
    M,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX,
    SG,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BF16_MUL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    g = tl.program_id(1)
    pid_m = tl.program_id(2)
    n = pid_n * BN + tl.arange(0, BN)
    m = pid_m * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)
    KS: tl.constexpr = K // 32
    acc = tl.zeros((BM, BN), tl.float32)
    m_mask = m[:, None] < M
    for i in range(K // BK):
        kk = i * BK + k
        a = tl.load(X + m[:, None] * SX + g * SG + kk[None, :], mask=m_mask, other=0.0)
        # [BN, BK] fp8 bytes, K contiguous (coalesced).
        wb = tl.load(W_u8 + (g * N + n[:, None]) * K + kk[None, :]).to(tl.int16)
        # e4m3 -> bf16 bit pattern with exponent bias 7 left in place; the
        # missing 2^120 is folded into the (exact power-of-two) block scale, so
        # e4m3 subnormals are handled exactly too.
        bits = ((wb & 0x80) << 8) | ((wb & 0x7F) << 4)
        sb = tl.load(
            S_u8
            + (g * N + n[:, None]) * KS
            + (i * (BK // 32) + tl.arange(0, BK // 32))[None, :]
        ).to(tl.int32)
        if BF16_MUL:
            # 2^(s-127) * 2^120 == 2^(s-7) as a bf16 (requires s <= 134).
            sbf = ((sb + 120) << 7).to(tl.int16).to(tl.bfloat16, bitcast=True)
            wv = tl.reshape(bits.to(tl.bfloat16, bitcast=True), (BN, BK // 32, 32))
            b = tl.reshape(wv * sbf[:, :, None], (BN, BK))
        else:
            wv = tl.reshape(
                bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
                * 1.3292279957849158e36,  # 2^120
                (BN, BK // 32, 32),
            )
            sf = (sb << 23).to(tl.float32, bitcast=True)  # 2^(s-127)
            b = tl.reshape(wv * sf[:, :, None], (BN, BK)).to(tl.bfloat16)
        tmp = tl.dot(a, tl.trans(b))
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[acc, tmp],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(
        Y + (m[:, None] * G + g) * N + n[None, :],
        acc.to(Y.dtype.element_ty),
        mask=m_mask,
    )


def _fast_cfg(m: int):
    for max_m, cfg in _WOA_FAST_CFGS:
        if m <= max_m:
            return cfg
    return None


# Split-K for decode-sized M: the (BM=16, BN=32) grid is only 1024/32 * G = 64
# CTAs at TP4, each streaming 128 KiB of K serially (a 16-deep latency chain,
# ~20 us cold-L2 on GA100). Splitting K eight ways (512 CTAs) takes 16.4 us; the fixed-order FP32
# reduce kernel keeps the result deterministic.
def _woa_splitk() -> int:
    import os

    return int(os.environ.get("VLLM_DSV41_WOA_SPLITK", "8"))


_WOA_SPLITK_MAX_M = 16


@triton.jit(do_not_specialize=["M"])
def wo_mxfp8_splitk_kernel(
    X,
    W_u8,
    S_u8,
    P,
    M,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX,
    SG,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BF16_MUL: tl.constexpr,
):
    pid0 = tl.program_id(0)
    g = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_n = pid0 // SPLIT
    ks = pid0 % SPLIT
    n = pid_n * BN + tl.arange(0, BN)
    m = pid_m * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)
    KS: tl.constexpr = K // 32
    KPER: tl.constexpr = K // SPLIT
    acc = tl.zeros((BM, BN), tl.float32)
    m_mask = m[:, None] < M
    for i in range(KPER // BK):
        kk = ks * KPER + i * BK + k
        a = tl.load(X + m[:, None] * SX + g * SG + kk[None, :], mask=m_mask, other=0.0)
        wb = tl.load(W_u8 + (g * N + n[:, None]) * K + kk[None, :]).to(tl.int16)
        bits = ((wb & 0x80) << 8) | ((wb & 0x7F) << 4)
        sb = tl.load(
            S_u8
            + (g * N + n[:, None]) * KS
            + ((ks * KPER + i * BK) // 32 + tl.arange(0, BK // 32))[None, :]
        ).to(tl.int32)
        if BF16_MUL:
            sbf = ((sb + 120) << 7).to(tl.int16).to(tl.bfloat16, bitcast=True)
            wv = tl.reshape(bits.to(tl.bfloat16, bitcast=True), (BN, BK // 32, 32))
            b = tl.reshape(wv * sbf[:, :, None], (BN, BK))
        else:
            wv = tl.reshape(
                bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
                * 1.3292279957849158e36,  # 2^120
                (BN, BK // 32, 32),
            )
            sf = (sb << 23).to(tl.float32, bitcast=True)  # 2^(s-127)
            b = tl.reshape(wv * sf[:, :, None], (BN, BK)).to(tl.bfloat16)
        tmp = tl.dot(a, tl.trans(b))
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[acc, tmp],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(
        P + ((ks * M + m[:, None]) * G + g) * N + n[None, :],
        acc,
        mask=m_mask,
    )


@triton.jit
def wo_splitk_reduce_kernel(P, Y, TOTAL, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    acc = tl.load(P + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT):
        v = tl.load(P + s * TOTAL + offs, mask=mask, other=0.0)
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[acc, v],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(Y + offs, acc.to(Y.dtype.element_ty), mask=mask)


def _launch_wo_mxfp8_splitk(x, w_u8, s_u8, bf16_mul, split):
    m, g, _ = x.shape
    bm, bn, bk = 16, 32, 128
    n_out, k_in = 1024, 4096
    part = torch.empty((split, m, g, n_out), device=x.device, dtype=torch.float32)
    out = torch.empty((m, g, n_out), device=x.device, dtype=x.dtype)
    wo_mxfp8_splitk_kernel[(triton.cdiv(n_out, bn) * split, g, triton.cdiv(m, bm))](
        x,
        w_u8,
        s_u8,
        part,
        m,
        g,
        n_out,
        k_in,
        x.stride(0),
        x.stride(1),
        split,
        bm,
        bn,
        bk,
        bf16_mul,
        num_warps=4,
        num_stages=3,
    )
    total = m * g * n_out
    wo_splitk_reduce_kernel[(triton.cdiv(total, 1024),)](
        part, out, total, split, 1024, num_warps=4
    )
    _READY_FAST.add((x.device.index, "splitk", split, g, bf16_mul))
    return out


def _launch_wo_mxfp8_fast(x, w_u8, s_u8, bf16_mul):
    m, g, _ = x.shape
    bm, bn, bk = _fast_cfg(m)
    out = torch.empty((m, g, 1024), device=x.device, dtype=x.dtype)
    wo_mxfp8_fast_kernel[(triton.cdiv(1024, bn), g, triton.cdiv(m, bm))](
        x,
        w_u8,
        s_u8,
        out,
        m,
        g,
        1024,
        4096,
        x.stride(0),
        x.stride(1),
        bm,
        bn,
        bk,
        bf16_mul,
        num_warps=4,
        num_stages=3,
    )
    _READY_FAST.add((x.device.index, bm, bn, bk, g, bf16_mul))
    return out


def _scale_bf16_ok(w_scale: torch.Tensor) -> bool | None:
    key = w_scale.data_ptr()
    ok = _SCALE_BF16_OK.get(key)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ok = bool(w_scale.view(torch.uint8).max().item() <= 134)
        _SCALE_BF16_OK[key] = ok
    return ok


def _fallback_dequant_einsum(
    x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor
) -> torch.Tensor:
    m, g, k = x.shape
    n_total = w_fp8.shape[0]
    r = n_total // g
    if m == 0:
        return torch.empty((0, g, r), device=x.device, dtype=x.dtype)
    if (
        w_fp8.is_cuda
        and w_fp8.is_contiguous()
        and w_scale.is_contiguous()
        and not torch.cuda.is_current_stream_capturing()
    ):
        from vllm.model_executor.layers.fused_moe.dsv4_mxfp4_decode import _ops_r2
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            get_dequant_mxfp8_scratch,
        )

        r2 = _ops_r2()
        if r2 is not None and hasattr(r2, "dequant_row_mxfp8"):
            w_bf16_2d, cached = get_dequant_mxfp8_scratch(
                x.device, n_total, k, w_fp8, w_scale
            )
            if not cached:
                r2.dequant_row_mxfp8(
                    w_fp8.view(torch.uint8), w_scale.view(torch.uint8), w_bf16_2d
                )
            return torch.einsum("tgd,grd->tgr", x, w_bf16_2d.view(g, r, k)).to(x.dtype)

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )

    w_bf16 = dequant_mxfp8_to_bf16(
        w_fp8.view(n_total, k), w_scale.view(torch.uint8).view(n_total, k // 32)
    ).view(g, r, k)
    return torch.einsum("tgd,grd->tgr", x, w_bf16).to(x.dtype)


def thin_woa_mxfp8(
    x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor
) -> torch.Tensor:
    m, g, k = x.shape
    if m == 0:
        return torch.empty((0, g, w_fp8.shape[0] // g), device=x.device, dtype=x.dtype)
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 3
        and g in _WOA_CFGS
        and k == 4096
        and x.stride(-1) == 1
        and w_fp8.is_contiguous()
        and w_scale.is_contiguous()
        and w_fp8.numel() == g * 1024 * 4096
        and w_scale.numel() == g * 1024 * 128
        and torch.cuda.get_device_capability(x.device) == (8, 0)
    ):
        return _fallback_dequant_einsum(x, w_fp8, w_scale)

    # Checked once per weight outside capture (the eager profile/warmup run).
    bf16_mul = bool(_scale_bf16_ok(w_scale))
    w_u8 = w_fp8.view(torch.uint8)
    s_u8 = w_scale.view(torch.uint8)
    cpp_out = _try_launch_wo_mxfp8_cpp(x, w_u8, s_u8, bf16_mul)
    if cpp_out is not None:
        return cpp_out
    split = _woa_splitk()
    if split > 1 and 1 <= m <= _WOA_SPLITK_MAX_M and 4096 % (split * 128) == 0:
        key = (x.device.index, "splitk", split, g, bf16_mul)
        if key in _READY_FAST or not torch.cuda.is_current_stream_capturing():
            return _launch_wo_mxfp8_splitk(x, w_u8, s_u8, bf16_mul, split)
    cfg = _fast_cfg(m) if m >= 1 else None
    if cfg is not None:
        key = (x.device.index, *cfg, g, bf16_mul)
        if key in _READY_FAST or not torch.cuda.is_current_stream_capturing():
            return _launch_wo_mxfp8_fast(x, w_u8, s_u8, bf16_mul)

    return _fallback_dequant_einsum(x, w_fp8, w_scale)


def thin_woa_mxfp8_lut(
    x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor
) -> torch.Tensor:
    """Previous LUT-dequant kernel path (kept for A/B; ~20x slower on sm_80)."""
    m, g, k = x.shape
    if m == 0:
        return torch.empty((0, g, w_fp8.shape[0] // g), device=x.device, dtype=x.dtype)
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 3
        and g in _WOA_CFGS
        and k == 4096
        and x.stride(-1) == 1
        and w_fp8.is_contiguous()
        and w_scale.is_contiguous()
        and w_fp8.numel() == g * 1024 * 4096
        and w_scale.numel() == g * 1024 * 128
        and torch.cuda.get_device_capability(x.device) == (8, 0)
    ):
        return _fallback_dequant_einsum(x, w_fp8, w_scale)

    bn, bk, warps, stages, max_m = _WOA_CFGS[g]
    if not (1 <= m <= max_m):
        return _fallback_dequant_einsum(x, w_fp8, w_scale)

    bm = max(16, triton.next_power_of_2(m))
    key = (x.device.index, bm, g)
    if key not in _READY_FP8 and torch.cuda.is_current_stream_capturing():
        return _fallback_dequant_einsum(x, w_fp8, w_scale)

    v_lut, s_lut = _get_fp8_bf16_lut(x.device)
    out = torch.empty((m, g, 1024), device=x.device, dtype=x.dtype)
    wo_mxfp8_kernel[(triton.cdiv(1024, bn), g)](
        x,
        w_fp8.view(torch.uint8),
        w_scale.view(torch.uint8),
        v_lut,
        s_lut,
        out,
        m,
        g,
        1024,
        4096,
        x.stride(0),
        x.stride(1),
        bm,
        bn,
        bk,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
    _READY_FP8.add(key)
    return out
