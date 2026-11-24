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


def warmup_thin_woa_mxfp8(device: torch.device, g: int) -> None:
    if g not in _WOA_CFGS:
        return
    wkey = (device.index, g)
    if wkey in _WARMED_DEVS:
        return
    w_u8 = torch.zeros((g * 1024, 4096), dtype=torch.uint8, device=device)
    s_u8 = torch.full((g * 1024, 128), 127, dtype=torch.uint8, device=device)
    for m_try, _cfg in _WOA_FAST_CFGS:
        x = torch.zeros((m_try, g, 4096), dtype=torch.bfloat16, device=device)
        for bf16_mul in (False, True):
            _launch_wo_mxfp8_fast(x, w_u8, s_u8, bf16_mul)
    _WARMED_DEVS.add(wkey)


# Register-dequant MXFP8 thin GEMM (no LUT gathers). Per M bucket:
# (max_m, (BM, BN, BK)); tuned on sm_80 CMP 170HX, G=2 (TP4). Above the last
# bucket the dequant + einsum fallback is used (prefill-sized M).
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
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )

    m, g, k = x.shape
    n_total = w_fp8.shape[0]
    r = n_total // g
    if m == 0:
        return torch.empty((0, g, r), device=x.device, dtype=x.dtype)
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
    cfg = _fast_cfg(m) if m >= 1 else None
    if cfg is not None:
        key = (x.device.index, *cfg, g, bf16_mul)
        if key in _READY_FAST or not torch.cuda.is_current_stream_capturing():
            return _launch_wo_mxfp8_fast(
                x, w_fp8.view(torch.uint8), w_scale.view(torch.uint8), bf16_mul
            )

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
