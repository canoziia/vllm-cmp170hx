# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-streaming MXFP4 MoE decode for DeepSeek V4.1 Flash on sm_80.

VLLM_DSV4_MXFP4_DECODE=1 (default 0 = unchanged). For small decode batches,
replaces the two ``moe_wna16_marlin_gemm`` calls of ``fused_marlin_moe`` with
the ``_dsv4_moe_C`` / ``_dsv4_moe2`` / ``_dsv41_r2_C`` kernels, which read the
same Marlin-packed MXFP4 weights and e8m0 scales. The weights are dequantised
exactly; only the fp32 accumulation order differs from Marlin.

Eligible calls: sm_80, bf16 activations, float4_e2m1f weights, no bias /
zero points / global scales / 8-bit activations, router weight applied on the
output, block-8 alignment, at most VLLM_DSV4_MXFP4_DECODE_MAX_TOKENS (default
192) tokens, the compiled shapes (hidden 5120, intermediate 2304, including
EP=4 E=96/32 topk=6/3), and e8m0 scale bounds checked once per weight tensor
outside CUDA-graph capture. Anything else uses the unchanged Marlin path.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_K, _NI = 5120, 2304
_NI_SUPPORTED = (2304, 576)
_OPS = None
_LOAD_ERROR: str | None = None
_SCALE_OK: dict[int | tuple[int, int], bool] = {}
_CTR: dict[tuple, torch.Tensor] = {}
_CTR_CAPTURE: dict[str, torch.Tensor] = {}
_LOGGED = False
_OPS2 = None
_OPS_R2 = None
_OPS_R2_FAILED = False
_SYNC: dict[tuple, torch.Tensor] = {}
_SYNC_CAPTURE: dict[str, torch.Tensor] = {}
_SYNC_SIZE = 65536  # 128 w2 tile counters + w13 (block, tile pair) counters (M*topk <= 1816)
_LOGGED2 = False
_FGTA_CNT: dict[tuple, torch.Tensor] = {}
_FGTA_CNT_CAPTURE: dict[str, torch.Tensor] = {}
_FGTA_SENT: dict[tuple, torch.Tensor] = {}
_FGTA_SENT_CAPTURE: dict[str, torch.Tensor] = {}
_PREALIGNED: dict[
    tuple[int | None, int],
    tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_PENDING_SHARED: dict[tuple[int | None, int], tuple[object, bool]] = {}
_FUSED_SHARED_OUT: dict[tuple[int | None, int], tuple[int, int]] = {}
_DEFER_W2_SUM: dict[tuple[int | None, int], bool] = {}
_DEFERRED_W2_SPLITK: dict[tuple[int | None, int], tuple] = {}


def enabled() -> bool:
    return os.environ.get("VLLM_DSV4_MXFP4_DECODE", "0") == "1"


def _max_tokens() -> int:
    return int(os.environ.get("VLLM_DSV4_MXFP4_DECODE_MAX_TOKENS", "192"))


def _dq() -> int:
    return int(os.environ.get("VLLM_DSV4_MXFP4_DQ", "0"))


def _fused_act() -> bool:
    return os.environ.get("VLLM_DSV4_MXFP4_FUSED_ACT", "0") == "1"


def _fused_sum() -> bool:
    return os.environ.get("VLLM_DSV4_MXFP4_FUSED_SUM", "0") == "1"


def _sum_order(moe_sum=None, expert_map=None) -> int:
    env = os.environ.get("VLLM_DSV4_MXFP4_FUSED_SUM_ORDER")
    if env is not None:
        return int(env)
    if expert_map is not None or moe_sum is not None:
        return 0
    return 1


def _sum_max_tokens() -> int:
    return int(os.environ.get("VLLM_DSV4_MXFP4_FUSED_SUM_MAX_TOKENS", "32"))


def _v4_tile_max_tokens() -> int:
    """Largest M that takes the (CR=16, TW=1) tile variant (cfg 4) instead of
    (CR=8, TW=2) (cfg 5). Cold-L2 scan on 108-SM GA100 (random top-6 of 384,
    EP=4): cfg 4 is 9-29% faster for W13+W2 at M = 12..48 and cfg 5 wins
    from M = 96."""
    return int(os.environ.get("VLLM_DSV4_MXFP4_V4_MAX_TOKENS", "64"))


def _w2_splitk_max_tokens() -> int:
    """Largest M whose W2 takes the split-K kernel plus its fused top-k sum
    (`_dsv41_r2_C.w2_splitk_ep`) instead of the KS=1 fused-sum GEMM. With
    KS=1 the W2 grid is n_experts * 80 items, too few to fill the GPU at
    decode sizes: split-K is 15-25% faster at M = 6..48 (top-6)."""
    return int(os.environ.get("VLLM_DSV4_MXFP4_W2_SPLITK_MAX_TOKENS", "64"))


def _w2_splitk_ks(M: int) -> int:
    return 4 if M < 20 else 3


def cuda_align_enabled() -> bool:
    return os.environ.get("VLLM_DSV4_CUDA_DET_ALIGN", "0") == "1"


def _ops2():
    global _OPS2
    if _OPS2 is None:
        _ops()
        ops2 = torch.ops._dsv4_moe2
        if int(ops2.abi()) != 1:
            raise RuntimeError("unexpected _dsv4_moe2 ABI")
        _OPS2 = ops2
    return _OPS2


def _ops_r2():
    global _OPS_R2, _OPS_R2_FAILED
    if _OPS_R2 is not None or _OPS_R2_FAILED:
        return _OPS_R2
    try:
        path = os.environ.get("VLLM_DSV41_R2_LIB")
        if not path:
            path = os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                ),
                "_dsv41_r2_C.abi3.so",
            )
        if not os.path.exists(path):
            _OPS_R2_FAILED = True
            return None
        torch.ops.load_library(path)
        _OPS_R2 = torch.ops._dsv41_r2_C
    except Exception as e:  # noqa: BLE001
        _OPS_R2_FAILED = True
        logger.warning("Failed to load _dsv41_r2_C: %s", e)
    return _OPS_R2


def _fgta_cnt(dev: torch.device) -> torch.Tensor | None:
    dev_key = str(dev)
    key = (dev_key, torch.cuda.current_stream(dev).cuda_stream)
    t = _FGTA_CNT.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return _FGTA_CNT_CAPTURE.get(dev_key)
        t = torch.zeros(4, dtype=torch.int32, device=dev)
        _FGTA_CNT[key] = t
        if dev_key not in _FGTA_CNT_CAPTURE:
            _FGTA_CNT_CAPTURE[dev_key] = torch.zeros(4, dtype=torch.int32, device=dev)
    return t


def _fgta_sent(dev: torch.device) -> torch.Tensor | None:
    dev_key = str(dev)
    key = (dev_key, torch.cuda.current_stream(dev).cuda_stream)
    t = _FGTA_SENT.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return _FGTA_SENT_CAPTURE.get(dev_key)
        t = torch.empty(16384, dtype=torch.float32, device=dev)
        t.view(torch.int32).fill_(-1)
        _FGTA_SENT[key] = t
        if dev_key not in _FGTA_SENT_CAPTURE:
            c = torch.empty(16384, dtype=torch.float32, device=dev)
            c.view(torch.int32).fill_(-1)
            _FGTA_SENT_CAPTURE[dev_key] = c
    return t


def set_defer_w2_sum(dev: torch.device, active: bool) -> None:
    key = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
    if active:
        _DEFER_W2_SUM[key] = True
    else:
        _DEFER_W2_SUM.pop(key, None)


def pop_deferred_w2_splitk(x: torch.Tensor):
    dev = x.device
    key = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
    entry = _DEFERRED_W2_SPLITK.pop(key, None)
    if entry is not None and entry[0] == x.data_ptr():
        return entry[1:]
    return None


def warmup_device(dev: torch.device) -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    _ops_r2()
    _sync(dev)
    _counters(dev)
    _fgta_cnt(dev)
    _fgta_sent(dev)


def pop_prealigned(
    topk_ids: torch.Tensor, local_e: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    dev = topk_ids.device
    key = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
    entry = _PREALIGNED.pop(key, None)
    if entry is None:
        return None
    ptr, le, bs, s_ids, e_ids, ntpp = entry
    if ptr == topk_ids.data_ptr() and le == local_e and bs == block_size:
        return s_ids, e_ids, ntpp
    return None


def clear_prealigned(dev: torch.device) -> None:
    _PREALIGNED.pop((dev.index, torch.cuda.current_stream(dev).cuda_stream), None)


def push_shared_experts(
    dev: torch.device, shared_experts: object, overlapping: bool
) -> None:
    key = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
    _PENDING_SHARED[key] = (shared_experts, overlapping)
    _FUSED_SHARED_OUT.pop(key, None)


def clear_shared_experts(dev: torch.device) -> None:
    _PENDING_SHARED.pop((dev.index, torch.cuda.current_stream(dev).cuda_stream), None)


def pop_fused_shared(fused_out: torch.Tensor, shared_out: torch.Tensor) -> bool:
    dev = fused_out.device
    key = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
    entry = _FUSED_SHARED_OUT.pop(key, None)
    return entry == (fused_out.data_ptr(), shared_out.data_ptr())


def try_fused_gate_topk_align(
    hidden_states: torch.Tensor,
    gate,
    router,
    routed_experts,
    indices_dtype: torch.dtype | None,
    input_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if os.environ.get("VLLM_DSV41_FAST_GATE_TOPK", "0") != "1":
        return None
    if hidden_states.ndim != 2 or not hidden_states.is_cuda:
        return None
    M, K = hidden_states.shape
    if (
        M < 1
        or M > 8
        or K != _K
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_contiguous()
        or hidden_states.data_ptr() % 16 != 0
    ):
        return None
    w = getattr(gate, "weight", None)
    if (
        w is None
        or not getattr(gate, "_router_gemm_no_bias", False)
        or getattr(gate, "allow_specialized_router_gemm", False)
        or getattr(gate, "allow_bf16x3_router_gemm", False)
        or getattr(gate, "out_dtype", torch.float32) != torch.float32
        or w.ndim != 2
        or w.shape[1] != K
        or w.dtype != torch.bfloat16
        or not w.is_contiguous()
        or w.data_ptr() % 16 != 0
    ):
        return None
    E = w.shape[0]
    topk = getattr(router, "top_k", -1)
    if (E, topk) not in ((384, 6), (128, 3)):
        return None
    cb_param = getattr(router, "e_score_correction_bias", None)
    if (
        cb_param is None
        or getattr(router, "scoring_func", "") != "sqrtsoftplus"
        or not getattr(router, "renormalize", False)
        or getattr(router, "_hash_indices_table", None) is not None
        or getattr(router, "num_fused_shared_experts", 0) != 0
        or getattr(router, "capture_fn", None) is not None
        or getattr(router, "eplb_enabled", False)
    ):
        return None
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        get_moe_padding_mask,
    )

    pad_mask = get_moe_padding_mask(M)
    if pad_mask is not None and not (
        pad_mask.is_cuda
        and pad_mask.is_contiguous()
        and pad_mask.dtype == torch.bool
        and pad_mask.numel() >= M
    ):
        return None
    cb = cb_param.data
    if not (cb.is_cuda and cb.is_contiguous() and cb.dtype == torch.float32 and cb.numel() == E):
        return None

    bvl_param = getattr(router, "bias_vl", None)
    sentinel = int(getattr(router, "image_sentinel_lo", 0))
    has_vl = bvl_param is not None and sentinel > 0
    bvl = None
    if has_vl:
        bvl = bvl_param.data
        if (
            input_ids is None
            or not input_ids.is_cuda
            or not input_ids.is_contiguous()
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.numel() < M
            or not bvl.is_cuda
            or not bvl.is_contiguous()
            or bvl.dtype != torch.float32
            or bvl.numel() != E
        ):
            return None

    r2 = _ops_r2()
    if r2 is None or not hasattr(r2, "fused_gate_topk_align"):
        return None
    done_cnt = _fgta_cnt(hidden_states.device)
    if done_cnt is None:
        return None

    out_idx_dtype = torch.int32 if indices_dtype is None else indices_dtype
    if out_idx_dtype not in (torch.int32, torch.int64):
        return None

    local_e = int(getattr(routed_experts, "local_num_experts", E))
    emap = getattr(routed_experts, "expert_map", None)
    do_align = (
        cuda_align_enabled()
        and type(routed_experts).__name__ == "RoutedExperts"
        and getattr(routed_experts, "_lora_context", None) is None
        and 0 < local_e <= 384
        and (
            emap is None
            or (
                emap.is_cuda
                and emap.is_contiguous()
                and emap.dtype == torch.int32
                and emap.numel() >= E
            )
        )
    )

    dev = hidden_states.device
    sk_scr = 4 if E == 384 else 10
    fgta_sent = _fgta_sent(dev) if M >= 3 else None
    use_sentinel = fgta_sent is not None and fgta_sent.numel() >= sk_scr * 8 * E
    logits_scr = (
        fgta_sent
        if use_sentinel
        else torch.empty((sk_scr * M * E,), dtype=torch.float32, device=dev)
    )
    topk_weights = torch.empty((M, topk), dtype=torch.float32, device=dev)
    topk_ids = torch.empty((M, topk), dtype=out_idx_dtype, device=dev)
    s_ids = e_ids = ntpp = None
    if do_align:
        # Since M * topk <= 48 < E (128 or 384), moe_align_block_size caps
        # max_num_tokens_padded at (M * topk) * 8 and max_num_m_blocks at M * topk.
        max_padded = M * topk * 8
        s_ids = torch.empty((max_padded,), dtype=torch.int32, device=dev)
        e_ids = torch.empty((M * topk,), dtype=torch.int32, device=dev)
        ntpp = torch.empty((1,), dtype=torch.int32, device=dev)

    r2.fused_gate_topk_align(
        hidden_states,
        w,
        cb,
        logits_scr,
        done_cnt,
        topk_weights,
        topk_ids,
        float(getattr(router, "routed_scaling_factor", 1.0)),
        input_ids if has_vl else None,
        bvl if has_vl else None,
        sentinel if has_vl else 0,
        do_align,
        local_e,
        emap,
        s_ids,
        e_ids,
        ntpp,
        pad_mask,
        use_sentinel,
    )
    if do_align:
        assert s_ids is not None and e_ids is not None and ntpp is not None
        key = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
        _PREALIGNED[key] = (topk_ids.data_ptr(), local_e, 8, s_ids, e_ids, ntpp)
        logger.info_once(
            "DSV41: single-kernel fused_gate_topk_align active (E=%d, topk=%d, local_e=%d, pad_mask=%s)",
            E,
            topk,
            local_e,
            pad_mask is not None,
        )
    return topk_weights, topk_ids


def _sync(dev: torch.device) -> torch.Tensor | None:
    """Zeroed int32 counters, self-resetting inside the kernels."""
    dev_key = str(dev)
    key = (dev_key, torch.cuda.current_stream(dev).cuda_stream)
    t = _SYNC.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            return _SYNC_CAPTURE.get(dev_key)
        t = torch.zeros(_SYNC_SIZE, dtype=torch.int32, device=dev)
        _SYNC[key] = t
        if dev_key not in _SYNC_CAPTURE:
            _SYNC_CAPTURE[dev_key] = torch.zeros(_SYNC_SIZE, dtype=torch.int32, device=dev)
        _fgta_cnt(dev)
    return t


def _plain_clamp_limit(activation, activation_func, activation_config) -> float | None:
    """clamp limit if the activation is exactly _C.silu_and_mul_with_clamp."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    if activation != MoEActivation.SILU:
        return None
    if activation_func is None:
        cfg = activation_config
    else:
        f = getattr(activation_func, "__func__", None)
        if f is None or f.__qualname__ != "FusedMoEExpertsModular.activation":
            return None
        cfg = getattr(activation_func.__self__, "activation_config", None)
    if cfg is None or cfg.clamp_limit is None:
        return None
    return float(cfg.clamp_limit)


def _plain_moe_sum(moe_sum) -> bool:
    if moe_sum is None or moe_sum is False:
        return moe_sum is None
    from vllm import _custom_ops as custom_ops

    if moe_sum is custom_ops.moe_sum:
        return True
    f = getattr(moe_sum, "__func__", None)
    return f is not None and f.__qualname__ == "MarlinExperts.moe_sum"


def try_det_align(
    topk_ids,
    num_experts,
    block_size,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad,
    expert_map,
    local_num_experts=None,
) -> bool:
    """CUDA deterministic align; False keeps the caller's path."""
    if not topk_ids.is_cuda or not topk_ids.is_contiguous():
        return False
    if topk_ids.dtype not in (torch.int32, torch.int64):
        return False
    s = topk_ids.numel()
    local_e = num_experts if local_num_experts is None else local_num_experts
    r2 = _ops_r2()
    max_s = 4096 if r2 is not None else 512
    if s == 0 or s > max_s or num_experts <= 0 or not 0 < local_e <= 1024:
        return False
    if expert_map is not None and (
        not expert_map.is_cuda
        or not expert_map.is_contiguous()
        or expert_map.dtype != torch.int32
        or expert_map.numel() < num_experts
    ):
        return False
    if not (
        sorted_ids.is_contiguous()
        and expert_ids.is_contiguous()
        and sorted_ids.dtype
        == expert_ids.dtype
        == num_tokens_post_pad.dtype
        == torch.int32
    ):
        return False
    if s > 512 and r2 is not None:
        r2.det_align_v2(
            topk_ids.view(-1),
            expert_map,
            num_experts,
            local_e,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
        return True
    if expert_map is None and local_num_experts is None:
        _ops2().align(
            topk_ids.view(-1),
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
    else:
        _ops2().align_ep(
            topk_ids.view(-1),
            expert_map,
            num_experts,
            local_e,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
    return True


def _ops():
    global _OPS, _LOAD_ERROR
    if _OPS is not None or _LOAD_ERROR is not None:
        return _OPS
    try:
        path = os.environ.get("VLLM_DSV4_MXFP4_DECODE_LIB")
        if not path:
            path = os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                ),
                "_dsv4_moe_C.abi3.so",
            )
        torch.ops.load_library(path)
        ops = torch.ops._dsv4_moe_C
        if int(ops.abi()) != 1:
            raise RuntimeError("unexpected _dsv4_moe_C ABI")
        _OPS = ops
    except Exception as e:  # noqa: BLE001
        _LOAD_ERROR = repr(e)
        raise RuntimeError(
            f"VLLM_DSV4_MXFP4_DECODE=1 but _dsv4_moe_C could not be loaded: {e}"
        ) from e
    return _OPS


def _scales_ok(s: torch.Tensor, limit: int = 128) -> bool | None:
    key = s.data_ptr() if limit == 128 else (s.data_ptr(), limit)
    ok = _SCALE_OK.get(key)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ok = int(s.view(torch.uint8).max().item()) <= limit
        _SCALE_OK[key] = ok
        if not ok:
            logger.warning(
                "VLLM_DSV4_MXFP4_DECODE: a layer has e8m0 scales > %d; "
                "it keeps the Marlin path",
                limit,
            )
    return ok


def _counters(dev: torch.device) -> torch.Tensor | None:
    dev_key = str(dev)
    key = (dev_key, torch.cuda.current_stream(dev).cuda_stream)
    ctr = _CTR.get(key)
    if ctr is None:
        if torch.cuda.is_current_stream_capturing():
            return _CTR_CAPTURE.get(dev_key)
        ctr = torch.zeros(2, dtype=torch.int32, device=dev)
        _CTR[key] = ctr
        if dev_key not in _CTR_CAPTURE:
            _CTR_CAPTURE[dev_key] = torch.zeros(2, dtype=torch.int32, device=dev)
    return ctr


def try_run(
    *,
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    quant_type,
    bias1,
    bias2,
    global_scale1,
    global_scale2,
    input_global_scale1=None,
    input_global_scale2=None,
    w1_zeros=None,
    w2_zeros=None,
    expert_map=None,
    apply_router_weight_on_input=False,
    block_size_m=8,
    input_dtype=None,
    activation=None,
    activation_func=None,
    activation_config=None,
    sorted_token_ids=None,
    expert_ids=None,
    num_tokens_post_padded=None,
    intermediate_cache13=None,
    intermediate_cache2=None,
    moe_sum=None,
    output=None,
    global_num_experts=-1,
    mul_topk_weights_in_gemm2=True,
):
    """Return (tensor, summed) or None to keep the Marlin path.

    summed=False: tensor is the [M*topk, K] w2 output (caller runs moe_sum);
    summed=True: tensor is the final [M, K] output (fused moe_sum, v2 only).
    """
    from vllm.scalar_type import scalar_types

    M, K = hidden_states.shape
    topk = topk_ids.size(1)
    ni = w2.size(1) * 16
    if (
        quant_type != scalar_types.float4_e2m1f
        or K != _K
        or ni not in _NI_SUPPORTED
        or M > _max_tokens()
        or block_size_m != 8
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_cuda
        or input_dtype is not None
        or apply_router_weight_on_input
        or any(
            t is not None
            for t in (
                bias1,
                bias2,
                global_scale1,
                global_scale2,
                input_global_scale1,
                input_global_scale2,
                w1_zeros,
                w2_zeros,
            )
        )
        or activation is None
        or not activation.is_gated
        or w1_scale.element_size() != 1
        or w2_scale.element_size() != 1
        or torch.cuda.get_device_capability(hidden_states.device) != (8, 0)
    ):
        return None
    if (
        not topk_ids.is_cuda
        or not topk_ids.is_contiguous()
        or topk_ids.dtype not in (torch.int32, torch.int64)
    ):
        return None
    global_e = (
        global_num_experts
        if global_num_experts > 0
        else (expert_map.numel() if expert_map is not None else w1.size(0))
    )
    if expert_map is not None and (
        not expert_map.is_cuda
        or not expert_map.is_contiguous()
        or expert_map.dtype != torch.int32
        or expert_map.numel() < global_e
    ):
        return None
    ops = _ops()
    dq = _dq()
    limit = (
        _plain_clamp_limit(activation, activation_func, activation_config)
        if _fused_act()
        else None
    )
    fsum = (
        mul_topk_weights_in_gemm2
        and _fused_sum()
        and M <= _sum_max_tokens()
        and _plain_moe_sum(moe_sum)
        and topk <= 8
        and (
            output is None
            or (
                output.dtype == torch.bfloat16
                and output.is_contiguous()
                and tuple(output.shape) == (M, K)
            )
        )
    )
    if (
        dq != 0
        or limit is not None
        or fsum
        or expert_map is not None
        or ni != _NI
        or not mul_topk_weights_in_gemm2
    ):
        return _try_run_v2(
            ops,
            dq,
            limit,
            fsum,
            ni,
            mul_topk_weights_in_gemm2,
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            expert_map,
            global_e,
            moe_sum,
            activation,
            activation_func,
            activation_config,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            intermediate_cache13,
            intermediate_cache2,
            output,
        )
    if not (_scales_ok(w1_scale) and _scales_ok(w2_scale)):
        return None
    slots = M * topk
    ctr = _counters(hidden_states.device)
    if ctr is None:
        return None
    if (
        intermediate_cache13.numel() < slots * max(2 * _NI, _K)
        or intermediate_cache2.numel() < slots * _NI
        or not intermediate_cache13.is_contiguous()
        or not intermediate_cache2.is_contiguous()
    ):
        return None
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        logger.info(
            "VLLM_DSV4_MXFP4_DECODE: weight-streaming MXFP4 MoE decode active "
            "(<= %d tokens)",
            _max_tokens(),
        )
    dev = hidden_states.device
    cfg = 2 if M <= 6 else 5
    part = torch.empty(4 * slots * 2 * _NI, dtype=torch.float32, device=dev)
    act_in = intermediate_cache13.reshape(-1)[: slots * 2 * _NI].view(slots, 2 * _NI)
    h = intermediate_cache2.reshape(-1)[: slots * _NI].view(slots, _NI)
    tw = topk_weights.view(-1)
    ks = ops.gemm(
        hidden_states,
        w1,
        w1_scale.view(torch.uint8),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        tw,
        topk,
        slots,
        _K,
        2 * _NI,
        True,
        cfg,
        ctr,
        part,
    )
    ops.sum(part, ks, slots, 2 * _NI, act_in)
    if activation_func is None:
        from vllm.model_executor.layers.fused_moe.activation import (
            ApplyMoEActivationConfig,
            apply_moe_activation,
        )

        apply_moe_activation(
            activation,
            h,
            act_in,
            activation_config=activation_config or ApplyMoEActivationConfig(),
            topk_ids=topk_ids,
            expert_map=None,
        )
    else:
        activation_func(activation, h, act_in, topk_ids=topk_ids, expert_map=None)
    out = intermediate_cache13.reshape(-1)[: slots * _K].view(slots, _K)
    ops.gemm(
        h,
        w2,
        w2_scale.view(torch.uint8),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        tw,
        topk,
        slots,
        _NI,
        _K,
        False,
        cfg,
        ctr,
        out,
    )
    return out, False


def _try_run_v2(
    ops,
    dq,
    limit,
    fsum,
    ni,
    mul_topk_weights_in_gemm2,
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    expert_map,
    global_e,
    moe_sum,
    activation,
    activation_func,
    activation_config,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    intermediate_cache13,
    intermediate_cache2,
    output,
):
    if dq not in (0, 1, 2, 3):
        return None
    smax = 128 if dq == 0 else 254
    if not (_scales_ok(w1_scale, smax) and _scales_ok(w2_scale, smax)):
        return None
    M = hidden_states.size(0)
    topk = topk_ids.size(1)
    slots = M * topk
    dev = hidden_states.device
    ctr = _counters(dev)
    sync = _sync(dev)
    if ctr is None or sync is None:
        return None
    if (
        intermediate_cache13.numel() < slots * max(2 * ni, _K)
        or intermediate_cache2.numel() < slots * ni
        or not intermediate_cache13.is_contiguous()
        or not intermediate_cache2.is_contiguous()
    ):
        return None
    hp = ni // 64
    if 128 + slots * hp > _SYNC_SIZE:
        return None
    ops2 = _ops2()
    s_order = _sum_order(moe_sum, expert_map)
    global _LOGGED2
    if not _LOGGED2:
        _LOGGED2 = True
        logger.info(
            "VLLM_DSV4_MXFP4_DECODE v2: ni=%d dq=%d fused_act=%s fused_sum=%s (order %d)",
            ni,
            dq,
            limit is not None,
            fsum,
            s_order,
        )
    if ni == 576:
        w13_cfg = 2 if dq in (0, 3) else 4
        w2_cfg = int(
            os.environ.get(
                "VLLM_DSV4_MXFP4_TP4_W2_CFG",
                "2" if M <= _v4_tile_max_tokens() else "5",
            )
        )
    else:
        w13_cfg = (2 if dq in (0, 3) else 4) if M <= _v4_tile_max_tokens() else 5
        w2_cfg = w13_cfg
    r2_ops = _ops_r2()
    w2_ks = None
    if (
        mul_topk_weights_in_gemm2
        and r2_ops is not None
        and dq in (0, 2, 3)
        and topk <= 8
        and _plain_moe_sum(moe_sum)
        and (topk <= 3 or M <= 2 or M <= _w2_splitk_max_tokens())
        and (
            output is None
            or (
                output.dtype == torch.bfloat16
                and output.is_contiguous()
                and tuple(output.shape) == (M, _K)
            )
        )
    ):
        if ni == 576:
            if M <= 2:
                w2_ks = int(os.environ.get("VLLM_DSV4_MXFP4_TP4_W2_KS_SMALL", "3"))
            else:
                w2_ks = int(
                    os.environ.get(
                        "VLLM_DSV4_MXFP4_TP4_W2_KS_MED",
                        "2" if (topk <= 3 or M <= 8) else "1",
                    )
                )
        else:
            w2_ks = 4 if (topk <= 3 or M <= 2) else _w2_splitk_ks(M)
    # W13 split-K partials [4, slots, 2*ni]; split-K W2 reuses the buffer as
    # [w2_ks, slots, K] after the activation consumed the W13 partials.
    part_numel = max(4 * slots * 2 * ni, (w2_ks or 0) * slots * _K)
    part = torch.empty(part_numel, dtype=torch.float32, device=dev)
    h = intermediate_cache2.reshape(-1)[: slots * ni].view(slots, ni)
    tw = topk_weights.view(-1)
    w1s = w1_scale.view(torch.uint8)
    w2s = w2_scale.view(torch.uint8)
    if limit is not None:
        if expert_map is None:
            ops2.gemm(
                hidden_states,
                w1,
                w1s,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                tw,
                topk,
                slots,
                _K,
                2 * ni,
                True,
                w13_cfg,
                dq,
                True,
                limit,
                0,
                ctr,
                sync,
                part,
                h,
            )
        else:
            ops2.gemm_ep(
                hidden_states,
                w1,
                w1s,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                tw,
                topk,
                slots,
                _K,
                2 * ni,
                True,
                w13_cfg,
                dq,
                True,
                limit,
                0,
                ctr,
                sync,
                part,
                h,
                topk_ids,
                expert_map,
                global_e,
            )
    else:
        dummy = torch.empty(0, dtype=torch.bfloat16, device=dev)
        if expert_map is None:
            ks = ops2.gemm(
                hidden_states,
                w1,
                w1s,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                tw,
                topk,
                slots,
                _K,
                2 * ni,
                True,
                w13_cfg,
                dq,
                False,
                0.0,
                0,
                ctr,
                sync,
                part,
                dummy,
            )
        else:
            ks = ops2.gemm_ep(
                hidden_states,
                w1,
                w1s,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                tw,
                topk,
                slots,
                _K,
                2 * ni,
                True,
                w13_cfg,
                dq,
                False,
                0.0,
                0,
                ctr,
                sync,
                part,
                dummy,
                topk_ids,
                expert_map,
                global_e,
            )
        act_in = intermediate_cache13.reshape(-1)[: slots * 2 * ni].view(slots, 2 * ni)
        ops.sum(part, ks, slots, 2 * ni, act_in)
        if activation_func is None:
            from vllm.model_executor.layers.fused_moe.activation import (
                ApplyMoEActivationConfig,
                apply_moe_activation,
            )

            apply_moe_activation(
                activation,
                h,
                act_in,
                activation_config=activation_config or ApplyMoEActivationConfig(),
                topk_ids=topk_ids,
                expert_map=expert_map,
            )
        else:
            activation_func(activation, h, act_in, topk_ids=topk_ids, expert_map=expert_map)
    out = intermediate_cache13.reshape(-1)[: slots * _K].view(slots, _K)
    if w2_ks is not None:
        # Split-K W2 + its own L2 top-k sum: the KS=1 W2 grid (n_experts * 80
        # items) cannot fill the GPU at decode sizes.
        y = output if output is not None else torch.empty_like(hidden_states)
        skey = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
        se_entry = _PENDING_SHARED.pop(skey, None)
        bias_bf16 = None
        if se_entry is not None:
            se, se_overlapping = se_entry
            se_out = se._output[se._output_idx]
            if (
                se_out is not None
                and se_out.dtype == torch.bfloat16
                and se_out.is_contiguous()
                and tuple(se_out.shape) == (M, _K)
            ):
                if se_overlapping:
                    se.wait()
                bias_bf16 = se_out
        part_w2 = part[: w2_ks * slots * _K]
        can_defer = (
            _DEFER_W2_SUM.get(skey, False)
            and os.environ.get("VLLM_DSV41_FUSED_W2_AR", "0") == "1"
            and 3 <= M <= 32
            and w2_ks in (3, 4)
            and topk in (3, 6)
            and bias_bf16 is not None
            and expert_map is not None
            and topk_ids is not None
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
            and expert_map.is_contiguous()
            and tw.is_contiguous()
        )
        y_arg = torch.empty(0, dtype=torch.bfloat16, device=dev) if can_defer else y
        r2_ops.w2_splitk_ep(
            h,
            w2,
            w2s,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            tw,
            topk_ids,
            expert_map,
            topk,
            global_e,
            w2_ks,
            dq,
            ctr,
            part_w2,
            y_arg,
            None if can_defer else bias_bf16,
        )
        if can_defer:
            _DEFERRED_W2_SPLITK[skey] = (
                y.data_ptr(),
                part_w2,
                tw,
                topk_ids,
                expert_map,
                bias_bf16,
                w2_ks,
                topk,
                global_e,
            )
        if bias_bf16 is not None:
            _FUSED_SHARED_OUT[skey] = (y.data_ptr(), bias_bf16.data_ptr())
        return y, True
    if fsum:
        y = output if output is not None else torch.empty_like(hidden_states)
        if expert_map is None:
            ops2.gemm(
                h,
                w2,
                w2s,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                tw,
                topk,
                slots,
                ni,
                _K,
                False,
                w2_cfg,
                dq,
                True,
                0.0,
                s_order,
                ctr,
                sync,
                out,
                y,
            )
        else:
            ops2.gemm_ep(
                h,
                w2,
                w2s,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                tw,
                topk,
                slots,
                ni,
                _K,
                False,
                w2_cfg,
                dq,
                True,
                0.0,
                s_order,
                ctr,
                sync,
                out,
                y,
                topk_ids,
                expert_map,
                global_e,
            )
        return y, True
    dummy = torch.empty(0, dtype=torch.bfloat16, device=dev)
    tw_w2 = tw if mul_topk_weights_in_gemm2 else torch.empty(0, dtype=torch.float32, device=dev)
    if expert_map is None:
        ops2.gemm(
            h,
            w2,
            w2s,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            tw_w2,
            topk,
            slots,
            ni,
            _K,
            False,
            w2_cfg,
            dq,
            False,
            0.0,
            0,
            ctr,
            sync,
            out,
            dummy,
        )
    else:
        ops2.gemm_ep(
            h,
            w2,
            w2s,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            tw_w2,
            topk,
            slots,
            ni,
            _K,
            False,
            w2_cfg,
            dq,
            False,
            0.0,
            0,
            ctr,
            sync,
            out,
            dummy,
            topk_ids,
            expert_map,
            global_e,
        )
    return out, False
