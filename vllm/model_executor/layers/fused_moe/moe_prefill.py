# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 MXFP4 split-list prefill; opt-in dispatch lives in marlin_moe.py.

Gated on VLLM_DSV41_MOE_PREFILL_SPLIT=1 (off by default on TEP4).
Same Marlin kernels, activation callback and caller's ordered sum. Not bitwise
identical: stream-K changes. Only immutable cover tables are cached; writable
scratch is per call (graph-private under capture). Missing tables in capture
fall back before writing output.
"""

import torch

from . import moe_split_align as split

_TABLES = {}

_MIN_M = {384: 512, 128: 1024, 96: 1024, 32: 2048}
_SUPPORTED_TOPK = (3, 6)


def eligible(
    x,
    w1,
    w2,
    sc1,
    sc2,
    ids,
    qt,
    bias1,
    bias2,
    expert_map,
    input_dtype=None,
    global_scale1=None,
    global_scale2=None,
    input_global_scale1=None,
    input_global_scale2=None,
    w1_zeros=None,
    w2_zeros=None,
    apply_router_weight_on_input=False,
):
    from vllm.scalar_type import scalar_types

    if not (
        x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and x.ndim == 2
    ):
        return False
    E = w1.shape[0] if w1.ndim == 3 else -1
    min_m = _MIN_M.get(E)
    if min_m is None:
        return False
    ni = w2.shape[1] * 16 if w2.ndim == 3 else -1
    if ni not in (2304, 1152, 576):
        return False
    return (
        min_m <= x.shape[0] <= 8192
        and x.shape[1] == 5120
        and tuple(w1.shape) == (E, 320, 4 * ni)
        and tuple(w2.shape) == (E, ni // 16, 10240)
        and ids.ndim == 2
        and ids.shape[0] == x.shape[0]
        and ids.shape[1] in _SUPPORTED_TOPK
        and ids.dtype in (torch.int32, torch.int64)
        and ids.is_cuda
        and qt == scalar_types.float4_e2m1f
        and sc1.dtype == torch.float8_e8m0fnu
        and sc2.dtype == torch.float8_e8m0fnu
        and tuple(sc1.shape) == (E, 160, 2 * ni)
        and tuple(sc2.shape) == (E, ni // 32, 5120)
        and (
            expert_map is None
            or (
                expert_map.is_cuda
                and expert_map.ndim == 1
                and expert_map.is_contiguous()
                and expert_map.dtype == torch.int32
                and expert_map.numel() >= E
            )
        )
        and all(
            v is None
            for v in (
                bias1,
                bias2,
                input_dtype,
                global_scale1,
                global_scale2,
                input_global_scale1,
                input_global_scale2,
                w1_zeros,
                w2_zeros,
            )
        )
        and not apply_router_weight_on_input
        and w1.is_contiguous()
        and w2.is_contiguous()
        and torch.cuda.get_device_capability(x.device) == (8, 0)
    )


def lists(ids, E=384, expert_map=None):
    key = (ids.device, ids.numel())
    entry = _TABLES.get(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        table = split.cover_table(ids.numel()).to(ids.device)
        torch.cuda.current_stream(ids.device).synchronize()
        _TABLES[key] = table
    else:
        table = entry
    T = ids.numel()

    def empty(n):
        return torch.empty(n, device=ids.device, dtype=torch.int32)

    buf = {
        "t_max": T,
        "E": E,
        "tab": table,
        "hist": empty(((T + split.CHUNK - 1) // split.CHUNK) * E),
        "info": empty(8 * E),
        "ntpp": empty(4),
        "s": [empty(T + bs * E) for bs in split.SIZES],
        "e": [empty((T + bs * E) // bs + 1) for bs in split.SIZES],
    }
    return split.split_align(ids, E, buf, expert_map=expert_map)


def split_gemm(block_lists, *args, **kwargs):
    from vllm import _custom_ops as ops

    args = list(args)
    for bs, sorted_ids, expert_ids, ntpp in block_lists:
        args[8:11] = [sorted_ids, expert_ids, ntpp]
        kwargs["moe_block_size"] = bs
        ops.moe_wna16_marlin_gemm(*args, **kwargs)
    return args[1]
