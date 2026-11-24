# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-kernel replacement for _moe_align_block_size_deterministic.

VLLM_DSV4_FAST_DET_MOE_ALIGN=1 (default 0 = unchanged). Produces exactly the
same sorted_ids / expert_ids / num_tokens_post_pad as the torch implementation
(entries ordered by (expert, flat index), invalid entries dropped, unused
sorted_ids slots = numel, blocks past the padded total = -1), in one Triton
program instead of ~25 small ATen kernels per MoE layer. Integer-only and
CUDA-graph capturable; no atomics, so the layout is deterministic.
Supports optional expert_map for EP; used for at most MAX_ENTRIES routed entries;
otherwise the caller keeps the fallback path.
"""

import os

import torch

from vllm.triton_utils import tl, triton

MAX_ENTRIES = 2048
MAX_EXPERTS = 1024
_CH = 64


@triton.jit
def _det_align_kernel(
    ids,
    emap,
    sorted_ids,
    expert_ids,
    ntpp,
    S,
    GE,
    LE,
    BS,
    CAP,
    NB,
    HAS_MAP: tl.constexpr,
    EP: tl.constexpr,
    CH: tl.constexpr,
    FILL: tl.constexpr,
):
    es = tl.arange(0, EP)
    j = tl.arange(0, CH)
    counts = tl.zeros([EP], dtype=tl.int32)
    for c0 in range(0, S, CH):
        i = c0 + j
        raw_e = tl.load(ids + i, mask=i < S, other=-1).to(tl.int64)
        in_g = (i < S) & (raw_e >= 0) & (raw_e < GE)
        if HAS_MAP:
            e = tl.load(emap + raw_e, mask=in_g, other=-1).to(tl.int32)
        else:
            e = tl.where(in_g, raw_e, -1).to(tl.int32)
        valid = in_g & (e >= 0) & (e < LE)
        hit = (es[:, None] == e[None, :]) & valid[None, :]
        counts += tl.sum(hit.to(tl.int32), 1)
    padded = ((counts + BS - 1) // BS) * BS
    start = tl.cumsum(padded, 0) - padded
    total = tl.sum(padded, 0)
    for p0 in range(0, CAP, FILL):
        p = p0 + tl.arange(0, FILL)
        tl.store(sorted_ids + p, S, mask=p < CAP)
    tl.debug_barrier()
    running = tl.zeros([EP], dtype=tl.int32)
    for c0 in range(0, S, CH):
        i = c0 + j
        raw_e = tl.load(ids + i, mask=i < S, other=-1).to(tl.int64)
        in_g = (i < S) & (raw_e >= 0) & (raw_e < GE)
        if HAS_MAP:
            e = tl.load(emap + raw_e, mask=in_g, other=-1).to(tl.int32)
        else:
            e = tl.where(in_g, raw_e, -1).to(tl.int32)
        valid = in_g & (e >= 0) & (e < LE)
        hit = (es[:, None] == e[None, :]) & valid[None, :]
        base = tl.sum(tl.where(hit, (start + running)[:, None], 0), 0)
        before = (
            (e[:, None] == e[None, :])
            & (j[None, :] < j[:, None])
            & valid[None, :]
        )
        rank = tl.sum(before.to(tl.int32), 1)
        tl.store(sorted_ids + base + rank, i, mask=valid & (base + rank < CAP))
        running += tl.sum(hit.to(tl.int32), 1)
    # Blocks past the padded total are inactive; each expert then writes its
    # own id over its [start, start + padded) block range (disjoint ranges).
    for b0 in range(0, NB, FILL):
        b = b0 + tl.arange(0, FILL)
        tl.store(expert_ids + b, -1, mask=(b < NB) & (b * BS >= total))
    nblk = padded // BS
    maxblk = tl.max(nblk, 0)
    sblk = start // BS
    live = es < LE
    for k in range(0, maxblk):
        tl.store(expert_ids + sblk + k, es, mask=live & (k < nblk) & (sblk + k < NB))
    tl.store(ntpp, total)


def enabled() -> bool:
    return os.environ.get("VLLM_DSV4_FAST_DET_MOE_ALIGN", "0") == "1"


def try_align(
    topk_ids,
    num_experts,
    block_size,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad,
    expert_map,
    local_num_experts=None,
) -> bool:
    """Fill the outputs and return True, or return False to keep fallback."""
    if not topk_ids.is_cuda:
        return False
    if not topk_ids.is_contiguous() or topk_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return False
    s = topk_ids.numel()
    local_e = num_experts if local_num_experts is None else local_num_experts
    if s == 0 or s > MAX_ENTRIES or num_experts <= 0 or not 0 < local_e <= MAX_EXPERTS:
        return False
    has_map = expert_map is not None
    if has_map and (
        not expert_map.is_cuda
        or not expert_map.is_contiguous()
        or expert_map.dtype != torch.int32
        or expert_map.numel() < num_experts
    ):
        return False
    if not (
        sorted_ids.is_contiguous()
        and expert_ids.is_contiguous()
        and sorted_ids.dtype == expert_ids.dtype == torch.int32
        and num_tokens_post_pad.dtype == torch.int32
    ):
        return False
    ep = triton.next_power_of_2(local_e)
    emap_arg = expert_map if has_map else sorted_ids
    with torch.cuda.device(topk_ids.device):
        _det_align_kernel[(1,)](
            topk_ids,
            emap_arg,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            s,
            num_experts,
            local_e,
            block_size,
            sorted_ids.numel(),
            expert_ids.numel(),
            HAS_MAP=has_map,
            EP=ep,
            CH=_CH,
            FILL=1024,
            num_warps=8,
        )
    return True
