# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split MoE block alignment for the Marlin W4A16 MoE at whole-expert prefill."""

import torch

from vllm.triton_utils import tl, triton

SIZES = (64, 48, 32, 16)
COST = {64: 33.3, 48: 26.4, 32: 20.3, 16: 14.1}
CHUNK = 128


@triton.jit(do_not_specialize=["T", "E", "GE"])
def _hist_kernel(
    ids_ptr,
    emap_ptr,
    hist_ptr,
    T,
    E,
    GE,
    HAS_MAP: tl.constexpr,
    C: tl.constexpr,
    NB: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * C + tl.arange(0, C)
    in_t = offs < T
    raw_e = tl.load(ids_ptr + offs, mask=in_t, other=-1)
    if HAS_MAP:
        in_g = in_t & (raw_e >= 0) & (raw_e < GE)
        e = tl.load(emap_ptr + raw_e, mask=in_g, other=-1)
    else:
        e = raw_e
    e = tl.where(in_t & (e >= 0) & (e < E), e, E)
    h = tl.histogram(e, NB)
    b = tl.arange(0, NB)
    tl.store(hist_ptr + pid * E + b, h, mask=b < E)


@triton.jit(do_not_specialize=["nch", "E"])
def _prefix_kernel(hist_ptr, info_ptr, nch, E, EB: tl.constexpr, CB: tl.constexpr):
    """Exclusive prefix of the per-chunk histograms over chunks (in place), for EB
    experts per program; totals r -> info[7*E + e]."""
    eb = tl.program_id(0) * EB + tl.arange(0, EB)
    em = eb < E
    carry = tl.zeros([EB], dtype=tl.int32)
    for c0 in range(0, nch, CB):
        c = c0 + tl.arange(0, CB)
        m = (c[:, None] < nch) & em[None, :]
        ptr = hist_ptr + c[:, None] * E + eb[None, :]
        h = tl.load(ptr, mask=m, other=0)
        inc = tl.cumsum(h, 0)
        tl.store(ptr, inc - h + carry[None, :], mask=m)
        carry += tl.sum(h, 0)
    tl.store(info_ptr + 7 * E + eb, carry, mask=em)


@triton.jit(do_not_specialize=["E"])
def _plan_kernel(
    E,
    tab_ptr,
    info_ptr,
    ntpp_ptr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    NB: tl.constexpr,
):
    """Per expert: block counts from the cover table, list offsets (prefix over
    experts), rank boundaries; list sizes -> ntpp."""
    b = tl.arange(0, NB)
    em = b < E
    r = tl.load(info_ptr + 7 * E + b, mask=em, other=0)
    n0 = tl.load(tab_ptr + r * 4 + 0, mask=em, other=0)
    n1 = tl.load(tab_ptr + r * 4 + 1, mask=em, other=0)
    n2 = tl.load(tab_ptr + r * 4 + 2, mask=em, other=0)
    n3 = tl.load(tab_ptr + r * 4 + 3, mask=em, other=0)
    rows0 = n0 * B0
    rows1 = n1 * B1
    rows2 = n2 * B2
    rows3 = n3 * B3
    tl.store(info_ptr + 0 * E + b, tl.cumsum(rows0, 0) - rows0, mask=em)
    tl.store(info_ptr + 1 * E + b, tl.cumsum(rows1, 0) - rows1, mask=em)
    tl.store(info_ptr + 2 * E + b, tl.cumsum(rows2, 0) - rows2, mask=em)
    tl.store(info_ptr + 3 * E + b, tl.cumsum(rows3, 0) - rows3, mask=em)
    tl.store(info_ptr + 4 * E + b, rows0, mask=em)
    tl.store(info_ptr + 5 * E + b, rows0 + rows1, mask=em)
    tl.store(info_ptr + 6 * E + b, rows0 + rows1 + rows2, mask=em)
    tl.store(ntpp_ptr + 0, tl.sum(rows0, 0))
    tl.store(ntpp_ptr + 1, tl.sum(rows1, 0))
    tl.store(ntpp_ptr + 2, tl.sum(rows2, 0))
    tl.store(ntpp_ptr + 3, tl.sum(rows3, 0))


@triton.jit
def _fill_list(
    e,
    r,
    off,
    lo,
    hi,
    is_last,
    T,
    s_ptr,
    e_ptr,
    BS: tl.constexpr,
    JB: tl.constexpr,
):
    rows = hi - lo
    pad = tl.where(is_last, hi - r, 0)
    j = tl.arange(0, 64)
    tl.store(s_ptr + off + rows - pad + j, T, mask=j < pad)
    nblk = rows // BS
    for j0 in range(0, nblk, JB):
        jb = j0 + tl.arange(0, JB)
        tl.store(e_ptr + off // BS + jb, e, mask=jb < nblk)


@triton.jit(do_not_specialize=["E", "T"])
def _fill_kernel(
    info_ptr,
    E,
    T,
    s0,
    s1,
    s2,
    s3,
    e0,
    e1,
    e2,
    e3,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
):
    """One program per expert: sentinel padding of its last block and its expert
    ids. Only the last (smallest) nonempty list of an expert is padded."""
    e = tl.program_id(0)
    r = tl.load(info_ptr + 7 * E + e)
    lo1 = tl.load(info_ptr + 4 * E + e)
    lo2 = tl.load(info_ptr + 5 * E + e)
    lo3 = tl.load(info_ptr + 6 * E + e)
    hi3 = lo3 + (r + B3 - 1 - lo3) // B3 * B3 * (r > lo3).to(tl.int32)
    last3 = hi3 > lo3
    last2 = (lo3 > lo2) & ~last3
    last1 = (lo2 > lo1) & (lo3 == lo2) & ~last3
    last0 = (lo1 > 0) & (lo2 == lo1) & (lo3 == lo2) & ~last3
    _fill_list(e, r, tl.load(info_ptr + e), 0, lo1, last0, T, s0, e0, B0, 64)
    _fill_list(e, r, tl.load(info_ptr + E + e), lo1, lo2, last1, T, s1, e1, B1, 64)
    _fill_list(e, r, tl.load(info_ptr + 2 * E + e), lo2, lo3, last2, T, s2, e2, B2, 64)
    _fill_list(e, r, tl.load(info_ptr + 3 * E + e), lo3, hi3, last3, T, s3, e3, B3, 64)


@triton.jit(do_not_specialize=["T", "E", "GE"])
def _scatter_kernel(
    ids_ptr,
    emap_ptr,
    hist_ptr,
    info_ptr,
    s0,
    s1,
    s2,
    s3,
    T,
    E,
    GE,
    HAS_MAP: tl.constexpr,
    C: tl.constexpr,
):
    pid = tl.program_id(0)
    i = tl.arange(0, C)
    offs = pid * C + i
    valid = offs < T
    raw_e = tl.load(ids_ptr + offs, mask=valid, other=-1)
    if HAS_MAP:
        in_g = valid & (raw_e >= 0) & (raw_e < GE)
        e = tl.load(emap_ptr + raw_e, mask=in_g, other=-1)
        valid = in_g & (e >= 0) & (e < E)
    else:
        e = raw_e
        valid = valid & (e >= 0) & (e < E)
    e = tl.where(valid, e, E + 1 + i)
    eq = (e[:, None] == e[None, :]) & (i[None, :] < i[:, None])
    rank_in = tl.sum(eq.to(tl.int32), 1)
    es = tl.where(valid, e, 0)
    rank = tl.load(hist_ptr + pid * E + es, mask=valid, other=0) + rank_in
    lo1 = tl.load(info_ptr + 4 * E + es, mask=valid, other=0)
    lo2 = tl.load(info_ptr + 5 * E + es, mask=valid, other=0)
    lo3 = tl.load(info_ptr + 6 * E + es, mask=valid, other=0)
    in3 = rank >= lo3
    in2 = (rank >= lo2) & ~in3
    in1 = (rank >= lo1) & ~in3 & ~in2
    in0 = ~in3 & ~in2 & ~in1
    o0 = tl.load(info_ptr + 0 * E + es, mask=valid & in0, other=0)
    o1 = tl.load(info_ptr + 1 * E + es, mask=valid & in1, other=0)
    o2 = tl.load(info_ptr + 2 * E + es, mask=valid & in2, other=0)
    o3 = tl.load(info_ptr + 3 * E + es, mask=valid & in3, other=0)
    tl.store(s0 + o0 + rank, offs, mask=valid & in0)
    tl.store(s1 + o1 + rank - lo1, offs, mask=valid & in1)
    tl.store(s2 + o2 + rank - lo2, offs, mask=valid & in2)
    tl.store(s3 + o3 + rank - lo3, offs, mask=valid & in3)


def cover_table(R: int) -> torch.Tensor:
    """[R+1, len(SIZES)] int32 (CPU): blocks of each size in the cheapest cover
    of r rows, for r = 0..R."""
    best = [0.0] * (R + 1)
    pick = [0] * (R + 1)
    for r in range(1, R + 1):
        c, p = min((COST[s] + best[max(r - s, 0)], s) for s in SIZES)
        best[r], pick[r] = c, p
    tab = [[0] * len(SIZES) for _ in range(R + 1)]
    for r in range(1, R + 1):
        tab[r] = list(tab[max(r - pick[r], 0)])
        tab[r][SIZES.index(pick[r])] += 1
    return torch.tensor(tab, dtype=torch.int32)


def buffers(t_max: int, E: int, device: torch.device) -> dict:
    """All scratch for ``split_align`` at up to ``t_max`` = M*topk routed rows."""
    nch = triton.cdiv(t_max, CHUNK)
    return {
        "t_max": t_max,
        "E": E,
        "hist": torch.empty(nch * E, device=device, dtype=torch.int32),
        "info": torch.empty(8 * E, device=device, dtype=torch.int32),
        "tab": cover_table(t_max).to(device),
        "s": [
            torch.empty(t_max + bs * E, device=device, dtype=torch.int32)
            for bs in SIZES
        ],
        "e": [
            torch.empty((t_max + bs * E) // bs + 1, device=device, dtype=torch.int32)
            for bs in SIZES
        ],
        "ntpp": torch.empty(len(SIZES), device=device, dtype=torch.int32),
    }


def split_align(
    topk_ids: torch.Tensor,
    E: int,
    buf: dict,
    expert_map: torch.Tensor | None = None,
):
    """-> [(block_size, sorted_ids, expert_ids, ntpp_1elem)], one per size in
    SIZES, into the fixed buffers ``buf`` (from ``buffers``)."""
    ids = topk_ids.reshape(-1)
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    T = ids.numel()
    assert 0 < T <= buf["t_max"] and E == buf["E"], "split_align buffers too small"
    has_map = expert_map is not None
    if has_map:
        assert (
            expert_map.is_cuda
            and expert_map.is_contiguous()
            and expert_map.dtype == torch.int32
        ), "expert_map must be contiguous CUDA int32"
        ge = expert_map.numel()
        emap_arg = expert_map
    else:
        ge = E
        emap_arg = ids
    nch = triton.cdiv(T, CHUNK)
    NB = max(triton.next_power_of_2(E + 1), 16)
    s, e, ntpp = buf["s"], buf["e"], buf["ntpp"]
    _hist_kernel[(nch,)](
        ids, emap_arg, buf["hist"], T, E, ge, HAS_MAP=has_map, C=CHUNK, NB=NB
    )
    B = dict(B0=SIZES[0], B1=SIZES[1], B2=SIZES[2], B3=SIZES[3])
    _prefix_kernel[(triton.cdiv(E, 32),)](
        buf["hist"], buf["info"], nch, E, EB=32, CB=64, num_warps=4
    )
    _plan_kernel[(1,)](E, buf["tab"], buf["info"], ntpp, NB=NB, num_warps=4, **B)
    _fill_kernel[(E,)](buf["info"], E, T, *s, *e, num_warps=1, **B)
    _scatter_kernel[(nch,)](
        ids,
        emap_arg,
        buf["hist"],
        buf["info"],
        *s,
        T,
        E,
        ge,
        HAS_MAP=has_map,
        C=CHUNK,
        num_warps=4,
    )
    return [(bs, s[k], e[k], ntpp[k : k + 1]) for k, bs in enumerate(SIZES)]
