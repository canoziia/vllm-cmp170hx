# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TP all-reduce of DeepSeek-V4.1's decode boundaries on SM80, in CuTe DSL.

Every decode layer of DeepSeek-V4.1 all-reduces two partial sums across the
tensor-parallel ranks: the attention output projection's and the MoE's, one
BF16 row of `hidden` values per token (24 rows at four requests with five
drafts each, 245 KB at TP8). vLLM's custom all-reduce does it by loading:
below 256 KB at TP8 each rank reads every peer's whole buffer over NVLink
(one-shot), above it a reduce-scatter and an all-gather separated by
barriers (two-shot). On A100 both are latency-bound at decode sizes, about
28 us at 24 rows and 25 us at 48, several times what the bytes need.

`TwoShotAllReduce` pushes instead of loading, and synchronizes through the
data rather than with barriers (a Lamport mailbox): every slot of a rank's
mailbox holds a sentinel, BF16 negative zero, until a peer's store replaces
it, so a reader polls a slot until none of its values is the sentinel.
Senders turn real negative zeros into positive zeros. One CTA per row, one
16-byte fragment (8 values) per thread, three phases:

1. **Scatter.** Each thread stores its fragment of the local partial into
   the mailbox of the rank that owns the fragment's slice of the row (rank
   `s` owns fragments `s * hidden / (8 * world)` onward).
2. **Reduce.** The row's fragments of this rank's slice, times the `world`
   sources, are one per thread: groups of `world` lanes poll one fragment
   from every source, add them in FP32 with a butterfly (every lane gets the
   same bits), round to BF16, and each lane stores the sum into one rank's
   mailbox.
3. **Gather.** Each thread polls its fragment of the reduced row in its own
   mailbox and writes it to the output.

Each rank sends `2 (world - 1) / world` of the row instead of
`world - 1` rows, and no phase waits for anything but its own data.

A reader resets the slots it consumed to the sentinel. The mailbox has three
generations, and each row (CTA) keeps the generation its next call uses,
advanced modulo three by every call: a peer writes the same generation of a
row again three calls later, which needs this rank's push of two calls later,
issued after this call completed. All ranks run the same sequence of calls
with the same row counts, so a row's generation agrees across ranks; it is
per row because a call of fewer rows leaves the other rows' generations
alone. The mailbox starts as sentinels and the generations at zero, so the
kernel needs no per-launch initialization and is CUDA-graph safe.

A row's CTA waits for the same row's CTA on every peer, so all of a call's
CTAs must be resident at once on every rank. A call therefore launches at
most one CTA per SM, whatever the kernel's registers, and above that many
rows each CTA takes rows `ctas` apart in increasing order: the lowest
unfinished row is then in progress on every rank, so every call completes.
This also needs every kernel that runs beside a call to finish without
waiting on it, which holds for the only one, the side-stream mHC
coefficients.

`TwoShotAllReducePostCollapse` is the same all-reduce followed, in the
same CTA, by the mHC boundary's post-mix, collapse and RMSNorm
(`mhc.PostCollapse`, whose CTAs have the same shape: a row per CTA, eight
values per thread). The loads that do not depend on the sum (the row's
streams, mixes and norm weight) are issued before the scatter, so they
overlap the waits, and the reduced row never goes back to memory: the
boundary saves a launch and a round trip of the sublayer output.

`TwoShotAllReduceFinalizePostCollapse` takes a MoE output whose top-k
reduction is still open (`UnfinalizedMoEOutput` with one unweighted GEMM2
row per route, routes of a token adjacent): before the scatter, each thread
forms its fragment of the rank's partial row from the token's `top_k`
route rows, their routing weights and the shared expert's row. The MoE's
top-k sum and shared-expert add then need no kernels of their own, as on
SM90, whose fused all-reduce does the same.

Numerics: each rank's slice sums the `world` partials in FP32 in a fixed
butterfly order and rounds once to BF16, so every rank gets the same bits;
the order differs from vLLM's kernels, so results differ from theirs by FP32
rounding of the sum. The fused variant's results are bitwise those of the
all-reduce followed by `PostCollapse`. The finalize variant forms the
partial row in FP32, a fused multiply-add per route in route order plus the
shared row, rounded once to BF16 (as SM90's kernel), so it differs from
vLLM's own top-k sum by rounding.
"""

from functools import cache
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.common.ops.moe_tail_fusion.primitives import (
    LAMPORT_GENERATIONS,
    NEGATIVE_ZERO_BF16_PAIR,
    current_cu_stream,
    fragment_has_negative_zero,
    sanitize_negative_zero_u32,
    store_global_u32x4,
)
from vllm.platforms import current_platform

from .codecs import f32_from_bits, load_s32, load_u32, store_u32
from .mhc import (
    HC,
    PostCollapse,
    _select_first_stream,
    bf16_high,
    bf16_low,
    fma,
    pack,
    pack_bf16x2,
    unpack,
)

# BF16 values per fragment (one 16-byte store) and bytes per fragment.
VEC = 8
FRAGMENT_BYTES = 16
# The most rows a call takes: decode steps of up to 32 requests with five
# drafts each (B32 x 6 = 192 rows).
MAX_ROWS = 192


def _asm(result, operands, text, constraints, side_effects, loc, ip):
    """One inline PTX statement."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_u64(address: Int64, *, loc=None, ip=None) -> Int64:
    """One 64-bit word that the kernel only reads (a mailbox base)."""
    return Int64(
        _asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u64 $0, [$1];",
            "=l,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def load_fragment(address: Int64, *, volatile: bool = False, loc=None, ip=None):
    """A 16-byte fragment as four words; `volatile` for polling a slot that
    peers store into."""
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        volatile,
        loc,
        ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), loaded, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


def store_fragment(address: Int64, words: list) -> None:
    """Store a 16-byte fragment, to this rank's memory or a peer's."""
    store_global_u32x4(Int64(address), [Uint32(word) for word in words], volatile=True)


@dsl_user_op
def _store_rel_sys_v4(
    address: Int64,
    w0: Uint32,
    w1: Uint32,
    w2: Uint32,
    w3: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            Uint32(w2).ir_value(loc=loc, ip=ip),
            Uint32(w3).ir_value(loc=loc, ip=ip),
        ],
        "st.relaxed.sys.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        True,
        loc,
        ip,
    )


def store_fragment_ring(address: Int64, words: list) -> None:
    """System-scope relaxed 16-byte P2P store for coalesced push-ring."""
    _store_rel_sys_v4(
        Int64(address),
        Uint32(words[0]),
        Uint32(words[1]),
        Uint32(words[2]),
        Uint32(words[3]),
    )


@cute.jit
def poll(address: Int64, words: cute.Tensor):
    """Wait until the slot at `address` holds no sentinel, read it into
    `words` and reset it to the sentinel."""
    fragment = load_fragment(address, volatile=True)
    for i in cutlass.range_constexpr(4):
        words[i] = fragment[i]
    dirty = fragment_has_negative_zero(words)
    while dirty:
        fragment = load_fragment(address, volatile=True)
        for i in cutlass.range_constexpr(4):
            words[i] = fragment[i]
        dirty = fragment_has_negative_zero(words)
    sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
    store_fragment(address, [sentinel] * 4)


@cute.jit
def poll_no_reset(address: Int64, words: cute.Tensor):
    """Wait until the slot at `address` holds no sentinel and read it into
    `words`, deferring the sentinel reset off the critical path."""
    fragment = load_fragment(address, volatile=True)
    for i in cutlass.range_constexpr(4):
        words[i] = fragment[i]
    dirty = fragment_has_negative_zero(words)
    while dirty:
        fragment = load_fragment(address, volatile=True)
        for i in cutlass.range_constexpr(4):
            words[i] = fragment[i]
        dirty = fragment_has_negative_zero(words)


class TwoShotAllReduce:
    """The push-based two-shot Lamport all-reduce; see the module docstring.

    `world` ranks, rows of `hidden` BF16 values, at most `capacity`
    rows a call.
    """

    def __init__(self, hidden: int, world: int, capacity: int):
        assert hidden % (VEC * world) == 0
        self.hidden = hidden
        self.world = world
        self.capacity = capacity
        # Fragments of a row (threads per CTA) and of a rank's slice.
        self.fragments = hidden // VEC
        self.slice_fragments = self.fragments // world
        # Byte offset of the gather slots after the scatter slots.
        self.gather_base = LAMPORT_GENERATIONS * capacity * hidden * 2

    def mailbox_bytes(self) -> int:
        """Bytes of a rank's mailbox: scatter and gather slots, three
        generations each."""
        return 2 * self.gather_base

    @cute.jit
    def __call__(
        self,
        x: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sc: cute.struct.MemRange[Uint32, self.fragments * 4]

        self.kernel(x, out, peers, generations, rank, rows, Storage).launch(
            grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        rows: Int32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSc = storage.sc.get_tensor(cute.make_layout(self.fragments * 4))
        ctas, _, _ = cute.arch.grid_dim()
        local = load_u64(peers + Int64(rank) * 8)
        owner = tid // self.slice_fragments
        owner_peer = load_u64(peers + Int64(owner) * 8)
        frag_c = tid % self.slice_fragments
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        # Rows in increasing order, ``ctas`` apart (see the module docstring).
        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            partial = load_fragment(
                x + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            self.scatter_coalesced(
                partial, owner, owner_peer, rank, generation, row, tid, frag_c, sSc
            )
            ga_addr = self.reduce_gather_coalesced(
                local, owner, owner_peer, rank, generation, row, tid, frag_c, sSc, words, sentinel
            )
            store_fragment(
                out + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES,
                [words[i] for i in range(4)],
            )
            if owner != rank:
                store_fragment(ga_addr, [sentinel] * 4)
            # Every thread has read the generation; advance it for the next call.
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas

    @cute.jit
    def scatter_coalesced(
        self,
        partial: list,
        owner: Int32,
        owner_peer: Int64,
        rank: Int32,
        generation: Int32,
        row: Int32,
        tid: Int32,
        frag_c: Int32,
        sSc: cute.Tensor,
    ):
        """Phase 1: 512B coalesced scatter push to remote peers ONLY; local slice stays in SMEM."""
        sanitized = [sanitize_negative_zero_u32(word) for word in partial]
        if owner != rank:
            store_fragment_ring(
                owner_peer + self.scatter_slot(generation, rank, row, frag_c),
                sanitized,
            )
        else:
            for i in cutlass.range_constexpr(4):
                sSc[tid * 4 + i] = sanitized[i]

    @cute.jit
    def reduce_gather_coalesced(
        self,
        local: Int64,
        owner: Int32,
        owner_peer: Int64,
        rank: Int32,
        generation: Int32,
        row: Int32,
        tid: Int32,
        frag_c: Int32,
        sSc: cute.Tensor,
        words: cute.Tensor,
        sentinel: Uint32,
    ) -> Int64:
        """Phase 2+3: 512B coalesced poll + exact tree sum + 512B coalesced gather (Skip-Local)."""
        sc_addr = local + self.scatter_slot(generation, owner, row, frag_c)
        if owner != rank:
            poll_no_reset(sc_addr, words)
            for i in cutlass.range_constexpr(4):
                sSc[tid * 4 + i] = words[i]
        cute.arch.barrier()
        s0 = unpack([sSc[(0 * self.slice_fragments + frag_c) * 4 + i] for i in range(4)])
        s1 = unpack([sSc[(1 * self.slice_fragments + frag_c) * 4 + i] for i in range(4)])
        s01 = [s0[e] + s1[e] for e in range(VEC)]
        if cutlass.const_expr(self.world >= 4):
            s2 = unpack([sSc[(2 * self.slice_fragments + frag_c) * 4 + i] for i in range(4)])
            s3 = unpack([sSc[(3 * self.slice_fragments + frag_c) * 4 + i] for i in range(4)])
            s23 = [s2[e] + s3[e] for e in range(VEC)]
            s0123 = [s01[e] + s23[e] for e in range(VEC)]
            if cutlass.const_expr(self.world == 8):
                s4 = unpack([sSc[(4 * self.slice_fragments + frag_c) * 4 + i] for i in range(4)])
                s5 = unpack([sSc[(5 * self.slice_fragments + frag_c) * 4 + i] for i in range(4)])
                s45 = [s4[e] + s5[e] for e in range(VEC)]
                s6 = unpack([sSc[(6 * self.slice_fragments + frag_c) * 4 + i] for i in range(4)])
                s7 = unpack([sSc[(7 * self.slice_fragments + frag_c) * 4 + i] for i in range(4)])
                s67 = [s6[e] + s7[e] for e in range(VEC)]
                s4567 = [s45[e] + s67[e] for e in range(VEC)]
                sums = [s0123[e] + s4567[e] for e in range(VEC)]
            else:
                sums = s0123
        else:
            sums = s01
        reduced = [
            sanitize_negative_zero_u32(pack_bf16x2(sums[2 * i], sums[2 * i + 1]))
            for i in range(4)
        ]
        if owner != rank:
            store_fragment_ring(
                owner_peer
                + self.gather_slot(generation, row, rank * self.slice_fragments + frag_c),
                reduced,
            )
            store_fragment(sc_addr, [sentinel] * 4)
        ga_addr = local + self.gather_slot(generation, row, tid)
        if owner != rank:
            poll_no_reset(ga_addr, words)
        else:
            for i in cutlass.range_constexpr(4):
                words[i] = reduced[i]
        return ga_addr

    @cute.jit
    def reduce_row(
        self,
        partial: list,
        peers: Int64,
        rank: Int32,
        generation: Int32,
        row: Int32,
        tid: Int32,
        words: cute.Tensor,
    ):
        """The three phases for row `row` (one CTA), from thread `tid`'s
        fragment of this rank's partial row (four packed words), leaving its
        fragment of the reduced row in `words`."""
        world = self.world
        slice_fragments = self.slice_fragments
        local = load_u64(peers + Int64(rank) * 8)

        # 1. Scatter: fragment tid of the row to the owner of its slice, at
        # scatter slot [generation][source rank][row][fragment in slice].
        owner = tid // slice_fragments
        store_fragment(
            load_u64(peers + Int64(owner) * 8)
            + self.scatter_slot(generation, rank, row, tid % slice_fragments),
            [sanitize_negative_zero_u32(word) for word in partial],
        )

        # 2. Reduce: lane group tid // world takes fragment tid // world of
        # this rank's slice, lane tid % world its copy from that source.
        source = tid % world
        poll(local + self.scatter_slot(generation, source, row, tid // world), words)
        sums = [
            bf16_low(words[i // 2]) if i % 2 == 0 else bf16_high(words[i // 2])
            for i in range(VEC)
        ]
        offset = 1
        for _ in cutlass.range_constexpr(world.bit_length() - 1):
            sums = [
                value + cute.arch.shuffle_sync_bfly(value, offset=offset)
                for value in sums
            ]
            offset *= 2
        reduced = [
            sanitize_negative_zero_u32(pack_bf16x2(sums[2 * i], sums[2 * i + 1]))
            for i in range(4)
        ]
        # The reduced fragment goes to rank `source` (each lane of the group
        # serves one rank), at gather slot [generation][row][fragment].
        store_fragment(
            load_u64(peers + Int64(source) * 8)
            + self.gather_slot(generation, row, rank * slice_fragments + tid // world),
            reduced,
        )

        # 3. Gather: this thread's fragment of the reduced row.
        poll(local + self.gather_slot(generation, row, tid), words)

    @cute.jit
    def scatter_slot(
        self, generation: Int32, source: Int32, row: Int32, fragment: Int32
    ) -> Int64:
        """Byte offset of a scatter slot in a mailbox."""
        return (
            ((Int64(generation) * self.world + source) * self.capacity + row)
            * self.slice_fragments
            + fragment
        ) * FRAGMENT_BYTES

    @cute.jit
    def gather_slot(self, generation: Int32, row: Int32, fragment: Int32) -> Int64:
        """Byte offset of a gather slot in a mailbox."""
        return (
            self.gather_base
            + ((Int64(generation) * self.capacity + row) * self.fragments + fragment)
            * FRAGMENT_BYTES
        )


class TwoShotAllReducePostCollapse(TwoShotAllReduce):
    """`TwoShotAllReduce` followed in the same CTA by `PostCollapse`
    (with the post-mix); see the module docstring."""

    def __init__(self, hidden: int, world: int, capacity: int, write_aux: bool):
        super().__init__(hidden, world, capacity)
        self.collapse = PostCollapse(hidden, has_post=True, write_aux=write_aux)

    @cute.jit
    def __call__(
        self,
        x: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]
            sc: cute.struct.MemRange[Uint32, self.fragments * 4]

        self.kernel(
            x,
            peers,
            generations,
            rank,
            residual,
            post,
            comb,
            pre,
            pre_stride,
            weight,
            residual_out,
            layer_input,
            aux,
            rows,
            eps,
            Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        sSc = storage.sc.get_tensor(cute.make_layout(self.fragments * 4))
        ctas, _, _ = cute.arch.grid_dim()
        local = load_u64(peers + Int64(rank) * 8)
        owner = tid // self.slice_fragments
        owner_peer = load_u64(peers + Int64(owner) * 8)
        frag_c = tid % self.slice_fragments
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        # Rows in increasing order, ``ctas`` apart (see the module docstring).
        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            partial = load_fragment(
                x + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            # Push 512B coalesced scatter immediately (Skip-Local) so PCIe flight overlaps load_row.
            self.scatter_coalesced(
                partial, owner, owner_peer, rank, generation, row, tid, frag_c, sSc
            )
            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )
            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            ga_addr = self.reduce_gather_coalesced(
                local, owner, owner_peer, rank, generation, row, tid, frag_c, sSc, words, sentinel
            )
            xs = unpack([words[i] for i in range(4)])
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            if owner != rank:
                store_fragment(ga_addr, [sentinel] * 4)
            # Every thread has read the generation; advance it for the next call.
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


class TwoShotAllReduceW2SplitKPostCollapse(TwoShotAllReducePostCollapse):
    """Fuses MXFP4 MoE W2 split-K + top-k + shared-expert reduction directly
    into push-first `TwoShotAllReducePostCollapse`, eliminating `w2_ks_moe_sum_ep`
    and its `[M, H]` BF16 round-trip."""

    def __init__(
        self,
        hidden: int,
        world: int,
        capacity: int,
        write_aux: bool,
        top_k: int,
        ks: int,
    ):
        super().__init__(hidden, world, capacity, write_aux)
        self.top_k = top_k
        self.ks = ks

    @cute.jit
    def __call__(
        self,
        part2: Int64,
        topk_w: Int64,
        topk_ids: Int64,
        emap: Int64,
        shared: Int64,
        global_e: Int32,
        ks_stride_bytes: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]

        self.kernel(
            part2,
            topk_w,
            topk_ids,
            emap,
            shared,
            global_e,
            ks_stride_bytes,
            peers,
            generations,
            rank,
            residual,
            post,
            comb,
            pre,
            pre_stride,
            weight,
            residual_out,
            layer_input,
            aux,
            rows,
            eps,
            Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        part2: Int64,
        topk_w: Int64,
        topk_ids: Int64,
        emap: Int64,
        shared: Int64,
        global_e: Int32,
        ks_stride_bytes: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        ctas, _, _ = cute.arch.grid_dim()
        local = load_u64(peers + Int64(rank) * 8)
        owner = tid // self.slice_fragments
        owner_peer = load_u64(peers + Int64(owner) * 8)
        source = tid % self.world
        source_peer = load_u64(peers + Int64(source) * 8)
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        top_k = self.top_k
        row_bytes_f32 = Int64(self.hidden) * 4
        col_bytes_f32 = Int64(tid) * 32
        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            shared_words = load_fragment(
                shared + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            acc = unpack(shared_words)
            for k in cutlass.range_constexpr(top_k):
                slot = row * top_k + k
                raw = load_s32(topk_ids + Int64(slot) * 4)
                valid = (raw >= Int32(0)) and (raw < global_e)
                if valid:
                    le = load_s32(emap + Int64(raw) * 4)
                    if le >= Int32(0):
                        w = f32_from_bits(load_u32(topk_w + Int64(slot) * 4))
                        base_p = part2 + Int64(slot) * row_bytes_f32 + col_bytes_f32
                        f0 = load_fragment(base_p)
                        f1 = load_fragment(base_p + Int64(16))
                        s = [
                            f32_from_bits(f0[0]),
                            f32_from_bits(f0[1]),
                            f32_from_bits(f0[2]),
                            f32_from_bits(f0[3]),
                            f32_from_bits(f1[0]),
                            f32_from_bits(f1[1]),
                            f32_from_bits(f1[2]),
                            f32_from_bits(f1[3]),
                        ]
                        for ks_i in cutlass.range_constexpr(1, self.ks):
                            p = base_p + Int64(ks_i) * ks_stride_bytes
                            g0 = load_fragment(p)
                            g1 = load_fragment(p + Int64(16))
                            s[0] = s[0] + f32_from_bits(g0[0])
                            s[1] = s[1] + f32_from_bits(g0[1])
                            s[2] = s[2] + f32_from_bits(g0[2])
                            s[3] = s[3] + f32_from_bits(g0[3])
                            s[4] = s[4] + f32_from_bits(g1[0])
                            s[5] = s[5] + f32_from_bits(g1[1])
                            s[6] = s[6] + f32_from_bits(g1[2])
                            s[7] = s[7] + f32_from_bits(g1[3])
                        acc = [fma(s[e], w, acc[e]) for e in range(VEC)]
            partial = pack(acc)
            store_fragment(
                owner_peer
                + self.scatter_slot(generation, rank, row, tid % self.slice_fragments),
                [sanitize_negative_zero_u32(word) for word in partial],
            )
            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )
            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            sc_addr = local + self.scatter_slot(generation, source, row, tid // self.world)
            poll_no_reset(sc_addr, words)
            sums = [
                bf16_low(words[i // 2]) if i % 2 == 0 else bf16_high(words[i // 2])
                for i in range(VEC)
            ]
            offset = 1
            for _ in cutlass.range_constexpr(self.world.bit_length() - 1):
                sums = [
                    value + cute.arch.shuffle_sync_bfly(value, offset=offset)
                    for value in sums
                ]
                offset *= 2
            reduced = [
                sanitize_negative_zero_u32(pack_bf16x2(sums[2 * i], sums[2 * i + 1]))
                for i in range(4)
            ]
            store_fragment(
                source_peer
                + self.gather_slot(
                    generation, row, rank * self.slice_fragments + tid // self.world
                ),
                reduced,
            )
            store_fragment(sc_addr, [sentinel] * 4)
            ga_addr = local + self.gather_slot(generation, row, tid)
            poll_no_reset(ga_addr, words)
            xs = unpack([words[i] for i in range(4)])
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            store_fragment(ga_addr, [sentinel] * 4)
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


class TwoShotAllReduceFinalizePostCollapse(TwoShotAllReducePostCollapse):
    """`TwoShotAllReducePostCollapse` of a MoE output with its top-k
    reduction open, `top_k` routes a token; see the module docstring."""

    def __init__(
        self, hidden: int, world: int, capacity: int, write_aux: bool, top_k: int
    ):
        super().__init__(hidden, world, capacity, write_aux)
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        routes: Int64,
        route_weights: Int64,
        shared: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]
            sc: cute.struct.MemRange[Uint32, self.fragments * 4]

        self.kernel(
            routes,
            route_weights,
            shared,
            peers,
            generations,
            rank,
            residual,
            post,
            comb,
            pre,
            pre_stride,
            weight,
            residual_out,
            layer_input,
            aux,
            rows,
            eps,
            Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        routes: Int64,
        route_weights: Int64,
        shared: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        sSc = storage.sc.get_tensor(cute.make_layout(self.fragments * 4))
        ctas, _, _ = cute.arch.grid_dim()
        local = load_u64(peers + Int64(rank) * 8)
        owner = tid // self.slice_fragments
        owner_peer = load_u64(peers + Int64(owner) * 8)
        frag_c = tid % self.slice_fragments
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        top_k = self.top_k
        # Rows in increasing order, ``ctas`` apart (see the module docstring).
        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            # Load only MoE route rows + weights + shared first so we can push
            # scatter ASAP and overlap PCIe flight with local collapse.load_row!
            route_words = [
                load_fragment(
                    routes
                    + ((Int64(row) * top_k + k) * self.fragments + tid) * FRAGMENT_BYTES
                )
                for k in range(top_k)
            ]
            scales = [
                f32_from_bits(load_u32(route_weights + (Int64(row) * top_k + k) * 4))
                for k in range(top_k)
            ]
            shared_words = load_fragment(
                shared + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            sums = unpack(route_words[0])
            sums = [value * scales[0] for value in sums]
            for k in cutlass.range_constexpr(1, top_k):
                values = unpack(route_words[k])
                sums = [fma(values[e], scales[k], sums[e]) for e in range(VEC)]
            shared_values = unpack(shared_words)
            partial = pack([sums[e] + shared_values[e] for e in range(VEC)])
            self.scatter_coalesced(
                partial, owner, owner_peer, rank, generation, row, tid, frag_c, sSc
            )
            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )
            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            ga_addr = self.reduce_gather_coalesced(
                local, owner, owner_peer, rank, generation, row, tid, frag_c, sSc, words, sentinel
            )
            xs = unpack([words[i] for i in range(4)])
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            if owner != rank:
                store_fragment(ga_addr, [sentinel] * 4)
            # Every thread has read the generation; advance it for the next call.
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


class OneShotAllReducePostCollapse:
    """One-shot push Lamport all-reduce fused with `PostCollapse` for tiny row
    counts (`rows <= 2`) on PCIe P2P, saving one full peer-to-peer PCIe hop
    versus two-shot scatter+gather while preserving bitwise-identical FP32
    butterfly reduction order."""

    def __init__(
        self,
        hidden: int,
        world: int,
        capacity: int,
        write_aux: bool,
        slot_base: int = 0,
    ):
        assert world in (2, 4, 8)
        self.hidden = hidden
        self.world = world
        self.capacity = capacity
        self.fragments = hidden // VEC
        self.slot_base = slot_base
        self.collapse = PostCollapse(hidden, has_post=True, write_aux=write_aux)

    def mailbox_bytes(self) -> int:
        return LAMPORT_GENERATIONS * self.world * self.capacity * self.hidden * 2

    @cute.jit
    def slot(
        self, generation: Int32, source: Int32, row: Int32, fragment: Int32
    ) -> Int64:
        return self.slot_base + (
            ((Int64(generation) * self.world + source) * self.capacity + row)
            * self.fragments
            + fragment
        ) * FRAGMENT_BYTES

    @cute.jit
    def __call__(
        self,
        x: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]

        self.kernel(
            x,
            peers,
            generations,
            rank,
            residual,
            post,
            comb,
            pre,
            pre_stride,
            weight,
            residual_out,
            layer_input,
            aux,
            rows,
            eps,
            Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        ctas, _, _ = cute.arch.grid_dim()
        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            partial = load_fragment(
                x + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            sanitized = [sanitize_negative_zero_u32(word) for word in partial]
            # Issue peer PCIe DMA stores first so they overlap local load_row.
            for p_off in cutlass.range_constexpr(1, self.world):
                dst = (rank + p_off) % self.world
                store_fragment(
                    load_u64(peers + Int64(dst) * 8)
                    + self.slot(generation, rank, row, tid),
                    sanitized,
                )
            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )
            local = load_u64(peers + Int64(rank) * 8)
            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            s01 = [Float32(0.0)] * VEC
            s23 = [Float32(0.0)] * VEC
            s45 = [Float32(0.0)] * VEC
            s67 = [Float32(0.0)] * VEC
            for src in cutlass.range_constexpr(self.world):
                vals = [Float32(0.0)] * VEC
                if src == rank:
                    vals = unpack(sanitized)
                else:
                    poll(local + self.slot(generation, src, row, tid), words)
                    vals = unpack([words[i] for i in range(4)])
                if cutlass.const_expr(src == 0):
                    s01 = vals
                elif cutlass.const_expr(src == 1):
                    s01 = [s01[e] + vals[e] for e in range(VEC)]
                elif cutlass.const_expr(src == 2):
                    s23 = vals
                elif cutlass.const_expr(src == 3):
                    s23 = [s23[e] + vals[e] for e in range(VEC)]
                elif cutlass.const_expr(src == 4):
                    s45 = vals
                elif cutlass.const_expr(src == 5):
                    s45 = [s45[e] + vals[e] for e in range(VEC)]
                elif cutlass.const_expr(src == 6):
                    s67 = vals
                elif cutlass.const_expr(src == 7):
                    s67 = [s67[e] + vals[e] for e in range(VEC)]
            sums = s01
            if cutlass.const_expr(self.world >= 4):
                sums = [sums[e] + s23[e] for e in range(VEC)]
            if cutlass.const_expr(self.world == 8):
                s47 = [s45[e] + s67[e] for e in range(VEC)]
                sums = [sums[e] + s47[e] for e in range(VEC)]
            reduced = [
                sanitize_negative_zero_u32(
                    pack_bf16x2(sums[2 * i], sums[2 * i + 1])
                )
                for i in range(4)
            ]
            xs = unpack(reduced)
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


class OneShotAllReduceFinalizePostCollapse(OneShotAllReducePostCollapse):
    """One-shot push Lamport all-reduce fused with MoE top-k finalize, shared-expert
    add, and `PostCollapse` for tiny row counts (`rows <= oneshot_capacity`, e.g.
    `M=1,2`) on PCIe P2P. Saves one full peer-to-peer PCIe hop versus two-shot
    finalize while avoiding separate W2 split-K sum and shared-add launches."""

    def __init__(
        self,
        hidden: int,
        world: int,
        capacity: int,
        write_aux: bool,
        top_k: int,
        slot_base: int = 0,
    ):
        super().__init__(hidden, world, capacity, write_aux, slot_base)
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        routes: Int64,
        route_weights: Int64,
        shared: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]

        self.kernel(
            routes,
            route_weights,
            shared,
            peers,
            generations,
            rank,
            residual,
            post,
            comb,
            pre,
            pre_stride,
            weight,
            residual_out,
            layer_input,
            aux,
            rows,
            eps,
            Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        routes: Int64,
        route_weights: Int64,
        shared: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        ctas, _, _ = cute.arch.grid_dim()
        top_k = self.top_k
        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            route_words = [
                load_fragment(
                    routes
                    + ((Int64(row) * top_k + k) * self.fragments + tid) * FRAGMENT_BYTES
                )
                for k in range(top_k)
            ]
            scales = [
                f32_from_bits(load_u32(route_weights + (Int64(row) * top_k + k) * 4))
                for k in range(top_k)
            ]
            shared_words = load_fragment(
                shared + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            acc = unpack(route_words[0])
            acc = [value * scales[0] for value in acc]
            for k in cutlass.range_constexpr(1, top_k):
                values = unpack(route_words[k])
                acc = [fma(values[e], scales[k], acc[e]) for e in range(VEC)]
            shared_values = unpack(shared_words)
            partial = pack([acc[e] + shared_values[e] for e in range(VEC)])
            sanitized = [sanitize_negative_zero_u32(word) for word in partial]
            # Issue peer PCIe DMA stores first so they overlap local load_row.
            for p_off in cutlass.range_constexpr(1, self.world):
                dst = (rank + p_off) % self.world
                store_fragment(
                    load_u64(peers + Int64(dst) * 8)
                    + self.slot(generation, rank, row, tid),
                    sanitized,
                )
            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )
            local = load_u64(peers + Int64(rank) * 8)
            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            s01 = [Float32(0.0)] * VEC
            s23 = [Float32(0.0)] * VEC
            s45 = [Float32(0.0)] * VEC
            s67 = [Float32(0.0)] * VEC
            for src in cutlass.range_constexpr(self.world):
                vals = [Float32(0.0)] * VEC
                if src == rank:
                    vals = unpack(sanitized)
                else:
                    poll(local + self.slot(generation, src, row, tid), words)
                    vals = unpack([words[i] for i in range(4)])
                if cutlass.const_expr(src == 0):
                    s01 = vals
                elif cutlass.const_expr(src == 1):
                    s01 = [s01[e] + vals[e] for e in range(VEC)]
                elif cutlass.const_expr(src == 2):
                    s23 = vals
                elif cutlass.const_expr(src == 3):
                    s23 = [s23[e] + vals[e] for e in range(VEC)]
                elif cutlass.const_expr(src == 4):
                    s45 = vals
                elif cutlass.const_expr(src == 5):
                    s45 = [s45[e] + vals[e] for e in range(VEC)]
                elif cutlass.const_expr(src == 6):
                    s67 = vals
                elif cutlass.const_expr(src == 7):
                    s67 = [s67[e] + vals[e] for e in range(VEC)]
            sums = s01
            if cutlass.const_expr(self.world >= 4):
                sums = [sums[e] + s23[e] for e in range(VEC)]
            if cutlass.const_expr(self.world == 8):
                s47 = [s45[e] + s67[e] for e in range(VEC)]
                sums = [sums[e] + s47[e] for e in range(VEC)]
            reduced = [
                sanitize_negative_zero_u32(
                    pack_bf16x2(sums[2 * i], sums[2 * i + 1])
                )
                for i in range(4)
            ]
            xs = unpack(reduced)
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


class RingAllReduce:
    """Single-kernel coalesced push-ring Lamport all-reduce for PCIe P2P at
    medium/large decode batches (`rows >= ring_min_rows`, e.g. `16..192`).

    Unlike TwoShot (which scatters all-to-all across PCIe switches and suffers
    switch incast above ~12 rows), each rank pushes strictly to
    `next = (rank + 1) % world` in contiguous 16B-vectorized warp strips along
    a unidirectional ring for `W - 1` Reduce-Scatter hops followed by `W - 1`
    All-Gather hops. Every rank gets bitwise-identical BF16 results, and 3
    Lamport generations make the kernel CUDA-graph safe without barriers.
    """

    def __init__(
        self,
        hidden: int,
        world: int,
        capacity: int,
        slot_base: int = 0,
    ):
        assert hidden % (VEC * world) == 0
        assert world in (2, 4, 8)
        self.hidden = hidden
        self.world = world
        self.capacity = capacity
        self.slot_base = slot_base
        self.fragments = hidden // VEC
        self.slice_fragments = self.fragments // world
        self.hops = 2 * (world - 1)

    def mailbox_bytes(self) -> int:
        return (
            LAMPORT_GENERATIONS
            * self.hops
            * self.capacity
            * self.slice_fragments
            * FRAGMENT_BYTES
        )

    @cute.jit
    def ring_slot(
        self, generation: Int32, hop: Int32, row: Int32, fragment: Int32
    ) -> Int64:
        return self.slot_base + (
            ((Int64(generation) * self.hops + hop) * self.capacity + row)
            * self.slice_fragments
            + fragment
        ) * FRAGMENT_BYTES

    @cute.jit
    def __call__(
        self,
        x: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            row_buf: cute.struct.MemRange[Uint32, self.fragments * 4]

        self.kernel(x, out, peers, generations, rank, rows, Storage).launch(
            grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        rows: Int32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sRow = storage.row_buf.get_tensor(cute.make_layout(self.fragments * 4))

        local = load_u64(peers + Int64(rank) * 8)
        next_rank = (rank + 1) % self.world
        next_peer = load_u64(peers + Int64(next_rank) * 8)
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        sf = self.slice_fragments
        W = self.world

        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            partial = load_fragment(
                x + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            sanitized = [sanitize_negative_zero_u32(word) for word in partial]
            for i in cutlass.range_constexpr(4):
                sRow[tid * 4 + i] = sanitized[i]
            cute.arch.barrier()

            if tid < sf:
                words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
                cur = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
                c0 = (rank + W - 1) % W
                base0 = (c0 * sf + tid) * 4
                for i in cutlass.range_constexpr(4):
                    cur[i] = sRow[base0 + i]
                for h in cutlass.range_constexpr(W - 1):
                    store_fragment_ring(
                        next_peer + self.ring_slot(generation, Int32(h), row, tid),
                        [cur[i] for i in range(4)],
                    )
                    in_addr = local + self.ring_slot(generation, Int32(h), row, tid)
                    poll_no_reset(in_addr, words)
                    rc = (rank + 2 * W - 2 - h) % W
                    rbase = (rc * sf + tid) * 4
                    in_f = unpack([words[i] for i in range(4)])
                    loc_f = unpack([sRow[rbase + i] for i in range(4)])
                    summed = [in_f[i] + loc_f[i] for i in range(VEC)]
                    for i in cutlass.range_constexpr(4):
                        cur[i] = sanitize_negative_zero_u32(
                            pack_bf16x2(summed[2 * i], summed[2 * i + 1])
                        )

                own_base = (rank * sf + tid) * 4
                for i in cutlass.range_constexpr(4):
                    sRow[own_base + i] = cur[i]

                for h in cutlass.range_constexpr(W - 1):
                    hop_idx = (W - 1) + h
                    store_fragment_ring(
                        next_peer
                        + self.ring_slot(generation, Int32(hop_idx), row, tid),
                        [cur[i] for i in range(4)],
                    )
                    in_addr = local + self.ring_slot(
                        generation, Int32(hop_idx), row, tid
                    )
                    poll_no_reset(in_addr, words)
                    rc = (rank + 2 * W - 1 - h) % W
                    rbase = (rc * sf + tid) * 4
                    for i in cutlass.range_constexpr(4):
                        val = words[i]
                        sRow[rbase + i] = val
                        cur[i] = val

            cute.arch.barrier()
            store_fragment(
                out + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES,
                [sRow[tid * 4 + i] for i in range(4)],
            )
            if tid < sf:
                for h in cutlass.range_constexpr(2 * (W - 1)):
                    store_fragment_ring(
                        local + self.ring_slot(generation, Int32(h), row, tid),
                        [sentinel] * 4,
                    )
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas



class RingBroadcast(RingAllReduce):
    """Single-kernel pipelined unidirectional push-ring broadcast from 
    (reuses RingAllReduce mailbox; 1.0x data vs 1.5x for RingAllReduce)."""

    @cute.jit
    def __call__(
        self,
        x: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        src_rank: Int32,
        rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            row_buf: cute.struct.MemRange[Uint32, self.fragments * 4]

        self.kernel(x, out, peers, generations, rank, src_rank, rows, Storage).launch(
            grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        src_rank: Int32,
        rows: Int32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sRow = storage.row_buf.get_tensor(cute.make_layout(self.fragments * 4))

        local = load_u64(peers + Int64(rank) * 8)
        next_rank = (rank + 1) % self.world
        next_peer = load_u64(peers + Int64(next_rank) * 8)
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        sf = self.slice_fragments
        W = self.world

        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            if rank == src_rank:
                partial = load_fragment(
                    x + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
                )
                sanitized = [sanitize_negative_zero_u32(word) for word in partial]
                for i in cutlass.range_constexpr(4):
                    sRow[tid * 4 + i] = sanitized[i]
                store_fragment(
                    out + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES,
                    sanitized,
                )
                cute.arch.barrier()
                if tid < sf:
                    for c in cutlass.range_constexpr(W):
                        cbase = (c * sf + tid) * 4
                        store_fragment_ring(
                            next_peer + self.ring_slot(generation, Int32(c), row, tid),
                            [sRow[cbase + i] for i in range(4)],
                        )
            else:
                if tid < sf:
                    words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
                    for c in cutlass.range_constexpr(W):
                        in_addr = local + self.ring_slot(generation, Int32(c), row, tid)
                        poll_no_reset(in_addr, words)
                        if next_rank != src_rank:
                            store_fragment_ring(
                                next_peer
                                + self.ring_slot(generation, Int32(c), row, tid),
                                [words[i] for i in range(4)],
                            )
                        cbase = (c * sf + tid) * 4
                        for i in cutlass.range_constexpr(4):
                            sRow[cbase + i] = words[i]
                cute.arch.barrier()
                store_fragment(
                    out + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES,
                    [sRow[tid * 4 + i] for i in range(4)],
                )
                if tid < sf:
                    for c in cutlass.range_constexpr(W):
                        store_fragment_ring(
                            local + self.ring_slot(generation, Int32(c), row, tid),
                            [sentinel] * 4,
                        )
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


class RingAllReducePostCollapse(RingAllReduce):
    """`RingAllReduce` fused with `PostCollapse` (post-mix, collapse, RMSNorm)
    in one CTA, overlapping Hop 0 PCIe flight with local residual loads."""

    def __init__(
        self,
        hidden: int,
        world: int,
        capacity: int,
        write_aux: bool,
        slot_base: int = 0,
    ):
        super().__init__(hidden, world, capacity, slot_base)
        self.collapse = PostCollapse(hidden, has_post=True, write_aux=write_aux)

    @cute.jit
    def __call__(
        self,
        x: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]
            row_buf: cute.struct.MemRange[Uint32, self.fragments * 4]

        self.kernel(
            x,
            peers,
            generations,
            rank,
            residual,
            post,
            comb,
            pre,
            pre_stride,
            weight,
            residual_out,
            layer_input,
            aux,
            rows,
            eps,
            Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        sRow = storage.row_buf.get_tensor(cute.make_layout(self.fragments * 4))

        local = load_u64(peers + Int64(rank) * 8)
        next_rank = (rank + 1) % self.world
        next_peer = load_u64(peers + Int64(next_rank) * 8)
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        sf = self.slice_fragments
        W = self.world

        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            partial = load_fragment(
                x + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            sanitized = [sanitize_negative_zero_u32(word) for word in partial]
            for i in cutlass.range_constexpr(4):
                sRow[tid * 4 + i] = sanitized[i]
            cute.arch.barrier()

            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            cur = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            if tid < sf:
                c0 = (rank + W - 1) % W
                base0 = (c0 * sf + tid) * 4
                for i in cutlass.range_constexpr(4):
                    cur[i] = sRow[base0 + i]
                store_fragment_ring(
                    next_peer + self.ring_slot(generation, Int32(0), row, tid),
                    [cur[i] for i in range(4)],
                )

            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )

            if tid < sf:
                for h in cutlass.range_constexpr(W - 1):
                    if cutlass.const_expr(h > 0):
                        store_fragment_ring(
                            next_peer
                            + self.ring_slot(generation, Int32(h), row, tid),
                            [cur[i] for i in range(4)],
                        )
                    in_addr = local + self.ring_slot(generation, Int32(h), row, tid)
                    poll_no_reset(in_addr, words)
                    rc = (rank + 2 * W - 2 - h) % W
                    rbase = (rc * sf + tid) * 4
                    in_f = unpack([words[i] for i in range(4)])
                    loc_f = unpack([sRow[rbase + i] for i in range(4)])
                    summed = [in_f[i] + loc_f[i] for i in range(VEC)]
                    for i in cutlass.range_constexpr(4):
                        cur[i] = sanitize_negative_zero_u32(
                            pack_bf16x2(summed[2 * i], summed[2 * i + 1])
                        )

                own_base = (rank * sf + tid) * 4
                for i in cutlass.range_constexpr(4):
                    sRow[own_base + i] = cur[i]

                for h in cutlass.range_constexpr(W - 1):
                    hop_idx = (W - 1) + h
                    store_fragment_ring(
                        next_peer
                        + self.ring_slot(generation, Int32(hop_idx), row, tid),
                        [cur[i] for i in range(4)],
                    )
                    in_addr = local + self.ring_slot(
                        generation, Int32(hop_idx), row, tid
                    )
                    poll_no_reset(in_addr, words)
                    rc = (rank + 2 * W - 1 - h) % W
                    rbase = (rc * sf + tid) * 4
                    for i in cutlass.range_constexpr(4):
                        val = words[i]
                        sRow[rbase + i] = val
                        cur[i] = val

            cute.arch.barrier()
            xs = unpack([sRow[tid * 4 + i] for i in range(4)])
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            if tid < sf:
                for h in cutlass.range_constexpr(2 * (W - 1)):
                    store_fragment_ring(
                        local + self.ring_slot(generation, Int32(h), row, tid),
                        [sentinel] * 4,
                    )
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


class RingAllReduceFinalizePostCollapse(RingAllReducePostCollapse):
    """`RingAllReducePostCollapse` fused with MoE top-k weighting/sum and
    shared-expert add in one CTA (`rows >= ring_min_rows`, e.g. `16..192`)."""

    def __init__(
        self,
        hidden: int,
        world: int,
        capacity: int,
        write_aux: bool,
        top_k: int,
        slot_base: int = 0,
    ):
        super().__init__(hidden, world, capacity, write_aux, slot_base)
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        routes: Int64,
        route_weights: Int64,
        shared: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]
            row_buf: cute.struct.MemRange[Uint32, self.fragments * 4]

        self.kernel(
            routes,
            route_weights,
            shared,
            peers,
            generations,
            rank,
            residual,
            post,
            comb,
            pre,
            pre_stride,
            weight,
            residual_out,
            layer_input,
            aux,
            rows,
            eps,
            Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        routes: Int64,
        route_weights: Int64,
        shared: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        sRow = storage.row_buf.get_tensor(cute.make_layout(self.fragments * 4))

        local = load_u64(peers + Int64(rank) * 8)
        next_rank = (rank + 1) % self.world
        next_peer = load_u64(peers + Int64(next_rank) * 8)
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        sf = self.slice_fragments
        W = self.world
        top_k = self.top_k

        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)

            route_words = [
                load_fragment(
                    routes
                    + ((Int64(row) * top_k + k) * self.fragments + tid)
                    * FRAGMENT_BYTES
                )
                for k in range(top_k)
            ]
            scales = [
                f32_from_bits(
                    load_u32(route_weights + (Int64(row) * top_k + k) * 4)
                )
                for k in range(top_k)
            ]
            shared_words = load_fragment(
                shared + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            acc = unpack(route_words[0])
            acc = [value * scales[0] for value in acc]
            for k in cutlass.range_constexpr(1, top_k):
                values = unpack(route_words[k])
                acc = [fma(values[e], scales[k], acc[e]) for e in range(VEC)]
            shared_values = unpack(shared_words)
            partial = pack([acc[e] + shared_values[e] for e in range(VEC)])
            sanitized = [sanitize_negative_zero_u32(word) for word in partial]
            for i in cutlass.range_constexpr(4):
                sRow[tid * 4 + i] = sanitized[i]
            cute.arch.barrier()

            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            cur = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            if tid < sf:
                c0 = (rank + W - 1) % W
                base0 = (c0 * sf + tid) * 4
                for i in cutlass.range_constexpr(4):
                    cur[i] = sRow[base0 + i]
                store_fragment_ring(
                    next_peer + self.ring_slot(generation, Int32(0), row, tid),
                    [cur[i] for i in range(4)],
                )

            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )

            if tid < sf:
                for h in cutlass.range_constexpr(W - 1):
                    if cutlass.const_expr(h > 0):
                        store_fragment_ring(
                            next_peer
                            + self.ring_slot(generation, Int32(h), row, tid),
                            [cur[i] for i in range(4)],
                        )
                    in_addr = local + self.ring_slot(generation, Int32(h), row, tid)
                    poll_no_reset(in_addr, words)
                    rc = (rank + 2 * W - 2 - h) % W
                    rbase = (rc * sf + tid) * 4
                    in_f = unpack([words[i] for i in range(4)])
                    loc_f = unpack([sRow[rbase + i] for i in range(4)])
                    summed = [in_f[i] + loc_f[i] for i in range(VEC)]
                    for i in cutlass.range_constexpr(4):
                        cur[i] = sanitize_negative_zero_u32(
                            pack_bf16x2(summed[2 * i], summed[2 * i + 1])
                        )

                own_base = (rank * sf + tid) * 4
                for i in cutlass.range_constexpr(4):
                    sRow[own_base + i] = cur[i]

                for h in cutlass.range_constexpr(W - 1):
                    hop_idx = (W - 1) + h
                    store_fragment_ring(
                        next_peer
                        + self.ring_slot(generation, Int32(hop_idx), row, tid),
                        [cur[i] for i in range(4)],
                    )
                    in_addr = local + self.ring_slot(
                        generation, Int32(hop_idx), row, tid
                    )
                    poll_no_reset(in_addr, words)
                    rc = (rank + 2 * W - 1 - h) % W
                    rbase = (rc * sf + tid) * 4
                    for i in cutlass.range_constexpr(4):
                        val = words[i]
                        sRow[rbase + i] = val
                        cur[i] = val

            cute.arch.barrier()
            xs = unpack([sRow[tid * 4 + i] for i in range(4)])
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            if tid < sf:
                for h in cutlass.range_constexpr(2 * (W - 1)):
                    store_fragment_ring(
                        local + self.ring_slot(generation, Int32(h), row, tid),
                        [sentinel] * 4,
                    )
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas




class HDBAllReduce:
    """Single-kernel 4-hop Recursive Halving-Doubling (Rabenseifner) push-Lamport
    all-reduce for world == 4 on PCIe Gen4 x16:
      - Exact same minimum 1.50x byte volume per rank as RingAllReduce
      - 4 serial hops instead of 6 (-33% hop latency)
      - 320-thread (5 KB) coalesced bursts on Hops 0 & 3
      - 640-thread parallel local mailbox reset (1.5 stores/thread vs 6)
      - Symmetric pairwise tree sum (r0+r1)+(r2+r3), bit-identical across ranks
    """

    def __init__(
        self,
        hidden: int,
        world: int,
        capacity: int,
        slot_base: int = 0,
    ):
        assert world == 4 and hidden % (VEC * 4) == 0
        self.hidden = hidden
        self.world = world
        self.capacity = capacity
        self.slot_base = slot_base
        self.fragments = hidden // VEC
        self.hf = self.fragments // 2
        self.qf = self.fragments // 4
        self.row_slots = 6 * self.qf  # 320 + 160 + 160 + 320 == 960 (same as Ring)

    def mailbox_bytes(self) -> int:
        return (
            LAMPORT_GENERATIONS
            * self.capacity
            * self.row_slots
            * FRAGMENT_BYTES
        )

    @cute.jit
    def hdb_slot(
        self, generation: Int32, hop_base: Int32, hop_width: Int32, row: Int32, fragment: Int32
    ) -> Int64:
        return self.slot_base + (
            Int64(generation) * self.capacity * self.row_slots
            + Int64(hop_base) * self.capacity
            + Int64(row) * hop_width
            + fragment
        ) * FRAGMENT_BYTES

    @cute.jit
    def __call__(
        self,
        x: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            row_buf: cute.struct.MemRange[Uint32, self.fragments * 4]

        self.kernel(x, out, peers, generations, rank, rows, Storage).launch(
            grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        rows: Int32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sRow = storage.row_buf.get_tensor(cute.make_layout(self.fragments * 4))

        r_lo = rank % 2
        r_hi = rank // 2
        peer0_rank = r_hi * 2 + (1 - r_lo)
        peer1_rank = (1 - r_hi) * 2 + r_lo
        keep_half = r_lo
        send_half = 1 - r_lo
        own_q = r_lo * 2 + r_hi
        send_q1 = r_lo * 2 + (1 - r_hi)

        local = load_u64(peers + Int64(rank) * 8)
        peer0 = load_u64(peers + Int64(peer0_rank) * 8)
        peer1 = load_u64(peers + Int64(peer1_rank) * 8)
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        hf = self.hf
        qf = self.qf

        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            partial = load_fragment(
                x + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            sanitized = [sanitize_negative_zero_u32(word) for word in partial]
            for i in cutlass.range_constexpr(4):
                sRow[tid * 4 + i] = sanitized[i]
            cute.arch.barrier()

            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            cur = cute.make_rmem_tensor(cute.make_layout(4), Uint32)

            # Hop 0: pairwise rank^1 exchange half row (320 threads = 5 KB)
            if tid < hf:
                sbase0 = (send_half * hf + tid) * 4
                store_fragment_ring(
                    peer0 + self.hdb_slot(generation, Int32(0), Int32(hf), row, tid),
                    [sRow[sbase0 + i] for i in range(4)],
                )
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(0), Int32(hf), row, tid), words
                )
                kbase0 = (keep_half * hf + tid) * 4
                in_f = unpack([words[i] for i in range(4)])
                loc_f = unpack([sRow[kbase0 + i] for i in range(4)])
                summed = [in_f[i] + loc_f[i] for i in range(VEC)]
                for i in cutlass.range_constexpr(4):
                    sRow[kbase0 + i] = sanitize_negative_zero_u32(
                        pack_bf16x2(summed[2 * i], summed[2 * i + 1])
                    )
            cute.arch.barrier()

            # Hop 1 + Hop 2: pairwise rank^2 exchange quarter row (160 threads = 2.5 KB)
            if tid < qf:
                sbase1 = (send_q1 * qf + tid) * 4
                store_fragment_ring(
                    peer1 + self.hdb_slot(generation, Int32(hf), Int32(qf), row, tid),
                    [sRow[sbase1 + i] for i in range(4)],
                )
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(hf), Int32(qf), row, tid), words
                )
                obase1 = (own_q * qf + tid) * 4
                in_f = unpack([words[i] for i in range(4)])
                loc_f = unpack([sRow[obase1 + i] for i in range(4)])
                summed = [in_f[i] + loc_f[i] for i in range(VEC)]
                for i in cutlass.range_constexpr(4):
                    v = sanitize_negative_zero_u32(
                        pack_bf16x2(summed[2 * i], summed[2 * i + 1])
                    )
                    cur[i] = v
                    sRow[obase1 + i] = v

                # Hop 2: AllGather stage 1 with peer1 (no barrier needed!)
                store_fragment_ring(
                    peer1 + self.hdb_slot(generation, Int32(hf + qf), Int32(qf), row, tid),
                    [cur[i] for i in range(4)],
                )
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(hf + qf), Int32(qf), row, tid),
                    words,
                )
                for i in cutlass.range_constexpr(4):
                    sRow[sbase1 + i] = words[i]
            cute.arch.barrier()

            # Hop 3: AllGather stage 2 with peer0 (320 threads = 5 KB)
            if tid < hf:
                kbase3 = (keep_half * hf + tid) * 4
                store_fragment_ring(
                    peer0 + self.hdb_slot(generation, Int32(hf + 2 * qf), Int32(hf), row, tid),
                    [sRow[kbase3 + i] for i in range(4)],
                )
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(hf + 2 * qf), Int32(hf), row, tid),
                    words,
                )
                sbase3 = (send_half * hf + tid) * 4
                for i in cutlass.range_constexpr(4):
                    sRow[sbase3 + i] = words[i]
            cute.arch.barrier()

            store_fragment(
                out + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES,
                [sRow[tid * 4 + i] for i in range(4)],
            )
            if tid < hf:
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(0), Int32(hf), row, tid),
                    [sentinel] * 4,
                )
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(hf + 2 * qf), Int32(hf), row, tid),
                    [sentinel] * 4,
                )
            elif tid < hf + qf:
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(hf), Int32(qf), row, tid - hf),
                    [sentinel] * 4,
                )
            else:
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(hf + qf), Int32(qf), row, tid - (hf + qf)),
                    [sentinel] * 4,
                )
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


class HDBAllReducePostCollapse(HDBAllReduce):
    """4-hop `HDBAllReduce` fused with `PostCollapse` for world == 4."""

    def __init__(
        self,
        hidden: int,
        world: int,
        capacity: int,
        write_aux: bool,
        slot_base: int = 0,
    ):
        super().__init__(hidden, world, capacity, slot_base)
        self.collapse = PostCollapse(hidden, HC, write_aux)
        assert self.fragments == self.collapse.threads

    @cute.jit
    def __call__(
        self,
        x: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]
            row_buf: cute.struct.MemRange[Uint32, self.fragments * 4]

        self.kernel(
            x, peers, generations, rank, residual, post, comb, pre, pre_stride,
            weight, residual_out, layer_input, aux, rows, eps, Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        sRow = storage.row_buf.get_tensor(cute.make_layout(self.fragments * 4))

        r_lo = rank % 2
        r_hi = rank // 2
        peer0_rank = r_hi * 2 + (1 - r_lo)
        peer1_rank = (1 - r_hi) * 2 + r_lo
        keep_half = r_lo
        send_half = 1 - r_lo
        own_q = r_lo * 2 + r_hi
        send_q1 = r_lo * 2 + (1 - r_hi)

        local = load_u64(peers + Int64(rank) * 8)
        peer0 = load_u64(peers + Int64(peer0_rank) * 8)
        peer1 = load_u64(peers + Int64(peer1_rank) * 8)
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        hf = self.hf
        qf = self.qf

        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)
            partial = load_fragment(
                x + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            sanitized = [sanitize_negative_zero_u32(word) for word in partial]
            for i in cutlass.range_constexpr(4):
                sRow[tid * 4 + i] = sanitized[i]
            cute.arch.barrier()

            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            cur = cute.make_rmem_tensor(cute.make_layout(4), Uint32)

            # Issue Hop 0 PCIe burst BEFORE collapse.load_row to overlap flight!
            if tid < hf:
                sbase0 = (send_half * hf + tid) * 4
                store_fragment_ring(
                    peer0 + self.hdb_slot(generation, Int32(0), Int32(hf), row, tid),
                    [sRow[sbase0 + i] for i in range(4)],
                )

            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )

            if tid < hf:
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(0), Int32(hf), row, tid), words
                )
                kbase0 = (keep_half * hf + tid) * 4
                in_f = unpack([words[i] for i in range(4)])
                loc_f = unpack([sRow[kbase0 + i] for i in range(4)])
                summed = [in_f[i] + loc_f[i] for i in range(VEC)]
                for i in cutlass.range_constexpr(4):
                    sRow[kbase0 + i] = sanitize_negative_zero_u32(
                        pack_bf16x2(summed[2 * i], summed[2 * i + 1])
                    )
            cute.arch.barrier()

            if tid < qf:
                sbase1 = (send_q1 * qf + tid) * 4
                store_fragment_ring(
                    peer1 + self.hdb_slot(generation, Int32(hf), Int32(qf), row, tid),
                    [sRow[sbase1 + i] for i in range(4)],
                )
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(hf), Int32(qf), row, tid), words
                )
                obase1 = (own_q * qf + tid) * 4
                in_f = unpack([words[i] for i in range(4)])
                loc_f = unpack([sRow[obase1 + i] for i in range(4)])
                summed = [in_f[i] + loc_f[i] for i in range(VEC)]
                for i in cutlass.range_constexpr(4):
                    v = sanitize_negative_zero_u32(
                        pack_bf16x2(summed[2 * i], summed[2 * i + 1])
                    )
                    cur[i] = v
                    sRow[obase1 + i] = v

                store_fragment_ring(
                    peer1 + self.hdb_slot(generation, Int32(hf + qf), Int32(qf), row, tid),
                    [cur[i] for i in range(4)],
                )
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(hf + qf), Int32(qf), row, tid),
                    words,
                )
                for i in cutlass.range_constexpr(4):
                    sRow[sbase1 + i] = words[i]
            cute.arch.barrier()

            if tid < hf:
                kbase3 = (keep_half * hf + tid) * 4
                store_fragment_ring(
                    peer0 + self.hdb_slot(generation, Int32(hf + 2 * qf), Int32(hf), row, tid),
                    [sRow[kbase3 + i] for i in range(4)],
                )
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(hf + 2 * qf), Int32(hf), row, tid),
                    words,
                )
                sbase3 = (send_half * hf + tid) * 4
                for i in cutlass.range_constexpr(4):
                    sRow[sbase3 + i] = words[i]
            cute.arch.barrier()

            xs = unpack([sRow[tid * 4 + i] for i in range(4)])
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            if tid < hf:
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(0), Int32(hf), row, tid),
                    [sentinel] * 4,
                )
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(hf + 2 * qf), Int32(hf), row, tid),
                    [sentinel] * 4,
                )
            elif tid < hf + qf:
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(hf), Int32(qf), row, tid - hf),
                    [sentinel] * 4,
                )
            else:
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(hf + qf), Int32(qf), row, tid - (hf + qf)),
                    [sentinel] * 4,
                )
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


class HDBAllReduceFinalizePostCollapse(HDBAllReducePostCollapse):
    """4-hop `HDBAllReducePostCollapse` fused with MoE top-k finalize + shared add."""

    def __init__(
        self,
        hidden: int,
        world: int,
        capacity: int,
        write_aux: bool,
        top_k: int,
        slot_base: int = 0,
    ):
        super().__init__(hidden, world, capacity, write_aux, slot_base)
        assert top_k > 0
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        routes: Int64,
        route_weights: Int64,
        shared: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        ctas: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            sums: cute.struct.MemRange[Float32, self.collapse.warps]
            row_buf: cute.struct.MemRange[Uint32, self.fragments * 4]

        self.kernel(
            routes, route_weights, shared, peers, generations, rank, residual,
            post, comb, pre, pre_stride, weight, residual_out, layer_input, aux,
            rows, eps, Storage,
        ).launch(grid=(ctas, 1, 1), block=[self.fragments, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        routes: Int64,
        route_weights: Int64,
        shared: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        residual: Int64,
        post: Int64,
        comb: Int64,
        pre: Int64,
        pre_stride: Int32,
        weight: Int64,
        residual_out: Int64,
        layer_input: Int64,
        aux: Int64,
        rows: Int32,
        eps: Float32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sSums = storage.sums.get_tensor(cute.make_layout(self.collapse.warps))
        sRow = storage.row_buf.get_tensor(cute.make_layout(self.fragments * 4))

        r_lo = rank % 2
        r_hi = rank // 2
        peer0_rank = r_hi * 2 + (1 - r_lo)
        peer1_rank = (1 - r_hi) * 2 + r_lo
        keep_half = r_lo
        send_half = 1 - r_lo
        own_q = r_lo * 2 + r_hi
        send_q1 = r_lo * 2 + (1 - r_hi)

        local = load_u64(peers + Int64(rank) * 8)
        peer0 = load_u64(peers + Int64(peer0_rank) * 8)
        peer1 = load_u64(peers + Int64(peer1_rank) * 8)
        sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR)
        hf = self.hf
        qf = self.qf
        top_k = self.top_k

        while row < rows:
            generation_address = generations + Int64(row) * 4
            generation = load_s32(generation_address)

            route_words = [
                load_fragment(
                    routes
                    + ((Int64(row) * top_k + k) * self.fragments + tid)
                    * FRAGMENT_BYTES
                )
                for k in range(top_k)
            ]
            scales = [
                f32_from_bits(
                    load_u32(route_weights + (Int64(row) * top_k + k) * 4)
                )
                for k in range(top_k)
            ]
            shared_words = load_fragment(
                shared + (Int64(row) * self.fragments + tid) * FRAGMENT_BYTES
            )
            acc = unpack(route_words[0])
            acc = [value * scales[0] for value in acc]
            for k in cutlass.range_constexpr(1, top_k):
                values = unpack(route_words[k])
                acc = [fma(values[e], scales[k], acc[e]) for e in range(VEC)]
            shared_values = unpack(shared_words)
            partial = pack([acc[e] + shared_values[e] for e in range(VEC)])
            sanitized = [sanitize_negative_zero_u32(word) for word in partial]
            for i in cutlass.range_constexpr(4):
                sRow[tid * 4 + i] = sanitized[i]
            cute.arch.barrier()

            words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            cur = cute.make_rmem_tensor(cute.make_layout(4), Uint32)

            if tid < hf:
                sbase0 = (send_half * hf + tid) * 4
                store_fragment_ring(
                    peer0 + self.hdb_slot(generation, Int32(0), Int32(hf), row, tid),
                    [sRow[sbase0 + i] for i in range(4)],
                )

            loaded = self.collapse.load_row(
                residual, post, comb, pre, pre_stride, weight, row, tid
            )

            if tid < hf:
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(0), Int32(hf), row, tid), words
                )
                kbase0 = (keep_half * hf + tid) * 4
                in_f = unpack([words[i] for i in range(4)])
                loc_f = unpack([sRow[kbase0 + i] for i in range(4)])
                summed = [in_f[i] + loc_f[i] for i in range(VEC)]
                for i in cutlass.range_constexpr(4):
                    sRow[kbase0 + i] = sanitize_negative_zero_u32(
                        pack_bf16x2(summed[2 * i], summed[2 * i + 1])
                    )
            cute.arch.barrier()

            if tid < qf:
                sbase1 = (send_q1 * qf + tid) * 4
                store_fragment_ring(
                    peer1 + self.hdb_slot(generation, Int32(hf), Int32(qf), row, tid),
                    [sRow[sbase1 + i] for i in range(4)],
                )
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(hf), Int32(qf), row, tid), words
                )
                obase1 = (own_q * qf + tid) * 4
                in_f = unpack([words[i] for i in range(4)])
                loc_f = unpack([sRow[obase1 + i] for i in range(4)])
                summed = [in_f[i] + loc_f[i] for i in range(VEC)]
                for i in cutlass.range_constexpr(4):
                    v = sanitize_negative_zero_u32(
                        pack_bf16x2(summed[2 * i], summed[2 * i + 1])
                    )
                    cur[i] = v
                    sRow[obase1 + i] = v

                store_fragment_ring(
                    peer1 + self.hdb_slot(generation, Int32(hf + qf), Int32(qf), row, tid),
                    [cur[i] for i in range(4)],
                )
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(hf + qf), Int32(qf), row, tid),
                    words,
                )
                for i in cutlass.range_constexpr(4):
                    sRow[sbase1 + i] = words[i]
            cute.arch.barrier()

            if tid < hf:
                kbase3 = (keep_half * hf + tid) * 4
                store_fragment_ring(
                    peer0 + self.hdb_slot(generation, Int32(hf + 2 * qf), Int32(hf), row, tid),
                    [sRow[kbase3 + i] for i in range(4)],
                )
                poll_no_reset(
                    local + self.hdb_slot(generation, Int32(hf + 2 * qf), Int32(hf), row, tid),
                    words,
                )
                sbase3 = (send_half * hf + tid) * 4
                for i in cutlass.range_constexpr(4):
                    sRow[sbase3 + i] = words[i]
            cute.arch.barrier()

            xs = unpack([sRow[tid * 4 + i] for i in range(4)])
            self.collapse.finish_row(
                xs, loaded, residual_out, layer_input, aux, eps, sSums, row, tid
            )
            if tid < hf:
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(0), Int32(hf), row, tid),
                    [sentinel] * 4,
                )
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(hf + 2 * qf), Int32(hf), row, tid),
                    [sentinel] * 4,
                )
            elif tid < hf + qf:
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(hf), Int32(qf), row, tid - hf),
                    [sentinel] * 4,
                )
            else:
                store_fragment_ring(
                    local + self.hdb_slot(generation, Int32(hf + qf), Int32(qf), row, tid - (hf + qf)),
                    [sentinel] * 4,
                )
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            row += ctas


@cache
def _hdb_kernel(
    device_index: int, hidden: int, world: int, capacity: int, slot_base: int
):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            HDBAllReduce(hidden, world, capacity, slot_base),
            *(Int64(0),) * 4,
            Int32(0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _hdb_collapse_kernel(
    device_index: int,
    hidden: int,
    world: int,
    capacity: int,
    write_aux: bool,
    slot_base: int,
):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            HDBAllReducePostCollapse(hidden, world, capacity, write_aux, slot_base),
            *(Int64(0),) * 3,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


@cache
def _hdb_finalize_kernel(
    device_index: int,
    hidden: int,
    world: int,
    capacity: int,
    write_aux: bool,
    top_k: int,
    slot_base: int,
):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            HDBAllReduceFinalizePostCollapse(
                hidden, world, capacity, write_aux, top_k, slot_base
            ),
            *(Int64(0),) * 5,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


@cache
def _ring_broadcast_kernel(
    device_index: int, hidden: int, world: int, capacity: int, slot_base: int
):
    """ compiled for a device and shape."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RingBroadcast(hidden, world, capacity, slot_base),
            *(Int64(0),) * 4,
            Int32(0),
            Int32(0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _ring_kernel(
    device_index: int, hidden: int, world: int, capacity: int, slot_base: int
):
    """`RingAllReduce` compiled for a device and shape."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RingAllReduce(hidden, world, capacity, slot_base),
            *(Int64(0),) * 4,
            Int32(0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _ring_collapse_kernel(
    device_index: int,
    hidden: int,
    world: int,
    capacity: int,
    write_aux: bool,
    slot_base: int,
):
    """`RingAllReducePostCollapse` compiled for a device and variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RingAllReducePostCollapse(hidden, world, capacity, write_aux, slot_base),
            *(Int64(0),) * 3,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


@cache
def _ring_finalize_kernel(
    device_index: int,
    hidden: int,
    world: int,
    capacity: int,
    write_aux: bool,
    top_k: int,
    slot_base: int,
):
    """`RingAllReduceFinalizePostCollapse` compiled for a device and variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RingAllReduceFinalizePostCollapse(
                hidden, world, capacity, write_aux, top_k, slot_base
            ),
            *(Int64(0),) * 5,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


@cache
def _oneshot_finalize_kernel(
    device_index: int,
    hidden: int,
    world: int,
    capacity: int,
    write_aux: bool,
    top_k: int,
    slot_base: int,
):
    """`OneShotAllReduceFinalizePostCollapse` compiled for a device and variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            OneShotAllReduceFinalizePostCollapse(
                hidden, world, capacity, write_aux, top_k, slot_base
            ),
            *(Int64(0),) * 5,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


@cache
def _oneshot_collapse_kernel(
    device_index: int,
    hidden: int,
    world: int,
    capacity: int,
    write_aux: bool,
    slot_base: int,
):
    """`OneShotAllReducePostCollapse` compiled for a device and variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            OneShotAllReducePostCollapse(
                hidden, world, capacity, write_aux, slot_base
            ),
            *(Int64(0),) * 3,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


@cache
def _kernel(device_index: int, hidden: int, world: int, capacity: int):
    """`TwoShotAllReduce` compiled for a device and shape."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            TwoShotAllReduce(hidden, world, capacity),
            *(Int64(0),) * 4,
            Int32(0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _collapse_kernel(
    device_index: int, hidden: int, world: int, capacity: int, write_aux: bool
):
    """`TwoShotAllReducePostCollapse` compiled for a device and variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            TwoShotAllReducePostCollapse(hidden, world, capacity, write_aux),
            *(Int64(0),) * 3,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


@cache
def _finalize_kernel(
    device_index: int,
    hidden: int,
    world: int,
    capacity: int,
    write_aux: bool,
    top_k: int,
):
    """`TwoShotAllReduceFinalizePostCollapse` compiled for a device and
    variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            TwoShotAllReduceFinalizePostCollapse(
                hidden, world, capacity, write_aux, top_k
            ),
            *(Int64(0),) * 5,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


@cache
def _w2_splitk_kernel(
    device_index: int,
    hidden: int,
    world: int,
    capacity: int,
    write_aux: bool,
    top_k: int,
    ks: int,
):
    """`TwoShotAllReduceW2SplitKPostCollapse` compiled for a device and variant."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            TwoShotAllReduceW2SplitKPostCollapse(
                hidden, world, capacity, write_aux, top_k, ks
            ),
            *(Int64(0),) * 5,
            Int32(384),
            Int64(0),
            *(Int64(0),) * 2,
            Int32(0),
            *(Int64(0),) * 4,
            Int32(4),
            *(Int64(0),) * 4,
            Int32(1),
            Int32(1),
            Float32(1e-6),
            make_fake_stream(),
        )


def ampere_peer_topology() -> str | None:
    """Return `'nvlink'` when vLLM's custom all-reduce reports full NVLink
    connectivity, `'pcie'` when PCIe P2P collectives are opted in and every TP
    peer pair on the node has direct CUDA P2P access behind a shared PCIe
    switch (`PIX`/`PXB`, or overridden via `VLLM_DSV41_AMPERE_PCIE_COLLECTIVES=force`),
    or `None` otherwise. Collective over the TP group so all ranks agree."""
    from vllm.distributed import get_tp_group
    from vllm.dsv41_opt_profile import NVML_TOPO_NAMES, pcie_switch_connected_nvml
    from vllm.logger import init_logger

    tp_group = get_tp_group()
    comm = getattr(tp_group.device_communicator, "ca_comm", None)
    if comm is not None and not comm.disabled and comm.fully_connected:
        return "nvlink"
    pcie_opt = (
        os.environ.get("VLLM_DSV41_AMPERE_PCIE_COLLECTIVES", "0").strip().lower()
    )
    if pcie_opt not in ("1", "force"):
        return None
    world_size = tp_group.world_size
    if world_size < 2 or world_size > torch.cuda.device_count():
        return None
    local_dev = torch.cuda.current_device()
    dev_tensor = torch.empty(world_size, dtype=torch.int32, device=tp_group.device)
    my_dev = torch.tensor([local_dev], dtype=torch.int32, device=tp_group.device)
    dist.all_gather_into_tensor(dev_tensor, my_dev, group=tp_group.device_group)
    peer_devs = [int(x) for x in dev_tensor.cpu().tolist()]
    ok = int(len(set(peer_devs)) == world_size)
    if ok:
        for d in peer_devs:
            if d != local_dev and not torch.cuda.can_device_access_peer(local_dev, d):
                ok = 0
                break
    if ok and pcie_opt != "force":
        phys_ids: list[int | str] = []
        for d in peer_devs:
            try:
                u = getattr(torch.cuda.get_device_properties(d), "uuid", None)
                if u is not None:
                    phys_ids.append(f"GPU-{u}")
                    continue
            except Exception:
                pass
            phys_ids.append(current_platform.visible_device_id_to_physical_device_id(d))
        switch_ok, worst_lvl = pcie_switch_connected_nvml(phys_ids)
        if worst_lvl is not None and not switch_ok:
            ok = 0
            topo_name = NVML_TOPO_NAMES.get(worst_lvl, f"level={worst_lvl}")
            init_logger(__name__).info_once(
                "DSV41: PCIe peers traverse CPU root complex (NVML topology=%s); "
                "disabling CuTe Lamport P2P collectives and falling back to PyNCCL "
                "(fine-grained 16B P2P writes congest cross-RC interconnects; "
                "set VLLM_DSV41_AMPERE_PCIE_COLLECTIVES=force to override, and "
                "consider NCCL_P2P_LEVEL=SYS if PyNCCL disables P2P).",
                topo_name,
            )
    flag = torch.tensor([ok], dtype=torch.int32, device=tp_group.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=tp_group.device_group)
    return "pcie" if int(flag.item()) == 1 else None


def default_ampere_ar_capacity(topology: str) -> int:
    """Default max rows for `AmpereAllReduce`: full `MAX_ROWS` (192) on both
    NVLink (TwoShot) and PCIe Gen4 x16 (3-regime OneShot -> TwoShot -> Ring).
    Overridable via `VLLM_DSV41_AMPERE_AR_MAX_ROWS`."""
    raw = os.environ.get("VLLM_DSV41_AMPERE_AR_MAX_ROWS", "").strip()
    if raw:
        return max(1, min(MAX_ROWS, int(raw)))
    return MAX_ROWS


def default_ampere_ag_capacity(
    topology: str, row_bytes: int, max_rows: int = MAX_ROWS
) -> int:
    """Default max rows for `AmpereAllGather`: full `max_rows` on NVLink (or
    tiny <=64B control/argmax rows on PCIe), else bounded by
    `VLLM_DSV41_AMPERE_AG_MAX_BYTES` (default 384 KiB/rank) on PCIe."""
    if topology == "nvlink" or row_bytes <= 64:
        return max_rows
    max_bytes = int(os.environ.get("VLLM_DSV41_AMPERE_AG_MAX_BYTES", "1048576"))
    pcie_row_cap = int(os.environ.get("VLLM_DSV41_AMPERE_AG_MAX_ROWS", "192"))
    return max(1, min(max_rows, pcie_row_cap, max_bytes // max(1, row_bytes)))


def default_ampere_ar_oneshot_max_rows(topology: str) -> int:
    """Default max rows for 1-hop `OneShotAllReducePostCollapse`: 0 on both
    NVLink and PCIe Gen4 x16 (all-3-phase coalesced skip-local TwoShot beats
    OneShot even at M=1,2 on PIX Gen4 x16: -0.04ms@M=1, -0.15ms@M=2)."""
    raw = os.environ.get("VLLM_DSV41_AMPERE_AR_ONESHOT_ROWS", "").strip()
    if raw:
        return max(0, min(MAX_ROWS, int(raw)))
    return 0


def default_ampere_ar_ring_min_rows(topology: str) -> int:
    """Default min rows for single-kernel push-ring `RingAllReduce*`: 20 rows
    on PCIe Gen4 x16 (Coalesced Skip-Local TwoShot M<=9, 4-hop HDB M=10..19,
    Row-Contiguous Parallel-Reset Push-Ring M>=20), 0 on NVLink."""
    raw = os.environ.get("VLLM_DSV41_AMPERE_AR_RING_MIN_ROWS", "").strip()
    if raw:
        return max(0, min(MAX_ROWS, int(raw)))
    return 40 if topology == "pcie" else 0


class AmpereAllReduce:
    """The mailbox and per-row generations of SM80 push-Lamport all-reduce for
    one TP group; construction is collective over the group.

    Supports three dispatch regimes on PCIe P2P:
      - `rows <= oneshot_capacity`: 1-hop `OneShotAllReduce*`
      - `oneshot_capacity < rows < ring_min_rows`: 2-hop `TwoShotAllReduce*`
      - `rows >= ring_min_rows`: coalesced push-ring `RingAllReduce*`

    Args:
        group: The TP group's device process group.
        hidden: Values per row.
        device: This rank's device.
        capacity: The most rows a call takes (`MAX_ROWS` by default).
        top_k: Routes a token of the MoE outputs `finalize_post_collapse`
            takes; None compiles no finalize kernels.
        oneshot_capacity: Optional tiny-row threshold (`<= capacity`) routed to
            1-hop `OneShotAllReduce*`.
        ring_min_rows: Optional min-row threshold (`<= capacity`) routed to
            coalesced push-ring `RingAllReduce*`; 0 disables ring.

    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        hidden: int,
        device: torch.device,
        capacity: int = MAX_ROWS,
        top_k: int | None = None,
        oneshot_capacity: int = 0,
        ring_min_rows: int = 0,
    ):
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.hidden = hidden
        self.capacity = capacity
        self.top_k = top_k
        self.oneshot_capacity = (
            min(capacity, oneshot_capacity)
            if oneshot_capacity > 0 and self.world in (2, 4, 8)
            else 0
        )
        if ring_min_rows > 0 and self.world in (2, 4, 8):
            eff_ring_min = max(self.oneshot_capacity + 1, ring_min_rows)
            self.ring_min_rows = eff_ring_min if eff_ring_min <= capacity else 0
        else:
            self.ring_min_rows = 0
        # The most CTAs a call launches, one per SM (see the module docstring).
        self.ctas = current_platform.num_compute_units(device.index)
        use_w2_splitk = (
            os.environ.get("VLLM_DSV41_FUSED_W2_AR", "0") == "1"
            and hidden == 5120
        )
        two_shot_capacity = (
            capacity
            if (self.ring_min_rows == 0 or use_w2_splitk)
            else max(1, self.ring_min_rows - 1)
        )
        self.two_shot_capacity = two_shot_capacity
        self.kernel = _kernel(device.index, hidden, self.world, two_shot_capacity)
        self.collapse_kernels = [
            _collapse_kernel(
                device.index, hidden, self.world, two_shot_capacity, write_aux
            )
            for write_aux in (False, True)
        ]
        two_shot_bytes = TwoShotAllReduce(
            hidden, self.world, two_shot_capacity
        ).mailbox_bytes()
        if self.oneshot_capacity > 0:
            self.oneshot_collapse_kernels = [
                _oneshot_collapse_kernel(
                    device.index,
                    hidden,
                    self.world,
                    self.oneshot_capacity,
                    write_aux,
                    two_shot_bytes,
                )
                for write_aux in (False, True)
            ]
            oneshot_bytes = OneShotAllReducePostCollapse(
                hidden, self.world, self.oneshot_capacity, False, two_shot_bytes
            ).mailbox_bytes()
        else:
            self.oneshot_collapse_kernels = []
            oneshot_bytes = 0
        ring_base = two_shot_bytes + oneshot_bytes
        # 4-hop HDB band [hdb_min_rows .. ring_min_rows - 1] for world==4 PCIe
        # (M=10..19: beats TwoShot by -0.24..-0.31ms and Ring by -0.05..-0.19ms!)
        hdb_min_env = int(os.environ.get("VLLM_DSV41_AMPERE_AR_HDB_MIN_ROWS", "10"))
        self.hdb_min_rows = (
            hdb_min_env
            if (
                self.world == 4
                and self.ring_min_rows > hdb_min_env
                and os.environ.get("VLLM_DSV41_AMPERE_AR_HDB", "1") == "1"
            )
            else 0
        )
        self.hdb_capacity = (self.ring_min_rows - 1) if self.hdb_min_rows > 0 else 0
        if self.ring_min_rows > 0:
            self.ring_kernel = _ring_kernel(
                device.index, hidden, self.world, capacity, ring_base
            )
            self.ring_broadcast_kernel = (
                _ring_broadcast_kernel(
                    device.index, hidden, self.world, capacity, ring_base
                )
                if self.world >= 2 and self.world <= 2 * (self.world - 1)
                else None
            )
            self.ring_collapse_kernels = [
                _ring_collapse_kernel(
                    device.index, hidden, self.world, capacity, write_aux, ring_base
                )
                for write_aux in (False, True)
            ]
            ring_bytes = RingAllReduce(
                hidden, self.world, capacity, ring_base
            ).mailbox_bytes()
        else:
            self.ring_kernel = None
            self.ring_broadcast_kernel = None
            self.ring_collapse_kernels = []
            ring_bytes = 0
        hdb_base = ring_base + ring_bytes
        if self.hdb_min_rows > 0:
            self.hdb_kernel = _hdb_kernel(
                device.index, hidden, self.world, self.hdb_capacity, hdb_base
            )
            self.hdb_collapse_kernels = [
                _hdb_collapse_kernel(
                    device.index, hidden, self.world, self.hdb_capacity, write_aux, hdb_base
                )
                for write_aux in (False, True)
            ]
            hdb_bytes = HDBAllReduce(
                hidden, self.world, self.hdb_capacity, hdb_base
            ).mailbox_bytes()
        else:
            self.hdb_kernel = None
            self.hdb_collapse_kernels = []
            hdb_bytes = 0
        # Compiled here: the first finalize may come while a CUDA graph is
        # being captured.
        self.finalize_kernels = (
            []
            if top_k is None
            else [
                _finalize_kernel(
                    device.index,
                    hidden,
                    self.world,
                    two_shot_capacity,
                    write_aux,
                    top_k,
                )
                for write_aux in (False, True)
            ]
        )
        self.oneshot_finalize_kernels = (
            []
            if top_k is None or self.oneshot_capacity <= 0
            else [
                _oneshot_finalize_kernel(
                    device.index,
                    hidden,
                    self.world,
                    self.oneshot_capacity,
                    write_aux,
                    top_k,
                    two_shot_bytes,
                )
                for write_aux in (False, True)
            ]
        )
        self.ring_finalize_kernels = (
            []
            if top_k is None or self.ring_min_rows <= 0
            else [
                _ring_finalize_kernel(
                    device.index,
                    hidden,
                    self.world,
                    capacity,
                    write_aux,
                    top_k,
                    ring_base,
                )
                for write_aux in (False, True)
            ]
        )
        self.hdb_finalize_kernels = (
            []
            if top_k is None or self.hdb_min_rows <= 0
            else [
                _hdb_finalize_kernel(
                    device.index,
                    hidden,
                    self.world,
                    self.hdb_capacity,
                    write_aux,
                    top_k,
                    hdb_base,
                )
                for write_aux in (False, True)
            ]
        )
        self.w2_splitk_kernels = {}
        if use_w2_splitk:
            for write_aux in (False, True):
                for tk in (6, 3):
                    for ks_val in (4, 3):
                        self.w2_splitk_kernels[(write_aux, tk, ks_val)] = (
                            _w2_splitk_kernel(
                                device.index,
                                hidden,
                                self.world,
                                two_shot_capacity,
                                write_aux,
                                tk,
                                ks_val,
                            )
                        )
        nbytes = two_shot_bytes + oneshot_bytes + ring_bytes + hdb_bytes
        self.mailbox = symm_mem.empty(nbytes // 2, dtype=torch.bfloat16, device=device)
        handle = symm_mem.rendezvous(self.mailbox, group)
        self.peers = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)
        # Every slot starts as the sentinel, BF16 negative zero.
        self.mailbox.view(torch.int16).fill_(-32768)
        ring_gen_len = capacity if self.ring_min_rows > 0 else 0
        hdb_gen_len = self.hdb_capacity if self.hdb_min_rows > 0 else 0
        self.generations_all = torch.zeros(
            two_shot_capacity + self.oneshot_capacity + ring_gen_len + hdb_gen_len,
            dtype=torch.int32,
            device=device,
        )
        o1 = two_shot_capacity
        o2 = o1 + self.oneshot_capacity
        o3 = o2 + ring_gen_len
        self.generations = self.generations_all[:o1]
        self.oneshot_generations = self.generations_all[o1:o2]
        self.ring_generations = self.generations_all[o2:o3]
        self.hdb_generations = self.generations_all[o3:]
        torch.accelerator.synchronize()
        dist.barrier(group=group)

    def supports(self, x: torch.Tensor) -> bool:
        """Whether a call takes `x`: up to `capacity` contiguous BF16
        rows of `hidden` values, 16-byte aligned."""
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= self.capacity
            and x.shape[1] == self.hidden
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.data_ptr() % FRAGMENT_BYTES == 0
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """The sum of `x` over the TP group, as a new tensor."""
        assert self.supports(x)
        out = torch.empty_like(x)
        rows = x.shape[0]
        if self.ring_min_rows > 0 and rows >= self.ring_min_rows:
            kern = self.ring_kernel
            gens = self.ring_generations
        elif self.hdb_min_rows > 0 and rows >= self.hdb_min_rows:
            kern = self.hdb_kernel
            gens = self.hdb_generations
        else:
            kern = self.kernel
            gens = self.generations
        kern(
            Int64(x.data_ptr()),
            Int64(out.data_ptr()),
            Int64(self.peers.data_ptr()),
            Int64(gens.data_ptr()),
            Int32(self.rank),
            Int32(x.shape[0]),
            Int32(min(x.shape[0], self.ctas)),
            current_cu_stream(),
        )
        return out

    def broadcast(self, x: torch.Tensor, src_rank: int) -> torch.Tensor:
        """Pipelined unidirectional push-ring broadcast of  from ."""
        assert self.supports(x) and self.ring_broadcast_kernel is not None
        out = torch.empty_like(x)
        self.ring_broadcast_kernel(
            Int64(x.data_ptr()),
            Int64(out.data_ptr()),
            Int64(self.peers.data_ptr()),
            Int64(self.ring_generations.data_ptr()),
            Int32(self.rank),
            Int32(src_rank),
            Int32(x.shape[0]),
            Int32(min(x.shape[0], self.ctas)),
            current_cu_stream(),
        )
        return out

    def post_collapse(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        pre_mix: torch.Tensor | None,
        norm_weight: torch.Tensor,
        norm_eps: float,
        write_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The sum of `x` over the TP group, post-mixed into the streams,
        collapsed and normalized as `mhc.mhc_post_collapse` does with the
        sum: the new streams, the layer input and the stream mean (empty
        without `write_aux`). `post_mix` is `[rows, 4]` or
        `[rows, 4, 1]`, `comb_mix` `[rows, 4, 4]`; `pre_mix` None
        selects stream zero."""
        assert self.supports(x)
        rows = x.shape[0]
        if rows <= self.oneshot_capacity:
            kern = self.oneshot_collapse_kernels[int(write_aux)]
            gens = self.oneshot_generations
        elif self.ring_min_rows > 0 and rows >= self.ring_min_rows:
            kern = self.ring_collapse_kernels[int(write_aux)]
            gens = self.ring_generations
        elif self.hdb_min_rows > 0 and rows >= self.hdb_min_rows:
            kern = self.hdb_collapse_kernels[int(write_aux)]
            gens = self.hdb_generations
        else:
            kern = self.collapse_kernels[int(write_aux)]
            gens = self.generations
        streams, layer_input, aux, args, _inputs = self._collapse_args(
            rows,
            residual,
            post_mix,
            comb_mix,
            pre_mix,
            norm_weight,
            write_aux,
            generations=gens,
        )
        kern(
            Int64(x.data_ptr()),
            *args,
            Float32(norm_eps),
            current_cu_stream(),
        )
        return streams, layer_input, aux

    def finalize_post_collapse(
        self,
        routes: torch.Tensor,
        route_weights: torch.Tensor,
        shared: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        pre_mix: torch.Tensor | None,
        norm_weight: torch.Tensor,
        norm_eps: float,
        write_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`post_collapse` of a MoE output with its top-k reduction open:
        the sublayer output of token `t` is `shared[t]` plus the sum over
        `k` of `route_weights[t, k] * routes[t * top_k + k]`. `routes`
        holds the unweighted GEMM2 rows, `route_weights` the FP32 routing
        weights `[rows, top_k]`, `shared` the shared expert's rows."""
        top_k = self.top_k
        assert top_k is not None, "built without finalize kernels"
        rows = shared.shape[0]
        assert self.supports(shared)
        assert route_weights.shape == (rows, top_k)
        assert route_weights.dtype == torch.float32 and route_weights.is_contiguous()
        assert routes.dim() == 2 and routes.shape[0] >= rows * top_k
        assert routes.shape[1] == self.hidden and routes.dtype == torch.bfloat16
        assert routes.is_contiguous() and routes.data_ptr() % FRAGMENT_BYTES == 0
        if rows <= self.oneshot_capacity and self.oneshot_finalize_kernels:
            kern = self.oneshot_finalize_kernels[int(write_aux)]
            gens = self.oneshot_generations
        elif self.ring_min_rows > 0 and rows >= self.ring_min_rows:
            kern = self.ring_finalize_kernels[int(write_aux)]
            gens = self.ring_generations
        elif self.hdb_min_rows > 0 and rows >= self.hdb_min_rows and self.hdb_finalize_kernels:
            kern = self.hdb_finalize_kernels[int(write_aux)]
            gens = self.hdb_generations
        else:
            kern = self.finalize_kernels[int(write_aux)]
            gens = self.generations
        streams, layer_input, aux, args, _inputs = self._collapse_args(
            rows,
            residual,
            post_mix,
            comb_mix,
            pre_mix,
            norm_weight,
            write_aux,
            generations=gens,
        )
        kern(
            Int64(routes.data_ptr()),
            Int64(route_weights.data_ptr()),
            Int64(shared.data_ptr()),
            *args,
            Float32(norm_eps),
            current_cu_stream(),
        )
        return streams, layer_input, aux

    def w2_splitk_post_collapse(
        self,
        part2: torch.Tensor,
        topk_w: torch.Tensor,
        topk_ids: torch.Tensor,
        emap: torch.Tensor,
        shared: torch.Tensor,
        ks: int,
        top_k: int,
        global_e: int,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        pre_mix: torch.Tensor | None,
        norm_weight: torch.Tensor,
        norm_eps: float,
        write_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused W2 split-K + top-k + shared add + TwoShotAllReduce + PostCollapse."""
        rows = shared.shape[0]
        assert self.supports(shared)
        kern = self.w2_splitk_kernels[(bool(write_aux), int(top_k), int(ks))]
        streams, layer_input, aux, args, _inputs = self._collapse_args(
            rows, residual, post_mix, comb_mix, pre_mix, norm_weight, write_aux
        )
        ks_stride_bytes = rows * top_k * self.hidden * 4
        kern(
            Int64(part2.data_ptr()),
            Int64(topk_w.data_ptr()),
            Int64(topk_ids.data_ptr()),
            Int64(emap.data_ptr()),
            Int64(shared.data_ptr()),
            Int32(global_e),
            Int64(ks_stride_bytes),
            *args,
            Float32(norm_eps),
            current_cu_stream(),
        )
        return streams, layer_input, aux

    def _collapse_args(
        self,
        rows: int,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        pre_mix: torch.Tensor | None,
        norm_weight: torch.Tensor,
        write_aux: bool,
        generations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple, tuple]:
        """The outputs of a collapsing call of `rows` rows, its kernel
        arguments after the sublayer output up to the CTA count, and the
        mixes those point to, which the caller keeps until the launch (they
        may be contiguous copies)."""
        device = residual.device
        gens = self.generations if generations is None else generations
        assert residual.shape == (rows, HC, self.hidden)
        assert residual.dtype == torch.bfloat16 and residual.is_contiguous()
        assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
        post_mix = post_mix.reshape(rows, HC).contiguous()
        comb_mix = comb_mix.contiguous()
        assert post_mix.dtype == comb_mix.dtype == torch.float32
        if pre_mix is None:
            pre, pre_stride = _select_first_stream(device.index), 0
        else:
            assert pre_mix.shape == (rows, HC) and pre_mix.dtype == torch.float32
            assert pre_mix.is_contiguous()
            pre, pre_stride = pre_mix, HC
        streams = torch.empty_like(residual)
        layer_input = torch.empty(
            (rows, self.hidden), dtype=torch.bfloat16, device=device
        )
        aux = torch.empty(
            (rows if write_aux else 0, self.hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        args = (
            Int64(self.peers.data_ptr()),
            Int64(gens.data_ptr()),
            Int32(self.rank),
            Int64(residual.data_ptr()),
            Int64(post_mix.data_ptr()),
            Int64(comb_mix.data_ptr()),
            Int64(pre.data_ptr()),
            Int32(pre_stride),
            Int64(norm_weight.data_ptr()),
            Int64(streams.data_ptr()),
            Int64(layer_input.data_ptr()),
            Int64(aux.data_ptr()),
            Int32(rows),
            Int32(min(rows, self.ctas)),
        )
        return streams, layer_input, aux, args, (post_mix, comb_mix, pre)
