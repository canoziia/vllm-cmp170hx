# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-kernel Lamport P2P Markov argmax reduction + finalize on SM80/SM90."""

from functools import cache

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
    current_cu_stream,
)

from .codecs import f32_from_bits, load_s32, store_u32

WARP_THREADS = 32
PACKED_EMPTY = -(1 << 63)


def _asm(result, operands, text, constraints, side_effects, loc, ip):
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
def load_s64(address: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.s64 $0, [$1];",
            "=l,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def store_s64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "st.global.s64 [$0], $1;",
        "l,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def load_u64_nc(address: Int64, *, loc=None, ip=None) -> Int64:
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
def load_volatile_u64(address: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u64 $0, [$1];",
            "=l,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def store_volatile_u64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "st.volatile.global.u64 [$0], $1;",
        "l,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def pack_b64(lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, {$1, $2};",
            "=l,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def unpack_b64_lo(word: Int64, *, loc=None, ip=None) -> Uint32:
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(word).ir_value(loc=loc, ip=ip)],
        "mov.b64 {$0, $1}, $2;",
        "=r,=r,l",
        False,
        loc,
        ip,
    )
    return Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip))


@dsl_user_op
def unpack_b64_hi(word: Int64, *, loc=None, ip=None) -> Uint32:
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(word).ir_value(loc=loc, ip=ip)],
        "mov.b64 {$0, $1}, $2;",
        "=r,=r,l",
        False,
        loc,
        ip,
    )
    return Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip))


class MarkovSymmKernel:
    """Fuses `_markov_reduce_scatter`, Lamport P2P 8-byte exchange, and
    `_markov_finalize` into one single-warp launch per draft step."""

    def __init__(self, world: int, max_reqs: int, has_d2t: bool):
        self.world = world
        self.max_reqs = max_reqs
        self.has_d2t = has_d2t

    def mailbox_bytes(self) -> int:
        return LAMPORT_GENERATIONS * self.world * self.max_reqs * 8

    @cute.jit
    def __call__(
        self,
        part: Int64,
        peers: Int64,
        generations: Int64,
        d2t: Int64,
        prev: Int64,
        draft_out: Int64,
        nblk: Int32,
        vocab_start: Int32,
        rank: Int32,
        draft_stride: Int32,
        step: Int32,
        draft_limit: Int32,
        target_limit: Int32,
        num_reqs: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            red: cute.struct.MemRange[Int64, WARP_THREADS]
            got: cute.struct.MemRange[Int64, self.world]

        self.kernel(
            part,
            peers,
            generations,
            d2t,
            prev,
            draft_out,
            nblk,
            vocab_start,
            rank,
            draft_stride,
            step,
            draft_limit,
            target_limit,
            Storage,
        ).launch(grid=(num_reqs, 1, 1), block=[WARP_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        part: Int64,
        peers: Int64,
        generations: Int64,
        d2t: Int64,
        prev: Int64,
        draft_out: Int64,
        nblk: Int32,
        vocab_start: Int32,
        rank: Int32,
        draft_stride: Int32,
        step: Int32,
        draft_limit: Int32,
        target_limit: Int32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        red = storage.red.get_tensor(cute.make_layout(WARP_THREADS))
        got_smem = storage.got.get_tensor(cute.make_layout(self.world))

        best = Int64(PACKED_EMPTY)
        row_base = part + Int64(b) * Int64(nblk) * 8
        k = tid
        while k < nblk:
            v = load_s64(row_base + Int64(k) * 8)
            if v > best:
                best = v
            k += WARP_THREADS
        red[tid] = best
        cute.arch.barrier()

        if tid == 0:
            wbest = red[0]
            for lane in cutlass.range_constexpr(WARP_THREADS):
                if lane > 0:
                    cand = red[lane]
                    if cand > wbest:
                        wbest = cand
            red[0] = wbest
        cute.arch.barrier()

        best = red[0]
        lo_u32 = unpack_b64_lo(best)
        hi_s32 = Int32(unpack_b64_hi(best))
        bits_s32 = hi_s32 ^ Int32(0x7FFFFFFF) if hi_s32 < Int32(0) else hi_s32
        val_bits = Uint32(bits_s32)
        gid = (Int32(0x7FFFFFFF) - Int32(lo_u32)) + vocab_start
        if gid < Int32(0):
            gid = Int32(0)
        payload = pack_b64(val_bits, Uint32(gid + Int32(1)))

        gen_addr = generations + Int64(b) * 4
        gen = load_s32(gen_addr)

        if tid < self.world:
            dst_base = load_u64_nc(peers + Int64(tid) * 8)
            push_off = (
                (Int64(gen) * self.world + Int64(rank)) * self.max_reqs + Int64(b)
            ) * 8
            store_volatile_u64(dst_base + push_off, payload)

            my_base = load_u64_nc(peers + Int64(rank) * 8)
            pull_off = (
                (Int64(gen) * self.world + Int64(tid)) * self.max_reqs + Int64(b)
            ) * 8
            w = load_volatile_u64(my_base + pull_off)
            while w == Int64(0):
                w = load_volatile_u64(my_base + pull_off)
            store_volatile_u64(my_base + pull_off, Int64(0))
            got_smem[tid] = w
        cute.arch.barrier()

        if tid == 0:
            store_u32(gen_addr, Uint32((gen + Int32(1)) % LAMPORT_GENERATIONS))

            vals = cute.make_rmem_tensor(cute.make_layout(self.world), Float32)
            gids = cute.make_rmem_tensor(cute.make_layout(self.world), Int32)
            any_nan = Int32(0)
            best_val = f32_from_bits(Uint32(0xFF800000))  # -inf
            fallback_gid = Int32(0)
            for s in cutlass.range_constexpr(self.world):
                gw = got_smem[s]
                vb = unpack_b64_lo(gw)
                gp1 = Int32(unpack_b64_hi(gw))
                g = gp1 - Int32(1)
                vf = f32_from_bits(vb)
                vals[s] = vf
                gids[s] = g
                if (vb & Uint32(0x7FFFFFFF)) > Uint32(0x7F800000):
                    any_nan = Int32(1)
                if vf > best_val:
                    best_val = vf
                if s == 0 or g > fallback_gid:
                    fallback_gid = g

            chosen = fallback_gid
            if any_nan == Int32(0):
                first_hit = Int32(1)
                for s in cutlass.range_constexpr(self.world):
                    if vals[s] == best_val:
                        if first_hit == Int32(1) or gids[s] < chosen:
                            chosen = gids[s]
                            first_hit = Int32(0)

            if chosen < Int32(0):
                chosen = Int32(0)
            if chosen >= draft_limit:
                chosen = draft_limit - Int32(1)
            out_id = Int64(chosen)
            if cutlass.const_expr(self.has_d2t):
                out_id = out_id + load_s64(d2t + out_id * 8)
                if out_id < Int64(0):
                    out_id = Int64(0)
                if out_id >= Int64(target_limit):
                    out_id = Int64(target_limit - Int32(1))

            store_s64(prev + Int64(b) * 8, out_id)
            store_s64(
                draft_out + (Int64(b) * Int64(draft_stride) + Int64(step)) * 8, out_id
            )


@cache
def _compile_markov_symm(
    device_index: int, world: int, max_reqs: int, has_d2t: bool
):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            MarkovSymmKernel(world, max_reqs, has_d2t),
            *(Int64(0),) * 6,
            *(Int32(1),) * 8,
            make_fake_stream(),
        )


class MarkovSymmExchange:
    """Collective symmetric-memory Lamport exchange for DSpark's Markov head."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        max_reqs: int,
        has_d2t: bool,
        device: torch.device,
    ):
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.max_reqs = max_reqs
        self.has_d2t = has_d2t
        self.kernel = _compile_markov_symm(device.index, self.world, max_reqs, has_d2t)
        nbytes = MarkovSymmKernel(self.world, max_reqs, has_d2t).mailbox_bytes()
        self.mailbox = symm_mem.empty(nbytes // 8, dtype=torch.int64, device=device)
        handle = symm_mem.rendezvous(self.mailbox, group)
        self.peers = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)
        self.mailbox.zero_()
        self.generations = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        torch.accelerator.synchronize()
        dist.barrier(group=group)

    def step(
        self,
        part: torch.Tensor,
        d2t: torch.Tensor | None,
        prev: torch.Tensor,
        draft_tokens: torch.Tensor,
        nblk: int,
        vocab_start: int,
        step_idx: int,
        draft_limit: int,
        target_limit: int,
        num_reqs: int,
    ) -> None:
        if num_reqs <= 0:
            return
        self.kernel(
            Int64(part.data_ptr()),
            Int64(self.peers.data_ptr()),
            Int64(self.generations.data_ptr()),
            Int64(d2t.data_ptr() if d2t is not None else 0),
            Int64(prev.data_ptr()),
            Int64(draft_tokens.data_ptr()),
            Int32(nblk),
            Int32(vocab_start),
            Int32(self.rank),
            Int32(draft_tokens.stride(0)),
            Int32(step_idx),
            Int32(draft_limit),
            Int32(target_limit),
            Int32(num_reqs),
            current_cu_stream(),
        )
