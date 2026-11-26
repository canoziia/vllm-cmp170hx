# SPDX-License-Identifier: Apache-2.0
"""SM80 Copy-Engine Ring DMA collective wrapper (lossless BF16).

Uses `_dsv41_pcie_dma_C` (pure `cudaMemcpyAsync` ring on dedicated Copy-Engine
streams + 1-CTA system-scope 64-bit PTX flags + <=64-CTA vectorized BF16 local
add) with a fixed O(1) triple-buffered slab (K = 3 slots regardless of
world_size) and lightweight bidirectional ACK flow-control.

Canonical chunk ownership: rank `r` owns chunk `r = [r * S : (r + 1) * S]`,
identical to `torch.distributed.reduce_scatter_single` and `all_gather_single`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_CHANNELS: dict[tuple[int, int], "PcieDmaRing | None"] = {}
_NEXT_CHAN_ID = 0
_RING_SLOTS = 3


def pcie_dma_ring_enabled() -> bool:
    raw = os.environ.get("VLLM_DSV41_PREFILL_DMA_RING", "0").strip().lower()
    return raw in ("1", "true", "yes", "auto", "force")


def prefill_row_shard_enabled() -> bool:
    raw = os.environ.get("VLLM_DSV41_PREFILL_ROW_SHARD", "0").strip().lower()
    return raw in ("1", "true", "yes", "auto", "force")


def prefill_ag_int8_enabled() -> bool:
    raw = os.environ.get("VLLM_DSV41_PREFILL_AG_INT8", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def prefill_2mb_oproj_enabled() -> bool:
    raw = os.environ.get("VLLM_DSV41_PREFILL_2MB_OPROJ", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def prefill_async_engram_enabled() -> bool:
    raw = os.environ.get("VLLM_DSV41_PREFILL_ASYNC_ENGRAM", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def prefill_2mb_moe_enabled(num_tokens: int, ep_size: int) -> bool:
    raw = os.environ.get("VLLM_DSV41_PREFILL_2MB_MOE", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "force"):
        return num_tokens >= 1024
    # Auto: Batch 1-B hardware proof on SM80 Marlin W4A16 shows splitting MoE
    # at EP=4, M<=3072 costs +2.16 ms/layer (net loss), while EP>=6 / M>=4096
    # costs only +0.66..0.93 ms/layer against 4-6+ ms collectives (big win).
    return (num_tokens >= 4096) or (ep_size >= 6 and num_tokens >= 3072)


def _load_ext():
    try:
        from vllm import _dsv41_pcie_dma_C  # type: ignore[attr-defined]

        return _dsv41_pcie_dma_C
    except Exception:
        try:
            import _dsv41_pcie_dma_C  # type: ignore[import-not-found]

            return _dsv41_pcie_dma_C
        except Exception:
            return None


class PcieDmaRing:
    """One initialized Copy-Engine Ring DMA channel over a TP group."""

    def __init__(
        self,
        *,
        ext: Any,
        chan_id: int,
        rank: int,
        world: int,
        max_chunk_bytes: int,
        device: torch.device,
        local_slab_ptr: int,
        local_flags_ptr: int,
        next_slab_ptr: int,
        next_flags_ptr: int,
        prev_flags_ptr: int,
    ) -> None:
        self._ext = ext
        self.chan_id = chan_id
        self.rank = rank
        self.world = world
        self.max_chunk_bytes = max_chunk_bytes
        self.device = device
        # Raw cudaMalloc base pointers (0-offset, safe for cudaIpcGetMemHandle).
        self._local_slab_ptr = local_slab_ptr
        self._local_flags_ptr = local_flags_ptr
        self._next_slab_ptr = next_slab_ptr
        self._next_flags_ptr = next_flags_ptr
        self._prev_flags_ptr = prev_flags_ptr

    def supports_full(self, x: torch.Tensor) -> bool:
        if (
            x.dtype != torch.bfloat16
            or not x.is_cuda
            or not x.is_contiguous()
            or x.device != self.device
        ):
            return False
        numel = x.numel()
        if numel == 0 or (numel % (self.world * 8)) != 0:
            return False
        return (numel // self.world) * 2 <= self.max_chunk_bytes

    def supports_shard(self, shard: torch.Tensor) -> bool:
        if (
            shard.dtype != torch.bfloat16
            or not shard.is_cuda
            or not shard.is_contiguous()
            or shard.device != self.device
        ):
            return False
        numel = shard.numel()
        if numel == 0 or (numel % 8) != 0:
            return False
        return numel * 2 <= self.max_chunk_bytes

    def reduce_scatter(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        s = stream if stream is not None else torch.cuda.current_stream(self.device)
        self._ext.ce_reduce_scatter(self.chan_id, inp, out, int(s.cuda_stream))

    def all_gather(
        self,
        shard: torch.Tensor,
        out: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        s = stream if stream is not None else torch.cuda.current_stream(self.device)
        self._ext.ce_all_gather(self.chan_id, shard, out, int(s.cuda_stream))

    def supports_shard_int8(self, shard: torch.Tensor) -> bool:
        if (
            not hasattr(self._ext, "ce_all_gather_int8")
            or shard.dtype != torch.bfloat16
            or not shard.is_cuda
            or not shard.is_contiguous()
            or shard.device != self.device
        ):
            return False
        numel = shard.numel()
        if numel == 0 or (numel % 32) != 0:
            return False
        packed_bytes = numel + (numel // 32) * 2
        return packed_bytes <= self.max_chunk_bytes

    def all_gather_int8(
        self,
        shard: torch.Tensor,
        out: torch.Tensor,
        q_scratch: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        s = stream if stream is not None else torch.cuda.current_stream(self.device)
        self._ext.ce_all_gather_int8(
            self.chan_id, shard, out, q_scratch, int(s.cuda_stream)
        )

    def all_reduce_inplace(
        self,
        buf: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        s = stream if stream is not None else torch.cuda.current_stream(self.device)
        self._ext.ce_all_reduce_inplace(self.chan_id, buf, int(s.cuda_stream))


def get_or_init_pcie_dma_ring(
    tp_group: Any,
    hidden_size: int,
    max_tokens: int,
) -> PcieDmaRing | None:
    """Collectively initialize (or fetch cached) Copy-Engine Ring DMA channel."""
    global _NEXT_CHAN_ID
    if not pcie_dma_ring_enabled():
        return None
    world = tp_group.world_size
    if world < 2 or world > 16:
        return None
    rank = tp_group.rank_in_group
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    key = (id(tp_group), int(device.index or 0))
    if key in _CHANNELS:
        return _CHANNELS[key]

    ext = _load_ext()
    if ext is None:
        logger.warning_once(
            "DSV4.1 PCIe DMA Ring requested but _dsv41_pcie_dma_C is unavailable; "
            "falling back to PyNCCL."
        )
        _CHANNELS[key] = None
        return None

    cpu_group = tp_group.cpu_group
    # 1. Verify bidirectional CUDA P2P access along the ring on every rank.
    my_dev = int(device.index or 0)
    all_devs: list[int | None] = [None] * world
    dist.all_gather_object(all_devs, my_dev, group=cpu_group)
    next_rank = (rank + 1) % world
    prev_rank = (rank - 1 + world) % world
    next_dev = int(all_devs[next_rank])  # type: ignore[arg-type]
    prev_dev = int(all_devs[prev_rank])  # type: ignore[arg-type]
    peer_ok = bool(ext.can_access_peer(my_dev, next_dev)) and bool(
        ext.can_access_peer(my_dev, prev_dev)
    )
    all_ok: list[bool | None] = [None] * world
    dist.all_gather_object(all_ok, peer_ok, group=cpu_group)
    if not all(all_ok):
        logger.info_once(
            "DSV4.1 PCIe DMA Ring: P2P access unavailable along ring (%s); "
            "falling back to PyNCCL.",
            all_ok,
        )
        _CHANNELS[key] = None
        return None

    # 2. Size O(1) triple-buffered slab (K=3 fixed slots independent of world_size).
    cap_tokens = max(1024, min(int(max_tokens), 8192))
    aligned_tokens = ((cap_tokens + world - 1) // world) * world
    chunk_rows = aligned_tokens // world
    max_chunk_bytes = chunk_rows * hidden_size * 2
    # 16-byte align
    max_chunk_bytes = ((max_chunk_bytes + 15) // 16) * 16
    num_slots = int(ext.ring_num_slots()) if hasattr(ext, "ring_num_slots") else _RING_SLOTS
    slab_bytes = num_slots * max_chunk_bytes

    # 2 * num_slots 128-byte cachelines (data + ack flags), page-aligned.
    flags_bytes = 4096
    local_slab_ptr = 0
    local_flags_ptr = 0
    next_slab_ptr = 0
    next_flags_ptr = 0
    prev_flags_ptr = 0
    try:
        # Allocate dedicated 0-offset cudaMalloc buffers outside PyTorch's
        # CUDACachingAllocator (cudaIpcGetMemHandle ignores sub-block offsets!).
        local_slab_ptr = int(ext.alloc_ipc_buffer(slab_bytes))
        local_flags_ptr = int(ext.alloc_ipc_buffer(flags_bytes))
        h_slab = ext.get_ipc_handle_bytes(local_slab_ptr)
        h_flags = ext.get_ipc_handle_bytes(local_flags_ptr)
        all_slabs: list[Any] = [None] * world
        all_flags: list[Any] = [None] * world
        dist.all_gather_object(all_slabs, h_slab, group=cpu_group)
        dist.all_gather_object(all_flags, h_flags, group=cpu_group)

        next_slab_ptr = int(ext.open_ipc_handle_bytes(all_slabs[next_rank]))
        next_flags_ptr = int(ext.open_ipc_handle_bytes(all_flags[next_rank]))
        # For world == 2, prev_rank == next_rank: reuse the open IPC handle
        # rather than opening the same handle twice (cudaErrorAlreadyMapped).
        if prev_rank == next_rank:
            prev_flags_ptr = next_flags_ptr
        else:
            prev_flags_ptr = int(ext.open_ipc_handle_bytes(all_flags[prev_rank]))

        chan_id = _NEXT_CHAN_ID
        _NEXT_CHAN_ID += 1
        ext.init_channel(
            chan_id,
            rank,
            world,
            max_chunk_bytes,
            local_slab_ptr,
            next_slab_ptr,
            local_flags_ptr,
            next_flags_ptr,
            prev_flags_ptr,
        )
        dist.barrier(group=cpu_group)
    except Exception as exc:
        if prev_flags_ptr and prev_flags_ptr != next_flags_ptr:
            ext.close_ipc_handle(prev_flags_ptr)
        if next_flags_ptr:
            ext.close_ipc_handle(next_flags_ptr)
        if next_slab_ptr:
            ext.close_ipc_handle(next_slab_ptr)
        if local_slab_ptr:
            ext.free_ipc_buffer(local_slab_ptr)
        if local_flags_ptr:
            ext.free_ipc_buffer(local_flags_ptr)
        logger.warning_once(
            "DSV4.1 PCIe DMA Ring initialization failed (%s); falling back to PyNCCL.",
            exc,
        )
        _CHANNELS[key] = None
        return None

    ring = PcieDmaRing(
        ext=ext,
        chan_id=chan_id,
        rank=rank,
        world=world,
        max_chunk_bytes=max_chunk_bytes,
        device=device,
        local_slab_ptr=local_slab_ptr,
        local_flags_ptr=local_flags_ptr,
        next_slab_ptr=next_slab_ptr,
        next_flags_ptr=next_flags_ptr,
        prev_flags_ptr=prev_flags_ptr,
    )
    _CHANNELS[key] = ring
    logger.info_once(
        "DSV4.1 Prefill: Copy-Engine Ring DMA enabled (world=%d, slots=%d, "
        "max_tokens=%d, slab=%.1f MiB/rank).",
        world,
        num_slots,
        aligned_tokens,
        slab_bytes / (1024 * 1024),
    )
    return ring
