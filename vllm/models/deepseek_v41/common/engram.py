# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram: n-gram hash lookups gated into the hyper-connection stream.

Port of the reference `inference/engram.py` + `Engram` /
`ParallelEngramEmbedding` from `inference/model.py` (DeepSeek V4.1
checkpoint layout). Engram modules live on the backbone layers listed in
`engram_layer_ids` only.

An n-gram at position `p` needs the token ids at `p-1..p-3` while vLLM
streams tokens chunk-by-chunk:

- `token_map`: token id -> compressed vocab id, built once from the
  model's tokenizer at init (deterministic; asserted against
  `engram_compressed_vocab_size`).
- `lookback_token_ids`: the ids just before each request's chunk start,
  which the runner reads from its device-resident token history.
"""

import ctypes
import hashlib
import mmap
import os
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import weakref
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from functools import cache
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

import vllm.envs as envs

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_engram_dp_group,
    get_engram_dp_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.device_communicators.shm_broadcast import (
    SHM_PATH,
    check_shm_free_space,
)
from vllm.distributed.parallel_state import GroupCoordinator, in_the_same_node_as
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.ops.query_quant import (
    can_fuse_query_quant,
    mxfp8_scale_bytes,
    mxfp8_scale_offset,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.cpu_resource_utils import get_available_memory
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

# Per-rank token slot from which the Engram DP exchange switches from peer
# reads to all-to-all.
ENGRAM_A2A_MIN_SLOT = 128

# Cache value for tokens that take no part in an n-gram (image spans).
DEAD_ID = -1


def _engram_lookup_thresholds(device: torch.device) -> tuple[int | None, int | None]:
    """Host lookup thresholds on `device`: the rows from which to sort, and the
    tokens (max over DP ranks) from which to run inline. None disables that path."""
    if current_platform.is_cuda() and current_platform.is_device_capability_family(
        100, device.index or 0
    ):
        # Measured on GB200.
        return 36864, 4096
    if current_platform.is_rocm():
        # Measured on MI355X with a 94 GiB table: unsorted lookups from 49K rows
        # take 4x longer than sorted ones; never inline.
        return 32768, None
    # TODO: Add hand-tuned thresholds for other GPU families.
    return None, None


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32 (avoids a sympy import)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize
    alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE"
    all hash the same way. The compressed size matters beyond bounds checking:
    every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # A private-use char, so a token that is exactly one space survives
    # Strip() instead of collapsing to the empty string and merging with
    # unrelated tokens.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # The raw Rust tokenizer, matching what training decodes with
    # (no clean_up_tokenization_spaces).
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token: nothing to normalize, so key it
            # by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers
    hash differently. Kept odd and bounded so `token_id * multiplier` cannot
    overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max), each
    split over `n_heads` heads. Every (n-gram size, head) pair owns its own
    prime-sized bucket range in the layer's table; the primes are drawn in
    order and never reused, which keeps the ranges disjoint.
    """

    def __init__(self, config) -> None:
        self.layer_ids: tuple[int, ...] = tuple(config.engram_layer_ids)
        self.num_embeddings: tuple[int, ...] = tuple(config.engram_num_embeddings)
        self.max_ngram_size: int = config.engram_max_ngram_size
        self.n_heads: int = config.engram_n_heads
        self.head_dim: int = config.engram_head_dim
        self.compressed_vocab_size: int = config.engram_compressed_vocab_size
        self.pad_token_id: int = config.engram_pad_token_id
        assert len(self.layer_ids) == len(self.num_embeddings)

        quant = getattr(config, "quantization_config", None) or {}
        embed_quant = quant.get("engram_embed_quant", "fp8")
        if embed_quant not in ("fp8", "mxfp4"):
            raise ValueError(f"Unsupported engram_embed_quant {embed_quant!r}")
        for key, expected in (
            ("engram_embed_block_size", 32),
            ("engram_embed_scale_dtype", "ue8m0"),
        ):
            if quant.get(key, expected) != expected:
                raise ValueError(f"{key} must be {expected!r}")
        embed_layers = quant.get("engram_embed_layers", self.layer_ids)
        if set(embed_layers) != set(self.layer_ids):
            raise ValueError("engram_embed_layers must match engram_layer_ids")
        self.packed_fp4 = embed_quant == "mxfp4"
        self.row_bytes = self.head_dim // 2 if self.packed_fp4 else self.head_dim

        primes = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        self.primes: tuple[tuple[tuple[int, ...], ...], ...] = tuple(primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.offsets = torch.tensor(np.array(offsets))  # [n_layers, n_hash_cols]

    @classmethod
    def from_config(cls, config) -> "EngramLayout | None":
        if not getattr(config, "engram_layer_ids", None):
            return None
        return cls(config)


# Keep request shapes out of the cache key so one warmed variant covers runtime batches.
@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_query_rows",
    ]
)
def _hash_ids_kernel(
    input_ids,
    token_map,
    dead_mask,
    positions,
    query_start_loc,
    multipliers,
    primes,
    offsets,
    lookback_token_ids,
    lookback_dead_mask,
    output,
    num_tokens,
    pad_id,
    input_stride,
    mask_stride,
    position_stride,
    query_stride,
    num_query_rows,
    MAX_NGRAM: tl.constexpr,
    num_heads,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    dead_id,
    lookback_depth,
    lookback_row_stride,
    lookback_col_stride,
    lookback_mask_row_stride,
    lookback_mask_col_stride,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    num_layers = tl.num_programs(1)
    valid = token < num_tokens
    # Upper bound in query_start_loc[1:], including repeated padding boundaries.
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.full((BLOCK_T,), num_query_rows, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(
            query_start_loc + (mid + 1) * query_stride,
            lo < hi,
            other=0,
        )
        right = token >= end
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    req = tl.minimum(lo, num_query_rows - 1).to(tl.int64)
    chunk_idx = tl.load(query_start_loc + req * query_stride)
    chunk_idx = tl.minimum(chunk_idx, num_tokens - 1).to(tl.int64)
    chunk_start = tl.load(positions + chunk_idx * position_stride)
    position = tl.load(positions + token * position_stride, valid, other=0).to(tl.int64)
    head = tl.arange(0, BLOCK_H)
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(MAX_NGRAM):
        lookback = position - shift
        in_batch = lookback >= chunk_start
        batch_idx = tl.maximum(token - shift, 0)
        batch_token = tl.load(
            input_ids + batch_idx * input_stride, valid & in_batch, other=0
        )
        batch_source = tl.load(token_map + batch_token, valid & in_batch, other=0)
        batch_dead = tl.load(
            dead_mask + batch_idx * mask_stride, valid & in_batch, other=False
        )
        batch_source = tl.where(batch_dead, dead_id, batch_source)

        col = chunk_start - 1 - lookback
        in_window = valid & ~in_batch & (col >= 0) & (col < lookback_depth)
        col = tl.minimum(tl.maximum(col, 0), lookback_depth - 1)
        window_token = tl.load(
            lookback_token_ids + req * lookback_row_stride + col * lookback_col_stride,
            in_window,
            other=-1,
        )
        known = in_window & (window_token >= 0)
        window_source = tl.load(token_map + window_token, known, other=0)
        window_dead = tl.load(
            lookback_dead_mask
            + req * lookback_mask_row_stride
            + col * lookback_mask_col_stride,
            known,
            other=False,
        )
        window_source = tl.where(window_dead, dead_id, window_source)

        source = tl.where(
            in_batch, batch_source, tl.where(known, window_source, pad_id)
        ).to(tl.int64)
        blocked |= (lookback < 0) | (source == dead_id)
        value = tl.where(blocked, pad_id, source)
        multiplier = tl.load(multipliers + layer * MAX_NGRAM + shift)
        rolling ^= value * multiplier
        if shift > 0:
            col = (shift - 1) * num_heads + head
            param_offset = layer * (MAX_NGRAM - 1) * num_heads + col
            prime = tl.load(primes + param_offset, head < num_heads, other=1)
            offset = tl.load(offsets + param_offset, head < num_heads, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            out_offset = (token.to(tl.int64) * num_layers + layer)[:, None] * (
                (MAX_NGRAM - 1) * num_heads
            ) + col[None, :]
            tl.store(output + out_offset, hashed, valid[:, None] & (head < num_heads))


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Stateless: the runner supplies every lookback token id.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layout: EngramLayout,
        swa_cache_module: nn.Module,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.swa_cache_module = swa_cache_module
        self.lookback_depth: int = layout.max_ngram_size - 1

        model_config = vllm_config.model_config
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != layout.compressed_vocab_size:
            raise ValueError(
                f"Compressed vocab size mismatch: built {vocab_size} from the "
                f"tokenizer, config expects {layout.compressed_vocab_size}; "
                "every hash multiplier derives from it, so the engram tables "
                "would be silently rehashed."
            )
        self.pad_id = token_map[layout.pad_token_id]
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer(
            "token_map", torch.tensor(token_map, dtype=torch.int32), persistent=False
        )
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", layout.offsets, persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        logger.info(
            "Built engram token map (%d -> %d ids) for layers %s",
            len(token_map),
            vocab_size,
            layout.layer_ids,
        )

    def ensure_cache(self) -> bool:
        """Return False while the KV cache is unbound (profile run); the caller
        skips engram hashing then."""
        return self.swa_cache_module.kv_cache.numel() > 0

    def dummy_hashes(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Participate in DP lookups without valid rows."""
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        hashes = input_ids.new_full(
            (num_tokens, num_layers, (max_ngram - 1) * num_heads),
            DEAD_ID,
            dtype=torch.int32,
        )
        keep = torch.zeros(num_tokens, dtype=torch.bool, device=input_ids.device)
        return hashes, keep

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute [tokens, layers, hash columns] int32 n-gram hashes.

        History comes from the current chunk, then the runner's lookback
        window.
        """
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        output = input_ids.new_empty(
            (num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int32
        )
        if num_tokens == 0:
            return output
        _hash_ids_kernel[(triton.cdiv(num_tokens, 32), num_layers)](
            input_ids,
            self.token_map,
            dead_mask,
            positions,
            query_start_loc,
            self.multipliers,
            self.primes,
            self.offsets,
            lookback_token_ids,
            lookback_dead_mask,
            output,
            num_tokens,
            self.pad_id,
            input_stride=input_ids.stride(0),
            mask_stride=dead_mask.stride(0),
            position_stride=positions.stride(0),
            query_stride=query_start_loc.stride(0),
            num_query_rows=query_start_loc.numel() - 1,
            MAX_NGRAM=max_ngram,
            num_heads=num_heads,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(num_heads),
            dead_id=DEAD_ID,
            lookback_depth=lookback_token_ids.shape[1],
            lookback_row_stride=lookback_token_ids.stride(0),
            lookback_col_stride=lookback_token_ids.stride(1),
            lookback_mask_row_stride=lookback_dead_mask.stride(0),
            lookback_mask_col_stride=lookback_dead_mask.stride(1),
            num_warps=4,
        )
        return output


def _engram_prefetch_sms() -> int:
    return int(os.environ.get("VLLM_DSV41_ENGRAM_PREFETCH_SMS", "0"))


def _engram_numa_bind_enabled() -> bool:
    return os.environ.get("VLLM_DSV41_ENGRAM_NUMA_BIND", "0") == "1"


def _engram_packed_host_enabled() -> bool:
    return os.environ.get("VLLM_DSV41_ENGRAM_PACKED_HOST", "0") == "1"


def _engram_packed_stride(min_row_bytes: int) -> int:
    default_stride = ((min_row_bytes + 15) // 16) * 16
    stride = int(
        os.environ.get("VLLM_DSV41_ENGRAM_PACKED_STRIDE", str(default_stride))
    )
    if stride < min_row_bytes:
        raise ValueError(
            f"VLLM_DSV41_ENGRAM_PACKED_STRIDE={stride} < minimum row size {min_row_bytes}"
        )
    return stride


def _engram_cpu_prefetch_enabled() -> bool:
    return os.environ.get("VLLM_DSV41_ENGRAM_CPU_PREFETCH", "0") == "1"


def _engram_cpu_min_tokens() -> int:
    return max(1, int(os.environ.get("VLLM_DSV41_ENGRAM_CPU_MIN_TOKENS", "256")))


def _engram_cpu_compact_h2d_enabled() -> bool:
    return os.environ.get("VLLM_DSV41_ENGRAM_CPU_COMPACT_H2D", "0") == "1"


def _is_graph_capture_active() -> bool:
    if torch.cuda.is_current_stream_capturing():
        return True
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    capture = BreakableCUDAGraphCapture.current()
    return bool(capture is not None and capture._capturing)


def _parse_cpulist(cpulist: str) -> set[int]:
    cpus: set[int] = set()
    for part in cpulist.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            cpus.update(range(int(lo_s), int(hi_s) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _get_cuda_device_numa_info(
    device_index: int | None = None,
) -> tuple[str, int, set[int]] | None:
    """Return `(pci_bus_id, numa_node, node_cpus)` for `device_index`, or None."""
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        return None
    try:
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        err, raw_bus = cudart.cudaDeviceGetPCIBusId(64, int(device_index))
        if int(err) != 0:
            return None
        if isinstance(raw_bus, bytes):
            pci = raw_bus.split(b"\x00", 1)[0].decode("ascii").strip().lower()
        else:
            pci = str(raw_bus).split("\x00", 1)[0].strip().lower()
        if not pci:
            return None

        candidates = [pci]
        if pci.startswith("00000000:"):
            candidates.append("0000:" + pci[9:])
        elif pci.startswith("0000:"):
            candidates.append("00000000:" + pci[5:])

        numa_node = -1
        resolved_pci = pci
        for cand in candidates:
            numa_path = Path(f"/sys/bus/pci/devices/{cand}/numa_node")
            if numa_path.is_file():
                numa_node = int(numa_path.read_text().strip())
                resolved_pci = cand
                break
        if numa_node < 0:
            return None

        cpulist_path = Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
        if not cpulist_path.is_file():
            return None
        cpus = _parse_cpulist(cpulist_path.read_text().strip())
        if not cpus:
            return None
        return resolved_pci, numa_node, cpus
    except Exception:
        return None


@contextmanager
def _numa_bind_current_gpu(
    device_index: int | None = None, strict_bind: bool = True
):
    """Bind CPU affinity & NUMA mempolicy (or MPOL_INTERLEAVE) during host alloc."""
    if not _engram_numa_bind_enabled() or not hasattr(os, "sched_getaffinity"):
        yield None
        return
    if os.environ.get("VLLM_DSV41_ENGRAM_NUMA_INTERLEAVE", "0") == "1":
        libc = None
        saved_pol = False
        orig_mode = ctypes.c_int(0)
        orig_mask = ctypes.c_ulong(0)
        if platform.system() == "Linux" and platform.machine().lower() in (
            "x86_64",
            "amd64",
        ):
            try:
                node_dirs = [
                    int(p.name[4:])
                    for p in Path("/sys/devices/system/node").glob("node[0-9]*")
                    if p.name[4:].isdigit() and int(p.name[4:]) < 63
                ]
                if len(node_dirs) > 1:
                    libc = ctypes.CDLL("libc.so.6", use_errno=True)
                    rc = libc.syscall(
                        ctypes.c_long(239),
                        ctypes.byref(orig_mode),
                        ctypes.byref(orig_mask),
                        ctypes.c_ulong(64),
                        ctypes.c_void_p(0),
                        ctypes.c_ulong(0),
                    )
                    if rc == 0:
                        mask_val = 0
                        for nd in node_dirs:
                            mask_val |= 1 << nd
                        target_mask = ctypes.c_ulong(mask_val)
                        rc_set = libc.syscall(
                            ctypes.c_long(238),
                            ctypes.c_int(3),
                            ctypes.byref(target_mask),
                            ctypes.c_ulong(64),
                        )
                        saved_pol = rc_set == 0
            except Exception:
                saved_pol = False
        try:
            yield ("interleave", -1) if saved_pol else None
        finally:
            if saved_pol and libc is not None:
                try:
                    if orig_mode.value == 0:
                        libc.syscall(
                            ctypes.c_long(238),
                            ctypes.c_int(0),
                            ctypes.c_void_p(0),
                            ctypes.c_ulong(0),
                        )
                    else:
                        libc.syscall(
                            ctypes.c_long(238),
                            orig_mode,
                            ctypes.byref(orig_mask),
                            ctypes.c_ulong(64),
                        )
                except Exception:
                    pass
        return
    info = _get_cuda_device_numa_info(device_index)
    if info is None:
        yield None
        return
    pci, numa_node, node_cpus = info
    try:
        orig_affinity = os.sched_getaffinity(0)
    except OSError:
        yield None
        return

    libc = None
    saved_pol = False
    orig_mode = ctypes.c_int(0)
    orig_mask = ctypes.c_ulong(0)
    if (
        platform.system() == "Linux"
        and platform.machine().lower() in ("x86_64", "amd64")
        and 0 <= numa_node < 63
    ):
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            SYS_set_mempolicy, SYS_get_mempolicy = 238, 239
            MPOL_PREFERRED, MPOL_BIND = 1, 2
            rc = libc.syscall(
                ctypes.c_long(SYS_get_mempolicy),
                ctypes.byref(orig_mode),
                ctypes.byref(orig_mask),
                ctypes.c_ulong(64),
                ctypes.c_void_p(0),
                ctypes.c_ulong(0),
            )
            if rc == 0:
                target_mask = ctypes.c_ulong(1 << numa_node)
                pol_mode = MPOL_BIND if strict_bind else MPOL_PREFERRED
                rc_set = libc.syscall(
                    ctypes.c_long(SYS_set_mempolicy),
                    ctypes.c_int(pol_mode),
                    ctypes.byref(target_mask),
                    ctypes.c_ulong(64),
                )
                saved_pol = rc_set == 0
        except Exception:
            saved_pol = False

    bound = False
    try:
        try:
            os.sched_setaffinity(0, node_cpus)
            bound = True
        except OSError:
            intersect = node_cpus & orig_affinity
            if intersect:
                os.sched_setaffinity(0, intersect)
                bound = True
        yield (pci, numa_node) if (bound or saved_pol) else None
    finally:
        if saved_pol and libc is not None:
            try:
                if orig_mode.value == 0:
                    libc.syscall(
                        ctypes.c_long(238),
                        ctypes.c_int(0),
                        ctypes.c_void_p(0),
                        ctypes.c_ulong(0),
                    )
                else:
                    libc.syscall(
                        ctypes.c_long(238),
                        orig_mode,
                        ctypes.byref(orig_mask),
                        ctypes.c_ulong(64),
                    )
            except Exception:
                pass
        if bound:
            try:
                os.sched_setaffinity(0, orig_affinity)
            except OSError:
                pass


_CPU_GATHER_LIB: ctypes.CDLL | None = None
_CPU_GATHER_LOCK = threading.Lock()


def _load_engram_cpu_lib() -> ctypes.CDLL:
    """Load or JIT-compile `engram_cpu_gather.cpp` (`engram_cpu_gather_bf16`)."""
    global _CPU_GATHER_LIB
    if _CPU_GATHER_LIB is not None:
        return _CPU_GATHER_LIB
    with _CPU_GATHER_LOCK:
        if _CPU_GATHER_LIB is not None:
            return _CPU_GATHER_LIB

        env_lib = os.environ.get("VLLM_DSV41_ENGRAM_CPU_LIB", "").strip()
        here = Path(__file__).resolve().parent
        prebuilt = here / "_dsv41_engram_cpu.so"
        so_path: Path | None = None
        if env_lib:
            so_path = Path(env_lib)
            if not so_path.is_file():
                raise FileNotFoundError(
                    f"VLLM_DSV41_ENGRAM_CPU_LIB not found: {so_path}"
                )
        elif prebuilt.is_file():
            so_path = prebuilt
        else:
            src_path = here / "engram_cpu_gather.cpp"
            if not src_path.is_file():
                raise FileNotFoundError(f"Missing C++ gather source: {src_path}")
            machine = platform.machine().lower()
            digest = hashlib.sha256(
                src_path.read_bytes() + machine.encode("utf-8")
            ).hexdigest()[:12]
            cache_dir = Path(tempfile.gettempdir())
            cached_so = cache_dir / f"dsv41_engram_cpu_{os.getuid()}_{digest}.so"
            if not cached_so.is_file():
                tmp_so = cache_dir / (
                    f".dsv41_engram_cpu_{os.getuid()}_{digest}"
                    f".{os.getpid()}.{threading.get_ident()}.so"
                )
                cxx = os.environ.get("CXX", "g++")
                arch_flags = (
                    ["-mavx2", "-mfma"] if machine in ("x86_64", "amd64") else []
                )
                cmd = [
                    cxx,
                    "-O3",
                    "-std=c++17",
                    *arch_flags,
                    "-fopenmp",
                    "-shared",
                    "-fPIC",
                    str(src_path),
                    "-o",
                    str(tmp_so),
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    os.replace(tmp_so, cached_so)
                finally:
                    if tmp_so.exists():
                        try:
                            tmp_so.unlink()
                        except OSError:
                            pass
            so_path = cached_so

        lib = ctypes.CDLL(str(so_path))
        lib.engram_cpu_gather_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        lib.engram_cpu_gather_bf16.restype = None
        lib.engram_cpu_gather_fp8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
        ]
        lib.engram_cpu_gather_fp8.restype = None
        _CPU_GATHER_LIB = lib
        return lib


class _AsyncPrefetchTask:
    __slots__ = ("done", "exc")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.exc: BaseException | None = None

    def wait(self) -> None:
        self.done.wait()
        if self.exc is not None:
            raise self.exc


class _EngramAsyncWorker:
    """Lightweight single-thread daemon worker for async CPU gather + H2D."""

    def __init__(self, device_index: int) -> None:
        self._device_index = device_index
        self._q: queue.SimpleQueue[tuple[object, _AsyncPrefetchTask]] = (
            queue.SimpleQueue()
        )
        self._thread = threading.Thread(
            target=self._loop, name="dsv41-engram-cpu-prefetch", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        torch.cuda.set_device(self._device_index)
        with torch.inference_mode():
            while True:
                fn, task = self._q.get()
                try:
                    fn()  # type: ignore[operator]
                except BaseException as exc:
                    task.exc = exc
                finally:
                    task.done.set()

    def submit(self, fn) -> _AsyncPrefetchTask:
        task = _AsyncPrefetchTask()
        self._q.put((fn, task))
        return task


_PINNED_KEEPALIVE: list[tuple[object, torch.Tensor, int]] = []


def _exact_pinned_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device_index: int | None = None,
) -> torch.Tensor | None:
    """Pinned host tensor occupying exact byte size via cudaHostAllocMapped."""
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        return None
    nbytes = dtype.itemsize
    for size in shape:
        nbytes *= size
    logger.info(
        "Allocating mapped pinned host memory: %d bytes (%.2f GiB); "
        "the driver faults and locks the pages, this can take a while...",
        nbytes,
        nbytes / 1024**3,
    )
    strict_bind = os.environ.get("VLLM_DSV41_ENGRAM_NUMA_STRICT", "0") == "1"
    with _numa_bind_current_gpu(device_index, strict_bind=strict_bind) as numa_bound:
        if numa_bound is not None:
            logger.info(
                "Binding pinned host allocation (%.2f GiB) to PCI %s NUMA node %d (%s)",
                nbytes / 1024**3,
                numa_bound[0],
                numa_bound[1],
                "MPOL_BIND" if strict_bind else "MPOL_PREFERRED",
            )
        err, ptr = cudart.cudaHostAlloc(nbytes, int(cudart.cudaHostAllocMapped))
    if int(err) != 0 and strict_bind and _engram_numa_bind_enabled():
        with _numa_bind_current_gpu(device_index, strict_bind=False):
            err, ptr = cudart.cudaHostAlloc(nbytes, int(cudart.cudaHostAllocMapped))
    if int(err) != 0:
        return None
    ptr = int(ptr)
    err, _ = cudart.cudaHostGetDevicePointer(ptr, 0)
    err_attr, attr = cudart.cudaPointerGetAttributes(ptr)
    if (
        int(err) != 0
        or int(err_attr) != 0
        or attr.type != cudart.cudaMemoryType.cudaMemoryTypeHost
    ):
        cudart.cudaFreeHost(ptr)
        return None
    try:
        buf = (ctypes.c_char * nbytes).from_address(ptr)
        tensor = torch.frombuffer(buf, dtype=torch.uint8).view(dtype).view(shape)
        torch.cuda.init()
        if not tensor.is_pinned():
            raise RuntimeError("cudaHostAlloc memory not reported as pinned")
    except Exception:
        cudart.cudaFreeHost(ptr)
        return None
    _PINNED_KEEPALIVE.append((buf, tensor, ptr))
    logger.info(
        "Engram pinned host memory at exact size: %d bytes (%.2f GiB)",
        nbytes,
        nbytes / 1024**3,
    )
    return tensor


def _stream_copy_engram(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Stream ~94 GiB safetensors shard in 1 GiB chunks with MADV_DONTNEED."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    MADV_SEQUENTIAL, MADV_DONTNEED = 2, 4
    PAGE = 4096

    def align_down(x: int) -> int:
        return x & ~(PAGE - 1)

    def align_up(x: int) -> int:
        return (x + PAGE - 1) & ~(PAGE - 1)

    base = src.data_ptr()
    nbytes = src.numel() * src.element_size()
    libc.madvise(
        ctypes.c_void_p(align_down(base)),
        ctypes.c_size_t(align_up(base + nbytes) - align_down(base)),
        MADV_SEQUENTIAL,
    )

    row_bytes = src.stride(0) * src.element_size()
    rows_per_chunk = max((1 << 30) // row_bytes, 1)
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    dst_u8 = (
        dst.view(torch.uint8)
        if not dst.is_cuda and dst.element_size() == 1
        else None
    )
    src_u8 = (
        src.view(torch.uint8)
        if dst_u8 is not None and src.element_size() == 1
        else None
    )
    try:
        for i in range(0, src.shape[0], rows_per_chunk):
            j = min(i + rows_per_chunk, src.shape[0])
            if dst_u8 is not None and src_u8 is not None:
                dst_u8[i:j].copy_(src_u8[i:j])
            else:
                dst[i:j].copy_(src[i:j])
            a = align_up(src[i:j].data_ptr())
            b = align_down(src[i:j].data_ptr() + (j - i) * row_bytes)
            if b > a:
                libc.madvise(
                    ctypes.c_void_p(a), ctypes.c_size_t(b - a), MADV_DONTNEED
                )
    finally:
        torch.set_num_threads(prev_threads)


def _engram_head_shard_weight_loader(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> None:
    """Load this rank's complete head buckets. ue8m0 scales arrive as
    float8_e8m0fnu; keep the raw bytes (the param stores uint8)."""
    part_rows = param.shape[0]
    if loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    shard = loaded_weight.narrow(0, param.engram_vocab_start, part_rows)
    assert shard.shape == param.shape, (
        f"engram shard {tuple(shard.shape)} does not fit param {tuple(param.shape)}"
    )
    if param.numel() > (1 << 30):
        _stream_copy_engram(param.data, shard)
    elif not param.data.is_cuda and param.data.element_size() == 1:
        param.data.view(torch.uint8).copy_(shard.view(torch.uint8))
    else:
        param.data.copy_(shard)
    param._engram_load_ver = getattr(param, "_engram_load_ver", 0) + 1


# Branching on SORTED at runtime intermittently crashes Triton 3.8's
# RemoveLayoutConversions.
# TODO: Remove the 3.8 check once Triton ships
# https://github.com/triton-lang/triton/pull/10706.
_SORTED_IS_CONSTEXPR = tl.constexpr(
    triton.__version__.startswith("3.8") and current_platform.is_rocm()
)


@triton.jit(
    do_not_specialize=[
        "vocab_start",
        "vocab_end",
        "num_rows",
        "ids_stride_t",
        "ids_stride_h",
        "GRID",
        *(() if _SORTED_IS_CONSTEXPR else ("SORTED",)),
    ]
)
def _engram_lookup_kernel(
    weight,
    scales,
    ids,
    sorted_dst,
    out,
    vocab_start,
    vocab_end,
    num_rows,
    ids_stride_t,
    ids_stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID,
    SORTED: tl.constexpr if _SORTED_IS_CONSTEXPR else None,  # type: ignore[valid-type]
    PACKED: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    PEER_ROWS: tl.constexpr = 0,
    WORD: tl.constexpr = tl.uint8,
    PACKED_FP4: tl.constexpr = False,
    SOFTWARE_FP8: tl.constexpr = False,
    W_STRIDE_R: tl.constexpr = 0,
    S_STRIDE_R: tl.constexpr = 0,
):
    """Gather FP8 or MXFP4 rows as dequantized BF16, or FP8 rows as packed
    values/scales (copied WORD at a time).

    Only this rank's heads are read; padded heads write zeros for all-gather.
    `weight`/`scales` may address pinned host memory through UVA. If SORTED,
    `ids` holds rows relative to vocab_start in table order and `sorted_dst`
    their output rows. SOFTWARE_FP8 (GPUs below SM89, which Triton gives no
    FP8 type): `weight` holds the E4M3 bytes as uint8 and they are decoded
    with integer operations, which is exact.
    """
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        if SORTED:
            local = tl.load(ids + rows, mask=valid, other=-1).to(tl.int64)
            dst = tl.load(sorted_dst + rows, mask=valid, other=0).to(tl.int32)
            owned = (local >= 0) & (local < vocab_end - vocab_start)
        else:
            head = HEAD_START + rows % LOCAL_HEADS
            token = (rows // LOCAL_HEADS).to(tl.int64)
            index = tl.load(
                ids + token * ids_stride_t + head * ids_stride_h,
                mask=valid & (head < TOTAL_HEADS),
                other=-1,
            ).to(tl.int64)
            dst = rows
            owned = valid & (head < TOTAL_HEADS)
            owned &= (index >= vocab_start) & (index < vocab_end)
            local = index - vocab_start
        local = tl.where(owned, local, 0)
        ROW_BYTES: tl.constexpr = DIM // 2 if PACKED_FP4 else DIM
        if PEER_ROWS:
            # `weight` holds each rank's buffer: PEER_ROWS rows, then scales.
            owner = local // PEER_ROWS
            peer = tl.load(weight + owner, mask=owned, other=0)
            local -= owner * PEER_ROWS
            if PACKED_FP4 or SOFTWARE_FP8:
                row_weight = peer.to(tl.pointer_type(tl.uint8))
            else:
                row_weight = peer.to(tl.pointer_type(tl.float8e4nv))
            row_scales = (peer + PEER_ROWS * ROW_BYTES).to(tl.pointer_type(tl.uint8))
            row_weight += local * ROW_BYTES
            row_scales += local * (DIM // QUANT_BLOCK)
        else:
            w_stride_r: tl.constexpr = W_STRIDE_R if W_STRIDE_R > 0 else ROW_BYTES
            s_stride_r: tl.constexpr = (
                S_STRIDE_R if S_STRIDE_R > 0 else (DIM // QUANT_BLOCK)
            )
            row_weight = weight + local * w_stride_r
            row_scales = scales + local * s_stride_r
        if PACKED:
            # Copy values and scales WORD at a time: one memory request a word.
            word_bytes: tl.constexpr = WORD.primitive_bitwidth // 8
            value_words = tl.arange(0, DIM // word_bytes)
            scale_words = tl.arange(0, DIM // QUANT_BLOCK // word_bytes)
            width: tl.constexpr = DIM + DIM // QUANT_BLOCK
            dst = dst.to(tl.int64)
            dst = dst // LOCAL_HEADS * OUT_STRIDE + dst % LOCAL_HEADS * width
            out_row = (out + dst).to(tl.pointer_type(WORD))
            # Load both first: a load after a store waits out its round trip.
            values = tl.load(
                row_weight.to(tl.pointer_type(WORD))[:, None] + value_words[None, :],
                mask=owned[:, None],
                other=0,
            )
            scale = tl.load(
                row_scales.to(tl.pointer_type(WORD))[:, None] + scale_words[None, :],
                mask=owned[:, None],
                other=0,
            )
            tl.store(
                out_row[:, None] + value_words[None, :], values, mask=valid[:, None]
            )
            out_row += DIM // word_bytes
            tl.store(
                out_row[:, None] + scale_words[None, :], scale, mask=valid[:, None]
            )
        else:
            if PACKED_FP4:
                # Even column in the low nibble. E2M1 magnitude with e = bits 2:1,
                # m = bit 0: m * 0.5 if e == 0, else (2 + m) * 2**(e - 2).
                byte = tl.load(
                    row_weight[:, None] + (cols // 2)[None, :],
                    mask=owned[:, None],
                    other=0,
                ).to(tl.int32)
                nib = tl.where(cols[None, :] % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
                e = (nib & 0x7) >> 1
                m = (nib & 0x1).to(tl.float32)
                pow2 = ((e - 2 + 127) << 23).to(tl.float32, bitcast=True)
                mag = tl.where(e == 0, m * 0.5, (2.0 + m) * pow2)
                values = tl.where((nib & 0x8) != 0, -mag, mag)
            else:
                values = tl.load(
                    row_weight[:, None] + cols[None, :], mask=owned[:, None], other=0.0
                )
            scale = tl.load(
                row_scales[:, None] + scale_cols[None, :], mask=owned[:, None], other=0
            )
            # ue8m0 is a power of two, so its byte *is* the fp32 exponent field.
            scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            if SOFTWARE_FP8:
                # E4M3: a normal code's exponent and mantissa shift into the
                # fp32 fields (bias 7 -> 127); a subnormal code is m * 2^-9.
                code = values.to(tl.int32)
                magnitude = code & 0x7F
                normal = ((magnitude << 20) + (120 << 23)).to(tl.float32, bitcast=True)
                subnormal = (magnitude & 7).to(tl.float32) * 0.001953125
                decoded = tl.where(magnitude < 8, subnormal, normal)
                decoded = tl.where((code & 0x80) != 0, -decoded, decoded)
            else:
                decoded = values.to(tl.float32)
            tl.store(
                out + dst[:, None] * DIM + cols[None, :],
                (decoded * scale).to(tl.bfloat16),
                mask=valid[:, None],
            )


@triton.jit
def _engram_dequant_staged_kernel(
    staged_w,
    staged_s,
    out,
    num_rows,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID: tl.constexpr,
):
    """Dequantize contiguous HBM-staged [num_rows, DIM] fp8 + ue8m0 -> bf16."""
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    s_row_cols = DIM // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        row_i64 = rows.to(tl.int64)
        code = tl.load(
            staged_w + row_i64[:, None] * DIM + cols[None, :],
            mask=valid[:, None],
            other=0,
        ).to(tl.int32)
        magnitude = code & 0x7F
        normal = ((magnitude << 20) + (120 << 23)).to(tl.float32, bitcast=True)
        subnormal = (magnitude & 7).to(tl.float32) * 0.001953125
        decoded = tl.where(magnitude < 8, subnormal, normal)
        decoded = tl.where((code & 0x80) != 0, -decoded, decoded)
        scale = tl.load(
            staged_s + row_i64[:, None] * s_row_cols + scale_cols[None, :],
            mask=valid[:, None],
            other=0,
        )
        scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(
            out + row_i64[:, None] * DIM + cols[None, :],
            (decoded * scale).to(tl.bfloat16),
            mask=valid[:, None],
        )


def engram_head_shard_rank() -> int:
    """This rank's slot among the hash-head shards of one engram table.

    TP-major, so the shards a DP gather brings in are contiguous heads and
    the following TP gather completes the head order.
    """
    dp_group = get_engram_dp_group()
    dp_size = dp_group.world_size if dp_group is not None else 1
    dp_rank = dp_group.rank_in_group if dp_group is not None else 0
    return get_tensor_model_parallel_rank() * dp_size + dp_rank


def engram_gathered_num_tokens() -> int:
    """Per-replica token slot for the node-local Engram DP group."""
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None:
        raise RuntimeError("a DP-shared engram table needs DP token metadata")
    group = get_engram_dp_group()
    assert group is not None
    # Engram groups are contiguous slices of the full DP group.
    start = get_dp_group().rank_in_group - group.rank_in_group
    return int(
        dp_metadata.num_tokens_across_dp_cpu[start : start + group.world_size].max()
    )


def gather_engram_hashes(
    hash_ids: torch.Tensor, *, dp_shared_memory: bool = False
) -> torch.Tensor:
    """Collect the n-gram ids of every DP replica sharing one table.

    Replicas are padded to a common token slot, so the gathered shape is
    static under CUDA graph capture (where DP already pads alike).
    """
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_shared_memory:
        return hash_ids
    slot = engram_gathered_num_tokens()
    if hash_ids.shape[0] > slot:
        raise ValueError("Engram token count exceeds the DP token slot")
    if hash_ids.shape[0] < slot:
        pad = hash_ids.new_full(
            (slot - hash_ids.shape[0], *hash_ids.shape[1:]), DEAD_ID
        )
        hash_ids = torch.cat((hash_ids, pad))
    return dp_group.all_gather(hash_ids, dim=0)


def describe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def gather_errors(group: GroupCoordinator, error: str | None) -> str:
    """Each rank's error, joined; empty if every rank succeeded."""
    errors: list[str | None] = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    return "; ".join(
        f"EDP rank {rank}: {error}" for rank, error in enumerate(errors) if error
    )


class EngramPeerGroup:
    """GPUs of several engines that split each Engram table over their HBM.

    They meet only at startup, through a file store; lookups then read peers'
    rows directly, so no engine's step ever waits on another's. Every GPU
    builds and loads the same tables in the same order, so its n-th exchange
    is every peer's n-th.
    """

    # Engines load their weights at their own pace before the group can serve.
    TIMEOUT = timedelta(minutes=30)

    def __init__(self, size: int, path: str) -> None:
        self.world_size = size
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.store = torch.distributed.FileStore(path, size)
        self.store.set_timeout(self.TIMEOUT)
        # Any assignment works: a rank only decides which rows a GPU holds.
        self.rank_in_group = self.store.add("next_rank", 1) - 1
        if self.rank_in_group >= size:
            raise RuntimeError(
                f"More than {size} GPUs joined the Engram peer group at {path}; "
                "check peer_group_size and use a fresh store path per launch"
            )
        self._step = 0

    def all_gather(self, value: bytes) -> list[bytes]:
        """Every GPU's value for the next exchange in rank order, once all have
        set theirs."""
        self._step += 1
        key = f"step{self._step}"
        self.store.set(f"{key}/{self.rank_in_group}", value)
        keys = [f"{key}/{rank}" for rank in range(self.world_size)]
        try:
            self.store.wait(keys)
        except RuntimeError as exc:
            missing = [
                rank for rank, name in enumerate(keys) if not self.store.check([name])
            ]
            raise TimeoutError(
                f"Engram peer GPUs {missing} did not reach startup step "
                f"{self._step} within {self.TIMEOUT}; every engine of the group "
                "must start, and none may restart alone"
            ) from exc
        return [self.store.get(name) for name in keys]

    def barrier(self) -> None:
        self.all_gather(b"")


@cache
def join_engram_peer_group(size: int, path: str) -> EngramPeerGroup:
    """This GPU's membership, shared by all of its process's tables."""
    return EngramPeerGroup(size, path)


def _shm_shortfall(num_bytes: int) -> str | None:
    """Why /dev/shm cannot hold a table of this size, or None if it can."""
    if not os.path.isdir(SHM_PATH):
        return f"{SHM_PATH} is not mounted"
    try:
        check_shm_free_space(num_bytes, allocation_name="Engram tables")
    except Exception as exc:
        return describe_error(exc)
    # tmpfs pages beyond available host memory are faulted in by the OOM killer.
    available = get_available_memory()
    if num_bytes > available:
        return (
            f"{num_bytes / 2**30:.1f} GiB exceeds the "
            f"{available / 2**30:.1f} GiB of available host memory"
        )
    return None


def _allow_any_numa_node(address: int, num_bytes: int) -> None:
    """Let a mapping's pages spill to any NUMA node despite --numa-bind.

    mbind(MPOL_LOCAL) overrides the task's membind for this range: pages still
    go to the faulting CPU's node first, then to any other node.
    """
    sys_mbind = {"aarch64": 235, "x86_64": 237}.get(platform.machine())
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    if sys_mbind is None or libc.syscall(
        ctypes.c_long(sys_mbind),
        ctypes.c_void_p(address),
        ctypes.c_ulong(num_bytes),
        ctypes.c_int(4),  # MPOL_LOCAL
        None,
        ctypes.c_ulong(0),
        ctypes.c_uint(0),
    ):
        logger.warning_once(
            "Engram mbind(MPOL_LOCAL) failed; /dev/shm tables follow the "
            "process memory policy: %s",
            os.strerror(ctypes.get_errno()),
        )


class DPSharedEngramStorage:
    """Host weights shared by a node-local DP group with one writer: host-NUMA
    cuMem where FABRIC handles work, opt-in HIP VMM host memory on ROCm, else a
    registered /dev/shm mapping."""

    def __init__(
        self,
        num_rows: int,
        dim: int,
        block_size: int,
        group: GroupCoordinator,
        packed_fp4: bool = False,
    ) -> None:
        self.group = group
        row_bytes = dim // 2 if packed_fp4 else dim
        weight_dtype = torch.uint8 if packed_fp4 else torch.float8_e4m3fn
        weight_bytes = num_rows * row_bytes
        num_bytes = weight_bytes + num_rows * (dim // block_size)

        def split_table(table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            weight = table[:weight_bytes].view(weight_dtype).view(-1, row_bytes)
            return weight, table[weight_bytes:].view(-1, dim // block_size)

        views = None
        if current_platform.is_cuda():
            from vllm.models.deepseek_v41.nvidia.ops import engram_fabric_table

            views = engram_fabric_table.share_host_table(num_bytes, group)
        elif current_platform.is_rocm():
            from vllm.models.deepseek_v41.amd.ops import engram_host_table

            views = engram_host_table.share_host_table(num_bytes, group)
        # The GPU maps cuMem host memory with 2M pages, /dev/shm with smaller ones.
        self.huge_pages = views is not None
        host, gpu = views or self.share_shm_table(num_bytes, group)
        self.weight, self.weight_scale_inv = split_table(host)
        self._views = split_table(gpu)

    @staticmethod
    def share_shm_table(
        num_bytes: int, group: GroupCoordinator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map and register one /dev/shm file across a node-local DP group.

        Returns CPU and GPU views of the file. Raises on every rank, leaving
        nothing mapped, if any rank fails.
        """
        with ExitStack() as stack:
            path = error = None
            if group.rank_in_group == 0 and not (error := _shm_shortfall(num_bytes)):
                try:
                    backing_file = stack.enter_context(
                        tempfile.NamedTemporaryFile(prefix="vllm_engram_", dir=SHM_PATH)
                    )
                    backing_file.truncate(num_bytes)
                    path = backing_file.name
                except Exception as exc:
                    error = describe_error(exc)
            path, error = group.broadcast_object((path, error))
            if error is not None:
                raise RuntimeError(
                    "Engram shared-memory creation failed on EDP rank 0: " + error
                )

            mapping = None
            try:
                with open(path, "r+b") as file:
                    mapping = mmap.mmap(file.fileno(), num_bytes, flags=mmap.MAP_SHARED)
            except Exception as exc:
                error = describe_error(exc)
            if failures := gather_errors(group, error):
                del mapping
                raise RuntimeError(
                    "Engram shared-memory initialization failed: " + failures
                )
            # Unlink while the file is still sparse, so a crash during the slow
            # registration below cannot orphan the table.
            stack.close()
            assert mapping is not None

            owner = tensor = None
            try:
                owner = np.frombuffer(mapping, dtype=np.uint8)
                pointer = owner.ctypes.data
                _allow_any_numa_node(pointer, num_bytes)
                tensor = torch.from_numpy(owner)
                result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
                if result.value != 0:
                    raise RuntimeError(f"cudaHostRegister failed: {result}")
                finalizer = weakref.finalize(
                    owner, DPSharedEngramStorage._unregister, mapping, pointer
                )
                finalizer.atexit = False  # type: ignore[misc]
                # The UVA helper otherwise allocates a private pinned copy.
                if not tensor.is_pinned():
                    raise RuntimeError(
                        "cudaHostRegister did not pin the shared Engram mapping"
                    )
            except Exception as exc:
                error = describe_error(exc)

            if failures := gather_errors(group, error):
                # Dropping the owner runs the finalizer; the mapping then unmaps.
                del tensor, owner, mapping
                raise RuntimeError(
                    "Engram shared-memory initialization failed: " + failures
                )
            assert tensor is not None
            return tensor, get_accelerator_view_from_cpu_tensor(tensor)

    @staticmethod
    def _unregister(mapping: mmap.mmap, pointer: int) -> None:
        # Torch storage retains the numpy owner, including through cached UVA views.
        # Keep its mmap alive until CUDA has released the registration.
        result = torch.cuda.cudart().cudaHostUnregister(pointer)
        if result.value != 0:
            logger.warning("Engram cudaHostUnregister failed: %s", result)

    def load_weight(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        if self.group.rank_in_group == 0:
            _engram_head_shard_weight_loader(param, loaded_weight)
        # Read order may differ across ranks. Equal load counts ensure all shared
        # weights are ready after the last weight-loader call returns.
        torch.distributed.barrier(group=self.group.cpu_group)

    def get_views(
        self, weight: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (weight.data_ptr(), scales.data_ptr()) != (
            self.weight.data_ptr(),
            self.weight_scale_inv.data_ptr(),
        ):
            raise RuntimeError("Shared Engram parameter storage must not be replaced")
        return self._views


def can_share_engram_tables(layout: EngramLayout, block_size: int = 32) -> bool:
    """Whether co-located DP replicas exist and can hold the full tables."""
    if get_engram_dp_size() == 1:
        logger.warning_once(
            "Engram DP replicas are not co-located on one node; "
            "storing the offloaded tables per rank instead of sharing them."
        )
        return False
    num_bytes = sum(layout.num_embeddings) * (
        layout.row_bytes + layout.head_dim // block_size
    )
    group = get_engram_dp_group()
    assert group is not None
    # Only the leader allocates, so every replica follows its check. /dev/shm
    # must fit even where cuMem will, since it is the fallback.
    error = _shm_shortfall(num_bytes) if group.rank_in_group == 0 else None
    error = group.broadcast_object(error)
    if error is not None:
        logger.warning_once(
            "Sharding the offloaded Engram tables across DP replicas: %s", error
        )
    return error is None


def _allocate_huge_page_storage(num_bytes: int) -> torch.Tensor | None:
    """Register prefaulted huge pages, or return None for pinned-memory fallback."""
    if sys.platform != "linux":
        return None
    try:
        mapping = mmap.mmap(-1, num_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        mapping.madvise(mmap.MADV_HUGEPAGE)
    except OSError as exc:
        logger.warning(
            "Engram huge-page allocation failed (%s); using pinned memory.", exc
        )
        return None

    owner = np.frombuffer(mapping, dtype=np.uint8)
    # Fault the pages in before CUDA pins them, so they can be huge.
    owner[:: mmap.PAGESIZE] = 0
    tensor = torch.from_numpy(owner)
    pointer = tensor.data_ptr()
    result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
    if result.value != 0:
        logger.warning(
            "Engram cudaHostRegister failed (%s); using pinned memory.", result
        )
        return None
    # Tensor views retain owner; its finalizer retains mapping until unregister.
    finalizer = weakref.finalize(
        owner, DPSharedEngramStorage._unregister, mapping, pointer
    )
    finalizer.atexit = False  # type: ignore[misc]
    if not tensor.is_pinned():
        logger.warning(
            "CUDA did not recognize Engram registration; using pinned memory."
        )
        return None
    return tensor


class ParallelEngramEmbedding(nn.Module):
    """Hash heads with FP8 or packed MXFP4 rows and per-block E8M0 scales.

    Heads are TP-sharded, and additionally DP-sharded across a node-local
    group unless the table lives in (DP-shared) pinned host memory or its rows
    are split over a peer group's HBM. MXFP4 rows are E2M1 values, two per
    byte (even column in the low nibble), stored as uint8 since torch has no
    4-bit dtype.
    """

    _weight_loader = staticmethod(_engram_head_shard_weight_loader)
    _shared_memory: DPSharedEngramStorage | None = None
    _packed: torch.Tensor | None = None
    _loaded = False
    packed_fp4: bool = False

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
        dp_shared_memory: bool = False,
        use_thp: bool = False,
        packed_fp4: bool = False,
        peer_group: EngramPeerGroup | None = None,
    ) -> None:
        super().__init__()
        assert head_sizes and all(size > 0 for size in head_sizes)
        assert sum(head_sizes) <= num_embeddings
        assert not packed_fp4 or dim % 2 == 0
        self.packed_fp4 = packed_fp4
        self.row_bytes = dim // 2 if packed_fp4 else dim
        self.cpu_offload = cpu_offload
        self.dp_shared_memory = dp_shared_memory
        self.use_thp = use_thp
        self.dp_size = get_engram_dp_size()
        if dp_shared_memory:
            if not cpu_offload:
                raise ValueError("dp_shared_memory requires cpu_offload=True")
            if self.dp_size <= 1:
                raise ValueError(
                    "dp_shared_memory requires a node-local Engram DP "
                    f"group with size > 1; effective Engram DP size is {self.dp_size}. "
                    "Check that the node layout and rank placement allow complete "
                    "DP replicas to be co-located."
                )
            self.dp_size = 1
        # GPU tables split rows over a peer group (the given GPUs of several
        # engines, else the DP group) and every rank reads its own tokens' rows
        # from the owning peer's HBM, so lookups need no collective.
        self._peer_group: EngramPeerGroup | GroupCoordinator | None = peer_group
        if peer_group is not None:
            self.dp_size = 1
        elif (
            not cpu_offload
            and self.dp_size > 1
            and get_tensor_model_parallel_world_size() == 1
        ):
            group = get_engram_dp_group()
            assert group is not None
            # Cross-node groups require embedding_across_dp's single NVLink domain;
            # node-local groups without NVLink P2P keep the DP collectives.
            peer = not all(in_the_same_node_as(group.cpu_group, source_rank=0))
            if not peer:
                physical_id = current_platform.visible_device_id_to_physical_device_id(
                    torch.accelerator.current_device_index()
                )
                physical_ids = [None] * group.world_size
                torch.distributed.all_gather_object(
                    physical_ids, physical_id, group=group.cpu_group
                )
                peer = current_platform.is_fully_connected(physical_ids)
            if peer:
                self._peer_group, self.dp_size = group, 1
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._view_src: tuple[int, int] | None = None
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.tp_size = get_tensor_model_parallel_world_size()
        num_shards, head_rank = self._get_shard_info()
        self.part_n_hash_cols = triton.cdiv(self.n_hash_cols, num_shards)
        # TODO: Support row-wise sharding when there are too few hash heads.
        assert (num_shards - 1) * self.part_n_hash_cols < self.n_hash_cols, (
            f"Engram sharding leaves ranks without hash heads: "
            f"{self.n_hash_cols} heads over {num_shards} shards"
        )
        self.head_start = head_rank * self.part_n_hash_cols
        head_end = self.head_start + self.part_n_hash_cols
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        # Peer tables keep the full lookup range but store one row slice.
        self.peer_rows, row_start = 0, self.vocab_start_idx
        if (peers := self._peer_group) is not None:
            self.peer_rows = triton.cdiv(self.vocab_end_idx, peers.world_size)
            row_start = min(peers.rank_in_group * self.peer_rows, self.vocab_end_idx)
            self.part_num_embeddings = min(
                self.peer_rows, self.vocab_end_idx - row_start
            )
        self._skip_layerwise_reload = self.cpu_offload and self.peer_rows == 0
        self._device_index = torch.accelerator.current_device_index()
        self.local_valid_heads = max(
            0, min(self.n_hash_cols - self.head_start, self.part_n_hash_cols)
        )
        self._num_sms = torch.cuda.get_device_properties(
            self._device_index
        ).multi_processor_count

        # Lazy state for VLLM_DSV41_ENGRAM_CPU_PREFETCH=1.
        self._cpu_cap: int = 0
        self._cpu_compact_mode: bool = False
        self._host_ids: torch.Tensor | None = None
        self._host_bf16: torch.Tensor | None = None
        self._host_fp8: torch.Tensor | None = None
        self._host_scales: torch.Tensor | None = None
        self._gpu_stage_fp8: torch.Tensor | None = None
        self._gpu_stage_scales: torch.Tensor | None = None
        self._ids_ready_event: torch.cuda.Event | None = None
        self._h2d_done_event: torch.cuda.Event | None = None
        self._h2d_in_flight: bool = False
        self._async_worker: _EngramAsyncWorker | None = None
        self._default_cpu_threads: int | None = None
        self._packed_host: torch.Tensor | None = None

        weight, scales = self._allocate_weights()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scales, requires_grad=False)
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {
                    "weight_loader": self._weight_loader,
                    "engram_vocab_start": row_start,
                },
            )
        if cpu_offload and not self.weight.is_meta:
            # Constant dummy values avoid randomizing huge CPU lookup tables.
            set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
            # The ue8m0 encoding of scale 1.0 is exponent byte 127.
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})
            logger.info(
                "Engram table offloaded to pinned host memory: %d rows x %d, "
                "%.2f GiB %s",
                self.part_num_embeddings,
                dim,
                self.part_num_embeddings
                * (self.row_bytes + dim // block_size)
                / 1024**3,
                "per rank"
                if self._shared_memory is None
                else "shared across DP replicas via "
                + (
                    ("HIP VMM" if current_platform.is_rocm() else "cuMem")
                    if self._shared_memory.huge_pages
                    else "/dev/shm"
                ),
            )

    def _get_shard_info(self) -> tuple[int, int]:
        if self.dp_size == 1:
            return self.tp_size, get_tensor_model_parallel_rank()
        return self.tp_size * self.dp_size, engram_head_shard_rank()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale_dim = self.dim // self.block_size
        weight_dtype = torch.uint8 if self.packed_fp4 else torch.float8_e4m3fn
        if torch.get_default_device().type == "meta" or not (
            self.cpu_offload or self.peer_rows
        ):
            return (
                torch.empty(
                    self.part_num_embeddings, self.row_bytes, dtype=weight_dtype
                ),
                torch.empty(self.part_num_embeddings, scale_dim, dtype=torch.uint8),
            )
        if self.peer_rows:
            peers = self._peer_group
            values = self.peer_rows * self.row_bytes
            num_bytes = values + self.peer_rows * scale_dim
            device = torch.device("cuda", torch.accelerator.current_device_index())
            if isinstance(peers, EngramPeerGroup):
                # Symmetric memory cannot span engines: its rendezvous rejects
                # equal device ordinals on one node, and each engine's start at 0.
                from vllm.models.deepseek_v41.nvidia.ops.engram_fabric_table import (
                    share_peer_table,
                )

                # Held here like the symmetric handle, so lookups keep every
                # slice mapped even if a loader replaces the parameters' data.
                buffer, self._peer_mappings = share_peer_table(num_bytes, peers)
                pointers = self._peer_mappings.addresses
            else:
                import torch.distributed._symmetric_memory as symm_mem

                assert peers is not None
                buffer = symm_mem.empty(num_bytes, dtype=torch.uint8, device=device)
                self._symm_handle = symm_mem.rendezvous(buffer, peers.device_group)
                pointers = self._symm_handle.buffer_ptrs
            self.register_buffer(
                "peer_ptrs",
                torch.tensor(pointers, dtype=torch.int64, device=device),
                persistent=False,
            )
            rows = self.part_num_embeddings
            return (
                buffer[: rows * self.row_bytes]
                .view(weight_dtype)
                .view(rows, self.row_bytes),
                buffer[values : values + rows * scale_dim].view(rows, scale_dim),
            )
        if self.dp_shared_memory:
            group = get_engram_dp_group()
            assert group is not None
            storage = DPSharedEngramStorage(
                self.part_num_embeddings,
                self.dim,
                self.block_size,
                group,
                self.packed_fp4,
            )
            self._shared_memory = storage
            self._weight_loader = storage.load_weight
            return storage.weight, storage.weight_scale_inv
        # pin_memory rounds up to a power of two. ROCm pins huge pages either way.
        if self.use_thp or current_platform.is_rocm():
            weight_bytes = self.part_num_embeddings * self.row_bytes
            scale_bytes = self.part_num_embeddings * scale_dim
            packed = _allocate_huge_page_storage(weight_bytes + scale_bytes)
            if packed is not None:
                self._packed = packed
                return (
                    packed[:weight_bytes].view(weight_dtype).view(-1, self.row_bytes),
                    packed[weight_bytes:].view(-1, scale_dim),
                )
        # Exact-size pinned allocation via cudaHostAllocMapped prevents PyTorch's
        # CachingHostAllocator from rounding 23.6 GiB shards up to 32 GiB (which
        # totals 256 GiB across 8 shards and OOMs 251 GiB hosts).
        if getattr(envs, "VLLM_ENGRAM_EXACT_PIN", True) and current_platform.is_cuda():
            if self.cpu_offload and _engram_packed_host_enabled():
                packed_stride = _engram_packed_stride(self.row_bytes + scale_dim)
                packed_host = _exact_pinned_tensor(
                    (self.part_num_embeddings, packed_stride),
                    torch.uint8,
                    self._device_index,
                )
                if packed_host is not None:
                    self._packed_host = packed_host
                    return (
                        packed_host[:, : self.row_bytes].view(weight_dtype),
                        packed_host[:, self.row_bytes : self.row_bytes + scale_dim],
                    )
            w_exact = _exact_pinned_tensor(
                (self.part_num_embeddings, self.row_bytes),
                weight_dtype,
                self._device_index,
            )
            s_exact = (
                _exact_pinned_tensor(
                    (self.part_num_embeddings, scale_dim),
                    torch.uint8,
                    self._device_index,
                )
                if w_exact is not None
                else None
            )
            if w_exact is not None and s_exact is not None:
                return w_exact, s_exact
            logger.warning(
                "Exact-size pinned allocation failed; falling back to "
                "torch's pinned allocator, which rounds up to the next power of two."
            )
        # Model initialization may be inside a CUDA device context.
        with _numa_bind_current_gpu(self._device_index):
            return (
                torch.empty(
                    self.part_num_embeddings,
                    self.row_bytes,
                    dtype=weight_dtype,
                    device="cpu",
                    pin_memory=True,
                ),
                torch.empty(
                    self.part_num_embeddings,
                    scale_dim,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                ),
            )

    def process_weights_after_loading(self) -> None:
        """Peers read this table's rows directly, so serve only once every GPU
        of the group has finished loading its own."""
        if not self.peer_rows:
            return
        assert self._peer_group is not None
        if self._loaded and isinstance(self._peer_group, EngramPeerGroup):
            # Other engines read these rows at any time, also while they reload.
            raise RuntimeError(
                "Engram tables split across engines (peer_group_size) cannot "
                "reload weights; restart the peer group instead"
            )
        torch.accelerator.synchronize()
        self._peer_group.barrier()
        self._loaded = True

    def collapse_huge_pages(self) -> None:
        """Best-effort MADV_COLLAPSE (Linux >= 6.1) of pages that faulted small."""
        if self._packed is None:
            return
        addr, num_bytes = self._packed.data_ptr(), self._packed.nbytes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        if libc.madvise(addr, num_bytes, 25) != 0:
            logger.warning(
                "Engram MADV_COLLAPSE failed; keeping existing pages: %s",
                os.strerror(ctypes.get_errno()),
            )

    @property
    def _uses_small_host_pages(self) -> bool:
        """Whether the GPU maps this host table with pages too small for its TLB."""
        if self._shared_memory is not None:
            return not self._shared_memory.huge_pages
        return self.cpu_offload and self._packed is None

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shared_memory is not None:
            return self._shared_memory.get_views(self.weight, self.weight_scale_inv)
        if not self.cpu_offload:
            return self.weight.data, self.weight_scale_inv.data
        packed_ptr = (
            self._packed_host.data_ptr() if self._packed_host is not None else 0
        )
        src = (self.weight.data_ptr(), self.weight_scale_inv.data_ptr(), packed_ptr)
        if self._view_src != src:
            scale_dim = self.dim // self.block_size
            if (
                self._packed_host is not None
                and self.weight.data_ptr() == packed_ptr
                and self.weight_scale_inv.data_ptr() == packed_ptr + self.row_bytes
            ):
                packed_uva = get_accelerator_view_from_cpu_tensor(self._packed_host)
                self._views = (
                    packed_uva[:, : self.row_bytes].view(self.weight.dtype),
                    packed_uva[:, self.row_bytes : self.row_bytes + scale_dim],
                )
            else:
                self._views = (
                    get_accelerator_view_from_cpu_tensor(self.weight.data),
                    get_accelerator_view_from_cpu_tensor(self.weight_scale_inv.data),
                )
            self._view_src = src
        assert self._views is not None
        return self._views

    def _resolve_cpu_threads(self) -> int:
        configured = int(os.environ.get("VLLM_DSV41_ENGRAM_CPU_THREADS", "0"))
        if configured > 0:
            return configured
        if self._default_cpu_threads is None:
            info = _get_cuda_device_numa_info(self._device_index)
            if info is not None:
                ncpus = len(info[2])
            elif hasattr(os, "sched_getaffinity"):
                ncpus = len(os.sched_getaffinity(0))
            else:
                ncpus = os.cpu_count() or 8
            self._default_cpu_threads = min(24, max(1, ncpus // 2))
        return self._default_cpu_threads

    def should_use_cpu_prefetch(self, num_tokens: int) -> bool:
        return (
            self.cpu_offload
            and not self.packed_fp4
            and self.dp_size == 1
            and not self.peer_rows
            and self._shared_memory is None
            and _engram_cpu_prefetch_enabled()
            and num_tokens >= _engram_cpu_min_tokens()
            and not self.weight.is_cuda
            and not self.weight_scale_inv.is_cuda
            and not _is_graph_capture_active()
        )

    def _ensure_cpu_staging(self, num_tokens: int, device: torch.device) -> bool:
        compact = _engram_cpu_compact_h2d_enabled()
        if self._ids_ready_event is None:
            self._ids_ready_event = torch.cuda.Event()
            self._h2d_done_event = torch.cuda.Event()
        has_buf = (
            self._host_fp8 is not None if compact else self._host_bf16 is not None
        )
        if (
            num_tokens <= self._cpu_cap
            and has_buf
            and self._cpu_compact_mode == compact
        ):
            return compact
        if self._h2d_in_flight and self._h2d_done_event is not None:
            self._h2d_done_event.synchronize()
            self._h2d_in_flight = False
        new_cap = max(4096, 1 << (num_tokens - 1).bit_length())
        valid_h = max(1, self.local_valid_heads)
        s_cols = self.dim // self.block_size
        with _numa_bind_current_gpu(self._device_index):
            self._host_ids = torch.empty(
                (new_cap, valid_h),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            if compact:
                self._host_fp8 = torch.empty(
                    (new_cap, self.part_n_hash_cols, self.dim),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
                self._host_scales = torch.empty(
                    (new_cap, self.part_n_hash_cols, s_cols),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
                self._host_bf16 = None
            else:
                self._host_bf16 = torch.empty(
                    (new_cap, self.part_n_hash_cols, self.dim),
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=True,
                )
                self._host_fp8 = None
                self._host_scales = None
        if compact:
            self._gpu_stage_fp8 = torch.empty(
                (new_cap, self.part_n_hash_cols, self.dim),
                dtype=torch.uint8,
                device=device,
            )
            self._gpu_stage_scales = torch.empty(
                (new_cap, self.part_n_hash_cols, s_cols),
                dtype=torch.uint8,
                device=device,
            )
        else:
            self._gpu_stage_fp8 = None
            self._gpu_stage_scales = None
        self._cpu_cap = new_cap
        self._cpu_compact_mode = compact
        return compact

    def _run_cpu_gather_kernel(self, num_tokens: int, compact: bool) -> None:
        assert self._host_ids is not None
        lib = _load_engram_cpu_lib()
        w_data = self.weight.data
        s_data = self.weight_scale_inv.data
        if compact:
            assert self._host_fp8 is not None and self._host_scales is not None
            lib.engram_cpu_gather_fp8(
                ctypes.c_void_p(w_data.data_ptr()),
                w_data.stride(0),
                ctypes.c_void_p(s_data.data_ptr()),
                s_data.stride(0),
                ctypes.c_void_p(self._host_ids.data_ptr()),
                ctypes.c_void_p(self._host_fp8.data_ptr()),
                ctypes.c_void_p(self._host_scales.data_ptr()),
                self.vocab_start_idx,
                self.vocab_end_idx,
                num_tokens,
                self.part_n_hash_cols,
                self.local_valid_heads,
                self.dim,
                self.block_size,
                self._resolve_cpu_threads(),
            )
        else:
            assert self._host_bf16 is not None
            lib.engram_cpu_gather_bf16(
                ctypes.c_void_p(w_data.data_ptr()),
                w_data.stride(0),
                ctypes.c_void_p(s_data.data_ptr()),
                s_data.stride(0),
                ctypes.c_void_p(self._host_ids.data_ptr()),
                ctypes.c_void_p(self._host_bf16.data_ptr()),
                self.vocab_start_idx,
                self.vocab_end_idx,
                num_tokens,
                self.part_n_hash_cols,
                self.local_valid_heads,
                self.dim,
                self.block_size,
                self._resolve_cpu_threads(),
            )

    def _upload_cpu_gathered(
        self, num_tokens: int, out: torch.Tensor, compact: bool, background: bool
    ) -> None:
        if compact:
            assert (
                self._host_fp8 is not None
                and self._host_scales is not None
                and self._gpu_stage_fp8 is not None
                and self._gpu_stage_scales is not None
            )
            self._gpu_stage_fp8[:num_tokens].copy_(
                self._host_fp8[:num_tokens], non_blocking=True
            )
            self._gpu_stage_scales[:num_tokens].copy_(
                self._host_scales[:num_tokens], non_blocking=True
            )
            rows = num_tokens * self.part_n_hash_cols
            tiles = triton.cdiv(rows, 16)
            if background:
                pref_sms = _engram_prefetch_sms()
                sm_budget = (
                    min(self._num_sms, pref_sms)
                    if pref_sms > 0
                    else min(self._num_sms, 8)
                )
            else:
                sm_budget = self._num_sms
            grid = min(tiles, sm_budget)
            _engram_dequant_staged_kernel[(grid,)](
                self._gpu_stage_fp8,
                self._gpu_stage_scales,
                out,
                rows,
                DIM=self.dim,
                QUANT_BLOCK=self.block_size,
                BLOCK_R=16,
                GRID=grid,
            )
        else:
            assert self._host_bf16 is not None
            out.copy_(self._host_bf16[:num_tokens], non_blocking=True)

    def _cpu_lookup_sync(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        num_tokens = indices.shape[0]
        if num_tokens == 0:
            return
        compact = self._ensure_cpu_staging(num_tokens, out.device)
        assert (
            self._host_ids is not None
            and self._ids_ready_event is not None
            and self._h2d_done_event is not None
        )
        stream = torch.cuda.current_stream(out.device)
        if self.local_valid_heads > 0:
            h_end = self.head_start + self.local_valid_heads
            ids_contig = (
                indices[:, self.head_start : h_end].to(dtype=torch.int64).contiguous()
            )
            self._host_ids[:num_tokens, : self.local_valid_heads].copy_(
                ids_contig, non_blocking=True
            )
            self._ids_ready_event.record(stream)
            self._ids_ready_event.synchronize()
        if self._h2d_in_flight:
            self._h2d_done_event.synchronize()
            self._h2d_in_flight = False
        self._run_cpu_gather_kernel(num_tokens, compact)
        self._upload_cpu_gathered(num_tokens, out, compact, background=False)
        self._h2d_done_event.record(stream)
        self._h2d_in_flight = True

    def submit_cpu_prefetch(
        self,
        indices: torch.Tensor,
        out: torch.Tensor,
        prefetch_stream: torch.cuda.Stream,
        start_event: torch.cuda.Event,
        done_event: torch.cuda.Event,
    ) -> _AsyncPrefetchTask:
        num_tokens = indices.shape[0]
        compact = self._ensure_cpu_staging(num_tokens, out.device)
        if self._async_worker is None:
            self._async_worker = _EngramAsyncWorker(out.device.index or 0)
        assert (
            self._host_ids is not None
            and self._ids_ready_event is not None
            and self._h2d_done_event is not None
        )
        ids_ready = self._ids_ready_event
        h2d_done = self._h2d_done_event
        ids_contig: torch.Tensor | None = None

        with torch.cuda.stream(prefetch_stream):
            prefetch_stream.wait_event(start_event)
            if self.local_valid_heads > 0:
                h_end = self.head_start + self.local_valid_heads
                ids_contig = (
                    indices[:, self.head_start : h_end]
                    .to(dtype=torch.int64)
                    .contiguous()
                )
                self._host_ids[:num_tokens, : self.local_valid_heads].copy_(
                    ids_contig, non_blocking=True
                )
                ids_ready.record(prefetch_stream)

        def _worker_job() -> None:
            _keep_alive = ids_contig
            if _keep_alive is not None:
                ids_ready.synchronize()
            if self._h2d_in_flight:
                h2d_done.synchronize()
                self._h2d_in_flight = False
            self._run_cpu_gather_kernel(num_tokens, compact)
            with torch.cuda.stream(prefetch_stream):
                self._upload_cpu_gathered(num_tokens, out, compact, background=True)
                h2d_done.record(prefetch_stream)
                self._h2d_in_flight = True
                done_event.record(prefetch_stream)

        return self._async_worker.submit(_worker_job)

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up local heads into BF16, or packed FP8 bytes followed by scales.

        `background` limits the grid to leave SMs for concurrent work.
        """
        if self.packed_fp4 and out.dtype == torch.uint8:
            raise ValueError("MXFP4 Engram tables do not support packed FP8 output")
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        if out.dtype == torch.bfloat16 and self.should_use_cpu_prefetch(indices.shape[0]):
            self._cpu_lookup_sync(indices, out)
            return
        weight, scales = self._storage()
        ids_stride_t, ids_stride_h = indices.stride()
        sort_min_rows, _ = _engram_lookup_thresholds(indices.device)
        sort = (
            self._uses_small_host_pages
            and sort_min_rows is not None
            and rows >= sort_min_rows
        )
        dst = indices
        if sort:
            # Small-page host tables miss the TLB per row; in table order, rows on
            # one page share it.
            head_end = self.head_start + self.part_n_hash_cols
            local = indices[:, self.head_start : head_end] - self.vocab_start_idx
            if (pad := self.part_n_hash_cols - local.shape[1]) > 0:
                local = nn.functional.pad(local, (0, pad), value=-1)
            indices, dst = local.flatten().sort()
            dst = dst.int()
        # The table dwarfs TLB reach, so a persistent grid near the SM count
        # beats one program per row; halve it to leave SMs for the main stream.
        if background:
            pref_sms = _engram_prefetch_sms()
            max_grid = (
                min(self._num_sms, pref_sms)
                if pref_sms > 0
                else max(1, self._num_sms // 2)
            )
        else:
            max_grid = self._num_sms
        # Host reads are latency bound: once the rows fill the grid with
        # 128-row tiles, each program keeps 8x the reads in flight.
        block_r = 128 if rows >= 128 * max_grid else 16
        packed = out.dtype == torch.uint8
        w_stride_r = 0 if self.peer_rows else weight.stride(0) * weight.element_size()
        s_stride_r = 0 if self.peer_rows else scales.stride(0) * scales.element_size()
        # Copy packed rows 8 bytes per load where their values, scales and
        # output rows are 8-byte aligned.
        wide = (
            packed
            and self.dim // self.block_size % 8 == 0
            and (out.stride(0) | out.data_ptr()) % 8 == 0
            and (self.peer_rows or (w_stride_r | s_stride_r) % 8 == 0)
        )
        grid = min(triton.cdiv(rows, block_r), max_grid)
        software_fp8 = (
            not self.packed_fp4 and not current_platform.has_device_capability(89)
        )
        if software_fp8 and not self.peer_rows:
            weight = weight.view(torch.uint8)
        _engram_lookup_kernel[(grid,)](
            self.peer_ptrs if self.peer_rows else weight,
            scales,
            indices,
            dst,
            out,
            self.vocab_start_idx,
            self.vocab_end_idx,
            rows,
            ids_stride_t,
            ids_stride_h,
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=block_r,
            GRID=grid,
            SORTED=sort,
            PACKED=packed,
            OUT_STRIDE=out.stride(0),
            PEER_ROWS=self.peer_rows,
            PACKED_FP4=self.packed_fp4,
            SOFTWARE_FP8=software_fp8,
            W_STRIDE_R=w_stride_r,
            S_STRIDE_R=s_stride_r,
            WORD=tl.uint64 if wide else tl.uint8,
            num_warps=8 if block_r == 128 else 4,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim]
        bf16, gathered from all shards for this replica's tokens."""
        num_tokens = indices.shape[0]
        if self.dp_size > 1:
            indices = gather_engram_hashes(indices)
        out = torch.empty(
            (indices.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=indices.device,
        )
        self.lookup(indices, out)
        if self.dp_size > 1:
            out = _gather_engram_rows(out, num_tokens)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=1)
        return out[:, : self.n_hash_cols]


@triton.jit(do_not_specialize=["num_kv_tokens"])
def _fused_engram_post_wkv_kernel(
    hidden_states,
    kv,
    q_weight,
    k_weight,
    token_mask,
    output,
    num_kv_tokens,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    kv_stride_t,
    kv_stride_d,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    mask_stride,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    eps,
    clamp_value,
    DIM: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    program_idx = tl.program_id(0)
    token_idx = program_idx // HC_MULT
    hc_idx = program_idx % HC_MULT
    token_idx = token_idx.to(tl.int64)
    source_idx = token_idx
    source_valid = source_idx < num_kv_tokens

    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_valid = dim_offsets < DIM
    hidden = tl.load(
        hidden_states
        + token_idx * hidden_stride_t
        + hc_idx * hidden_stride_h
        + dim_offsets * hidden_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        kv + source_idx * kv_stride_t + (hc_idx * DIM + dim_offsets) * kv_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    q = tl.load(
        q_weight + hc_idx * q_stride_h + dim_offsets * q_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_weight + hc_idx * k_stride_h + dim_offsets * k_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)

    hidden_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / DIM + eps)
    key_rms = tl.rsqrt(tl.sum(key * key, axis=0) / DIM + eps)
    dot = tl.sum(hidden * q * k * key, axis=0)
    dot *= hidden_rms * key_rms * tl.rsqrt(DIM * 1.0)
    gate_input = tl.sqrt(tl.maximum(tl.abs(dot), clamp_value))
    gate_input = tl.where(dot < 0.0, -gate_input, gate_input)
    gate = tl.sigmoid(gate_input)
    if HAS_MASK:
        active = tl.load(
            token_mask + source_idx * mask_stride,
            mask=source_valid,
            other=0,
        )
        gate = tl.where(active, gate, 0.0)

    value = tl.load(
        kv + source_idx * kv_stride_t + (HC_MULT * DIM + dim_offsets) * kv_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output
        + token_idx * output_stride_t
        + hc_idx * output_stride_h
        + dim_offsets * output_stride_d,
        hidden + gate * value,
        mask=dim_valid,
    )


@triton.jit(do_not_specialize=["num_kv_tokens"])
def _engram_partial_dot_kernel(
    hidden_states,
    kv_shard,
    q_weight,
    k_weight,
    stats,
    num_kv_tokens,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    kv_stride_t,
    kv_stride_d,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    TP_RANK: tl.constexpr,
    C_PART: tl.constexpr,
    DIM: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_idx = tl.program_id(0)
    token_idx = (program_idx // HC_MULT).to(tl.int64)
    hc_idx = program_idx % HC_MULT
    g_lo = TP_RANK * C_PART
    g_hi = g_lo + C_PART
    k_lo = hc_idx * DIM
    k_hi = k_lo + DIM
    ov_lo = tl.maximum(g_lo, k_lo)
    ov_hi = tl.minimum(g_hi, k_hi)
    ov_len = tl.maximum(ov_hi - ov_lo, 0)
    offs = tl.arange(0, BLOCK_SIZE)
    valid = (offs < ov_len) & (token_idx < num_kv_tokens)
    d_idx = (ov_lo - k_lo) + offs
    kv_idx = (ov_lo - g_lo) + offs

    hidden = tl.load(
        hidden_states
        + token_idx * hidden_stride_t
        + hc_idx * hidden_stride_h
        + d_idx * hidden_stride_d,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        kv_shard + token_idx * kv_stride_t + kv_idx * kv_stride_d,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    q = tl.load(
        q_weight + hc_idx * q_stride_h + d_idx * q_stride_d,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_weight + hc_idx * k_stride_h + d_idx * k_stride_d,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    ksq = tl.sum(key * key, axis=0)
    dot = tl.sum(hidden * q * k * key, axis=0)
    base = (token_idx * HC_MULT + hc_idx) * 2
    tl.store(stats + base, ksq)
    tl.store(stats + base + 1, dot)


@triton.jit(do_not_specialize=["num_kv_tokens"])
def _engram_apply_gate_v_kernel(
    hidden_states,
    v_buf,
    stats,
    token_mask,
    output,
    num_kv_tokens,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    v_stride_t,
    v_stride_d,
    mask_stride,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    eps,
    clamp_value,
    DIM: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    program_idx = tl.program_id(0)
    token_idx = (program_idx // HC_MULT).to(tl.int64)
    hc_idx = program_idx % HC_MULT
    source_valid = token_idx < num_kv_tokens

    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_valid = dim_offsets < DIM
    hidden = tl.load(
        hidden_states
        + token_idx * hidden_stride_t
        + hc_idx * hidden_stride_h
        + dim_offsets * hidden_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    base = (token_idx * HC_MULT + hc_idx) * 2
    ksq = tl.load(stats + base)
    dot = tl.load(stats + base + 1)

    hidden_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / DIM + eps)
    key_rms = tl.rsqrt(ksq / DIM + eps)
    dot *= hidden_rms * key_rms * tl.rsqrt(DIM * 1.0)
    gate_input = tl.sqrt(tl.maximum(tl.abs(dot), clamp_value))
    gate_input = tl.where(dot < 0.0, -gate_input, gate_input)
    gate = tl.sigmoid(gate_input)
    if HAS_MASK:
        active = tl.load(
            token_mask + token_idx * mask_stride,
            mask=source_valid,
            other=0,
        )
        gate = tl.where(active, gate, 0.0)

    value = tl.load(
        v_buf + token_idx * v_stride_t + dim_offsets * v_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output
        + token_idx * output_stride_t
        + hc_idx * output_stride_h
        + dim_offsets * output_stride_d,
        hidden + gate * value,
        mask=dim_valid,
    )


@triton.jit(do_not_specialize=["num_tokens", "token_start", "num_elements"])
def _engram_select_rows_kernel(
    gathered,
    output,
    num_tokens,
    token_start,
    num_elements,
    LOCAL_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tokens = token_start + offsets // WIDTH
    cols = offsets % WIDTH
    source = (cols // LOCAL_WIDTH * num_tokens + tokens) * LOCAL_WIDTH
    source += cols % LOCAL_WIDTH
    values = tl.load(
        gathered + source, (offsets < num_elements) & (tokens < num_tokens), other=0
    )
    tl.store(output + offsets, values, offsets < num_elements)


def _engram_select_rows(
    gathered: torch.Tensor,
    output: torch.Tensor,
    source_tokens: int,
    token_start: int,
    local_width: int,
) -> None:
    """Copy one token window out of a rank-major gathered buffer.

    Both gathers land rank-major (`[rank][token][local width]`); this walks the
    window the rank keeps and lays its ranks out side by side as width.
    """
    if output.numel() == 0:
        return
    _engram_select_rows_kernel[(triton.cdiv(output.numel(), 1024),)](
        gathered,
        output,
        source_tokens,
        token_start,
        output.numel(),
        local_width,
        output.shape[1] * output.shape[2],
        BLOCK_SIZE=1024,
    )


def _gather_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Exchange DP tokens for heads, retaining only this replica's tokens."""
    dp_group = get_engram_dp_group()
    assert dp_group is not None
    slot, remainder = divmod(staged.shape[0], dp_group.world_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    gathered = dp_group.all_gather(staged, dim=0)
    local_heads, dim = staged.shape[1:]
    rows = staged.new_empty((num_tokens, dp_group.world_size * local_heads, dim))
    _engram_select_rows(
        gathered,
        rows,
        staged.shape[0],
        dp_group.rank_in_group * slot,
        local_heads * dim,
    )
    return rows


@triton.jit(do_not_specialize=["num_tokens"])
def _engram_unpack_fp8_kernel(
    packed,
    values,
    scales,
    num_tokens,
    values_stride,
    scales_stride,
    HEADS: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    LAYERS: tl.constexpr,
    RANK: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    PEER: tl.constexpr,
):
    """Split `[rank][token][layer][local head]` packed rows into MXFP8 values and
    F8_128x4 scales, one layer per grid column; `packed` holds per-rank
    pointers if PEER. Scales past `num_tokens` zero-pad the 128-row tile.
    """
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    layer = tl.program_id(1).to(tl.int64)
    token, head = row // HEADS, row % HEADS
    if PEER:
        src = tl.load(packed + head // LOCAL_HEADS)
        src = tl.multiple_of(src, DIM // 32).to(tl.pointer_type(tl.uint8))
        source = RANK * num_tokens + token
    else:
        src = packed
        source = head // LOCAL_HEADS * num_tokens + token
    source = (source * LAYERS + layer) * LOCAL_HEADS + head % LOCAL_HEADS
    # Packed rows are only scale-width aligned.
    src += tl.multiple_of(source * (DIM + DIM // 32), DIM // 32)
    valid = token < num_tokens
    cols = tl.arange(0, DIM)
    data = tl.load(src[:, None] + cols[None, :], mask=valid[:, None], other=0)
    values += layer * values_stride + row * DIM
    tl.store(values[:, None] + cols[None, :], data, mask=valid[:, None])
    scale_cols = tl.arange(0, DIM // 32)
    scale = tl.load(
        src[:, None] + DIM + scale_cols[None, :], mask=valid[:, None], other=0
    )
    group = head[:, None] * (DIM // 32) + scale_cols[None, :]
    offset = mxfp8_scale_offset(token[:, None], group, HEADS * DIM // 32)
    tl.store(scales + layer * scales_stride + offset, scale)


class EngramBatch:
    """Stage every Engram layer's WKV input as MXFP8 with one exchange."""

    def __init__(self, layers: list["Engram"], max_tokens: int) -> None:
        first = layers[0].layer_hash_index
        hash_indices = [layer.layer_hash_index for layer in layers]
        assert hash_indices == list(range(first, first + len(layers)))
        self.layers = layers
        table = layers[0].embed_tokens
        device = layers[0].staged_rows.device
        self.packed_rows = torch.empty(
            max_tokens * table.dp_size,
            len(layers),
            table.part_n_hash_cols,
            table.dim + table.dim // 32,
            dtype=torch.uint8,
            device=device,
        )
        # Same allocator as the send buffer: NCCL hangs this all-to-all when its
        # buffers mix the cumem pool and the caching allocator.
        self.exchanged_rows = torch.empty_like(self.packed_rows)
        width = table.n_hash_cols * table.dim
        self.fp8_values = torch.empty(
            len(layers), max_tokens, width, dtype=torch.uint8, device=device
        )
        self.fp8_scales = torch.empty(
            len(layers),
            mxfp8_scale_bytes(max_tokens, width),
            dtype=torch.uint8,
            device=device,
        )
        max_peer_slot = min(max_tokens, ENGRAM_A2A_MIN_SLOT - 1)
        self.peer_exchange = None
        if table.cpu_offload:
            from vllm.models.deepseek_v41.nvidia.ops.engram_peer import (
                EngramPeerExchange,
            )

            self.peer_exchange = EngramPeerExchange.create(layers, max_peer_slot)
        for layer in layers:
            layer._batch = self
            # The layers read this batch's staging instead of their BF16 rows.
            layer.staged_rows = layer.staged_rows.new_empty(
                (0, *layer.staged_rows.shape[1:])
            )

    @classmethod
    def create(cls, layers: list["Engram"], max_tokens: int) -> "EngramBatch | None":
        """Batch DP-sharded tables whose WKV takes FlashInfer MXFP8 input."""
        if not layers:
            return None
        table = layers[0].embed_tokens
        if (
            table.dp_size == 1
            or any(layer.embed_tokens.packed_fp4 for layer in layers)
            or table.tp_size > 1
            or table.dim % 128
            or get_current_vllm_config().lora_config is not None
            or not can_fuse_query_quant([layer.wkv for layer in layers])
        ):
            return None
        return cls(layers, max_tokens)

    def prepare_embeddings(self, hashes: torch.Tensor) -> None:
        """Stage every layer's WKV input before the decoder layers.

        DP-sharded tables decide on the group's slot, so every rank agrees.
        """
        slot = engram_gathered_num_tokens()
        peer = self.peer_exchange is not None and slot < ENGRAM_A2A_MIN_SLOT
        self._stage(hashes if peer else gather_engram_hashes(hashes), slot, peer)

    @eager_break_during_capture
    def _stage(self, hashes: torch.Tensor, slot: int, peer: bool) -> None:
        if slot == 0:
            return
        if peer:
            assert self.peer_exchange is not None
            rows = self.peer_exchange.exchange(hashes, slot)
            rank = self.peer_exchange.dp_rank
        else:
            rows, rank = self.packed_rows[: hashes.shape[0]], 0
            for index, layer in enumerate(self.layers):
                layer.embed_tokens.lookup(
                    hashes[:, layer.layer_hash_index], rows[:, index]
                )
            group = get_engram_dp_group()
            assert group is not None
            sent, rows = rows, self.exchanged_rows[: rows.shape[0]]
            torch.distributed.all_to_all_single(rows, sent, group=group.device_group)
        table = self.layers[0].embed_tokens
        heads = table.n_hash_cols
        grid = (triton.cdiv(slot, 128) * 128 * heads // 16, len(self.layers))
        _engram_unpack_fp8_kernel[grid](
            rows,
            self.fp8_values,
            self.fp8_scales,
            slot,
            self.fp8_values.stride(0),
            self.fp8_scales.stride(0),
            HEADS=heads,
            LOCAL_HEADS=table.part_n_hash_cols,
            LAYERS=len(self.layers),
            RANK=rank,
            DIM=table.dim,
            BLOCK=16,
            PEER=peer,
        )

    def wkv_input(self, layer: "Engram", num_tokens: int) -> QuantizedActivation:
        index = layer.layer_hash_index - self.layers[0].layer_hash_index
        values = self.fp8_values[index, :num_tokens].view(torch.float8_e4m3fn)
        return QuantizedActivation(
            values,
            self.fp8_scales[index, : mxfp8_scale_bytes(num_tokens, values.shape[1])],
            torch.bfloat16,
            values.shape,
            kMxfp8Dynamic,
        )


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it
    matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per
    hc copy plus a shared value. The gate is a normalized dot product of the
    stream against the key, signed-sqrt'ed before the sigmoid (matching the
    training kernel).
    """

    _prefetch_stream: torch.cuda.Stream | None = None
    _prefetch_done: torch.cuda.Event | None = None
    _batch: EngramBatch | None = None
    # A model's own TP all-gather of a 2-D tensor's last dimension for the
    # TP-sharded heads and wkv output (set_column_gather); None uses the TP
    # group's.
    _gather_columns: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        layout: EngramLayout,
        layer_hash_index: int,
        use_sequence_parallel: bool,
        prefix: str,
        prefetch_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        # Layers sharing one stream serialize their offloaded lookups, so they
        # take turns instead of jointly starving decoder compute of SMs.
        self._prefetch_stream = prefetch_stream
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        self.use_sequence_parallel = use_sequence_parallel

        # Named ``embed_tokens`` so the checkpoint's ``engram.embed.weight``
        # survives the mapper's ``embed.weight`` -> ``embed_tokens.weight``
        # suffix rule.
        self.embed_tokens = self._create_embedding(layout, layer_hash_index)
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = self._create_wkv(
            n_hash_cols * layout.head_dim, quant_config, f"{prefix}.wkv"
        )
        self.q_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )

        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        self._init_staging(max_tokens, layout.head_dim)

    def _create_wkv(
        self, input_size: int, quant_config: QuantizationConfig | None, prefix: str
    ) -> ColumnParallelLinear:
        # Without sequence parallelism every TP rank holds every token, so
        # shard the output columns instead of replicating the projection.
        return ColumnParallelLinear(
            input_size,
            self.dim * (self.hc_mult + 1),
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            return_bias=False,
            prefix=prefix,
            disable_tp=self.use_sequence_parallel,
        )

    def _create_embedding(
        self, layout: EngramLayout, layer_hash_index: int
    ) -> ParallelEngramEmbedding:
        engram_config = get_current_vllm_config().engram_config
        assert engram_config is not None
        peer_group = None
        if engram_config.peer_group_store is not None:
            peer_group = join_engram_peer_group(
                engram_config.peer_group_size, engram_config.peer_group_store
            )
        return ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=engram_config.cpu_offload,
            dp_shared_memory=bool(engram_config.dp_shared_memory),
            use_thp=engram_config.use_thp,
            packed_fp4=layout.packed_fp4,
            peer_group=peer_group,
        )

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        # Persistent storage keeps lookup addresses stable across graph replays.
        self.staged_rows = torch.empty(
            max_tokens * self.embed_tokens.dp_size,
            self.embed_tokens.part_n_hash_cols,
            head_dim,
            dtype=torch.bfloat16,
        )
        if not self.embed_tokens.cpu_offload:
            self._prefetch_stream = None
            return
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=self.staged_rows.device)
        self._prefetch_start = torch.cuda.Event()
        self._prefetch_done = torch.cuda.Event()
        self._pending_cpu_task: _AsyncPrefetchTask | None = None

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Look up, or prefetch from host memory, rows before the decoder layers
        consume them; DP-sharded tables take the group's gathered hash IDs."""
        rows = self.staged_rows[: hash_ids.shape[0]]
        assert rows.shape[0] == hash_ids.shape[0], "engram staging buffer too small"
        if self._prefetch_stream is None:
            self.embed_tokens.lookup(hash_ids, rows)
        else:
            self._start_prefetch(hash_ids, rows)

    @eager_break_during_capture
    def _start_prefetch(self, hash_ids: torch.Tensor, rows: torch.Tensor) -> None:
        # Eager boundaries let the lookup span piecewise graph segments, and
        # decide per replay (on the replay stream) whether it runs inline.
        num_tokens = hash_ids.shape[0]
        # All DP ranks must agree, or EP collectives wait on the slowest one.
        if is_forward_context_available() and (dp := get_forward_context().dp_metadata):
            num_tokens = int(dp.num_tokens_across_dp_cpu.max())
        current = stream = torch.cuda.current_stream()
        self._pending_cpu_task = None
        if self.embed_tokens.should_use_cpu_prefetch(hash_ids.shape[0]):
            assert (
                self._prefetch_stream is not None
                and self._prefetch_start is not None
                and self._prefetch_done is not None
            )
            self._prefetch_start.record(current)
            self._pending_cpu_task = self.embed_tokens.submit_cpu_prefetch(
                hash_ids,
                rows,
                self._prefetch_stream,
                self._prefetch_start,
                self._prefetch_done,
            )
            return
        # Big lookups stall persistent main-stream kernels anyway: run them inline.
        _, inline_min_tokens = _engram_lookup_thresholds(hash_ids.device)
        if inline_min_tokens is not None and num_tokens >= inline_min_tokens:
            self.embed_tokens.lookup(hash_ids, rows)
        else:
            stream = self._prefetch_stream
            assert stream is not None
            stream.wait_stream(current)
            # Keep temporary hash storage alive until lookup finishes reading it.
            if not torch.cuda.is_current_stream_capturing():
                hash_ids.record_stream(stream)
            with torch.cuda.stream(stream):
                self.embed_tokens.lookup(hash_ids, rows, background=True)
        assert self._prefetch_done is not None
        self._prefetch_done.record(stream)

    @eager_break_during_capture
    def _finish_prefetch(self, event: torch.cuda.Event) -> None:
        # Wait for this layer's lookup only, not for later layers on the stream.
        if self._pending_cpu_task is not None:
            self._pending_cpu_task.wait()
            self._pending_cpu_task = None
        torch.cuda.current_stream().wait_event(event)

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        if self._prefetch_done is not None:
            self._finish_prefetch(self._prefetch_done)
        if self.embed_tokens.dp_size > 1:
            slot = engram_gathered_num_tokens()
            staged = self.staged_rows[: slot * self.embed_tokens.dp_size]
            return _gather_engram_rows(staged, num_tokens)
        return self.staged_rows[:num_tokens]

    def set_column_gather(self, gather: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """All-gather the TP-sharded heads and wkv output with `gather` (a
        2-D tensor's last dimension, every rank's columns side by side)
        instead of the TP group's all-gather. For TP-sharded tables without
        sequence parallelism or a DP-shared table."""
        assert self.embed_tokens.tp_size > 1 and self.embed_tokens.dp_size == 1
        assert not self.use_sequence_parallel and self._batch is None
        self._gather_columns = gather
        self.wkv.gather_output = False

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Gather heads, returning only local tokens when SP is enabled."""
        rows = self._ready_rows(hash_ids.shape[0])
        if self.embed_tokens.tp_size == 1:
            return rows[:, : self.embed_tokens.n_hash_cols]
        if self._gather_columns is not None:
            dim = rows.shape[-1]
            rows = self._gather_columns(rows.flatten(1)).unflatten(1, (-1, dim))
            return rows[:, : self.embed_tokens.n_hash_cols]
        if self.use_sequence_parallel:
            tp_size = self.embed_tokens.tp_size
            num_tokens, local_heads, dim = rows.shape
            gathered = tensor_model_parallel_all_gather(rows, dim=0)
            chunk = (num_tokens + tp_size - 1) // tp_size
            out = rows.new_empty((chunk, self.embed_tokens.n_hash_cols, dim))
            _engram_select_rows(
                gathered,
                out,
                num_tokens,
                get_tensor_model_parallel_rank() * chunk,
                local_heads * dim,
            )
            return out
        rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, : self.embed_tokens.n_hash_cols]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """hidden_states: [T, hc_mult, dim]; hash_ids: [T, n_hash_cols] (all
        tokens, pre sequence-parallel shard); token_mask: [T], False shuts
        the gate so those positions pass through untouched."""
        if self._batch is not None:
            kv = self.wkv(self._batch.wkv_input(self, hash_ids.shape[0]))
        else:
            kv = self.wkv(self.embed(hash_ids).flatten(-2))
        num_kv_tokens = hash_ids.shape[0]
        assert token_mask is None or token_mask.shape == (num_kv_tokens,)
        if (
            self._gather_columns is not None
            and not self.use_sequence_parallel
            and num_kv_tokens >= 48
            and os.environ.get("VLLM_DSV41_ENGRAM_LOCAL_DOT", "1") == "1"
        ):
            tp_size = get_tensor_model_parallel_world_size()
            c_part = kv.shape[1]
            if (
                tp_size > 1
                and c_part * tp_size == (self.hc_mult + 1) * self.dim
                and c_part >= self.dim
            ):
                return self._post_wkv_sharded(
                    hidden_states, kv, token_mask, num_kv_tokens
                )
        if self._gather_columns is not None:
            kv = self._gather_columns(kv)
        if self.use_sequence_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = (num_kv_tokens + tp_size - 1) // tp_size
            assert hidden_states.shape[0] == shard_size
            start = min(tp_rank * shard_size, num_kv_tokens)
            num_kv_tokens = min(shard_size, num_kv_tokens - start)
            if token_mask is not None:
                token_mask = token_mask[start : start + num_kv_tokens]

        return self._post_wkv(hidden_states, kv, token_mask, num_kv_tokens)

    def _post_wkv_sharded(
        self,
        hidden_states: torch.Tensor,
        kv_shard: torch.Tensor,
        token_mask: torch.Tensor | None,
        num_kv_tokens: int,
    ) -> torch.Tensor:
        """Lossless 5x-bandwidth-reduced Engram update from TP-column-sharded
        `kv_shard` (`[M, c_part]`): computes local FP32 partial `(key^2, dot)`
        scalars per stream, all-reduces only `[M, 2*hc_mult]` FP32 scalars and
        broadcasts only `V` (`[M, dim]`, 1/5 of `kv`) from the last TP rank."""
        num_tokens, hc_mult, dim = hidden_states.shape
        assert hc_mult == self.hc_mult and dim == self.dim
        output = torch.empty_like(hidden_states)
        if num_tokens == 0:
            return output

        tp_group = get_tp_group()
        tp_size = tp_group.world_size
        tp_rank = tp_group.rank_in_group
        c_part = kv_shard.shape[1]
        block_size = triton.next_power_of_2(dim)
        num_warps = 8 if block_size >= 2048 else 4

        stats = torch.empty(
            (num_tokens, hc_mult * 2), dtype=torch.float32, device=hidden_states.device
        )
        _engram_partial_dot_kernel[(num_tokens * hc_mult,)](
            hidden_states,
            kv_shard,
            self.q_weight,
            self.k_weight,
            stats,
            num_kv_tokens,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            kv_shard.stride(0),
            kv_shard.stride(1),
            self.q_weight.stride(0),
            self.q_weight.stride(1),
            self.k_weight.stride(0),
            self.k_weight.stride(1),
            TP_RANK=tp_rank,
            C_PART=c_part,
            DIM=dim,
            HC_MULT=hc_mult,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        # P2P Lamport all-gather of tiny [M, 8] FP32 stats (32B/row) -> local sum
        from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.columns import (
            _gathers as _ampere_col_gathers,
            all_gather_columns as _ampere_ag_cols,
        )
        if stats.shape[1] * stats.element_size() in _ampere_col_gathers:
            stats = (
                _ampere_ag_cols(stats)
                .view(num_tokens, tp_size, hc_mult * 2)
                .sum(dim=1)
            )
        else:
            dist.all_reduce(stats, group=tp_group.device_group)

        from vllm.models.deepseek_v41.nvidia.ops.mhc import (
            _ampere_all_reduce as _ampere_ar,
        )
        if tp_rank == tp_size - 1:
            v_buf = kv_shard[:, c_part - dim : c_part].contiguous()
        else:
            v_buf = torch.empty(
                (num_kv_tokens, dim), dtype=kv_shard.dtype, device=kv_shard.device
            )
        if _ampere_ar is not None and _ampere_ar.supports(v_buf):
            if v_buf.shape[0] >= 16 and getattr(_ampere_ar, "ring_broadcast_kernel", None) is not None:
                v_buf = _ampere_ar.broadcast(v_buf, tp_size - 1)
            else:
                if tp_rank != tp_size - 1:
                    v_buf.zero_()
                v_buf = _ampere_ar(v_buf)
        else:
            dist.broadcast(
                v_buf, src=tp_group.ranks[tp_size - 1], group=tp_group.device_group
            )

        mask = token_mask if token_mask is not None else hidden_states
        _engram_apply_gate_v_kernel[(num_tokens * hc_mult,)](
            hidden_states,
            v_buf,
            stats,
            mask,
            output,
            num_kv_tokens,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            v_buf.stride(0),
            v_buf.stride(1),
            token_mask.stride(0) if token_mask is not None else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            self.eps,
            self.clamp_value,
            DIM=dim,
            HC_MULT=hc_mult,
            BLOCK_SIZE=block_size,
            HAS_MASK=token_mask is not None,
            num_warps=num_warps,
        )
        return output

    def _post_wkv(
        self,
        hidden_states: torch.Tensor,
        kv: torch.Tensor,
        token_mask: torch.Tensor | None,
        num_kv_tokens: int,
    ) -> torch.Tensor:
        """The gated update from the `wkv` output of `num_kv_tokens` tokens."""
        num_tokens, hc_mult, dim = hidden_states.shape
        assert hc_mult == self.hc_mult and dim == self.dim
        assert kv.ndim == 2 and kv.shape[1] == (hc_mult + 1) * dim
        output = torch.empty_like(hidden_states)
        if num_tokens == 0:
            return output

        block_size = triton.next_power_of_2(dim)
        num_warps = 8 if block_size >= 2048 else 4
        mask = token_mask if token_mask is not None else hidden_states
        _fused_engram_post_wkv_kernel[(num_tokens * hc_mult,)](
            hidden_states,
            kv,
            self.q_weight,
            self.k_weight,
            mask,
            output,
            num_kv_tokens,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            kv.stride(0),
            kv.stride(1),
            self.q_weight.stride(0),
            self.q_weight.stride(1),
            self.k_weight.stride(0),
            self.k_weight.stride(1),
            token_mask.stride(0) if token_mask is not None else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            self.eps,
            self.clamp_value,
            DIM=dim,
            HC_MULT=hc_mult,
            BLOCK_SIZE=block_size,
            HAS_MASK=token_mask is not None,
            num_warps=num_warps,
        )
        return output
