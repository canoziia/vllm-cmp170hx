# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 prefill scheduler helpers: quadratic cost-budget chunking
and prefix-cache tracking for YoCo consumer-layer prefill truncation.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Any

DEFAULT_YOCO_KEEP_TOKENS = 2688
_ALPHA_US = 77.0
_BETA_US_PER_1K = 0.95
_MAX_TRACKED_HASHES = 262144


def yoco_prefill_trunc_enabled() -> bool:
    return os.environ.get("VLLM_DSV41_YOCO_PREFILL_TRUNC", "0") == "1"


def yoco_keep_tokens() -> int:
    raw = int(
        os.environ.get("VLLM_DSV41_YOCO_KEEP_TOKENS", str(DEFAULT_YOCO_KEEP_TOKENS))
    )
    return ((max(raw, 256) + 127) // 128) * 128


def cost_prefill_chunk_enabled() -> bool:
    return os.environ.get("VLLM_DSV41_COST_PREFILL_CHUNK", "0") == "1"


def solve_cost_chunk_tokens(
    prefix_tokens: int,
    remaining_tokens: int,
    max_chunk_tokens: int,
    block_size: int = 128,
) -> int:
    """Solve quadratic cost(q, c) <= target_budget for chunk size q."""
    if remaining_tokens <= 512 or max_chunk_tokens <= 512 or prefix_tokens < 65536:
        return min(remaining_tokens, max_chunk_tokens)

    ref_q = float(max_chunk_tokens)
    budget_us = float(
        os.environ.get(
            "VLLM_DSV41_PREFILL_QUOTA_US",
            str(
                ref_q
                * (_ALPHA_US + _BETA_US_PER_1K * (65536.0 + 0.5 * ref_q) / 1024.0)
            ),
        )
    )
    a = 0.5 * _BETA_US_PER_1K / 1024.0
    b = _ALPHA_US + _BETA_US_PER_1K * (float(prefix_tokens) / 1024.0)
    disc = b * b + 4.0 * a * budget_us
    q_float = (-b + math.sqrt(max(0.0, disc))) / (2.0 * a)

    min_chunk = max(512, block_size)
    q = int(q_float // block_size) * block_size
    q = max(min_chunk, min(q, max_chunk_tokens, remaining_tokens))

    leftover = remaining_tokens - q
    if 0 < leftover < min_chunk:
        if remaining_tokens <= max_chunk_tokens:
            q = remaining_tokens
        elif q - min_chunk >= min_chunk:
            q = ((remaining_tokens // 2) // block_size) * block_size
            q = max(min_chunk, q)
    return q


class Dsv41PrefillHelper:
    """Tracks YoCo-truncated block hashes and computes cost-aware chunk limits."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self._truncated_hashes: OrderedDict[Any, None] = OrderedDict()

    def _mark_truncated(self, h: Any) -> None:
        if h in self._truncated_hashes:
            self._truncated_hashes.move_to_end(h)
            return
        if len(self._truncated_hashes) >= _MAX_TRACKED_HASHES:
            self._truncated_hashes.popitem(last=False)
        self._truncated_hashes[h] = None

    def _mark_clean(self, h: Any) -> None:
        self._truncated_hashes.pop(h, None)

    def maybe_cap_prefill_chunk(
        self, num_computed: int, num_new: int, max_chunk: int
    ) -> int:
        if not cost_prefill_chunk_enabled() or num_new <= 512:
            return num_new
        return solve_cost_chunk_tokens(
            prefix_tokens=num_computed,
            remaining_tokens=num_new,
            max_chunk_tokens=max_chunk,
            block_size=self.block_size,
        )

    def record_scheduled_prefill(
        self, request: Any, c_start: int, num_new: int
    ) -> None:
        if not yoco_prefill_trunc_enabled() or num_new <= 0:
            return
        prompt_len = int(getattr(request, "num_prompt_tokens", 0))
        c_end = c_start + num_new
        if c_start >= prompt_len:
            return
        params = getattr(request, "sampling_params", None)
        if params is not None and getattr(params, "prompt_logprobs", None) is not None:
            return
        block_hashes = getattr(request, "block_hashes", None)
        if not block_hashes:
            return

        keep = yoco_keep_tokens()
        bs = self.block_size
        if prompt_len - c_end >= keep:
            b0 = c_start // bs
            b1 = min(len(block_hashes), c_end // bs)
            for b in range(b0, b1):
                self._mark_truncated(block_hashes[b])
        else:
            cutoff = max(0, prompt_len - keep)
            b_trunc_end = min(len(block_hashes), cutoff // bs)
            for b in range(c_start // bs, b_trunc_end):
                self._mark_truncated(block_hashes[b])
            clean_from = max(c_start, cutoff + max(0, keep - 256))
            b_clean_start = (clean_from + bs - 1) // bs if clean_from > 0 else 0
            b_clean_end = min(len(block_hashes), min(c_end, prompt_len) // bs)
            for b in range(b_clean_start, b_clean_end):
                self._mark_clean(block_hashes[b])

    def adjust_prefix_cache_hit(
        self,
        request: Any,
        kv_cache_manager: Any,
        new_computed_blocks: Any,
        num_local: int,
        num_external: int,
        load_kv_async: bool,
    ) -> tuple[Any, int, int, bool]:
        if not yoco_prefill_trunc_enabled() or not self._truncated_hashes:
            return new_computed_blocks, num_local, num_external, load_kv_async
        total_hit = num_local + num_external
        bs = self.block_size
        if total_hit < bs:
            return new_computed_blocks, num_local, num_external, load_kv_async
        block_hashes = getattr(request, "block_hashes", None)
        last_blk = (total_hit // bs) - 1
        if not block_hashes or last_blk >= len(block_hashes):
            return new_computed_blocks, num_local, num_external, load_kv_async
        if block_hashes[last_blk] not in self._truncated_hashes:
            return new_computed_blocks, num_local, num_external, load_kv_async

        keep = yoco_keep_tokens()
        target_hit = max(0, ((total_hit - keep) // bs) * bs)
        if target_hit >= num_local:
            new_ext = target_hit - num_local
            if new_ext == 0:
                load_kv_async = False
            return new_computed_blocks, num_local, new_ext, load_kv_async
        trimmed_blocks = kv_cache_manager.truncate_computed_blocks(
            new_computed_blocks, target_hit
        )
        return trimmed_blocks, target_hit, 0, False
