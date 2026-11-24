# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only FP8 copy of the lm_head for DSpark *draft* logits only.

``VLLM_DSPARK_DRAFT_HEAD_FP8=1`` (default 0 = off, bitwise unchanged).

The DSpark drafter re-uses the target's bf16 lm_head ([129280 x 5120],
1.32 GB) for its base draft logits. On the last PP stage that is a second
full-vocabulary, purely bandwidth-bound pass every step (~0.85 ms on CMP
170HX). With this switch the drafter reads a per-output-channel FP8 (e4m3)
copy through the Marlin W8A16 kernel instead (0.66 GB). The target's
verification logits keep the bf16 head, so the generated text distribution is
unchanged: greedy drafts can only differ in acceptance, and probabilistic
drafting samples and rejection-tests against the *same* (fp8-derived) q.

Memory: +N*K bytes (+0.62 GiB) on the stage that hosts the drafter. Built in
``load_draft_model`` (before KV sizing and before CUDA-graph capture), in row
chunks so the transient peak is about 2x the fp8 copy.
"""

from __future__ import annotations

import os

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    fp8_fused_exponent_bias_into_scales,
)

FP8_MAX = 448.0
_QUANT_CHUNK_ROWS = 8192


def draft_head_fp8_enabled() -> bool:
    return os.environ.get("VLLM_DSPARK_DRAFT_HEAD_FP8", "0") == "1"


def quantize_rows_fp8(
    weight: torch.Tensor, chunk_rows: int = _QUANT_CHUNK_ROWS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (= per output channel) symmetric e4m3 quantization.

    Returns ``(fp8 [N, K], scale [N] in weight.dtype)``. The scale is rounded to
    ``weight.dtype`` *before* dividing, so dequant ``fp8 * scale`` uses exactly
    the value Marlin sees; values are clamped to +-448 (torch's fp8 cast does not
    saturate).
    """
    assert weight.dim() == 2
    n, k = weight.shape
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty((n,), dtype=weight.dtype, device=weight.device)
    for s in range(0, n, chunk_rows):
        e = min(n, s + chunk_rows)
        w = weight[s:e].float()
        amax = w.abs().amax(dim=1).clamp_min(1e-12)
        sc = (amax / FP8_MAX).to(weight.dtype)
        sc = torch.where(sc == 0, torch.ones_like(sc), sc)
        scf = sc.float()
        q[s:e] = (w / scf[:, None]).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        scale[s:e] = sc
        del w
    return q, scale


class Fp8DraftHead:
    """Marlin W8A16 (per-channel) projection [M, K] bf16 -> [M, N] bf16."""

    def __init__(self, weight: torch.Tensor) -> None:
        n, k = weight.shape
        if n % 64 != 0 or k % 128 != 0:
            raise ValueError(
                f"VLLM_DSPARK_DRAFT_HEAD_FP8: lm_head shape {tuple(weight.shape)} "
                "is not Marlin tile aligned (N%64, K%128)"
            )
        if weight.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"VLLM_DSPARK_DRAFT_HEAD_FP8: unsupported lm_head dtype {weight.dtype}"
            )
        self.size_n, self.size_k = n, k
        self.dtype = weight.dtype
        q, scale = quantize_rows_fp8(weight)
        # GPTQ layout: [K/4, N] int32, 4 consecutive k per word.
        packed = q.view(torch.int32).T.contiguous()
        del q
        self.qweight = ops.gptq_marlin_repack(
            b_q_weight=packed, size_k=k, size_n=n, num_bits=8, is_a_8bit=False
        )
        del packed
        s = marlin_permute_scales(
            s=scale.view(1, n), size_k=k, size_n=n, group_size=-1
        )
        self.scale = fp8_fused_exponent_bias_into_scales(s)
        # Keep the plain scale for tests / reference dequant.
        self.row_scale = scale
        self.workspace = marlin_make_workspace_new(weight.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == self.dtype, (x.dtype, self.dtype)
        return apply_fp8_marlin_linear(
            input=x,
            weight=self.qweight,
            weight_scale=self.scale,
            workspace=self.workspace,
            size_n=self.size_n,
            size_k=self.size_k,
            bias=None,
        )

    def nbytes(self) -> int:
        return (
            self.qweight.numel() * self.qweight.element_size()
            + self.scale.numel() * self.scale.element_size()
        )
