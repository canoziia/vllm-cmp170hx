# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the compiled Marlin MoE decode test and benchmark (0006).

Builds one GLM-5.3-Flash routed-expert layer in exactly the format our fork
produces for canada-quant/GLM-5.3-Flash-W4A16-MTP (compressed-tensors
pack-quantized INT4, symmetric, group 128 -> uint4b8 Marlin):

    GPTQ-packed int32 [E, K/8, 2N] --gptq_marlin_moe_repack--> [E, K/16, 4N]
    bf16 scales     [E, K/128, 2N] --marlin_moe_permute_scales--> same shape

with random nibbles and random positive scales, using the same vLLM helpers
as ``oracle/int_wna16.py`` (so a layout mismatch between our repack and the
compiled kernel shows up here). Real dimensions by default: E=288, top-8,
K=4096, N=2048 (PP whole experts); N=512 is the TP4 shard.

Each expert's raw tensors are regenerated from a per-expert seed, so the fp32
dense reference can dequantise only the experts a test routes to.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch

E_REAL, TOPK_REAL, K_REAL, N_REAL, GROUP = 288, 8, 4096, 2048, 128
CLAMP = float(os.environ.get("GLM_SWIGLU_LIMIT", "10.0"))


def _raw_expert(e: int, K: int, N2: int, seed: int, device):
    """GPTQ int32 [K/8, N2] (8 nibbles along K) and bf16 scales [K/128, N2]."""
    g = torch.Generator(device=device).manual_seed(seed * 100003 + e)
    q = torch.randint(-(2**31), 2**31 - 1, (K // 8, N2), generator=g,
                      device=device, dtype=torch.int64).to(torch.int32)
    s = (0.002 * (0.5 + torch.rand(K // GROUP, N2, generator=g, device=device))
         ).to(torch.bfloat16)
    return q, s


def dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """[K/8, N2] int32 + [K/128, N2] bf16 -> fp32 [K, N2] = (nib - 8) * s."""
    k8, n2 = q.shape
    shifts = torch.arange(0, 32, 4, device=q.device, dtype=torch.int32)
    nib = (q.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF        # [K/8, 8, N2]
    w = nib.reshape(k8 * 8, n2).float() - 8.0
    return w * s.float().repeat_interleave(GROUP, dim=0)


@dataclass
class MoELayer:
    E: int
    K: int
    N: int
    topk: int
    seed: int
    w1: torch.Tensor          # Marlin [E, K/16, 4N] int32
    w2: torch.Tensor          # Marlin [E, N/16, 2K] int32
    w1_scale: torch.Tensor    # [E, K/128, 2N] bf16 (Marlin-permuted)
    w2_scale: torch.Tensor    # [E, N/128, K] bf16
    stub: SimpleNamespace     # what marlin_decode.run() reads from MarlinExperts


def build_layer(E=E_REAL, K=K_REAL, N=N_REAL, topk=TOPK_REAL, seed=0,
                device="cuda") -> MoELayer:
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_moe_permute_scales,
    )
    from vllm.scalar_type import scalar_types

    w1 = torch.empty(E, K // 16, 4 * N, dtype=torch.int32, device=device)
    w2 = torch.empty(E, N // 16, 2 * K, dtype=torch.int32, device=device)
    s1 = torch.empty(E, K // GROUP, 2 * N, dtype=torch.bfloat16, device=device)
    s2 = torch.empty(E, N // GROUP, K, dtype=torch.bfloat16, device=device)
    for e in range(E):  # one expert at a time keeps the peak small (8 GB cards)
        q1, sc1 = _raw_expert(e, K, 2 * N, seed, device)
        q2, sc2 = _raw_expert(e, N, K, seed + 1, device)
        w1[e] = ops.gptq_marlin_moe_repack(q1[None], K, 2 * N, 4)[0]
        w2[e] = ops.gptq_marlin_moe_repack(q2[None], N, K, 4)[0]
        s1[e] = marlin_moe_permute_scales(sc1[None], K, 2 * N, GROUP)[0]
        s2[e] = marlin_moe_permute_scales(sc2[None], N, K, GROUP)[0]
    cfg = ApplyMoEActivationConfig(clamp_limit=CLAMP)
    stub = SimpleNamespace(
        w1_scale=s1, w2_scale=s2, activation_config=cfg,
        quant_type_id=scalar_types.uint4b8.id, input_dtype=None,
        w1_zp=None, w2_zp=None, w1_bias=None, w2_bias=None, g1_alphas=None,
        g2_alphas=None, a1_gscale=None, a2_gscale=None, _lora_context=None,
        moe_sum=lambda inp, out, ids, em: ops.moe_sum(inp, out),
    )
    return MoELayer(E, K, N, topk, seed, w1, w2, s1, s2, stub)


def routing(M: int, layer: MoELayer, seed: int, device="cuda", pool: int = 0):
    """Random routing; ``pool`` > 0 draws every token's experts from one random
    subset of that many experts (speculative-decode tokens of one request
    share most experts; pool=0 means independent tokens)."""
    g = torch.Generator().manual_seed(seed)
    if pool:
        assert layer.topk <= pool <= layer.E
        subset = torch.randperm(layer.E, generator=g)[:pool]
        ids = torch.stack([subset[torch.randperm(pool, generator=g)[:layer.topk]]
                           for _ in range(M)]).to(torch.int32)
    else:
        ids = torch.stack([torch.randperm(layer.E, generator=g)[:layer.topk]
                           for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, layer.topk, generator=g).softmax(-1).float()
    x = (torch.randn(M, layer.K, generator=g) * 0.5).to(torch.bfloat16)
    return x.to(device), ids.to(device), w.to(device)


def run_marlin(layer: MoELayer, x, ids, w):
    """The incumbent: fused_marlin_moe, as MarlinExperts.apply calls it."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )

    if not hasattr(layer, "_ws"):
        layer._ws = marlin_make_workspace_new(x.device, 4)
    out = torch.empty_like(x)
    ret = fused_marlin_moe(
        hidden_states=x, w1=layer.w1, w2=layer.w2, bias1=None, bias2=None,
        w1_scale=layer.w1_scale, w2_scale=layer.w2_scale, topk_weights=w,
        topk_ids=ids, quant_type_id=layer.stub.quant_type_id,
        global_num_experts=layer.E, activation=MoEActivation.SILU,
        activation_config=layer.stub.activation_config, moe_sum=layer.stub.moe_sum,
        output=out, workspace=layer._ws,
    )
    # Our fused_marlin_moe writes into `output` and returns None.
    return out if ret is None else ret


def run_compiled(layer: MoELayer, x, ids, w, variant=None, out=None):
    from vllm.models.glm5next.nvidia.ops.marlin_decode import run

    out = torch.empty_like(x) if out is None else out
    return run(layer.stub, out, x, layer.w1, layer.w2, w, ids, variant_name=variant)


def run_dense_fp32(layer: MoELayer, x, ids, w):
    """fp32 dense reference from the raw (pre-repack) tensors."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in ids.unique().tolist():
        q1, sc1 = _raw_expert(e, layer.K, 2 * layer.N, layer.seed, x.device)
        q2, sc2 = _raw_expert(e, layer.N, layer.K, layer.seed + 1, x.device)
        W1, W2 = dequant(q1, sc1), dequant(q2, sc2)         # [K,2N], [N,K]
        rows, cols = (ids == e).nonzero(as_tuple=True)
        y = xf[rows] @ W1
        g = y[:, :layer.N].clamp(max=CLAMP)
        u = y[:, layer.N:].clamp(-CLAMP, CLAMP)
        h = torch.nn.functional.silu(g) * u
        out.index_add_(0, rows, (h @ W2) * w[rows, cols, None])
    return out


def rel_l2(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
