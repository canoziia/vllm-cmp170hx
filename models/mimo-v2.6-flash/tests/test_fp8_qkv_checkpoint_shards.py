#!/usr/bin/env python3
"""patch 0001: fused FP8 qkv_proj stored pre-sharded for the export TP.

MiMo-V2.6-Flash stores qkv_proj as 4 row shards (export TP = 4), each
[Q_s | K_s | V_s] with its own 128x128 block-scale grid padded to whole
blocks: SWA 4 x 3712 rows (29 scale rows each, 116 total), global
4 x 3392 rows (27 scale rows each, 108 total although 13568 rows are only
106 blocks). Builds such a matrix and checks every TP rank against the exact
dequantized reference. Needs a CUDA device for the vLLM import:

    python3 test_fp8_qkv_checkpoint_shards.py
"""
import types

import torch

from vllm.model_executor.models.mimo_v2 import (
    _shard_fp8_qkv_proj,
    mimo_v2_qkv_checkpoint_shards,
)

B = 128


def block_quant(w):
    rows, cols = w.shape
    rb, cb = -(-rows // B), -(-cols // B)
    pad = torch.zeros(rb * B, cb * B)
    pad[:rows, :cols] = w
    blocks = pad.view(rb, B, cb, B)
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    q = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.view(rb * B, cb * B)[:rows, :cols].contiguous(), scale


def dequant(q, s):
    se = s.repeat_interleave(B, 0).repeat_interleave(B, 1)[: q.shape[0], : q.shape[1]]
    return q.float() * se


cfg = types.SimpleNamespace(num_key_value_heads=4, swa_num_key_value_heads=8)
N = mimo_v2_qkv_checkpoint_shards(cfg)
assert N == 4, N

torch.manual_seed(0)
hidden = 256
hd, vd = 192, 128
for name, heads, kv in (("swa", 64, 8), ("global", 64, 4)):
    qh, kh = heads // N, kv // N
    # reference full Q/K/V in head order
    Q = torch.randn(heads * hd, hidden)
    K = torch.randn(kv * hd, hidden) * 3
    V = torch.randn(kv * vd, hidden) * 0.5
    ws, ss, refs = [], [], []
    for i in range(N):
        shard = torch.cat([Q[i * qh * hd:(i + 1) * qh * hd],
                           K[i * kh * hd:(i + 1) * kh * hd],
                           V[i * kh * vd:(i + 1) * kh * vd]])
        wq, sq = block_quant(shard)  # per-shard grid, padded to whole blocks
        ws.append(wq); ss.append(sq); refs.append(dequant(wq, sq))
    w_full, s_full = torch.cat(ws), torch.cat(ss)
    ref = torch.cat(refs)
    shard_rows = ref.shape[0] // N
    q_rows, k_rows = qh * hd, kh * hd
    RQ = torch.cat([ref[i * shard_rows:i * shard_rows + q_rows] for i in range(N)])
    RK = torch.cat([ref[i * shard_rows + q_rows:i * shard_rows + q_rows + k_rows] for i in range(N)])
    RV = torch.cat([ref[i * shard_rows + q_rows + k_rows:(i + 1) * shard_rows] for i in range(N)])
    for tp in (1, 2, 4):
        for rank in range(tp):
            w_r, s_r = _shard_fp8_qkv_proj(
                w_full, s_full, num_heads=heads, num_kv_heads=kv, head_dim=hd,
                v_head_dim=vd, tp_rank=rank, tp_size=tp, ckpt_shards=N)
            got = dequant(w_r, s_r)
            rq, rk, rv = heads // tp * hd, kv // tp * hd, kv // tp * vd
            exp = torch.cat([RQ[rank * rq:(rank + 1) * rq], RK[rank * rk:(rank + 1) * rk],
                             RV[rank * rv:(rank + 1) * rv]])
            assert got.shape == exp.shape, (name, tp, rank, got.shape, exp.shape)
            rel = ((got - exp).norm() / exp.norm()).item()
            limit = 0.0 if tp == N else 0.05  # tp == N: no re-quantisation
            assert rel <= limit, (name, tp, rank, rel)
    print(f"PASS {name}: {N} checkpoint shards x {shard_rows} rows, "
          f"{s_full.shape[0]} scale rows; tp 1/2/4 match the reference")
