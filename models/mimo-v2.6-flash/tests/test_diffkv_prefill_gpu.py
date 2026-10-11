"""TRITON_ATTN_DIFFKV prefill BLOCK_M (patch 0010) vs the default and fp64.

    python3 test_diffkv_prefill_gpu.py

One 512-token chunk at contexts 0 / 3000 (MiMo full layer: 64 q heads, 4 kv
heads, qk 192, v 128, block 64; and an SWA-128 layer with 8 kv heads + sinks).
BLOCK_M 64 must be bitwise equal to the default; 128 must not be less accurate
than the default against an fp64 reference.
"""
import os
import sys

import torch

import vllm.v1.attention.ops.triton_unified_attention_diffkv as m

dev = "cuda"
fails = []


def ref64(q, k, v, ctx, scale, window, sinks, hq, hkv):
    Q, L = q.shape[0], ctx + q.shape[0]
    kk = k.double().repeat_interleave(hq // hkv, dim=1)  # [L, hq, d]
    vv = v.double().repeat_interleave(hq // hkv, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.double(), kk) * scale
    qpos = torch.arange(ctx, L, device=dev)[:, None]
    kpos = torch.arange(L, device=dev)[None, :]
    mask = kpos <= qpos
    if window:
        mask &= (qpos - kpos) < window
    s = s.masked_fill(~mask[None], float("-inf"))
    if sinks is not None:
        s = torch.cat([s, sinks.double()[:, None, None].expand(hq, Q, 1)], dim=-1)
    p = s.softmax(-1)[..., :L]
    return torch.einsum("hqk,khd->qhd", p, vv)


def case(ctx, hq, hkv, window, use_sinks):
    Q, BS, DQK, DV = 512, 64, 192, 128
    L = ctx + Q
    nb = (L + BS - 1) // BS
    torch.manual_seed(ctx + hkv)
    kc = torch.randn(nb, BS, hkv, DQK, dtype=torch.bfloat16, device=dev)
    vc = torch.randn(nb, BS, hkv, DV, dtype=torch.bfloat16, device=dev)
    q = torch.randn(Q, hq, DQK, dtype=torch.bfloat16, device=dev)
    perm = torch.randperm(nb, device=dev, dtype=torch.int32)
    bt = perm.view(1, -1)
    cu = torch.tensor([0, Q], dtype=torch.int32, device=dev)
    sk = torch.tensor([L], dtype=torch.int32, device=dev)
    sinks = torch.randn(hq, dtype=torch.float32, device=dev) if use_sinks else None
    scale = DQK ** -0.5
    outs = {}
    os.environ["VLLM_MIMO26_DIFFKV_QK_SPLIT"] = "0"
    for bm in ("0", "64", "128", "256", "s64", "s128", "s256"):
        if bm.startswith("s"):
            os.environ["VLLM_MIMO26_DIFFKV_QK_SPLIT"] = "1"
        os.environ["VLLM_MIMO26_DIFFKV_PREFILL_BM"] = bm.lstrip("s")
        out = torch.empty(Q, hq, DV, dtype=torch.bfloat16, device=dev)
        m.unified_attention_diffkv(q, kc, vc, out, cu, sk, scale, True,
                                   (window - 1 if window else -1, 0), bt, 0.0,
                                   max_seqlen_q=Q, sinks=sinks)
        outs[bm] = out
    k_lin = kc[perm.long()].reshape(-1, hkv, DQK)[:L]
    v_lin = vc[perm.long()].reshape(-1, hkv, DV)[:L]
    r = ref64(q, k_lin, v_lin, ctx, scale, window, sinks, hq, hkv)
    e = {bm: (o.double() - r).abs().mean().item() for bm, o in outs.items()}
    eq64 = torch.equal(outs["64"], outs["0"])
    print(f"ctx={ctx:5d} hkv={hkv} window={window} sinks={use_sinks}: err default {e['0']:.3e} "
          f"bm64 {e['64']:.3e} (bitwise {eq64}) bm128 {e['128']:.3e} "
          f"(bitwise {torch.equal(outs['128'], outs['0'])}) bm256 {e['256']:.3e}")
    if not eq64:
        fails.append(f"bm64 not bitwise ctx={ctx} hkv={hkv}")
    for bm in ("64", "128", "256"):
        if not torch.equal(outs["s" + bm], outs[bm]):
            fails.append(f"qk split != padded bm{bm} ctx={ctx} hkv={hkv}")
    print("   qk split == padded (64/128/256):",
          [torch.equal(outs["s" + bm], outs[bm]) for bm in ("64", "128", "256")])
    if max(e["128"], e["256"]) > 1.05 * e["0"] + 1e-9:
        fails.append(f"bm128 accuracy ctx={ctx} hkv={hkv}")


for ctx in (0, 3000):
    case(ctx, 64, 4, 0, False)       # full attention layer
    case(ctx, 64, 8, 128, True)      # SWA-128 layer with sinks
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
