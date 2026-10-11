#!/usr/bin/env python3
"""GPU checks for Qwen3.8 GDN RecoverSSM (VLLM_QWEN38_GDN_RECOVER).

  python3 test_gdn_recover_gpu.py

1. The recording verify kernel returns the baseline outputs bitwise and leaves
   the committed state untouched.
2. Replaying n = 1..K+1 recorded tokens reproduces the baseline per-token
   state bitwise.
3. 40 multi-step verify steps of one GDN layer (real causal_conv1d_update and
   recurrent kernel, random acceptance, conv compaction, 2 requests) give
   bitwise identical outputs to baseline speculative state blocks.
4. Same for the PLE short conv (compaction vs num_accepted offset).
"""
import random
import sys

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.qwen4_exp.nvidia import gdn_recover as gr
from vllm.models.qwen4_exp.nvidia.ops.ple import ple_conv
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update as F,
)

dev = "cuda"
N, T, H, HV, K, V, KS = 2, 6, 16, 48, 128, 128, 4
i32 = lambda x: torch.tensor(x, device=dev, dtype=torch.int32)  # noqa: E731
i64 = lambda x: torch.tensor(x, device=dev, dtype=torch.int64)  # noqa: E731
ok = True


def report(name, good):
    global ok
    print(("PASS" if good else "FAIL"), name)
    ok &= bool(good)


def replay(state, rec, slots, plan, bt):
    gr._replay_kernel[(V // 32, slots.numel(), HV)](
        slots, plan, i64([bt.data_ptr()]), bt.stride(0), i32([0]),
        i64([state.data_ptr()]), i64([state.stride(0)]), i64([rec.data_ptr()]), state,
        H=H, HV=HV, K=K, V=V, BV=32, REC_T=rec.shape[1], REC_F=rec.shape[2], num_warps=4)


def compact(conv, slots, plan, bt, dims, hist):
    gr._compact_conv_kernel[((dims + 127) // 128, slots.numel(), 1)](
        slots, plan, i64([bt.data_ptr()]), bt.stride(0), i32([0]),
        i64([conv.data_ptr()]), i64([conv.stride(0)]), i64([conv.stride(1)]),
        i64([conv.stride(2)]), i32([dims]), i32([hist]), conv,
        BLOCK_D=128, BLOCK_HIST=triton_pow2(hist))


def triton_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


torch.manual_seed(0)
random.seed(0)
A_log = torch.randn(HV, device=dev) * 0.5
dt_bias = torch.randn(HV, device=dev) * 0.5

# 1 + 2: single verify step
state = torch.randn(16, HV, V, K, device=dev) * 0.1
q = torch.randn(1, N * T, H, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, N * T, H, K, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, N * T, HV, V, device=dev, dtype=torch.bfloat16)
a = torch.randn(N * T, HV, device=dev, dtype=torch.bfloat16)
b = torch.randn(N * T, HV, device=dev, dtype=torch.bfloat16)
cu = torch.arange(0, N * T + 1, T, device=dev, dtype=torch.int32)
src = torch.arange(1, N + 1, device=dev, dtype=torch.int32)
idx = torch.stack([src] + [3 + (c - 1) * N + torch.arange(N, device=dev, dtype=torch.int32) for c in range(1, T)], 1)
ones = torch.ones(N, device=dev, dtype=torch.int32)
base = state.clone()
ob, _ = F(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, initial_state=base,
          inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=idx,
          num_accepted_tokens=ones, use_qk_l2norm_in_kernel=True)
rs = state.clone()
rec = torch.zeros(4, T, HV + H * K + HV * V, device=dev)
slots = i32([3, 0])
orr, _ = F(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, initial_state=rs,
           inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=idx,
           num_accepted_tokens=ones, use_qk_l2norm_in_kernel=True, rec=rec, rec_slots=slots)
report("record verify outputs bitwise, state untouched", torch.equal(ob, orr) and torch.equal(rs, state))
good = True
for n in range(1, T + 1):
    out = state.clone()
    bt = torch.zeros(4, 2, device=dev, dtype=torch.int32)
    plan = torch.zeros(4, 5, device=dev, dtype=torch.int32)
    for r in range(N):
        s = int(slots[r])
        bt[s, 0] = int(src[r])
        bt[s, 1] = 13 + r
        plan[s] = i32([n, 0, 1, -1, 0])
    replay(out, rec, slots, plan, bt)
    good &= all(torch.equal(base[int(idx[r, n - 1])], out[13 + r]) for r in range(N))
report("replay n=1..K+1 bitwise", good)

# 3: multi-step GDN layer
D = 2 * H * K + HV * V
W = KS - 1 + T - 1
w = torch.randn(D, KS, device=dev, dtype=torch.bfloat16) * 0.3
bias = torch.randn(D, device=dev, dtype=torch.bfloat16) * 0.1
ssm0 = torch.randn(13, HV, V, K, device=dev) * 0.05
conv0 = torch.randn(13, D, W, device=dev, dtype=torch.bfloat16)
ssm_b, conv_b, ssm_r, conv_r = ssm0.clone(), conv0.clone(), ssm0.clone(), conv0.clone()
idx_b = i32([[1 + 6 * r + c for c in range(T)] for r in range(N)])
idx_r = idx_b[:, :1].expand(-1, T).contiguous()
na_b, na_r = ones.clone(), ones.clone()
slots = i32([1, 0])
rec = torch.zeros(4, T, HV + H * K + HV * V, device=dev)


def layer(x, conv, ssm, ix, na, extra):
    y = causal_conv1d_update(x, conv, w, bias, "silu", conv_state_indices=ix[:, 0],
                             num_accepted_tokens=na, query_start_loc=cu, max_query_len=T)
    qq = y[:, : H * K].reshape(1, -1, H, K)
    kk = y[:, H * K: 2 * H * K].reshape(1, -1, H, K)
    vv = y[:, 2 * H * K:].reshape(1, -1, HV, V)
    o, _ = F(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=qq, k=kk, v=vv, initial_state=ssm,
             inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=ix,
             num_accepted_tokens=na, use_qk_l2norm_in_kernel=True, **extra)
    return o


good = True
for step in range(40):
    x = torch.randn(N * T, D, device=dev, dtype=torch.bfloat16)
    a = torch.randn(N * T, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N * T, HV, device=dev, dtype=torch.bfloat16)
    good &= torch.equal(layer(x.clone(), conv_b, ssm_b, idx_b, na_b, {}),
                        layer(x.clone(), conv_r, ssm_r, idx_r, na_r, {"rec": rec, "rec_slots": slots}))
    n = [random.randint(1, T) for _ in range(N)]
    na_b = i32(n)
    bt = torch.zeros(4, 2, device=dev, dtype=torch.int32)
    plan = torch.zeros(4, 5, device=dev, dtype=torch.int32)
    for r in range(N):
        s = int(slots[r])
        bt[s, 0] = 1 + 6 * r
        plan[s] = i32([n[r], 0, 0, -1, 0])
    replay(ssm_r, rec, slots, plan, bt)
    compact(conv_r, slots, plan, bt, D, KS - 1)
    na_r = ones.clone()
report("40-step GDN layer (conv + recurrent, random acceptance) bitwise", good)

# 4: PLE short conv
C, DIL = 10240, 3
SL = (KS - 1) * DIL
wts = torch.randn(C, KS, device=dev, dtype=torch.bfloat16) * 0.3
st0 = torch.randn(8, C, SL + T - 1, device=dev, dtype=torch.bfloat16)
st_b, st_r = st0.clone(), st0.clone()
pidx = i32([1, 4])
na_b, na_r = ones.clone(), ones.clone()
good = True
for step in range(40):
    x = torch.randn(N * T, C, device=dev, dtype=torch.bfloat16)
    res = torch.randn(N * T, C, device=dev, dtype=torch.bfloat16)
    rb, rr = res.clone(), res.clone()
    ple_conv(x.clone(), rb, st_b, wts, pidx, mode="spec", dilation=DIL, query_start_loc=cu,
             num_accepted_tokens=na_b, spec_query_len=T)
    ple_conv(x.clone(), rr, st_r, wts, pidx, mode="spec", dilation=DIL, query_start_loc=cu,
             num_accepted_tokens=na_r, spec_query_len=T)
    good &= torch.equal(rb, rr)
    n = [random.randint(1, T) for _ in range(N)]
    na_b = i32(n)
    bt = torch.zeros(4, 2, device=dev, dtype=torch.int32)
    plan = torch.zeros(4, 5, device=dev, dtype=torch.int32)
    for r in range(N):
        s = int(slots[r])
        bt[s, 0] = int(pidx[r])
        plan[s] = i32([n[r], 0, 0, -1, 0])
    compact(st_r, slots, plan, bt, C, SL)
    na_r = ones.clone()
report("40-step PLE short conv bitwise", good)

print("QWEN38_GDN_RECOVER_GPU", "ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
