#!/usr/bin/env python3
"""GPU check for the native GDN MTP recording kernel (_qwen38_gdn_mtp_C).

  python3 test_gdn_mtp_record_native.py [path/to/_qwen38_gdn_mtp_C.abi3.so]

Against the image's fused CUDA kernel (torch.ops._C.fused_gdn_decode_post_conv_mtp):
1. without recording: outputs and every per-token state bit-identical;
2. with recording: outputs bit-identical, committed state untouched, and
   replaying n = 1..T recorded tokens (decay mode) reproduces the per-token
   states bit-identically.
"""
import glob
import sys

import torch

import vllm  # noqa: F401  (loads torch.ops._C)
from vllm import _custom_ops as ops
from vllm.models.qwen4_exp.nvidia import gdn_recover as gr

so = sys.argv[1] if len(sys.argv) > 1 else (glob.glob(
    "/usr/local/lib/python3.12/dist-packages/vllm/_qwen38_gdn_mtp_C*.so") + [None])[0]
torch.ops.load_library(so)
native = torch.ops._qwen38_gdn_mtp_C.post_conv_mtp

dev = "cuda"
torch.manual_seed(0)
N, T, H, HV, K, V = 3, 6, 16, 48, 128, 128
L = N * T
ok = True


def report(name, good):
    global ok
    print(("PASS" if good else "FAIL"), name)
    ok &= bool(good)


mixed = torch.randn(L, 2 * H * K + HV * V, device=dev, dtype=torch.bfloat16)
a = torch.randn(L, HV, device=dev, dtype=torch.bfloat16)
b = torch.randn(L, HV, device=dev, dtype=torch.bfloat16)
A_log = torch.randn(HV, device=dev) * 0.5
dt_bias = torch.randn(HV, device=dev)
gate = torch.randn(L, HV, V, device=dev, dtype=torch.bfloat16)
normw = torch.randn(V, device=dev, dtype=torch.bfloat16)
cu = torch.arange(0, L + 1, T, device=dev, dtype=torch.int32)
na = torch.ones(N, device=dev, dtype=torch.int32)
src = torch.tensor([1, 2, 3], device=dev, dtype=torch.int32)
idx = torch.stack([src] + [4 + (c - 1) * N + torch.arange(N, device=dev, dtype=torch.int32)
                           for c in range(1, T)], 1).contiguous()
nblk = 24
state0 = torch.randn(nblk, HV, V, K, device=dev) * 0.1

for act in ("silu", "sigmoid"):
    s_ref = state0.clone()
    o_ref = torch.empty(L, HV, V, device=dev, dtype=torch.bfloat16)
    torch.ops._C.fused_gdn_decode_post_conv_mtp(mixed, a, b, A_log, dt_bias, idx, cu, na, s_ref,
                                                 gate, normw, o_ref, K ** -0.5, 1e-6, act)
    s_nat = state0.clone()
    o_nat = torch.empty_like(o_ref)
    native(mixed, a, b, A_log, dt_bias, idx, cu, na, s_nat, gate, normw, o_nat, K ** -0.5, 1e-6, act, None, None)
    report(f"[{act}] port without recording: outputs and states bitwise",
           torch.equal(o_ref, o_nat) and torch.equal(s_ref, s_nat))

    s_rec = state0.clone()
    o_rec = torch.empty_like(o_ref)
    rec = torch.zeros(4, T, HV + H * K + HV * V, device=dev)
    slots = torch.tensor([2, 0, 3], device=dev, dtype=torch.int32)
    idx0 = idx[:, :1].expand(-1, T).contiguous()
    native(mixed, a, b, A_log, dt_bias, idx0, cu, na, s_rec, gate, normw, o_rec, K ** -0.5, 1e-6, act, rec, slots)
    report(f"[{act}] recording: outputs bitwise, committed state untouched",
           torch.equal(o_ref, o_rec) and torch.equal(s_rec, state0))

    good = True
    for n in range(1, T + 1):
        out = state0.clone()
        bt = torch.zeros(4, 2, device=dev, dtype=torch.int32)
        plan = torch.zeros(4, 5, device=dev, dtype=torch.int32)
        for r in range(N):
            s = int(slots[r])
            bt[s, 0] = int(src[r])
            bt[s, 1] = 20 + r
            plan[s] = torch.tensor([n, 0, 1, -1, 0], dtype=torch.int32)
        i64 = lambda x: torch.tensor(x, device=dev, dtype=torch.int64)  # noqa: E731
        gr._replay_kernel[(V // 32, N, HV)](
            slots, plan, i64([bt.data_ptr()]), bt.stride(0), torch.zeros(1, device=dev, dtype=torch.int32),
            i64([out.data_ptr()]), i64([out.stride(0)]), i64([rec.data_ptr()]), out,
            H=H, HV=HV, K=K, V=V, BV=32, REC_T=T, REC_F=rec.shape[2], USE_DECAY=True, num_warps=4)
        for r in range(N):
            good &= torch.equal(s_ref[int(idx[r, n - 1])], out[20 + r])
    report(f"[{act}] replay n=1..T (decay mode) bitwise vs fused CUDA per-token states", good)

print("QWEN38_GDN_MTP_RECORD_NATIVE", "ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
