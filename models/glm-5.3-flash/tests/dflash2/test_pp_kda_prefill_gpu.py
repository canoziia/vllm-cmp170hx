"""GPU ONLY (sm_80, CMP 170HX / A100). Not executed by the author session.

Patch 0020: Morrowmake's 64-head sm_80 KDA chunk prefill
(vllm/models/glm5next/nvidia/ops/kda_prefill_pp.py) vs the incumbent FLA path
(vllm/models/glm5next/nvidia/ops/third_party/kda.chunk_kda_with_fused_gate) and
an fp64 token-sequential reference.

Run inside the patched container (0001..0018 + 0020 applied):

    python -m pytest -q -s tests/test_pp_kda_prefill_gpu.py
    python tests/test_pp_kda_prefill_gpu.py --bench      # timing table
    python tests/test_pp_kda_prefill_gpu.py --report     # error table

Shapes: 64 heads, head dim 128, chunk 64, production input layout (q, k, v
column views of one [T, 3*64*128] bf16 buffer), fp32 beta (pre-sigmoided),
fp32 A_log / dt_bias, fp32 state, safe gate at lower_bound -5, l2norm in
kernel, varlen cu_seqlens, with and without initial state.

Acceptance gate (pre-registered, not measured; MM's own criteria,
tests/kernels/test_ampere_pp_kda_prefill.py:9-14):
  * error vs fp64 of the candidate <= 1.10x the incumbent's on the mean and
    <= 1.25x on the max, for o and for final_state, every case;
  * candidate vs incumbent close at rounding level (reported, loose atol);
  * two candidate runs bitwise equal;
  * gather -> chunk -> scatter on a fake state pool: rows not addressed by the
    batch are bitwise untouched, addressed rows equal the returned final
    state, rows with has_initial_state=False start from zero (both paths).
"""
import itertools
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

H, D = 64, 128
P = H * D
LB = -5.0


def _mods():
    from vllm.models.glm5next.nvidia.ops import kda_prefill_pp as cand
    from vllm.models.glm5next.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate as inc,
    )

    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("candidate is sm_80-only by design")
    return cand, inc


# name -> (seqlens, which sequences have an initial state)
CASES = {
    "first_2304_noinit": ([2304], [False]),
    "cont_2304_init": ([2304], [True]),
    "tail_1282_init": ([1282], [True]),
    "varlen7_mixed": ([1000, 777, 64, 1, 63, 65, 334], [True, False, True, True, False, True, False]),
    "seqs16_mixed": ([144] * 15 + [148], [i % 2 == 0 for i in range(16)]),
    "short_1_init": ([1], [True]),
}


def make_inputs(seqlens, init_mask, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(seqlens)
    dev = "cuda"
    # post-conv silu activations, as in kda.py (q|k|v of one buffer)
    buf = torch.nn.functional.silu(torch.randn(T, 3 * P, device=dev, generator=g)).to(torch.bfloat16)
    raw_g = (torch.randn(1, T, H, D, device=dev, generator=g) * 2.0).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, T, H, device=dev, generator=g)).float()
    a_log = torch.log(torch.rand(1, 1, H, 1, device=dev, generator=g) * 15 + 1).float()
    dt = (torch.randn(P, device=dev, generator=g) * 0.5).float()
    N = len(seqlens)
    h0 = torch.randn(N, H, D, D, device=dev, generator=g) * 0.05
    h0[~torch.tensor(init_mask, device=dev)] = 0  # gather_initial_states semantics
    cu = torch.tensor([0] + list(itertools.accumulate(seqlens)), dtype=torch.int32, device=dev)
    return buf, raw_g, beta, a_log, dt, h0.float().contiguous(), cu


def call(fn, buf, raw_g, beta, a_log, dt, h0, cu):
    T = buf.shape[0]
    b = buf.clone()  # the incumbent writes o into its contiguous v copy; keep inputs pristine
    q, k, v = (b[:, i * P:(i + 1) * P].view(1, T, H, D) for i in range(3))
    return fn(q=q, k=k, v=v, raw_g=raw_g.clone(), beta=beta.clone(), A_log=a_log, g_bias=dt,
              initial_state=h0.clone(), output_final_state=True, use_qk_l2norm_in_kernel=True,
              cu_seqlens=cu, safe_gate=True, lower_bound=LB)


@torch.no_grad()
def reference(buf, raw_g, beta, a_log, dt, h0, cu):
    """fp64 token-sequential KDA; state S[n, h] is [V, K] (decode-pool layout)."""
    T = buf.shape[0]
    x = buf.double()
    q, k, v = (x[:, i * P:(i + 1) * P].view(T, H, D) for i in range(3))
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    gg = raw_g[0].double() + dt.double().view(H, D)
    gate = LB * torch.sigmoid(torch.exp(a_log.double().view(H, 1)) * gg)  # [T, H, D], natural log
    bt = beta[0].double()
    scale = D ** -0.5
    o = torch.empty(T, H, D, dtype=torch.float64, device=buf.device)
    hs = []
    cuh = cu.tolist()
    for n in range(len(cuh) - 1):
        S = h0[n].double().clone()  # [H, V, K]
        for t in range(cuh[n], cuh[n + 1]):
            S = S * torch.exp(gate[t])[:, None, :]
            u = bt[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", S, k[t]))
            S = S + u[:, :, None] * k[t][:, None, :]
            o[t] = scale * torch.einsum("hvk,hk->hv", S, q[t])
        hs.append(S)
    return o, torch.stack(hs)


def err(x, ref):
    d = (x.double() - ref).abs()
    return d.mean().item(), d.max().item()


def compare(name, seed=0):
    cand, inc = _mods()
    seqlens, mask = CASES[name]
    args = make_inputs(seqlens, mask, seed)
    o_i, s_i = call(inc, *args)
    o_c, s_c = call(cand.chunk_kda_with_fused_gate, *args)
    o_r, s_r = reference(*args)
    T = sum(seqlens)
    o_r = o_r.view(1, T, H, D)
    row = dict(
        case=name,
        o_inc=err(o_i, o_r), o_cand=err(o_c, o_r),
        s_inc=err(s_i, s_r), s_cand=err(s_c, s_r),
        o_diff=(o_c.float() - o_i.float()).abs().max().item(),
        s_diff=(s_c - s_i).abs().max().item(),
        o_bitwise=bool(torch.equal(o_c, o_i)), s_bitwise=bool(torch.equal(s_c, s_i)),
        o_scale=o_r.abs().max().item(), s_scale=s_r.abs().max().item(),
    )
    return row, (o_c, s_c, args)


@pytest.mark.parametrize("name", list(CASES))
def test_error_vs_fp64(name):
    row, _ = compare(name)
    print(row)
    for key in ("o", "s"):
        (mi, xi), (mc, xc) = row[f"{key}_inc"], row[f"{key}_cand"]
        floor = 1e-6  # both essentially exact: do not divide noise by noise
        assert mc <= 1.10 * mi + floor, (key, mc, mi)
        assert xc <= 1.25 * xi + floor, (key, xc, xi)
    # sanity: the reference itself is right (incumbent close to it)
    assert row["o_inc"][1] <= 0.05 * max(1.0, row["o_scale"])
    assert row["s_inc"][1] <= 0.05 * max(1.0, row["s_scale"])
    # rounding-level agreement between the two implementations
    assert row["o_diff"] <= 0.05 * max(1.0, row["o_scale"])
    assert row["s_diff"] <= 0.05 * max(1.0, row["s_scale"])


@pytest.mark.parametrize("name", ["cont_2304_init", "varlen7_mixed"])
def test_deterministic(name):
    cand, _ = _mods()
    args = make_inputs(*CASES[name])
    a = call(cand.chunk_kda_with_fused_gate, *args)
    b = call(cand.chunk_kda_with_fused_gate, *args)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("which", ["inc", "cand"])
def test_state_pool_semantics(which):
    """gather_initial_states -> chunk -> scatter_states, as kda.py does."""
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import gather_initial_states
    from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states

    cand, inc = _mods()
    fn = inc if which == "inc" else cand.chunk_kda_with_fused_gate
    seqlens, mask = CASES["varlen7_mixed"]
    buf, raw_g, beta, a_log, dt, _, cu = make_inputs(seqlens, mask)
    N = len(seqlens)
    pool = torch.randn(32, H, D, D, device="cuda") * 0.05
    before = pool.clone()
    idx = torch.tensor([3, 9, 17, 4, 30, 21, 11], dtype=torch.int32, device="cuda")
    has = torch.tensor(mask, device="cuda")
    h0 = gather_initial_states(pool, idx, has)
    assert torch.equal(h0[~has], torch.zeros_like(h0[~has]))
    T = buf.shape[0]
    q, k, v = (buf.clone()[:, i * P:(i + 1) * P].view(1, T, H, D) for i in range(3))
    _, fs = fn(q=q, k=k, v=v, raw_g=raw_g, beta=beta, A_log=a_log, g_bias=dt,
               initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True,
               cu_seqlens=cu, safe_gate=True, lower_bound=LB)
    assert fs.shape == (N, H, D, D) and fs.dtype == torch.float32 and fs.is_contiguous()
    scatter_states(pool, fs, idx)
    touched = torch.zeros(32, dtype=torch.bool, device="cuda")
    touched[idx.long()] = True
    assert torch.equal(pool[~touched], before[~touched])
    assert torch.equal(pool[idx.long()], fs)


def test_gates():
    cand, inc = _mods()
    assert cand.layer_closed_reason(64, 128, torch.bfloat16, True, -5.0, (8, 0)) == ""
    assert cand.layer_closed_reason(16, 128, torch.bfloat16, True, -5.0, (8, 0))
    assert cand.layer_closed_reason(64, 128, torch.bfloat16, True, -5.0, (8, 6))
    f32, bf = torch.float32, torch.bfloat16
    ok = (2304, 1, bf, bf, f32, f32, f32)
    assert cand.select_chunk_fn(inc, *ok) is cand.chunk_kda_with_fused_gate
    assert cand.select_chunk_fn(inc, 2313, 1, bf, bf, f32, f32, f32) is inc
    assert cand.select_chunk_fn(inc, 2304, 17, bf, bf, f32, f32, f32) is inc


def bench(iters=20):
    cand, inc = _mods()
    plans = {
        "first_2304_noinit": CASES["first_2304_noinit"],
        "cont_2304_init": CASES["cont_2304_init"],
        "tail_1282_init": CASES["tail_1282_init"],
        "varlen7_mixed": CASES["varlen7_mixed"],
    }
    print(f"{'case':22s} {'FLA ms':>9s} {'MM ms':>9s} {'speedup':>8s}")
    for name, (seqlens, mask) in plans.items():
        args = make_inputs(seqlens, mask)
        T = args[0].shape[0]
        res = {}
        for tag, fn in (("inc", inc), ("cand", cand.chunk_kda_with_fused_gate)):
            b = args[0].clone()
            q, k, v = (b[:, i * P:(i + 1) * P].view(1, T, H, D) for i in range(3))
            kw = dict(q=q, k=k, v=v, raw_g=args[1], beta=args[2], A_log=args[3], g_bias=args[4],
                      initial_state=args[5], output_final_state=True, use_qk_l2norm_in_kernel=True,
                      cu_seqlens=args[6], safe_gate=True, lower_bound=LB)
            for _ in range(3):
                fn(**kw)
            torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); fn(**kw); e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            res[tag] = sorted(ts)[len(ts) // 2]
        print(f"{name:22s} {res['inc']:9.3f} {res['cand']:9.3f} {res['inc'] / res['cand']:7.2f}x")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench(int(os.environ.get("ITERS", "20")))
    if "--report" in sys.argv:
        for name in CASES:
            print(compare(name)[0])
