"""GPU ONLY (sm_80, CMP 170HX / A100). Not executed by the author session.

Patch 0017: 64-head Gluon sparse-MLA prefill vs the incumbent Triton kernel
(ops/triton_mla_sparse_kernel.triton_mla_sparse_attention) and an fp64 reference.

Run inside the patched container (0001..0014, 0016, 0017 applied):

    python -m pytest -q -s tests/test_pp_sparse_prefill_gpu.py
    python tests/test_pp_sparse_prefill_gpu.py --bench   # timing table

Acceptance gate (pre-registered, not measured):
  * the kernel compiles on this Triton build (test_compiles);
  * for every case, candidate error vs fp64 <= 1.10x the incumbent's error
    (max-abs and RMS), plus an absolute bound atol=0.02 on the output;
  * all-invalid rows give exactly 0 output;
  * graph replay output == eager output bitwise; two eager runs bitwise equal.
"""
import math
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

H, D, W = 64, 512, 2176
TOPK = 2048


def _mods():
    from vllm.models.glm5next.nvidia.ops import sparse_prefill_mla_pp as cand
    from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
        triton_mla_sparse_attention,
    )

    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("candidate is sm_80-only by design")
    return cand, triton_mla_sparse_attention


def make_case(kind: str, T: int, ctx: int, rows: int, seed: int = 123):
    """Cache rows are a flat [rows, 1, 512] bf16 view; a request's tokens sit at
    rows [base, base + ctx). Indices mimic the kpool indexer output: 2048 top-k
    slots (-1 where the causal prefix is shorter) + 128-slot local section that
    is mostly -1 (only kpool-1 = 3 tail tokens valid)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = (torch.randn(rows, 1, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    q = (torch.randn(T, H, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    base = 4608 * 7  # not block 0, like a real allocation
    assert base + ctx <= rows
    idx = torch.full((T, 1, W), -1, dtype=torch.int32, device="cuda")
    # query t sits at position (ctx - T + t)
    pos = torch.arange(ctx - T, ctx, device="cuda")
    for t0 in range(0, T, 256):
        t1 = min(T, t0 + 256)
        p = pos[t0:t1]
        n_vis = p + 1  # causal prefix length
        if kind == "first":  # causal first chunk: <= 2048 visible -> all of them
            k = torch.arange(TOPK, device="cuda")[None, :].expand(t1 - t0, -1)
            sel = torch.where(k < n_vis[:, None], k, -1)
        else:  # random top-k over the visible prefix (unique per row)
            r = torch.rand(t1 - t0, ctx, device="cuda", generator=g)
            r = torch.where(
                torch.arange(ctx, device="cuda")[None, :] < n_vis[:, None], r, -1.0
            )
            sel = r.topk(TOPK, dim=1).indices
            sel = torch.where(sel < n_vis[:, None], sel, -1)
        sel = torch.where(sel >= 0, sel + base, -1).to(torch.int32)
        idx[t0:t1, 0, :TOPK] = sel
        # local section: 3 valid tail tokens, rest -1
        tail = (p[:, None] - torch.arange(3, device="cuda")[None, :]).clamp_min(0) + base
        idx[t0:t1, 0, TOPK : TOPK + 3] = tail.to(torch.int32)
    if kind == "empty_rows":
        idx[: min(T, 7)] = -1
    return q, kv, idx


def reference(q, kv, idx, scale, rows_chunk=64):
    """fp64 gathered attention; masked slots excluded; empty row -> 0."""
    T = q.shape[0]
    out = torch.zeros(T, H, D, dtype=torch.float64, device="cuda")
    for t0 in range(0, T, rows_chunk):
        t1 = min(T, t0 + rows_chunk)
        ii = idx[t0:t1, 0].long()
        valid = (ii >= 0) & (ii < kv.shape[0])
        k = kv[ii.clamp_min(0), 0].double()  # [t, W, D]
        s = torch.einsum("thd,twd->thw", q[t0:t1].double(), k) * scale
        s = s.masked_fill(~valid[:, None, :], -math.inf)
        m = s.amax(-1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(s - m)
        den = p.sum(-1, keepdim=True)
        o = torch.einsum("thw,twd->thd", p, k) / torch.where(den > 0, den, 1.0)
        out[t0:t1] = o
    return out


def errs(a, ref):
    d = (a.double() - ref).abs()
    return d.max().item(), d.pow(2).mean().sqrt().item()


def test_compiles():
    cand, _ = _mods()
    cand.warmup("cuda")


CASES = [  # (kind, T, ctx)
    ("first", 2312, 2312),
    ("cont", 2312, 8192),
    ("cont", 2296, 32768),
    ("cont", 1282, 65536),
    ("cont", 384, 16384),
    ("empty_rows", 512, 4096),
]


@pytest.mark.parametrize("kind,T,ctx", CASES)
def test_numerics(kind, T, ctx):
    cand, inc = _mods()
    rows = 4608 * 7 + ctx + 4608
    q, kv, idx = make_case(kind, T, ctx, rows)
    scale = 1.0 / math.sqrt(D)
    assert cand.closed(q, kv, idx, D, 0, None) == ""
    out_c, mx, lse = cand.sparse_mla_prefill(q, kv, idx, scale)
    out_i = inc(q, kv, idx, sm_scale=scale)
    ref = reference(q, kv, idx, scale)
    ec, ei = errs(out_c, ref), errs(out_i, ref)
    pair = (out_c.float() - out_i.float()).abs().max().item()
    print(f"\n{kind} T={T} ctx={ctx}: cand max/rms {ec[0]:.3e}/{ec[1]:.3e}  "
          f"incumbent {ei[0]:.3e}/{ei[1]:.3e}  pair max {pair:.3e}")
    assert torch.isfinite(out_c.float()).all()
    assert ec[0] <= 1.10 * ei[0] + 1e-6 and ec[1] <= 1.10 * ei[1] + 1e-7
    torch.testing.assert_close(out_c.double(), ref, atol=0.02, rtol=0.0)
    if kind == "empty_rows":
        assert (out_c[:7] == 0).all()
    # run-to-run bitwise
    out_c2, _, _ = cand.sparse_mla_prefill(q, kv, idx, scale)
    assert torch.equal(out_c, out_c2)


def test_graph_replay_bitwise():
    cand, _ = _mods()
    q, kv, idx = make_case("cont", 1024, 8192, 4608 * 7 + 8192 + 4608)
    scale = 1.0 / math.sqrt(D)
    eager, _, _ = cand.sparse_mla_prefill(q, kv, idx, scale)
    out = torch.empty_like(eager)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        cand.sparse_mla_prefill(q, kv, idx, scale, out=out)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cand.sparse_mla_prefill(q, kv, idx, scale, out=out)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_backend_dispatch_gate(monkeypatch):
    """Flag off -> untouched; flag on but below MIN_TOKENS -> untouched."""
    cand, _ = _mods()
    q, kv, idx = make_case("cont", 64, 4096, 4608 * 7 + 4096 + 4608)
    assert cand.closed(q, kv, idx[:, :, :2048].contiguous(), D, 0, None) != ""
    assert cand.closed(q[:, :16].contiguous(), kv, idx, D, 0, None) != ""


def bench():
    cand, inc = _mods()
    scale = 1.0 / math.sqrt(D)
    # Large cache (like a production pool) so 128k-context gathers miss L2.
    rows_big = int(os.environ.get("BENCH_ROWS", str(1_400_000)))
    print("kind,T,ctx,incumbent_ms,candidate_ms,speedup")
    for kind, T, ctx in [("first", 2312, 2312), ("cont", 2312, 8192),
                         ("cont", 2312, 32768), ("cont", 2312, 114688),
                         ("cont", 2312, 229376)]:
        rows = max(rows_big, 4608 * 7 + ctx + 4608)
        q, kv, idx = make_case(kind, T, ctx, rows)
        res = []
        for fn in (lambda: inc(q, kv, idx, sm_scale=scale),
                   lambda: cand.sparse_mla_prefill(q, kv, idx, scale)):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(True), torch.cuda.Event(True)
            ts = []
            for _ in range(10):
                st.record(); fn(); en.record(); en.synchronize()
                ts.append(st.elapsed_time(en))
            ts.sort()
            res.append(ts[len(ts) // 2])
        print(f"{kind},{T},{ctx},{res[0]:.3f},{res[1]:.3f},{res[0]/res[1]:.2f}")
        del q, kv, idx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        sys.exit(pytest.main([__file__, "-q", "-s"]))
