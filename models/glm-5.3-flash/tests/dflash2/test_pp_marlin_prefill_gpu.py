"""GPU ONLY (sm_80, CMP 170HX / A100). Not executed by the author session.

Patch 0019: split-block Marlin MoE prefill (marlin_prefill_split.py) vs the
incumbent fused_marlin_moe, at the real PP4 shape M=2304, E=288, top-8,
K=4096, N=2048, uint4b8 group 128, SiLU clamp 10.

Run inside the patched container (0001..0019 applied):

    VLLM_GLM5_PP_MARLIN_PREFILL=1 python -m pytest -q -s tests/test_pp_marlin_prefill_gpu.py
    VLLM_GLM5_PP_MARLIN_PREFILL=1 python tests/test_pp_marlin_prefill_gpu.py --bench

Weights: NB=8 distinct experts quantised with marlin_quantize (so an fp32
reference exists), replicated to 288 experts (expert e uses base e % NB).
Routing is over all 288 experts, so the block lists / launches are the real
ones; only the weight bytes repeat (L2 effects may be slightly optimistic for
both paths alike).  Memory ~5 GB.

Acceptance (pre-registered, not measured):
  * candidate vs incumbent: NOT required bitwise (stream-K split differs);
    report the fraction of bitwise-equal elements and max |diff| in bf16 ulps;
  * vs fp32 reference: candidate max-abs and RMS error <= 1.10x incumbent's;
  * two eager candidate runs bitwise equal; CUDA-graph replay == eager bitwise;
  * gate closed (returns False, output untouched) for M < MIN_TOKENS and M=8.
"""
import os
import sys

os.environ.setdefault("VLLM_GLM5_PP_MARLIN_PREFILL", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

# Import the fused-MoE package first so marlin_utils is fully initialised
# before marlin_utils_test (avoids a circular import in this tree).
import vllm.model_executor.layers.fused_moe.experts.marlin_moe  # noqa: E402,F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

E, TOPK, K, N, G, NB = 288, 8, 4096, 2048, 128, 8
CLAMP = 10.0
_STATE = {}


class FakeLayer:
    """The attributes MarlinExperts exposes to the hook and fused path."""

    def __init__(self, w1_scale, w2_scale):
        from vllm.model_executor.layers.fused_moe.activation import (
            ApplyMoEActivationConfig,
        )
        from vllm.scalar_type import scalar_types

        self.w1_scale, self.w2_scale = w1_scale, w2_scale
        self.input_dtype = None
        for n in ("w1_zp", "w2_zp", "w1_bias", "w2_bias", "g1_alphas",
                  "g2_alphas", "a1_gscale", "a2_gscale"):
            setattr(self, n, None)
        self.quant_type_id = scalar_types.uint4b8.id
        self.activation_config = ApplyMoEActivationConfig(clamp_limit=CLAMP)
        self._ws = None

    def marlin_workspace(self, device):
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
        )

        if self._ws is None:
            self._ws = marlin_make_workspace_new(device, 4)
        return self._ws

    def activation(self, activation, output, input, *, topk_ids=None,
                   expert_map=None, valid_token_counts=None):
        from vllm.model_executor.layers.fused_moe.activation import (
            apply_moe_activation,
        )

        apply_moe_activation(activation, output, input,
                             activation_config=self.activation_config,
                             topk_ids=topk_ids, expert_map=expert_map)

    def moe_sum(self, input, output, topk_ids=None, expert_map=None):
        import vllm._custom_ops as ops

        ops.moe_sum(input, output)


def _setup():
    if _STATE:
        return _STATE
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("candidate is sm_80-only by design")
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.scalar_type import scalar_types

    torch.manual_seed(0)
    dev = torch.device("cuda")
    qt = scalar_types.uint4b8
    q1, s1, r1, q2, s2, r2 = [], [], [], [], [], []
    for _ in range(NB):
        w = (torch.randn(K, 2 * N, device=dev) * 0.02).to(torch.bfloat16)
        ref, q, s = marlin_quantize(w, qt, G)
        q1.append(q); s1.append(s); r1.append(ref.float())
        w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        ref, q, s = marlin_quantize(w, qt, G)
        q2.append(q); s2.append(s); r2.append(ref.float())
    idx = torch.arange(E) % NB
    st = lambda xs: torch.stack(xs)[idx.to(dev)].contiguous()  # noqa: E731
    w1, w1s, w2, w2s = st(q1), st(s1), st(q2), st(s2)
    assert w1.shape == (E, K // 16, 2 * N * 2) and w2.shape == (E, N // 16, K * 2)
    assert w1s.shape == (E, K // G, 2 * N) and w2s.shape == (E, N // G, K)
    _STATE.update(dev=dev, w1=w1, w2=w2, layer=FakeLayer(w1s, w2s),
                  r1=torch.stack(r1), r2=torch.stack(r2))
    return _STATE


def _inputs(M, skew=True, seed=1, ids_dtype=torch.int32):
    st = _setup()
    g = torch.Generator(device=st["dev"]).manual_seed(seed)
    x = torch.randn(M, K, device=st["dev"], generator=g).to(torch.bfloat16)
    logits = torch.randn(M, E, device=st["dev"], generator=g)
    if skew:  # uneven expert load, like a real router
        logits += torch.linspace(0, 2.5, E, device=st["dev"])[
            torch.randperm(E, device=st["dev"], generator=g)]
    w, ids = torch.topk(logits.softmax(-1), TOPK, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).float().contiguous()
    return x, w, ids.to(ids_dtype).contiguous()


def _ws(M):
    dev = _STATE["dev"]
    ws13 = torch.empty(M * TOPK, max(N, K), device=dev, dtype=torch.bfloat16)
    ws2 = torch.empty(M * TOPK * max(2 * N, K), device=dev, dtype=torch.bfloat16)
    return ws13, ws2


def run_incumbent(x, w, ids):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )

    st, L = _STATE, _STATE["layer"]
    out = torch.empty_like(x)
    ws13, ws2 = _ws(x.size(0))
    fused_marlin_moe(
        hidden_states=x, w1=st["w1"], w2=st["w2"], bias1=None, bias2=None,
        w1_scale=L.w1_scale, w2_scale=L.w2_scale, topk_weights=w, topk_ids=ids,
        quant_type_id=L.quant_type_id, global_num_experts=E,
        activation=MoEActivation.SILU, activation_func=L.activation,
        activation_config=L.activation_config, moe_sum=L.moe_sum,
        output=out, intermediate_cache13=ws2, intermediate_cache2=ws13,
        workspace=L.marlin_workspace(x.device))
    return out


def run_candidate(x, w, ids, out=None, ws=None):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.models.glm5next.nvidia.ops import marlin_prefill_split as cand

    st = _STATE
    out = torch.empty_like(x) if out is None else out
    ws13, ws2 = _ws(x.size(0)) if ws is None else ws
    ok = cand.maybe_apply(st["layer"], out, x, st["w1"], st["w2"], w, ids,
                          MoEActivation.SILU, E, None, False, ws13, ws2)
    return ok, out


def reference(x, w, ids):
    """fp32 per (token, slot): silu(clamp(g)) * clamp(u), weighted, summed."""
    st = _STATE
    M = x.size(0)
    xf = x.float()
    out = torch.zeros(M, K, device=x.device)
    base = (ids.long() % NB)
    for b in range(NB):
        t, s = (base == b).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        h = xf[t] @ st["r1"][b]
        gt = h[:, :N].clamp(max=CLAMP)
        up = h[:, N:].clamp(min=-CLAMP, max=CLAMP)
        y = (gt * torch.sigmoid(gt) * up) @ st["r2"][b]
        out.index_add_(0, t, y * w[t, s, None])
    return out


def _ulps(a, b):
    ai = a.view(torch.int16).int()
    bi = b.view(torch.int16).int()
    ai = torch.where(ai < 0, -32768 - ai, ai)
    bi = torch.where(bi < 0, -32768 - bi, bi)
    return (ai - bi).abs()


def compare(M=2304, skew=True, ids_dtype=torch.int32, verbose=True):
    torch.backends.cuda.matmul.allow_tf32 = False
    x, w, ids = _inputs(M, skew, ids_dtype=ids_dtype)
    ref_i = run_incumbent(x, w, ids)
    ok, ref_c = run_candidate(x, w, ids)
    assert ok, "gate closed at the validated shape (see log)"
    ok2, ref_c2 = run_candidate(x, w, ids)
    torch.cuda.synchronize()
    ref = reference(x, w, ids)
    eq = (ref_i.view(torch.int16) == ref_c.view(torch.int16)).float().mean().item()
    ul = _ulps(ref_i, ref_c)
    ei = (ref_i.float() - ref).abs()
    ec = (ref_c.float() - ref).abs()
    r = dict(M=M, skew=skew, ids=str(ids_dtype), bitwise_equal_frac=eq,
             max_ulp=int(ul.max()), p999_ulp=float(ul.float().quantile(0.999))
             if ul.numel() < 2**24 else None,
             max_abs_diff=(ref_i.float() - ref_c.float()).abs().max().item(),
             inc_max=ei.max().item(), cand_max=ec.max().item(),
             inc_rms=ei.pow(2).mean().sqrt().item(),
             cand_rms=ec.pow(2).mean().sqrt().item(),
             ref_rms=ref.pow(2).mean().sqrt().item(),
             cand_deterministic=bool(torch.equal(ref_c, ref_c2)))
    if verbose:
        print(r)
    return r


@pytest.mark.parametrize("M,skew,ids_dtype", [
    (2304, True, torch.int32), (2304, False, torch.int32),
    (2304, True, torch.int64), (384, True, torch.int32),
    (1000, True, torch.int32)])
def test_accuracy(M, skew, ids_dtype):
    r = compare(M, skew, ids_dtype)
    assert r["cand_deterministic"]
    assert r["cand_max"] <= 1.10 * r["inc_max"] + 1e-6
    assert r["cand_rms"] <= 1.10 * r["inc_rms"] + 1e-9


@pytest.mark.parametrize("M", [8, 4, 383])
def test_gate_closed_small_m(M):
    x, w, ids = _inputs(M)
    sentinel = torch.full_like(x, 7.0)
    ok, out = run_candidate(x, w, ids, out=sentinel.clone())
    assert not ok and torch.equal(out, sentinel)


def test_cuda_graph_replay():
    from vllm.models.glm5next.nvidia.ops import marlin_prefill_split as cand

    M = 2304
    x, w, ids = _inputs(M, seed=3)
    cand.warmup(x.device, E, M, TOPK)  # as kernel_warmup does
    ws = _ws(M)
    out = torch.empty_like(x)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            run_candidate(x, w, ids, out=out, ws=ws)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ok, _ = run_candidate(x, w, ids, out=out, ws=ws)
    assert ok, "fell back during capture: buffers not warmed"
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    _, eager = run_candidate(x, w, ids)
    assert torch.equal(out, eager)


def _time(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def bench():
    from vllm.models.glm5next.nvidia.ops import moe_split_align
    from vllm.models.glm5next.nvidia.ops import marlin_prefill_split as cand

    _setup()
    print("M  skew  incumbent_ms(med/min)  candidate_ms(med/min)  speedup  "
          "split_align_ms  padded_rows(bs64 -> split)")
    for M in (384, 1024, 2304):
        for skew in (False, True):
            x, w, ids = _inputs(M, skew)
            ws = _ws(M)
            out = torch.empty_like(x)
            ti = _time(lambda: run_incumbent(x, w, ids))
            tc = _time(lambda: run_candidate(x, w, ids, out=out, ws=ws))
            buf = cand._buffers(x.device, E, M * TOPK, create=True)
            ta = _time(lambda: moe_split_align.split_align(ids, E, buf))
            lists = moe_split_align.split_align(ids, E, buf)
            torch.cuda.synchronize()
            split_rows = sum(int(nt.item()) for _, _, _, nt in lists)
            cnt = torch.bincount(ids.flatten().long(), minlength=E)
            bs = next((b for b in (8, 16, 32, 48, 64)
                       if M * TOPK / E / b < 0.9), 64)
            pad = int(((cnt + bs - 1) // bs * bs).sum())
            print(f"{M:5d} {int(skew)}  {ti[0]:.3f}/{ti[1]:.3f}  "
                  f"{tc[0]:.3f}/{tc[1]:.3f}  {ti[0] / tc[0]:.3f}x  {ta[0]:.3f}  "
                  f"{pad} (bs{bs}) -> {split_rows} / {M * TOPK}")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        for M in (2304, 384):
            for skew in (True, False):
                compare(M, skew)
