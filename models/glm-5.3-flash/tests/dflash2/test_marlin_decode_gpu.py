# SPDX-License-Identifier: Apache-2.0
"""GPU test for patch 0006 (compiled sm_80 Marlin MoE decode).

Needs: an sm_80 GPU, torch + vLLM with patches 0001-0006 overlaid, and the
library built by ``ampere_marlin/build.py``:

    VLLM_GLM5_MARLIN_DECODE_LIB=/work/out/_ampere_marlin_C.abi3.so \
        python -m pytest -q -s tests/test_marlin_decode_gpu.py

Real GLM-5.3-Flash routed-expert shapes (E=288, top-8, K=4096, N=2048,
INT4 uint4b8 group 128, bf16), repacked by the same vLLM helpers our
int_wna16 oracle uses. For M in {4, 8} (k=3 and k=7 at one request; extra M
via MARLIN_TEST_TOKENS=1,2,...) and both variants:

* compiled vs fused_marlin_moe: relative L2 <= 1e-2 and max |diff| <= 2 % of
  max |ref| (a layout bug gives ~1);
* both vs an fp32 dense reference from the raw GPTQ tensors: the compiled
  error is within 1.25x Marlin's own (+1e-3);
* bitwise repeatable; ``exact`` variant also reported vs Marlin bitwise;
* CUDA graph: capture of a 3-call sequence (counters must cycle), replay
  with new inputs equals eager bitwise, repeated replays stable;
* gate: an M outside the regime returns False and leaves ``output`` intact;
* TP4 shard shape N=512 (smaller E for memory) as a smoke test.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an sm_80 GPU")

TOKENS = [int(t) for t in os.environ.get("MARLIN_TEST_TOKENS", "4,8").split(",")]
VARIANTS = ["orig", "exact"]


@pytest.fixture(scope="module")
def mdec():
    from vllm.models.glm5next.nvidia.ops import marlin_decode

    marlin_decode.require_extension()  # fails loudly with build instructions
    return marlin_decode


@pytest.fixture(scope="module")
def layer(mdec):
    from marlin_decode_common import build_layer

    torch.cuda.empty_cache()
    return build_layer()


def test_build_info(mdec):
    ops = mdec.require_extension()
    assert mdec.require_extension() is ops  # cached, registered once
    for name in ("decode_gemm", "decode_act", "decode_gemm_orig", "decode_act_orig"):
        assert hasattr(ops, name)
    print("\nruntime build info:", mdec._runtime_build_info())


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("M", TOKENS)
def test_matches_marlin(mdec, layer, M, variant):
    from marlin_decode_common import (
        rel_l2,
        routing,
        run_compiled,
        run_dense_fp32,
        run_marlin,
    )

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    x0, ids0, w0 = routing(M, layer, 0)
    why = mdec.structural_reason(layer.stub, x0, layer.w1, layer.w2, w0, ids0,
                                 MoEActivation.SILU, -1, None, False)
    assert why is None, why
    worst = 0.0
    for seed in range(3):
        x, ids, w = routing(M, layer, 100 * M + seed)
        ref = run_marlin(layer, x, ids, w)
        out = run_compiled(layer, x, ids, w, variant)
        out2 = run_compiled(layer, x, ids, w, variant)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        assert torch.equal(out, out2), "compiled path is not bitwise repeatable"
        d = rel_l2(out, ref)
        mx = (out.float() - ref.float()).abs().max().item()
        assert d <= 1e-2, f"rel L2 vs Marlin {d:.3e}"
        assert mx <= 0.02 * ref.float().abs().max().item(), f"max |diff| {mx:.3e}"
        dense = run_dense_fp32(layer, x, ids, w)
        e_c, e_m = rel_l2(out, dense), rel_l2(ref, dense)
        assert e_c <= 1.25 * e_m + 1e-3, f"compiled {e_c:.3e} vs Marlin {e_m:.3e}"
        worst = max(worst, d)
        print(f"\nM={M} {variant} seed={seed}: vs Marlin relL2={d:.2e} "
              f"max={mx:.2e} bitwise={torch.equal(out, ref)}; vs fp32 dense: "
              f"compiled {e_c:.2e}, Marlin {e_m:.2e}")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("M", TOKENS)
def test_cuda_graph(mdec, layer, M, variant):
    from marlin_decode_common import routing, run_compiled

    mdec._counters(torch.device("cuda", torch.cuda.current_device()))  # warm
    xs, idss, ws = zip(*[routing(M, layer, 7000 + i) for i in range(3)])
    sx, sid, sw = xs[0].clone(), idss[0].clone(), ws[0].clone()
    outs = [torch.empty_like(sx) for _ in range(3)]
    # eager warm-up on the capture inputs (first launch sets kernel attributes)
    run_compiled(layer, sx, sid, sw, variant, out=outs[0])
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for o in outs:  # three MoE "layers" in one graph: counters must cycle
            run_compiled(layer, sx, sid, sw, variant, out=o)
    for i in range(3):
        sx.copy_(xs[i]); sid.copy_(idss[i]); sw.copy_(ws[i])
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            eager = run_compiled(layer, xs[i], idss[i], ws[i], variant)
            torch.cuda.synchronize()
            for o in outs:
                assert torch.equal(o, eager), f"graph replay != eager (input {i})"


def test_gate_outside_regime(mdec, layer):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from marlin_decode_common import routing

    M = next(m for m in range(1, 33) if m not in mdec.regime_tokens())
    x, ids, w = routing(M, layer, 1)
    out = torch.full_like(x, 3.0)
    took = mdec.maybe_apply(layer.stub, out, x, layer.w1, layer.w2, w, ids,
                            MoEActivation.SILU, -1, None, False)
    assert took is False and bool((out == 3.0).all())
    M = min(mdec.regime_tokens())
    x, ids, w = routing(M, layer, 2)
    assert mdec.maybe_apply(layer.stub, torch.empty_like(x), x, layer.w1, layer.w2, w, ids,
                            MoEActivation.SILU, -1, None, False)


def test_tp4_shape_smoke(mdec):
    from marlin_decode_common import build_layer, rel_l2, routing, run_compiled, run_marlin

    small = build_layer(E=64, N=512, seed=5)
    for M in TOKENS:
        x, ids, w = routing(M, small, 9)
        d = rel_l2(run_compiled(small, x, ids, w), run_marlin(small, x, ids, w))
        assert d <= 1e-2, d
    del small
    torch.cuda.empty_cache()
