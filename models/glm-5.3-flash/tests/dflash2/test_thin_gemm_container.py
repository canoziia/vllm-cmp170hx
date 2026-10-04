"""Patch 0005 (VLLM_GLM5_THIN_GEMM): torch tests for the container.

Needs torch + the patched vLLM (patched/vllm overlaid on the installed package,
see tests/README.md). Three tiers, each skipped when unavailable:

  CPU      gates, the off path, the custom op's F.linear fallback.
  INTERP   the real Triton kernel under TRITON_INTERPRET=1 on CPU tensors
           (subprocess; skipped if this Triton's interpreter lacks a feature).
  GPU      numerics against an FP64 reference on the real PP4 shapes, at least
           as accurate as cuBLAS; bitwise run-to-run determinism; CUDA-graph
           capture/replay; the not-warmed capture fallback; two streams.
           The kernel is tuned for sm_80, but correctness holds on any NVIDIA
           GPU Triton supports, so the GPU tier runs on whatever card is there
           (dispatch itself stays sm_80-only and is checked separately).

    python -m pytest -q tests/test_thin_gemm_container.py
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

import vllm.envs as envs
from vllm.models.glm5next.nvidia.ops import thin_gemm as tg

BF16 = torch.bfloat16
HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")

# PP4 full-width shapes of GLM-5.3-Flash W4A16 + DFlash2 (N, K).
PP4_SHAPES = [
    (24896, 4096),  # kda in_proj
    (4096, 8192),  # kda o_proj (cuBLAS from M=17)
    (8192, 128),  # kda f/g b proj
    (16384, 1536),  # mla q_b
    (32768, 512),  # mla kv_b
    (4096, 16384),  # mla o_proj
    (128, 4096),  # kpool compress gate (split-K 8/16)
    (160, 4096),  # indexer wk+weights (split-K 8)
    (4096, 4096),  # shared expert gate_up
    (4096, 2048),  # shared expert down
    (4096, 20480),  # drafter fc (split-K 2)
    (154880, 4096),  # lm_head
]
MS = (1, 2, 4, 8, 16, 24, 32)


@pytest.fixture
def env(monkeypatch):
    def set_(**kv):
        for k, v in kv.items():
            monkeypatch.setenv(k, v)
        tg.dispatch_threshold.cache_clear()
        tg._device_supported.cache_clear()

    yield set_
    tg.dispatch_threshold.cache_clear()
    tg._device_supported.cache_clear()


# ----------------------------------------------------------------------- CPU


def test_env_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_GLM5_THIN_GEMM", raising=False)
    monkeypatch.delenv("VLLM_GLM5_THIN_GEMM_MAX_TOKENS", raising=False)
    assert envs.VLLM_GLM5_THIN_GEMM is False
    assert envs.VLLM_GLM5_THIN_GEMM_MAX_TOKENS == 32
    assert tg.use_thin_gemm() is False


def test_dispatch_off_returns_previous_callable(monkeypatch):
    """Flag unset: the dispatcher returns exactly what it returned before."""
    from vllm.model_executor.layers import utils

    monkeypatch.delenv("VLLM_GLM5_THIN_GEMM", raising=False)
    got = utils.dispatch_unquantized_gemm()
    assert got is not tg.glm5_thin_unquantized_gemm
    if HAS_CUDA:
        assert got is utils.default_unquantized_gemm


def test_static_and_row_gates(env):
    env(VLLM_GLM5_THIN_GEMM_MAX_TOKENS="32")
    x, w = torch.zeros(4, 4096, dtype=BF16), torch.zeros(128, 4096, dtype=BF16)
    assert tg.static_supported(x, w, None)
    assert not tg.static_supported(x, w, torch.zeros(128, dtype=BF16))  # bias
    assert not tg.static_supported(x.float(), w, None)
    assert not tg.static_supported(x, w.t().contiguous().t(), None)  # K-strided
    assert not tg.static_supported(x[:, :64], w[:, :64], None)  # K < 128
    assert not tg.static_supported(x.view(2, 2, 4096), w, None)  # 3-D
    assert tg.rows_supported(32, 4096, 4096) and not tg.rows_supported(33, 4096, 4096)
    assert tg.rows_supported(16, 4096, 8192) and not tg.rows_supported(17, 4096, 8192)
    env(VLLM_GLM5_THIN_GEMM_MAX_TOKENS="8")
    assert not tg.rows_supported(9, 4096, 4096)


def test_custom_op_falls_back_to_f_linear_above_bound(env):
    env(VLLM_GLM5_THIN_GEMM_MAX_TOKENS="32")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 256, generator=g).to(BF16)
    w = torch.randn(64, 256, generator=g).to(BF16)
    # The op is registered for the platform's dispatch key (CUDA), so the CPU
    # tier calls its implementation directly.
    out = tg._glm5_thin_linear_impl(x, w)
    assert torch.equal(out, torch.nn.functional.linear(x, w))
    # the unquantized-gemm drop-in keeps bias semantics through F.linear
    b = torch.randn(64, generator=g).to(BF16)  # bias -> never the custom op
    assert torch.equal(
        tg.glm5_thin_unquantized_gemm(None, x, w, b),
        torch.nn.functional.linear(x, w, b),
    )


def test_fake_impl_shape():
    x, w = torch.empty(3, 512, dtype=BF16), torch.empty(77, 512, dtype=BF16)
    out = tg._glm5_thin_linear_fake(x, w)
    assert out.shape == (3, 77) and out.dtype == BF16


# -------------------------------------------------------------------- INTERP

_INTERP = textwrap.dedent(
    """
    import torch
    from vllm.models.glm5next.nvidia.ops import thin_gemm as tg
    tg.num_sms = lambda: 70
    g = torch.Generator().manual_seed(1)
    cases = [  # (M, N, K, (BLOCK_N, BLOCK_K, SPLIT_K, warps, stages))
        (1, 48, 256, (16, 64, 1, 2, 2)),
        (5, 40, 384, (16, 64, 2, 2, 2)),   # split-K, ragged N, EVEN_K
        (8, 32, 320, (32, 64, 4, 2, 2)),   # split-K, ragged K (masked)
        (17, 64, 512, (32, 128, 2, 2, 2)), # BLOCK_M 32
    ]
    for M, N, K, cfg in cases:
        tg._CONFIG_OVERRIDES[(N, K, M)] = cfg
        x = torch.randn(M, K, generator=g).to(torch.bfloat16)
        w = torch.randn(N, K, generator=g).to(torch.bfloat16)
        y = tg.thin_gemm(x, w)
        y2 = tg.thin_gemm(x, w)
        ref = (x.double() @ w.double().T)
        err = (y.double() - ref).abs().max().item()
        ref_bf = ref.float().to(torch.bfloat16).double()
        tol = (ref_bf - ref).abs().max().item() * 2 + 1e-2
        assert torch.equal(y, y2), "not deterministic"
        assert err <= tol, (M, N, K, err, tol)
        print("interp ok", M, N, K, err)
    """
)


def test_kernel_under_triton_interpreter():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, "-c", _INTERP], env=env, capture_output=True, text=True
    )
    if r.returncode != 0:
        tail = (r.stderr or r.stdout)[-1500:]
        if "AssertionError" in tail:
            pytest.fail(tail)
        pytest.skip("Triton interpreter cannot run this kernel here:\n" + tail)
    assert r.stdout.count("interp ok") == 4, r.stdout


# ----------------------------------------------------------------------- GPU


def _ref(x, w, chunk=8192):
    """FP64 x @ w.T and K * 2^-24 * |x| @ |w|.T, chunked over N (lm_head is
    154880 x 4096: a whole FP64 copy would be 5 GB)."""
    K = x.shape[1]
    xd, xa = x.double(), x.double().abs()
    refs, accs = [], []
    for n0 in range(0, w.shape[0], chunk):
        wd = w[n0 : n0 + chunk].double()
        refs.append(xd @ wd.T)
        accs.append(K * 2.0**-24 * (xa @ wd.abs().T))
    return torch.cat(refs, 1), torch.cat(accs, 1)


@needs_cuda
@pytest.mark.parametrize("N,K", PP4_SHAPES)
def test_gpu_accuracy_vs_fp64_and_cublas(N, K):
    g = torch.Generator(device="cuda").manual_seed(N * 7 + K)
    w = torch.randn(N, K, device="cuda", generator=g, dtype=BF16).mul_(K**-0.5)
    for M in MS:
        x = torch.randn(M, K, device="cuda", generator=g, dtype=BF16)
        ref, acc = _ref(x, w)
        thin = tg.thin_gemm(x, w)
        e_t = (thin.double() - ref).abs()
        e_c = (torch.nn.functional.linear(x, w).double() - ref).abs()
        # Same contract as MM's gate: as accurate as cuBLAS (both round one
        # FP32 accumulator to BF16), with slack for the summation order; and
        # within one BF16 ulp plus the worst-case FP32 summation error
        # (K * 2^-24 * sum|x*w|) of the exact result, element by element.
        _, exp = torch.frexp(ref.abs().clamp_min(1e-30))
        ulp = torch.ldexp(torch.ones_like(ref), (exp - 8).to(torch.int32))
        assert (e_t <= ulp + acc).all(), (N, K, M)
        assert e_t.mean() <= e_c.mean() * 1.10 + 1e-7, (N, K, M)
        assert torch.equal(thin, tg.thin_gemm(x, w)), ("not deterministic", N, K, M)


@needs_cuda
def test_gpu_cuda_graph_capture_and_replay(env):
    env(VLLM_GLM5_THIN_GEMM_MAX_TOKENS="32")
    N, K, M = 128, 4096, 8  # split-K shape: exercises partials + counters
    w = torch.randn(N, K, device="cuda").to(BF16)
    x = torch.randn(M, K, device="cuda").to(BF16)
    torch.ops.vllm.glm5_thin_linear(x, w)  # warm: compile + workspace
    assert (x.device.index, M, N, K) in tg._READY
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        y = torch.ops.vllm.glm5_thin_linear(x, w)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        x.copy_(torch.randn(M, K, device="cuda").to(BF16))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, tg.thin_gemm(x, w))


@needs_cuda
def test_gpu_capture_without_warmup_falls_back(env):
    env(VLLM_GLM5_THIN_GEMM_MAX_TOKENS="32")
    N, K, M = 96, 2048, 3  # a shape nothing else in this file uses
    w = torch.randn(N, K, device="cuda").to(BF16)
    x = torch.randn(M, K, device="cuda").to(BF16)
    assert (x.device.index, M, N, K) not in tg._READY
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        y = torch.ops.vllm.glm5_thin_linear(x, w)
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, torch.nn.functional.linear(x, w))
    assert (x.device.index, M, N, K) not in tg._READY


@needs_cuda
def test_gpu_two_streams_different_shapes():
    """Shared-expert-on-aux-stream pattern: concurrent split-K GEMMs of
    different shapes must not share partials or counters."""
    a = (torch.randn(8, 4096, device="cuda").to(BF16),
         torch.randn(128, 4096, device="cuda").to(BF16))
    b = (torch.randn(8, 4096, device="cuda").to(BF16),
         torch.randn(160, 4096, device="cuda").to(BF16))
    ra, rb = tg.thin_gemm(*a), tg.thin_gemm(*b)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    for _ in range(50):
        with torch.cuda.stream(s1):
            ya = tg.thin_gemm(*a)
        with torch.cuda.stream(s2):
            yb = tg.thin_gemm(*b)
        torch.cuda.synchronize()
        assert torch.equal(ya, ra) and torch.equal(yb, rb)


@needs_cuda
def test_gpu_dispatch_on(env):
    from vllm.model_executor.layers import utils
    from vllm.platforms import current_platform

    env(VLLM_GLM5_THIN_GEMM="1")
    got = utils.dispatch_unquantized_gemm()
    if current_platform.is_device_capability(80):
        assert got is tg.glm5_thin_unquantized_gemm
    else:
        assert got is utils.default_unquantized_gemm
