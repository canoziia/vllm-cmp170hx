"""GPU ONLY (sm_80, CMP 170HX / A100). Not executed by the author session.

Patch 0022: pre-compile every Triton specialization the GLM5Next kpool indexer
prefill reaches, so the first long-context request does not JIT mid-request.
See ../JIT-WARMUP.md.

What it checks, in fresh subprocesses (fresh in-memory JIT caches, and a
fresh empty TRITON_CACHE_DIR so a compile costs what it costs on a cold boot):

  1. ``--warmup``: after the 0022 warmup (logits all-variants warmup + the
     prefill-chunk metadata kernel warmup keys), a simulated single-request
     chunked prefill at contexts 2k, 6k, 48k, 74k, 110k, 200k (+ ragged
     variants, kpool 4, 2304-token chunks, logits budget 128 MiB -> query
     sub-chunking) causes 0 new compilations of ``_fp8_mqa_logits_kernel``
     and of ``BuildPrefillChunkMetadataKernel.kernel``.
  2. ``--no-warmup`` (pre-0022 behaviour) on the same sweep: reports how many
     specializations are compiled mid-sweep and where (expected > 0 on sm_80:
     KV_GROUP=8 at N >= 16384, N % 16 != 0 at ragged ends, metadata kernel at
     the first query sub-chunk). Informational, not asserted.
  3. Every logits / ks / ke output of 1 and 2 is bitwise identical (sha256),
     and also bitwise identical with the autotuner pinned to num_stages=2 and
     to num_stages=4 (the warmup moves the autotune sweep to a different
     shape; this shows the choice cannot change outputs).

Run inside the patched container (0001..0022 applied):

    python -m pytest -q -s tests/test_indexer_jit_warmup_gpu.py
    python tests/test_indexer_jit_warmup_gpu.py --warmup --out /tmp/a.json
    python tests/test_indexer_jit_warmup_gpu.py --no-warmup --out /tmp/b.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

H, D = 64, 128  # logits kernel heads as launched (index_n_heads=64), head dim
KPOOL = 4
CHUNK = 2304
LOGITS_BUDGET_ELEMS = 128 * 1024 * 1024 // 4  # VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
# Contexts at the END of the simulated chunk. +37 variants make N ragged
# (ctx // 4 not a multiple of 16), like the last chunk of a real prompt.
CONTEXTS = [2048, 6000, 48000, 48037, 74000, 74037, 110000, 110037, 200000, 200037]


# --------------------------------------------------------------------------
# compile counting
# --------------------------------------------------------------------------
def _jit_cache_size(jit_fn) -> int:
    """Number of compiled binaries held by a triton JITFunction (all devices)."""
    caches = getattr(jit_fn, "device_caches", None)
    if caches is None:  # very old triton
        return sum(len(v) for v in getattr(jit_fn, "cache", {}).values())
    total = 0
    for entry in caches.values():
        kernel_cache = entry[0] if isinstance(entry, tuple) else entry
        total += len(kernel_cache)
    return total


def _kernels():
    from vllm.v1.attention.backends.mla.indexer import BuildPrefillChunkMetadataKernel
    from vllm.v1.attention.ops import mqa_logits_triton as m

    logits_jit = m._fp8_mqa_logits_kernel.fn  # Autotuner -> JITFunction
    meta_jit = BuildPrefillChunkMetadataKernel.kernel
    meta_jit = getattr(meta_jit, "__func__", meta_jit)
    return {"logits": logits_jit, "chunk_meta": meta_jit}


def _counts():
    return {k: _jit_cache_size(v) for k, v in _kernels().items()}


def _install_compile_log(log: list):
    """Also record compiles through triton.knobs when available."""
    try:
        from triton import knobs
    except ImportError:
        return
    prev = knobs.runtime.jit_post_compile_hook

    def hook(*args, **kwargs):
        fn = kwargs.get("fn")
        name = getattr(getattr(fn, "jit_function", fn), "__name__", None) or str(fn)
        log.append({"name": name, "key": str(kwargs.get("key"))[:200]})
        if prev is not None:
            return prev(*args, **kwargs)
        return None

    knobs.runtime.jit_post_compile_hook = hook


# --------------------------------------------------------------------------
# simulated chunked prefill
# --------------------------------------------------------------------------
def _fake_vllm_config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2312),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(index_kpool=KPOOL)),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
    )


def _sweep(device, record):
    from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata
    from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton

    gen = torch.Generator(device=device).manual_seed(1234)
    for ctx in CONTEXTS:
        T = min(CHUNK, ctx)
        n = ctx // KPOOL  # compressed total_seq_lens of this chunk
        max_q = max(1, LOGITS_BUDGET_ELEMS // n)
        q_all = (torch.randn(T, H, D, device=device, generator=gen) * 0.5).to(
            torch.float8_e4m3fn
        )
        k = (torch.randn(n, D, device=device, generator=gen) * 0.5).to(
            torch.float8_e4m3fn
        )
        k_scale = torch.rand(n, device=device, generator=gen) + 0.01
        w = torch.randn(T, H, device=device, generator=gen)

        qsl_cpu = torch.tensor([0, T], dtype=torch.int32)
        qsl = qsl_cpu.to(device)
        seq = torch.tensor([ctx], dtype=torch.int32, device=device)
        cseq_cpu = torch.tensor([n], dtype=torch.int32)
        cseq = cseq_cpu.to(device)
        block_table = torch.zeros(1, 1, dtype=torch.int32, device=device)

        for q0 in range(0, T, max_q):
            q1 = min(T, q0 + max_q)
            before = _counts()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            meta = build_prefill_chunk_metadata(
                0, 1, qsl, qsl_cpu, seq, cseq, cseq_cpu, block_table, KPOOL,
                query_slice=slice(q0, q1), skip_kv_gather=q0 > 0,
            )
            logits = fp8_mqa_logits_triton(
                q_all[q0:q1], (k, k_scale), w[q0:q1],
                meta.cu_seqlen_ks, meta.cu_seqlen_ke, clean_logits=False,
            )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            after = _counts()
            record.append({
                "ctx": ctx, "N": n, "M": q1 - q0, "q0": q0,
                "new": {kk: after[kk] - before[kk] for kk in after},
                "ms": round(dt * 1e3, 2),
                "sha_logits": hashlib.sha256(
                    logits.contiguous().cpu().numpy().tobytes()).hexdigest(),
                "sha_ks_ke": hashlib.sha256(
                    torch.cat([meta.cu_seqlen_ks, meta.cu_seqlen_ke])
                    .cpu().numpy().tobytes()).hexdigest(),
            })
            del logits, meta


def run(warmup: bool, stages: int | None, out: str | None) -> dict:
    device = torch.device("cuda", torch.cuda.current_device())
    from vllm.v1.attention.ops import mqa_logits_triton as m

    if stages is not None:
        m._fp8_mqa_logits_kernel.configs = [
            c for c in m._fp8_mqa_logits_kernel.configs if c.num_stages == stages
        ]
        assert m._fp8_mqa_logits_kernel.configs, stages
    log: list = []
    _install_compile_log(log)

    t0 = time.perf_counter()
    if warmup:
        from vllm.model_executor.layers.sparse_attn_indexer_kpool import (
            _warmup_prefill_logits_once,
        )
        from vllm.v1.attention.backends.mla.indexer import (
            _BUILD_PREFILL_CHUNK_METADATA_KERNEL,
        )

        _warmup_prefill_logits_once(H, D, device)
        _BUILD_PREFILL_CHUNK_METADATA_KERNEL.warmup(_fake_vllm_config())
        torch.cuda.synchronize()
    warm_s = time.perf_counter() - t0
    after_warm = _counts()
    n_log_warm = len(log)

    record: list = []
    _sweep(device, record)
    res = {
        "warmup": warmup, "stages": stages, "sm80": m._IS_SM80,
        "warmup_s": round(warm_s, 2), "after_warmup": after_warm,
        "final": _counts(),
        "new_in_sweep": {k: sum(r["new"][k] for r in record) for k in after_warm},
        "sweep_compile_log": log[n_log_warm:],
        "calls": record,
    }
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
    return res


# --------------------------------------------------------------------------
# pytest: each mode in its own process with an empty Triton cache
# --------------------------------------------------------------------------
def _subprocess(*flags) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "res.json")
        env = dict(os.environ, TRITON_CACHE_DIR=os.path.join(tmp, "triton"))
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), *flags, "--out", out],
            check=True, env=env,
        )
        with open(out) as f:
            return json.load(f)


def _digest(res):
    return [(c["ctx"], c["q0"], c["sha_logits"], c["sha_ks_ke"]) for c in res["calls"]]


def test_no_new_compiles_after_warmup_and_bitwise():
    fixed = _subprocess("--warmup")
    old = _subprocess("--no-warmup")
    print("\nwarmup:", fixed["warmup_s"], "s, after warmup", fixed["after_warmup"],
          "new in sweep", fixed["new_in_sweep"])
    print("no-warmup: new in sweep", old["new_in_sweep"])
    for c in old["calls"]:
        if any(c["new"].values()):
            print("  pre-0022 mid-sweep compile at ctx=%d N=%d M=%d q0=%d %s %.0f ms"
                  % (c["ctx"], c["N"], c["M"], c["q0"], c["new"], c["ms"]))
    assert fixed["new_in_sweep"] == {"logits": 0, "chunk_meta": 0}, fixed["sweep_compile_log"]
    assert _digest(fixed) == _digest(old)

    for st in (2, 4):
        pinned = _subprocess("--warmup", "--stages", str(st))
        assert pinned["new_in_sweep"] == {"logits": 0, "chunk_meta": 0}
        assert _digest(pinned) == _digest(fixed), f"num_stages={st} differs"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--warmup", action="store_true")
    g.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--stages", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = run(a.warmup, a.stages, a.out)
    print(json.dumps({k: r[k] for k in
                      ("warmup", "stages", "sm80", "warmup_s", "after_warmup",
                       "new_in_sweep")}))
    for c in r["calls"]:
        print(c["ctx"], c["N"], c["M"], c["q0"], c["new"], c["ms"], "ms")
