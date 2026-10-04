#!/usr/bin/env python3
"""Patch 0005 (VLLM_GLM5_THIN_GEMM): torch-free checks. Runs anywhere.

    python3 tests/test_thin_gemm_pure.py        (or pytest)

What it checks, without torch or a GPU:

1. Schedule table / selector (extracted by AST from the patched
   thin_gemm.py and executed with a stand-in `triton.cdiv` and 70 SMs):
   for every GLM-5.3-Flash PP4 / TP4 / DFlash2 weight shape and every
   M in 1..32,
     - BLOCK_M covers M (one M tile; the kernel masks rows against BLOCK_M),
     - the pipelined smem fits or num_stages was trimmed to 2,
     - block sizes are powers of two, SPLIT_K >= 1;
2. The kernel's index arithmetic, re-played in Python from the same
   constexpr values: every k in [0, K) is read by exactly one (pid_k,
   iteration) pair and, when EVEN_K skips the k mask, no read is out of
   range; N tiles cover N and EVEN_N stores are in range.
3. The split-K "last arriver reduces" protocol, simulated for every arrival
   order of up to 5 splits: the output is the fixed-order sum 0..S-1
   (bitwise independent of the arrival order) and the counter is left at 0.
4. The row gate: M <= bound, kda_o_proj (4096, 8192) to cuBLAS from M=17.
5. Default-off invariants by AST/text on the patched files: env defaults
   "0"/"32"; the dispatch hook is the first thing guarded by
   `envs.VLLM_GLM5_THIN_GEMM` and imports lazily; the kernel_warmup hook and
   the kpool-gate site are guarded the same way, and the off branch of the
   kpool gate is the original `F.linear` line.
"""

import os
import ast
import itertools
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
# GLM_DFLASH2_TREE: a directory holding the patched vllm/ package
# (source tree after scripts/apply-glm53-dflash2-patches.sh, or site-packages).
PATCHED = pathlib.Path(os.environ.get("GLM_DFLASH2_TREE", "/usr/local/lib/python3.12/dist-packages")) / "vllm"
TG = PATCHED / "models/glm5next/nvidia/ops/thin_gemm.py"

# (N, K) of the BF16 GEMMs GLM-5.3-Flash W4A16 + DFlash2 runs at PP4 (full
# width) and TP4 (per rank). [inference from config shapes + MM's table]
SHAPES = sorted({
    # PP4 target
    (24896, 4096), (4096, 8192), (8192, 128), (24576, 4096), (4096, 12288),
    (2112, 4096), (16384, 1536), (32768, 512), (4096, 16384), (160, 4096),
    (128, 4096), (4096, 4096), (4096, 2048), (154880, 4096),
    # DFlash2 drafter, replicated
    (4096, 20480), (6144, 4096), (10240, 4096),
    # TP4 rows of MM's table
    (6416, 4096), (6144, 4096), (4096, 3072), (2048, 4096), (4096, 1536),
    (1024, 4096), (8192, 512), (4096, 512), (288, 4096), (38720, 4096),
    (1536, 4096), (4096, 1024), (2048, 128),
})


def _load():
    tree = ast.parse(TG.read_text())
    keep_fn = {"_pow2_at_least", "_heuristic", "_select_config", "rows_supported",
               "_nearest_pinned_same_block_m"}
    keep_var = {"_MIN_K", "_CUBLAS_FROM_M", "_CTA_PER_SM_TARGET",
                "_MIN_K_PER_SPLIT", "_MAX_SMEM", "_CONFIG_OVERRIDES"}
    body = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in keep_fn:
            n.decorator_list = []
            body.append(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tgt = n.targets[0] if isinstance(n, ast.Assign) else n.target
            if isinstance(tgt, ast.Name) and tgt.id in keep_var:
                body.append(n)
    ns = {
        "triton": type("T", (), {"cdiv": staticmethod(lambda a, b: -(-a // b))}),
        "num_sms": lambda: 70,
        "dispatch_threshold": lambda: 32,
    }
    exec(compile(ast.Module(body, []), str(TG), "exec"), ns)
    return ns


NS = _load()


def _pow2(v):
    return v > 0 and v & (v - 1) == 0


def test_table_and_selector_invariants():
    sel = NS["_select_config"]
    for (N, K), M in itertools.product(SHAPES, range(1, 33)):
        BM, BN, BK, SK, nw, ns = sel(M, N, K)
        assert BM >= M and _pow2(BM) and BM >= 16, (N, K, M, BM)
        assert _pow2(BN) and _pow2(BK) and SK >= 1 and nw in (1, 2, 4, 8)
        assert ns == 2 or ns * (BN + BM) * BK * 2 <= NS["_MAX_SMEM"], (N, K, M)
    # every pinned row is reachable and well formed
    for (N, K, M), cfg in NS["_CONFIG_OVERRIDES"].items():
        assert len(cfg) == 5 and all(isinstance(v, int) for v in cfg), (N, K, M)


def test_pp4_unpinned_m_reuses_same_block_m_row():
    """[ours] k=3 / k=7 verify sizes on PP4 full-width shapes use a measured
    row with the same BLOCK_M, never the heuristic MM's sweep beat."""
    sel, tab = NS["_select_config"], NS["_CONFIG_OVERRIDES"]
    for N, K in ((24896, 4096), (24576, 4096), (154880, 4096), (32768, 512)):
        for M in (2, 3, 4, 5, 6, 7, 8):
            cfg = sel(M, N, K)[1:]
            pinned16 = [c for (n, k, m), c in tab.items()
                        if (n, k) == (N, K) and m is not None and m <= 16]
            assert cfg[:4] in [c[:4] for c in pinned16], (N, K, M, cfg)
    # exact row wins; any-M row still wins over the nearest-row rule
    assert sel(8, 24896, 4096)[1:] == tab[(24896, 4096, 8)]
    assert sel(3, 6416, 4096)[1:] == tab[(6416, 4096, None)]
    # lm_head at M=8 reuses the M=1 row (747 us vs cuBLAS 948 us in MM's sweep)
    assert sel(8, 154880, 4096)[1:] == tab[(154880, 4096, 1)]


def _k_reads(K, BK, SK, even_k):
    """(pid_k, iteration) -> k indices the kernel loads (mirrors x_ptrs/w_ptrs)."""
    step = BK * SK
    n_iter = -(-K // step)
    seen = []
    for pid_k in range(SK):
        for i in range(n_iter):
            ks = [i * step + pid_k * BK + o for o in range(BK)]
            if not even_k:
                ks = [k for k in ks if k < K]
            seen.extend(ks)
    return seen


def test_kernel_index_coverage():
    sel = NS["_select_config"]
    for (N, K), M in itertools.product(SHAPES, (1, 4, 8, 16, 24, 32)):
        BM, BN, BK, SK, _, _ = sel(M, N, K)
        even_k = K % (BK * SK) == 0
        reads = _k_reads(K, BK, SK, even_k)
        assert all(0 <= k < K for k in reads), ("OOB k", N, K, M)
        assert sorted(reads) == list(range(K)), ("k not covered once", N, K, M)
        tiles_n = -(-N // BN)
        cols = [t * BN + o for t in range(tiles_n) for o in range(BN)]
        assert set(c for c in cols if c < N) == set(range(N))
        if N % BN == 0:  # EVEN_N: unmasked store
            assert max(cols) < N


def test_split_k_last_arriver_protocol():
    import random
    rnd = random.Random(0)
    for S in range(2, 6):
        partials = [rnd.uniform(-1, 1) * 10 ** rnd.randint(-3, 3) for _ in range(S)]
        ref = 0.0
        for p in partials:  # fixed order 0..S-1
            ref += p
        for order in itertools.permutations(range(S)):
            lock, out, stored = 0, None, {}
            for pid_k in order:
                stored[pid_k] = partials[pid_k]
                arrived, lock = lock, lock + 1  # atomic_add returns old value
                if arrived == S - 1:
                    tot = 0.0
                    for k in range(S):
                        tot += stored[k]
                    out, lock = tot, 0  # atomic_xchg(lock, 0)
            assert out is not None and out == ref and lock == 0, (S, order)


def test_row_gate():
    rows = NS["rows_supported"]
    assert rows(1, 4096, 4096) and rows(32, 4096, 4096)
    assert not rows(33, 4096, 4096) and not rows(0, 4096, 4096)
    assert rows(16, 4096, 8192) and not rows(17, 4096, 8192)


def _src(rel):
    return (PATCHED / rel).read_text()


def test_default_off_invariants():
    envs = _src("envs.py")
    assert 'os.getenv("VLLM_GLM5_THIN_GEMM", "0")' in envs
    assert 'os.getenv("VLLM_GLM5_THIN_GEMM_MAX_TOKENS", "32")' in envs
    assert "VLLM_GLM5_THIN_GEMM: bool = False" in envs

    utils = ast.parse(_src("model_executor/layers/utils.py"))
    fn = next(n for n in utils.body if isinstance(n, ast.FunctionDef)
              and n.name == "dispatch_unquantized_gemm")
    ifs = [n for n in fn.body if isinstance(n, ast.If)]
    thin = [n for n in ifs if ast.unparse(n.test) == "envs.VLLM_GLM5_THIN_GEMM"]
    assert len(thin) == 1
    # the import of the kernel module happens only inside the guarded branch
    mod = "vllm.models.glm5next.nvidia.ops.thin_gemm"
    assert all(n.module != mod for n in ast.walk(utils)
               if isinstance(n, ast.ImportFrom) and n not in ast.walk(thin[0]))
    # nothing after the platform checks precedes the guard except the guard
    src_fn = ast.unparse(fn)
    assert src_fn.index("envs.VLLM_GLM5_THIN_GEMM") < src_fn.index(
        "_FLASHINFER_BF16_BACKENDS.get")

    warm = _src("model_executor/warmup/kernel_warmup.py")
    assert ("    if envs.VLLM_GLM5_THIN_GEMM:\n        from vllm.models.glm5next."
            "nvidia.ops.thin_gemm import warmup_thin_gemm") in warm

    attn = _src("models/glm5next/nvidia/attention.py")
    assert ("        else:\n            gate_score = F.linear(hidden_states, "
            "self.index_kpool_compress_gate)\n") in attn
    assert "    if not envs.VLLM_GLM5_THIN_GEMM:\n        return False\n" in attn
    # the kernel module is never imported at attention.py module scope
    top = ast.parse(attn).body
    assert not any(isinstance(n, ast.ImportFrom) and n.module == mod for n in top)


if __name__ == "__main__":
    fails = 0
    for name, f in sorted(globals().items()):
        if name.startswith("test_") and callable(f):
            try:
                f()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    print(f"\n{'FAILED: %d' % fails if fails else 'all passed'}")
    sys.exit(1 if fails else 0)
