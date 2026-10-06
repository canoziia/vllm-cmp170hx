#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 decode small-kernel fusions (/tmp/ds-work/C diff): bitwise
tests + micro-benchmarks. Run inside the image on ONE sm80 GPU:

    python3 test_dsv41_small_kernels_gpu.py [--vllm-root DIR] [--bench]

--vllm-root: directory that CONTAINS a patched ``vllm`` package (prepended to
sys.path). Omit it when the diff is applied to the installed package.

Every fusion is integer-only or a pure re-routing of the same stores, so all
checks are exact (torch.equal on raw bits); no tolerance is used anywhere.

Switches under test (all default off; off == deployed code path):
  VLLM_DSV41_TOPK_RAGGED_FUSED   1 launch for topk lens+scan+pack (was 5)
  VLLM_DSV41_TOPK_RAGGED_REUSE   reuse that metadata across layers per step
  VLLM_DSV4_ATTN_DIRECT_OUT      decode writes layer output directly (no DtoD
                                 copy) + no dummy indptr fill on SWA-only layers
  VLLM_DSV41_SWA_RAGGED_INPLACE  SWA builder writes ragged into graph buffers
  VLLM_DSV41_INDEXER_Q_LUT_FUSED 1 gather launch for indexer q LUT (was 2)
"""
import argparse
import os
import re
import sys
import tempfile
import time

ap = argparse.ArgumentParser()
ap.add_argument("--vllm-root", default=None)
ap.add_argument("--bench", action="store_true")
ap.add_argument("--lib", default=None, help="unused (no native lib); accepted for uniformity")
args = ap.parse_args()
if args.vllm_root:
    sys.path.insert(0, args.vllm_root)

for k in ("VLLM_DSV41_TOPK_RAGGED_FUSED", "VLLM_DSV41_TOPK_RAGGED_REUSE",
          "VLLM_DSV4_ATTN_DIRECT_OUT", "VLLM_DSV41_SWA_RAGGED_INPLACE",
          "VLLM_DSV41_INDEXER_Q_LUT_FUSED"):
    os.environ.pop(k, None)

import torch  # noqa: E402

import vllm.model_executor.layers.fused_moe  # noqa: E402,F401  (import order)
import vllm.v1.attention.ops.rocm_aiter_mla_sparse as SP  # noqa: E402
import vllm.v1.attention.ops.mqa_logits_triton as ML  # noqa: E402
import vllm.models.deepseek_v4_1.amd.rocm as R  # noqa: E402
import vllm.models.deepseek_v4_1.attention as A  # noqa: E402

print("vllm from:", os.path.dirname(SP.__file__))
assert hasattr(R, "fused_global_topk_ragged_indices_and_indptr"), "diff not applied"
dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL:", msg)


def set_flag(name, on, cached_fn=None):
    if on:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)
    if cached_fn is not None:
        cached_fn.cache_clear()


def graph_time_us(fn, inner=20, outer=20):
    for _ in range(3):
        fn()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(outer):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (inner * outer) * 1e6


def eager_time_us(fn, iters=200):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def graph_nodes(fn):
    """Count kernel / memcpy / memset nodes of one call via the graph dump."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        g.enable_debug_mode()
    except Exception:
        return None
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        fn()
    with tempfile.NamedTemporaryFile(suffix=".dot") as f:
        try:
            g.debug_dump(f.name)
        except Exception:
            return None
        txt = open(f.name).read()
    # node labels look like 'KERNEL', 'MEMCPY', 'MEMSET' in cudaGraphDebugDotPrint
    return {k: len(re.findall(rf"\b{k}\b", txt)) for k in ("KERNEL", "MEMCPY", "MEMSET")}


# ---------------------------------------------------------------------------
# 1. topk ragged: fused single kernel vs deployed 5-launch chain (bitwise)
# ---------------------------------------------------------------------------
print("\n[1] topk ragged lens+indptr+pack: fused vs deployed")


def make_topk_case(m, topk, block_size, pattern, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    nreq = max(1, (m + 5) // 6)
    width = 4096 // block_size + 64
    block_table = torch.randperm(nreq * width * 4, generator=g, device=dev)[
        : nreq * width].to(torch.int32).view(nreq, width)
    tok2req = (torch.arange(m, device=dev) // 6).clamp(max=nreq - 1).to(torch.int32)
    n_local = width * block_size
    idx = torch.randint(0, n_local, (m, topk), generator=g, device=dev, dtype=torch.int32)
    if pattern == "prefix":       # indexer output: valid prefix, -1 tail
        lens = torch.randint(0, topk + 1, (m,), generator=g, device=dev)
        lens[0] = topk
        col = torch.arange(topk, device=dev)
        idx[col[None, :] >= lens[:, None]] = -1
    elif pattern == "scatter":    # arbitrary -1 holes
        idx[torch.rand(m, topk, generator=g, device=dev) < 0.3] = -1
    elif pattern == "allneg":
        idx.fill_(-1)
    valid = torch.ones(m, dtype=torch.bool, device=dev)
    if m > 2:
        valid[-2:] = False        # cudagraph padding rows
    # buffer wider in rows than m, as in production (slice of the max buffer)
    buf = torch.full((m + 7, topk), -1, dtype=torch.int32, device=dev)
    buf[:m] = idx
    return buf[:m], tok2req, block_table, block_size, valid


def cmp_topk(ref, out, tag):
    r_rag, r_ptr, r_len = ref
    o_rag, o_ptr, o_len = out
    ok = torch.equal(r_ptr, o_ptr) and torch.equal(r_len, o_len)
    nnz = int(r_ptr[-1].item())
    ok = ok and torch.equal(r_rag[:nnz], o_rag[:nnz])
    ok = ok and r_rag.shape == o_rag.shape and r_rag.dtype == o_rag.dtype
    check(ok, f"topk ragged mismatch {tag}")
    return ok


for topk in (512, 1024):
    for block_size in (128, 64):
        for m in (1, 6, 12, 48, 64, 200, 256, 300):
            for pattern in ("prefix", "scatter", "allneg"):
                case = make_topk_case(m, topk, block_size, pattern, m * 7 + topk)
                ref = R.compute_global_topk_ragged_indices_and_indptr(*case)
                out = R.fused_global_topk_ragged_indices_and_indptr(*case)
                cmp_topk(ref, out, f"m={m} topk={topk} bs={block_size} {pattern}")
print("  done ({} failures so far)".format(len(FAILS)))

# 1b. reuse semantics (epoch)
print("[1b] cross-layer reuse + epoch invalidation")
set_flag("VLLM_DSV41_TOPK_RAGGED_FUSED", True, R._topk_ragged_fused_enabled)
set_flag("VLLM_DSV41_TOPK_RAGGED_REUSE", True, R._topk_ragged_reuse_enabled)
case = make_topk_case(48, 512, 128, "prefix", 1)
A.bump_topk_epoch()
r1 = R.decode_global_topk_ragged(*case)
r2 = R.decode_global_topk_ragged(*case)
check(r1[0] is r2[0], "reuse: second consumer in the same epoch did not hit")
case2 = make_topk_case(48, 512, 64, "prefix", 1)  # other block size -> other key
r3 = R.decode_global_topk_ragged(*case2)
check(r3[0] is not r1[0], "reuse: different block_size must not share")
case[0].copy_(make_topk_case(48, 512, 128, "scatter", 2)[0])  # indexer rewrite
A.bump_topk_epoch()
r4 = R.decode_global_topk_ragged(*case)
check(r4[0] is not r1[0], "reuse: epoch bump did not invalidate")
cmp_topk(R.compute_global_topk_ragged_indices_and_indptr(*case), r4, "after bump")
set_flag("VLLM_DSV41_TOPK_RAGGED_REUSE", False, R._topk_ragged_reuse_enabled)
set_flag("VLLM_DSV41_TOPK_RAGGED_FUSED", False, R._topk_ragged_fused_enabled)
r5 = R.decode_global_topk_ragged(*case)
cmp_topk(R.compute_global_topk_ragged_indices_and_indptr(*case), r5, "switch off")

if args.bench:
    print("  bench (graph-replay us/call; 'x4' = one index source + 3 consumers):")
    for m in (6, 48):
        case = make_topk_case(m, 512, 128, "prefix", 3)
        t_ref = graph_time_us(lambda: R.compute_global_topk_ragged_indices_and_indptr(*case))
        t_fus = graph_time_us(lambda: R.fused_global_topk_ragged_indices_and_indptr(*case))
        e_ref = eager_time_us(lambda: R.compute_global_topk_ragged_indices_and_indptr(*case))
        e_fus = eager_time_us(lambda: R.fused_global_topk_ragged_indices_and_indptr(*case))
        n_ref = graph_nodes(lambda: R.compute_global_topk_ragged_indices_and_indptr(*case))
        n_fus = graph_nodes(lambda: R.fused_global_topk_ragged_indices_and_indptr(*case))
        print(f"   M={m:2d}: graph deployed {t_ref:6.2f} fused {t_fus:6.2f} | "
              f"eager(host+gpu) deployed {e_ref:6.1f} fused {e_fus:6.1f} | "
              f"x4 eager deployed {4 * e_ref:6.1f} fused+reuse {e_fus:6.1f} | "
              f"nodes {n_ref} -> {n_fus}")

# ---------------------------------------------------------------------------
# 2. sparse attention decode: direct output (no DtoD) + SWA-only dummy indptr
# ---------------------------------------------------------------------------
print("\n[2] rocm_sparse_attn_decode direct-out: flag on vs off (bitwise)")
H, NOPE, ROPE, BS = 64, 448, 64, 64


def make_cache(nblocks, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    c = torch.randint(0, 256, (nblocks, BS, 584), dtype=torch.uint8, device=dev, generator=g)
    flat = c.view(nblocks, -1)
    data = flat[:, : BS * 576].view(nblocks, BS, 576)
    rope = (torch.randn(nblocks, BS, ROPE, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    data[:, :, NOPE:].copy_(rope.view(torch.uint8).view(nblocks, BS, ROPE * 2))
    nope = data[:, :, :NOPE]
    nope[(nope & 0x7F) == 0x7F] = 0x3C
    sc = flat[:, BS * 576:].view(nblocks, BS, 8)
    sc.copy_(torch.randint(110, 135, sc.shape, dtype=torch.uint8, device=dev, generator=g))
    return c


def attn_case(nq, main_len, extra_len, swa_only, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    main = make_cache(64, seed)
    extra = None if swa_only else make_cache(256, seed + 1)
    q = (torch.randn(nq, H, NOPE + ROPE, device=dev, generator=g) * 0.3).to(torch.bfloat16)
    mi = torch.randint(0, 64 * BS, (nq * main_len,), dtype=torch.int32, device=dev, generator=g)
    mp = torch.arange(0, nq + 1, dtype=torch.int32, device=dev) * main_len
    ei = ep = elen = None
    if not swa_only:
        ei = torch.randint(0, 256 * BS, (nq * extra_len,), dtype=torch.int32, device=dev, generator=g)
        ei[::17] = -1
        ep = torch.arange(0, nq + 1, dtype=torch.int32, device=dev) * extra_len
        elen = torch.full((nq,), extra_len, dtype=torch.int32, device=dev)
    sink = torch.randn(H, device=dev, generator=g)
    # layer output = decode slice of a larger o_padded buffer
    o_padded = torch.full((nq + 3, H, NOPE + ROPE), float("nan"), dtype=torch.bfloat16, device=dev)
    return dict(
        q=q, kv_cache=extra, swa_k_cache=main, swa_only=swa_only, topk_indices=None,
        topk_lens=elen, swa_indices=mi.view(nq, main_len),
        swa_lens=torch.full((nq,), main_len, dtype=torch.int32, device=dev),
        swa_ragged_indices=mi, swa_ragged_indptr=mp,
        topk_ragged_indices=ei, topk_ragged_indptr=ep, attn_sink=sink,
        scale=0.0441941738, head_dim=NOPE + ROPE, nope_head_dim=NOPE,
        rope_head_dim=ROPE, output=o_padded[:nq]), o_padded


for nq in (1, 6, 17, 48):
    for swa_only in (True, False):
        for fast in ("0", "1"):
            os.environ["VLLM_DSV4_SPARSE_DECODE_FAST"] = fast
            kw, buf = attn_case(nq, 128, 512, swa_only, nq * 3 + swa_only)
            set_flag("VLLM_DSV4_ATTN_DIRECT_OUT", False, SP._dsv4_attn_direct_out_enabled)
            SP.rocm_sparse_attn_decode(**kw)
            ref = buf.clone()
            buf.fill_(float("nan"))
            set_flag("VLLM_DSV4_ATTN_DIRECT_OUT", True, SP._dsv4_attn_direct_out_enabled)
            SP.rocm_sparse_attn_decode(**kw)
            check(torch.equal(ref.view(torch.int16), buf.view(torch.int16)),
                  f"direct-out nq={nq} swa_only={swa_only} fast={fast}")
            if args.bench and nq in (6, 48) and fast == "1":
                set_flag("VLLM_DSV4_ATTN_DIRECT_OUT", False, SP._dsv4_attn_direct_out_enabled)
                t0 = graph_time_us(lambda: SP.rocm_sparse_attn_decode(**kw), inner=10)
                n0 = graph_nodes(lambda: SP.rocm_sparse_attn_decode(**kw))
                set_flag("VLLM_DSV4_ATTN_DIRECT_OUT", True, SP._dsv4_attn_direct_out_enabled)
                t1 = graph_time_us(lambda: SP.rocm_sparse_attn_decode(**kw), inner=10)
                n1 = graph_nodes(lambda: SP.rocm_sparse_attn_decode(**kw))
                print(f"   nq={nq:2d} swa_only={swa_only!s:5}: deployed {t0:7.2f} us "
                      f"direct {t1:7.2f} us | nodes {n0} -> {n1}")
set_flag("VLLM_DSV4_ATTN_DIRECT_OUT", False, SP._dsv4_attn_direct_out_enabled)
os.environ.pop("VLLM_DSV4_SPARSE_DECODE_FAST", None)

# ---------------------------------------------------------------------------
# 3. SWA ragged metadata in place (no 2x DtoD per step)
# ---------------------------------------------------------------------------
print("\n[3] SWA ragged build: in place vs build + copy")
for m, width in ((6, 128), (48, 128), (48, 133), (1, 128)):
    g = torch.Generator(device=dev).manual_seed(m + width)
    dense = torch.randint(0, 1 << 20, (m, width), dtype=torch.int32, device=dev, generator=g)
    lens = torch.randint(0, width + 1, (m,), dtype=torch.int32, device=dev, generator=g)
    buf_i_a = torch.full((4096 * width,), -7, dtype=torch.int32, device=dev)
    buf_p_a = torch.full((4097,), -7, dtype=torch.int32, device=dev)
    buf_i_b, buf_p_b = buf_i_a.clone(), buf_p_a.clone()
    ri, rp = SP.build_ragged_indices_from_dense(dense, lens)
    ri, rp = R._copy_ragged_to_graph_buffers(ri, rp, buf_i_a, buf_p_a, m, width)
    oi, op = R._build_ragged_into_graph_buffers(dense, lens, buf_i_b, buf_p_b, m, width)
    nnz = int(rp[-1].item())
    ok = (torch.equal(rp, op) and torch.equal(ri[:nnz], oi[:nnz]) and ri.shape == oi.shape
          and ri.data_ptr() == oi.data_ptr() - (buf_i_b.data_ptr() - buf_i_a.data_ptr())
          and op.data_ptr() == buf_p_b.data_ptr())
    check(ok, f"swa in-place m={m} width={width}")
if args.bench:
    dense = torch.randint(0, 1 << 20, (48, 128), dtype=torch.int32, device=dev)
    lens = torch.full((48,), 128, dtype=torch.int32, device=dev)
    bi = torch.empty(4096 * 128, dtype=torch.int32, device=dev)
    bp = torch.empty(4097, dtype=torch.int32, device=dev)

    def old():
        a, b = SP.build_ragged_indices_from_dense(dense, lens)
        R._copy_ragged_to_graph_buffers(a, b, bi, bp, 48, 128)

    def new():
        R._build_ragged_into_graph_buffers(dense, lens, bi, bp, 48, 128)
    print(f"   M=48: deployed {graph_time_us(old):6.2f} us in-place {graph_time_us(new):6.2f} us"
          f" | nodes {graph_nodes(old)} -> {graph_nodes(new)}")

# ---------------------------------------------------------------------------
# 4. indexer q LUT decode: 1 gather vs cast + index_select; end-to-end logits
# ---------------------------------------------------------------------------
print("\n[4] indexer q LUT gather + fp8_paged_mqa_logits_triton (bitwise)")
lut = ML._get_e4m3fn_bf16_lut(dev)
allb = torch.arange(256, dtype=torch.uint8, device=dev)
check(torch.equal(ML.paged_q_lut_decode(allb, lut).view(torch.int16),
                  lut.index_select(0, allb.to(torch.int32)).view(torch.int16)),
      "lut gather all 256 codes")
q_bf16_default = ML._paged_q_bf16_default(dev)
print("   _paged_q_bf16_default on this GPU:", q_bf16_default,
      "(False => deployed path never pre-decodes q; fusion inert here)")


def logits_case(B, next_n, ctx, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    heads, dim, block = 64, 128, 128
    nblk = (ctx + block - 1) // block
    q = (torch.randn(B, next_n, heads, dim, device=dev, generator=g) * 0.25).to(torch.float8_e4m3fn)
    cache = torch.empty(B * nblk + 3, block, 1, dim + 4, device=dev, dtype=torch.uint8)
    flat = cache.view(cache.shape[0], -1)
    keys = (torch.randn(cache.shape[0], block, dim, device=dev, generator=g) * 0.25).to(torch.float8_e4m3fn)
    flat[:, : block * dim] = keys.view(torch.uint8).reshape(cache.shape[0], -1)
    flat[:, block * dim:] = torch.rand(cache.shape[0], block, device=dev, generator=g).view(
        torch.uint8).reshape(cache.shape[0], -1)
    w = torch.rand(B * next_n, heads, device=dev, generator=g)
    lens = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    bt = torch.randperm(B * nblk + 3, device=dev, generator=g)[: B * nblk].to(torch.int32).view(B, nblk)
    return q, cache, w, lens, bt, ctx


for (B, nn, ctx) in ((1, 6, 1000), (8, 6, 3000), (1, 6, 40000)):
    case = logits_case(B, nn, ctx, B + ctx)
    q8, cache, w, lens, bt, mx = case
    outs = []
    for on in (False, True):
        set_flag("VLLM_DSV41_INDEXER_Q_LUT_FUSED", on, ML._paged_q_lut_fused_enabled)
        lg = ML.fp8_paged_mqa_logits_triton(q8, cache, w, lens, bt, max_model_len=mx, clean_logits=True)
        outs.append(lg.clone())
    check(torch.equal(outs[0].view(torch.int32), outs[1].view(torch.int32)),
          f"paged logits B={B} ctx={ctx}")
    if args.bench and ctx < 10000:
        res = []
        for on in (False, True):
            set_flag("VLLM_DSV41_INDEXER_Q_LUT_FUSED", on, ML._paged_q_lut_fused_enabled)
            fn = lambda: ML.fp8_paged_mqa_logits_triton(q8, cache, w, lens, bt, max_model_len=mx, clean_logits=False)  # noqa: E731
            res.append((graph_time_us(fn, inner=10), graph_nodes(fn)))
        print(f"   B={B} next_n={nn} ctx={ctx}: deployed {res[0][0]:6.2f} us fused {res[1][0]:6.2f} us"
              f" | nodes {res[0][1]} -> {res[1][1]}")
set_flag("VLLM_DSV41_INDEXER_Q_LUT_FUSED", False, ML._paged_q_lut_fused_enabled)

print("\nFAILURES:", FAILS if FAILS else "none")
sys.exit(1 if FAILS else 0)
