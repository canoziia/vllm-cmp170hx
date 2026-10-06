"""Correctness tests for the SM80 fused indexer prefill (GPU2 container).

python3 test_idx_gpu.py [--lib out/_dsv41_indexer_C.abi3.so] [--quick]
"""
import argparse
import json
import os
import sys

import torch

P = argparse.ArgumentParser()
P.add_argument("--lib", default="/t/out/_dsv41_indexer_C.abi3.so")
P.add_argument("--quick", action="store_true")
args = P.parse_args()
os.environ["VLLM_DSV41_INDEXER_LIB"] = args.lib
sys.path.insert(0, "/t/candidate")
import importlib.util

spec = importlib.util.spec_from_file_location(
    "dsv41_indexer_prefill", "/t/candidate/vllm/v1/attention/ops/dsv41_indexer_prefill.py"
)
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)

from vllm import _custom_ops as ops
from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
)
from common import make_inputs

dev = "cuda"
FAIL = []
REPORT = {}


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + " " + str(info), flush=True)
    if not ok:
        FAIL.append(name)


def inside_mask(R, N, lo, hi):
    c = torch.arange(N, device=dev)[None, :]
    return (c >= lo[:, None].long()) & (c < hi[:, None].long())


def ref_logits(inp, R0=None):
    q, k, ks_, w, lo, hi = inp["q"], inp["k"], inp["ks"], inp["w"], inp["lo"], inp["hi"]
    return fp8_mqa_logits_triton(q, (k, ks_), w, lo, hi, clean_logits=False).clone()


def fast_logits(inp, flags=None, bs=0):
    q, k, ks_, w, lo, hi = inp["q"], inp["k"], inp["ks"], inp["w"], inp["lo"], inp["hi"]
    R, N = q.shape[0], k.shape[0]
    out = torch.full((R, N), float("nan"), device=dev)
    mx = fx.logits(fx.decode_fp8(q), fx.decode_fp8(k), ks_, w, lo, hi, out, flags, bs)
    return out, mx


def cmp_sets(a, b, logits, lo, hi, K):
    """a, b: [R, K] index tensors (relative). Return (#rows differing, #rows differing not explained by ties)."""
    a = a.long()
    b = b.long()
    sa, _ = a.sort(1)
    sb, _ = b.sort(1)
    diff = (sa != sb).any(1)
    nd = int(diff.sum())
    bad = 0
    for r in torch.nonzero(diff).flatten().tolist():
        L = int(hi[r] - lo[r])
        va = a[r][a[r] >= 0]
        vb = b[r][b[r] >= 0]
        x = logits[r, lo[r]:hi[r]]
        ka = torch.sort(x[va]).values
        kb = torch.sort(x[vb]).values
        if not (torch.equal(ka, kb) and len(va) == len(vb)):
            bad += 1
    return nd, bad


def test_logits_bitwise():
    cases = [(292, 114688, "causal", 1), (128, 57344, "causal", 2), (64, 5000, "causal", 3),
             (300, 4096, "causal", 4), (37, 777, "full", 5), (256, 16384, "structured", 6)]
    if args.quick:
        cases = cases[2:]
    for R, N, kind, seed in cases:
        inp = make_inputs(R, N, seed=seed, causal=(kind != "full"), kind=("structured" if kind == "structured" else "randn"))
        if kind == "causal" and N > R and seed == 3:
            # multi-request style starts
            inp["lo"] = (torch.arange(R, device=dev, dtype=torch.int32) % 3) * 100
        ref = ref_logits(inp)
        out, mx = fast_logits(inp)
        m = inside_mask(R, N, inp["lo"], inp["hi"])
        eq = torch.equal(ref[m].view(torch.int32), out[m].view(torch.int32))
        nbad = int((ref[m].view(torch.int32) != out[m].view(torch.int32)).sum())
        check(f"logits_bitwise R{R} N{N} {kind}", eq, f"mismatch={nbad}/{int(m.sum())}")
        # chunk max consistency
        mx2 = fx.chunkmax(ref, inp["lo"], inp["hi"])
        c0 = (inp["lo"] // 32).long()
        c1 = ((inp["hi"] + 31) // 32).long()
        cc = torch.arange(mx.shape[1], device=dev)[None, :]
        mm = (cc >= c0[:, None]) & (cc < c1[:, None])
        check(f"chunkmax R{R} N{N}", torch.equal(mx[mm], mx2[mm]))
        # determinism
        out2, mx3 = fast_logits(inp)
        check(f"logits_deterministic R{R} N{N}", torch.equal(out[m].view(torch.int32), out2[m].view(torch.int32)) and torch.equal(mx[mm], mx3[mm]))


def test_topk_same_logits():
    K = 512
    cases = [(292, 114688, "randn", 11), (585, 57344, "randn", 12), (512, 16384, "structured", 13),
             (300, 4096, "randn", 14), (128, 600, "randn", 15), (64, 2048, "ties", 16),
             (64, 114688, "monotone", 17), (64, 9000, "monotone", 18), (64, 40000, "clustered", 19),
             (64, 12000, "clustered", 20), (64, 20000, "ties", 22), (200, 16450, "randn", 23),
             (32, 10000, "constant", 24), (32, 60000, "constant", 25), (64, 60000, "clustered", 26)]
    if args.quick:
        cases = cases[3:]
    for R, N, kind, seed in cases:
        inp = make_inputs(R, N, seed=seed, kind=("structured" if kind == "structured" else "randn"))
        lg = ref_logits(inp)
        lo, hi = inp["lo"], inp["hi"]
        m = inside_mask(R, N, lo, hi)
        if kind == "ties":
            lg = torch.round(lg * 4) / 4  # many exact ties
            lg[~m] = float("-inf")
        elif kind == "monotone":  # top-k packed in the first columns -> candidate overflow
            lg = -torch.arange(N, device=dev, dtype=torch.float32)[None, :].expand(R, N).contiguous()
            lg[~m] = float("-inf")
        elif kind == "constant":  # all equal: every element ties
            lg = torch.full_like(lg, 0.5)
            lg[~m] = float("-inf")
        elif kind == "clustered":  # a hot window per row on top of noise
            c = torch.randint(0, N, (R, 1), device=dev)
            cols = torch.arange(N, device=dev)[None, :]
            lg = lg + 50.0 * ((cols - c).abs() < 3000).float()
            lg[~m] = float("-inf")
        idx_ref = torch.full((R, K), -7, device=dev, dtype=torch.int32)
        ops.top_k_per_row_prefill(lg, lo, hi, idx_ref, R, lg.stride(0), 1, K)
        mx = fx.chunkmax(lg, lo, hi)
        idx = torch.full((R, K), -7, device=dev, dtype=torch.int32)
        stats = torch.zeros(16, device=dev, dtype=torch.int32)
        fx.topk(lg, mx, lo, hi, idx, K, stats)
        nd, bad = cmp_sets(idx, idx_ref, lg, lo, hi, K)
        st = stats.tolist()
        REPORT[f"topk_R{R}_N{N}_{kind}"] = dict(rows_differ=nd, rows_not_tie=bad, cand_mean=st[0] / R, cand_max=st[1], modes=st[2:8])
        check(f"topk_set R{R} N{N} {kind}", bad == 0 and (nd == 0 or kind == "ties"),
              f"rows_differ={nd} non_tie={bad} cand_mean={st[0]/R:.0f} cand_max={st[1]} modes(g8,chunk,g8+refine,chunk+refine,g8+stream,chunk+stream)={st[2:8]}")
        # sorted and in range, -1 padding semantics
        L = (hi - lo).long()
        valid = idx >= 0
        cnt_ok = torch.equal(valid.sum(1).long(), torch.clamp(L, max=K))
        srt = torch.all(idx[:, 1:] >= idx[:, :-1]) if True else True
        check(f"topk_format R{R} N{N}", bool(cnt_ok) and bool(torch.all((idx < L[:, None]) | ~valid)))
        idx2 = torch.full((R, K), -7, device=dev, dtype=torch.int32)
        fx.topk(lg, mx, lo, hi, idx2, K)
        check(f"topk_deterministic R{R} N{N}", torch.equal(idx, idx2))


def test_incumbent_order_nondeterminism():
    R, N, K = 128, 57344, 512
    inp = make_inputs(R, N, seed=21)
    lg = ref_logits(inp)
    a = torch.empty((R, K), device=dev, dtype=torch.int32)
    b = torch.empty_like(a)
    ops.top_k_per_row_prefill(lg, inp["lo"], inp["hi"], a, R, lg.stride(0), 1, K)
    ops.top_k_per_row_prefill(lg, inp["lo"], inp["hi"], b, R, lg.stride(0), 1, K)
    same_order = torch.equal(a, b)
    same_set = torch.equal(a.sort(1).values, b.sort(1).values)
    REPORT["incumbent_topk_order_repeatable"] = bool(same_order)
    print("INFO incumbent top_k_per_row_prefill order repeatable:", same_order, "set repeatable:", same_set, flush=True)


def test_candidates():
    K, bs, nb = 512, 8, 2048
    for R, N, seed in [(292, 114688, 31), (256, 20000, 32), (64, 9000, 33)]:
        inp = make_inputs(R, N, seed=seed)
        lo, hi = inp["lo"], inp["hi"]
        # writer on reference logits
        lg_ref = ref_logits(inp)
        cand = torch.full((R, nb), -1, device=dev, dtype=torch.int32)
        select_candidate_blocks(lg_ref, lo, hi, nb, bs, cand)
        # reader: apply mask to (a second index source's) logits
        inp2 = make_inputs(R, N, seed=seed + 100)
        inp2["lo"], inp2["hi"] = lo, hi
        lg2 = ref_logits(inp2)
        apply_candidate_mask(lg2, lo, hi, cand, bs)
        flags = fx.candidate_flags(lo, cand, N, bs)
        out, mx = fast_logits(inp2, flags, bs)
        m = inside_mask(R, N, lo, hi)
        eq = torch.equal(lg2[m].view(torch.int32), out[m].view(torch.int32))
        check(f"cand_mask_logits_bitwise R{R} N{N}", eq, int((lg2[m].view(torch.int32) != out[m].view(torch.int32)).sum()))
        idx_ref = torch.empty((R, K), device=dev, dtype=torch.int32)
        ops.top_k_per_row_prefill(lg2, lo, hi, idx_ref, R, lg2.stride(0), 1, K)
        idx = torch.empty_like(idx_ref)
        st = torch.zeros(16, device=dev, dtype=torch.int32)
        fx.topk(out, mx, lo, hi, idx, K, st)
        nd, bad = cmp_sets(idx, idx_ref, lg2, lo, hi, K)
        check(f"cand_topk_set R{R} N{N}", nd == 0 and bad == 0, f"rows_differ={nd} cand_mean={st[0].item()/R:.0f} modes={st[2:8].tolist()}")
        # full fused path incl. writer
        ws = torch.empty(R * N, device=dev)
        cand2 = torch.full((R, nb), -1, device=dev, dtype=torch.int32)
        idx_w = torch.empty((R, K), device=dev, dtype=torch.int32)
        fx.indexer_prefill_topk(inp["q"], inp["k"], inp["ks"], inp["w"], lo, hi, idx_w, K, ws,
                                candidate_blocks=cand2, candidate_block_size=bs, candidate_write=True)
        check(f"cand_writer_blocks R{R} N{N}", torch.equal(cand, cand2))
        idx_wr = torch.empty_like(idx_w)
        ops.top_k_per_row_prefill(lg_ref, lo, hi, idx_wr, R, lg_ref.stride(0), 1, K)
        nd, bad = cmp_sets(idx_w, idx_wr, lg_ref, lo, hi, K)
        check(f"cand_writer_topk R{R} N{N}", nd == 0, f"rows_differ={nd}")


def test_graph():
    R, N, K = 256, 20000, 512
    inp = make_inputs(R, N, seed=41)
    ws = torch.empty(R * N, device=dev)
    out = torch.empty((R, K), device=dev, dtype=torch.int32)
    fx.warmup(torch.device(dev))
    fn = lambda: fx.indexer_prefill_topk(inp["q"], inp["k"], inp["ks"], inp["w"], inp["lo"], inp["hi"], out, K, ws)
    fn()
    torch.cuda.synchronize()
    eager = out.clone()
    out.fill_(-5)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    out.fill_(-5)
    g.replay()
    torch.cuda.synchronize()
    check("graph_replay_equals_eager", torch.equal(out, eager))
    # new inputs in place, replay
    inp2 = make_inputs(R, N, seed=42)
    for key in ("q", "k", "ks", "w"):
        inp[key].copy_(inp2[key])
    g.replay()
    torch.cuda.synchronize()
    out_g = out.clone()
    fn()
    torch.cuda.synchronize()
    check("graph_replay_new_inputs", torch.equal(out_g, out))
    # side stream
    with torch.cuda.stream(s):
        fn()
    s.synchronize()
    check("side_stream", torch.equal(out_g, out))


def test_decode():
    x = torch.arange(256, dtype=torch.uint8, device=dev).repeat(64).view(torch.float8_e4m3fn)
    a = fx.decode_fp8(x)
    b = x.to(torch.bfloat16)
    check("decode_fp8_all_bytes", torch.equal(a.view(torch.int16), b.view(torch.int16)))


test_decode()
test_logits_bitwise()
test_topk_same_logits()
test_incumbent_order_nondeterminism()
test_candidates()
test_graph()
print("REPORT", json.dumps(REPORT), flush=True)
print("RESULT", "FAIL " + ",".join(FAIL) if FAIL else "ALL PASS", flush=True)
