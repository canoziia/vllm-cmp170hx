#!/usr/bin/env python3
"""GPU ONLY (sm_80). Patch 0023 (VLLM_GLM5_PP_FOLD_DRAFT_FC). Not run by the
author session.

Run inside the patched container (0001..0023 applied, vllm importable):

    python tests/test_pp_fold_draft_fc_gpu.py \
        --draft /models/GLM-5.3-Flash-DFlash2 --json /tmp/fold0023.json
    python -m pytest -q -s tests/test_pp_fold_draft_fc_gpu.py   # random W

What it checks, on real-shape tensors (H = 4096, 5 aux layers, the drafter's
real fc.weight when --draft is given, else a random one with the same
scale):

  1. Numerics, M in {1, 4, 8, 16, 32, 64, 256, 2304} rows:
     ref64 = fc(cat(aux)) in float64; base = F.linear(cat, W) in bf16 (the
     unfolded path, cuBLAS); fold = the stage-ordered fp32 partial sums of
     0023 (recipe split 13,11,11,10 -> stage slots [0],[1],[2,3],[4], using the
     patched model's _mm_fp32), rounded once to bf16.
     Asserts: every fold element is within 1 bf16 ulp of ref64; fold's mean
     abs error to ref64 <= 1.05 x base's (no accuracy loss); reports the
     fraction of elements where fold != base (expected small, rounding-level).
     Also checks that _mm_fp32 took the torch.mm(out_dtype=fp32) path.
  2. CUDA graph: the fold chain captured once (M = 8) and replayed with new
     inputs is bitwise equal to eager.
  3. Timing (informational): last-stage fc work per call, unfolded (cat +
     [M, 5H] x [5H, H]) vs folded (one [M, H] x [H, H] block + cast), and
     each other stage's added work, inside CUDA graphs of 50 calls.

The end-to-end check of draft ids (same prompt, fold off vs on, draft
traces from patch 0015) is in ../DRAFT-TAIL.md §5.
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F

H = 4096
NUM_AUX = 5
STAGE_SLOTS = [[0], [1], [2, 3], [4]]  # 13,11,11,10 with aux (6,15,25,34,43)
ROWS = [1, 4, 8, 16, 32, 64, 256, 2304]


def load_w(draft):
    if not draft:
        g = torch.Generator().manual_seed(0)
        w = torch.randn(H, NUM_AUX * H, generator=g) / (NUM_AUX * H) ** 0.5
        return w.to(torch.bfloat16).cuda(), "random"
    from vllm.v1.worker.gpu.spec_decode.eagle.aux_fc_fold import (
        _draft_checkpoint_files,
        load_fc_column_blocks,
    )
    from types import SimpleNamespace

    files = _draft_checkpoint_files(
        SimpleNamespace(draft_model_config=SimpleNamespace(model=draft))
    )
    blocks = load_fc_column_blocks(files, list(range(NUM_AUX)), H, NUM_AUX)
    w = torch.cat([blocks[i] for i in range(NUM_AUX)], dim=1)
    return w.to(torch.bfloat16).cuda(), draft


def make_aux(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Later layers carry larger states; norms roughly as in the 0015 traces.
    return [
        (torch.randn(m, H, device="cuda", generator=g) * (0.5 + i)).to(torch.bfloat16)
        for i in range(NUM_AUX)
    ]


def fold(aux, blocks, mm):
    partial = None
    for slots in STAGE_SLOTS:  # stage order, as on the wire
        if partial is None:
            partial = torch.zeros(aux[0].shape[0], H, dtype=torch.float32,
                                  device="cuda")
        for s in slots:
            partial = partial + mm(aux[s], blocks[s])
    return partial.to(torch.bfloat16)


def ulp_bf16(x):
    # spacing of bf16 at |x| (8 significant bits)
    e = torch.floor(torch.log2(x.abs().clamp_min(2.0**-126)))
    return torch.pow(2.0, e - 7)


def numerics(w, mm):
    blocks = [w[:, i * H:(i + 1) * H].contiguous() for i in range(NUM_AUX)]
    rows = []
    for m in ROWS:
        aux = make_aux(m, 100 + m)
        cat = torch.cat(aux, dim=-1)
        ref64 = cat.double() @ w.double().t()
        base = F.linear(cat, w)
        f = fold(aux, blocks, mm)
        e_f = (f.double() - ref64).abs()
        e_b = (base.double() - ref64).abs()
        mag = torch.maximum(ref64.float().abs(), f.float().abs())
        within = (e_f <= ulp_bf16(mag).double() + 1e-30).float().mean().item()
        row = {
            "rows": m,
            "fold_within_1ulp": within,
            "fold_mean_err": e_f.mean().item(),
            "base_mean_err": e_b.mean().item(),
            "fold_ne_base_frac": (f != base).float().mean().item(),
        }
        rows.append(row)
        print(json.dumps(row))
        assert within == 1.0, row
        assert row["fold_mean_err"] <= 1.05 * row["base_mean_err"] + 1e-12, row
    return rows


def graph_check(w, mm):
    blocks = [w[:, i * H:(i + 1) * H].contiguous() for i in range(NUM_AUX)]
    static = make_aux(8, 1)
    out = fold(static, blocks, mm)  # warm-up / probe outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fold(static, blocks, mm)
    for seed in (2, 3, 4):
        new = make_aux(8, seed)
        for s, n in zip(static, new):
            s.copy_(n)
        g.replay()
        torch.cuda.synchronize()
        eager = fold(new, blocks, mm)
        assert torch.equal(out, eager), seed
    print("cuda graph replay == eager (bitwise)")


def time_graph(fn, n=50):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    for _ in range(5):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / (5 * n)  # us per call


def timing(w, mm):
    blocks = [w[:, i * H:(i + 1) * H].contiguous() for i in range(NUM_AUX)]
    rows = []
    for m in (4, 8, 16, 32, 64):
        aux = make_aux(m, 7)
        z = torch.zeros(m, H, dtype=torch.float32, device="cuda")
        unfolded = time_graph(lambda: F.linear(torch.cat(aux, -1), w))
        last = time_graph(lambda: (z + mm(aux[4], blocks[4])).to(torch.bfloat16))
        one = time_graph(lambda: z + mm(aux[0], blocks[0]))
        two = time_graph(lambda: z + mm(aux[2], blocks[2]) + mm(aux[3], blocks[3]))
        row = {"rows": m, "last_unfolded_us": unfolded, "last_folded_us": last,
               "stage_1block_us": one, "stage_2blocks_us": two}
        rows.append(row)
        print(json.dumps(row))
    return rows


def run(draft=None, out=None):
    assert torch.cuda.is_available()
    import vllm.models.glm5next.nvidia.model as glm

    mm = glm._mm_fp32
    w, src = load_w(draft)
    print("weight:", src, tuple(w.shape))
    res = {"weight": src, "numerics": numerics(w, mm)}
    assert glm._MM_OUT_DTYPE_OK is True, "torch.mm(out_dtype=fp32) unavailable"
    graph_check(w, mm)
    res["timing"] = timing(w, mm)
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
    return res


def test_fold_gpu():
    run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    run(a.draft, a.json)
    print("ALL OK")
    sys.exit(0)
