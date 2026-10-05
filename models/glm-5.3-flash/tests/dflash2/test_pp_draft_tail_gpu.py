#!/usr/bin/env python3
"""Patch 0021 (VLLM_PP_DRAFT_TAIL_STAGE): single-GPU kernel-level test.

Needs: one CUDA GPU (sm_80 for the thin-GEMM tier) and the patched vLLM
importable (the GLM DFlash2 image).

What it checks, with the real kernels of the tail (thin-M GEMM / cuBLAS via
``dispatch_unquantized_gemm``, the vocabulary top-k ``_topk`` (FlashInfer
radix or torch), ``_score_edges`` and the Triton ``_selector_walk_kernel``)
at production shapes (H 4096, vocab 154,880, k 7, rows 1..8):

1. Fused tail captured in a CUDA graph on the main stream (what the last
   stage replays with the switch off) == split path: ``pack_tail_payload``
   captured in a graph, payload bytes copied to a fresh buffer, tail run
   eagerly on a side stream inside ``tail_side_stream_workspaces`` (what the
   tail stage runs). Draft tokens and realized scores must be bitwise equal.
2. The side-stream tail runs while the main stream replays a "busy" graph
   that launches thin GEMMs of exactly the tail's shapes in the shared
   workspace namespace (the hazard the private scope removes), repeated many
   times with random inputs. Any race shows up as a mismatch.
3. Graph padding: num_reqs < rows with padded rows marked -1.

Usage (in the container, patched tree on PYTHONPATH):
  VLLM_GLM5_THIN_GEMM=1 python test_pp_draft_tail_gpu.py \
      [--draft /models/GLM-5.3-Flash-DFlash2] [--iters 200] [--json out.json]
  # also run once with VLLM_GLM5_THIN_GEMM=0 (cuBLAS path)
  # negative control (expected to be able to fail when thin GEMM is on):
  #   ... --no-private
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default=None, help="draft checkpoint dir (config.json)")
    ap.add_argument("--vocab", type=int, default=154880)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--busy", type=int, default=8, help="busy-graph repeats")
    ap.add_argument("--no-private", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from vllm import envs
    from vllm.model_executor.layers.logits_processor import _topk
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
    from vllm.model_executor.models.qwen3_dflash2 import _score_edges
    from vllm.v1.worker.gpu import pp_draft_tail as T
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        _selector_walk_kernel,
    )
    from vllm.triton_utils import triton

    rank, top_k = args.rank, args.top_k
    if args.draft:
        cfg = json.load(open(os.path.join(args.draft, "config.json")))
        dc = cfg.get("dflash_config", {})
        rank = rank or int(dc["selector_rank"])
        top_k = top_k or int(dc["selector_top_k"])
    rank = rank or 256
    top_k = top_k or 8

    dev = torch.device("cuda")
    torch.manual_seed(0)
    k, H, V, R = args.k, args.hidden, args.vocab, args.max_rows
    Vp = (V + 63) // 64 * 64  # ParallelLMHead pads the vocabulary to 64
    num_pad = Vp - V
    q = k + 1
    bf = torch.bfloat16
    thin = bool(envs.VLLM_GLM5_THIN_GEMM)
    gemm = dispatch_unquantized_gemm()
    print(f"thin_gemm={thin} gemm={getattr(gemm, '__name__', gemm)} "
          f"V={V} Vp={Vp} H={H} k={k} rank={rank} top_k={top_k}")

    W = (torch.randn(Vp, H, device=dev) * 0.02).to(bf)
    pred = (torch.randn(V, rank, device=dev) * 0.1).to(bf)
    succ = (torch.randn(V, rank, device=dev) * 0.1).to(bf)
    proj = (torch.randn(rank, H, device=dev) * 0.02).to(bf)
    scale, soft_cap = 1.0, None

    def compute_candidates(hidden):  # LogitsProcessor.get_top_k_tokens, TP1
        logits = gemm(None, hidden, W, None)
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        values, ids = _topk(logits, top_k)
        ids = ids.to(torch.int64)
        values = values.float()
        if scale != 1.0:
            values = values * scale
        if soft_cap is not None:
            values = torch.tanh(values / soft_cap) * soft_cap
        return ids, values

    def select(candidate_ids, unary_logits, hidden_states, anchor):
        hidden = gemm(None, hidden_states, proj, None)
        return _score_edges(pred, succ, candidate_ids, unary_logits, hidden,
                            anchor, top_k)

    # Last-stage buffers (as DFlashSpeculator / DFlash2Speculator).
    last_hidden = torch.zeros(R * q, H, dtype=bf, device=dev)
    input_ids = torch.zeros(R * q, dtype=torch.int32, device=dev)
    anchor_idx = torch.arange(R, dtype=torch.int64, device=dev) * q
    sample_indices = torch.zeros(R * k, dtype=torch.int64, device=dev)
    sample_pos = torch.zeros(R * k, dtype=torch.int64, device=dev)
    sample_map = torch.full((R * k,), -1, dtype=torch.int32, device=dev)
    temperature = torch.zeros(R, dtype=torch.float32, device=dev)
    seeds = torch.zeros(R, dtype=torch.int64, device=dev)
    draft_tokens = torch.zeros(R, k, dtype=torch.int64, device=dev)
    sel_scores = torch.zeros(R, k, top_k, dtype=torch.float32, device=dev)
    layout = T.TailPayloadLayout(R, k, H, bf, torch.int32)
    staging = torch.zeros(layout.nbytes(R), dtype=torch.uint8, device=dev)
    row_ids = torch.arange(R * k, dtype=torch.int32, device=dev) // k

    def fused(rows):  # DFlash2Speculator._generate_draft after the forward
        n = rows * k
        hs = last_hidden[sample_indices[:n]].view(rows, k, -1)
        cand, unary = compute_candidates(hs.flatten(0, 1))
        cand = cand.view(rows, k, top_k)
        unary = unary.view_as(cand)
        anchor = input_ids[anchor_idx[:rows]]
        scores = select(cand, unary, hs, anchor)
        _selector_walk_kernel[(rows,)](
            scores.contiguous(), cand.contiguous(), sample_pos, sample_map,
            temperature, seeds, draft_tokens, sel_scores,
            num_steps=k, top_k=top_k, BLOCK_K=triton.next_power_of_2(top_k),
            SAMPLE_PROBABILISTIC=False, USE_FP64=False, num_warps=1,
        )  # fmt: skip

    def pack(rows):
        T.pack_tail_payload(
            layout.views(staging, rows), rows, k, last_hidden, sample_indices,
            input_ids, anchor_idx, sample_pos, sample_map, temperature, seeds,
            row_ids,
        )  # fmt: skip

    # Busy work for the main stream: thin GEMMs of the tail's exact shapes in
    # the shared namespace.
    busy_x = torch.randn(R * k, H, device=dev).to(bf)

    def busy(rows):
        for _ in range(args.busy):
            gemm(None, busy_x[: rows * k], W, None)
            gemm(None, busy_x[: rows * k], proj, None)

    rows_set = [r for r in (1, 2, 3, 4, 5, 6, 7, 8) if r <= R]
    graphs = {}
    stream = torch.cuda.Stream()
    for rows in rows_set:
        for name, fn in (("fused", fused), ("pack", pack), ("busy", busy)):
            # Warm-up eagerly first (as CudaGraphManager.capture does), so the
            # thin-GEMM shapes are ready and the graph uses them.
            fn(rows)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(rows)
            graphs[(name, rows)] = g
    torch.cuda.synchronize()

    out_tokens = torch.zeros(R * k, dtype=torch.int64, device=dev)
    realized = torch.zeros(R * k * top_k, dtype=torch.float32, device=dev)
    holder: dict = {}
    if not args.no_private:
        # The runtime validates the FlashInfer cache contract at startup
        # before any private-workspace use; do the same here.
        T.validate_flashinfer_topk_contract(
            torch.device("cuda", torch.cuda.current_device()), holder)

    def tail(payload, rows, num_reqs, out):
        views = layout.views(payload, rows)

        def walk(cand, scores, views, rows):
            T.launch_walk(cand, scores, views, rows, out_tokens, realized, k,
                          top_k, False, False)

        ctx = (contextlib.nullcontext() if args.no_private
               else T.tail_side_stream_workspaces(torch.device("cuda", torch.cuda.current_device()), holder))
        with ctx:
            T.run_draft_tail(compute_candidates, select, walk, views, rows, k,
                             top_k)
        out.copy_(out_tokens.view(-1, k)[:num_reqs])

    # Warm the side-stream path for every row count (as finalize does).
    with torch.cuda.stream(stream):
        for rows in rows_set:
            buf = torch.zeros(layout.nbytes(rows), dtype=torch.uint8, device=dev)
            layout.views(buf, rows).row_state.fill_(-1)
            tail(buf, rows, 0, torch.zeros(0, k, dtype=torch.int64, device=dev))
    torch.cuda.synchronize()

    g = torch.Generator(device="cpu").manual_seed(1)
    mismatches = 0
    checked = 0
    for it in range(args.iters):
        rows = rows_set[it % len(rows_set)]
        num_reqs = int(torch.randint(1, rows + 1, (1,), generator=g))
        # What prepare_dflash_inputs leaves for num_reqs real requests.
        last_hidden.copy_(torch.randn(R * q, H, generator=g).to(bf))
        input_ids.copy_(torch.randint(0, V, (R * q,), generator=g, dtype=torch.int32))
        slots = torch.randperm(R, generator=g)[:num_reqs]
        temperature.copy_(torch.rand(R, generator=g))
        seeds.copy_(torch.randint(0, 1 << 30, (R,), generator=g))
        si = torch.zeros(R * k, dtype=torch.int64)
        sp = torch.zeros(R * k, dtype=torch.int64)
        sm = torch.full((R * k,), -1, dtype=torch.int32)
        for r in range(num_reqs):
            for j in range(k):
                si[r * k + j] = r * q + 1 + j
                sp[r * k + j] = 5000 + 31 * r + j
                sm[r * k + j] = int(slots[r])
        sample_indices.copy_(si)
        sample_pos.copy_(sp)
        sample_map.copy_(sm)
        torch.cuda.synchronize()

        graphs[("fused", rows)].replay()
        ref_tokens = draft_tokens[:num_reqs].clone()
        ref_scores = sel_scores[:rows].clone()
        graphs[("pack", rows)].replay()
        wire = staging[: layout.nbytes(rows)].clone()
        torch.cuda.synchronize()

        payload = torch.empty_like(wire)
        payload.copy_(wire)
        got = torch.empty(num_reqs, k, dtype=torch.int64, device=dev)
        # Main stream busy with same-shape thin GEMMs while the tail runs.
        graphs[("busy", rows)].replay()
        stream.wait_stream(torch.cuda.current_stream())  # payload ready
        graphs[("busy", rows)].replay()
        with torch.cuda.stream(stream):
            tail(payload, rows, num_reqs, got)
        graphs[("busy", rows)].replay()
        torch.cuda.synchronize()

        ok_t = torch.equal(got, ref_tokens)
        valid = (sm.view(-1, k)[:rows, 0] >= 0).to(dev)
        got_s = realized.view(R, k, top_k)[:rows]
        ok_s = torch.equal(got_s[valid], ref_scores[valid])
        checked += 1
        if not (ok_t and ok_s):
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH it={it} rows={rows} num_reqs={num_reqs} "
                      f"tokens={ok_t} scores={ok_s}")
    res = {
        "thin_gemm": thin,
        "private_workspace": not args.no_private,
        "checked": checked,
        "mismatches": mismatches,
        "rows": rows_set,
        "rank": rank,
        "top_k": top_k,
    }
    print(json.dumps(res))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(res) + "\n")
    if args.no_private:
        return 0  # informational
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
