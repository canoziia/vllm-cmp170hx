#!/usr/bin/env python3
"""GPU test + micro-benchmark for VLLM_DSPARK_DRAFT_HEAD_FP8 (fp8 draft lm_head).

Usage (inside the vLLM image, one GPU):
  python3 test_draft_head_fp8_gpu.py [--src PATH/dspark_fp8_head.py] [--bench]
         [--lm-head /models/DeepSeek-V4.1-Flash] [--hidden hidden.pt]

--src      load the module from this file instead of the installed (patched)
           vllm.models.deepseek_v4_1.nvidia.dspark_fp8_head
--lm-head  checkpoint dir; reads the target head weight ("head.weight" or
           "lm_head.weight") via the safetensors index. Default: random
           N(0, 0.02) [129280, 5120] bf16.
--hidden   optional .pt with real pre-norm-applied hidden rows [M, 5120] bf16
           (e.g. dumped sample_hidden after model.norm) for a top-1 agreement
           estimate on real activations.

Checks:
  1. Marlin fp8 output vs fp64 reference of the *dequantized* weight
     (kernel correctness; tolerance relative to bf16 output rounding).
  2. fp8 head vs bf16 head: logit error, top-1 agreement, top-1 margin of
     disagreements (acceptance-risk proxy).
  3. CUDA graph capture/replay bitwise equal to eager, and run-to-run
     determinism.
  4. --bench: bf16 F.linear vs fp8 Marlin at M in {5,6,10,12,20,30,40,48}.
"""

import argparse
import importlib.util
import json
import os
import sys

import torch
import vllm.model_executor.layers.fused_moe  # noqa: F401  (import order)
import torch.nn.functional as F

N_VOCAB, HIDDEN = 129280, 5120


def load_module(src):
    if src is None:
        from vllm.models.deepseek_v4_1.nvidia import dspark_fp8_head as m

        return m
    spec = importlib.util.spec_from_file_location("dspark_fp8_head", src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_head(path, dev):
    if path is None:
        g = torch.Generator(device=dev).manual_seed(0)
        w = torch.randn(N_VOCAB, HIDDEN, device=dev, generator=g) * 0.02
        return w.to(torch.bfloat16)
    from safetensors import safe_open

    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    for name in ("head.weight", "lm_head.weight"):
        if name in wm:
            with safe_open(os.path.join(path, wm[name]), "pt", device="cpu") as f:
                return f.get_tensor(name).to(dev, torch.bfloat16)
    raise SystemExit("head weight not found in index")


def bench(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--lm-head")
    ap.add_argument("--hidden")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    m = load_module(a.src)
    w = load_head(a.lm_head, dev)
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    head = m.Fp8DraftHead(w)
    torch.cuda.synchronize()
    print(f"fp8 head built: +{(torch.cuda.memory_allocated()-m0)/2**30:.3f} GiB "
          f"(peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB)")
    ok = True

    # Reference dequantized weight (rebuild fp8 rows for the reference).
    q, sc = m.quantize_rows_fp8(w)
    assert torch.equal(sc, head.row_scale)

    g = torch.Generator(device=dev).manual_seed(1)
    # Unit-RMS rows (the drafter feeds model.norm(h) times norm weight).
    x = torch.randn(48, HIDDEN, device=dev, generator=g).to(torch.bfloat16)
    if a.hidden:
        xh = torch.load(a.hidden, map_location=dev).to(torch.bfloat16)
        x = xh.reshape(-1, HIDDEN)
    M = x.shape[0]

    y8 = head(x)
    assert y8.shape == (M, N_VOCAB) and y8.dtype == torch.bfloat16

    # 1) kernel correctness vs fp64 ref on the dequantized weight (row chunks).
    max_err = 0.0
    max_ref = 0.0
    for s in range(0, N_VOCAB, 16384):
        e = min(N_VOCAB, s + 16384)
        wd = q[s:e].double() * sc[s:e].double()[:, None]
        ref = x.double() @ wd.T
        max_err = max(max_err, (y8[:, s:e].double() - ref).abs().max().item())
        max_ref = max(max_ref, ref.abs().max().item())
    # bf16 output rounding alone is up to 2^-8 relative to |ref|.
    tol = max_ref * 2**-7
    print(f"[1] marlin vs fp64(dequant): max_abs_err={max_err:.4e} "
          f"max|ref|={max_ref:.3f} tol={tol:.3e}")
    ok &= max_err <= tol

    # 2) fp8 vs bf16 head.
    y16 = F.linear(x, w)
    d = (y8.float() - y16.float()).abs()
    agree = (y8.float().argmax(-1) == y16.float().argmax(-1))
    top2 = y16.float().topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1])
    print(f"[2] fp8 vs bf16: max_abs={d.max().item():.4e} mean_abs={d.mean().item():.4e} "
          f"top1_agree={agree.float().mean().item()*100:.2f}% ({int(agree.sum())}/{M})")
    if (~agree).any():
        print(f"    bf16 top1-top2 margin on disagreeing rows: "
              f"{margin[~agree].tolist()[:16]}")
    if not a.hidden:
        print("    (random hidden/weights: agreement is only indicative; "
              "use --lm-head/--hidden for an acceptance proxy)")

    # 3) graph capture == eager, determinism.
    for mm in (5, 6, 30, 48):
        xs = x[:mm].clone()
        eager = head(xs)
        again = head(xs)
        ok &= torch.equal(eager, again)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            head(xs)
        torch.cuda.current_stream().wait_stream(st)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = head(xs)
        gr.replay()
        torch.cuda.synchronize()
        eq = torch.equal(out, eager)
        print(f"[3] M={mm}: graph==eager {eq}, deterministic {torch.equal(eager, again)}")
        ok &= eq

    if a.bench:
        print("[4] bench (us): M  bf16_linear  fp8_marlin  saving")
        for mm in (5, 6, 10, 12, 20, 30, 40, 48):
            xs = torch.randn(mm, HIDDEN, device=dev).to(torch.bfloat16)
            t16 = bench(lambda: F.linear(xs, w))
            t8 = bench(lambda: head(xs))
            print(f"    {mm:3d}  {t16:9.1f}  {t8:9.1f}  {t16-t8:8.1f}")
        print(f"    bf16 head {w.numel()*2/1e9:.3f} GB, fp8 head {head.nbytes()/1e9:.3f} GB")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
