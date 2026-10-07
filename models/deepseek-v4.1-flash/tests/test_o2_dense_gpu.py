"""Final end-to-end validation through dsv4_o2_dense (O2/O4/O5 dispatch + Marlin fallback).
--acc: fp64 accuracy + determinism over many M;  --safety: graph capture;  --seq: 8-layer sequence."""
import argparse, os, statistics, time
os.environ["VLLM_DSV4_O2_DENSE"] = "1"; os.environ["VLLM_DSV4_O2_LIB"] = "out/_o2.so"
import torch
import dsv4_o2_dense as D
from o2_common import SHAPES, clone, emit, graph_of, make, marlin, nbytes, time_graph
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/final_all.jsonl")
for k in ("acc", "safety", "seq"): ap.add_argument("--" + k, action="store_true")
a = ap.parse_args(); f = open(a.out, "a"); fails = []


def mk(K, N, seed, ref=False, lo=115, hi=123):
    L, r = make(K, N, seed, ref=ref, exp_lo=lo, exp_hi=hi)
    L.input_size_per_partition, L.output_size_per_partition = K, N
    D.prepare(L); assert L._o2_ok
    return L, r


def apply(x, L):
    o = D.try_apply(x, L, L.N, L.K, None)
    return o if o is not None else marlin(x, L)


MS = [1, 3, 6, 8, 9, 13, 16, 22, 32, 40, 48, 57, 64, 65, 80, 96, 97, 110, 127, 128, 129, 140, 150, 170, 191, 192]
if a.acc:
    for K, N, nm in SHAPES:
        for seed, (lo, hi) in ((5, (115, 123)), (6, (100, 135))):
            L, ref = mk(K, N, 500 * seed + K + N, True, lo, hi)
            g = torch.Generator(device="cuda").manual_seed(seed)
            for kind in ("normal", "wide"):
                for M in MS:
                    x = torch.randn(M, K, device="cuda", generator=g)
                    if kind == "wide": x = x * torch.exp2(torch.randint(-8, 8, (M, K), device="cuda", generator=g).float())
                    x = x.to(torch.bfloat16)
                    sel = D.select(M, K, N)
                    r = x.double() @ ref; sc = r.abs().mean().item()
                    o = apply(x, L); eo = (o.double() - r).abs(); em = (marlin(x, L).double() - r).abs()
                    det = torch.equal(o, apply(x, L))
                    rec = dict(exp="acc_all", shape=nm, seed=seed, dist=kind, M=M, path=sel[0] if sel else "marlin",
                               ours_mean=eo.mean().item() / sc, ours_max=eo.max().item() / sc,
                               marlin_mean=em.mean().item() / sc, marlin_max=em.max().item() / sc, det=det, nan=bool(torch.isnan(o).any()))
                    emit(f, **rec)
                    if not det or rec["nan"] or rec["ours_mean"] > 1.05 * rec["marlin_mean"] or rec["ours_max"] > 1.5 * rec["marlin_max"]:
                        fails.append((nm, seed, kind, M))
            del L, ref
    print("ACC FAILURES:", fails, flush=True)

if a.safety:
    for K, N, nm in SHAPES:
        L, _ = mk(K, N, 9)
        for M in (6, 40, 100, 160):
            x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
            out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
            g = graph_of([lambda: out.copy_(apply(x, L))])
            ok = True
            for i in range(5):
                x.copy_(torch.randn(M, K, device="cuda").to(torch.bfloat16)); g.replay()
                ok &= torch.equal(out, apply(x, L))
            emit(f, exp="safety_all_graph", shape=nm, M=M, ok=ok)
            if not ok: fails.append(("graph", nm, M))
    print("SAFETY FAILURES:", fails, flush=True)

if a.seq:
    big = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    for M in (1, 6, 16, 32, 48, 64, 96, 128, 160, 192):
        layers = [[mk(k, n, 100 * l + i)[0] for i, (k, n, _) in enumerate(SHAPES)] for l in range(8)]
        xs = {k: torch.randn(M, k, device="cuda").to(torch.bfloat16) for k, _, _ in SHAPES}
        gm = graph_of([lambda L=L: marlin(xs[L.K], L) for lay in layers for L in lay])
        go = graph_of([lambda L=L: apply(xs[L.K], L) for lay in layers for L in lay])
        tt = time.time()
        while time.time() - tt < 0.2: big.add_(1)
        rm, ro = [], []
        for rep in range(7):
            for g, r in ((gm, rm), (go, ro)) if rep % 2 == 0 else ((go, ro), (gm, rm)):
                r.append(time_graph(g, 40, replays=8, trials=3)["us"] * 5)
        um, uo = statistics.median(rm), statistics.median(ro)
        tot = sum(nbytes(k, n) for k, n, _ in SHAPES) / 1e3
        emit(f, exp="seq_all", M=M, marlin_layer_us=round(um, 2), ours_layer_us=round(uo, 2), speedup=round(um / uo, 3),
             paths={nm: (D.select(M, k, n) or ("marlin",))[0] for k, n, nm in SHAPES}, ours_GBs=round(tot / uo), per_rep=[round(p / q, 3) for p, q in zip(rm, ro)])
        del gm, go, layers; torch.cuda.empty_cache()
print("FAILURES:", fails if fails else "none")
