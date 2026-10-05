#!/usr/bin/env python3
"""Patch 0017 (mHC decode v2, full port): numerics, graph replay, microbench.

sm_80 only. OURS_V2 defaults to the file in $GLM_DFLASH2_TREE (default: the
image site-packages).
Prints one JSON object per line.

Implementations compared on identical inputs:
  ours     OURS_V2 = path to ours  .../nvidia/ops/mhc_decode_v2.py (loaded by file path)
  mm       MM_V2   = path to MM    .../vllm/ampere_decode/mhc_decode_v2.py (by file path)
  tilelang the installed vllm's mhc_fused_post_pre_tilelang (incumbent). In the MM
           image it is forced to TileLang with VLLM_GLM5_DECODE_KERNELS=0.
  gate     (our image only, GATE=1) vllm.models.glm5next.nvidia.ops.mhc_decode_v2_gate
           .maybe_fused_post_pre with VLLM_GLM5_DECODE_MHC_V2=1 -- the model's real
           entry point; must equal `ours` bitwise and return None outside the gate.
  fp64     float64 recomputation with the op's two bf16 rounding points.

Env: MS (default 1,2,4,8,16), CASES (default 8), BENCH=1, LAYERS (default 90),
     REPLAYS (default 50), REAL_CKPT=<dir with model.safetensors.index.json> to use
     real hc_{attn,ffn}_{fn,base,scale} and norm weights of layer REAL_LAYER (default 3).
"""

import importlib.util
import json
import os
import sys

import torch

H, HC, SINK = 4096, 4, 20
NOUT = HC * 2 + HC * HC
EPS, RMS_EPS, NORM_EPS, POST_MULT = 1e-6, 1e-5, 1e-5, 2.0
DEV = "cuda"


def emit(**kw):
    print(json.dumps(kw), flush=True)


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- inputs
_REAL = None


def real_params():
    global _REAL
    ck = os.environ.get("REAL_CKPT")
    if not ck or _REAL is not None:
        return _REAL
    from safetensors import safe_open
    L = int(os.environ.get("REAL_LAYER", "3"))
    idx = json.load(open(os.path.join(ck, "model.safetensors.index.json")))["weight_map"]
    want = {k: f"model.layers.{L}.{k}" for k in (
        "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale", "post_attention_layernorm.weight")}
    out = {}
    for k, full in want.items():
        cand = [n for n in idx if n.endswith(full)]
        if not cand:
            emit(event="real_ckpt_missing", key=full)
            return None
        with safe_open(os.path.join(ck, idx[cand[0]]), "pt", device=DEV) as f:
            out[k] = f.get_tensor(cand[0])
    _REAL = dict(fn=out["hc_ffn_fn"].float().contiguous(),
                 hc_base=out["hc_ffn_base"].float().contiguous(),
                 hc_scale=out["hc_ffn_scale"].float().contiguous(),
                 norm_weight=out["post_attention_layernorm.weight"].to(torch.bfloat16).contiguous())
    emit(event="real_ckpt", layer=L)
    return _REAL


def make_case(M, seed, outliers=False):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, mul=1.0):
        return torch.randn(*s, generator=g, device=DEV) * mul

    comb = torch.softmax(rn(M, HC, HC), -1)
    for _ in range(5):
        comb = comb / comb.sum(-2, keepdim=True)
        comb = comb / comb.sum(-1, keepdim=True)
    c = dict(
        x=rn(M, H).to(torch.bfloat16),
        residual=rn(M, HC, H).to(torch.bfloat16),
        post_layer_mix=torch.sigmoid(rn(M, HC, 1)) * POST_MULT,
        comb_res_mix=comb.contiguous(),
        fn=rn(NOUT, HC * H, mul=0.02),
        hc_scale=torch.ones(3, device=DEV) + rn(3, mul=0.1),
        hc_base=rn(NOUT, mul=0.1),
        norm_weight=(1.0 + rn(H, mul=0.1)).to(torch.bfloat16),
    )
    if outliers:
        idx = torch.randint(0, H, (16,), generator=g, device=DEV)
        r = c["residual"].float()
        r[:, :, idx] *= 1000.0
        c["residual"] = r.to(torch.bfloat16)
    rp = real_params()
    if rp is not None:
        c.update({k: v.clone() for k, v in rp.items()})
    return c


SCAL = dict(rms_eps=RMS_EPS, hc_pre_eps=EPS, hc_sinkhorn_eps=EPS,
            hc_post_mult_value=POST_MULT, sinkhorn_repeat=SINK, norm_eps=NORM_EPS)


def ref64(c):
    d = torch.float64
    x, res = c["x"].to(d), c["residual"].to(d)
    post = c["post_layer_mix"].reshape(-1, HC).to(d)
    comb, fn = c["comb_res_mix"].to(d), c["fn"].to(d)
    sc, base, nw = c["hc_scale"].to(d), c["hc_base"].to(d), c["norm_weight"].to(d)
    M = x.shape[0]
    rc = torch.einsum("mkj,mkh->mjh", comb, res) + post[:, :, None] * x[:, None, :]
    rcb = rc.to(torch.bfloat16)
    flat = rc.reshape(M, -1)
    mixes = (flat @ fn.t()) * torch.rsqrt(flat.square().sum(-1, keepdim=True)
                                          / flat.shape[1] + RMS_EPS)
    pre = torch.sigmoid(mixes[:, :HC] * sc[0] + base[:HC]) + EPS
    po = torch.sigmoid(mixes[:, HC:2 * HC] * sc[1] + base[HC:2 * HC]) * POST_MULT
    cm = mixes[:, 2 * HC:].reshape(M, HC, HC) * sc[2] + base[2 * HC:].reshape(1, HC, HC)
    cm = torch.softmax(cm, -1) + EPS
    cm = cm / (cm.sum(-2, keepdim=True) + EPS)
    for _ in range(SINK - 1):
        cm = cm / (cm.sum(-1, keepdim=True) + EPS)
        cm = cm / (cm.sum(-2, keepdim=True) + EPS)
    o = (pre[:, :, None] * rcb.to(d)).sum(1)
    rn = torch.rsqrt(o.square().sum(-1, keepdim=True) / H + NORM_EPS)
    li = o.to(torch.bfloat16).to(d) * rn * nw   # pre-final-rounding, fp64
    return rcb, po.unsqueeze(-1), cm, li


NAMES = ("rc", "post", "comb", "li")


# ---------------------------------------------------------------- impls
def build_impls():
    impls = {}
    for key, env in (("ours", "OURS_V2"), ("mm", "MM_V2")):
        p = os.environ.get(env)
        if key == "ours" and not p:
            p = os.path.join(os.environ.get("GLM_DFLASH2_TREE", "/usr/local/lib/python3.12/dist-packages"),
                             "vllm/models/glm5next/nvidia/ops/mhc_decode_v2.py")
        if p:
            mod = load_by_path(f"_mhc_v2_{key}", p)
            impls[key] = (lambda mod: lambda c: mod.mhc_fused_post_pre(
                c["x"], c["residual"], c["post_layer_mix"], c["comb_res_mix"],
                c["fn"], c["hc_scale"], c["hc_base"], RMS_EPS, EPS, EPS,
                POST_MULT, SINK, 1, 1, c["norm_weight"], NORM_EPS))(mod)
            impls[key + "_mod"] = mod
    try:
        from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang as tl

        # MM image: launch with VLLM_GLM5_DECODE_KERNELS=0 so this is TileLang,
        # not MM's own v1/v2 dispatch inside the same function.
        if os.environ.get("VLLM_GLM5_DECODE_KERNELS", "0") != "0":
            emit(event="warning", msg="VLLM_GLM5_DECODE_KERNELS!=0: 'tilelang' may be MM v1/v2")
        impls["tilelang"] = lambda c: tl(**c, **SCAL)
    except Exception as e:  # noqa: BLE001
        emit(event="tilelang_unavailable", err=repr(e))
    if os.environ.get("GATE") == "1":
        os.environ["VLLM_GLM5_DECODE_MHC_V2"] = "1"
        from vllm.models.glm5next.nvidia.ops import mhc_decode_v2_gate as gate
        impls["gate_mod"] = gate
        impls["gate"] = lambda c: gate.maybe_fused_post_pre(
            c["x"], c["residual"], c["post_layer_mix"], c["comb_res_mix"],
            c["fn"], c["hc_scale"], c["hc_base"], **SCAL,
            norm_weight=c["norm_weight"])
    return impls


def err(a, b):
    d = (a.to(torch.float64) - b.to(torch.float64)).abs()
    return float(d.max()), float(d.mean())


# ---------------------------------------------------------------- numerics
def numerics(impls, ms, cases):
    runners = {k: v for k, v in impls.items() if not k.endswith("_mod")}
    for M in ms:
        for mode in ("rand", "outliers"):
            acc = {}
            bit = {}
            for s in range(cases):
                c = make_case(M, 1000 * M + s, outliers=(mode == "outliers"))
                ref = ref64(c)
                outs = {k: f(c) for k, f in runners.items()}
                torch.cuda.synchronize()
                for k, o in outs.items():
                    if o is None:
                        acc.setdefault(k, {})["none"] = True
                        continue
                    o2 = runners[k](c)
                    bit.setdefault(f"{k}_deterministic", True)
                    bit[f"{k}_deterministic"] &= all(torch.equal(a, b) for a, b in zip(o, o2))
                    for i, n in enumerate(NAMES):
                        mx, mn = err(o[i], ref[i])
                        a = acc.setdefault(k, {}).setdefault(n, [0.0, 0.0])
                        a[0] = max(a[0], mx)
                        a[1] += mn / cases
                for x_, y_ in (("ours", "mm"), ("ours", "tilelang"), ("mm", "tilelang"),
                               ("gate", "ours")):
                    if outs.get(x_) is None or outs.get(y_) is None:
                        continue
                    for i, n in enumerate(NAMES):
                        key = f"{x_}=={y_}:{n}"
                        bit[key] = bit.get(key, True) and torch.equal(outs[x_][i], outs[y_][i])
                        mx, _ = err(outs[x_][i], outs[y_][i])
                        dk = f"{x_}-{y_}:{n}:maxabs"
                        bit[dk] = max(bit.get(dk, 0.0), mx)
                    neq = int((outs[x_][0].view(torch.int16) != outs[y_][0].view(torch.int16)).sum())
                    bit[f"{x_}-{y_}:rc_bf16_mismatch"] = bit.get(f"{x_}-{y_}:rc_bf16_mismatch", 0) + neq
            emit(event="numerics", M=M, mode=mode, cases=cases,
                 err_vs_fp64={k: {n: {"max": v[0], "mean": v[1]} for n, v in d.items()
                                  if n != "none"} | ({"returned_none": True} if "none" in d else {})
                              for k, d in acc.items()},
                 compare=bit)


def gate_edges(impls):
    gate = impls.get("gate_mod")
    if gate is None:
        return
    c = make_case(33, 1)
    r33 = impls["gate"](c)
    os.environ["VLLM_GLM5_DECODE_MHC_V2"] = "0"
    flag_off = gate.flag_on()
    os.environ["VLLM_GLM5_DECODE_MHC_V2"] = "1"
    emit(event="gate_edges", M33_returns_none=r33 is None, flag_on_after_unset=flag_off,
         max_tokens=gate.max_tokens())


# ---------------------------------------------------------------- graph replay
def graph_replay(impls, ms, n_replay=20):
    runners = {k: v for k, v in impls.items() if not k.endswith("_mod") and k != "gate"}
    if "gate" in impls:
        runners["gate"] = impls["gate"]
    for M in ms:
        for k, f in runners.items():
            static = make_case(M, 7)
            f(static)  # eager warm (compile) before capture
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                f(static)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            mem0 = torch.cuda.memory_allocated()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = f(static)
            if out is None:
                emit(event="graph", impl=k, M=M, captured_none=True)
                continue
            ok = True
            for r in range(n_replay):
                fresh = make_case(M, 100 + r)
                for name in ("x", "residual", "post_layer_mix", "comb_res_mix"):
                    static[name].copy_(fresh[name])
                g.replay()
                torch.cuda.synchronize()
                eager = f(static)
                ok &= all(torch.equal(a, b) for a, b in zip(out, eager))
            torch.cuda.synchronize()
            emit(event="graph", impl=k, M=M, replays=n_replay, bitwise_eq_eager=bool(ok),
                 mem_growth_after_capture=torch.cuda.memory_allocated() - mem0)


# ---------------------------------------------------------------- bench
def bench(impls, ms, layers, replays):
    runners = {k: v for k, v in impls.items() if not k.endswith("_mod") and k != "gate"}
    for M in ms:
        row = {}
        for k, f in runners.items():
            cs = [make_case(M, 5000 + i) for i in range(layers)]   # distinct fn per layer: no L2 reuse
            for c in cs[:2]:
                f(c)
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for c in cs:
                    f(c)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for c in cs:
                    f(c)
            for _ in range(3):
                g.replay()
            times = []
            for _ in range(replays):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                g.replay()
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) * 1000.0 / layers)
            times.sort()
            row[k] = {"us_per_call_median": times[len(times) // 2], "us_per_call_min": times[0]}
            del g, cs
            torch.cuda.empty_cache()
        if "tilelang" in row:
            for k in row:
                row[k]["speedup_vs_tilelang"] = (row["tilelang"]["us_per_call_median"]
                                                 / row[k]["us_per_call_median"])
        emit(event="bench", M=M, layers=layers, per_call=row,
             note="per layer-step = 2 calls (attn boundary + ffn boundary)")


def main():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0), "sm_80 only"
    import triton
    emit(event="env", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         triton=triton.__version__)
    ms = [int(m) for m in os.environ.get("MS", "1,2,4,8,16").split(",")]
    impls = build_impls()
    emit(event="impls", have=sorted(k for k in impls if not k.endswith("_mod")))
    for k in ("ours_mod", "mm_mod"):
        if k in impls:
            impls[k].warmup(ms + [32])
    numerics(impls, ms, int(os.environ.get("CASES", "8")))
    gate_edges(impls)
    graph_replay(impls, ms)
    if os.environ.get("BENCH", "1") == "1":
        bench(impls, ms, int(os.environ.get("LAYERS", "90")), int(os.environ.get("REPLAYS", "50")))
    emit(event="done")


if __name__ == "__main__":
    main()
