"""0013 route v2: real-GPU correctness test (sm_80, patched vllm installed).

Never auto-runs from CPU checks.  Prints JSON lines; exits non-zero on failure.

    python3 tests/test_route_v2_gpu.py [--rounds 200] [--quick]

Reference ("incumbent") = what the unpatched path does for these shapes:
  logits  GateLinear tier 3.5 bf16_gemv (M <= 8) / cuBLAS torch.mm (M > 8)
  routing ops.grouped_topk via fused_grouped_topk (single_group_topk kernel)
  align   independent CPU reference (expert-major, flattened-lane order,
          sentinel numel, expert tail -1), plus 00010 `_align` bitwise
          (M 4/8) and upstream moe_align_block_size as per-block sets.

Checks
  gate mode : ids (ordered), weights (bit pattern) and alignment BITWISE.
  tc   mode : (a) route stage == fused_grouped_topk on tc's own logits,
              bitwise; (b) vs incumbent logits: max |dlogit|, per-row expert
              set agreement; any set flip must be a near tie (fp64 gap of
              the 8th/9th key < FLIP_GAP) else FAIL; weight rel. error.
  inputs    random, exact ties (duplicated gate rows), engineered near ties
            (bias moved to tie / +-1 ulp of the 8th key), saturated logits,
            non-finite rows.
  graphs    M=1/4/8 x both modes captured separately, alternately replayed
            with fresh inputs every round, compared to eager reference.
  streams   two streams launching concurrently (per-stream scratch).
  hook      maybe_route -> GroupedTopKRouter.select_experts -> take_align
            handoff hit; in-place edit of ids -> miss.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_GLM5_ROUTE_V2", "1")
os.environ.setdefault("VLLM_GLM5_MARLIN_DECODE_CUDA", "1")
os.environ.setdefault("VLLM_GLM5_MARLIN_DECODE_TOKENS", "1,4,8")
os.environ["VLLM_GLM5_ROUTER_ALIGN_DECODE"] = "1"   # 00010 comparison only

import torch  # noqa: E402

from vllm.model_executor.kernels.linear.gemv_triton import (  # noqa: E402
    bf16_gemv, should_use_triton_gemv)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (  # noqa: E402
    fused_grouped_topk)
from vllm.models.glm5next.nvidia.ops import route_v2_decode as rv  # noqa: E402

E, K, TOPK, RSF = 288, 4096, 8, 2.5
FLIP_GAP = 1e-5
FAIL = []


def emit(**kw):
    print(json.dumps(kw), flush=True)


def check(ok, **kw):
    kw["pass"] = bool(ok)
    emit(**kw)
    if not ok:
        FAIL.append(kw)


def gate_logits(x, w):
    if should_use_triton_gemv(x, w):
        return bf16_gemv(x, w, torch.float32)
    return torch.mm(x, w.T, out_dtype=torch.float32)


def ref_route(x, logits, bias):
    return fused_grouped_topk(x, logits, TOPK, True, bias, 1, 1, "sigmoid", RSF)


def cpu_align(ids):
    flat = ids.cpu().flatten().tolist()
    s = len(flat)
    sorted_ids, owners = [], []
    for e in range(E):
        rows = [i for i, v in enumerate(flat) if v == e]
        if rows:
            pad = rows + [s] * ((-len(rows)) % 8)
            sorted_ids += pad
            owners += [e] * (len(pad) // 8)
    return sorted_ids, owners


def check_align(tag, ids, out):
    srt, eids, ntpp = [t.cpu() for t in out]
    exp_s, exp_o = cpu_align(ids)
    n = int(ntpp.item())
    s = ids.numel()
    ok = (n == len(exp_s) and srt[:n].tolist() == exp_s
          and eids[:n // 8].tolist() == exp_o
          and bool((srt[n:] == s).all()) and bool((eids[n // 8:] == -1).all()))
    res = dict(check="align_cpu_ref", tag=tag, ntpp=n)
    if ids.shape[0] in (4, 8):
        from vllm.models.glm5next.nvidia.ops.router_align_decode import maybe_align
        a = maybe_align(ids, E)
        if a is not None:
            m = min(a[0].numel(), srt.numel())
            same = (torch.equal(a[0][:m].cpu(), srt[:m]) and int(a[2].item()) == n
                    and torch.equal(a[1][:n // 8].cpu(), eids[:n // 8]))
            res["eq_00010"] = bool(same)
            ok = ok and same
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)
    us, ue, un = moe_align_block_size(ids, 8, E, None, ignore_invalid_experts=True)
    un = int(un.item())
    blocks_up = sorted((int(ue[b]), tuple(sorted(us[b * 8:b * 8 + 8].tolist())))
                       for b in range(un // 8))
    blocks_me = sorted((int(eids[b]), tuple(sorted(srt[b * 8:b * 8 + 8].tolist())))
                       for b in range(n // 8))
    res["eq_upstream_sets"] = un == n and blocks_up == blocks_me
    ok = ok and res["eq_upstream_sets"]
    check(ok, **res)


def keys64(logits, bias):
    lg = logits.double()
    return 0.5 * torch.tanh(0.5 * lg) + 0.5 + bias.double()


def bits(t):
    return t.contiguous().view(torch.int32)


def make_inputs(kind, M, gen, dev):
    w = (torch.randn(E, K, generator=gen) * 0.02).to(torch.bfloat16)
    bias = (torch.randn(E, generator=gen) * 0.05).float()
    x = torch.randn(M, K, generator=gen).to(torch.bfloat16)
    if kind == "exact_tie":
        w[1::2] = w[0::2]
        bias[1::2] = bias[0::2]
    elif kind == "saturated":
        x = (x.float() * 40).to(torch.bfloat16)
        bias.zero_()
    elif kind == "nonfinite":
        x[0, 5] = float("inf")
        if M > 1:
            x[M - 1, 7] = float("nan")
    w, bias, x = w.to(dev), bias.to(dev), x.to(dev)
    if kind.startswith("near_tie"):
        # Move bias of row-0's 9th expert to the 8th's key (tie), or +-1 ulp.
        delta = {"near_tie_eq": 0, "near_tie_up": 1, "near_tie_dn": -1}[kind]
        lg = gate_logits(x, w)
        key = (0.5 * torch.tanh(0.5 * lg[0]) + 0.5) + bias
        order = torch.argsort(key, descending=True, stable=True)
        e8, e9 = int(order[7]), int(order[8])
        s9 = (0.5 * torch.tanh(0.5 * lg[0, e9]) + 0.5)
        target = key[e8]
        if delta:
            target = torch.nextafter(target, torch.tensor(
                float("inf") if delta > 0 else float("-inf"), device=dev))
        bias[e9] = target - s9
    return x, w, bias


def run_case(kind, M, seed, dev):
    gen = torch.Generator().manual_seed(seed)
    x, w, bias = make_inputs(kind, M, gen, dev)
    lg_ref = gate_logits(x, w)
    w_ref, i_ref = ref_route(x, lg_ref, bias)
    tag = f"{kind}/M{M}/s{seed}"

    # gate mode: routes the incumbent's own logits
    lg, tw, ti, srt, eids, ntpp = rv.route_v2(bias, logits=lg_ref, renormalize=True,
                                              routed_scaling_factor=RSF)
    check(torch.equal(ti, i_ref) and torch.equal(bits(tw), bits(w_ref)),
          check="gate_bitwise", tag=tag,
          ids_eq=bool(torch.equal(ti, i_ref)),
          w_bits_eq=bool(torch.equal(bits(tw), bits(w_ref))),
          w_max_abs=float((tw - w_ref).abs().max()))
    check_align("gate/" + tag, ti, (srt, eids, ntpp))

    # tc mode
    lg_t, tw_t, ti_t, srt_t, eids_t, ntpp_t = rv.route_v2(
        bias, x=x, weight=w, renormalize=True, routed_scaling_factor=RSF)
    w_own, i_own = ref_route(x, lg_t, bias)
    check(torch.equal(ti_t, i_own) and torch.equal(bits(tw_t), bits(w_own)),
          check="tc_route_stage_bitwise", tag=tag)
    fin = torch.isfinite(lg_ref).all(1)
    dl = (lg_t - lg_ref)[fin].abs()
    flips, bad = [], []
    k64 = keys64(lg_ref, bias)
    for r in range(M):
        if not bool(fin[r]):
            continue
        if set(ti_t[r].tolist()) != set(i_ref[r].tolist()):
            srt_k = torch.sort(k64[r], descending=True).values
            gap = float(srt_k[7] - srt_k[8])
            flips.append(dict(row=r, gap=gap))
            if not gap < FLIP_GAP:
                bad.append(r)
    same = torch.tensor([bool(fin[r]) and set(ti_t[r].tolist()) == set(i_ref[r].tolist())
                         for r in range(M)], device=dev)
    rel = 0.0
    if same.any():
        a = torch.sort(ti_t[same], 1)
        b = torch.sort(i_ref[same], 1)
        wa = torch.gather(tw_t[same], 1, a.indices)
        wb = torch.gather(w_ref[same], 1, b.indices)
        rel = float(((wa - wb).abs() / wb.abs().clamp_min(1e-30)).max())
    check(not bad, check="tc_vs_incumbent", tag=tag,
          logit_max_abs=float(dl.max()) if dl.numel() else 0.0,
          set_flips=flips, nontie_flips=bad, w_max_rel=rel)
    check_align("tc/" + tag, ti_t, (srt_t, eids_t, ntpp_t))


def graph_test(rounds, dev):
    gen = torch.Generator().manual_seed(7)
    states = []
    for M in (1, 4, 8):
        for m in ("gate", "tc"):
            x, w, bias = make_inputs("random", M, gen, dev)

            def body(x=x, w=w, bias=bias, m=m):
                if m == "tc":
                    return rv.route_v2(bias, x=x, weight=w, renormalize=True,
                                       routed_scaling_factor=RSF)
                return rv.route_v2(bias, logits=gate_logits(x, w), renormalize=True,
                                   routed_scaling_factor=RSF)
            body()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = body()
            states.append((M, m, g, x, w, bias, out))
    bad = 0
    for step in range(rounds):
        for M, m, g, x, w, bias, out in states:
            x.copy_(torch.randn(M, K, generator=gen).to(torch.bfloat16))
            if step % 17 == 0:
                x.mul_(40)
            g.replay()
            lg, tw, ti, srt, eids, ntpp = out
            lg_ref = lg if m == "tc" else gate_logits(x, w)
            if m == "gate" and not torch.equal(bits(lg), bits(lg_ref)):
                bad += 1
                continue
            w_ref, i_ref = ref_route(x, lg_ref, bias)
            exp_s, exp_o = cpu_align(i_ref)
            n = int(ntpp.item())
            ok = (torch.equal(ti, i_ref) and torch.equal(bits(tw), bits(w_ref))
                  and n == len(exp_s) and srt[:n].tolist() == exp_s
                  and eids[:n // 8].tolist() == exp_o
                  and bool((srt[n:] == ti.numel()).all())
                  and bool((eids[n // 8:] == -1).all()))
            bad += not ok
    check(bad == 0, check="graph_alternating_replay", rounds=rounds,
          graphs=len(states), mismatches=bad)


def stream_test(dev):
    gen = torch.Generator().manual_seed(11)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    bad = 0
    for _ in range(50):
        jobs = []
        for s in (s1, s2):
            x, w, bias = make_inputs("random", 8, gen, dev)
            torch.cuda.current_stream().synchronize()
            with torch.cuda.stream(s):
                out = rv.route_v2(bias, x=x, weight=w, renormalize=True,
                                  routed_scaling_factor=RSF)
            jobs.append((s, x, bias, out))
        torch.cuda.synchronize()
        for s, x, bias, (lg, tw, ti, srt, eids, ntpp) in jobs:
            w_ref, i_ref = ref_route(x, lg, bias)
            exp_s, _ = cpu_align(i_ref)
            n = int(ntpp.item())
            bad += not (torch.equal(ti, i_ref) and srt[:n].tolist() == exp_s)
    check(bad == 0, check="two_streams_concurrent", iters=50, mismatches=bad)


class _Gate:
    """GateLinear stand-in: same weight attributes, same tier-3.5/4 output."""

    def __init__(self, w):
        self.weight, self.bias, self.out_dtype = w, None, torch.float32

    def __call__(self, t):
        return gate_logits(t, self.weight), None


def hook_test(dev):
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        GroupedTopKRouter)
    gen = torch.Generator().manual_seed(5)
    for M in (1, 4, 8):
        x, w, bias = make_inputs("random", M, gen, dev)
        router = GroupedTopKRouter(top_k=8, global_num_experts=E, num_expert_group=1,
                                   topk_group=1, renormalize=True,
                                   scoring_func="sigmoid", routed_scaling_factor=RSF,
                                   e_score_correction_bias=bias)
        gate_fn = _Gate(w)
        why = rv.reason(gate_fn, router, x)
        before = rv.stats()
        lg = rv.maybe_route(gate_fn, router, x)
        tw, ti = router.select_experts(x, lg, topk_indices_dtype=torch.int32)
        aligned = rv.take_align(ti, E)
        after = rv.stats()
        w_ref, i_ref = ref_route(x, gate_logits(x, w), bias)
        ok = (why is None and lg is not None and aligned is not None
              and torch.equal(ti, i_ref) and torch.equal(bits(tw), bits(w_ref))
              and after["routing_hit"] == before["routing_hit"] + 1)
        # in-place edit after the stash -> must miss (version counter)
        lg2 = rv.maybe_route(gate_fn, router, x)
        tw2, ti2 = router.select_experts(x, lg2, topk_indices_dtype=torch.int32)
        ti2.add_(0)
        miss = rv.take_align(ti2, E) is None
        check(ok and miss, check="hook_handoff", M=M, reason=why,
              aligned_hit=aligned is not None, inplace_edit_misses=miss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    emit(gpu=torch.cuda.get_device_name(), cap=torch.cuda.get_device_capability(),
         torch=torch.__version__)
    seeds = (0,) if a.quick else (0, 1, 2, 3)
    kinds = ("random", "exact_tie", "near_tie_eq", "near_tie_up", "near_tie_dn",
             "saturated", "nonfinite")
    for M in (1, 4, 8, 2, 16, 32):
        for kind in kinds:
            for seed in seeds:
                run_case(kind, M, seed, dev)
    graph_test(a.rounds, dev)
    stream_test(dev)
    hook_test(dev)
    emit(summary="FAIL" if FAIL else "PASS", failures=len(FAIL), stats=rv.stats())
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
