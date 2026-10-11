#!/usr/bin/env python3
"""CPU checks for the Qwen3.8 measured MTP verification width policy.

  python3 test_mtp_width_policy.py SOURCE_TREE [K]

Simulated feedback with measured per-position acceptance (counting, code,
prose) and the policy's own step-cost table: high-acceptance workloads must
keep full K at every concurrency; low-acceptance prose at high concurrency
trims; growth is at most one width per decision; a failing policy keeps K; a
batch shares the maximum width; manual spec_k requests are not managed.
"""
import importlib.util
import logging
import random
import sys
import types

path = f"{sys.argv[1]}/vllm/v1/core/sched/qwen38_verification_policy.py"
spec = importlib.util.spec_from_file_location("qwen38_policy", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
log = logging.getLogger("test")


def req(rid, limit=None):
    return types.SimpleNamespace(
        request_id=rid, verification_draft_limit=limit, is_prefill_chunk=False
    )


def simulate(cum, conc, steps=400, seed=0):
    random.seed(seed)
    policy = m.MeasuredWidthPolicy(log, K)
    r = req("a")
    policy.admit(r)
    t, widths, last = 0.0, [], K
    cost = m.costs(conc, K)
    for _ in range(steps):
        k = policy.next_width([r], conc)
        assert k <= last + 1, "growth above one width"
        last = k
        widths.append(k)
        u = random.random()
        accepted = sum(1 for j in range(k) if u < cum[j])
        out = types.SimpleNamespace(scheduled_spec_decode_tokens={"a": [0] * k})
        policy.scheduled(out, [r], conc)
        t += cost[k - 1] * (1 + random.gauss(0, 0.05))
        policy.feedback_time = t
        policy.feedback(out, r, k, accepted, accepted + 1, True)
    return widths


ok = True
HIGH = {"counting": [0.99, 0.98, 0.97, 0.96, 0.95], "code": [0.96, 0.89, 0.84, 0.78, 0.72]}
for name, cum in ((n, c[:K]) for n, c in HIGH.items()):
    for conc in (1, 8, 32):
        w = simulate(cum, conc)
        good = w.count(K) >= 0.99 * len(w)
        print(("PASS" if good else "FAIL"), f"{name} c{conc} keeps K ({w.count(K)}/{len(w)})")
        ok &= good
w = simulate([0.61, 0.38, 0.19, 0.10, 0.05][:K], 32)
good = sum(x < K for x in w[100:]) > 0.8 * len(w[100:])
print(("PASS" if good else "FAIL"), f"prose c32 trims (narrow {sum(x < K for x in w[100:])}/{len(w[100:])})")
ok &= good

policy = m.MeasuredWidthPolicy(log, K)
a, b, manual = req("a"), req("b"), req("m", limit=1)
for r in (a, b, manual):
    policy.admit(r)
policy.active["a"].k, policy.active["b"].k = 1, 2
width = policy.next_width([a, b], 8)
good = width == 2 and "m" not in policy.active and not policy.manages(manual)
print(("PASS" if good else "FAIL"), "batch drafts the maximum width; manual spec_k unmanaged")
ok &= good

entry = policy.active["a"]
entry.fresh = True
entry.plugin.choose = lambda ctx: (_ for _ in ()).throw(RuntimeError("boom"))
width = policy.next_width([a], 8)
good = width == K and entry.failed
print(("PASS" if good else "FAIL"), "policy failure keeps full K")
ok &= good

print("QWEN38_MTP_WIDTH_POLICY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
