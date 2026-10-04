#!/usr/bin/env python3
"""Patch 0004 (adaptive DFlash depth): torch-free tests.

Runs the real patched code, not a mirror:
* ``vllm/v1/spec_decode/dynamic/adaptive_k.py`` is imported by path with a
  stub ``vllm.logger`` (its only vllm import);
* ``SpeculativeConfig._maybe_enable_glm5_load_adaptive_depth`` /
  ``_glm5_accept_depth_config`` and ``Scheduler._select_adaptive_k`` are
  extracted from the patched files with ``ast`` and run on stand-ins;
* the new ``envs.py`` entries are extracted and evaluated.

Policy cases are adapted from Morrowmake
``tests/v1/spec_decode/test_glm5_dflash_{adaptive_k,accept_depth}.py``.

Run: python3 tests/test_adaptive_k_pure.py   (or pytest)
"""

import ast
import importlib.util
import logging
import os
import pathlib
import random
import sys
import types
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
# GLM_DFLASH2_TREE: a directory holding the patched vllm/ package
# (source tree after scripts/apply-glm53-dflash2-patches.sh, or site-packages).
PATCHED = pathlib.Path(os.environ.get("GLM_DFLASH2_TREE", "/usr/local/lib/python3.12/dist-packages")) / "vllm"

# ---- load the real module ------------------------------------------------------
_vllm = types.ModuleType("vllm")
_vllm_logger = types.ModuleType("vllm.logger")
_vllm_logger.init_logger = logging.getLogger
sys.modules.setdefault("vllm", _vllm)
sys.modules["vllm.logger"] = _vllm_logger
_spec = importlib.util.spec_from_file_location(
    "adaptive_k", PATCHED / "v1/spec_decode/dynamic/adaptive_k.py"
)
ak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ak)
AdaptiveKConfig, AdaptiveKPolicy = ak.AdaptiveKConfig, ak.AdaptiveKPolicy


def _method(path: pathlib.Path, cls: str, name: str, ns: dict):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == name:
                    mod = ast.Module(body=[fn], type_ignores=[])
                    exec(compile(mod, str(path), "exec"), ns)
                    return ns[name]
    raise KeyError(f"{cls}.{name}")


def _env_entries(prefix: str) -> dict:
    tree = ast.parse((PATCHED / "envs.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", "") == "environment_variables"
        ):
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value.startswith(prefix):
                    out[k.value] = eval(
                        compile(ast.Expression(v), "envs", "eval"), {"os": os}
                    )
            return out
    raise KeyError("environment_variables")


ENV = _env_entries("VLLM_GLM5_DFLASH_ADAPTIVE_K")


class _Envs:
    def __getattr__(self, name):
        return ENV[name]()


class _Log:
    def __init__(self):
        self.messages = []

    def warning_once(self, msg, *args):
        self.messages.append(msg % args if args else msg)

    info = warning_once


SPEC_PY = PATCHED / "config/speculative.py"
_envs_mod = types.ModuleType("vllm.envs")
_envs_mod.__getattr__ = lambda name: ENV[name]()  # PEP 562
sys.modules["vllm.envs"] = _envs_mod
_vllm.envs = _envs_mod


def _rewrite(spec, log=None):
    log = log or _Log()
    ns = {"logger": log, "Any": object}
    enable = _method(SPEC_PY, "SpeculativeConfig", "_maybe_enable_glm5_load_adaptive_depth", ns)
    accept = _method(SPEC_PY, "SpeculativeConfig", "_glm5_accept_depth_config", dict(ns))
    spec._glm5_accept_depth_config = lambda by_load: accept(spec, by_load)
    enable(spec)
    del spec._glm5_accept_depth_config
    return spec, log


def _spec(pp=4, **kw):
    attrs = dict(
        method="dflash",
        num_speculative_tokens=3,
        adaptive_k=None,
        num_speculative_tokens_per_batch_size=None,
        enable_adaptive_verification=False,
        target_parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
    )
    attrs.update(kw)
    return SimpleNamespace(**attrs)


class _Env:
    """Context manager setting/unsetting env vars."""

    def __init__(self, **kv):
        self.kv = kv
        self.old = {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


ALL_OFF = {k: None for k in ENV}
FLAG = "VLLM_GLM5_DFLASH_ADAPTIVE_K"

# ---- envs ----------------------------------------------------------------------


def test_env_defaults_off():
    with _Env(**ALL_OFF):
        assert ENV[FLAG]() is False
        assert ENV[FLAG + "_DEPTHS"]() == "7,5"
        assert ENV[FLAG + "_ACCEPT"]() is False
        assert ENV[FLAG + "_COSTS"]() == ""
        assert ENV[FLAG + "_HYST"]() == 0.03
        assert ENV[FLAG + "_PRIOR"]() == 0.75
        assert ENV[FLAG + "_LOG"]() == 0


# ---- config rewrite --------------------------------------------------------------


def test_off_leaves_config_untouched():
    with _Env(**ALL_OFF):
        spec = _spec()
        before = dict(vars(spec))
        _rewrite(spec)
        assert vars(spec) == before


def test_on_drafts_deepest_and_verifies_by_load():
    with _Env(**{**ALL_OFF, FLAG: "1"}):
        spec, _ = _rewrite(_spec())
    assert spec.num_speculative_tokens == 7  # ring (0002) and buffers sized for 7
    assert spec.adaptive_k == {"by_load": [7, 5, 3], "log_interval": 0}
    cfg = AdaptiveKConfig.from_dict(spec.adaptive_k, 7)
    assert cfg.allowed == (3, 5, 7)
    assert [cfg.k_for_load(n) for n in range(0, 10)] == [7, 7, 5, 3, 3, 3, 3, 3, 3, 3]


def test_rewrite_idempotent():
    with _Env(**{**ALL_OFF, FLAG: "1"}):
        spec, _ = _rewrite(_spec())
        once = dict(vars(spec))
        _rewrite(spec)
        assert vars(spec) == once


def test_gate_closed_says_why():
    cases = [
        (dict(method="mtp"), "7,5", "needs the DFlash drafter"),
        (dict(adaptive_k={"x": 1}), "7,5", "already configured"),
        (dict(num_speculative_tokens_per_batch_size=[(1, 8, 3)]), "7,5", "per_batch_size"),
        (dict(enable_adaptive_verification=True), "7,5", "adaptive_verification"),
        (dict(num_speculative_tokens=None), "7,5", "not set"),
        (dict(), "3,2", "exceeds num_speculative_tokens=3"),
        (dict(num_speculative_tokens=7), "7,5", "exceeds num_speculative_tokens=7"),
        (dict(), "five", "is not a list"),
        (dict(), "0,5", "empty or < 1"),
    ]
    for overrides, depths, needle in cases:
        with _Env(**{**ALL_OFF, FLAG: "1", FLAG + "_DEPTHS": depths}):
            spec = _spec(**overrides)
            before = dict(vars(spec))
            _, log = _rewrite(spec)
            assert vars(spec) == before, overrides
            assert any("depth is off" in m and needle in m for m in log.messages), (
                needle, log.messages)


def test_accept_default_costs_by_layout():
    for pp, costs in ((1, [1.0, 1.08, 1.16, 1.24, 1.32]), (4, [1.0, 1.105, 1.21, 1.315, 1.42])):
        with _Env(**{**ALL_OFF, FLAG: "1", FLAG + "_ACCEPT": "1"}):
            spec, _ = _rewrite(_spec(pp=pp))
        assert spec.adaptive_k["accept"] == {"costs": costs, "hysteresis": 0.03}
        cfg = AdaptiveKConfig.from_dict(spec.adaptive_k, 7)
        assert cfg.accept and cfg.allowed == (3, 4, 5, 6, 7)


def test_accept_custom_and_bad_costs():
    env = {**ALL_OFF, FLAG: "1", FLAG + "_ACCEPT": "1",
           FLAG + "_COSTS": "1,1.077,1.154,1.231,1.308",
           FLAG + "_COSTS_MULTI": "1,1.04,1.08,1.12,1.16"}
    with _Env(**env):
        spec, _ = _rewrite(_spec())
    assert spec.adaptive_k["accept"] == {
        "costs": [1, 1.077, 1.154, 1.231, 1.308],
        "costs_multi": [1, 1.04, 1.08, 1.12, 1.16], "hysteresis": 0.03}
    with _Env(**{**env, FLAG + "_COSTS": "1,1.1"}):
        spec, log = _rewrite(_spec())
    assert "accept" not in spec.adaptive_k
    assert any("ACCEPT=1 set but off" in m for m in log.messages)
    with _Env(**{**ALL_OFF, FLAG + "_ACCEPT": "1"}):
        spec, log = _rewrite(_spec())
    assert spec.adaptive_k is None
    assert any("needs VLLM_GLM5_DFLASH_ADAPTIVE_K=1" in m for m in log.messages)


# ---- config object -----------------------------------------------------------------


def _raises(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"no ValueError ({needle})")


def test_config_validation():
    _raises(lambda: AdaptiveKConfig.from_dict({"by_load": []}, 5), "non-empty")
    _raises(lambda: AdaptiveKConfig.from_dict({"by_load": [6, 3]}, 5), "outside")
    _raises(lambda: AdaptiveKConfig.from_dict({"min": 1}, 5), "Unknown")
    _raises(lambda: AdaptiveKConfig.from_dict(
        {"by_load": [5, 4, 3], "accept": {"costs": [1.0, 1.1]}}, 5), "one positive cost")
    _raises(lambda: AdaptiveKConfig.from_dict(
        {"by_load": [5, 4, 3], "accept": {"costs": [1.0, 0.0, 1.2]}}, 5), "one positive cost")


def test_graph_ceiling_per_depth():
    cfg = AdaptiveKConfig.from_dict({"by_load": [7, 5, 3]}, 7)
    assert [cfg.max_reqs_for(k, 8) for k in (3, 5, 7)] == [8, 2, 1]
    assert cfg.max_reqs_for(4, 8) == 0  # never chosen without accept
    k7 = AdaptiveKConfig.from_dict(
        {"by_load": [7, 5, 3], "accept": {"costs": [1, 1.1, 1.2, 1.3, 1.4]}}, 7)
    assert [k7.max_reqs_for(k, 8) for k in (3, 4, 5, 6, 7)] == [8, 2, 2, 1, 1]
    deep = AdaptiveKConfig.from_dict(
        {"by_load": [7] * 8 + [3], "accept": {"costs": [1, 1.1, 1.2, 1.3, 1.4]}}, 7)
    assert [deep.max_reqs_for(k, 8) for k in (3, 4, 5, 6, 7)] == [8] * 5


# ---- policy (adapted from MM test_glm5_dflash_accept_depth.py) ----------------------

TP_COSTS = [1.0, 1.08, 1.16]
ACC = {"by_load": [5, 4, 3], "accept": {"costs": TP_COSTS, "hysteresis": 0.03}}
K7 = {"by_load": [7, 5, 3],
      "accept": {"costs": [1.0, 1.08, 1.16, 1.24, 1.32], "hysteresis": 0.03}}


def _policy(raw=ACC, k=5):
    return AdaptiveKPolicy(AdaptiveKConfig.from_dict(raw, k))


def _feed(policy, req, p, steps, rng, depth_of=lambda: 5):
    for _ in range(steps):
        k = depth_of()
        accepted = 0
        while accepted < k and rng.random() < p:
            accepted += 1
        policy.observe(req, k, accepted)


def test_expected_tokens():
    assert AdaptiveKPolicy.expected_tokens(0.0, 5) == 1.0
    assert abs(AdaptiveKPolicy.expected_tokens(0.5, 3) - 1.875) < 1e-9
    assert abs(AdaptiveKPolicy.expected_tokens(1.0, 3) - 4.0) < 0.04


def test_low_and_high_acceptance():
    p = _policy()
    _feed(p, "prose", 0.45, 60, random.Random(0))
    assert p.select_by_acceptance(["prose"], cap=5) == 3
    p = _policy()
    _feed(p, "code", 0.9, 60, random.Random(1), depth_of=lambda: 3)
    assert p.select_by_acceptance(["code"], cap=5) == 5


def test_load_width_caps_choice():
    p = _policy()
    _feed(p, "code", 0.95, 60, random.Random(2))
    assert p.select_by_acceptance(["code"], cap=4) == 4
    assert p.select_by_acceptance(["code"], cap=3) == 3


def test_new_request_starts_at_fixed_prior():
    p = _policy()
    assert abs(p.accept_rate("new") - 0.75) < 1e-12
    rng = random.Random(3)
    for i in range(20):
        _feed(p, f"old{i}", 0.4, 30, rng)
    assert abs(p.accept_rate("fresh") - 0.75) < 1e-12
    assert p.observed_acceptance() < 0.5


def test_hysteresis():
    p = _policy()
    p._acc_s["r"], p._acc_f["r"] = 72.0, 28.0
    first = p.select_by_acceptance(["r"], cap=5)
    p._acc_s["r"], p._acc_f["r"] = 73.0, 27.0
    assert p.select_by_acceptance(["r"], cap=5) == first
    p._acc_s["r"], p._acc_f["r"] = 20.0, 80.0
    assert p.select_by_acceptance(["r"], cap=5) == 3


def test_padding_and_forget():
    p = _policy()
    p.mark_padded("r")
    p.observe("r", 5, 0)
    assert "r" not in p._acc_s
    p.observe("r", 5, 5)
    assert p._acc_s["r"] == 5 and p._acc_f["r"] == 0
    p.forget("r")
    assert "r" not in p._acc_s and "r" not in p._acc_f


def test_k7_choice_stays_inside_captured_graphs():
    cfg = AdaptiveKConfig.from_dict(K7, 7)
    p = AdaptiveKPolicy(cfg)
    rng = random.Random(9)
    for prob in (0.2, 0.5, 0.7, 0.9, 0.99):
        for n in range(1, 9):
            reqs = [f"{prob}-{i}" for i in range(n)]
            for r in reqs:
                _feed(p, r, prob, 40, rng)
            k = p.want(n, reqs)
            assert k in cfg.allowed and k <= cfg.k_for_load(n)
            assert cfg.max_reqs_for(k, 8) >= n, (prob, n, k)


def test_multi_request_costs():
    raw = {"by_load": [7, 5, 3], "accept": {
        "costs": [1.0, 1.08, 1.16, 1.24, 1.32],
        "costs_multi": [1.0, 1.5, 2.0, 2.5, 3.0], "hysteresis": 0.0}}
    p = AdaptiveKPolicy(AdaptiveKConfig.from_dict(raw, 7))
    for r in ("a", "b"):
        p._acc_s[r], p._acc_f[r] = 95.0, 5.0
    assert p.select_by_acceptance(["a"], cap=5) == 5
    assert p.select_by_acceptance(["a", "b"], cap=5) == 3


def test_our_measured_costs_pick_expected_depths():
    """With costs derived from our PP4 numbers (README), prose-like acceptance
    (~0.55 per draft) stays at 3, counting-like (~0.99) goes to 7."""
    raw = {"by_load": [7, 5, 3], "accept": {"costs": [1.0, 1.077, 1.154, 1.231, 1.308]}}
    p = AdaptiveKPolicy(AdaptiveKConfig.from_dict(raw, 7))
    p._acc_s["prose"], p._acc_f["prose"] = 55.0, 45.0
    p._acc_s["count"], p._acc_f["count"] = 99.0, 1.0
    assert p.select_by_acceptance(["prose"], cap=7) == 3
    assert p.select_by_acceptance(["count"], cap=7) == 7


def test_identical_requests_identical_depths_after_unrelated_traffic():
    def run(policy, req, pattern):
        depths = []
        for limit in pattern:
            k = policy.select_by_acceptance([req], 7)
            depths.append(k)
            policy.observe(req, k, min(k, limit))
        policy.forget(req)
        return depths

    rng = random.Random(13)
    pattern = [rng.choice([0, 1, 2, 3, 5, 7, 7, 7]) for _ in range(80)]
    p = AdaptiveKPolicy(AdaptiveKConfig.from_dict(K7, 7))
    first = run(p, "a", pattern)
    for i in range(30):
        _feed(p, f"x{i}", rng.choice([0.2, 0.95]), 20, rng)
        p.select_by_acceptance([f"x{i}"], 7)
        p.select_by_acceptance([f"x{i}", f"y{i}"], 5)
    assert run(p, "b", pattern) == first
    assert len(set(first)) > 1
    assert run(AdaptiveKPolicy(AdaptiveKConfig.from_dict(K7, 7)), "c", pattern) == first


# ---- scheduler._select_adaptive_k (real code, fake scheduler) ------------------------

SCHED_PY = PATCHED / "v1/core/sched/scheduler.py"
_select = _method(SCHED_PY, "Scheduler", "_select_adaptive_k", {})


class _Req:
    def __init__(self, rid, drafts=7, eligible_at=0, prefill=False):
        self.request_id = rid
        self.spec_token_ids = list(range(11, 11 + drafts))
        self.next_decode_eligible_step = eligible_at
        self.is_prefill_chunk = prefill


class _Sched:
    def __init__(self, raw, running, waiting=0, step=1, k=7):
        self.adaptive_k = AdaptiveKPolicy(AdaptiveKConfig.from_dict(raw, k))
        self.running = running
        self.waiting = [None] * waiting
        self.skipped_waiting = []
        self.current_step = step
        self.num_spec_tokens = k

    def select(self):
        return _select(self)


BY_LOAD = {"by_load": [7, 5, 3]}


def test_sched_verifies_by_load_and_truncates_prefix():
    for n, k in ((1, 7), (2, 5), (3, 3), (8, 3)):
        reqs = [_Req(f"r{i}") for i in range(n)]
        s = _Sched(BY_LOAD, reqs)
        assert s.select() == k
        for r in reqs:
            assert r.spec_token_ids == list(range(11, 11 + k))  # prefix, unaltered


def test_sched_waiting_counts_as_load():
    s = _Sched(BY_LOAD, [_Req("a")], waiting=1)
    assert s.select() == 5
    s = _Sched(BY_LOAD, [_Req("a")], waiting=2)
    assert s.select() == 3


def test_sched_in_flight_requests_are_not_truncated():
    """PP + async: requests not eligible this step keep their drafts and do
    not cap the width (fix over MM, which truncated every running request)."""
    a = _Req("a", drafts=7, eligible_at=0)
    b = _Req("b", drafts=3, eligible_at=5)  # in flight
    s = _Sched(BY_LOAD, [a, b], step=1)
    assert s.select() == 5  # load 2
    assert len(a.spec_token_ids) == 5 and len(b.spec_token_ids) == 3


def test_sched_short_list_snaps_down():
    s = _Sched(BY_LOAD, [_Req("a", drafts=6)])
    assert s.select() == 5  # wants 7, carries 6 -> widest captured <= 6


def test_sched_climbs_back_when_load_drops():
    reqs = [_Req(f"r{i}") for i in range(3)]
    s = _Sched(BY_LOAD, reqs)
    assert s.select() == 3
    reqs[0].spec_token_ids = list(range(11, 18))  # fresh full-width drafts
    s.running = reqs[:1]
    assert s.select() == 7


def test_sched_no_eligible_requests():
    s = _Sched(BY_LOAD, [_Req("a", eligible_at=9)])
    assert s.select() == 7


# ---- losslessness ---------------------------------------------------------------------


def test_any_depth_sequence_gives_the_target_output():
    """Toy greedy speculative decoding: whatever prefix width the scheduler
    verifies, committed tokens are the target's (accepted drafts equal the
    target's tokens by construction; the bonus is the target's). Shows that
    adaptive depth can change only the number of steps, not the text."""
    rng = random.Random(42)
    target = [rng.randrange(1000) for _ in range(400)]

    def drafter(pos, width):
        return [t if rng.random() < 0.8 else -t - 1 for t in target[pos:pos + width]]

    def run(choose):
        out, steps = [], 0
        while len(out) < len(target) - 8:
            drafts = drafter(len(out), 7)[: choose(steps)]  # verified prefix
            acc = 0
            while acc < len(drafts) and drafts[acc] == target[len(out) + acc]:
                acc += 1
            out += drafts[:acc] + [target[len(out) + acc]]
            steps += 1
        return out, steps

    ref, _ = run(lambda s: 0)  # no speculation
    for choose in (lambda s: 3, lambda s: 7, lambda s: rng.choice([3, 4, 5, 6, 7])):
        out, _ = run(choose)
        n = min(len(out), len(ref))
        assert out[:n] == ref[:n]


# ---- the force-file diagnostic (old 0014) is not part of the production series ----

def test_force_file_diagnostic_absent():
    assert "VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE" not in (PATCHED / "envs.py").read_text()
    assert "_forced_depth" not in SCHED_PY.read_text()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
