#!/usr/bin/env python3
"""DeepSeek V4 DSpark verification widths: profile, gating and cohort policy.

usage: test_dsv4_verification.py SOURCE_TREE (patched checkout containing vllm/)
"""
import importlib, importlib.util, logging, os, sys, types
from pathlib import Path

root = Path(sys.argv[1]) / "vllm"
spec = importlib.util.spec_from_file_location("profile_under_test", root / "dsv41_opt_profile.py")
prof = importlib.util.module_from_spec(spec); spec.loader.exec_module(prof)

# Gate: V4 (both resolved architectures) and V4.1 only.
for arch in ("DeepseekV4ForCausalLM", "DeepseekV4ForConditionalGeneration", "DeepseekV41ForCausalLM"):
    assert prof.uses_dsv4_verification_widths([arch]), arch
assert not prof.uses_dsv4_verification_widths(["Qwen3ForCausalLM"])
assert not prof.uses_dsv4_verification_widths(None)

# Profile: opt-in only, never selects adaptive width by itself.
keys = ("VLLM_DSV41_VERIFICATION", "VLLM_DSV41_FAST_METADATA", "VLLM_DSV41_METADATA_GRAPHS",
        "VLLM_DSPARK_DRAFT_HEAD_FP8")
saved = dict(os.environ)
try:
    for name, want in (("off", None), ("v4-spec", "1")):
        os.environ.clear(); os.environ["VLLM_DSV41_OPT_PROFILE"] = name; prof.apply()
        assert all(os.environ.get(k) == want for k in keys), name
        assert "VLLM_PP_DECODE_COHORT_LIVE_LOAD" not in os.environ
        assert not any(k.startswith("VLLM_DSV41_DENSE") or k.startswith("VLLM_DSV4_MXFP4") for k in os.environ)
finally:
    os.environ.clear(); os.environ.update(saved)

# Native DSpark block: V4 drafts dspark_block_size tokens, not 3 MTP stages.
assert 'getattr(draft_hf_config, "dspark_block_size", None)' in (root / "config/speculative.py").read_text()

# Cohort policy: per-request choices collapse to one shared width.
pkg = types.ModuleType("sched_under_test"); pkg.__path__ = [str(root / "v1/core/sched")]
sys.modules["sched_under_test"] = pkg
mp = importlib.import_module("sched_under_test.dsv41_measured_policy")
fp = importlib.import_module("sched_under_test.dsv41_fixed_policy")
table = mp.COST_MS_DSV4_PP3
assert all(list(v) == sorted(v) for v in table.values())
policy = fp.FixedHistoryPolicy(logging.getLogger("t"), cost_ms=table, uniform=True)
reqs = [types.SimpleNamespace(request_id=f"r{i}", verification_draft_limit=None) for i in range(4)]
for r in reqs:
    policy.admit(r)
for step in range(40):
    for i, r in enumerate(reqs):
        e = policy.active[r.request_id]
        acc = min(e.k, (step + i) % 2)  # low-acceptance (prose-like) stream
        e.plugin.observe(dict(k=e.k, accepted=acc, seconds=mp.costs(4, table)[e.k - 1],
                              output_tokens=acc + 1, timing_valid=True, load_epoch=1))
        e.fresh = True
    widths = policy.choose(reqs, 4)
assert len(set(widths.values())) == 1 and widths["r0"] < 5, widths
print("PASS gating, v4-spec profile, native block, uniform cohort width")
