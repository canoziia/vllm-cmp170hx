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

# Profile: opt-in only (off sets nothing); dsv4-flash carries the speculative
# switches and never selects adaptive width by itself (that is the
# speculative-config's enable_adaptive_verification).
keys = ("VLLM_DSV41_VERIFICATION", "VLLM_DSV41_FAST_METADATA", "VLLM_DSV41_METADATA_GRAPHS",
        "VLLM_DSPARK_DRAFT_HEAD_FP8", "VLLM_DSV41_MOE_PREFILL_SPLIT", "VLLM_DSV4_SPARSE_PREFILL_H32",
        "VLLM_DSV4_SPARSE_PREFILL_REQS")
assert "v4-spec" not in prof.PROFILES
saved = dict(os.environ)
try:
    for name, want in (("off", None), ("dsv4-flash", "1")):
        os.environ.clear(); os.environ["VLLM_DSV41_OPT_PROFILE"] = name; prof.apply()
        assert all(os.environ.get(k) == want for k in keys), name
        assert not any("ADAPTIVE" in k for k in os.environ), name
        if name == "off":
            assert set(os.environ) == {"VLLM_DSV41_OPT_PROFILE"}, sorted(os.environ)
finally:
    os.environ.clear(); os.environ.update(saved)

# Prefill: the MoE split accepts V4's expert shapes.
assert "(256,4096,2048)" in (root / "model_executor/layers/fused_moe/moe_prefill.py").read_text()

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
print("PASS gating, dsv4-flash profile, native block, uniform cohort width")
