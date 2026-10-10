#!/usr/bin/env python3
"""Static checks of the pieces MiMo-V2.6 relies on in a patched source tree.

Usage: test_mimo_patches.py SOURCE_TREE
"""
import pathlib
import py_compile
import sys

root = pathlib.Path(sys.argv[1])
mimo = root / "vllm/model_executor/models/mimo_v2.py"
omni = root / "vllm/model_executor/models/mimo_v2_omni.py"

# 0001: fused FP8 qkv_proj is stored as export-TP row shards [Q_s|K_s|V_s]
# with per-shard padded block scales; main model and MTP both reshard it.
src = mimo.read_text()
assert "def mimo_v2_qkv_checkpoint_shards(config) -> int:" in src
assert "ckpt_shards=mimo_v2_qkv_checkpoint_shards(self.config)," in src
assert "scale_rows_per_group" not in src
mtp = (root / "vllm/model_executor/models/mimo_v2_mtp.py").read_text()
assert "ckpt_shards=mimo_v2_qkv_checkpoint_shards(cfg)," in mtp
assert "loaded_weight.chunk(tp_size, dim=0)[tp_rank]" not in mtp

# 0002: VLLM_MIMO26_MTP_LAYERS loads several MTP layers (multi-module MTP).
spec = (root / "vllm/config/speculative.py").read_text()
assert "def _mimo_v2_num_mtp_layers(hf_config, default: int) -> int:" in spec
assert 'num_mtp_layers = getattr(config, "num_nextn_predict_layers", None) or 1' in mtp
assert "def use_multi_module_mtp(self) -> bool:" in spec
assert "return out, previous_hidden_states" in mtp

# 0003: VLLM_MIMO26_OPT_PROFILE applied from env_override.
prof = (root / "vllm/mimo26_opt_profile.py").read_text()
assert '"VLLM_MIMO26_MTP_LAYERS": "3"' in prof
assert '"VLLM_DSV4_FAST_DET_MOE_ALIGN": "1"' in prof
assert '"VLLM_MIMO26_SORT_MOE_ALIGN": "1"' in prof
assert '"VLLM_MIMO26_DRAFT_HEAD_FP8": "1"' in prof

# 0004: single-sort MoE align (bitwise test: tests/test_sort_moe_align_gpu.py).
align = (root / "vllm/model_executor/layers/fused_moe/fast_det_align.py").read_text()
assert "def _sort_align_kernel(" in align and "def sort_enabled() -> bool:" in align
assert "_mimo26_apply_opt_profile()" in (root / "vllm/env_override.py").read_text()

# 0005: fp8 draft head built after the target lm_head is shared.
assert "def after_target_sharing(self) -> None:" in (
    root / "vllm/model_executor/models/mimo_v2_mtp.py").read_text()
assert 'def enabled(switch: str = "VLLM_MIMO26_DRAFT_HEAD_FP8") -> bool:' in (
    root / "vllm/model_executor/layers/fp8_draft_head.py").read_text()
assert 'getattr(self.model, "after_target_sharing", None)' in (
    root / "vllm/v1/worker/gpu/spec_decode/speculator.py").read_text()

# Upstream pieces the deployment relies on (fail loudly if a rebase drops them).
assert '"MiMoV2ForCausalLM": ("mimo_v2", "MiMoV2ForCausalLM")' in (
    root / "vllm/model_executor/models/registry.py"
).read_text()
# MiMo-V2.6 checkpoints (vision_config present) resolve to the Omni class.
assert '"MiMoV2OmniForCausalLM": ("mimo_v2_omni", "MiMoV2OmniForCausalLM")' in (
    root / "vllm/model_executor/models/registry.py"
).read_text()
assert '"mimo_v2_mtp"' in (root / "vllm/config/speculative.py").read_text()
assert 'store_dtype == "mxfp4"' in (
    root / "vllm/model_executor/layers/quantization/fp8.py"
).read_text()

for f in (mimo, omni, root / "vllm/model_executor/models/mimo_v2_mtp.py",
          root / "vllm/mimo26_opt_profile.py"):
    py_compile.compile(str(f), doraise=True)
print("PASS 0001-0005 present; mimo_v2 / mimo_v2_omni / MTP / fp8+mxfp4 upstream hooks intact")
