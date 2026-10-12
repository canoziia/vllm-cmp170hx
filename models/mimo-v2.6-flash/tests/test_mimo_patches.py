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

# 0006: DFlash drafter under PP (aux hidden states, value scale).
src2 = mimo.read_text()
assert "supports_aux_hidden_states_over_pp = True" in src2
assert "**self.pack_local_aux_hidden_states(aux_hidden_states)," in src2
assert "SupportsEagle3" in omni.read_text()
dfl = (root / "vllm/model_executor/models/qwen3_dflash.py").read_text()
assert 'value_scale=dflash_config.get("attention_value_scale")' in dfl
assert "all_v.mul_(value_scale)" in dfl
assert '"VLLM_MIMO26_DFLASH_DRAFT_HEAD_FP8"' in dfl
assert "or spec_decode_needs_target_embed(vllm_config)" in src2

# 0007: MTP draft widths + measured width policy.
sched = (root / "vllm/v1/core/sched/scheduler.py").read_text()
assert "def _mimo26_plan_drafts(self, scheduler_output: SchedulerOutput) -> None:" in sched
assert "def _spec_verify_limit(self, request: Request, history_limits) -> int | None:" in sched
pol = (root / "vllm/v1/core/sched/mimo26_verification_policy.py").read_text()
assert "def mimo26_widths_enabled(vllm_config) -> bool:" in pol and "COST_MS = {" in pol
assert "def _mimo26_draft_widths(self, speculative_config) -> bool:" in (
    root / "vllm/v1/worker/gpu/cudagraph_utils.py").read_text()

# 0007 spec_k: MiMo-only module, gated call sites, generic layer untouched.
assert "def validate_request(extra_args, speculative_config) -> None:" in (
    root / "vllm/v1/core/sched/mimo26_spec_k.py").read_text()
assert "from vllm.v1.core.sched.mimo26_spec_k import validate_request" in (
    root / "vllm/v1/engine/input_processor.py").read_text()
assert "from vllm.v1.core.sched.mimo26_spec_k import draft_limit" in sched
for generic in ("vllm/sampling_params.py", "vllm/entrypoints/serve/engine/protocol.py"):
    assert "spec_k" not in (root / generic).read_text(), generic

# 0008: PP feedback receive posted when consumed.
ppu = (root / "vllm/v1/worker/gpu/pp_utils.py").read_text()
assert "def _post_recv(self, slot: PendingRecv) -> None:" in ppu
assert '"VLLM_PP_DEFERRED_FEEDBACK_RECV", "0") == "1"' in ppu
assert "def flush_deferred(self) -> None:" in ppu
assert "_flush_pp_deferred(self.worker)" in (
    root / "vllm/v1/executor/multiproc_executor.py").read_text()
assert (root / "vllm/v1/worker/gpu/model_runner.py").read_text().count(
    "self.pp_handler.flush_deferred()") == 3

prof = (root / "vllm/mimo26_opt_profile.py").read_text()
assert '"VLLM_MIMO26_MXFP4_DQ": "2"' in prof
assert '"VLLM_MIMO26_DIFFKV_PREFILL_BM": "128"' in prof and '"VLLM_MIMO26_DIFFKV_QK_SPLIT": "1"' in prof
assert '"VLLM_MIMO26_ADAPTIVE_WIDTH": "1"' in prof and '"VLLM_PP_DEFERRED_FEEDBACK_RECV": "1"' in prof

# 0009: MiMo MXFP4 decode kernels hook.
assert "_mimo26_mxfp4_decode.try_run(" in (
    root / "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py").read_text()
assert "_K, _NI = 4096, 2048" in (
    root / "vllm/model_executor/layers/fused_moe/mimo26_mxfp4_decode.py").read_text()

# 0010: diffkv prefill BLOCK_M.
assert '"VLLM_MIMO26_DIFFKV_PREFILL_BM", "0"' in (
    root / "vllm/v1/attention/ops/triton_unified_attention_diffkv.py").read_text()

# 0011: split-KV (3D) attention for MTP verification, on in the profile.
_diffkv = (root / "vllm/v1/attention/ops/triton_unified_attention_diffkv.py").read_text()
assert '"VLLM_MIMO26_DIFFKV_SPEC_3D", "0") == "1"' in _diffkv
assert "and window_size[0] < 0" in _diffkv
assert '"VLLM_MIMO26_DIFFKV_SPEC_3D": "1"' in prof

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
print("PASS 0001-0011 present; mimo_v2 / mimo_v2_omni / MTP / fp8+mxfp4 upstream hooks intact")
