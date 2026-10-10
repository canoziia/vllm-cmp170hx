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

for f in (mimo, omni, root / "vllm/model_executor/models/mimo_v2_mtp.py"):
    py_compile.compile(str(f), doraise=True)
print("PASS 0001 present; mimo_v2 / mimo_v2_omni / MTP / fp8+mxfp4 upstream hooks intact")
