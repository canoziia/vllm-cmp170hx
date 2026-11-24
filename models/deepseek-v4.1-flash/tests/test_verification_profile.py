#!/usr/bin/env python3
"""Profile defaults do not silently enable the new execution paths, and auto selects by model + parallelism."""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

p = Path(sys.argv[1]) / "vllm/dsv41_opt_profile.py"
s = importlib.util.spec_from_file_location("profile_under_test", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)

keys = (
    "VLLM_DSV41_VERIFICATION",
    "VLLM_PP_DECODE_COHORT_LIVE_LOAD",
    "VLLM_DSV41_FAST_METADATA",
    "VLLM_DSV41_METADATA_GRAPHS",
)
for k in keys:
    assert k not in m.PROFILES["default"]

original_env = dict(os.environ)
original_argv = list(sys.argv)
try:
    for profile in ("", "0", "off", "default", "verification"):
        os.environ.clear()
        os.environ["VLLM_DSV41_OPT_PROFILE"] = profile
        m.apply()
        for k in keys:
            assert os.environ.get(k) == ("1" if profile == "verification" else None), (
                profile,
                k,
            )
        assert "VLLM_DSV41_HISTORY_POLICY" not in os.environ
        assert all("enable_adaptive_verification" not in k.lower() for k in os.environ)

    # Explicit component override is retained
    os.environ.clear()
    os.environ["VLLM_DSV41_OPT_PROFILE"] = "verification"
    os.environ[keys[0]] = "0"
    m.apply()
    assert os.environ[keys[0]] == "0"

    with tempfile.TemporaryDirectory() as td:
        dsv41_dir = Path(td) / "my-model-a"
        qwen_dir = Path(td) / "my-model-b"
        dsv41_dir.mkdir()
        qwen_dir.mkdir()
        (dsv41_dir / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]})
        )
        (qwen_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
        )

        # Non-DeepSeek model: auto is completely inert
        os.environ.clear()
        os.environ["VLLM_DSV41_OPT_PROFILE"] = "auto"
        sys.argv = ["vllm", "serve", str(qwen_dir), "-tp", "4", "--enable-expert-parallel"]
        m.apply()
        assert os.environ["VLLM_DSV41_OPT_PROFILE"] == "auto"
        assert "VLLM_DSV41_FUSED_W2_AR" not in os.environ
        assert "VLLM_DSV41_DENSE_MXFP8" not in os.environ

        assert m.resolve_auto_profile(["vllm", "serve", "Qwen/Qwen3-32B", "-tp", "4"]) is None

        # DeepSeek-V4.1 TEP4 -> tep4_fused, propagated for spawned workers
        os.environ.clear()
        os.environ["VLLM_DSV41_OPT_PROFILE"] = "auto"
        os.environ["VLLM_DSV41_WOA_FP8"] = "0"  # explicit user override must survive
        sys.argv = [
            "vllm",
            "serve",
            str(dsv41_dir),
            "--tensor-parallel-size=4",
            "--enable-expert-parallel",
        ]
        m.apply()
        assert os.environ["VLLM_DSV41_OPT_PROFILE"] == "tep4_fused"
        assert os.environ["VLLM_DSV41_FUSED_W2_AR"] == "1"
        assert os.environ["VLLM_DSV41_WOA_FP8"] == "0"

        # Spawned worker with different argv inherits resolved tep4_fused
        sys.argv = ["python3", "-c", "from multiprocessing.spawn import spawn_main"]
        m.apply()
        assert os.environ["VLLM_DSV41_OPT_PROFILE"] == "tep4_fused"
        assert os.environ["VLLM_DSV41_FUSED_W2_AR"] == "1"

        # DeepSeek-V4.1 PP6 -> default / verification
        assert m.resolve_auto_profile(["vllm", "serve", str(dsv41_dir), "-pp", "6"]) == "default"
        assert (
            m.resolve_auto_profile(
                [
                    "vllm",
                    "serve",
                    str(dsv41_dir),
                    "--pipeline-parallel-size=6",
                    '--speculative-config={"enable_adaptive_verification":true}',
                ]
            )
            == "verification"
        )

        # DeepSeek-V4.1 TP8 -> tp_auto
        assert (
            m.resolve_auto_profile(["vllm", "serve", "deepseek-ai/DeepSeek-V4.1-Flash", "-tp", "8"])
            == "tp_auto"
        )
finally:
    os.environ.clear()
    os.environ.update(original_env)
    sys.argv[:] = original_argv

print("PASS default/off inert; verification opt-in; auto model+parallelism routing; explicit override retained")
