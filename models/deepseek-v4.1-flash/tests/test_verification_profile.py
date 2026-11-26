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

        # DeepSeek-V4.1 Pure TP4 (default) -> tp4_fused, propagated for spawned workers
        os.environ.clear()
        os.environ["VLLM_DSV41_OPT_PROFILE"] = "auto"
        os.environ["VLLM_DSV41_WOA_FP8"] = "0"  # explicit user override must survive
        sys.argv = [
            "vllm",
            "serve",
            str(dsv41_dir),
            "--tensor-parallel-size=4",
        ]
        m.apply()
        assert os.environ["VLLM_DSV41_OPT_PROFILE"] == "tp4_fused"
        assert os.environ["VLLM_DSV41_COST_MS_FUSED"] == "tp4"
        assert os.environ["VLLM_DSV41_DEFER_FINALIZE_MIN_TOKENS"] == "1"
        assert os.environ["VLLM_DSV41_FUSED_W2_AR"] == "0"
        assert os.environ["VLLM_DSV41_WOA_FP8"] == "0"

        # Spawned worker with different argv inherits resolved tp4_fused
        sys.argv = ["python3", "-c", "from multiprocessing.spawn import spawn_main"]
        m.apply()
        assert os.environ["VLLM_DSV41_OPT_PROFILE"] == "tp4_fused"
        assert os.environ["VLLM_DSV41_COST_MS_FUSED"] == "tp4"

        # DeepSeek-V4.1 legacy TEP4 (--enable-expert-parallel) -> tep4_fused
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
        assert os.environ["VLLM_DSV41_COST_MS_FUSED"] == "1"
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

        # DeepSeek-V4.1 TP8 -> tp_auto, and >=6 sharded GPUs disables vision/embed CPU offload
        assert (
            m.resolve_auto_profile(["vllm", "serve", "deepseek-ai/DeepSeek-V4.1-Flash", "-tp", "8"])
            == "tp_auto"
        )

        # Vision/embed CPU offload policy: enabled for <6 weight-sharding GPUs, disabled for >=6
        offload_cases = [
            (["vllm", "serve", str(dsv41_dir), "-tp", "4"], "tp4_fused", "1"),
            (["vllm", "serve", str(dsv41_dir), "-tp", "4", "-dp", "2"], "tp4_fused", "1"),
            (["vllm", "serve", str(dsv41_dir), "-tp", "2"], "tp_auto", "1"),
            (
                ["vllm", "serve", str(dsv41_dir), "-tp", "2", "-dp", "3", "--enable-expert-parallel"],
                "tp_auto",
                "0",
            ),
            (["vllm", "serve", str(dsv41_dir), "-tp", "8"], "tp_auto", "0"),
            (
                ["vllm", "serve", str(dsv41_dir), "-tp", "4", "-dp", "2", "--enable-expert-parallel"],
                "tp_auto",
                "0",
            ),
            (["vllm", "serve", str(dsv41_dir), "-pp", "6"], "default", "0"),
        ]
        for argv, expected_prof, expected_offload in offload_cases:
            os.environ.clear()
            os.environ["VLLM_DSV41_OPT_PROFILE"] = "auto"
            sys.argv = list(argv)
            m.apply()
            assert os.environ["VLLM_DSV41_OPT_PROFILE"] == expected_prof, (argv, os.environ["VLLM_DSV41_OPT_PROFILE"])
            assert os.environ.get("VLLM_DSV41_OFFLOAD_VISION_EMBED") == expected_offload, (
                argv,
                os.environ.get("VLLM_DSV41_OFFLOAD_VISION_EMBED"),
            )
            # Spawned worker must inherit parent's offload decision unchanged
            sys.argv = ["python3", "-c", "from multiprocessing.spawn import spawn_main"]
            m.apply()
            assert os.environ.get("VLLM_DSV41_OFFLOAD_VISION_EMBED") == expected_offload, argv

        # Explicit user override of VLLM_DSV41_OFFLOAD_VISION_EMBED always survives
        os.environ.clear()
        os.environ["VLLM_DSV41_OPT_PROFILE"] = "auto"
        os.environ["VLLM_DSV41_OFFLOAD_VISION_EMBED"] = "1"
        sys.argv = ["vllm", "serve", str(dsv41_dir), "-tp", "2", "-dp", "3", "--enable-expert-parallel"]
        m.apply()
        assert os.environ["VLLM_DSV41_OFFLOAD_VISION_EMBED"] == "1"

        os.environ.clear()
        os.environ["VLLM_DSV41_OPT_PROFILE"] = "auto"
        os.environ["VLLM_DSV41_OFFLOAD_VISION_EMBED"] = "0"
        sys.argv = ["vllm", "serve", str(dsv41_dir), "-tp", "4"]
        m.apply()
        assert os.environ["VLLM_DSV41_OFFLOAD_VISION_EMBED"] == "0"

        # NCCL_PROTO=Simple only set on switch-connected PCIe (PIX/PXB), not cross-RC (PHB/NODE/SYS)
        orig_nvl = m._has_active_nvlink_nvml
        orig_sw = m.pcie_switch_connected_nvml
        try:
            m._has_active_nvlink_nvml = lambda: False
            m.pcie_switch_connected_nvml = lambda device_ids=None: (True, 10)  # PIX
            os.environ.clear()
            os.environ["VLLM_DSV41_OPT_PROFILE"] = "tep4_fused"
            m.apply()
            assert os.environ.get("NCCL_PROTO") == "Simple"
            assert os.environ.get("VLLM_DSV41_PREFILL_DMA_RING") == "auto"
            assert os.environ.get("VLLM_DSV41_PREFILL_ROW_SHARD") == "auto"
            assert os.environ.get("VLLM_DSV41_PREFILL_2MB_MOE") == "auto"

            m.pcie_switch_connected_nvml = lambda device_ids=None: (False, 40)  # NODE
            os.environ.clear()
            os.environ["VLLM_DSV41_OPT_PROFILE"] = "tep4_fused"
            m.apply()
            assert "NCCL_PROTO" not in os.environ

            os.environ.clear()
            os.environ["VLLM_DSV41_OPT_PROFILE"] = "tep4_fused"
            os.environ["VLLM_DSV41_AMPERE_PCIE_COLLECTIVES"] = "force"
            m.apply()
            assert os.environ.get("NCCL_PROTO") == "Simple"
        finally:
            m._has_active_nvlink_nvml = orig_nvl
            m.pcie_switch_connected_nvml = orig_sw
finally:
    os.environ.clear()
    os.environ.update(original_env)
    sys.argv[:] = original_argv

print("PASS default/off inert; verification opt-in; auto model+parallelism routing; explicit override retained")
