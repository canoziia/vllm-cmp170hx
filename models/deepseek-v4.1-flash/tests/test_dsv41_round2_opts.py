#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline build-time validation for DeepSeek V4.1 round-2 optimisations."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    root = Path(sys.argv[1])
    orig_env = dict(os.environ)
    try:
        os.environ.clear()
        sched = _load_module(
            "dsv41_prefill_sched_under_test",
            root / "vllm/v1/core/sched/dsv41_prefill_sched.py",
        )
        # 1. Default-off inertness
        assert sched.yoco_prefill_trunc_enabled() is False
        assert sched.cost_prefill_chunk_enabled() is False
        helper = sched.Dsv41PrefillHelper(block_size=128)
        assert helper.maybe_cap_prefill_chunk(262144, 4096, 4096) == 4096

        # 2. Quadratic cost-budget chunk solver when enabled
        os.environ["VLLM_DSV41_COST_PREFILL_CHUNK"] = "1"
        assert helper.maybe_cap_prefill_chunk(0, 4096, 4096) == 4096
        q131 = helper.maybe_cap_prefill_chunk(131072, 16384, 4096)
        q524 = helper.maybe_cap_prefill_chunk(524288, 16384, 4096)
        assert 4096 > q131 > q524 >= 512
        assert q131 % 128 == 0 and q524 % 128 == 0

        # 3. YoCo prefix-cache truncation tracking & backoff
        os.environ["VLLM_DSV41_YOCO_PREFILL_TRUNC"] = "1"
        hashes = [f"h{i}" for i in range(128)]  # 16,384 tokens
        req = SimpleNamespace(
            num_prompt_tokens=16384,
            block_hashes=hashes,
            sampling_params=SimpleNamespace(prompt_logprobs=None),
        )
        for c in range(0, 16384, 4096):
            helper.record_scheduled_prefill(req, c, 4096)

        kvm = SimpleNamespace(
            truncate_computed_blocks=lambda blks, n: blks[: n // 128]
        )
        # End-of-turn hit requires zero backoff
        _, loc_end, ext_end, _ = helper.adjust_prefix_cache_hit(
            req, kvm, list(range(128)), 16384, 0, False
        )
        assert loc_end + ext_end == 16384
        # Mid-prompt hit at 8,192 backs off by 2,688 tokens
        _, loc_mid, ext_mid, _ = helper.adjust_prefix_cache_hit(
            req, kvm, list(range(64)), 8192, 0, False
        )
        assert loc_mid + ext_mid == 8192 - 2688

        # 4. Source tree structural assertions for new opt-in switches
        profile_src = (root / "vllm/dsv41_opt_profile.py").read_text()
        for sw in (
            "VLLM_DSV41_EXPANDABLE_AFTER_KV",
            "VLLM_DSV41_DECODE_LOGITS_CHUNK",
            "VLLM_DSV41_INDEXER_DECODE_SPARSE",
            "VLLM_DSV41_YOCO_PREFILL_TRUNC",
            "VLLM_DSV41_RELAY_SHADOW_CANDIDATES",
            "VLLM_DSV41_COST_PREFILL_CHUNK",
            "VLLM_DSV41_SPARSE_PREFILL_64H",
            "VLLM_DSV41_ENGRAM_ASYNC_PREFETCH",
        ):
            assert sw in profile_src, f"Missing switch in dsv41_opt_profile.py: {sw}"
        mr_src = (root / "vllm/v1/worker/gpu/model_runner.py").read_text()
        assert "cu_num_logits_np[-1]" not in mr_src, (
            "YoCo prefill check must not gate on cu_num_logits_np[-1] == 0 in V1 GPUModelRunner"
        )
        nv_model_src = (
            root / "vllm/models/deepseek_v41/nvidia/model.py"
        ).read_text()
        assert "VLLM_DSV41_PREFILL_AR_OVERLAP" in nv_model_src
        sp_src = (
            root / "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
        ).read_text()
        assert "cu_num_logits_np[-1]" not in sp_src
    finally:
        os.environ.clear()
        os.environ.update(orig_env)
    print("PASS DeepSeek V4.1 round-2 optimisation offline checks")


if __name__ == "__main__":
    main()
