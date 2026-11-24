# SPDX-License-Identifier: Apache-2.0
"""One switch for the DeepSeek V4.1 Flash SM80 optimisation patches.

VLLM_DSV41_OPT_PROFILE=default|verification|tep4|tep4_verification sets every
per-patch switch to its deployed value *unless that variable is already set*,
so a single switch can still be overridden.
Unset / "0" / "off": profile defaults stay off (except ensuring 1x-HBM MXFP8
emulation dequant opt-out when dense MXFP8 decode is explicitly enabled).
Applied from vllm.env_override before any other vLLM module is imported, in
the API server and in every spawned worker.
"""

import os

PROFILES: dict[str, dict[str, str]] = {
    "default": {
        # 0017 expandable allocator segments after KV IPC registration
        "VLLM_DSV41_EXPANDABLE_AFTER_KV": "1",
        # 0018 bounded decode indexer workspace + candidate-sparse decode scoring
        "VLLM_DSV41_DECODE_LOGITS_CHUNK": "1",
        "VLLM_DSV41_INDEXER_DECODE_SPARSE": "1",
        # 0019 Engram async stream overlap, NUMA binding, packed host rows & CPU gather
        "VLLM_DSV41_ENGRAM_NUMA_BIND": "1",
        "VLLM_DSV41_ENGRAM_PACKED_HOST": "1",
        "VLLM_DSV41_ENGRAM_PACKED_STRIDE": "264",
        "VLLM_DSV41_ENGRAM_ASYNC_PREFETCH": "1",
        "VLLM_DSV41_ENGRAM_PREFETCH_SMS": "4",
        "VLLM_DSV41_ENGRAM_CPU_PREFETCH": "1",
        "VLLM_DSV41_ENGRAM_CPU_COMPACT_H2D": "1",
        # 0020 fused wq_b+indexer.wq_b projection + bitwise fast MoE router
        "VLLM_DSV41_FUSED_WQB_IDX": "1",
        "VLLM_DSV41_FAST_GATE_TOPK": "1",
        # 0021 32H sparse MLA prefill + fast prefill writer & compact reader indexer
        "VLLM_DSV41_SPARSE_PREFILL_64H": "1",
        "VLLM_DSV41_SPARSE_PREFILL_REQS": "1",
        "VLLM_DSV41_INDEXER_WRITER_FAST": "1",
        "VLLM_DSV41_INDEXER_READER_COMPACT": "1",
        # 0022 exact YoCo consumer-layer prefill truncation, shadow relay & cost chunking
        "VLLM_DSV41_YOCO_PREFILL_TRUNC": "1",
        "VLLM_DSV41_RELAY_SHADOW_CANDIDATES": "1",
        "VLLM_DSV41_COST_PREFILL_CHUNK": "1",
        # 0003 single-kernel deterministic MoE align
        "VLLM_DSV4_FAST_DET_MOE_ALIGN": "1",
        # 0004 weight-streaming MXFP4 MoE decode
        "VLLM_DSV4_MXFP4_DECODE": "1",
        # 0005 faster sm80 sparse-attention decode
        "VLLM_DSV4_SPARSE_DECODE_FAST": "1",
        # 0006 fused small decode kernels
        "VLLM_DSV4_ATTN_DIRECT_OUT": "1",
        "VLLM_DSV41_TOPK_RAGGED_FUSED": "1",
        "VLLM_DSV41_SWA_RAGGED_INPLACE": "1",
        "VLLM_DSV41_INDEXER_Q_LUT_FUSED": "1",
        # 0007 Gluon mHC decode v2
        "VLLM_DSV4_MHC_V2": "1",
        # 0008 fused MoE activation / top-k sum, CUDA deterministic align
        "VLLM_DSV4_MXFP4_DQ": "2",
        "VLLM_DSV4_MXFP4_FUSED_ACT": "1",
        "VLLM_DSV4_MXFP4_FUSED_SUM": "1",
        "VLLM_DSV4_MXFP4_FUSED_SUM_ORDER": "1",
        "VLLM_DSV4_CUDA_DET_ALIGN": "1",
        # 0009 fp8 draft head
        "VLLM_DSPARK_DRAFT_HEAD_FP8": "1",
        # 0010 thin-M wo_a GEMM
        "VLLM_DSV41_THIN_WOA": "1",
        # 0011 MXFP4 MoE prefill split-list
        "VLLM_DSV41_MOE_PREFILL_SPLIT": "1",
        # 0012 fast prefill indexer
        "VLLM_DSV41_INDEXER_PREFILL_FAST": "1",
        # 0013 1x-HBM dense MXFP8 decode GEMM
        "VLLM_DSV41_DENSE_MXFP8": "1",
        "VLLM_DSV41_DENSE_MXFP8_DECODE": "1",
        "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD": "0",
        "VLLM_DSV41_DENSE_MXFP8_LIB": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_dsv41_dense_mxfp8.so"
        ),
    },
}

# Opt-in extension; the deployed default profile remains unchanged.
PROFILES["verification"] = {
    **PROFILES["default"],
    "VLLM_DSV41_VERIFICATION": "1",
    "VLLM_DSV41_MIXED_SPEC_FULL_GRAPH": "1",
    "VLLM_PP_DECODE_COHORT_LIVE_LOAD": "1",
    "VLLM_DSV41_FAST_METADATA": "1",
    "VLLM_DSV41_METADATA_GRAPHS": "1",
}

PROFILES["tep4_dcp4"] = {
    **PROFILES["default"],
    "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD": "0",
    "VLLM_DSV41_DENSE_MXFP8": "1",
    "VLLM_DSV41_DENSE_MXFP8_DECODE": "1",
    "VLLM_DSV41_ENGRAM_NUMA_INTERLEAVE": "1",
    "VLLM_DSV41_WOA_FP8": "1",
    "VLLM_DSV41_OFFLOAD_VISION_EMBED": "1",
    "VLLM_DSV41_MARLIN_CLEAN_REPACK": "1",
    "VLLM_DSV41_COST_PREFILL_CHUNK": "0",
    "VLLM_DSV41_YOCO_PARTIAL_FFN": "1",
    "VLLM_DSV41_TOPK_RAGGED_REUSE": "1",
    "VLLM_DSV41_PREFILL_AR_OVERLAP": "1",
    "VLLM_DSV41_DECODE_REQ_SIZES": "1,2,4,8,16,24,32",
    "VLLM_DSV41_MIXED_SPEC_TOKENS": "16,24,32,40,48,64,80,96,128,160,192",
    "VLLM_DSV41_SUBWIDTH_GRAPHS": "1,2:2-5;4,8,16,32:2-6",
    "VLLM_DSV41_METADATA_SUBWIDTH_RANGES": "0",
    "VLLM_DSV41_METADATA_FORK_STREAMS": "0",
    "VLLM_DSV41_SHARE_CAPTURE_STREAM": "1",
    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "64",
    "VLLM_DSPARK_DRAFT_HEAD_FP8": "0",
    "VLLM_DSV4_MXFP4_DECODE_MAX_TOKENS": "192",
    "VLLM_DSV4_MXFP4_FUSED_SUM_MAX_TOKENS": "32",
    "VLLM_DSV41_MOE_PREFILL_SPLIT": "0",
    "VLLM_SPARSE_RAGGED_FAST_SCAN": "1",
    "VLLM_DSV41_ENGRAM_CPU_THREADS": "6",
    "VLLM_INDEXER_QUERY_SHARD": "1",
    "VLLM_INDEXER_DECODE_SHARD_MIN_REQS": "0",
}

PROFILES["tep4_dcp4_verification"] = {
    **PROFILES["verification"],
    "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD": "0",
    "VLLM_DSV41_DENSE_MXFP8": "1",
    "VLLM_DSV41_DENSE_MXFP8_DECODE": "1",
    "VLLM_DSV41_ENGRAM_NUMA_INTERLEAVE": "1",
    "VLLM_DSV41_WOA_FP8": "1",
    "VLLM_DSV41_OFFLOAD_VISION_EMBED": "1",
    "VLLM_DSV41_MARLIN_CLEAN_REPACK": "1",
    "VLLM_DSV41_COST_PREFILL_CHUNK": "0",
    "VLLM_DSV41_YOCO_PARTIAL_FFN": "1",
    "VLLM_DSV41_TOPK_RAGGED_REUSE": "1",
    "VLLM_DSV41_PREFILL_AR_OVERLAP": "1",
    "VLLM_DSV41_DECODE_REQ_SIZES": "1,2,4,8,16,24,32",
    "VLLM_DSV41_MIXED_SPEC_TOKENS": "16,24,32,40,48,64,80,96,128,160,192",
    "VLLM_DSV41_SUBWIDTH_GRAPHS": "1,2:2-5;4,8,16,32:2-6",
    "VLLM_DSV41_METADATA_SUBWIDTH_RANGES": "0",
    "VLLM_DSV41_METADATA_FORK_STREAMS": "0",
    "VLLM_DSV41_SHARE_CAPTURE_STREAM": "1",
    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "64",
    "VLLM_DSPARK_DRAFT_HEAD_FP8": "0",
    "VLLM_DSV4_MXFP4_DECODE_MAX_TOKENS": "192",
    "VLLM_DSV4_MXFP4_FUSED_SUM_MAX_TOKENS": "32",
    "VLLM_DSV41_MOE_PREFILL_SPLIT": "0",
    "VLLM_SPARSE_RAGGED_FAST_SCAN": "1",
    "VLLM_DSV41_ENGRAM_CPU_THREADS": "6",
    "VLLM_INDEXER_QUERY_SHARD": "1",
    "VLLM_INDEXER_DECODE_SHARD_MIN_REQS": "0",
}

# Aliases pointing to the exact same dicts for TEP4 (DCP=1 on official SM80).
PROFILES["tep4"] = PROFILES["tep4_dcp4"]
PROFILES["tep4_verification"] = PROFILES["tep4_dcp4_verification"]


def apply() -> None:
    name = os.environ.get("VLLM_DSV41_OPT_PROFILE", "").strip().lower()
    if name not in ("", "0", "off", "none"):
        if name not in PROFILES:
            raise ValueError(
                f"VLLM_DSV41_OPT_PROFILE={name!r}: unknown profile "
                f"(known: {', '.join(sorted(PROFILES))})"
            )
        for key, value in PROFILES[name].items():
            os.environ.setdefault(key, value)

    # Whenever dense MXFP8 decode is enabled (via profile or standalone env),
    # keep MXFP8 weights as 1-byte MXFP8 on GPU instead of 2x BF16 at load.
    if (
        os.environ.get("VLLM_DSV41_DENSE_MXFP8_DECODE", "0") == "1"
        or os.environ.get("VLLM_DSV41_DENSE_MXFP8", "0") == "1"
    ):
        os.environ.setdefault("VLLM_DSV41_DENSE_MXFP8_DECODE", "1")
        os.environ.setdefault("VLLM_DSV41_DENSE_MXFP8", "1")
        os.environ.setdefault("VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD", "0")
