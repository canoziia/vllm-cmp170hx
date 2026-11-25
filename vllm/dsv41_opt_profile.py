# SPDX-License-Identifier: Apache-2.0
"""One switch for the DeepSeek V4.1 Flash SM80 optimisation patches.

VLLM_DSV41_OPT_PROFILE=auto|default|verification|tp_auto|tp_auto_verification|tp4_fused|tep4|tep4_verification|tep4_fused
sets every per-patch switch to its deployed value *unless that variable is
already set*, so a single switch can still be overridden.

In ``auto`` mode, the launch command line and model ``config.json`` are
inspected before vLLM imports:
  * non-DeepSeek-V4/V4.1 models are left completely untouched (official vLLM);
  * DeepSeek-V4/V4.1 with TP=4, PP=1, EP disabled (default) selects ``tp4_fused``
    (which further auto-detects NVLink vs PCIe P2P via NVML/peer topology);
  * DeepSeek-V4/V4.1 with TP=4, PP=1, EP enabled selects ``tep4_fused``
    (legacy TEP4 path, also auto-detecting NVLink vs PCIe P2P);
  * DeepSeek-V4/V4.1 with PP > 1 selects ``verification`` when
    ``enable_adaptive_verification`` is set in ``--speculative-config``, else
    ``default``;
  * DeepSeek-V4/V4.1 with other TP > 1 (PP=1) splits selects ``tp_auto``
    (or ``tp_auto_verification``);
  * DeepSeek-V4/V4.1 single-GPU (TP=1, PP=1) selects ``default`` / ``verification``.
The resolved profile name is written back to ``VLLM_DSV41_OPT_PROFILE`` so
spawned EngineCore and GPU workers inherit the exact same configuration.

Unset / "0" / "off" / "none": profile defaults stay off (except ensuring
1x-HBM MXFP8 emulation dequant opt-out when dense MXFP8 decode is explicitly
enabled). Applied from vllm.env_override before any other vLLM module is
imported, in the API server and in every spawned worker.
"""

import json
import os
import sys

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

# Topology-aware deep fusion profile for 4x SM80 TEP4 (auto-detects NVLink vs PCIe P2P).
PROFILES["tep4_fused"] = {
    **PROFILES["tep4_dcp4_verification"],
    "VLLM_DSV41_AMPERE_PCIE_COLLECTIVES": "1",
    "VLLM_DSV41_MARKOV_SYMM_MEM": "1",
    "VLLM_DSV41_IN_GRAPH_TEXT_EMBED": "1",
    "VLLM_DSV41_MOE_PREFILL_CHUNK_TOKENS": "0",
    "VLLM_DSV41_MOE_PREFILL_SPLIT": "1",
    "VLLM_DSV41_PREFILL_MXFP8_CUBLAS": "1",
    "VLLM_DSV41_COHORT_UNIFORM_K": "1",
    "VLLM_DSV41_MIXED_SPEC_FULL_GRAPH": "0",
    "VLLM_DSV41_DECODE_REQ_SIZES": "1,2,4,8,16,32",
    "VLLM_DSV41_SUBWIDTH_GRAPHS": "1,2,4,8,16,32:2-5",
    "VLLM_DSV41_COST_MS_FUSED": "1",
    "VLLM_DSV41_TP_LOCAL_GREEDY": "1",
    "VLLM_DSV41_WOA_CPP": "1",
    "VLLM_DSV41_WOA_MARLIN": "1",
    "VLLM_DSV41_DENSE_SPLITK_M8": "1",
    "VLLM_DSV41_DENSE_KSPLIT_NOBAR": "1",
    "VLLM_DSV4_MXFP4_DQ": "3",
    "VLLM_DSV41_FUSED_SHARED_ADD": "1",
    "VLLM_DSV41_FUSED_W2_AR": "1",
    "VLLM_DSV41_PREFILL_DMA_RING": "auto",
    "VLLM_DSV41_PREFILL_ROW_SHARD": "auto",
    "VLLM_DSV41_PREFILL_2MB_MOE": "auto",
}

# Topology-aware deep fusion profile for 4x SM80 pure TP4 (default: EP=1, NI=576).
PROFILES["tp4_fused"] = {
    **PROFILES["tep4_fused"],
    "VLLM_DSV41_COST_MS_FUSED": "tp4",
    "VLLM_DSV41_DEFER_FINALIZE_MIN_TOKENS": "3",
}

# General TP-safe SM80 optimisations for non-TEP4 TP>1 splits (e.g. TP2/TP8).
_TP_COMMON_EXTRAS: dict[str, str] = {
    "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD": "0",
    "VLLM_DSV41_DENSE_MXFP8": "1",
    "VLLM_DSV41_DENSE_MXFP8_DECODE": "1",
    "VLLM_DSV41_ENGRAM_NUMA_INTERLEAVE": "1",
    "VLLM_DSV41_WOA_FP8": "1",
    "VLLM_DSV41_OFFLOAD_VISION_EMBED": "1",
    "VLLM_DSV41_MARLIN_CLEAN_REPACK": "1",
    "VLLM_DSV41_YOCO_PARTIAL_FFN": "1",
    "VLLM_DSV41_TOPK_RAGGED_REUSE": "1",
    "VLLM_DSV41_PREFILL_AR_OVERLAP": "1",
    "VLLM_DSV41_AMPERE_PCIE_COLLECTIVES": "1",
    "VLLM_DSV41_PREFILL_MXFP8_CUBLAS": "1",
    "VLLM_SPARSE_RAGGED_FAST_SCAN": "1",
    "VLLM_INDEXER_QUERY_SHARD": "1",
    "VLLM_DSV41_WOA_CPP": "1",
    "VLLM_DSV41_WOA_MARLIN": "1",
    "VLLM_DSV41_FUSED_SHARED_ADD": "1",
    "VLLM_DSV41_PREFILL_DMA_RING": "auto",
    "VLLM_DSV41_PREFILL_ROW_SHARD": "auto",
    "VLLM_DSV41_PREFILL_2MB_MOE": "auto",
}
PROFILES["tp_auto"] = {**PROFILES["default"], **_TP_COMMON_EXTRAS}
PROFILES["tp_auto_verification"] = {**PROFILES["verification"], **_TP_COMMON_EXTRAS}


def _parse_launch_args(argv: list[str]) -> tuple[str | None, int, int, bool, bool]:
    """Extract (model, tp, pp, ep, adaptive_verification) from vLLM CLI argv."""
    model: str | None = None
    positional_model: str | None = None
    tp = 1
    pp = 1
    ep = False
    adaptive = False

    def _safe_int(val: str, default: int) -> int:
        try:
            return max(1, int(val))
        except ValueError:
            return default

    def _check_spec(raw: str) -> bool:
        try:
            cfg = json.loads(raw)
        except Exception:
            return False
        return isinstance(cfg, dict) and bool(cfg.get("enable_adaptive_verification"))

    args = argv[1:] if len(argv) > 1 else []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "serve":
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                positional_model = args[i + 1]
                i += 2
                continue
        elif tok == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
            continue
        elif tok.startswith("--model="):
            model = tok.split("=", 1)[1]
        elif tok in ("-tp", "--tensor-parallel-size") and i + 1 < len(args):
            tp = _safe_int(args[i + 1], tp)
            i += 2
            continue
        elif tok.startswith("-tp="):
            tp = _safe_int(tok.split("=", 1)[1], tp)
        elif tok.startswith("--tensor-parallel-size="):
            tp = _safe_int(tok.split("=", 1)[1], tp)
        elif tok in ("-pp", "--pipeline-parallel-size") and i + 1 < len(args):
            pp = _safe_int(args[i + 1], pp)
            i += 2
            continue
        elif tok.startswith("-pp="):
            pp = _safe_int(tok.split("=", 1)[1], pp)
        elif tok.startswith("--pipeline-parallel-size="):
            pp = _safe_int(tok.split("=", 1)[1], pp)
        elif tok == "--enable-expert-parallel":
            ep = True
        elif tok == "--speculative-config" and i + 1 < len(args):
            adaptive = _check_spec(args[i + 1])
            i += 2
            continue
        elif tok.startswith("--speculative-config="):
            adaptive = _check_spec(tok.split("=", 1)[1])
        i += 1

    return (model or positional_model, tp, pp, ep, adaptive)


def _is_deepseek_v41_model(model: str | None) -> bool:
    """Return True iff *model* identifies a DeepSeek-V4/V4.1 checkpoint."""
    if not model:
        return False
    cfg_path = os.path.join(model, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return False
        if not isinstance(cfg, dict):
            return False
        candidates: list[str] = []
        for mt in (
            cfg.get("model_type"),
            (cfg.get("text_config") or {}).get("model_type")
            if isinstance(cfg.get("text_config"), dict)
            else None,
        ):
            if isinstance(mt, str):
                candidates.append(mt.lower())
        archs = cfg.get("architectures")
        if isinstance(archs, list):
            candidates.extend(a.lower() for a in archs if isinstance(a, str))
        return any("deepseek_v4" in c or "deepseekv4" in c for c in candidates)

    lower = model.lower()
    return "deepseek" in lower and any(
        tok in lower for tok in ("v4.1", "v4_1", "v4-1", "deepseek-v4", "deepseek_v4")
    )


def resolve_auto_profile(argv: list[str] | None = None) -> str | None:
    """Resolve the best SM80 profile for the launch command, or None for other models."""
    model, tp, pp, ep, adaptive = _parse_launch_args(
        sys.argv if argv is None else argv
    )
    if not _is_deepseek_v41_model(model):
        return None
    if pp > 1:
        return "verification" if adaptive else "default"
    if tp == 4 and not ep:
        return "tp4_fused"
    if tp == 4 and ep:
        return "tep4_fused"
    if tp > 1:
        return "tp_auto_verification" if adaptive else "tp_auto"
    return "verification" if adaptive else "default"


def _has_active_nvlink_nvml() -> bool:
    """Query host NVML via ctypes without initializing CUDA or loading libnccl."""
    import ctypes

    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit_v2() != 0:
            return False
        try:
            count = ctypes.c_uint(0)
            if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
                return False
            for idx in range(int(count.value)):
                handle = ctypes.c_void_p()
                if nvml.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(handle)) != 0:
                    continue
                for link in range(18):
                    active = ctypes.c_uint(0)
                    rc = nvml.nvmlDeviceGetNvLinkState(
                        handle, link, ctypes.byref(active)
                    )
                    if rc != 0:
                        break
                    if active.value:
                        return True
            return False
        finally:
            nvml.nvmlShutdown()
    except Exception:
        return False


# NVML nvmlGpuTopologyLevel_t enum values (from nvml.h):
#   0=INTERNAL, 10=SINGLE (PIX), 20=MULTIPLE (PXB),
#   30=HOSTBRIDGE (PHB), 40=NODE, 50=SYSTEM (SYS).
# Fine-grained 16-byte GPU SM P2P BAR1 writes (CuTe Lamport) require a shared
# PCIe switch (<= 20, PIX/PXB); across CPU root complexes (>= 30, PHB/NODE/SYS)
# posted-write credit exhaustion degrades low-M decode by 2-3x vs PyNCCL DMA.
NVML_TOPO_SWITCH_MAX = 20
NVML_TOPO_NAMES: dict[int, str] = {
    0: "INTERNAL",
    10: "PIX",
    20: "PXB",
    30: "PHB",
    40: "NODE",
    50: "SYS",
}


def pcie_switch_connected_nvml(
    device_ids: list[int | str] | None = None,
) -> tuple[bool, int | None]:
    """Return ``(all_switch_connected, worst_nvml_topo_level)`` without CUDA init.

    Checks ``nvmlDeviceGetTopologyCommonAncestor`` across every pair of
    devices. Returns ``(True, worst)`` iff all pairs share a PCIe switch
    (``INTERNAL`` / ``PIX`` / ``PXB``, level <= 20). Returns ``(False, None)``
    if NVML is unavailable or fewer than 2 devices are visible.
    """
    import ctypes

    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit_v2() != 0:
            return (False, None)
        try:
            targets: list[int | str]
            if device_ids is not None:
                targets = list(device_ids)
            else:
                raw_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
                if raw_vis:
                    targets = []
                    for tok in raw_vis.split(","):
                        tok = tok.strip()
                        if not tok:
                            continue
                        targets.append(int(tok) if tok.isdigit() else tok)
                else:
                    count = ctypes.c_uint(0)
                    if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
                        return (False, None)
                    targets = list(range(int(count.value)))
            if len(targets) < 2:
                return (False, None)

            handles: list[ctypes.c_void_p] = []
            for dev in targets:
                handle = ctypes.c_void_p()
                if isinstance(dev, int):
                    rc = nvml.nvmlDeviceGetHandleByIndex_v2(
                        ctypes.c_uint(dev), ctypes.byref(handle)
                    )
                else:
                    rc = nvml.nvmlDeviceGetHandleByUUID(
                        dev.encode("utf-8"), ctypes.byref(handle)
                    )
                    if rc != 0 and not dev.startswith("GPU-"):
                        rc = nvml.nvmlDeviceGetHandleByUUID(
                            f"GPU-{dev}".encode("utf-8"), ctypes.byref(handle)
                        )
                if rc != 0:
                    return (False, None)
                handles.append(handle)

            worst: int | None = None
            for i in range(len(handles)):
                for j in range(i + 1, len(handles)):
                    lvl = ctypes.c_int(-1)
                    rc = nvml.nvmlDeviceGetTopologyCommonAncestor(
                        handles[i], handles[j], ctypes.byref(lvl)
                    )
                    if rc != 0:
                        return (False, None)
                    val = int(lvl.value)
                    worst = val if worst is None else max(worst, val)
            if worst is None:
                return (False, None)
            return (worst <= NVML_TOPO_SWITCH_MAX, worst)
        finally:
            nvml.nvmlShutdown()
    except Exception:
        return (False, None)


def apply() -> None:
    name = os.environ.get("VLLM_DSV41_OPT_PROFILE", "").strip().lower()
    if name == "auto":
        resolved = resolve_auto_profile()
        if resolved is None:
            name = ""
        else:
            name = resolved
            # Propagate resolved profile so spawned workers inherit it cleanly
            # even though worker argv lacks the CLI serve flags.
            os.environ["VLLM_DSV41_OPT_PROFILE"] = resolved
    if name not in ("", "0", "off", "none"):
        if name not in PROFILES:
            raise ValueError(
                f"VLLM_DSV41_OPT_PROFILE={name!r}: unknown profile "
                f"(known: auto, {', '.join(sorted(PROFILES))})"
            )
        for key, value in PROFILES[name].items():
            os.environ.setdefault(key, value)

    # Set NCCL_PROTO=Simple BEFORE torch/libnccl is imported ONLY when running
    # on switch-connected PCIe GPUs (PIX/PXB, zero active NVLinks) where CuTe
    # Lamport handles small decode messages (M<=60) and NCCL only handles
    # larger Ring/prefill transfers. On cross-RC topologies (PHB/NODE/SYS),
    # Lamport is disabled so PyNCCL handles small decode too and benefits from
    # NCCL's automatic LL protocol selection.
    pcie_opt = os.environ.get("VLLM_DSV41_AMPERE_PCIE_COLLECTIVES", "0").strip().lower()
    if (
        pcie_opt in ("1", "force")
        and "NCCL_PROTO" not in os.environ
        and not _has_active_nvlink_nvml()
        and (pcie_opt == "force" or pcie_switch_connected_nvml()[0])
    ):
        os.environ["NCCL_PROTO"] = "Simple"

    # Whenever dense MXFP8 decode is enabled (via profile or standalone env),
    # keep MXFP8 weights as 1-byte MXFP8 on GPU instead of 2x BF16 at load.
    if (
        os.environ.get("VLLM_DSV41_DENSE_MXFP8_DECODE", "0") == "1"
        or os.environ.get("VLLM_DSV41_DENSE_MXFP8", "0") == "1"
    ):
        os.environ.setdefault("VLLM_DSV41_DENSE_MXFP8_DECODE", "1")
        os.environ.setdefault("VLLM_DSV41_DENSE_MXFP8", "1")
        os.environ.setdefault("VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD", "0")
