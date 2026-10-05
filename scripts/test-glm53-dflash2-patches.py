#!/usr/bin/env python3
"""Static regression for the GLM-5.3-Flash W4A16 + DFlash2 series
(models/glm-5.3-flash/patches/dflash2/series).

Usage: test-glm53-dflash2-patches.py <SOURCE_TREE>

Run against a tree that already has the DeepSeek V4.1 series, the shared
series, the adaptive series and the dflash2 series applied (the apply script
does that and then calls this file). CPU only, no torch: it checks that every
patch left its marker, that every switch keeps its default, that the
diagnostic-only patches are absent, and that every touched Python file
compiles.
"""
import py_compile
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCH_DIR = REPO / "models/glm-5.3-flash/patches/dflash2"
root = Path(sys.argv[1]).resolve()
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' :: ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(name)


def src(rel: str) -> str:
    path = root / rel
    return path.read_text() if path.is_file() else ""


def env_default(text: str, name: str, default: str) -> bool:
    pattern = rf"""os\.(?:getenv|environ\.get)\(\s*["']{re.escape(name)}["']\s*,\s*["']{re.escape(default)}["']"""
    return re.search(pattern, text) is not None


series = [line.strip() for line in (PATCH_DIR / "series").read_text().splitlines()
          if line.strip() and not line.startswith("#")]
check("series has 28 patches", len(series) == 28, str(len(series)))

touched: set[str] = set()
for name in series:
    text = (PATCH_DIR / name).read_text()
    check(f"{name}: has a Subject header", text.startswith("From: ") and "\nSubject: [PATCH] " in text)
    for path in re.findall(r"^\+\+\+ b/(\S+)", text, flags=re.M):
        touched.add(path)
check("series only touches vllm/", all(p.startswith("vllm/") for p in touched), str(sorted(touched)))

envs = src("vllm/envs.py")
model = "vllm/models/glm5next/nvidia/model.py"
ops = "vllm/models/glm5next/nvidia/ops/"

print("markers:")
markers = [
    ("0001 aux hidden states", model, "SupportsEagle3"),
    ("0001 aux tensor switch", model, "VLLM_GLM5_AUX_HIDDEN_TENSOR"),
    ("0002 tail ring", "vllm/models/glm5next/nvidia/attention.py", "def tail_ring_size("),
    ("0003 drafter KV groups", "vllm/v1/core/kv_cache_utils.py", "ride the MLA tensors"),
    ("0004 adaptive depth", "vllm/v1/spec_decode/dynamic/adaptive_k.py", ""),
    ("0005 thin GEMM", ops + "thin_gemm.py", ""),
    ("0006 compiled Marlin loader", ops + "marlin_decode.py", 'EXT_NAME = "_ampere_marlin_C"'),
    ("0006 default import vllm._ampere_marlin_C", ops + "marlin_decode.py",
     'importlib.import_module(f"vllm.{EXT_NAME}")'),
    ("0006 hook in MarlinExperts", "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py", "marlin_decode"),
    ("0006 early check in Worker.init_device", "vllm/v1/worker/gpu_worker.py", "check_startup"),
    ("0007 fused grouped conv", "vllm/model_executor/models/qwen3_dflash2.py", "dflash2_grouped_conv"),
    ("0008 deterministic alignment", ops + "router_align_decode.py", "def maybe_align("),
    ("0009 sparse MLA schedule", "vllm/v1/attention/ops/triton_mla_sparse_mm_experimental.py", ""),
    ("0009 sparse MLA switch", "vllm/v1/attention/backends/mla/triton_mla_sparse.py",
     "VLLM_GLM5_SPARSE_MLA_MM_EXPERIMENTAL"),
    ("0010 route v2", ops + "route_v2_decode.py", "def take_align("),
    ("0010 route v2 falls back to 0008", ops + "marlin_decode.py", "maybe_align"),
    ("0011 gather clamp", "vllm/v1/attention/backends/mla/indexer.py", "VLLM_GLM5_INDEXER_GATHER_CLAMP"),
    ("0012 sparse MLA prefill", ops + "sparse_prefill_mla_pp.py", ""),
    ("0013 Marlin prefill split", ops + "marlin_prefill_split.py", ""),
    ("0013 split alignment", ops + "moe_split_align.py", ""),
    ("0014 KDA prefill", ops + "kda_prefill_pp.py", ""),
    ("0014 KDA prefill hook", "vllm/models/glm5next/nvidia/kda.py", "VLLM_GLM5_PP_KDA_PREFILL"),
    ("0015 indexer JIT warmup", "vllm/model_executor/layers/sparse_attn_indexer_kpool.py",
     "VLLM_GLM5_INDEXER_JIT_WARMUP"),
    ("0016 fc fold", "vllm/v1/worker/gpu/spec_decode/eagle/aux_fc_fold.py", 'AUX_FC_PARTIAL_KEY = "aux_fc_partial"'),
    ("0017 mHC decode v2 kernels", ops + "mhc_decode_v2.py", ""),
    ("0017 mHC decode v2 gate", ops + "mhc_decode_v2_gate.py", "def maybe_fused_post_pre("),
    ("0017 hook in the decoder layer", model, "mhc_decode_v2_gate import maybe_fused_post_pre"),
    ("0017 warmup before capture", "vllm/model_executor/warmup/kernel_warmup.py", "VLLM_GLM5_DECODE_MHC_V2"),
    ("0018 prologue fuse", "vllm/v1/worker/gpu/prologue_fuse.py", ""),
    ("0018 hook in GDN builder", "vllm/v1/attention/backends/gdn_attn.py", "prologue_fuse.enabled()"),
    ("0019 packed HtoD", "vllm/v1/worker/gpu/prologue_h2d.py", ""),
    ("0019 hook in runner", "vllm/v1/worker/gpu/model_runner.py", "prologue_h2d.stage_prologue_inputs"),
    ("0020 device req ids", "vllm/v1/attention/backends/mla/req_id_per_token.py", ""),
    ("0020 hook in sparse builder", "vllm/v1/attention/backends/mla/xpu_mla_sparse.py",
     "VLLM_GLM5_SPARSE_MLA_DEVICE_REQ_IDS"),
    ("0021 draft tail", "vllm/v1/worker/gpu/pp_draft_tail.py", ""),
    ("0021 audited tail (workspaces before KV sizing)", "vllm/v1/worker/gpu/model_runner.py",
     "self.draft_tail.prepare_workspaces()"),
    ("0021 tail broadcast", "vllm/v1/worker/gpu/pp_utils.py", "def send_draft_tail("),
    ("0022 MLA profile clamp", "vllm/model_executor/layers/attention/mla_attention.py",
     "VLLM_GLM5_MLA_PROFILE_WS_CLAMP"),
    ("0023 route first", ops + "route_v2_decode.py", "def route_first("),
    ("0023 runner hook", "vllm/model_executor/layers/fused_moe/runner/moe_runner.py", "route_first()"),
    ("0024 idx glue", ops + "idx_glue_0024.py", "native_fp8_cast_supported"),
    ("0024 attention hook", "vllm/models/glm5next/nvidia/attention.py", "fwht128_quant_fp8_wscale"),
    ("0024 kpool hook", "vllm/model_executor/layers/sparse_attn_indexer_kpool.py", "idx_glue_0024"),
    ("0025 fp32 prenorm", "vllm/model_executor/kernels/mhc/tilelang.py", "VLLM_GLM5_TARGET_PRENORM_FP32_0026"),
    ("0026 moe sum add", ops + "moe_sum_add.py", ""),
    ("0026 runner hook", "vllm/model_executor/layers/fused_moe/runner/moe_runner.py", "VLLM_GLM5_MOE_SUM_ADD"),
    ("0028 bf16x3 prenorm kernel (vendored, Apache-2.0)", "vllm/model_executor/kernels/mhc/mm_prenorm_bf16x3.py",
     "SPDX-License-Identifier: Apache-2.0"),
    ("0028 provenance", "vllm/model_executor/kernels/mhc/mm_prenorm_bf16x3.py",
     "3a2bf16dae8b97f5ff2c7e9bc5809d24545e6340"),
    ("0028 hook", "vllm/model_executor/kernels/mhc/tilelang.py", "mm_prenorm_bf16x3 import"),
    ("0027 shared reorder", "vllm/model_executor/layers/fused_moe/runner/shared_experts.py",
     "VLLM_GLM5_SHARED_EXPERT_REORDER"),
]
for label, rel, needle in markers:
    text = src(rel)
    check(label, bool(text) and needle in text, rel)

print("defaults (every switch keeps its patch default):")
defaults = [
    (envs, "VLLM_GLM5_DFLASH_ADAPTIVE_K", "0"),
    (envs, "VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS", "7,5"),
    (envs, "VLLM_GLM5_DFLASH_ADAPTIVE_K_ACCEPT", "0"),
    (envs, "VLLM_GLM5_THIN_GEMM", "0"),
    (envs, "VLLM_GLM5_THIN_GEMM_MAX_TOKENS", "32"),
    (envs, "VLLM_GLM5_MARLIN_DECODE_CUDA", "0"),
    (envs, "VLLM_GLM5_MARLIN_DECODE_LIB", ""),
    (envs, "VLLM_GLM5_INDEXER_GATHER_CLAMP", "0"),
    (envs, "VLLM_GLM5_PP_SPARSE_MLA_PREFILL", "0"),
    (envs, "VLLM_GLM5_PP_SPARSE_MLA_PREFILL_MIN_TOKENS", "384"),
    (envs, "VLLM_GLM5_PP_MARLIN_PREFILL", "0"),
    (envs, "VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS", "384"),
    (envs, "VLLM_GLM5_PP_KDA_PREFILL", "0"),
    (envs, "VLLM_GLM5_PP_KDA_PREFILL_MAX_TOKENS", "2312"),
    (envs, "VLLM_GLM5_INDEXER_JIT_WARMUP", "1"),
    (envs, "VLLM_GLM5_PP_FOLD_DRAFT_FC", "0"),
    (src(model), "VLLM_GLM5_AUX_HIDDEN_TENSOR", "stream_mean"),
    (src("vllm/model_executor/models/qwen3_dflash2.py"), "VLLM_DFLASH2_FUSED_GROUPED_CONV", "0"),
    (src(ops + "router_align_decode.py"), "VLLM_GLM5_ROUTER_ALIGN_DECODE", "0"),
    (src("vllm/v1/attention/backends/mla/triton_mla_sparse.py"), "VLLM_GLM5_SPARSE_MLA_MM_EXPERIMENTAL", "0"),
    (src(ops + "route_v2_decode.py"), "VLLM_GLM5_ROUTE_V2", "0"),
    (src(ops + "route_v2_decode.py"), "VLLM_GLM5_ROUTE_V2_GEMV", "gate"),
    (envs, "VLLM_GLM5_DECODE_MHC_V2", "0"),
    (envs, "VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS", "32"),
    (envs, "VLLM_GLM5_PROLOGUE_FUSE", "0"),
    (envs, "VLLM_GLM5_PROLOGUE_FUSE_GDN", "1"),
    (envs, "VLLM_GLM5_PROLOGUE_PACKED_H2D", "0"),
    (src("vllm/v1/worker/gpu/prologue_h2d.py"), "VLLM_GLM5_PROLOGUE_PACKED_H2D", "0"),
    (envs, "VLLM_GLM5_SPARSE_MLA_DEVICE_REQ_IDS", "0"),
    (envs, "VLLM_PP_DRAFT_TAIL_STAGE", "-1"),
    (envs, "VLLM_PP_DRAFT_TAIL_VERIFY", "0"),
    (envs, "VLLM_GLM5_MLA_PROFILE_WS_CLAMP", "0"),
    (src(ops + "route_v2_decode.py"), "VLLM_GLM5_ROUTE_V2_FIRST", "0"),
    (src(ops + "idx_glue_0024.py"), "VLLM_GLM5_DECODE_IDX_GLUE_0024", "0"),
    (src("vllm/model_executor/kernels/mhc/tilelang.py"), "VLLM_GLM5_TARGET_PRENORM_FP32_0026", "0"),
    (src("vllm/model_executor/layers/fused_moe/runner/moe_runner.py"), "VLLM_GLM5_MOE_SUM_ADD", "0"),
    (src(ops + "moe_sum_add.py"), "VLLM_GLM5_MOE_SUM_ADD", "0"),
    (src("vllm/model_executor/layers/fused_moe/runner/moe_runner.py"), "VLLM_GLM5_SHARED_EXPERT_REORDER", "0"),
    (src("vllm/model_executor/kernels/mhc/tilelang.py"), "VLLM_GLM5_TARGET_PRENORM_FP32_0026B_MIN_TOKENS", "384"),
    (envs, "VLLM_MHC_POST_FUSE_SQRSUM", "0"),
]
for text, name, default in defaults:
    check(f"{name} defaults to {default!r}", env_default(text, name, default))
# 0028 must only be reachable inside 0025's VLLM_GLM5_TARGET_PRENORM_FP32_0026=1
# branch, so the series default (0025 off) is unchanged.
def _gated_by_0025(text: str) -> bool:
    lines = text.splitlines()
    gate = [i for i, l in enumerate(lines)
            if 'os.environ.get("VLLM_GLM5_TARGET_PRENORM_FP32_0026", "0") == "1"' in l]
    if len(gate) != 1:
        return False
    i = gate[0]
    while not lines[i].rstrip().endswith("):"):  # end of the if condition
        i += 1
    body_indent = len(lines[i]) - len(lines[i].lstrip()) + 4
    j = i + 1
    while j < len(lines) and (not lines[j].strip()
                              or len(lines[j]) - len(lines[j].lstrip()) >= body_indent):
        j += 1
    inside = "\n".join(lines[i + 1:j])
    outside = "\n".join(lines[:i + 1] + lines[j:])
    needles = ("VLLM_GLM5_TARGET_PRENORM_FP32_0026B_MIN_TOKENS", "mm_prenorm_bf16x3")
    return all(n in inside and n not in outside for n in needles)


check("0028 bf16x3 route only inside the 0025 (FP32_0026=1) branch",
      _gated_by_0025(src("vllm/model_executor/kernels/mhc/tilelang.py")))
route_v2 = src(ops + "route_v2_decode.py")
check("VLLM_GLM5_ROUTE_V2_FIRST is opt-in (only '1' enables it)",
      '"VLLM_GLM5_ROUTE_V2_FIRST", "0").strip() == "1"' in route_v2)
check("VLLM_GLM5_MARLIN_DECODE_VARIANT defaults to 'orig'",
      re.search(r'"VLLM_GLM5_MARLIN_DECODE_VARIANT",\s*"orig"', envs) is not None)

print("diagnostic / unvalidated patches are not part of the series:")
absent = {
    "force-file (old 0014)": "VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE",
    "draft trace (old 0015)": "VLLM_DFLASH_DRAFT_TRACE",
    "PP pipeline trace (old 0021)": "VLLM_PP_PIPELINE_TRACE",
    "mHC v1 numerics (old 0018)": "VLLM_GLM5_TARGET_MHC_V1",
    "mHC v2 (old 0009)": "VLLM_GLM5_TARGET_MHC_V2",
    "PP metadata cache (dev 0025)": "VLLM_PP_METADATA_CACHE_0025",
    "PP pack hop tensors (dev 0029)": "VLLM_PP_PACK_TENSORS_0029",
    "same-history overlay": "VLLM_SAME_HISTORY_",
    "same-history overlay module": "same_history",
    "KDA dual projection (old 0011)": "VLLM_GLM5_KDA_GATE_PROJECTION",
    "context KV graph (old 0007)": "VLLM_DFLASH_CONTEXT_KV_GRAPH",
}
# Excluded dev 0025/0029 would also touch parallel_state.py.
scanned = sorted(touched | {"vllm/distributed/parallel_state.py"})
haystack = "\n".join((root / p).read_text() for p in scanned if (root / p).is_file())
for label, needle in absent.items():
    check(label, needle not in haystack)
for rel in ("vllm/v1/worker/gpu/pp_trace.py", "vllm/v1/worker/gpu/spec_decode/dflash/trace.py",
            ops + "mhc_decode_v1.py", ops + "kda_gate_projection.py",
            "vllm/distributed/pp_pack_0029.py",
            "vllm/v1/worker/gpu/spec_decode/dflash/same_history.py"):
    check(f"{rel} absent", not (root / rel).exists())

print("py_compile:")
for rel in sorted(touched):
    if not rel.endswith(".py"):
        continue
    try:
        with tempfile.TemporaryDirectory() as tmp:
            py_compile.compile(str(root / rel), cfile=f"{tmp}/x.pyc", doraise=True)
        ok, detail = True, ""
    except (py_compile.PyCompileError, FileNotFoundError) as exc:
        ok, detail = False, str(exc)
    check(rel, ok, detail)

if failures:
    print(f"GLM DFlash2 series: {len(failures)} check(s) FAILED", file=sys.stderr)
    sys.exit(1)
print("GLM DFlash2 series: all static checks passed.")
