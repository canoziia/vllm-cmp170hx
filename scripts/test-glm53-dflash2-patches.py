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
check("series has 16 patches", len(series) == 16, str(len(series)))

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
]
for text, name, default in defaults:
    check(f"{name} defaults to {default!r}", env_default(text, name, default))
check("VLLM_GLM5_MARLIN_DECODE_VARIANT defaults to 'orig'",
      re.search(r'"VLLM_GLM5_MARLIN_DECODE_VARIANT",\s*"orig"', envs) is not None)

print("diagnostic / unvalidated patches are not part of the series:")
absent = {
    "force-file (old 0014)": "VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE",
    "draft trace (old 0015)": "VLLM_DFLASH_DRAFT_TRACE",
    "PP pipeline trace (old 0021)": "VLLM_PP_PIPELINE_TRACE",
    "mHC v1 numerics (old 0018)": "VLLM_GLM5_TARGET_MHC_V1",
    "mHC v2 (old 0009)": "VLLM_GLM5_TARGET_MHC_V2",
    "KDA dual projection (old 0011)": "VLLM_GLM5_KDA_GATE_PROJECTION",
    "context KV graph (old 0007)": "VLLM_DFLASH_CONTEXT_KV_GRAPH",
}
haystack = "\n".join((root / p).read_text() for p in sorted(touched) if (root / p).is_file())
for label, needle in absent.items():
    check(label, needle not in haystack)
for rel in ("vllm/v1/worker/gpu/pp_trace.py", "vllm/v1/worker/gpu/spec_decode/dflash/trace.py",
            ops + "mhc_decode_v1.py", ops + "mhc_decode_v2.py", ops + "kda_gate_projection.py"):
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
