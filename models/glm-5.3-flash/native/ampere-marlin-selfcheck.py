#!/usr/bin/env python3
"""Image-build self-check for the GLM DFlash2 image (CPU only, no GPU).

1. vllm._ampere_marlin_C is importable from the vllm package (no
   VLLM_GLM5_MARLIN_DECODE_LIB) and passes the real dflash2/0006 loader
   validation (torch/CUDA/ABI build_info + four op schemas with CUDA kernels).
   marlin_decode.py is loaded by path so the GLM model package (which probes
   the platform) is not imported during the build.
2. Every switch of the series keeps its default: the image changes nothing
   until the deployment's environment turns a feature on.
"""
import importlib.util
import os

import vllm
import vllm.envs as envs

assert "VLLM_GLM5_MARLIN_DECODE_LIB" not in os.environ
import vllm._ampere_marlin_C as ext  # noqa: E402

pkg = os.path.dirname(vllm.__file__)
assert os.path.dirname(ext.__file__) == pkg, ext.__file__

path = os.path.join(pkg, "models/glm5next/nvidia/ops/marlin_decode.py")
spec = importlib.util.spec_from_file_location("_glm5_marlin_decode_selfcheck", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.require_extension()
print("ampere_marlin:", ext.__file__, ext.build_info())

off = (
    "VLLM_GLM5_DFLASH_ADAPTIVE_K", "VLLM_GLM5_THIN_GEMM", "VLLM_GLM5_MARLIN_DECODE_CUDA",
    "VLLM_GLM5_INDEXER_GATHER_CLAMP", "VLLM_GLM5_PP_SPARSE_MLA_PREFILL",
    "VLLM_GLM5_PP_MARLIN_PREFILL", "VLLM_GLM5_PP_KDA_PREFILL", "VLLM_GLM5_PP_FOLD_DRAFT_FC",
    "VLLM_GLM5_DECODE_MHC_V2", "VLLM_GLM5_PROLOGUE_FUSE", "VLLM_GLM5_PROLOGUE_PACKED_H2D",
    "VLLM_GLM5_SPARSE_MLA_DEVICE_REQ_IDS", "VLLM_PP_DRAFT_TAIL_VERIFY",
    "VLLM_GLM5_MLA_PROFILE_WS_CLAMP",
)
for name in off:
    assert name not in os.environ, name
    assert not getattr(envs, name), name
assert envs.VLLM_GLM5_INDEXER_JIT_WARMUP  # the one switch that defaults on
assert envs.VLLM_GLM5_MARLIN_DECODE_VARIANT == "orig"
assert envs.VLLM_GLM5_MARLIN_DECODE_LIB == ""
assert envs.VLLM_PP_DRAFT_TAIL_STAGE == -1
print("glm dflash2 defaults: ok")
