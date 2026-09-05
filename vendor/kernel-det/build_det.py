# SPDX-License-Identifier: Apache-2.0
"""Build the upstream QSA top-k kernel under an independent operator namespace."""
import os
from pathlib import Path
from torch.utils.cpp_extension import load

source = Path(__file__).resolve().parent
build = Path(os.environ.get("DET_BUILD_DIR", str(source / "build")))
build.mkdir(parents=True, exist_ok=True)
arch = os.environ.get("DET_ARCH", "121a")
if arch not in {"120a", "121a"}:
    raise ValueError("Expected an explicitly supported SM12x architecture")
result = load(
    name="_C_det",
    sources=[str(source / "topk_det.cu"), str(source / "bindings_det.cpp")],
    extra_include_paths=[str(source)],
    extra_cflags=["-O3", "-std=c++17", "-DUSE_CUDA"],
    extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                      "-DTORCH_STABLE_ONLY", "-DUSE_CUDA",
                      f"-gencode=arch=compute_{arch},code=sm_{arch}"],
    build_directory=str(build),
    is_python_module=False,
    verbose=True,
)
print(f"Built independent QSA top-k extension: {result}")
