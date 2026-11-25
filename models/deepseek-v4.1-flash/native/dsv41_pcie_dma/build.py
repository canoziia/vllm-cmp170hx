"""Build the DeepSeek V4.1 PCIe Copy-Engine Ring DMA extension (_dsv41_pcie_dma_C, sm_80)."""

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "_dsv41_pcie_dma_C"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ.setdefault("MAX_JOBS", "8")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), "/usr/local/cuda/bin", os.environ.get("PATH", "")]
    )
    from torch.utils import cpp_extension

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    build_dir = Path(a.build_dir or out / "build_pcie_dma")
    build_dir.mkdir(parents=True, exist_ok=True)
    cuda_flags = [
        "-O3",
        "-std=c++20",
        "-DUSE_CUDA",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    cpp_extension.load(
        name=NAME,
        sources=[str(HERE / "dsv41_pcie_dma_kernel.cu")],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=c++20"],
        build_directory=str(build_dir),
        is_python_module=True,
        verbose=a.verbose,
    )
    shutil.copy2(build_dir / f"{NAME}.so", out / f"{NAME}.so")
    print("built", out / f"{NAME}.so", flush=True)


if __name__ == "__main__":
    main()
