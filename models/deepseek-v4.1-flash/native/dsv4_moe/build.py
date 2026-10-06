"""Build the DeepSeek V4.1 MXFP4 decode extension (_dsv4_moe_C, sm_80).

    python3 build.py --out DIR [--build-dir DIR]

No GPU is needed. Produces DIR/_dsv4_moe_C.abi3.so (ATen C++ API: locked to
the torch build it was compiled against).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "_dsv4_moe_C"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ.setdefault("MAX_JOBS", "4")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), "/usr/local/cuda/bin", os.environ.get("PATH", "")])
    from torch.utils import cpp_extension

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    build_dir = Path(a.build_dir or out / "build")
    build_dir.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-std=c++20"]
    cuda_flags = [*flags, "--expt-relaxed-constexpr", "-lineinfo",
                  "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                  "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"]
    cpp_extension.load(name=a.name, sources=[str(HERE / "mxfp4_decode.cu")],
                       extra_cuda_cflags=cuda_flags, extra_cflags=flags,
                       build_directory=str(build_dir), is_python_module=False,
                       verbose=a.verbose)
    shutil.copy2(build_dir / f"{a.name}.so", out / f"{a.name}.abi3.so")
    print("built", out / f"{a.name}.abi3.so")


if __name__ == "__main__":
    main()
