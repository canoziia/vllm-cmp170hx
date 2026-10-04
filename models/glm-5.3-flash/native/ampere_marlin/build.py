# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the decode-only _ampere_marlin_C extension (sm_80) without touching vLLM.

Adapted from Morrowmake vllm-cmp170hx
csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py (@ 3a2bf16dae).
Differences: only the decode translation units are built (decode.cu,
decode_orig.cu, module.cpp); the prefill GEMM (ops.cu, kernels_sm80.cu), which
needs MM's modified csrc/libtorch_stable/moe/marlin_moe_wna16 templates, is
left out. The sources here therefore need only the PyTorch/CUDA headers.

No GPU is used: CUDA_VISIBLE_DEVICES is cleared and the arch list is pinned to
8.0. Output goes only to --out; the installed vLLM is never modified.

    python3 build.py --out /work/out [--build-dir /work/build] [--verbose]

Produces <out>/_ampere_marlin_C.abi3.so (the ".abi3" tag only mirrors MM's file
name; the library uses the ATen C++ API and is locked to the exact torch build
it was compiled against, which the runtime loader checks via build_info()).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "_ampere_marlin_C"
SOURCES = ("decode.cu", "decode_orig.cu", "module.cpp")


def _preflight() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ.setdefault("MAX_JOBS", str(min(4, os.cpu_count() or 1)))
    if "CUDA_HOME" not in os.environ and Path("/usr/local/cuda/bin/nvcc").exists():
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
    # ninja may live next to the interpreter (pip-installed).
    os.environ["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent),
         str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin"),
         os.environ.get("PATH", "")])
    missing = [t for t in ("nvcc", "ninja") if shutil.which(t) is None]
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        missing.append("c++/g++")
    if missing:
        raise SystemExit(f"build.py: missing tools: {', '.join(missing)}")
    for tool in ("nvcc", cxx):
        out = subprocess.run([tool, "--version"], capture_output=True, text=True)
        print(f"[build] {tool}: {(out.stdout or out.stderr).strip().splitlines()[-1]}")


def build(out: Path, build_dir: Path, verbose: bool) -> Path:
    _preflight()
    import torch
    from torch.utils import cpp_extension

    if torch.version.cuda is None or cpp_extension.CUDA_HOME is None:
        raise SystemExit("A CUDA-enabled PyTorch and a CUDA toolkit are required")
    print(f"[build] torch {torch.__version__} (CUDA {torch.version.cuda}), "
          f"CUDA_HOME={cpp_extension.CUDA_HOME}, cxx11_abi="
          f"{int(torch._C._GLIBCXX_USE_CXX11_ABI)}")
    build_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    maps = [
        f"-ffile-prefix-map={HERE}=ampere_marlin",
        f"-ffile-prefix-map={build_dir}=build",
        f"-ffile-prefix-map={Path(torch.__file__).resolve().parent}=torch",
    ]
    flags = ["-O3", "-std=c++20"]
    cuda_flags = [
        *flags, "--expt-relaxed-constexpr", "--threads=4",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        *[f"-Xcompiler={flag}" for flag in maps],
    ]
    # Same flag as MM's builder; only understood by nvcc >= 12.8.
    nvcc_help = subprocess.run(["nvcc", "--help"], capture_output=True, text=True).stdout
    if "static-global-template-stub" in nvcc_help:
        cuda_flags.append("-static-global-template-stub=false")
    # is_python_module=False: cpp_extension dlopens the result through
    # torch.ops.load_library, which registers the ops (no CUDA context needed).
    cpp_extension.load(
        name=NAME,
        sources=[str(HERE / s) for s in SOURCES],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=[*flags, *maps],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=verbose,
    )
    source = build_dir / f"{NAME}.so"
    destination = out / f"{NAME}.abi3.so"
    with tempfile.NamedTemporaryFile(dir=out, prefix=NAME, suffix=".tmp",
                                     delete=False) as f:
        temporary = Path(f.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


_VERIFY = r"""
import importlib.util, sys
import torch
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("vllm._ampere_marlin_C", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for n in ("decode_gemm", "decode_act", "decode_gemm_orig", "decode_act_orig"):
    q = "_ampere_marlin_C::" + n
    schema = torch._C._dispatch_find_schema_or_throw(q, "").schema()
    assert torch._C._dispatch_has_kernel_for_dispatch_key(q, "CUDA"), q
    print("[build]   " + str(schema))
print("[build] build_info:", module.build_info())
"""


def verify(path: Path) -> None:
    """CPU-only check in a fresh interpreter (this one already registered the
    ops through cpp_extension.load; a second registration would abort)."""
    subprocess.run([sys.executable, "-c", _VERIFY, str(path)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for _ampere_marlin_C.abi3.so")
    parser.add_argument("--build-dir", type=Path,
                        help="persistent build directory (default: temporary)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.build_dir is None:
        with tempfile.TemporaryDirectory(prefix="ampere_marlin_") as d:
            dest = build(args.out.resolve(), Path(d), args.verbose)
    else:
        dest = build(args.out.resolve(), args.build_dir.resolve(), args.verbose)
    print(f"[build] wrote {dest} ({dest.stat().st_size} bytes)")
    verify(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
