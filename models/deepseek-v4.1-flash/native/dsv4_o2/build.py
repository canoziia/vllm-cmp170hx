"""Build an sm_80 torch extension from .cu sources (no GPU needed).
    python3 build.py --out DIR --name NAME src.cu [...]
Writes DIR/NAME.so and keeps ptxas -v output in the log (spills/registers)."""
import argparse, os, shutil, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--name", default="_o2")
ap.add_argument("srcs", nargs="+")
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ.setdefault("MAX_JOBS", "8")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), "/usr/local/cuda/bin", os.environ.get("PATH", "")])
from torch.utils import cpp_extension
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
bd = out / ("build_" + a.name); bd.mkdir(parents=True, exist_ok=True)
flags = ["-O3", "-std=c++17"]
cuda_flags = [*flags, "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-v", "--keep", "--keep-dir", str(bd),
              "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
              "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"]
cpp_extension.load(name=a.name, sources=[str(Path(s).resolve()) for s in a.srcs], extra_cuda_cflags=cuda_flags,
                   extra_cflags=flags, build_directory=str(bd), is_python_module=False, verbose=True)
shutil.copy2(bd / f"{a.name}.so", out / f"{a.name}.so")
print("built", out / f"{a.name}.so")
