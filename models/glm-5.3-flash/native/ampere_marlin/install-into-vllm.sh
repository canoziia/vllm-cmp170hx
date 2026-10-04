#!/usr/bin/env bash
# Build _ampere_marlin_C inside the serving image and install it into the vllm
# package, so patch dflash2/0006 imports it as vllm._ampere_marlin_C without
# VLLM_GLM5_MARLIN_DECODE_LIB. No GPU is used (build.py clears
# CUDA_VISIBLE_DEVICES and pins TORCH_CUDA_ARCH_LIST=8.0).
#
# Usage (Containerfile): bash install-into-vllm.sh [SOURCE_DIR]
set -euo pipefail

SRC=$(realpath "${1:-$(dirname "${BASH_SOURCE[0]}")}")
PY=${PYTHON:-python3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_HOME
# Parity with Morrowmake's build_standalone.py (opt-in switch); build.py here
# has no guard, the variable is harmless.
export VLLM_BUILD_AMPERE_MARLIN=1

VLLM_DIR=$("$PY" -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')
NV_INC=${NVIDIA_PIP_INCLUDE:-}
if [[ -z $NV_INC ]]; then
  NV_INC=$("$PY" -c 'import glob, nvidia; print(next(iter(sorted(p for b in nvidia.__path__ for p in glob.glob(b + "/cu*/include"))), ""))' 2>/dev/null || true)
fi

# The image's CUDA toolkit lacks some library headers (cusparse.h,
# cusolverDn.h, ...) that torch's headers include; pip ships them under
# nvidia/cu13/include. Putting that whole directory on CPATH clashes with the
# toolkit's CUDA runtime headers (__cudaLaunch macro errors), so link only the
# headers the toolkit does not have into a private directory.
SHIM=$(mktemp -d /tmp/ampere-marlin-inc.XXXXXX)
trap 'rm -rf "$SHIM"' EXIT
toolkit_dirs=("$CUDA_HOME/include")
[[ -d "$CUDA_HOME/targets/x86_64-linux/include" ]] && toolkit_dirs+=("$CUDA_HOME/targets/x86_64-linux/include")
linked=0
if [[ -n $NV_INC && -d $NV_INC ]]; then
  # Header files only (the verified workaround); directories such as crt/
  # would bring a second CUDA runtime header set back in.
  for entry in "$NV_INC"/*; do
    [[ -f "$entry" ]] || continue
    name=$(basename "$entry")
    present=0
    for dir in "${toolkit_dirs[@]}"; do
      [[ -e "$dir/$name" ]] && { present=1; break; }
    done
    if [[ $present == 0 ]]; then
      ln -s "$entry" "$SHIM/$name"
      linked=$((linked + 1))
    fi
  done
fi
echo "[ampere_marlin] linked $linked missing toolkit header(s) from ${NV_INC:-<none>}"
export CPATH="$SHIM${CPATH:+:$CPATH}"

"$PY" "$SRC/build.py" --out "$VLLM_DIR"
test -f "$VLLM_DIR/_ampere_marlin_C.abi3.so"
echo "[ampere_marlin] installed $VLLM_DIR/_ampere_marlin_C.abi3.so"
