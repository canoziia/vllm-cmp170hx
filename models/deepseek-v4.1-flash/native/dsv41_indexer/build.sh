#!/usr/bin/env bash
# Build _dsv41_indexer_C inside the DeepSeek V4.1 image (no GPU needed).
# Usage: bash build.sh OUT_DIR
# The image's CUDA toolkit lacks some library headers (cusparse.h, ...) that
# torch's headers include; link only those from pip's nvidia/cu13/include
# (putting the whole directory on CPATH clashes with the toolkit runtime
# headers).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?usage: build.sh OUT_DIR}
PY=${PYTHON:-python3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_HOME
NV_INC=$("$PY" -c 'import glob, nvidia; print(next(iter(sorted(p for b in nvidia.__path__ for p in glob.glob(b + "/cu*/include"))), ""))' 2>/dev/null || true)
SHIM=$(mktemp -d /tmp/dsv4-moe-inc.XXXXXX)
trap 'rm -rf "$SHIM"' EXIT
dirs=("$CUDA_HOME/include")
[[ -d "$CUDA_HOME/targets/x86_64-linux/include" ]] && dirs+=("$CUDA_HOME/targets/x86_64-linux/include")
if [[ -n $NV_INC && -d $NV_INC ]]; then
  for entry in "$NV_INC"/*; do
    [[ -f "$entry" ]] || continue
    name=$(basename "$entry"); present=0
    for d in "${dirs[@]}"; do [[ -e "$d/$name" ]] && { present=1; break; }; done
    [[ $present == 1 ]] || ln -s "$entry" "$SHIM/$name"
  done
fi
export CPATH="$SHIM${CPATH:+:$CPATH}"
shift
"$PY" "$HERE/build.py" --out "$OUT" "$@"
