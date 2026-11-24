#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?usage: build.sh OUT_DIR}
PY=${PYTHON:-python3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_HOME
NV_INC=$("$PY" -c 'import glob, nvidia; print(next(iter(sorted(p for b in nvidia.__path__ for p in glob.glob(b + "/cu*/include"))), ""))' 2>/dev/null || true)
SHIM=$(mktemp -d /tmp/dsv41-swa-inc.XXXXXX)
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
