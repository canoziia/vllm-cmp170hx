#!/usr/bin/env bash
# Build _dsv41_engram_cpu.so (OpenMP + AVX2 CPU gather/dequant helper for Engram).
# Usage: bash build.sh [OUT_DIR] [SRC_CPP]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT="${1:-${HERE}/../../vllm-src/vllm/models/deepseek_v4_1/common}"
SRC="${2:-${OUT}/engram_cpu_gather.cpp}"
mkdir -p "$OUT"
CXX=${CXX:-g++}
ARCH_FLAGS=()
if [[ "$(uname -m)" == "x86_64" ]]; then
  ARCH_FLAGS=(-mavx2 -mfma)
fi
"$CXX" -O3 -std=c++17 "${ARCH_FLAGS[@]}" -fopenmp -shared -fPIC "$SRC" -o "$OUT/_dsv41_engram_cpu.so"
echo "built $OUT/_dsv41_engram_cpu.so"
