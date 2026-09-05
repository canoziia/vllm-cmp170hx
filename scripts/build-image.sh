#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENGINE=${CONTAINER_ENGINE:-docker}
BASE_IMAGE=${BASE_IMAGE:-docker.io/vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e}
SOURCE_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD)
OUTPUT_IMAGE=${OUTPUT_IMAGE:-vllm-qwen38-nvfp4-spark:latest}
INSTALL_ROOT=${INSTALL_ROOT:-/usr/local/lib/python3.12/dist-packages}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
command -v "$ENGINE" >/dev/null || { echo "Container engine '$ENGINE' not found" >&2; exit 2; }
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/vllm-qwen38-spark-build.XXXXXX")
CID=""
cleanup() {
  if [[ -n "$CID" ]]; then "$ENGINE" rm -f "$CID" >/dev/null 2>&1 || true; fi
  if [[ "$KEEP_WORKDIR" == 1 ]]; then echo "Keeping build workdir: $WORKDIR"; else rm -rf "$WORKDIR"; fi
}
trap cleanup EXIT
mkdir -p "$WORKDIR/context/native" "$WORKDIR/context/vendor"
cp "$REPO_ROOT/native/ple_pread.c" "$WORKDIR/context/native/ple_pread.c"
cp -a "$REPO_ROOT/vendor/kernel-det" "$WORKDIR/context/vendor/kernel-det"
CID=$("$ENGINE" create "$BASE_IMAGE" /bin/true)
while read -r _hash relpath; do
  mkdir -p "$WORKDIR/context/$(dirname "$relpath")"
  "$ENGINE" cp "$CID:$INSTALL_ROOT/$relpath" "$WORKDIR/context/$relpath"
done <"$REPO_ROOT/manifests/base-qwen38-nvfp4-spark.sha256"
"$ENGINE" rm "$CID" >/dev/null
CID=""
"$REPO_ROOT/scripts/apply-patches.sh" "$WORKDIR/context"
{
  echo "FROM $BASE_IMAGE"
  echo "COPY native/ple_pread.c /tmp/ple_pread.c"
  echo "RUN gcc -O3 -fopenmp -shared -fPIC -o /usr/local/lib/libvllm_ple_pread.so /tmp/ple_pread.c && rm /tmp/ple_pread.c"
  echo "COPY vendor/kernel-det /opt/llm/kernel-det/src"
  echo "RUN cd /opt/llm/kernel-det/src && DET_BUILD_DIR=/opt/llm/kernel-det/build DET_ARCH=121a python3 build_det.py && cp /opt/llm/kernel-det/build/_C_det.so /opt/llm/kernel-det/_C_det.so"
  while read -r _hash relpath; do
    echo "COPY $relpath $INSTALL_ROOT/$relpath"
  done <"$REPO_ROOT/manifests/final-qwen38-nvfp4-spark.sha256"
  echo 'LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" vllm-cmp170hx.variant="qwen3.8-flash-next-nvfp4-spark"'
  echo "LABEL org.opencontainers.image.revision=\"$SOURCE_REVISION\""
} >"$WORKDIR/context/Containerfile"
"$ENGINE" build -t "$OUTPUT_IMAGE" -f "$WORKDIR/context/Containerfile" "$WORKDIR/context"
echo "Built $OUTPUT_IMAGE from $BASE_IMAGE for $(uname -m)"
