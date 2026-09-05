#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENGINE=${CONTAINER_ENGINE:-docker}
BASE_IMAGE=${BASE_IMAGE:-docker.io/vllm/vllm-openai:qwen38-flash-next}
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
mkdir -p "$WORKDIR/context/native"
cp "$REPO_ROOT/native/ple_pread.c" "$WORKDIR/context/native/ple_pread.c"
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
  while read -r _hash relpath; do
    echo "COPY $relpath $INSTALL_ROOT/$relpath"
  done <"$REPO_ROOT/manifests/final-qwen38-nvfp4-spark.sha256"
  echo 'LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" vllm-cmp170hx.variant="qwen3.8-flash-next-nvfp4-spark"'
} >"$WORKDIR/context/Containerfile"
"$ENGINE" build -t "$OUTPUT_IMAGE" -f "$WORKDIR/context/Containerfile" "$WORKDIR/context"
echo "Built $OUTPUT_IMAGE from $BASE_IMAGE for $(uname -m)"
