#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENGINE=${CONTAINER_ENGINE:-podman}
BASE_IMAGE=${BASE_IMAGE:-docker.io/vllm/vllm-openai:qwen38-flash-next}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-localhost/vllm-cmp170hx:qwen3.8-flash-next}
INSTALL_ROOT=${INSTALL_ROOT:-/usr/local/lib/python3.12/dist-packages}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}

command -v "$ENGINE" >/dev/null || {
  echo "Container engine '$ENGINE' not found" >&2
  exit 2
}

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/vllm-cmp170hx-build.XXXXXX")
CID=""
cleanup() {
  if [[ -n "$CID" ]]; then
    "$ENGINE" rm -f "$CID" >/dev/null 2>&1 || true
  fi
  if [[ "$KEEP_WORKDIR" == 1 ]]; then
    echo "Keeping build workdir: $WORKDIR"
  else
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

mkdir -p "$WORKDIR/context/native"
cp "$REPO_ROOT/native/ple_pread.c" "$WORKDIR/context/native/ple_pread.c"
CID=$("$ENGINE" create "$BASE_IMAGE" /bin/true)

while read -r _hash relpath; do
  mkdir -p "$WORKDIR/context/$(dirname "$relpath")"
  "$ENGINE" cp "$CID:$INSTALL_ROOT/$relpath" "$WORKDIR/context/$relpath"
done <"$REPO_ROOT/manifests/base-qwen38-flash-next.sha256"

"$ENGINE" rm "$CID" >/dev/null
CID=""

"$REPO_ROOT/scripts/apply-patches.sh" "$WORKDIR/context"

cat >"$WORKDIR/context/Containerfile" <<CONTAINERFILE
FROM $BASE_IMAGE
COPY native/ple_pread.c /tmp/ple_pread.c
RUN gcc -O3 -fopenmp -shared -fPIC -o /usr/local/lib/libvllm_ple_pread.so /tmp/ple_pread.c && rm /tmp/ple_pread.c
COPY vllm/models/qwen3_8_flash_next/nvidia/model.py $INSTALL_ROOT/vllm/models/qwen3_8_flash_next/nvidia/model.py
COPY vllm/models/qwen3_8_flash_next/nvidia/model_state.py $INSTALL_ROOT/vllm/models/qwen3_8_flash_next/nvidia/model_state.py
COPY vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py $INSTALL_ROOT/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
COPY vllm/models/qwen3_8_flash_next/nvidia/ple_nvme.py $INSTALL_ROOT/vllm/models/qwen3_8_flash_next/nvidia/ple_nvme.py
COPY vllm/v1/ple_offload/connector.py $INSTALL_ROOT/vllm/v1/ple_offload/connector.py
COPY vllm/v1/ple_offload/worker.py $INSTALL_ROOT/vllm/v1/ple_offload/worker.py
COPY vllm/v1/worker/gpu_worker.py $INSTALL_ROOT/vllm/v1/worker/gpu_worker.py
COPY vllm/v1/worker/gpu_model_runner.py $INSTALL_ROOT/vllm/v1/worker/gpu_model_runner.py
COPY vllm/models/qwen3_8_flash_next/nvidia/mtp.py $INSTALL_ROOT/vllm/models/qwen3_8_flash_next/nvidia/mtp.py
COPY vllm/v1/worker/gpu/model_runner.py $INSTALL_ROOT/vllm/v1/worker/gpu/model_runner.py
COPY vllm/v1/worker/gpu/pp_utils.py $INSTALL_ROOT/vllm/v1/worker/gpu/pp_utils.py
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \
      vllm-cmp170hx.base="$BASE_IMAGE" \
      vllm-cmp170hx.variant="qwen3.8-flash-next"
CONTAINERFILE

"$ENGINE" build -t "$OUTPUT_IMAGE" -f "$WORKDIR/context/Containerfile" "$WORKDIR/context"
echo "Built $OUTPUT_IMAGE from $BASE_IMAGE"
