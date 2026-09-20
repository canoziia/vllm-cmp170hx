#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/source.env"
ENGINE=${CONTAINER_ENGINE:-podman}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-localhost/deepseek-v41-cmp170hx:latest}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
ENABLE_PERF_DEBUG=${ENABLE_PERF_DEBUG:-0}
[[ $ENABLE_PERF_DEBUG == 0 || $ENABLE_PERF_DEBUG == 1 ]] || {
  echo "ENABLE_PERF_DEBUG must be 0 or 1" >&2
  exit 2
}
PATCH_LABEL=early-pp-communicator-primer
if [[ $ENABLE_PERF_DEBUG == 1 ]]; then
  PATCH_LABEL=early-pp-primer+hot-perf-debug
fi
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/dsv41-pp6-build.XXXXXX")
cleanup() {
  if [[ "$KEEP_WORKDIR" == 1 ]]; then
    echo "Keeping build workdir: $WORKDIR"
  else
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

command -v "$ENGINE" >/dev/null || { echo "Missing container engine: $ENGINE" >&2; exit 2; }
command -v git >/dev/null || { echo "Missing git" >&2; exit 2; }

mkdir -p "$WORKDIR/source" "$WORKDIR/context"
git -C "$WORKDIR/source" init -q
git -C "$WORKDIR/source" remote add origin "$SOURCE_REPO"
git -C "$WORKDIR/source" fetch --depth=1 origin "$SOURCE_COMMIT"
git -C "$WORKDIR/source" checkout -q --detach FETCH_HEAD
ENABLE_PERF_DEBUG=$ENABLE_PERF_DEBUG \
  "$REPO_ROOT/scripts/apply-patches.sh" "$WORKDIR/source"

cp -a "$WORKDIR/source/vllm" "$WORKDIR/context/vllm"
cat >"$WORKDIR/context/Containerfile" <<CONTAINERFILE
FROM $BASE_IMAGE
USER root
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \\
      org.opencontainers.image.revision="$SOURCE_COMMIT" \\
      io.canoziia.upstream="$SOURCE_REPO@$SOURCE_COMMIT" \\
      io.canoziia.patch="$PATCH_LABEL" \
      io.canoziia.perf-debug="$ENABLE_PERF_DEBUG"
CONTAINERFILE

build_args=(-t "$OUTPUT_IMAGE" -f "$WORKDIR/context/Containerfile")
if [[ "$ENGINE" == podman ]]; then
  build_args=(--format docker "${build_args[@]}")
fi
"$ENGINE" build "${build_args[@]}" "$WORKDIR/context"
echo "Built $OUTPUT_IMAGE"
