#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR="$REPO_ROOT/models/qwen3.8-flash-next"
# shellcheck disable=SC1091
source "$MODEL_DIR/manifests/source.env"
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/lmcache.env"
ENGINE=${CONTAINER_ENGINE:-podman}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-$DEFAULT_OUTPUT_IMAGE}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/qwen38-cmp170hx-build.XXXXXX")
PAYLOAD_CID=

cleanup() {
  [[ -z $PAYLOAD_CID ]] || "$ENGINE" rm -f "$PAYLOAD_CID" >/dev/null 2>&1 || true
  if [[ $KEEP_WORKDIR == 1 ]]; then
    echo "Keeping build workdir: $WORKDIR"
  else
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

for command in "$ENGINE" git patch; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done

mkdir -p "$WORKDIR/source" "$WORKDIR/context/lmcache-payload"
git -C "$WORKDIR/source" init -q
git -C "$WORKDIR/source" remote add origin "$SOURCE_REPO"
git -C "$WORKDIR/source" fetch --depth=1 origin "$SOURCE_COMMIT"
git -C "$WORKDIR/source" checkout -q --detach FETCH_HEAD
"$REPO_ROOT/scripts/apply-qwen38-patches.sh" "$WORKDIR/source"

# Extract the complete ABI-matched official package, then apply the shared
# LMCache series used by every model image in this repository.
# The patched LMCache payload has a single source: the shared server image.
# It is built on demand here (REBUILD_LMCACHE_IMAGE=1 forces a rebuild), and the
# tree is taken from it rather than extracted and patched a second time, so the
# client runs exactly what the server runs.
LMCACHE_SERVER_IMAGE=$("$REPO_ROOT/scripts/ensure-lmcache-server-image.sh")
PAYLOAD_CID=$("$ENGINE" create --entrypoint /bin/true "$LMCACHE_SERVER_IMAGE")
"$ENGINE" cp "$PAYLOAD_CID:/opt/lmcache-patched/." "$WORKDIR/context/lmcache-payload/"
"$ENGINE" rm "$PAYLOAD_CID" >/dev/null
PAYLOAD_CID=

cp -a "$WORKDIR/source/vllm" "$WORKDIR/context/vllm"
cp "$MODEL_DIR/native/ple_pread.c" "$WORKDIR/context/ple_pread.c"
cp "$MODEL_DIR/scripts/lmcache" "$WORKDIR/context/lmcache-cli"
cp "$REPO_ROOT/scripts/test-lmcache-patches.py" "$WORKDIR/context/test-lmcache-patches.py"
cp -a "$MODEL_DIR/tests" "$WORKDIR/context/qwen-cache-tests"
cp "$MODEL_DIR/Containerfile" "$WORKDIR/context/Containerfile"

args=(
  --build-arg "QWEN_BASE_IMAGE=$BASE_IMAGE"
  --build-arg "SOURCE_REVISION=$SOURCE_COMMIT"
  --build-arg "INTEGRATION_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  -f "$WORKDIR/context/Containerfile"
  -t "$OUTPUT_IMAGE"
)
[[ $ENGINE != podman ]] || args=(--format docker "${args[@]}")
"$ENGINE" build "${args[@]}" "$WORKDIR/context"
echo "Built $OUTPUT_IMAGE"
