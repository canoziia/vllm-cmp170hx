#!/usr/bin/env bash
# Build localhost/vllm-backport:mimo-v2.6-flash from BASE_IMAGE:
#   pinned DeepSeek V4.1 source + DeepSeek/shared/adaptive series (exactly as
#   build-deepseek-v41-image.sh) + models/mimo-v2.6-flash/patches/series,
#   plus the shared LMCache client payload (same tree as the server image).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DS_DIR="$REPO_ROOT/models/deepseek-v4.1-flash"
MIMO_DIR="$REPO_ROOT/models/mimo-v2.6-flash"
# shellcheck disable=SC1091
source "$DS_DIR/manifests/source.env"    # SOURCE_REPO, SOURCE_COMMIT, BASE_IMAGE
# shellcheck disable=SC1091
source "$MIMO_DIR/manifests/source.env"  # DEFAULT_OUTPUT_IMAGE (MiMo), model pin
ENGINE=${CONTAINER_ENGINE:-podman}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-$DEFAULT_OUTPUT_IMAGE}
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/mimo-v26-build.XXXXXX")
PAYLOAD_CID=
cleanup() {
  if [[ -n $PAYLOAD_CID ]]; then
    "$ENGINE" rm -f "$PAYLOAD_CID" >/dev/null 2>&1 || true
  fi
  if [[ $KEEP_WORKDIR == 1 ]]; then
    echo "Keeping build workdir: $WORKDIR"
  else
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

for command in "$ENGINE" git patch sha256sum; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done

mkdir -p "$WORKDIR/source" "$WORKDIR/context/lmcache-payload"
git -C "$WORKDIR/source" init -q
git -C "$WORKDIR/source" remote add origin "$SOURCE_REPO"
git -C "$WORKDIR/source" fetch --depth=1 origin "$SOURCE_COMMIT"
git -C "$WORKDIR/source" checkout -q --detach FETCH_HEAD
"$REPO_ROOT/scripts/apply-mimo-v26-patches.sh" "$WORKDIR/source"

# Same LMCache client payload as every other image in this repository.
LMCACHE_SERVER_IMAGE=$("$REPO_ROOT/scripts/ensure-lmcache-server-image.sh")
LMCACHE_SERVER_ID=$("$ENGINE" image inspect --format '{{.Id}}' "$LMCACHE_SERVER_IMAGE" | cut -c1-12)
PAYLOAD_CID=$("$ENGINE" create --entrypoint /bin/true "$LMCACHE_SERVER_IMAGE")
"$ENGINE" cp "$PAYLOAD_CID:/opt/lmcache-patched/." "$WORKDIR/context/lmcache-payload/"
"$ENGINE" rm "$PAYLOAD_CID" >/dev/null
PAYLOAD_CID=

find "$WORKDIR/source/vllm" -name __pycache__ -type d -prune -exec rm -rf {} +
cp -a "$WORKDIR/source/vllm" "$WORKDIR/context/vllm"
cp "$REPO_ROOT/scripts/test-lmcache-patches.py" "$WORKDIR/context/test-lmcache-patches.py"
INTEGRATION_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD)
SERIES_SHA=$(sha256sum "$MIMO_DIR/patches/series" | cut -c1-12)

cat >"$WORKDIR/context/Containerfile" <<CONTAINERFILE
FROM $BASE_IMAGE
USER root
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY lmcache-payload/ /opt/lmcache-patched/
COPY test-lmcache-patches.py /tmp/test-lmcache-patches.py
ENV PYTHONPATH=/opt/lmcache-patched
RUN python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm /opt/lmcache-patched/lmcache \\
    && python3 -c 'import lmcache,sys,torch,lmcache.cuda_ops,lmcache.lmcache_native,lmcache.lmcache_fs; from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector; assert lmcache.__version__ == "0.5.5"; assert lmcache.__file__.startswith("/opt/lmcache-patched/"); assert sys.version_info[:2] == (3,12); assert torch.version.cuda == "13.0"; print(lmcache.__version__, lmcache.__file__, torch.__version__, torch.version.cuda)' \\
    && python3 /tmp/test-lmcache-patches.py \\
    && rm /tmp/test-lmcache-patches.py
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \\
      org.opencontainers.image.revision="$SOURCE_COMMIT" \\
      io.canoziia.integration.revision="$INTEGRATION_REVISION" \\
      io.canoziia.upstream="$SOURCE_REPO@$SOURCE_COMMIT" \\
      io.canoziia.patch="early-pp-communicator-primer+adaptive+mimo-v2.6" \\
      io.canoziia.mimo.series="$SERIES_SHA" \\
      io.canoziia.mimo.model="$MODEL_ID@$MODEL_REVISION" \\
      io.canoziia.perf-debug="0" \\
      io.canoziia.lmcache.role="client" \\
      io.canoziia.lmcache.payload-from="$LMCACHE_SERVER_IMAGE ($LMCACHE_SERVER_ID)"
CONTAINERFILE

build_args=(-t "$OUTPUT_IMAGE" -f "$WORKDIR/context/Containerfile")
if [[ $ENGINE == podman ]]; then
  build_args=(--format docker "${build_args[@]}")
fi
"$ENGINE" build "${build_args[@]}" "$WORKDIR/context"
echo "Built $OUTPUT_IMAGE"
