#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR="$REPO_ROOT/models/deepseek-v41"
# shellcheck disable=SC1091
source "$MODEL_DIR/manifests/source.env"
ENGINE=${CONTAINER_ENGINE:-podman}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
ENABLE_PERF_DEBUG=${ENABLE_PERF_DEBUG:-0}
if [[ ${ENABLE_HOT_DSPARK_TOGGLE:-0} != 0 ]]; then
  echo "Use ENABLE_PERF_DEBUG=1 for the combined debug package" >&2
  exit 2
fi
[[ $ENABLE_PERF_DEBUG == 0 || $ENABLE_PERF_DEBUG == 1 ]] || {
  echo "ENABLE_PERF_DEBUG must be 0 or 1" >&2
  exit 2
}
if [[ $ENABLE_PERF_DEBUG == 1 ]]; then
  OUTPUT_IMAGE=${OUTPUT_IMAGE:-${DEFAULT_OUTPUT_IMAGE}-debug}
  PATCH_LABEL=early-pp-primer+hot-perf-debug+hot-dspark
else
  OUTPUT_IMAGE=${OUTPUT_IMAGE:-$DEFAULT_OUTPUT_IMAGE}
  PATCH_LABEL=early-pp-communicator-primer
fi
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/dsv41-pp6-build.XXXXXX")
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

for command in "$ENGINE" git patch; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done

mkdir -p "$WORKDIR/source" "$WORKDIR/context/lmcache-payload"
git -C "$WORKDIR/source" init -q
git -C "$WORKDIR/source" remote add origin "$SOURCE_REPO"
git -C "$WORKDIR/source" fetch --depth=1 origin "$SOURCE_COMMIT"
git -C "$WORKDIR/source" checkout -q --detach FETCH_HEAD
ENABLE_PERF_DEBUG=$ENABLE_PERF_DEBUG \
  "$REPO_ROOT/scripts/apply-deepseek-v41-patches.sh" "$WORKDIR/source"

# The server image is shared with Qwen. Reuse :latest unless it is missing or
# REBUILD_LMCACHE_IMAGE=1 was explicitly requested. Client and server get the
# same patched LMCache tree; no second extraction or patch application here.
LMCACHE_SERVER_IMAGE=$("$REPO_ROOT/scripts/ensure-lmcache-server-image.sh")
LMCACHE_SERVER_ID=$("$ENGINE" image inspect --format '{{.Id}}' "$LMCACHE_SERVER_IMAGE" | cut -c1-12)
PAYLOAD_CID=$("$ENGINE" create --entrypoint /bin/true "$LMCACHE_SERVER_IMAGE")
"$ENGINE" cp "$PAYLOAD_CID:/opt/lmcache-patched/." "$WORKDIR/context/lmcache-payload/"
"$ENGINE" rm "$PAYLOAD_CID" >/dev/null
PAYLOAD_CID=

cp -a "$WORKDIR/source/vllm" "$WORKDIR/context/vllm"
cp "$REPO_ROOT/scripts/test-lmcache-patches.py" "$WORKDIR/context/test-lmcache-patches.py"
INTEGRATION_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD)

# One build, as for Qwen: the pinned vLLM tree and shared LMCache payload are
# copied into the same final client image. There is no deployable stage1 tag.
cat >"$WORKDIR/context/Containerfile" <<CONTAINERFILE
FROM $BASE_IMAGE
USER root
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY lmcache-payload/ /opt/lmcache-patched/
COPY test-lmcache-patches.py /tmp/test-lmcache-patches.py
ENV PYTHONPATH=/opt/lmcache-patched
RUN python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm /opt/lmcache-patched/lmcache \
    && python3 -c 'import lmcache,sys,torch,lmcache.cuda_ops,lmcache.lmcache_native,lmcache.lmcache_fs; from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector; assert lmcache.__version__ == "0.5.5"; assert lmcache.__file__.startswith("/opt/lmcache-patched/"); assert sys.version_info[:2] == (3,12); assert torch.version.cuda == "13.0"; print(lmcache.__version__, lmcache.__file__, torch.__version__, torch.version.cuda)' \
    && python3 /tmp/test-lmcache-patches.py \
    && rm /tmp/test-lmcache-patches.py
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \
      org.opencontainers.image.revision="$SOURCE_COMMIT" \
      io.canoziia.integration.revision="$INTEGRATION_REVISION" \
      io.canoziia.upstream="$SOURCE_REPO@$SOURCE_COMMIT" \
      io.canoziia.patch="$PATCH_LABEL" \
      io.canoziia.perf-debug="$ENABLE_PERF_DEBUG" \
      io.canoziia.hot-dspark="$ENABLE_PERF_DEBUG" \
      io.canoziia.lmcache.role="client" \
      io.canoziia.lmcache.payload-from="$LMCACHE_SERVER_IMAGE ($LMCACHE_SERVER_ID)"
CONTAINERFILE

build_args=(-t "$OUTPUT_IMAGE" -f "$WORKDIR/context/Containerfile")
if [[ $ENGINE == podman ]]; then
  build_args=(--format docker "${build_args[@]}")
fi
"$ENGINE" build "${build_args[@]}" "$WORKDIR/context"
echo "Built $OUTPUT_IMAGE"
