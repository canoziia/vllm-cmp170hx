#!/usr/bin/env bash
set -euo pipefail

# Build the shared LMCache server image.
#
# The server is model-agnostic: it stores and serves opaque KV blocks and learns
# the KV layout from the client at registration, so one image serves every
# deployment here. It is also the single source of the patched LMCache payload:
# client images take /opt/lmcache-patched from this image instead of extracting
# and patching the official payload themselves, so client and server provably
# run the same tree.
#
# Identity: the repository revision. The image is tagged both :latest and
# :<short-rev>, and ensure-lmcache-server-image.sh reuses :<short-rev> when it is
# already present. Any commit therefore produces a new image, which is coarse but
# cannot go stale.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/lmcache.env"
ENGINE=${CONTAINER_ENGINE:-podman}
REVISION=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)
IMAGE_BASE=${LMCACHE_SERVER_IMAGE_BASE:-localhost/lmcache-server}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-$IMAGE_BASE:$REVISION}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/lmcache-server-build.XXXXXX")
payload_cid=

cleanup() {
  if [[ -n $payload_cid ]]; then
    "$ENGINE" rm -f "$payload_cid" >/dev/null 2>&1 || true
  fi
  if [[ $KEEP_WORKDIR == 1 ]]; then
    echo "Keeping build workdir: $WORKDIR"
  else
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

for command in "$ENGINE" patch; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done

mkdir -p "$WORKDIR/context/lmcache-payload"

# Pull by digest and extract the exact official package. The payload image is a
# tiny carrier for /payload, including compiled .so files and dist-info from the
# matching lmcache/vllm-openai release image.
"$ENGINE" pull "$LMCACHE_PAYLOAD_IMAGE"
payload_cid=$("$ENGINE" create --entrypoint /bin/true "$LMCACHE_PAYLOAD_IMAGE")
"$ENGINE" cp "$payload_cid:/payload/." "$WORKDIR/context/lmcache-payload/"
"$ENGINE" rm "$payload_cid" >/dev/null
payload_cid=

"$REPO_ROOT/scripts/apply-lmcache-patches.sh" "$WORKDIR/context/lmcache-payload"
cp "$REPO_ROOT/scripts/test-lmcache-patches.py" "$WORKDIR/context/"

cat > "$WORKDIR/context/Containerfile" <<CONTAINERFILE
FROM $LMCACHE_OFFICIAL_IMAGE
USER root
COPY lmcache-payload/ /opt/lmcache-patched/
COPY test-lmcache-patches.py /opt/lmcache-patched/
ENV PYTHONPATH=/opt/lmcache-patched
RUN python3 -m compileall -q /opt/lmcache-patched/lmcache && python3 -c 'import lmcache,sys,torch,lmcache.cuda_ops,lmcache.lmcache_native,lmcache.lmcache_fs; from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector; assert lmcache.__version__ == "0.5.5"; assert lmcache.__file__.startswith("/opt/lmcache-patched/"); assert sys.version_info[:2] == (3,12); assert torch.version.cuda == "13.0"; print(lmcache.__version__, lmcache.__file__, torch.__version__, torch.version.cuda)' && python3 /opt/lmcache-patched/test-lmcache-patches.py
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \
      org.opencontainers.image.revision="$REVISION" \
      io.canoziia.lmcache.role="server" \
      io.canoziia.lmcache.payload-for="client images take /opt/lmcache-patched from here" \
      io.canoziia.lmcache.source="$LMCACHE_SOURCE_REPO@$LMCACHE_SOURCE_COMMIT" \
      io.canoziia.lmcache.official-image="$LMCACHE_OFFICIAL_IMAGE" \
      io.canoziia.lmcache.payload-image="$LMCACHE_PAYLOAD_IMAGE" \
      io.canoziia.lmcache.patch-series="patches/lmcache/series"
CONTAINERFILE

if [[ $ENGINE == podman ]]; then
  "$ENGINE" build --format docker -f "$WORKDIR/context/Containerfile" -t "$OUTPUT_IMAGE" "$WORKDIR/context"
else
  "$ENGINE" build -f "$WORKDIR/context/Containerfile" -t "$OUTPUT_IMAGE" "$WORKDIR/context"
fi

# Keep :latest pointing at the most recently built revision. The :<revision> tag
# is what reuse and pinning key on.
"$ENGINE" tag "$OUTPUT_IMAGE" "$IMAGE_BASE:latest" >/dev/null
echo "Built LMCache server image: $OUTPUT_IMAGE (also tagged $IMAGE_BASE:latest)"
