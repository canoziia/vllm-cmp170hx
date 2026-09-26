#!/usr/bin/env bash
set -euo pipefail

# Layer the shared patched LMCache payload onto a DeepSeek client image.
#
# The payload is taken from the shared server image rather than extracted and
# patched again here: one build, one tree, so the client's LMCacheMPConnector and
# the server it talks to provably run the same code. The server image is built on
# demand (scripts/ensure-lmcache-server-image.sh).
#
# Run scripts/build-deepseek-v41-image.sh first to produce the client base image,
# then point DEEPSEEK_BASE_IMAGE at it.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENGINE=${CONTAINER_ENGINE:-podman}
DEEPSEEK_BASE_IMAGE=${DEEPSEEK_BASE_IMAGE:-localhost/deepseek-v41-cmp170hx:latest}
CLIENT_OUTPUT_IMAGE=${CLIENT_OUTPUT_IMAGE:-localhost/deepseek-v41-cmp170hx:official-lmcache-v0.5.5-patched}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/lmcache-client-build.XXXXXX")
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

LMCACHE_SERVER_IMAGE=$("$REPO_ROOT/scripts/ensure-lmcache-server-image.sh")

mkdir -p "$WORKDIR/context/lmcache-payload"
payload_cid=$("$ENGINE" create --entrypoint /bin/true "$LMCACHE_SERVER_IMAGE")
"$ENGINE" cp "$payload_cid:/opt/lmcache-patched/." "$WORKDIR/context/lmcache-payload/"
"$ENGINE" rm "$payload_cid" >/dev/null
payload_cid=

# The client environment differs from the server's (system site-packages vs the
# official image's venv), so re-assert version and native extension loading here.
cat > "$WORKDIR/context/Containerfile" <<CONTAINERFILE
FROM $DEEPSEEK_BASE_IMAGE
USER root
COPY lmcache-payload/ /opt/lmcache-patched/
ENV PYTHONPATH=/opt/lmcache-patched
RUN python3 -m compileall -q /opt/lmcache-patched/lmcache && python3 -c 'import lmcache,sys,torch,lmcache.cuda_ops,lmcache.lmcache_native,lmcache.lmcache_fs; from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector; assert lmcache.__version__ == "0.5.5"; assert lmcache.__file__.startswith("/opt/lmcache-patched/"); assert sys.version_info[:2] == (3,12); assert torch.version.cuda == "13.0"; print(lmcache.__version__, lmcache.__file__, torch.__version__, torch.version.cuda)'
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \
      io.canoziia.lmcache.role="client" \
      io.canoziia.lmcache.payload-from="$LMCACHE_SERVER_IMAGE"
CONTAINERFILE

if [[ $ENGINE == podman ]]; then
  "$ENGINE" build --format docker -f "$WORKDIR/context/Containerfile" -t "$CLIENT_OUTPUT_IMAGE" "$WORKDIR/context"
else
  "$ENGINE" build -f "$WORKDIR/context/Containerfile" -t "$CLIENT_OUTPUT_IMAGE" "$WORKDIR/context"
fi

echo "Built DeepSeek client image with the shared patched payload: $CLIENT_OUTPUT_IMAGE"
