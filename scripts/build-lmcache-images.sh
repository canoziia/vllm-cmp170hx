#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/lmcache.env"
ENGINE=${CONTAINER_ENGINE:-podman}
DEEPSEEK_BASE_IMAGE=${DEEPSEEK_BASE_IMAGE:-localhost/deepseek-v41-cmp170hx:latest}
SERVER_OUTPUT_IMAGE=${SERVER_OUTPUT_IMAGE:-localhost/deepseek-v41-lmcache:official-v0.5.5-patched}
CLIENT_OUTPUT_IMAGE=${CLIENT_OUTPUT_IMAGE:-localhost/deepseek-v41-cmp170hx:official-lmcache-v0.5.5-patched}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/lmcache-official-build.XXXXXX")

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

# Pull by digest and extract the exact official package. The official payload
# image is intentionally a tiny carrier for /payload, including compiled .so
# files and dist-info from the matching lmcache/vllm-openai release image.
"$ENGINE" pull "$LMCACHE_PAYLOAD_IMAGE"
payload_cid=$("$ENGINE" create --entrypoint /bin/true "$LMCACHE_PAYLOAD_IMAGE")
mkdir -p "$WORKDIR/context/lmcache-payload"
"$ENGINE" cp "$payload_cid:/payload/." "$WORKDIR/context/lmcache-payload/"
"$ENGINE" rm "$payload_cid" >/dev/null
payload_cid=

"$REPO_ROOT/scripts/apply-lmcache-patches.sh" "$WORKDIR/context/lmcache-payload"
cp "$REPO_ROOT/scripts/test-lmcache-patches.py" "$WORKDIR/context/"

cat > "$WORKDIR/context/Containerfile" <<CONTAINERFILE
FROM $LMCACHE_OFFICIAL_IMAGE AS server
USER root
COPY lmcache-payload/ /opt/lmcache-patched/
COPY test-lmcache-patches.py /opt/lmcache-patched/
ENV PYTHONPATH=/opt/lmcache-patched
RUN python3 -m compileall -q /opt/lmcache-patched/lmcache && python3 -c 'import lmcache,sys,torch,lmcache.cuda_ops,lmcache.lmcache_native,lmcache.lmcache_fs; assert lmcache.__version__ == "0.5.5"; assert lmcache.__file__.startswith("/opt/lmcache-patched/"); assert sys.version_info[:2] == (3,12); assert torch.version.cuda == "13.0"; print(lmcache.__version__, lmcache.__file__, torch.__version__, torch.version.cuda)' && python3 /opt/lmcache-patched/test-lmcache-patches.py
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \
      io.canoziia.lmcache.source="$LMCACHE_SOURCE_REPO@$LMCACHE_SOURCE_COMMIT" \
      io.canoziia.lmcache.official-image="$LMCACHE_OFFICIAL_IMAGE" \
      io.canoziia.lmcache.payload-image="$LMCACHE_PAYLOAD_IMAGE" \
      io.canoziia.lmcache.patch-series="patches/lmcache/series"

FROM $DEEPSEEK_BASE_IMAGE AS client
USER root
COPY lmcache-payload/ /opt/lmcache-patched/
ENV PYTHONPATH=/opt/lmcache-patched
RUN python3 -m compileall -q /opt/lmcache-patched/lmcache && python3 -c 'import lmcache,sys,torch,lmcache.cuda_ops,lmcache.lmcache_native,lmcache.lmcache_fs; from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector; assert lmcache.__version__ == "0.5.5"; assert lmcache.__file__.startswith("/opt/lmcache-patched/"); assert sys.version_info[:2] == (3,12); assert torch.version.cuda == "13.0"; print(lmcache.__version__, lmcache.__file__, torch.__version__, torch.version.cuda)'
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \
      io.canoziia.lmcache.source="$LMCACHE_SOURCE_REPO@$LMCACHE_SOURCE_COMMIT" \
      io.canoziia.lmcache.official-image="$LMCACHE_OFFICIAL_IMAGE" \
      io.canoziia.lmcache.payload-image="$LMCACHE_PAYLOAD_IMAGE" \
      io.canoziia.lmcache.patch-series="patches/lmcache/series"
CONTAINERFILE

build_common=(-f "$WORKDIR/context/Containerfile")
if [[ $ENGINE == podman ]]; then
  build_common=(--format docker "${build_common[@]}")
fi
"$ENGINE" build "${build_common[@]}" --target server -t "$SERVER_OUTPUT_IMAGE" "$WORKDIR/context"
"$ENGINE" build "${build_common[@]}" --target client -t "$CLIENT_OUTPUT_IMAGE" "$WORKDIR/context"

echo "Built server image: $SERVER_OUTPUT_IMAGE"
echo "Built client image: $CLIENT_OUTPUT_IMAGE"
