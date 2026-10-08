#!/usr/bin/env bash
# Build localhost/vllm-backport:glm-5.3-flash-w4a16-dflash2 from BASE_IMAGE:
#   pinned DeepSeek V4.1 source + DeepSeek/shared/adaptive series (exactly as
#   build-deepseek-v41-image.sh) + models/glm-5.3-flash/patches/dflash2,
#   the shared LMCache client payload, and the compiled sm_80 Marlin decode
#   extension installed as vllm._ampere_marlin_C.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DS_DIR="$REPO_ROOT/models/deepseek-v4.1-flash"
GLM_DIR="$REPO_ROOT/models/glm-5.3-flash"
# shellcheck disable=SC1091
source "$DS_DIR/manifests/source.env"   # SOURCE_REPO, SOURCE_COMMIT, BASE_IMAGE
# shellcheck disable=SC1091
source "$GLM_DIR/manifests/source.env"  # DEFAULT_OUTPUT_IMAGE (GLM), MM pins
ENGINE=${CONTAINER_ENGINE:-podman}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-$DEFAULT_OUTPUT_IMAGE}
MARLIN_MAX_JOBS=${MARLIN_MAX_JOBS:-4}
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/glm53-dflash2-build.XXXXXX")
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
"$REPO_ROOT/scripts/apply-glm53-dflash2-patches.sh" "$WORKDIR/source"

# Same LMCache client payload as the DeepSeek image, so this image stays a
# superset of localhost/vllm-backport:deepseek-v4.1-flash.
LMCACHE_SERVER_IMAGE=$("$REPO_ROOT/scripts/ensure-lmcache-server-image.sh")
LMCACHE_SERVER_ID=$("$ENGINE" image inspect --format '{{.Id}}' "$LMCACHE_SERVER_IMAGE" | cut -c1-12)
PAYLOAD_CID=$("$ENGINE" create --entrypoint /bin/true "$LMCACHE_SERVER_IMAGE")
"$ENGINE" cp "$PAYLOAD_CID:/opt/lmcache-patched/." "$WORKDIR/context/lmcache-payload/"
"$ENGINE" rm "$PAYLOAD_CID" >/dev/null
PAYLOAD_CID=

find "$WORKDIR/source/vllm" -name __pycache__ -type d -prune -exec rm -rf {} +
cp -a "$WORKDIR/source/vllm" "$WORKDIR/context/vllm"
cp -a "$GLM_DIR/native/ampere_marlin" "$WORKDIR/context/ampere_marlin"
cp "$REPO_ROOT/scripts/test-lmcache-patches.py" "$WORKDIR/context/test-lmcache-patches.py"
cp "$GLM_DIR/native/ampere-marlin-selfcheck.py" "$WORKDIR/context/ampere-marlin-selfcheck.py"
cp "$GLM_DIR/tests/dflash2/test_kv_draft_pages.py" "$GLM_DIR/tests/dflash2/test_chat_template.py" "$WORKDIR/context/"
cp "$GLM_DIR/chat_template.jinja" "$WORKDIR/context/chat_template.jinja"
INTEGRATION_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD)
SERIES_SHA=$(sha256sum "$GLM_DIR/patches/dflash2/series" | cut -c1-12)

cat >"$WORKDIR/context/Containerfile" <<CONTAINERFILE
FROM $BASE_IMAGE
USER root
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY lmcache-payload/ /opt/lmcache-patched/
COPY test-lmcache-patches.py /tmp/test-lmcache-patches.py
COPY ampere_marlin/ /tmp/ampere_marlin/
COPY ampere-marlin-selfcheck.py /tmp/ampere-marlin-selfcheck.py
COPY test_kv_draft_pages.py test_chat_template.py /tmp/
# GLM-5.3 chat template with a real thinking-off prompt (--chat-template)
COPY chat_template.jinja /opt/glm-5.3-flash/chat_template.jinja
ENV PYTHONPATH=/opt/lmcache-patched
# Compiled sm_80 W4A16 MoE decode (dflash2/0006): no GPU needed to build.
RUN MAX_JOBS=$MARLIN_MAX_JOBS bash /tmp/ampere_marlin/install-into-vllm.sh /tmp/ampere_marlin \\
    && rm -rf /tmp/ampere_marlin
RUN python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm /opt/lmcache-patched/lmcache \\
    && python3 -c 'import lmcache,sys,torch,lmcache.cuda_ops,lmcache.lmcache_native,lmcache.lmcache_fs; from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector; assert lmcache.__version__ == "0.5.5"; assert lmcache.__file__.startswith("/opt/lmcache-patched/"); assert sys.version_info[:2] == (3,12); assert torch.version.cuda == "13.0"; print(lmcache.__version__, lmcache.__file__, torch.__version__, torch.version.cuda)' \\
    && python3 /tmp/test-lmcache-patches.py \\
    && rm /tmp/test-lmcache-patches.py \\
    && env -u VLLM_GLM5_MARLIN_DECODE_LIB python3 /tmp/ampere-marlin-selfcheck.py \\
    && rm /tmp/ampere-marlin-selfcheck.py \\
    && python3 /tmp/test_kv_draft_pages.py \\
    && python3 /tmp/test_chat_template.py /opt/glm-5.3-flash/chat_template.jinja \\
    && rm /tmp/test_kv_draft_pages.py /tmp/test_chat_template.py
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \\
      org.opencontainers.image.revision="$SOURCE_COMMIT" \\
      io.canoziia.integration.revision="$INTEGRATION_REVISION" \\
      io.canoziia.upstream="$SOURCE_REPO@$SOURCE_COMMIT" \\
      io.canoziia.patch="early-pp-communicator-primer+adaptive+glm53-dflash2" \\
      io.canoziia.glm.dflash2-series="$SERIES_SHA" \\
      io.canoziia.glm.port-from="$MM_SOURCE_REPO@$MM_SOURCE_COMMIT" \\
      io.canoziia.glm.recipe="$MM_RECIPE_REPO@$MM_RECIPE_COMMIT" \\
      io.canoziia.ampere-marlin="vllm._ampere_marlin_C (sm_80, built in image)" \\
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
