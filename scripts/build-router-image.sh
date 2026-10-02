#!/usr/bin/env bash
set -euo pipefail

# Build the patched vLLM production-stack router image.
#
# Everything network-dependent happens on the HOST: the pinned release tarball
# is fetched (through $ROUTER_FETCH_PROXY when set) and cached, then the patch
# series in patches/router/ is applied to the extracted tree. The slim build
# container only needs the PyPI mirror for the router's dependencies, and the
# patched vllm_router/ tree is copied over the installed one so what ships is
# exactly what the series produced.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/router.env"
ENGINE=${CONTAINER_ENGINE:-podman}
REVISION=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)
OUTPUT_IMAGE=${OUTPUT_IMAGE:-$DEFAULT_ROUTER_OUTPUT_IMAGE:$REVISION}
SERIES="$REPO_ROOT/patches/router/series"
CACHE=${ROUTER_TARBALL_CACHE:-/root/app/.cmp-build/router}
TARBALL="$CACHE/production-stack-$ROUTER_VERSION.tar.gz"
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/router-image-build.XXXXXX")
cleanup() { [[ ${KEEP_WORKDIR:-0} == 1 ]] && echo "Keeping $WORKDIR" || rm -rf "$WORKDIR"; }
trap cleanup EXIT

(cd "$REPO_ROOT" && sha256sum -c manifests/patches.sha256 >/dev/null)

for command in "$ENGINE" curl tar patch; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done

mkdir -p "$CACHE"
if [[ ! -s $TARBALL ]]; then
  echo "Fetching $ROUTER_VERSION"
  HTTPS_PROXY=${ROUTER_FETCH_PROXY:-${HTTPS_PROXY:-}} ALL_PROXY=${ROUTER_FETCH_PROXY:-${ALL_PROXY:-}} \
    curl -fsSL --retry 3 -o "$TARBALL.tmp" \
      "https://codeload.github.com/vllm-project/production-stack/tar.gz/refs/tags/$ROUTER_VERSION"
  mv "$TARBALL.tmp" "$TARBALL"
fi
mkdir -p "$WORKDIR/context"
tar xzf "$TARBALL" -C "$WORKDIR/context" --strip-components=1
cp "$TARBALL" "$WORKDIR/context/router.tar.gz"

while IFS= read -r patch_name; do
  [[ -n $patch_name && ${patch_name:0:1} != "#" ]] || continue
  echo "Applying $patch_name"
  patch --batch --forward -d "$WORKDIR/context/src" -p1 < "$REPO_ROOT/patches/router/$patch_name"
done < "$SERIES"

cp "$REPO_ROOT/scripts/test-router-patches.py" "$WORKDIR/context/"
cat > "$WORKDIR/context/Containerfile" <<CONTAINERFILE
FROM $ROUTER_BASE_IMAGE
USER root
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \\
    SETUPTOOLS_SCM_PRETEND_VERSION=${ROUTER_VERSION#vllm-stack-} \\
    PYTHONUNBUFFERED=1
COPY router.tar.gz /tmp/router.tar.gz
RUN pip install -q --disable-pip-version-check /tmp/router.tar.gz \\
 && rm -f /tmp/router.tar.gz
COPY src/vllm_router/ /usr/local/lib/python3.12/site-packages/vllm_router/
COPY test-router-patches.py /tmp/
RUN python3 -m compileall -q /usr/local/lib/python3.12/site-packages/vllm_router \\
 && python3 /tmp/test-router-patches.py
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \\
      org.opencontainers.image.revision="$REVISION" \\
      io.canoziia.router.version="$ROUTER_VERSION" \\
      io.canoziia.router.patch-series="patches/router/series"
CONTAINERFILE

unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy
if [[ $ENGINE == podman ]]; then
  "$ENGINE" build --format docker -f "$WORKDIR/context/Containerfile" -t "$OUTPUT_IMAGE" "$WORKDIR/context"
else
  "$ENGINE" build -f "$WORKDIR/context/Containerfile" -t "$OUTPUT_IMAGE" "$WORKDIR/context"
fi
"$ENGINE" tag "$OUTPUT_IMAGE" "$DEFAULT_ROUTER_OUTPUT_IMAGE:latest" >/dev/null
echo "Built router image: $OUTPUT_IMAGE (also tagged $DEFAULT_ROUTER_OUTPUT_IMAGE:latest)"
