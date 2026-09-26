#!/usr/bin/env bash
set -euo pipefail

# Print the LMCache server image reference for the current repository revision,
# building it only if that revision's image is not present yet.
#
# stdout carries nothing but the image reference, so callers do:
#     LMCACHE_SERVER_IMAGE=$(scripts/ensure-lmcache-server-image.sh)
# All progress goes to stderr.
#
# REBUILD_LMCACHE_IMAGE=1 forces a rebuild even when the image exists.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENGINE=${CONTAINER_ENGINE:-podman}
REVISION=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)
IMAGE_BASE=${LMCACHE_SERVER_IMAGE_BASE:-localhost/lmcache-server}
IMAGE="$IMAGE_BASE:$REVISION"

if [[ ${REBUILD_LMCACHE_IMAGE:-0} != 1 ]] && "$ENGINE" image exists "$IMAGE" >/dev/null 2>&1; then
  echo "Reusing LMCache server image $IMAGE" >&2
else
  echo "Building LMCache server image $IMAGE" >&2
  OUTPUT_IMAGE="$IMAGE" "$REPO_ROOT/scripts/build-lmcache-server-image.sh" >&2
fi

echo "$IMAGE"
