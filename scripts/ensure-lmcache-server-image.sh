#!/usr/bin/env bash
set -euo pipefail

# Print the LMCache server image reference that client builds should take their
# payload from.
#
# Reuse is keyed on the moving ":latest" tag, not on the repository revision:
# most commits do not touch LMCache, so rebuilding the server on every commit
# would be waste. The build produces two tags - ":latest" and ":<short-rev>" -
# and ":latest" is what this script hands out and what the deployments run.
#
# Rebuild only when:
#   * ":latest" does not exist yet, or
#   * REBUILD_LMCACHE_IMAGE=1 is set.
#
# If ":latest" was built at a different revision, a note is printed, because that
# is the one case worth a human glance: the pinned payload digest or the patch
# series may have moved since. It is a note, not a rebuild.
#
# stdout carries nothing but the image reference, so callers do:
#     LMCACHE_SERVER_IMAGE=$(scripts/ensure-lmcache-server-image.sh)
# Everything else goes to stderr.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENGINE=${CONTAINER_ENGINE:-podman}
REVISION=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)
IMAGE_BASE=${LMCACHE_SERVER_IMAGE_BASE:-localhost/lmcache-server}
LATEST="$IMAGE_BASE:latest"

if [[ ${REBUILD_LMCACHE_IMAGE:-0} != 1 ]] && "$ENGINE" image exists "$LATEST" >/dev/null 2>&1; then
  built_rev=$("$ENGINE" image inspect --format '{{index .Labels "org.opencontainers.image.revision"}}' "$LATEST" 2>/dev/null || true)
  if [[ -n $built_rev && $built_rev != "$REVISION" ]]; then
    echo "note: $LATEST was built at revision $built_rev, repository is at $REVISION." >&2
    echo "note: fine if this commit does not touch LMCache; otherwise set REBUILD_LMCACHE_IMAGE=1." >&2
  fi
  echo "Reusing LMCache server image $LATEST (built at ${built_rev:-unknown})" >&2
else
  echo "Building LMCache server image $LATEST (at revision $REVISION)" >&2
  "$REPO_ROOT/scripts/build-lmcache-server-image.sh" >&2
fi

echo "$LATEST"
