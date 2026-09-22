#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_DIR=${1:?usage: apply-lmcache-patches.sh SOURCE_DIR}
SERIES="$REPO_ROOT/patches/lmcache/series"

[[ -f "$SOURCE_DIR/lmcache/__init__.py" ]] || {
  echo "Not an LMCache source/payload root: $SOURCE_DIR" >&2
  exit 2
}

while IFS= read -r patch_name; do
  [[ -n "$patch_name" && ${patch_name:0:1} != "#" ]] || continue
  patch_file="$REPO_ROOT/patches/lmcache/$patch_name"
  [[ -f "$patch_file" ]] || { echo "Missing patch: $patch_file" >&2; exit 2; }
  patch --batch --forward -d "$SOURCE_DIR" -p1 < "$patch_file"
done < "$SERIES"

python3 -m compileall -q "$SOURCE_DIR/lmcache"
echo "Applied official LMCache patch series to $SOURCE_DIR"
