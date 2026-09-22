#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 SOURCE_TREE" >&2; exit 2; }
SOURCE_TREE=$(realpath "$1")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SERIES="$REPO_ROOT/patches/vllm/series"

(cd "$REPO_ROOT" && sha256sum -c manifests/patches.sha256 >/dev/null)

[[ -d "$SOURCE_TREE/vllm" ]] || {
  echo "SOURCE_TREE must contain vllm/" >&2
  exit 3
}

while IFS= read -r patch_name; do
  [[ -n "$patch_name" && ${patch_name:0:1} != "#" ]] || continue
  patch_file="$REPO_ROOT/patches/vllm/$patch_name"
  [[ -f "$patch_file" ]] || { echo "Missing common vLLM patch: $patch_file" >&2; exit 4; }
  echo "Applying common vLLM patch: $patch_name"
  git -C "$SOURCE_TREE" apply --check "$patch_file"
  git -C "$SOURCE_TREE" apply "$patch_file"
done < "$SERIES"
