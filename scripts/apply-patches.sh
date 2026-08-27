#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 SOURCE_TREE" >&2
  echo "SOURCE_TREE must contain vllm/... paths extracted from the official image." >&2
}

[[ $# -eq 1 ]] || { usage; exit 2; }
SOURCE_TREE=$(realpath "$1")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST="$REPO_ROOT/manifests/base-qwen38-flash-next.sha256"
FINAL_MANIFEST="$REPO_ROOT/manifests/final-v2-mtp-r9.sha256"

[[ -d "$SOURCE_TREE/vllm" ]] || {
  echo "Missing $SOURCE_TREE/vllm" >&2
  exit 3
}

echo "Verifying official-image source hashes..."
(cd "$SOURCE_TREE" && sha256sum -c "$MANIFEST")

for patch_file in "$REPO_ROOT"/patches/*.patch; do
  echo "Applying $(basename "$patch_file")"
  (cd "$SOURCE_TREE" && git apply --check "$patch_file")
  (cd "$SOURCE_TREE" && git apply "$patch_file")
done

echo "Verifying patched output hashes..."
(cd "$SOURCE_TREE" && sha256sum -c "$FINAL_MANIFEST")
echo "Patch series applied successfully."
