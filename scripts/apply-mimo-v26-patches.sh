#!/usr/bin/env bash
# Apply the MiMo-V2.6-Flash source stack to a clean checkout of the pinned
# DeepSeek V4.1 source:
#   DeepSeek V4.1 series -> shared vLLM series -> adaptive series   (as the
#   localhost/vllm-backport:deepseek-v4.1-flash image)
#   -> models/mimo-v2.6-flash/patches/series
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 SOURCE_TREE" >&2; exit 2; }
SOURCE_TREE=$(realpath "$1")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR="$REPO_ROOT/models/mimo-v2.6-flash"
SERIES="$MODEL_DIR/patches/series"

(cd "$REPO_ROOT" && sha256sum -c --quiet models/mimo-v2.6-flash/manifests/patches.sha256)

ENABLE_PERF_DEBUG=0 ENABLE_ADAPTIVE_VERIFICATION=1 \
  "$REPO_ROOT/scripts/apply-deepseek-v41-patches.sh" "$SOURCE_TREE"

while IFS= read -r patch_name; do
  [[ -n "$patch_name" && ${patch_name:0:1} != "#" ]] || continue
  patch_file="$MODEL_DIR/patches/$patch_name"
  [[ -f "$patch_file" ]] || { echo "Missing MiMo patch: $patch_file" >&2; exit 6; }
  echo "Applying MiMo patch: $patch_name"
  git -C "$SOURCE_TREE" apply --check "$patch_file"
  git -C "$SOURCE_TREE" apply "$patch_file"
done < "$SERIES"

git -C "$SOURCE_TREE" diff --check
python3 "$MODEL_DIR/tests/test_mimo_patches.py" "$SOURCE_TREE"
echo "MiMo-V2.6-Flash source validation completed successfully."
