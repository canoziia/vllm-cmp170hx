#!/usr/bin/env bash
# Apply the GLM-5.3-Flash W4A16 + DFlash2 source stack to a clean checkout of
# the pinned DeepSeek V4.1 source:
#   DeepSeek V4.1 series -> shared vLLM series -> adaptive series   (as the
#   localhost/vllm-backport:deepseek-v4.1-flash image)
#   -> models/glm-5.3-flash/patches/dflash2/series
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 SOURCE_TREE" >&2; exit 2; }
SOURCE_TREE=$(realpath "$1")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR="$REPO_ROOT/models/glm-5.3-flash"
SERIES="$MODEL_DIR/patches/dflash2/series"

(cd "$REPO_ROOT" && sha256sum -c --quiet models/glm-5.3-flash/manifests/patches.sha256)

# Same base as the DeepSeek image: revision/clean checks, DeepSeek + shared +
# adaptive series. No perf-debug package.
ENABLE_PERF_DEBUG=0 ENABLE_ADAPTIVE_VERIFICATION=1 \
  "$REPO_ROOT/scripts/apply-deepseek-v41-patches.sh" "$SOURCE_TREE"

while IFS= read -r patch_name; do
  [[ -n "$patch_name" && ${patch_name:0:1} != "#" ]] || continue
  patch_file="$MODEL_DIR/patches/dflash2/$patch_name"
  [[ -f "$patch_file" ]] || { echo "Missing GLM DFlash2 patch: $patch_file" >&2; exit 6; }
  echo "Applying GLM DFlash2 patch: $patch_name"
  git -C "$SOURCE_TREE" apply --check "$patch_file"
  git -C "$SOURCE_TREE" apply "$patch_file"
done < "$SERIES"

git -C "$SOURCE_TREE" diff --check
python3 "$REPO_ROOT/scripts/test-glm53-dflash2-patches.py" "$SOURCE_TREE"
echo "GLM-5.3-Flash DFlash2 source validation completed successfully."
