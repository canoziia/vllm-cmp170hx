#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 SOURCE_TREE" >&2; exit 2; }
SOURCE_TREE=$(realpath "$1")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/source.env"

(cd "$REPO_ROOT" && sha256sum -c manifests/patches.sha256)

[[ -e "$SOURCE_TREE/.git" && -d "$SOURCE_TREE/vllm" ]] || {
  echo "SOURCE_TREE must be a git checkout containing vllm/" >&2
  exit 3
}
actual=$(git -C "$SOURCE_TREE" rev-parse HEAD)
[[ "$actual" == "$SOURCE_COMMIT" ]] || {
  echo "Source revision mismatch: expected $SOURCE_COMMIT, got $actual" >&2
  exit 4
}
[[ -z $(git -C "$SOURCE_TREE" status --short) ]] || {
  echo "Source checkout is not clean" >&2
  exit 5
}

while read -r patch; do
  [[ -n "$patch" && $patch != \#* ]] || continue
  patch_path="$REPO_ROOT/patches/$patch"
  [[ -f "$patch_path" ]] || { echo "Missing series patch: $patch" >&2; exit 6; }
  echo "Applying $patch"
  git -C "$SOURCE_TREE" apply --check "$patch_path"
  git -C "$SOURCE_TREE" apply "$patch_path"
done <"$REPO_ROOT/patches/series"

# The current pinned author revision has the formerly out-of-tree PP support.
grep -q 'supports_aux_hidden_states_over_pp.*True' \
  "$SOURCE_TREE/vllm/models/deepseek_v4_1/nvidia/model.py"
grep -q 'spec_decode_needs_target_embed(vllm_config)' \
  "$SOURCE_TREE/vllm/models/deepseek_v4_1/nvidia/model.py"

git -C "$SOURCE_TREE" diff --check
python3 -m py_compile \
  "$SOURCE_TREE/vllm/models/deepseek_v4_1/nvidia/model.py" \
  "$SOURCE_TREE/vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
echo "Pinned DeepSeek V4.1 source validation completed successfully."
