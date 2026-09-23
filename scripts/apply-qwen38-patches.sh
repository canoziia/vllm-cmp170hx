#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 SOURCE_TREE" >&2; exit 2; }
SOURCE_TREE=$(realpath "$1")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR="$REPO_ROOT/models/qwen3.8-flash-next"
# shellcheck disable=SC1091
source "$MODEL_DIR/manifests/source.env"

(cd "$REPO_ROOT" && sha256sum -c models/qwen3.8-flash-next/manifests/files.sha256 >/dev/null)

[[ -e "$SOURCE_TREE/.git" && -d "$SOURCE_TREE/vllm" ]] || {
  echo "SOURCE_TREE must be a git checkout containing vllm/" >&2
  exit 3
}
actual=$(git -C "$SOURCE_TREE" rev-parse HEAD)
[[ $actual == "$SOURCE_COMMIT" ]] || {
  echo "Source revision mismatch: expected $SOURCE_COMMIT, got $actual" >&2
  exit 4
}
[[ -z $(git -C "$SOURCE_TREE" status --short) ]] || {
  echo "Source checkout is not clean" >&2
  exit 5
}

apply_model_series() {
  local series=$1 patch_name patch_file
  while IFS= read -r patch_name; do
    [[ -n "$patch_name" && ${patch_name:0:1} != "#" ]] || continue
    patch_file="$MODEL_DIR/patches/$patch_name"
    [[ -f "$patch_file" ]] || {
      echo "Missing Qwen patch: $patch_file" >&2
      exit 6
    }
    echo "Applying Qwen patch: $patch_name"
    git -C "$SOURCE_TREE" apply --check "$patch_file"
    git -C "$SOURCE_TREE" apply "$patch_file"
  done < "$series"
}

apply_model_series "$MODEL_DIR/patches/series"
"$REPO_ROOT/scripts/apply-vllm-common-patches.sh" "$SOURCE_TREE"
# Generated-history checkpointing builds on the shared PP/Mamba lifecycle
# fixes, so it is intentionally applied after the shared series.
apply_model_series "$MODEL_DIR/patches/series.post-common"

python3 "$MODEL_DIR/tests/test_decode_checkpoint_scheduler.py" "$SOURCE_TREE"
python3 "$MODEL_DIR/tests/test_qwen_mtp_group_annotation.py" "$SOURCE_TREE"
python3 "$MODEL_DIR/tests/test_qsa_logits_workspace.py" "$SOURCE_TREE"
git -C "$SOURCE_TREE" diff --check
python3 -m compileall -q \
  "$SOURCE_TREE/vllm/config" \
  "$SOURCE_TREE/vllm/model_executor/layers/ple_offload_layer.py" \
  "$SOURCE_TREE/vllm/models/qwen4_exp" \
  "$SOURCE_TREE/vllm/v1/core" \
  "$SOURCE_TREE/vllm/v1/ple_offload" \
  "$SOURCE_TREE/vllm/v1/worker/gpu"
echo "Pinned Qwen3.8 source validation completed successfully."
