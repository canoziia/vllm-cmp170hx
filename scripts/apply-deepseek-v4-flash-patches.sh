#!/usr/bin/env bash
# Apply the DeepSeek-V4-Flash source stack to a clean checkout of the pinned
# DeepSeek V4.1 source:
#   DeepSeek V4.1 series -> shared vLLM series -> adaptive series   (as the
#   localhost/vllm-backport:deepseek-v4.1-flash image)
#   -> models/deepseek-v4-flash/patches/series
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 SOURCE_TREE" >&2; exit 2; }
SOURCE_TREE=$(realpath "$1")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR="$REPO_ROOT/models/deepseek-v4-flash"
SERIES="$MODEL_DIR/patches/series"

(cd "$REPO_ROOT" && sha256sum -c --quiet models/deepseek-v4-flash/manifests/patches.sha256)

ENABLE_PERF_DEBUG=${ENABLE_PERF_DEBUG:-0} ENABLE_ADAPTIVE_VERIFICATION=1 \
  "$REPO_ROOT/scripts/apply-deepseek-v41-patches.sh" "$SOURCE_TREE"

while IFS= read -r patch_name; do
  [[ -n "$patch_name" && ${patch_name:0:1} != "#" ]] || continue
  patch_file="$MODEL_DIR/patches/$patch_name"
  [[ -f "$patch_file" ]] || { echo "Missing DeepSeek-V4-Flash patch: $patch_file" >&2; exit 6; }
  echo "Applying DeepSeek-V4-Flash patch: $patch_name"
  git -C "$SOURCE_TREE" apply --check "$patch_file"
  git -C "$SOURCE_TREE" apply "$patch_file"
done < "$SERIES"

git -C "$SOURCE_TREE" diff --check
python3 -m py_compile \
  "$SOURCE_TREE/vllm/dsv41_opt_profile.py" \
  "$SOURCE_TREE/vllm/model_executor/layers/fused_moe/dsv4_mxfp4_decode.py" \
  "$SOURCE_TREE/vllm/model_executor/kernels/mhc/dsv4_mhc_v2_gate.py" \
  "$SOURCE_TREE/vllm/model_executor/kernels/linear/mxfp8/dsv4_dense_fp8.py" \
  "$SOURCE_TREE/vllm/model_executor/kernels/linear/scaled_mm/marlin.py" \
  "$SOURCE_TREE/vllm/v1/worker/gpu/pp_utils.py"
grep -q '"dsv4-flash"' "$SOURCE_TREE/vllm/dsv41_opt_profile.py"
python3 "$MODEL_DIR/tests/test_dsv4_verification.py" "$SOURCE_TREE"
echo "DeepSeek-V4-Flash source validation completed successfully."
