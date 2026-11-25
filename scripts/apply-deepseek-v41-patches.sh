#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 SOURCE_TREE" >&2; exit 2; }
SOURCE_TREE=$(realpath "$1")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR="$REPO_ROOT/models/deepseek-v4.1-flash"
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/vllm.env"
PYTHON=${PYTHON:-python3}

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

apply_series() {
  local series=$1 patch patch_path
  while read -r patch; do
    [[ -n "$patch" && $patch != \#* ]] || continue
    patch_path="$MODEL_DIR/patches/$patch"
    [[ -f "$patch_path" ]] || { echo "Missing series patch: $patch" >&2; exit 6; }
    echo "Applying $patch"
    git -C "$SOURCE_TREE" apply --check "$patch_path"
    git -C "$SOURCE_TREE" apply "$patch_path"
  done <"$series"
}

# Adaptive verification + topology-aware fusions are compiled in by
# default and remain inert unless enabled via runtime config / VLLM_DSV41_OPT_PROFILE.
ENABLE_ADAPTIVE_VERIFICATION=${ENABLE_ADAPTIVE_VERIFICATION:-1}
[[ $ENABLE_ADAPTIVE_VERIFICATION == 0 || $ENABLE_ADAPTIVE_VERIFICATION == 1 ]] || exit 2

apply_series "$MODEL_DIR/patches/series"
if [[ $ENABLE_ADAPTIVE_VERIFICATION == 1 ]]; then
  apply_series "$MODEL_DIR/patches/adaptive/series"
fi

# Verify all 45 shared vLLM invariants (async-PP, Mamba, structured-output spec, NVFP4 amax).
"$PYTHON" "$REPO_ROOT/scripts/test-vllm-common-patches.py" "$SOURCE_TREE"

# Verify DeepSeek-V4.1 upstream 0e3c8db structural & offline test suites.
grep -q '_pp_communicators_primed' "$SOURCE_TREE/vllm/v1/worker/gpu_worker.py"
"$PYTHON" "$MODEL_DIR/tests/test_triton_logits_workspace.py" "$SOURCE_TREE"
"$PYTHON" "$MODEL_DIR/tests/test_dsv41_encoding_1017.py" "$SOURCE_TREE"
"$PYTHON" "$MODEL_DIR/tests/test_deepseek_thinking.py" "$SOURCE_TREE"
"$PYTHON" "$MODEL_DIR/tests/test_dsv41_round2_opts.py" "$SOURCE_TREE"

compile_files=(
  "$SOURCE_TREE/vllm/dsv41_memory.py"
  "$SOURCE_TREE/vllm/dsv41_opt_profile.py"
  "$SOURCE_TREE/vllm/v1/core/sched/dsv41_prefill_sched.py"
  "$SOURCE_TREE/vllm/models/deepseek_v41/common/engram.py"
  "$SOURCE_TREE/vllm/models/deepseek_v41/attention.py"
  "$SOURCE_TREE/vllm/models/deepseek_v41/nvidia/model.py"
  "$SOURCE_TREE/vllm/models/deepseek_v41/nvidia/vl_model.py"
  "$SOURCE_TREE/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/indexer.py"
  "$SOURCE_TREE/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/mqa_logits.py"
  "$SOURCE_TREE/vllm/model_executor/kernels/linear/mxfp8/dsv41_dense_mxfp8.py"
  "$SOURCE_TREE/vllm/model_executor/layers/fused_moe/dsv4_mxfp4_decode.py"
  "$SOURCE_TREE/vllm/model_executor/layers/fused_moe/moe_prefill.py"
  "$SOURCE_TREE/vllm/v1/attention/ops/dsv41_thin_woa.py"
  "$SOURCE_TREE/vllm/v1/worker/gpu/spec_decode/dspark/markov_argmax.py"
  "$SOURCE_TREE/vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
  "$SOURCE_TREE/vllm/v1/worker/gpu_worker.py"
)

if [[ $ENABLE_ADAPTIVE_VERIFICATION == 1 ]]; then
  grep -q 'enable_adaptive_verification: bool = False' \
    "$SOURCE_TREE/vllm/config/speculative.py"
  "$PYTHON" "$MODEL_DIR/tests/test_adaptive_switch_routing.py" "$SOURCE_TREE"
  "$PYTHON" "$MODEL_DIR/tests/test_verification_profile.py" "$SOURCE_TREE"
  "$PYTHON" "$MODEL_DIR/tests/test_measured_selection.py" "$SOURCE_TREE"
  VLLM_DSV41_COST_MS_FUSED=1 "$PYTHON" "$MODEL_DIR/tests/test_measured_selection.py" "$SOURCE_TREE"
  VLLM_DSV41_COST_MS_FUSED=tp4 "$PYTHON" "$MODEL_DIR/tests/test_measured_selection.py" "$SOURCE_TREE"
  "$PYTHON" "$MODEL_DIR/tests/test_cohort_packing.py" "$SOURCE_TREE"
  "$PYTHON" "$MODEL_DIR/tests/test_verification_dispatch.py" "$SOURCE_TREE"
  compile_files+=(
    "$SOURCE_TREE/vllm/models/deepseek_v41/nvidia/model_state.py"
    "$SOURCE_TREE/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/all_reduce.py"
    "$SOURCE_TREE/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/columns.py"
    "$SOURCE_TREE/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/markov_symm.py"
    "$SOURCE_TREE/vllm/v1/core/sched/dsv41_fixed_policy.py"
    "$SOURCE_TREE/vllm/v1/core/sched/dsv41_measured_policy.py"
    "$SOURCE_TREE/vllm/v1/core/sched/scheduler.py"
    "$SOURCE_TREE/vllm/v1/worker/gpu/cudagraph_utils.py"
    "$SOURCE_TREE/vllm/v1/worker/gpu/model_runner.py"
    "$SOURCE_TREE/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"
  )
fi

git -C "$SOURCE_TREE" diff --check
"$PYTHON" -m py_compile "${compile_files[@]}"
echo "Pinned DeepSeek V4.1 upstream ($SOURCE_COMMIT) validation completed successfully."
