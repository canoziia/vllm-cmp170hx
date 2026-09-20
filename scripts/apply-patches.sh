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

apply_series() {
  local series=$1 patch patch_path
  while read -r patch; do
    [[ -n "$patch" && $patch != \#* ]] || continue
    patch_path="$REPO_ROOT/patches/$patch"
    [[ -f "$patch_path" ]] || { echo "Missing series patch: $patch" >&2; exit 6; }
    echo "Applying $patch"
    git -C "$SOURCE_TREE" apply --check "$patch_path"
    git -C "$SOURCE_TREE" apply "$patch_path"
  done <"$series"
}

ENABLE_PERF_DEBUG=${ENABLE_PERF_DEBUG:-0}
[[ $ENABLE_PERF_DEBUG == 0 || $ENABLE_PERF_DEBUG == 1 ]] || exit 2
if [[ ${ENABLE_HOT_DSPARK_TOGGLE:-0} != 0 ]]; then
  echo "Use ENABLE_PERF_DEBUG=1 for the combined debug package" >&2
  exit 2
fi
apply_series "$REPO_ROOT/patches/series"
if [[ $ENABLE_PERF_DEBUG == 1 ]]; then
  apply_series "$REPO_ROOT/patches/optional/series.perf-debug"
fi

# The pinned author revision provides native PP6+DSpark; the local patch adds
# the early, idempotent PP communicator primer.
grep -q 'supports_aux_hidden_states_over_pp.*True' \
  "$SOURCE_TREE/vllm/models/deepseek_v4_1/nvidia/model.py"
grep -q 'spec_decode_needs_target_embed(vllm_config)' \
  "$SOURCE_TREE/vllm/models/deepseek_v4_1/nvidia/model.py"
grep -q '_pp_communicators_primed' \
  "$SOURCE_TREE/vllm/v1/worker/gpu_worker.py"
grep -q 'handler.broadcast_group' \
  "$SOURCE_TREE/vllm/v1/worker/gpu_worker.py"

compile_files=(
  "$SOURCE_TREE/vllm/models/deepseek_v4_1/nvidia/model.py"
  "$SOURCE_TREE/vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
  "$SOURCE_TREE/vllm/v1/worker/gpu_worker.py"
)
if [[ ${ENABLE_PERF_DEBUG:-0} == 1 ]]; then
  grep -q 'signal.SIGUSR2' \
    "$SOURCE_TREE/vllm/v1/worker/gpu/perf_debug.py"
  grep -q 'if not self.enabled and not self._reload_requested' \
    "$SOURCE_TREE/vllm/v1/worker/gpu/perf_debug.py"
  compile_files+=(
    "$SOURCE_TREE/vllm/v1/worker/gpu/perf_debug.py"
    "$SOURCE_TREE/vllm/v1/worker/gpu/model_runner.py"
    "$SOURCE_TREE/vllm/v1/worker/gpu/pp_utils.py"
  )
else
  [[ ! -e "$SOURCE_TREE/vllm/v1/worker/gpu/perf_debug.py" ]]
  ! grep -q 'perf_debug' "$SOURCE_TREE/vllm/v1/worker/gpu_worker.py"
fi

if [[ $ENABLE_PERF_DEBUG == 1 ]]; then
  compile_files+=(
    "$SOURCE_TREE/vllm/v1/core/sched/scheduler.py"
    "$SOURCE_TREE/vllm/v1/worker/gpu/model_runner.py"
    "$SOURCE_TREE/vllm/v1/worker/gpu/cudagraph_utils.py"
    "$SOURCE_TREE/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
  )
fi
git -C "$SOURCE_TREE" diff --check
python3 -m py_compile "${compile_files[@]}"
echo "Pinned DeepSeek V4.1 source validation completed successfully."
