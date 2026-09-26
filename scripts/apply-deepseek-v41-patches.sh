#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 SOURCE_TREE" >&2; exit 2; }
SOURCE_TREE=$(realpath "$1")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
MODEL_DIR="$REPO_ROOT/models/deepseek-v41"
source "$MODEL_DIR/manifests/source.env"

(cd "$REPO_ROOT" && sha256sum -c models/deepseek-v41/manifests/patches.sha256)

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

ENABLE_PERF_DEBUG=${ENABLE_PERF_DEBUG:-0}
[[ $ENABLE_PERF_DEBUG == 0 || $ENABLE_PERF_DEBUG == 1 ]] || exit 2
# Adaptive verification is compiled in by default. It stays inert unless the
# runtime spec config sets enable_adaptive_verification, so a production image
# carries the code without changing behaviour. Set
# ENABLE_ADAPTIVE_VERIFICATION=0 to reproduce an image built without it.
ENABLE_ADAPTIVE_VERIFICATION=${ENABLE_ADAPTIVE_VERIFICATION:-1}
[[ $ENABLE_ADAPTIVE_VERIFICATION == 0 || $ENABLE_ADAPTIVE_VERIFICATION == 1 ]] || exit 2
if [[ ${ENABLE_HOT_DSPARK_TOGGLE:-0} != 0 ]]; then
  echo "Use ENABLE_PERF_DEBUG=1 for the combined debug package" >&2
  exit 2
fi
apply_series "$MODEL_DIR/patches/series"
"$REPO_ROOT/scripts/apply-vllm-common-patches.sh" "$SOURCE_TREE"
# Order matters: the tracing package is maintained against the adaptive-modified
# PPHandler and draft-propose code, so the adaptive series has to go first.
if [[ $ENABLE_ADAPTIVE_VERIFICATION == 0 && $ENABLE_PERF_DEBUG == 1 ]]; then
  echo "ENABLE_PERF_DEBUG requires the adaptive series: optional/0002-hot-perf-debug.patch is" >&2
  echo "maintained against the adaptive-modified PPHandler/draft-propose code. Use" >&2
  echo "ENABLE_ADAPTIVE_VERIFICATION=1 (the default), or build the tracing image from the" >&2
  echo "commit that predates this rebase to reproduce the older combination." >&2
  exit 2
fi
if [[ $ENABLE_ADAPTIVE_VERIFICATION == 1 ]]; then
  apply_series "$MODEL_DIR/patches/adaptive/series"
fi
if [[ $ENABLE_PERF_DEBUG == 1 ]]; then
  apply_series "$MODEL_DIR/patches/optional/series.perf-debug"
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
python3 "$MODEL_DIR/tests/test_triton_logits_workspace.py" "$SOURCE_TREE"

compile_files=(
  "$SOURCE_TREE/vllm/model_executor/layers/sparse_attn_indexer.py"
  "$SOURCE_TREE/vllm/models/deepseek_v4_1/nvidia/model.py"
  "$SOURCE_TREE/vllm/v1/attention/ops/mqa_logits_triton.py"
  "$SOURCE_TREE/vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
  "$SOURCE_TREE/vllm/v1/worker/gpu_worker.py"
)
if [[ ${ENABLE_ADAPTIVE_VERIFICATION:-1} == 1 ]]; then
  grep -q 'def record_confidence_rows' \
    "$SOURCE_TREE/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"
  # Inert unless enabled: the runtime default must stay false, and every PP hook
  # the series adds must sit behind an adaptive check, otherwise shipping it by
  # default would silently change ordinary production runs.
  grep -q 'enable_adaptive_verification: bool = False' \
    "$SOURCE_TREE/vllm/config/speculative.py"
  grep -q 'if self.use_pp and self.adaptive_verification is not None:' \
    "$SOURCE_TREE/vllm/v1/worker/gpu/model_runner.py"
  grep -q 'if budget is not None or lengths is not None:' \
    "$SOURCE_TREE/vllm/v1/worker/gpu_worker.py"
  grep -q 'and not cls.supports_device_cpu_query_lens_mismatch()' \
    "$SOURCE_TREE/vllm/v1/attention/backend.py"
  grep -q 'use_adaptive_verification' \
    "$SOURCE_TREE/vllm/v1/attention/backend.py"
  grep -q 'supports_aux_hidden_states_over_pp' \
    "$SOURCE_TREE/vllm/models/deepseek_v4_1/nvidia/model.py"
  compile_files+=(
    "$SOURCE_TREE/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"
    "$SOURCE_TREE/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"
  )
fi

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
