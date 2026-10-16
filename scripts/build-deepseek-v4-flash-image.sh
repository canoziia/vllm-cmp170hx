#!/usr/bin/env bash
# DeepSeek-V4-Flash image: the DeepSeek V4.1 image build (same pinned source,
# base image, native kernels and LMCache payload) with the V4-Flash series on
# top. Every V4-Flash switch defaults off (VLLM_DSV41_OPT_PROFILE=dsv4-flash).
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO_ROOT/models/deepseek-v4-flash/manifests/source.env"
if [[ ${ENABLE_PERF_DEBUG:-0} == 1 ]]; then
  export OUTPUT_IMAGE=${OUTPUT_IMAGE:-${DEFAULT_OUTPUT_IMAGE}-debug}
else
  export OUTPUT_IMAGE=${OUTPUT_IMAGE:-$DEFAULT_OUTPUT_IMAGE}
fi
export APPLY_SCRIPT="$REPO_ROOT/scripts/apply-deepseek-v4-flash-patches.sh"
exec "$REPO_ROOT/scripts/build-deepseek-v41-image.sh"
