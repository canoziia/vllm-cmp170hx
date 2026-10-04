#!/usr/bin/env bash
# CPU-only check of the GLM-5.3-Flash W4A16 + DFlash2 series on a plain source
# tree (no torch, no GPU, no container):
#   1. scripts/apply-glm53-dflash2-patches.sh on a clean pinned checkout
#      (sha256 manifests, DeepSeek + shared + adaptive series, then every
#      dflash2 patch with git apply --check + git apply, markers, defaults,
#      diagnostics absent, py_compile);
#   2. the torch-free unit tests against the real patched files.
#
# Usage: check-dflash2-series.sh [CLEAN_PINNED_CHECKOUT]
# Without an argument the pinned source is fetched (network needed).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
TESTS="$REPO_ROOT/models/glm-5.3-flash/tests/dflash2"
# shellcheck disable=SC1091
source "$REPO_ROOT/models/deepseek-v4.1-flash/manifests/source.env"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/glm53-dflash2-check.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

if [[ $# -ge 1 ]]; then
  cp -a "$(realpath "$1")" "$WORK/clean"
else
  git -C "$WORK" init -q clean
  git -C "$WORK/clean" remote add origin "$SOURCE_REPO"
  git -C "$WORK/clean" fetch -q --depth=1 origin "$SOURCE_COMMIT"
  git -C "$WORK/clean" checkout -q --detach FETCH_HEAD
fi
cp -a "$WORK/clean" "$WORK/before"
cp -a "$WORK/clean" "$WORK/after"

echo "== tree before the dflash2 series (= deepseek-v4.1-flash image source)"
ENABLE_PERF_DEBUG=0 ENABLE_ADAPTIVE_VERIFICATION=1 \
  "$REPO_ROOT/scripts/apply-deepseek-v41-patches.sh" "$WORK/before" >/dev/null
echo "== full GLM DFlash2 stack"
"$REPO_ROOT/scripts/apply-glm53-dflash2-patches.sh" "$WORK/after"

echo "== the series is exactly the difference between the two trees"
{ diff -r -q -x .git -x __pycache__ "$WORK/before/vllm" "$WORK/after/vllm" || true; } \
  | sed -E 's#^Files [^ ]*/after/(vllm/[^ ]*) and .*#M \1#; s#^Files [^ ]*/before/(vllm/[^ ]*) and .*#M \1#; s#^Only in [^ ]*/after/(vllm[^:]*): (.*)#A \1/\2#' \
  | sort >"$WORK/changed"
grep -h '^+++ b/' "$REPO_ROOT"/models/glm-5.3-flash/patches/dflash2/0*.patch \
  | sed 's#^+++ b/##' | sort -u >"$WORK/touched"
if ! diff <(awk '{print $2}' "$WORK/changed" | sort -u) "$WORK/touched"; then
  echo "trees differ outside the files the series touches" >&2
  exit 1
fi
echo "  $(wc -l <"$WORK/touched") files, all accounted for"

echo "== torch-free unit tests on the patched files"
export GLM_DFLASH2_TREE="$WORK/after" GLM_DFLASH2_PRISTINE_TREE="$WORK/before"
export PYTHONDONTWRITEBYTECODE=1
for test in test_kv_layout_dflash_pure.py test_adaptive_k_pure.py test_thin_gemm_pure.py; do
  echo "-- $test"
  python3 "$TESTS/$test" | tail -n 1
done
echo "GLM DFlash2 CPU check passed."
