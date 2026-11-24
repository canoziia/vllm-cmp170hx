#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR="$REPO_ROOT/models/deepseek-v4.1-flash"
# shellcheck disable=SC1091
source "$MODEL_DIR/manifests/source.env"
# shellcheck disable=SC1091
source "$REPO_ROOT/manifests/lmcache.env"
ENGINE=${CONTAINER_ENGINE:-podman}
PYTHON=${PYTHON:-python3}
KEEP_WORKDIR=${KEEP_WORKDIR:-0}
WITH_LMCACHE=${WITH_LMCACHE:-1}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
# PREPARE_CONTEXT_ONLY: when set to a directory path, populate the full build
# context + Containerfile there and exit without invoking `$ENGINE build`
# (used by GitHub Actions `docker/build-push-action`).
PREPARE_CONTEXT_ONLY=${PREPARE_CONTEXT_ONLY:-}
# LMCACHE_PAYLOAD_SOURCE:
#   "auto" (default): reuse local lmcache-server:latest if already present,
#       otherwise extract directly from the small LMCACHE_PAYLOAD_IMAGE (~19 MiB)
#       and apply patches/lmcache/ without pulling the 17 GB server image.
#   "payload-image": always extract & patch directly from LMCACHE_PAYLOAD_IMAGE.
#   "server-image": ensure/build lmcache-server:latest and copy from it.
LMCACHE_PAYLOAD_SOURCE=${LMCACHE_PAYLOAD_SOURCE:-auto}
# SOURCE_MODE:
#   "vllm-branch" (default): build directly from this repo's `vllm` branch.
#   "patch-upstream": clone official upstream SOURCE_COMMIT and apply patches/.
SOURCE_MODE=${SOURCE_MODE:-vllm-branch}
VLLM_REF=${VLLM_REF:-$VLLM_BRANCH}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-$DEFAULT_OUTPUT_IMAGE}
PATCH_LABEL="upstream-sm80-tep4"

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/dsv41-main-sm80-build.XXXXXX")
PAYLOAD_CID=
cleanup() {
  if [[ -n $PAYLOAD_CID ]]; then
    "$ENGINE" rm -f "$PAYLOAD_CID" >/dev/null 2>&1 || true
  fi
  if [[ $KEEP_WORKDIR == 1 ]]; then
    echo "Keeping build workdir: $WORKDIR"
  else
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

req_cmds=(git "$PYTHON")
if [[ $VALIDATE_ONLY != 1 ]]; then
  req_cmds+=("$ENGINE")
fi
if [[ $SOURCE_MODE == "patch-upstream" || $WITH_LMCACHE == 1 ]]; then
  req_cmds+=(patch)
fi
for command in "${req_cmds[@]}"; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 2; }
done

mkdir -p "$WORKDIR/source" "$WORKDIR/context"

validate_vllm_tree() {
  local src=$1
  "$PYTHON" "$REPO_ROOT/scripts/test-vllm-common-patches.py" "$src"
  grep -q '_pp_communicators_primed' "$src/vllm/v1/worker/gpu_worker.py"
  "$PYTHON" "$MODEL_DIR/tests/test_triton_logits_workspace.py" "$src"
  "$PYTHON" "$MODEL_DIR/tests/test_dsv41_encoding_1017.py" "$src"
  "$PYTHON" "$MODEL_DIR/tests/test_deepseek_thinking.py" "$src"
  "$PYTHON" "$MODEL_DIR/tests/test_dsv41_round2_opts.py" "$src"
  grep -q 'enable_adaptive_verification: bool = False' "$src/vllm/config/speculative.py"
  "$PYTHON" "$MODEL_DIR/tests/test_adaptive_switch_routing.py" "$src"
  "$PYTHON" "$MODEL_DIR/tests/test_verification_profile.py" "$src"
  "$PYTHON" "$MODEL_DIR/tests/test_measured_selection.py" "$src"
  VLLM_DSV41_COST_MS_FUSED=1 "$PYTHON" "$MODEL_DIR/tests/test_measured_selection.py" "$src"
  "$PYTHON" "$MODEL_DIR/tests/test_cohort_packing.py" "$src"
  "$PYTHON" "$MODEL_DIR/tests/test_verification_dispatch.py" "$src"
  "$PYTHON" -m py_compile \
    "$src/vllm/dsv41_memory.py" \
    "$src/vllm/dsv41_opt_profile.py" \
    "$src/vllm/v1/core/sched/dsv41_prefill_sched.py" \
    "$src/vllm/models/deepseek_v41/common/engram.py" \
    "$src/vllm/models/deepseek_v41/attention.py" \
    "$src/vllm/models/deepseek_v41/nvidia/model.py" \
    "$src/vllm/models/deepseek_v41/nvidia/vl_model.py" \
    "$src/vllm/models/deepseek_v41/nvidia/model_state.py" \
    "$src/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/all_reduce.py" \
    "$src/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/columns.py" \
    "$src/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/markov_symm.py" \
    "$src/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/indexer.py" \
    "$src/vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/mqa_logits.py" \
    "$src/vllm/model_executor/kernels/linear/mxfp8/dsv41_dense_mxfp8.py" \
    "$src/vllm/model_executor/layers/fused_moe/dsv4_mxfp4_decode.py" \
    "$src/vllm/model_executor/layers/fused_moe/moe_prefill.py" \
    "$src/vllm/v1/attention/ops/dsv41_thin_woa.py" \
    "$src/vllm/v1/core/sched/dsv41_fixed_policy.py" \
    "$src/vllm/v1/core/sched/dsv41_measured_policy.py" \
    "$src/vllm/v1/core/sched/scheduler.py" \
    "$src/vllm/v1/worker/gpu/cudagraph_utils.py" \
    "$src/vllm/v1/worker/gpu/model_runner.py" \
    "$src/vllm/v1/worker/gpu/spec_decode/dspark/markov_argmax.py" \
    "$src/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py" \
    "$src/vllm/v1/worker/gpu/spec_decode/dspark/utils.py" \
    "$src/vllm/v1/worker/gpu_worker.py"
}

VLLM_SOURCE_LABEL=
if [[ $SOURCE_MODE == "vllm-branch" ]]; then
  if git -C "$REPO_ROOT" rev-parse --verify --quiet "${VLLM_REF}^{commit}" >/dev/null; then
    VLLM_REV=$(git -C "$REPO_ROOT" rev-parse "${VLLM_REF}^{commit}")
    echo "Exporting vLLM source from local branch/ref $VLLM_REF ($VLLM_REV)"
    git -C "$REPO_ROOT" archive "$VLLM_REV" | tar -x -C "$WORKDIR/source"
  elif git -C "$REPO_ROOT" rev-parse --verify --quiet "origin/${VLLM_REF}^{commit}" >/dev/null; then
    VLLM_REV=$(git -C "$REPO_ROOT" rev-parse "origin/${VLLM_REF}^{commit}")
    echo "Exporting vLLM source from origin/$VLLM_REF ($VLLM_REV)"
    git -C "$REPO_ROOT" archive "$VLLM_REV" | tar -x -C "$WORKDIR/source"
  else
    echo "Fetching vLLM branch $VLLM_REF from $VLLM_BRANCH_REPO"
    git -C "$WORKDIR/source" init -q
    git -C "$WORKDIR/source" remote add origin "$VLLM_BRANCH_REPO"
    git -C "$WORKDIR/source" fetch --depth=1 origin "$VLLM_REF"
    git -C "$WORKDIR/source" checkout -q --detach FETCH_HEAD
    VLLM_REV=$(git -C "$WORKDIR/source" rev-parse HEAD)
  fi
  if [[ -n ${VLLM_BRANCH_COMMIT:-} && ${ALLOW_UNPINNED_VLLM_BRANCH:-0} != 1 && $VLLM_REV != "$VLLM_BRANCH_COMMIT" ]]; then
    echo "Warning: $VLLM_REF is at $VLLM_REV (manifest pins $VLLM_BRANCH_COMMIT); set ALLOW_UNPINNED_VLLM_BRANCH=1 to suppress." >&2
  fi
  validate_vllm_tree "$WORKDIR/source"
  VLLM_SOURCE_LABEL="${VLLM_BRANCH}@${VLLM_REV}"
elif [[ $SOURCE_MODE == "patch-upstream" ]]; then
  git -C "$WORKDIR/source" init -q
  git -C "$WORKDIR/source" remote add origin "$SOURCE_REPO"
  git -C "$WORKDIR/source" fetch --depth=1 origin "$SOURCE_COMMIT"
  git -C "$WORKDIR/source" checkout -q --detach FETCH_HEAD
  APPLY_SCRIPT=${APPLY_SCRIPT:-$REPO_ROOT/scripts/apply-deepseek-v41-patches.sh}
  PYTHON="$PYTHON" "$APPLY_SCRIPT" "$WORKDIR/source"
  VLLM_SOURCE_LABEL="${SOURCE_REPO}@${SOURCE_COMMIT}+patches"
else
  echo "Unknown SOURCE_MODE=$SOURCE_MODE (expected vllm-branch or patch-upstream)" >&2
  exit 2
fi

if [[ $VALIDATE_ONLY == 1 ]]; then
  echo "Validated vLLM source ($VLLM_SOURCE_LABEL) successfully (VALIDATE_ONLY=1)."
  exit 0
fi

LMCACHE_LABEL="none"
if [[ $WITH_LMCACHE == 1 ]]; then
  mkdir -p "$WORKDIR/context/lmcache-payload"
  use_server_image=0
  if [[ $LMCACHE_PAYLOAD_SOURCE == "server-image" ]]; then
    use_server_image=1
  elif [[ $LMCACHE_PAYLOAD_SOURCE == "auto" && ${REBUILD_LMCACHE_IMAGE:-0} != 1 ]]; then
    LMCACHE_SERVER_TAG="${LMCACHE_SERVER_IMAGE_BASE:-localhost/lmcache-server}:latest"
    if "$ENGINE" image exists "$LMCACHE_SERVER_TAG" >/dev/null 2>&1 || \
       "$ENGINE" image inspect "$LMCACHE_SERVER_TAG" >/dev/null 2>&1; then
      use_server_image=1
    fi
  fi

  if [[ $use_server_image == 1 ]]; then
    LMCACHE_SERVER_IMAGE=$(CONTAINER_ENGINE="$ENGINE" "$REPO_ROOT/scripts/ensure-lmcache-server-image.sh")
    LMCACHE_SERVER_ID=$("$ENGINE" image inspect --format '{{.Id}}' "$LMCACHE_SERVER_IMAGE" | cut -c1-12)
    PAYLOAD_CID=$("$ENGINE" create --entrypoint /bin/true "$LMCACHE_SERVER_IMAGE")
    "$ENGINE" cp "$PAYLOAD_CID:/opt/lmcache-patched/." "$WORKDIR/context/lmcache-payload/"
    "$ENGINE" rm "$PAYLOAD_CID" >/dev/null
    PAYLOAD_CID=
    LMCACHE_LABEL="$LMCACHE_SERVER_IMAGE ($LMCACHE_SERVER_ID)"
  else
    "$ENGINE" pull "$LMCACHE_PAYLOAD_IMAGE"
    PAYLOAD_CID=$("$ENGINE" create --entrypoint /bin/true "$LMCACHE_PAYLOAD_IMAGE")
    "$ENGINE" cp "$PAYLOAD_CID:/payload/." "$WORKDIR/context/lmcache-payload/"
    "$ENGINE" rm "$PAYLOAD_CID" >/dev/null
    PAYLOAD_CID=
    PYTHON="$PYTHON" "$REPO_ROOT/scripts/apply-lmcache-patches.sh" "$WORKDIR/context/lmcache-payload"
    "$PYTHON" "$REPO_ROOT/scripts/test-lmcache-bounded-pd-groups.py" \
      "$WORKDIR/context/lmcache-payload/lmcache/v1/kv_layer_groups.py"
    LMCACHE_LABEL="$LMCACHE_PAYLOAD_IMAGE+patches"
  fi
  cp "$REPO_ROOT/scripts/test-lmcache-patches.py" "$WORKDIR/context/test-lmcache-patches.py"
fi

# Copy only source Python/C++ files from clean vLLM tree (preserving _version.py,
# compiled *.so extensions, vllm_flash_attn, and vllm-rs inside $BASE_IMAGE).
cp -a "$WORKDIR/source/vllm" "$WORKDIR/context/vllm"
rm -f "$WORKDIR/context/vllm/_version.py"
for nat in dsv4_moe dsv41_indexer dsv41_dense_mxfp8 dsv41_engram_cpu dsv41_swa dsv41_r2; do
  cp -a "$MODEL_DIR/native/$nat" "$WORKDIR/context/$nat"
done
INTEGRATION_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD)

cat >"$WORKDIR/context/Containerfile" <<CONTAINERFILE
FROM $BASE_IMAGE
USER root
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY dsv4_moe/ /tmp/dsv4_moe/
COPY dsv41_indexer/ /tmp/dsv41_indexer/
COPY dsv41_dense_mxfp8/ /tmp/dsv41_dense_mxfp8/
COPY dsv41_engram_cpu/ /tmp/dsv41_engram_cpu/
COPY dsv41_swa/ /tmp/dsv41_swa/
COPY dsv41_r2/ /tmp/dsv41_r2/
RUN (command -v g++ >/dev/null || (apt-get update -y && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*)) \\
    && bash /tmp/dsv4_moe/build.sh /usr/local/lib/python3.12/dist-packages/vllm \\
    && rm -rf /usr/local/lib/python3.12/dist-packages/vllm/build /tmp/dsv4_moe /root/.cache/torch_extensions \\
    && test -f /usr/local/lib/python3.12/dist-packages/vllm/_dsv4_moe_C.abi3.so \\
    && bash /tmp/dsv41_indexer/build.sh /usr/local/lib/python3.12/dist-packages/vllm \\
    && rm -rf /usr/local/lib/python3.12/dist-packages/vllm/build /tmp/dsv41_indexer /root/.cache/torch_extensions \\
    && test -f /usr/local/lib/python3.12/dist-packages/vllm/_dsv41_indexer_C.abi3.so \\
    && bash /tmp/dsv41_dense_mxfp8/build.sh /tmp/dsv41_dense_mxfp8/out --name _dsv41_dense_mxfp8 /tmp/dsv41_dense_mxfp8/dense_mxfp8.cu \\
    && cp /tmp/dsv41_dense_mxfp8/out/_dsv41_dense_mxfp8.so /usr/local/lib/python3.12/dist-packages/vllm/_dsv41_dense_mxfp8.so \\
    && rm -rf /tmp/dsv41_dense_mxfp8 /root/.cache/torch_extensions \\
    && test -f /usr/local/lib/python3.12/dist-packages/vllm/_dsv41_dense_mxfp8.so \\
    && bash /tmp/dsv41_swa/build.sh /usr/local/lib/python3.12/dist-packages/vllm \\
    && rm -rf /usr/local/lib/python3.12/dist-packages/vllm/build_swa /tmp/dsv41_swa /root/.cache/torch_extensions \\
    && test -f /usr/local/lib/python3.12/dist-packages/vllm/_dsv41_swa_C.abi3.so \\
    && bash /tmp/dsv41_r2/build.sh /usr/local/lib/python3.12/dist-packages/vllm \\
    && rm -rf /usr/local/lib/python3.12/dist-packages/vllm/build_r2 /tmp/dsv41_r2 /root/.cache/torch_extensions \\
    && test -f /usr/local/lib/python3.12/dist-packages/vllm/_dsv41_r2_C.abi3.so \\
    && bash /tmp/dsv41_engram_cpu/build.sh /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common \\
    && rm -rf /tmp/dsv41_engram_cpu \\
    && test -f /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common/_dsv41_engram_cpu.so \\
    && python3 -m compileall -f -q /usr/local/lib/python3.12/dist-packages/vllm
CONTAINERFILE

if [[ $WITH_LMCACHE == 1 ]]; then
  cat >>"$WORKDIR/context/Containerfile" <<'CONTAINERFILE'
COPY lmcache-payload/ /opt/lmcache-patched/
COPY test-lmcache-patches.py /tmp/test-lmcache-patches.py
ENV PYTHONPATH=/opt/lmcache-patched
RUN python3 -m compileall -q /opt/lmcache-patched/lmcache \
    && python3 /tmp/test-lmcache-patches.py \
    && rm /tmp/test-lmcache-patches.py
CONTAINERFILE
fi

cat >>"$WORKDIR/context/Containerfile" <<CONTAINERFILE
LABEL org.opencontainers.image.source="https://github.com/canoziia/vllm-cmp170hx" \\
      org.opencontainers.image.revision="$SOURCE_COMMIT" \\
      io.canoziia.integration.revision="$INTEGRATION_REVISION" \\
      io.canoziia.vllm.source="$VLLM_SOURCE_LABEL" \\
      io.canoziia.upstream="$SOURCE_REPO@$SOURCE_COMMIT" \\
      io.canoziia.base-image="$BASE_IMAGE" \\
      io.canoziia.patch="$PATCH_LABEL" \\
      io.canoziia.lmcache.payload-from="$LMCACHE_LABEL"
CONTAINERFILE

if [[ -n $PREPARE_CONTEXT_ONLY ]]; then
  rm -rf "$PREPARE_CONTEXT_ONLY"
  mkdir -p "$(dirname "$PREPARE_CONTEXT_ONLY")"
  mv "$WORKDIR/context" "$PREPARE_CONTEXT_ONLY"
  echo "Prepared build context in $PREPARE_CONTEXT_ONLY from $VLLM_SOURCE_LABEL"
  exit 0
fi

build_args=(-t "$OUTPUT_IMAGE" -f "$WORKDIR/context/Containerfile")
if [[ $ENGINE == podman ]]; then
  build_args=(--format docker "${build_args[@]}")
fi
"$ENGINE" build "${build_args[@]}" "$WORKDIR/context"
echo "Built $OUTPUT_IMAGE from $VLLM_SOURCE_LABEL"
