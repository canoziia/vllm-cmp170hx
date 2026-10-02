#!/usr/bin/env bash
# Layer NIXL onto an already-built engine image.
#
# Why an overlay: the engine images ship without NIXL, and NIXL is what vLLM's
# NixlConnector uses for the prefill->decode KV handoff (the official P/D shape is
# MultiConnector[NixlConnector + LMCacheMPConnector]). The `nixl-cu13` wheel is
# self-contained - it bundles the transfer engine, its UCX build *with* the GPU
# device API (UCX reports DRAM_SEG + VRAM_SEG, so KV can move GPU-to-GPU on a
# single host without any RDMA NIC) and the plugins (UCX, POSIX, GDS, ...).
#
# Usage: build-nixl-overlay-image.sh [engine-image] [output-tag]
#   defaults: localhost/vllm-backport:qwen3.8-flash-next-nvfp4  ->  <that>-nixl
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-podman}
BASE_IMAGE=${1:-localhost/vllm-backport:qwen3.8-flash-next-nvfp4}
OUTPUT_IMAGE=${2:-${BASE_IMAGE}-nixl}
NIXL_VERSION=${NIXL_VERSION:-1.5.0}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
WORKDIR=${WORKDIR:-$REPO_ROOT/.build/nixl-overlay}
mkdir -p "$WORKDIR/context"

cat > "$WORKDIR/context/Containerfile" <<CONTAINERFILE
ARG ENGINE_IMAGE
FROM \${ENGINE_IMAGE}

ARG NIXL_VERSION
ARG PIP_INDEX_URL
ARG EXPECT_NCCL=unknown
USER root

# Install exactly two wheels and nothing else:
#   nixl        - the python API (nixl/__init__.py, _api.py)
#   nixl-cu13   - the compiled bindings plus the bundled transfer engine, UCX
#                 (with the GPU device API) and the plugins
# --no-deps is deliberate. The wheels' own Requires-Dist is just torch+numpy,
# both already in the engine image, but the `nixl` *meta* package also pins
# nixl-cu12, and resolving that drags in a CUDA-12 stack and DOWNGRADES
# nvidia-nccl-cu13 (2.30.7 -> 2.29.7), i.e. it silently changes a component the
# engines rely on for pipeline parallelism. --no-deps keeps NCCL intact.
RUN python3 -m pip install --no-cache-dir --no-deps --index-url "\${PIP_INDEX_URL}" \
      "nixl==\${NIXL_VERSION}" "nixl-cu13==\${NIXL_VERSION}" \\
 && python3 -c "import nixl, nixl_cu13; assert nixl.HAVE_UCX_GPU_DEVICE_API, 'bundled UCX lacks the GPU device API'; print('nixl overlay OK')"

ARG EXPECT_NCCL=unknown
RUN python3 -c "import importlib.metadata as md, os, sys; \
      want = os.environ['EXPECT_NCCL']; got = md.version('nvidia-nccl-cu13'); \
      print('nvidia-nccl-cu13:', got); \
      assert want in ('unknown', got), f'NCCL changed by the overlay: {want} -> {got}'"

LABEL io.canoziia.nixl.version="\${NIXL_VERSION}" \\
      io.canoziia.nixl.engine-image="\${ENGINE_IMAGE}"
CONTAINERFILE

# The engine images deliberately override torch's NCCL pin (torch 2.13.0+cu130 asks
# for nvidia-nccl-cu13==2.29.7 while the image ships 2.30.7), and they were never
# re-resolved by pip. Any plain `pip install` makes the resolver "fix" that and
# silently downgrades NCCL, so the overlay installs with --no-deps and asserts here
# that the base image's NCCL survived.
BASE_NCCL=$("$ENGINE" run --rm --entrypoint python3 "$BASE_IMAGE" -c \
  "import importlib.metadata as md; print(md.version('nvidia-nccl-cu13'))" 2>/dev/null || echo unknown)
echo "base image NCCL: $BASE_NCCL (must be preserved)"

"$ENGINE" build --format docker \
  --build-arg ENGINE_IMAGE="$BASE_IMAGE" \
  --build-arg NIXL_VERSION="$NIXL_VERSION" \
  --build-arg PIP_INDEX_URL="$PIP_INDEX_URL" \
  --build-arg EXPECT_NCCL="$BASE_NCCL" \
  -t "$OUTPUT_IMAGE" "$WORKDIR/context"
echo "Built NIXL overlay image: $OUTPUT_IMAGE (from $BASE_IMAGE)"
