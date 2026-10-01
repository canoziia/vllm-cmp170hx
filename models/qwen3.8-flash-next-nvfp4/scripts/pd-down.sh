#!/bin/bash
set -euo pipefail
podman rm -f qwen3.8-flash-next-nvfp4-decode qwen3.8-flash-next-nvfp4-prefill \
              qwen3.8-flash-next-nvfp4-lmcache-prefill qwen3.8-flash-next-nvfp4-lmcache-decode >/dev/null 2>&1 || true
echo stopped
