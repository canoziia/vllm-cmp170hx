#!/bin/bash
set -euo pipefail
podman rm -f deepseek-v4-flash-decode deepseek-v4-flash-prefill \
              deepseek-v4-flash-lmcache-prefill deepseek-v4-flash-lmcache-decode >/dev/null 2>&1 || true
echo stopped
