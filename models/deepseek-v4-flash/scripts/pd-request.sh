#!/bin/bash
# Two-step disaggregated request: prefill the prompt, then decode it.
#   1. prompt -> prefill server, max_tokens=1 (computes and stores the KV);
#   2. same prompt -> decode server (loads that KV and generates).
# Usage: pd-request.sh "<prompt>" [max_tokens]
set -euo pipefail
: "${VLLM_API_KEY:?export VLLM_API_KEY}"
PREFILL=${PREFILL:-http://127.0.0.1:8201}
DECODE=${DECODE:-http://127.0.0.1:8202}
MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Flash}
PROMPT=${1:?prompt required}
MAXTOK=${2:-32}
enc() { printf '%s' "$1" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'; }
echo "=== prefill (max_tokens=1) ==="
time curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d "$(printf '{"model":"%s","prompt":%s,"max_tokens":1,"temperature":0}' "$MODEL" "$(enc "$PROMPT")")" \
  "$PREFILL/v1/completions" | head -c 300; echo
echo
echo "=== decode (max_tokens=$MAXTOK) ==="
time curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d "$(printf '{"model":"%s","prompt":%s,"max_tokens":%s,"temperature":0}' "$MODEL" "$(enc "$PROMPT")" "$MAXTOK")" \
  "$DECODE/v1/completions" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(json.dumps(d.get("choices",[{}])[0].get("text","")[:400]))'
