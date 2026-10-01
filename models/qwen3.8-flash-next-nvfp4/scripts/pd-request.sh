#!/bin/bash
# Drive a disaggregated prefill -> decode request pair against the PD compose.
#
#   1. the prompt is sent to the prefill server with max_tokens=1, so only the
#      prompt is computed and its KV is stored in the shared LMCache tier;
#   2. the same prompt is sent to the decode server, which is expected to load
#      the prompt KV from LMCache and generate the answer.
#
# Usage: pd-request.sh "<prompt>" [max_tokens]
set -euo pipefail
: "${VLLM_API_KEY:?export VLLM_API_KEY}"
PREFILL=${PREFILL:-http://127.0.0.1:8101}
DECODE=${DECODE:-http://127.0.0.1:8102}
MODEL=${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}
PROMPT=${1:?prompt required}
MAXTOK=${2:-32}

body_prefill=$(printf '{"model":"%s","prompt":%s,"max_tokens":1,"temperature":0}' "$MODEL" "$(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')")
body_decode=$(printf '{"model":"%s","prompt":%s,"max_tokens":%s,"temperature":0}' "$MODEL" "$(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" "$MAXTOK")

echo "=== prefill (max_tokens=1) ==="
time curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d "$body_prefill" "$PREFILL/v1/completions" | head -c 400; echo

echo
echo "=== decode (max_tokens=$MAXTOK) ==="
time curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d "$body_decode" "$DECODE/v1/completions" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get("choices",[{}])[0].get("text","")[:400]))'
