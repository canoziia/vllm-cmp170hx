#!/bin/bash
# Launch the DeepSeek-V4-Flash PD stack with plain podman.
#
# Topology (same shape as the Qwen PD deployment, which is verified):
#   prefill engine (GPU 4,5,6) -> lmcache server A :5558  -.
#                                                           >- shared L2 on
#   decode  engine (GPU 7,8,9) -> lmcache server B :5568  -'  $LMCACHE_L2_PATH
# One server per engine: the LMCache MP server binds a single engine's KV layout
# and segfaults when a second engine registers. Disaggregation happens through
# the shared L2 filesystem, so prefill's stored KV is read by decode.
#
# --ipc=host is mandatory (CUDA IPC); podman-compose 1.3.0 ignores `ipc`.
#
# Model-specific flags (tokenizer mode, KV dtype, speculative method) are the
# first things to check if the engine refuses to start: see .env.
#
# Usage: pd-up.sh [prefill|decode|both]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
NS=deepseek-v4-flash
SEC=(--security-opt label=disable --security-opt apparmor=unconfined)
MODEL=/models/DeepSeek-V4-Flash
SERVED=${DSFLASH_SERVED_NAME:-deepseek-ai/DeepSeek-V4-Flash}
KV_DTYPE=${DSFLASH_KV_CACHE_DTYPE:-fp8_ds_mla}
TOK_MODE=${DSFLASH_TOKENIZER_MODE:-deepseek_v41}
SPEC=${DSFLASH_SPECULATIVE_CONFIG:-'{"method":"mtp","num_speculative_tokens":1}'}
MAXLEN=${DSFLASH_MAX_MODEL_LEN:-1048576}
L2ADAPTER='--l2-adapter={"type":"fs_native","base_path":"/lmcache-l2","num_workers":2,"use_odirect":false,"max_capacity_gb":500,"adopt_existing":true,"eviction":{"eviction_policy":"LRU"}}'

start_server() { # role port http gpus...
  local role=$1 port=$2 http=$3; shift 3
  local devs=(); for g in "$@"; do devs+=(--device nvidia.com/gpu=$g); done
  podman rm -f $NS-lmcache-$role >/dev/null 2>&1 || true
  podman run -d --name $NS-lmcache-$role --network host --ipc=host "${SEC[@]}" "${devs[@]}" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e PYTHONUNBUFFERED=1 \
    -v "$LMCACHE_L2_PATH:/lmcache-l2" \
    --entrypoint lmcache "$LMCACHE_IMAGE" server --host=127.0.0.1 --port=$port \
      --http-host=127.0.0.1 --http-port=$http --chunk-size=${LMCACHE_CHUNK_SIZE:-1600} \
      --separate-object-groups --l1-size-gb=64 --eviction-policy=LRU --max-workers=8 "$L2ADAPTER" >/dev/null
}

start_engine() { # role
  local role=$1 port kvport cache server; local -a gpus; local -a extra=()
  case $role in
    prefill) port=8201; kvport=14581; cache=$VLLM_CACHE_PREFILL; server=5558; gpus=(4 5 6);;
    decode)  port=8202; kvport=14582; cache=$VLLM_CACHE_DECODE;  server=5568; gpus=(7 8 9)
             extra=(--enable-auto-tool-choice --tool-call-parser=${DSFLASH_TOOL_CALL_PARSER:-deepseek_v41}
                    --reasoning-parser=${DSFLASH_REASONING_PARSER:-deepseek_v41});;
  esac
  podman rm -f $NS-$role >/dev/null 2>&1 || true
  podman run -d --name $NS-$role --network host --ipc=host --stop-timeout 600 "${SEC[@]}" \
    --device nvidia.com/gpu=${gpus[0]} --device nvidia.com/gpu=${gpus[1]} --device nvidia.com/gpu=${gpus[2]} \
    --ulimit nofile=1048576:1048576 \
    -v "$MODEL_ROOT:/models:ro" -v "$cache:/root/.cache" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e HF_HUB_OFFLINE=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_PP_LAYER_PARTITION=15,15,13 \
    -e VLLM_CPU_OFFLOAD_GB_PER_RANK=0,0,0 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 -e VLLM_PP_DECODE_COHORT_BALANCE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False -e TMPDIR=/root/.cache/tmp \
    -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_IB_DISABLE=1 -e PYTHONUNBUFFERED=1 \
    "$DSFLASH_IMAGE" "$MODEL" --host=0.0.0.0 --port=$port --served-model-name="$SERVED" \
      --pipeline-parallel-size=3 --tensor-parallel-size=1 --trust-remote-code \
      --max-model-len=$MAXLEN --kv-cache-dtype=$KV_DTYPE --kv-cache-memory=6442450944 \
      --max-num-batched-tokens=${LMCACHE_CHUNK_SIZE:-1600} \
      --prefix-cache-retention-interval=1024 --disable-custom-all-reduce --enable-prefix-caching \
      --tokenizer-mode=$TOK_MODE --speculative-config="$SPEC" \
      "--kv-transfer-config={\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_ip\":\"127.0.0.1\",\"kv_port\":$kvport,\"kv_connector_extra_config\":{\"lmcache.mp.server_urls\":[\"127.0.0.1:$server\"]}}" \
      "${extra[@]}" --enable-prompt-tokens-details --api-key="$VLLM_API_KEY" >/dev/null
}

wait_health() { for _ in $(seq 1 120); do
    [ "$(timeout 5 curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null)" = 200 ] && return 0
    sleep 20; done; return 1; }

which=${1:-both}
podman ps --format '{{.Names}}' | grep -qx "$NS-lmcache-prefill" || start_server prefill 5558 18558 4 5 6
podman ps --format '{{.Names}}' | grep -qx "$NS-lmcache-decode"  || start_server decode  5568 18568 7 8 9
case $which in
  prefill) start_engine prefill; wait_health 8201 || { echo "prefill failed"; exit 1; }; echo "prefill ready";;
  decode)  start_engine decode;  wait_health 8202 || { echo "decode failed";  exit 1; }; echo "decode ready";;
  both)    start_engine prefill; start_engine decode
           wait_health 8201 || { echo "prefill failed"; exit 1; }; echo "prefill ready"
           wait_health 8202 || { echo "decode failed";  exit 1; }; echo "decode ready";;
esac
