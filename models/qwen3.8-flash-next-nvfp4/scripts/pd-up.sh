#!/bin/bash
# Launch the Qwen PD stack with plain podman.
#
# Topology (this is the part that matters):
#
#   prefill engine (GPU 0,1)  ->  lmcache server A :5557  -.
#                                                            >-- shared L2 on
#   decode  engine (GPU 2,3)  ->  lmcache server B :5567  -'   $LMCACHE_L2_PATH
#
# The LMCache MP server binds one engine's KV layout and segfaults in
# cuda/torch when a second engine registers against it, so each role gets its own
# server. The disaggregation still works because both servers are backed by the
# same L2 filesystem: prefill stores its prompt KV there, decode looks the same
# chunk keys up in the same directory and loads them instead of recomputing.
#
# Why not podman-compose? Version 1.3.0 silently ignores `ipc`, so every
# container would get a private /dev/shm. The connector moves KV through CUDA IPC
# and node1's working deployment runs everything with IpcMode=host, hence
# --ipc=host here. compose.pd.yml documents the same stack declaratively.
#
# Usage: pd-up.sh [prefill|decode|both]   (default: both, in parallel)
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
NS=qwen3.8-flash-next-nvfp4
SEC=(--security-opt label=disable --security-opt apparmor=unconfined)
MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4
REV=${QWEN_REVISION:-fc694b54fb0174e0913e6adf86691ef85a4ead47}
HFO='--hf-overrides={"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
L2ADAPTER='--l2-adapter={"type":"fs_native","base_path":"/lmcache-l2","num_workers":2,"use_odirect":false,"max_capacity_gb":500,"adopt_existing":true,"eviction":{"eviction_policy":"LRU"}}'

start_server() { # $1 role  $2 port  $3 http  $4.. gpus
  local role=$1 port=$2 http=$3; shift 3
  local devs=(); for g in "$@"; do devs+=(--device nvidia.com/gpu=$g); done
  podman rm -f $NS-lmcache-$role >/dev/null 2>&1 || true
  podman run -d --replace --name $NS-lmcache-$role --network host --ipc=host "${SEC[@]}" "${devs[@]}" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e PYTHONUNBUFFERED=1 \
    -v "$LMCACHE_L2_PATH:/lmcache-l2" \
    --entrypoint lmcache "$LMCACHE_IMAGE" server --host=127.0.0.1 --port=$port \
      --http-host=127.0.0.1 --http-port=$http --chunk-size=1600 --separate-object-groups \
      --l1-size-gb=${LMCACHE_L1_SIZE_GB:-8} --eviction-policy=LRU --max-workers=8 "$L2ADAPTER" >/dev/null
}

start_engine() { # $1 role
  local role=$1 port kvport cache server batched; local -a gpus
  case $role in
    prefill) port=8101; kvport=14579; cache=$VLLM_CACHE_PREFILL; server=5557; gpus=(0 1); batched=1600;;
    # step size must equal the resolved block size (1600): a prefill step that
    # advances more than one block desynchronises the PP ranks with MTP
    # enabled (observed: "hidden_states has 40 rows but this step expects 1536")
    # and is also the case the LMCache patch series warns corrupts mamba state.
    decode)  port=8102; kvport=14580; cache=$VLLM_CACHE_DECODE;  server=5567; gpus=(2 3); batched=1600;;
  esac
  podman rm -f $NS-$role >/dev/null 2>&1 || true
  podman run -d --replace --name $NS-$role --network host --ipc=host --stop-timeout 600 "${SEC[@]}" \
    --device nvidia.com/gpu=${gpus[0]} --device nvidia.com/gpu=${gpus[1]} \
    --ulimit nofile=1048576:1048576 \
    -v "$QWEN_MODEL_CACHE:/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4:ro" \
    -v "$cache:/root/.cache/vllm" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_PP_LAYER_PARTITION=26,22 -e VLLM_PP_DECODE_COHORT_BALANCE=1 \
    -e VLLM_PLE_CPU_OFFLOAD=1 \
    -e "VLLM_PLE_NVME_PATH=/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/$REV" \
    -e VLLM_PLE_NVME_BACKEND=pread -e VLLM_PLE_NVME_PREAD_WORKERS=48 -e VLLM_PLE_NEXT_CHUNK_PREFETCH=1 \
    -e SAFETENSORS_FAST_GPU=0 -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_IB_DISABLE=1 -e PYTHONUNBUFFERED=1 \
    "$QWEN_IMAGE" "$MODEL" --host=0.0.0.0 --port=$port --served-model-name=$MODEL --revision=$REV \
      --pipeline-parallel-size=2 --tensor-parallel-size=1 --quantization=modelopt --trust-remote-code \
      --max-model-len=1000000 --kv-cache-dtype=bfloat16 --kv-cache-memory=17179869184 \
      --max-num-batched-tokens=$batched \
      --speculative-config='{"method":"mtp","num_speculative_tokens":3}' \
      --disable-custom-all-reduce --no-enable-flashinfer-autotune \
      --enable-prefix-caching --prefix-match-unit=32 --mamba-cache-mode=align \
      --prefix-cache-retention-interval=1600 "$HFO" \
      "--kv-transfer-config={\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_ip\":\"127.0.0.1\",\"kv_port\":$kvport,\"kv_connector_extra_config\":{\"lmcache.mp.server_urls\":[\"127.0.0.1:$server\"]}}" \
      --enable-prompt-tokens-details --api-key="$VLLM_API_KEY" >/dev/null
}

wait_health() { for _ in $(seq 1 90); do
    [ "$(timeout 5 curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$1/health 2>/dev/null)" = 200 ] && return 0
    sleep 20; done; return 1; }

which=${1:-both}
if podman ps --format '{{.Names}}' | grep -qx "$NS-lmcache-prefill"; then echo "server A already running"; else start_server prefill 5557 18557 0 1; fi
if podman ps --format '{{.Names}}' | grep -qx "$NS-lmcache-decode";  then echo "server B already running"; else start_server decode  5567 18567 2 3; fi
case $which in
  prefill) start_engine prefill; wait_health 8101 || { echo "prefill failed"; exit 1; }; echo "prefill ready";;
  decode)  start_engine decode;  wait_health 8102 || { echo "decode failed";  exit 1; }; echo "decode ready";;
  both)    start_engine prefill; start_engine decode
           wait_health 8101 || { echo "prefill failed"; exit 1; }; echo "prefill ready"
           wait_health 8102 || { echo "decode failed";  exit 1; }; echo "decode ready";;
esac
