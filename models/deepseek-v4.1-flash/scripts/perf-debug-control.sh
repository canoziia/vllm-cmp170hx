#!/usr/bin/env bash
set -euo pipefail

CONTAINER=${CONTAINER:-deepseek-v4.1-flash}
CACHE_ROOT=${VLLM_CACHE:-/srv/deepseek-v41/cache}
DEBUG_DIR=${VLLM_PERF_DEBUG_HOST_DIR:-$CACHE_ROOT/vllm-perf-debug}
CONTROL=$DEBUG_DIR/control.json
mkdir -p "$DEBUG_DIR"

# Signal only the PP workers of $CONTAINER. Deployments run with --pid=host, so a
# /proc scan from inside the container sees every process on the host (including
# other engines' "VLLM::Worker" processes, for which SIGUSR2 is fatal; `podman
# top` lists every host process too). A worker is therefore selected only if its
# cgroup equals the cgroup of the container's init process, and the number found
# must equal the engine's pipeline-parallel size, otherwise nothing is signaled.
signal_workers() {
  local init_pid cg pp pid comm
  local -a pids=()
  init_pid=$(podman inspect --format '{{.State.Pid}}' "$CONTAINER")
  [[ $init_pid =~ ^[0-9]+$ && $init_pid -gt 0 ]] || { echo "Container $CONTAINER is not running" >&2; return 1; }
  cg=$(cat "/proc/$init_pid/cgroup")
  pp=${PP_SIZE:-$(podman inspect --format '{{join .Args " "}}' "$CONTAINER" |
    grep -oE -- '--pipeline-parallel-size[= ]+[0-9]+' | grep -oE '[0-9]+$' || true)}
  [[ $pp =~ ^[0-9]+$ && $pp -gt 0 ]] || { echo "Cannot determine PP size; set PP_SIZE" >&2; return 1; }
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    comm=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    [[ $comm == VLLM::Worker* ]] || continue
    [[ $(cat "/proc/$pid/cgroup" 2>/dev/null) == "$cg" ]] || continue
    pids+=("$pid")
  done
  if [[ ${#pids[@]} -ne $pp ]]; then
    echo "Refusing to signal: found ${#pids[@]} worker(s) in $CONTAINER (${pids[*]:-none}), expected PP size $pp" >&2
    return 1
  fi
  kill -USR2 "${pids[@]}"
  echo "Signaled ${#pids[@]} PP workers of $CONTAINER: ${pids[*]}"
}

case "${1:-}" in
  enable)
    session=${2:-$(date +%Y%m%d-%H%M%S)}
    every=${3:-10}
    samples=${4:-256}
    ranks=${5:-all}
    if [[ $ranks == all ]]; then ranks_json='"all"'; else ranks_json="[$ranks]"; fi
    cat >"$CONTROL" <<JSON
{"enabled":true,"session":"$session","sample_every":$every,"max_samples":$samples,"max_pending":512,"flush_every":16,"ranks":$ranks_json,"torch_profile_steps":0}
JSON
    signal_workers
    echo "Enabled sampled timing: $CONTROL"
    ;;
  detail)
    echo "Disabled: per-worker eager switching can desynchronize PP shapes; use enable/profile" >&2
    exit 2
    session=${2:-$(date +%Y%m%d-%H%M%S)}
    samples=${3:-16}
    ranks=${4:-all}
    [[ $ranks == all ]] || {
      echo "detail changes PP execution shape and must run on all PP ranks" >&2
      exit 2
    }
    cat >"$CONTROL" <<JSON
{"enabled":true,"session":"$session","sample_every":1,"max_samples":$samples,"max_pending":128,"flush_every":1,"ranks":"all","force_eager":true,"module_detail":true,"torch_profile_steps":0}
JSON
    signal_workers
    echo "Enabled $samples detailed eager samples on all PP ranks: $CONTROL"
    ;;
  profile)
    echo "Disabled: repeated hot CPU profiler start/stop caused PP worker hangs; use async enable or external py-spy" >&2
    exit 2
    session=${2:-$(date +%Y%m%d-%H%M%S)}
    steps=${3:-8}
    rank=${4:-5}
    cat >"$CONTROL" <<JSON
{"enabled":true,"session":"$session","sample_every":1,"max_samples":$steps,"max_pending":64,"flush_every":1,"ranks":[$rank],"torch_profile_steps":$steps,"profile_ranks":[$rank],"profile_record_shapes":false,"profile_memory":false,"profile_with_stack":false}
JSON
    signal_workers
    echo "Enabled $steps-step CPU/dispatch torch profile on PP rank $rank: $CONTROL"
    ;;
  disable)
    printf '%s\n' '{"enabled":false}' >"$CONTROL"
    signal_workers
    echo "Disabled timing and flushed completed samples."
    ;;
  status)
    echo "Control: $CONTROL"
    [[ -f $CONTROL ]] && cat "$CONTROL" || echo '(missing: disabled)'
    echo "Outputs:"
    find "$DEBUG_DIR" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %f\n' | sort
    ;;
  *)
    cat >&2 <<'USAGE'
Usage:
  perf-debug-control.sh enable [session] [sample_every=10] [max_samples=256] [ranks=all|0,2,5]
  perf-debug-control.sh detail [session] [samples=16] [ranks=all]
  perf-debug-control.sh profile [session] [steps=8] [rank=5]
  perf-debug-control.sh disable
  perf-debug-control.sh status

Sampled timing is asynchronous and does not synchronize CUDA streams. `detail`
temporarily dispatches sampled steps eagerly and installs module hooks for
per-layer Engram/attention/MoE/shadow/draft GPU timing; it is intentionally
intrusive but auto-removes the hooks. `profile` provides CPU/dispatch traces;
CMP 170HX does not expose CUPTI CUDA kernel activities. The container must mount
VLLM_CACHE at /root/.cache (the production Compose does).
USAGE
    exit 2
    ;;
esac
