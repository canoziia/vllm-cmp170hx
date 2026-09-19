#!/usr/bin/env bash
set -euo pipefail

# The deployment is managed explicitly with `podman compose up -d`. Disable the
# stale systemd wrapper (which referenced another compose file) without running
# its ExecStop against a live container.
systemctl disable deepseek-v41.service 2>/dev/null || true

# The old watcher assumed 48 vCPUs and two vNUMA regions. The current VM exposes
# 40 CPUs and four memory nodes, while virtual GPUs have no guest NUMA locality.
systemctl disable --now vllm-numa-pin.service 2>/dev/null || true

# Stopping the watcher does not undo taskset on already-running workers.
online_cpus=$(cat /sys/devices/system/cpu/online)
while read -r pid; do
  [[ -n "$pid" ]] || continue
  taskset -apc "$online_cpus" "$pid"
done < <(pgrep -f '^VLLM::Worker_PP' || true)

echo "Legacy startup wrapper disabled; NUMA watcher stopped."
echo "Existing vLLM workers, if any, can run on CPUs $online_cpus."
