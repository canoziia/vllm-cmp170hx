# monitoring

Standalone Prometheus + Grafana for all vLLM engines (plus optional
vllm-router, LMCache server and GPU exporter). It only scrapes HTTP `/metrics`
and never touches the engines; run it on one node (node2, which has disk
space) and point it at every node.

```text
monitoring/
├── compose.yml                 # prometheus v3.5.0, grafana 12.1.1, gpu-exporter (profile gpu)
├── .env.example                # targets, ports, retention, data dir, admin password
├── prometheus/entrypoint.sh    # renders prometheus.yml + file_sd JSON from env, then execs prometheus
├── grafana/provisioning/       # datasource (uid prometheus) + dashboard provider (folder "vLLM")
├── grafana/dashboards/         # generated, do not edit by hand
│   ├── vllm-throughput.json    # our throughput overview (home dashboard)
│   ├── vllm-official.json      # upstream vLLM dashboard + server/role/instance variables
│   └── vllm-performance.json   # upstream performance statistics + the same variables
└── tools/
    ├── gen_dashboards.py       # regenerates grafana/dashboards/ (upstream copies in tools/upstream/)
    └── check_dashboards.py     # runs every panel query through the Grafana API (read-only)
```

## Run

Run from the repository checkout, like the model compose files (on node2:
`/root/app/vllm-cmp170hx/monitoring`, `.env` there with mode 0600). Data lives
outside the checkout in `MONITORING_DATA_DIR` (default `/root/app/monitoring-data`).

```bash
cd /root/app/vllm-cmp170hx/monitoring
cp .env.example .env && chmod 600 .env   # set GRAFANA_ADMIN_PASSWORD and targets
podman compose up -d                     # add `--profile gpu` for GPU metrics
```

- Grafana: `http://<host>:${GRAFANA_PORT:-13000}` (folder "vLLM", home = throughput overview).
- Prometheus: `127.0.0.1:${PROM_PORT:-19090}` on the host only (no auth);
  use `ssh -L 19090:127.0.0.1:19090` for `/targets`.
- Data: `${MONITORING_DATA_DIR}/{prometheus,grafana}`, bind mounts with podman
  `:U` (chowned to the container users). Retention: `PROM_RETENTION` (180d) and
  `PROM_RETENTION_SIZE` (50GB), whichever is hit first.
- Change targets: edit `.env`, then `podman compose down && podman compose up -d`
  (add `--profile gpu` to both if used; data is kept). podman-compose 1.3's
  `up --force-recreate <svc>` only restarts the old container, so new env is
  not picked up.
- Check: `set -a; . ./.env; set +a; python3 tools/check_dashboards.py`.

All services use host networking, like the model compose files: LMCache and
vllm-router bind 127.0.0.1, so only the node running this stack can scrape
its own ones, and no `host-gateway` / `host.docker.internal` is needed.

## Targets

```
VLLM_METRICS_TARGETS=NAME=HOST:PORT[;server=..;model=..;role=..;path=..],...
```

Every target gets the labels

| label | value |
|---|---|
| `instance` | `NAME` (replaces host:port; the address is kept in `address`) |
| `server` | `VLLM_SERVER_NAMES` mapping of HOST (`162.105.151.94=node2`), else HOST |
| `role` | `prefill` / `decode` if NAME contains it, else `unified` |
| `model` | NAME minus `-prefill`, `-decode`, `-nodeN` (vLLM's own `model_name` stays as well) |

`ROUTER_METRICS_TARGETS`, `LMCACHE_METRICS_TARGETS` and `GPU_METRICS_TARGETS`
use the same format (jobs `vllm-router`, `lmcache`, `gpu`); empty means no
targets. Unreachable targets just show `up == 0`.

**Auth**: vLLM's `--api-key` middleware only guards paths under `/v1`
(`vllm/entrypoints/serve/middleware/authenticate.py`, checked on our 0.13.1
images; `curl /metrics` without a key returns 200). No token is needed. If a
proxy in front requires one, put it in a 0600 file and set
`VLLM_METRICS_TOKEN_FILE`; it is mounted read-only and used as
`Authorization: Bearer` for the `vllm` job only.

## Dashboards and accounting

Metric names are those exposed by our images (vLLM V1 engine, `vllm:` prefix):
`generation_tokens_total`, `prompt_tokens_total`, `prompt_tokens_by_source_total{source}`,
`prompt_tokens_cached_total`, `prefix_cache_*`, `external_prefix_cache_*`,
`kv_cache_usage_perc`, `num_requests_running/waiting`,
`spec_decode_num_{drafts,draft_tokens,accepted_tokens,accepted_tokens_per_pos}_total`,
TTFT / ITL / E2E / queue histograms. The older `gpu_cache_usage_perc`,
`vllm:prompt_tokens` without `_total` etc. are not used.

PD pairs are not double counted (also noted on the dashboard):

- generated tokens and finished requests: `role!="prefill"` (prefill emits one
  token per request; decode finishes the user request);
- user prompt tokens: `role!="decode"` (decode sees the same prompt again,
  delivered as `external_kv_transfer`);
- `prompt tokens by source` shows each instance's real work: `local_compute`,
  `local_cache_hit` (GPU prefix cache), `external_kv_transfer` (LMCache / PD);
- spec decode: acceptance = accepted / draft tokens; tokens per step =
  1 + accepted / drafts; per-position acceptance = accepted_per_pos / drafts.

The upstream dashboards filter by `model_name`, which merged instances of the
same model; the generated copies add `server`, `role` and `instance` variables
and group histograms `by (le, instance)`.

## GPU metrics

`--profile gpu` starts `utkuozdemir/nvidia_gpu_exporter` on the monitoring
node; other nodes run only the exporter from their checkout:

```bash
cd monitoring   # .env: GPU_EXPORTER_LISTEN=<node IP reachable by Prometheus>
podman compose -f compose.gpu-exporter.yml up -d
```

and are added to `GPU_METRICS_TARGETS` on the Prometheus node. The exporter
has no auth; bind it to a LAN address and restrict the source (node1:
nftables table `vllm_monitoring`, only 162.105.151.94 and loopback may reach
19835).

Cost: each scrape runs one `nvidia-smi --query-gpu` (NVML, persistence mode on,
no CUDA context, no kernel launches on the GPUs). Measured on node1 (10 GPUs):
all fields ~1.8 s of driver/sys time, the default `GPU_QUERY_FIELDS` ~0.35 s;
with the 30 s `GPU_SCRAPE_INTERVAL` that is ~1% of one CPU core.

## Known limitations

- Counters reset when an engine restarts; `rate`/`increase` handle that, but
  the in-flight minute around a restart is lost. Per-hour/per-day bars need at
  least one full hour/day of history.
- Instances that are not running show as down (e.g. a PD decode that is
  stopped); totals simply omit them.
- `gpu` profile uses `utkuozdemir/nvidia_gpu_exporter` (nvidia-smi based) via
  CDI `nvidia.com/gpu=all`; dcgm-exporter is not used on CMP 170HX. It only
  sees the GPUs of its own node.
