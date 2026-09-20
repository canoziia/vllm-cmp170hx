# DeepSeek V4.1 Flash on six CMP 170HX GPUs

Reproducible minimal patches and a Podman Compose deployment for
`deepseek-ai/DeepSeek-V4.1-Flash` on six SM80 CMP 170HX GPUs.

## Source and image

- Author source: `https://github.com/344303947/dsv41-flash-pp5-170hx.git`
- Pinned source revision: `d63af5a472dc76b12d7a73d50a5af142844c15d1`
- Pinned SM80 base image:
  `docker.io/lazymio/vllm-backport@sha256:8094fcbab905a04a480b327f2761255e3d17cd8d39470cac9d76450bbb567f7e`
- Default output image: `localhost/deepseek-v41-cmp170hx:latest`

The model checkpoint is mounted read-only and is not modified. The two 94.4-GiB
Engram tables use the author's exact-size pinned CPU offload path. The pinned
author revision includes native PP6+DSpark support. The sole default patch
primes all persistent PP communicators before model and KV allocation so
first-use NCCL resource creation is deterministic and included in memory
accounting. Performance-debug runtime code is optional and is not included in
default images.

## Runtime configuration

```text
TP1 x PP6, partition 7,7,7,7,7,5
DSpark 5, local argmax reduction
max_model_len=1,048,576
max_num_batched_tokens=4096
max_num_seqs=32
KV=fp8_ds_mla, fixed at 8 GiB per rank
CPU Engram offload, prefix caching
NCCL Ring/Simple, P2P and IB disabled
```

No `--compilation-config` is supplied. vLLM automatically derives CUDA Graph
capture sizes from the six-token DSpark target verification width,
`max_num_seqs`, and the platform ceiling. Do not substitute request-count powers
of two: `cudagraph_capture_sizes` is measured in expanded forward tokens, not
HTTP concurrency.

GPU selection is done only through numeric NVIDIA CDI devices. The redundant
`NVIDIA_VISIBLE_DEVICES` variable is intentionally absent;
`CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA's selected-device ordering stable.

## Build

```bash
CONTAINER_ENGINE=podman \
OUTPUT_IMAGE=localhost/deepseek-v41-cmp170hx:latest \
bash scripts/build-image.sh
```

The default build applies only `patches/series` and does **not** include the
hot performance-debug runtime. Build a separate diagnostic image explicitly:

```bash
ENABLE_PERF_DEBUG=1 \
OUTPUT_IMAGE=localhost/deepseek-v41-cmp170hx:debug \
bash scripts/build-image.sh
```

The build checks out the pinned author revision, verifies that it is clean,
applies the default early-communicator patch and, only when requested, the
optional debug series. It validates and compiles the resulting Python files,
then copies the complete pinned `vllm/` tree over the SM80 image.

To inspect only the source result:

```bash
git clone https://github.com/344303947/dsv41-flash-pp5-170hx.git /tmp/dsv41
cd /tmp/dsv41
git checkout d63af5a472dc76b12d7a73d50a5af142844c15d1
/path/to/this/repo/scripts/apply-patches.sh /tmp/dsv41
```

## Deploy

```bash
cp .env.example .env
# Set VLLM_API_KEY and adjust model/cache paths if needed.

# One-time cleanup of stale local services from the previous deployment:
sudo bash scripts/disable-legacy-services.sh

podman compose -f compose.yml config
podman compose -f compose.yml up -d
podman logs -f deepseek-v41
```

The DAX-backed 286-GiB VM took about 53 minutes to become healthy in one clean
load, so the health-check start period is 75 minutes. The container does not
auto-restart: a failed load is expensive and GPU passthrough health must be
verified before trying again.

## Stop safely

Do not reset the VM or kill the container. A hard stop can leave CMP GSP/ACR
state set and make the next guest driver probe fail.

```bash
podman stop -t 600 deepseek-v41
podman rm deepseek-v41
```

The ten-minute grace period is also encoded in Compose.

## Optional LMCache deployment

`compose.lmcache.yml` is a standalone alternative to `compose.yml`: PP6,
seq32, 6 GiB GPU KV per rank, and a companion LMCache MP server using the same
image. It reserves a 16 GB L1 CPU cache, uses 1024-token chunks, separate object
groups, LRU and no disk L2. Both services use host IPC and the same six CDI GPUs.
LMCache binds its RPC and HTTP ports to loopback (5556 and 18556). Its healthcheck
checks both listening sockets; cache correctness still requires request testing.
The pinned image must include the author's compatible LMCache fork.

```bash
podman compose -f compose.lmcache.yml -f compose.debug.yml up -d
```

Use `compose.debug.yml` only with a debug-enabled image. Performance tracing
remains off initially. This deployment is experimental: added retention and
transfer resources need runtime memory/correctness validation. Neither LMCache
nor debug is enabled by the default Compose file.

## Combined optional debug package: DSpark compute toggle

```bash
ENABLE_PERF_DEBUG=1 OUTPUT_IMAGE=localhost/deepseek-v41-cmp170hx:debug bash scripts/build-image.sh
```

Use that image with `compose.yml` plus `compose.debug.yml`. Both performance
tracing and the DSpark compute toggle are included by this one build flag;
default builds include neither. The control-file environment must be present
at startup to capture both K=0 and K=5 target graphs. Performance tracing stays
disabled until explicitly enabled. DSpark initially stays on.

Write `0` (off) or `1` (on) atomically to `dspark-enabled` in the host cache
mount. The scheduler polls at most once per second; in-flight batches finish
under their original settings. Missing/invalid files retain the current mode
(initially on). No signal or restart is needed after startup.

```bash
printf '0\n' > "$VLLM_CACHE/dspark-enabled.tmp"
mv "$VLLM_CACHE/dspark-enabled.tmp" "$VLLM_CACHE/dspark-enabled"
# Repeat with 1 to enable.
```

Off skips draft backbone/Markov sampling/graph replay, but continues draft
context-KV maintenance so ongoing requests can safely resume drafting. Weights,
aux outputs, fixed-shape PP feedback and draft caches stay resident. This is
not a zero-overhead non-speculative baseline. Requires MRV2 DSpark, DP=1 and
adaptive verification disabled. Combined-image runtime validation is pending.

## Optional hot performance diagnostics

Default images contain no diagnostic runtime code or hot-path hooks. Build and
deploy a separate image with `ENABLE_PERF_DEBUG=1` when diagnosis is needed.
Within that diagnostic image the tracer is still disabled by default: disabled
workers do not create CUDA events, synchronize streams, copy timing tensors,
read control files, or write logs. Runtime control uses a shared JSON file plus
`SIGUSR2`, so subsequent enable/disable cycles do not require another restart.

Sample asynchronous timings on every tenth step, up to 256 samples on all PP
ranks:

```bash
bash scripts/perf-debug-control.sh enable decode-ab 10 256 all
# Run the benchmark, then inspect or explicitly stop early:
bash scripts/perf-debug-control.sh status
bash scripts/perf-debug-control.sh disable
python3 scripts/summarize-perf-debug.py \
  /root/app/deepseek-v41/cache/vllm-perf-debug/steps-decode-ab-rank*.jsonl
```

Captured JSONL includes PP rank, real/padded tokens, graph dispatch mode,
request/cohort sizes, CPU PP waits/enqueues, asynchronous GPU target/sampler/
draft/postprocess timings, feedback-broadcast timings, and per-request accepted
and rejected token counts.

CMP 170HX does not expose CUPTI CUDA kernel activities. For detailed GPU timing,
request a short diagnostic window that temporarily dispatches sampled steps
eagerly and installs per-module CUDA Event hooks:

```bash
bash scripts/perf-debug-control.sh detail dspark-detail 16 all
```

This reports per-layer target Engram, attention, MoE, ShadowSource, and last-rank
draft-module timing. Hooks are removed automatically when the sample limit is
reached and normal CUDA Graph dispatch resumes without restart. A bounded CPU
and dispatch `torch.profiler` trace is also available:

```bash
bash scripts/perf-debug-control.sh profile dspark-cpu 8 5
```

Both `detail` and `profile` are intentionally intrusive. Use ordinary `enable`
for low-overhead measurements in the production Graph path. The control
directory is inside the existing cache mount at
`/root/.cache/vllm-perf-debug`.

## Validation gates

After startup, verify:

```bash
podman inspect deepseek-v41 --format '{{.State.Healthcheck.Status}}'
curl -fsS http://127.0.0.1:8000/health
podman exec deepseek-v41 nvidia-smi -L
swapon --show
```

KV cache memory is fixed at 8 GiB per rank. `max_num_seqs=32` is an admission
limit, not capacity for 32 one-million-token requests. Record the reported token
capacity and PP2 free HBM for every new image because Graph coverage and model
allocations can change the runtime headroom.
