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
author revision now includes native PP6+DSpark support; the former independent
patches remain under [`patches/upstreamed/`](patches/README.md) for audit only.

## Runtime configuration

```text
TP1 x PP6, partition 7,7,7,7,7,5
DSpark 5, local argmax reduction
max_model_len=1,048,576
max_num_batched_tokens=4096
max_num_seqs=64
KV=fp8_ds_mla, fixed 8 GiB per PP rank
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

The build checks out the pinned author revision, verifies that it is clean,
applies the active `patches/series` (currently empty because the required changes
were upstreamed), validates the PP+DSpark capabilities, compiles the relevant
Python files, and copies the complete pinned `vllm/` tree over the SM80 image.

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

## Validation gates

After startup, verify:

```bash
podman inspect deepseek-v41 --format '{{.State.Healthcheck.Status}}'
curl -fsS http://127.0.0.1:8000/health
podman exec deepseek-v41 nvidia-smi -L
swapon --show
```

Expected GPU KV capacity with the recorded fixed budget was 5,523,045 tokens.
The fixed 8-GiB value is per PP rank, not a global allocation. `max_num_seqs=64`
is an admission limit, not capacity for 64 one-million-token requests.
