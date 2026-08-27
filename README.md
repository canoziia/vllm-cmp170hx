# vllm-cmp170hx

Reproducible patches and container build tooling for serving
`Qwen/Qwen3.8-Flash-Next-FP8` on four NVIDIA CMP 170HX GPUs.

The build starts from Qwen's dedicated vLLM image:

```text
docker.io/vllm/vllm-openai:qwen38-flash-next
```

and produces:

```text
localhost/vllm-cmp170hx:qwen3.8-flash-next
```

The Qwen checkpoint is **not modified**. The image contains only Python runtime
patches; model weights remain in the original Hugging Face safetensors snapshot.

## Features

- TP1 x PP4 with an explicit `12,12,12,12` layer partition.
- V2 Model Runner support.
- Native one-step Qwen MTP on the last PP stage.
- Correct sampled-token and draft-token propagation through V2 PP pipeline slots.
- 1,000,000-token context using Qwen static YaRN (`factor=4`).
- BF16 QSA KV cache and BF16 GDN recurrent state for target/draft geometry parity.
- The 47.68-GiB PLE/n-gram embedding stays in the original safetensors files.
- Native parallel `pread` row access directly into the FP8 output buffer; `mmap` and a pure-Python fallback remain available.
- Prefix caching, chunked prefill, xgrammar structured outputs, reasoning and tool parsers.
- Marlin FP8 MoE on SM80/CMP 170HX; no additional lossy weight or activation conversion.

## Tested hardware and software

- 4 x NVIDIA CMP 170HX, 65,344 MiB each, SM80
- PCIe-limited topology, therefore PP4 is preferred over TP4
- Podman with NVIDIA CDI (`nvidia.com/gpu=all`)
- vLLM base image `vllm/vllm-openai:qwen38-flash-next`
- Qwen snapshot `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`

The source manifests intentionally pin the files copied from the base image. If
the tag is updated upstream, the build stops rather than applying patches to
unknown source.

## Build

Requirements:

- `podman` (default) or Docker
- `git`, `sha256sum`, and Bash
- access to `docker.io/vllm/vllm-openai:qwen38-flash-next`

Build with Podman:

```bash
./scripts/build-image.sh
```

Override the engine, source image, or output tag:

```bash
CONTAINER_ENGINE=docker \
BASE_IMAGE=docker.io/vllm/vllm-openai:qwen38-flash-next \
OUTPUT_IMAGE=localhost/vllm-cmp170hx:qwen3.8-flash-next \
./scripts/build-image.sh
```

The script performs these steps:

1. Creates a stopped container from the official Qwen image.
2. Copies only the ten source files touched by the patch series.
3. Verifies their SHA-256 hashes against `manifests/base-qwen38-flash-next.sha256`.
4. Applies `patches/*.patch` in lexical order.
5. Verifies all eleven final files against `manifests/final-qwen38-flash-next.sha256`.
6. Builds a thin derived image containing the patched Python files and the small native PLE `pread` helper.

To inspect or apply the patches without building a container:

```bash
./scripts/apply-patches.sh /path/to/extracted/source-tree
```

See [`patches/README.md`](patches/README.md) for the functional split.

## Model files

Download `Qwen/Qwen3.8-Flash-Next-FP8` into the Hugging Face cache. The Compose
files pass the explicit model ID `Qwen/Qwen3.8-Flash-Next-FP8` to vLLM and mount
the cached repository at its standard Hugging Face hub path. `QWEN_MODEL_CACHE`
points to the repository directory containing `snapshots/`, not to an individual
safetensors file:

```text
/root/app/vllm/cache/huggingface/hub/
└── models--Qwen--Qwen3.8-Flash-Next-FP8/
    └── snapshots/
        └── bcd9f01ddc9cff2316eb84281bebcd5b058bddce/
```

Copy and edit the example environment:

```bash
cp .env.example .env
$EDITOR .env
```

## Run the default production configuration without MTP

The repository default is the final patched image with MTP disabled. PP4, V2,
1M context, NVMe PLE, prefix caching, chunked prefill and structured outputs
remain enabled.

```bash
podman compose config
podman compose up -d
curl -H "Authorization: Bearer $VLLM_API_KEY" \
  http://127.0.0.1:8000/health
```

## Enable MTP

The image always contains the MTP fixes. To enable native Qwen MTP, use the
explicit MTP Compose file:

```bash
podman compose -f compose.mtp.yml config
podman compose -f compose.mtp.yml up -d
```

The production-tested MTP setting is deliberately one step:

```json
{"method":"mtp","num_speculative_tokens":1}
```

The checkpoint has one MTP layer. Reusing it for more speculative steps is not
the tested configuration and may reduce acceptance quality.

## PLE / n-gram disk offload

The checkpoint's PLE table is approximately 47.68 GiB:

```text
320,001,536 rows x 160 FP8 values
128 logical shards in 33 safetensors files
```

The following variables keep it in the original files and read only requested
rows:

```yaml
VLLM_PLE_CPU_OFFLOAD: "1"
VLLM_PLE_NVME_PATH: /root/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/<snapshot>
VLLM_PLE_NVME_BACKEND: pread
VLLM_PLE_NVME_PREAD_WORKERS: "8"
```

Set `VLLM_PLE_NVME_BACKEND=mmap` to use the mmap backend. The production example
uses eight-worker `pread` because it performed better for random row access on
the tested host.

The NVMe path currently requires TP=1 and DP=1. PP=4 is supported; only PP0 owns
the PLE layer and offload connector.

## Long context

The checkpoint natively declares 262,144 positions. The Compose files request
approximately 1M using static YaRN:

```json
{
  "rope_type": "yarn",
  "factor": 4.0,
  "original_max_position_embeddings": 262144,
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10]
}
```

`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` is required because the requested length exceeds
the native declaration.

## Observed results

These numbers describe one PCIe-limited 4 x CMP 170HX host and are not universal:

| Configuration | Result |
| --- | ---: |
| V2, no MTP, 256-token steady decode | 50.47 tok/s |
| V2, MTP=1, steady decode run 1 | 61.91 tok/s |
| V2, MTP=1, steady decode run 2 | 65.74 tok/s |
| MTP acceptance in those runs | 62.0% / 64.1% |
| MTP KV capacity | 2,868,442 tokens |
| Maximum 1M-token concurrency | 2.87x |
| Two concurrent 192-token requests | 46.14 aggregate tok/s |
| Prefix-cache hit observed | 4,896 tokens |

The first request after a fresh build may trigger Triton JIT and is not a steady
performance measurement.

## Patch boundaries

The final image replaces eleven files:

- Eight files for Qwen PP ownership and PLE NVMe offload.
- Three files for Qwen MTP and V2 PP speculative feedback.

It does **not** include the discarded profiling kernels, Humming experiments,
BF16 linear pre-dequantization, activation quantization, zero-PLE diagnostics,
or custom small-M MoE kernels.

## Safety notes

- Change `VLLM_API_KEY` before exposing the service.
- Validate a new image with restart disabled before enabling an automatic restart loop.
- A healthy HTTP process alone does not prove that EngineCore workers are alive;
  monitor worker logs for fatal errors as well.
- The example requests access to all NVIDIA CDI devices and is intended for a
  dedicated four-GPU host.
