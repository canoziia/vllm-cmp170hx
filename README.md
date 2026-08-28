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

- TP1 x PP4: `12,12,12,12` without MTP and an MTP-balanced `13,13,13,9` profile.
- V2 Model Runner support.
- Native Qwen MTP on the last PP stage, including recursive multi-step drafting with the checkpoint's single MTP layer.
- Correct sampled-token and draft-token propagation through V2 PP pipeline slots.
- 1,000,000-token context using Qwen static YaRN (`factor=4`).
- BF16 QSA KV cache and BF16 GDN recurrent state for target/draft geometry parity.
- The 47.68-GiB PLE/n-gram embedding stays in the original safetensors files.
- Native parallel `pread` row access directly into the FP8 output buffer; `mmap` and a pure-Python fallback remain available.
- Bounded next-chunk PLE prefetch for V2: future rows are gathered during the current GPU forward and reused only after exact request, position, token, and n-gram-context validation.
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
2. Copies only the eleven source files touched by the patch series.
3. Verifies their SHA-256 hashes against `manifests/base-qwen38-flash-next.sha256`.
4. Applies `patches/*.patch` in lexical order.
5. Verifies all twelve final files against `manifests/final-qwen38-flash-next.sha256`.
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
remain enabled. The production scheduler allows 4,096 batched tokens, keeps up
to 32 request slots, and leaves `long_prefill_token_threshold=0` so a lone long
prefill can use the full available batch budget.

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

The MTP Compose profile uses three recursive draft steps and moves three target
layers away from PP3, which also owns the MTP layer, LM head, and sampler:

```text
VLLM_PP_LAYER_PARTITION=13,13,13,9
```

```json
{"method":"mtp","num_speculative_tokens":3}
```

It also keeps the Mamba recurrent state in BF16 so target and MTP cache geometry
matches across all four PP stages. The checkpoint has one MTP layer; MTP3 runs
that same layer recursively rather than loading three copies. Acceptance beyond
the first draft position is workload-dependent. The MTP1 benchmark results below
remain historical reference measurements.

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
VLLM_PLE_NVME_PREAD_WORKERS: "48"
VLLM_PLE_NEXT_CHUNK_PREFETCH: "1"
```

Set `VLLM_PLE_NVME_BACKEND=mmap` to use the mmap backend. The production example
uses 48 native `pread` workers because it performed best for random row access
on the tested host.

Next-chunk prefetch is bounded to at most the current scheduler batch and stores
only one future PLE result batch. It is enabled only on PP0, does not cache or
copy the 47.68-GiB table, and does not modify the scheduler. A future result is
used only when request IDs, absolute starts, token IDs, padded batch size, and
n-gram contexts all match the actual next batch; otherwise the worker falls back
to the normal synchronous gather. Mixed decode/prefill batches are not
prefetched, avoiding additional disk work on the decode path.

For correctness testing only, set `VLLM_PLE_NEXT_CHUNK_VERIFY=1`. Every prefetch
hit is then gathered synchronously a second time and compared byte-for-byte.
This deliberately adds duplicate disk I/O and must be disabled for performance
measurements and production.

Prefetch telemetry separates candidate mismatches and uncovered eligible
prefills from decode or mixed batches that are inherently ineligible. It reports
candidate hit rate, eligible-prefill step coverage, and valid-token coverage;
ineligible decode steps do not reduce these rates. Logs advance once per 128
eligible prefill lookups rather than once per 128 total model forwards.

The NVMe path currently requires TP=1 and DP=1. PP=4 is supported; only PP0 owns
the PLE layer, offload connector, and future-prompt state.

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
| V2, no MTP, 32K high-entropy prefill before native PLE `pread` | 724.8 tok/s |
| V2, no MTP, 32K high-entropy prefill after native PLE `pread` | 2,935 tok/s cold / 6,638 tok/s warm |
| Native PLE random-row microbenchmark, 1,024-token batch | approximately 4,100–4,770 tok/s |
| Strict random-token 128K prefill, batch 1,024, before next-chunk prefetch | 2,594 tok/s |
| Strict random-token 128K prefill, batch 1,024, with next-chunk prefetch | 3,611–3,791 tok/s |
| Four concurrent strict-random 32K prefills before next-chunk prefetch | 2,819 aggregate tok/s |
| Four concurrent strict-random 32K prefills with next-chunk prefetch | 3,724 aggregate tok/s |
| V2, no MTP, 16 concurrent independent requests | 3,061 aggregate prompt+output tok/s |
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

The final image replaces twelve files:

- Nine files for Qwen PP ownership, PLE NVMe offload, and bounded next-chunk prefetch.
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
