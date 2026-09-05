# Qwen3.8 Flash Next NVFP4 on DGX Spark

This branch builds an ARM64/SM121 image for serving:

```text
RadixArk/Qwen3.8-Flash-Next-NVFP4
```

on one 128-GiB NVIDIA DGX Spark. It keeps the checkpoint's approximately 47.68-GiB PLE/n-gram embedding table in its original safetensors files and reads only selected FP8 rows from local NVMe.

The branch is intentionally TP1 x PP1. Multi-Spark support is not included yet.

## Verified checkpoint metadata

Revision pinned by `.env.example`:

```text
7b719225242aacd3dbd3f9407468c2ee9a9d2594
```

The published metadata declares:

```text
architecture=Qwen4ExpForConditionalGeneration
model_type=qwen4_exp
quant_method=modelopt
quant_algo=NVFP4
checkpoint size=135,195,303,851 bytes (125.91 GiB)
PLE dtype=float8_e4m3fn
PLE logical shards=128
```

The PLE is still FP8 E4M3, not NVFP4. Removing its 47.68 GiB from unified-memory residency leaves about 78.23 GiB of checkpoint data before runtime packing and allocator overhead.

## Included patches

Five Spark/PP1 patches are carried:

1. Safetensors-backed native `pread`/`mmap` PLE row access.
2. Bounded V2 next-chunk PLE prefetch with exact key validation.
3. Hybrid Mamba scheduler-block alignment.
4. MTP-safe fine-grained recurrent partial-tail caching.
5. `UniProcExecutor` PLE worker lifecycle support required by TP1×PP1.

See [`patches/README.md`](patches/README.md) and [`SPARK-PORT.md`](SPARK-PORT.md).

## Build on the Spark

The official dedicated Qwen image has a Linux ARM64 manifest. Build natively on the Spark so the OpenMP helper is compiled for ARM64:

```bash
cp .env.example .env
./scripts/build-image.sh
```

Defaults:

```text
base image=docker.io/vllm/vllm-openai:qwen38-flash-next
output image=vllm-qwen38-nvfp4-spark:latest
container engine=docker
```

The build:

1. Extracts only the ten touched Python files from the official ARM64 image.
2. Verifies base SHA256 hashes.
3. Applies patches `0003`, `0007`, `0008`, `0009`, and `0010` in order.
4. Verifies final SHA256 hashes.
5. Compiles `native/ple_pread.c` inside the ARM64 image with GCC/OpenMP.
6. Produces a thin image on top of the official base.

Override values when needed:

```bash
CONTAINER_ENGINE=docker \
BASE_IMAGE=docker.io/vllm/vllm-openai:qwen38-flash-next \
OUTPUT_IMAGE=vllm-qwen38-nvfp4-spark:latest \
./scripts/build-image.sh
```

## Model files

Download the pinned revision into the Hugging Face cache, then set `.env` so `QWEN_MODEL_CACHE` points to the repository directory containing `snapshots/`:

```text
/home/nvidia/.cache/huggingface/hub/
└── models--RadixArk--Qwen3.8-Flash-Next-NVFP4/
    └── snapshots/
        └── 7b719225242aacd3dbd3f9407468c2ee9a9d2594/
```

Do not copy or convert the PLE table. `VLLM_PLE_NVME_PATH` points directly to this original snapshot.

## Run

```bash
docker compose config
docker compose up -d
docker compose logs -f
```

The initial single-Spark profile enables all four ported features together:

```text
TP1 x PP1
text-only
native 262,144-token context
MTP3
BF16 QSA KV cache
BF16 Mamba recurrent state
mamba-cache-mode=align
prefix-match-unit=32
prefix caching
chunked prefill
max_num_batched_tokens=4096
max_num_seqs=4
gpu_memory_utilization=0.90
PLE native pread workers=16
PLE next-chunk prefetch=enabled
```

The image contains all features in one build. For debugging only, next-chunk prefetch can be disabled without rebuilding:

```yaml
VLLM_PLE_NEXT_CHUNK_PREFETCH: "0"
```

The synchronous NVMe reader remains active.

## Correctness checks

Before treating the port as validated, verify on the Spark:

1. The build selects an optimized SM121 NVFP4 Linear backend.
2. The build selects an optimized SM121 NVFP4 MoE backend rather than emulation.
3. All 128 PLE shards are discovered and total row count matches the config.
4. Random PLE rows match `safetensors.safe_open` byte-for-byte.
5. A deterministic text completion is coherent.
6. Prefix caching works across repeated requests with MTP3 and `prefix-match-unit=32`.
7. A long uncached prompt reports PLE prefetch hits with no verification failure.
8. Startup and runtime logs contain no real OOM, worker death, or EngineCore failure.

For PLE verification only:

```yaml
VLLM_PLE_NEXT_CHUNK_VERIFY: "1"
```

This duplicates successful prefetch reads and must be disabled for performance measurements.

## DGX Spark validation

Validated on one NixOS DGX Spark (`aarch64`, GB10, SM121) with image:

```text
localhost/vllm-qwen38-nvfp4-spark:latest
d3c91d8246f63ae47020d76bf325e7960635c40e9d3ba0783c0edfc0e18d126e
```

Observed startup/runtime state:

```text
V2 Model Runner
quantization=modelopt_fp4
NVFP4 MoE backend=FLASHINFER_CUTLASS
resident model memory=78.1 GiB
weights + non-torch after warmup=81.68 GiB
CUDA Graph memory=0.95 GiB
KV cache=26.44 GiB / 966,121 tokens
262,144-token request concurrency=3.69x
health=200
restart count=0
fatal errors=0
```

PLE validation:

```text
47.68 GiB
320,001,536 rows × 160 bytes
128 logical shards / 10 physical PLE files
ARM64 native pread helper loaded
128 random rows byte-equal to safetensors.safe_open
next-chunk prefetch enabled
candidate hit rate=91.7%
prefill token coverage=93.7%
```

### Prefill throughput

Uncached random token-ID prompts, `max_tokens=1`, `max_num_batched_tokens=4096`:

| Workload | Runs | Throughput |
| --- | --- | ---: |
| Single 32K (first new-shape run) | 1 | 684.27 tok/s |
| Single 32K (warmed) | 1 | 815.18 tok/s |
| Single 128K | 3 | 773.07 / 771.18 / 772.27 tok/s |
| 4 × 32K concurrent | 1 | 809.67 aggregate tok/s |

All measured prefill responses reported `cached_tokens=0`.

### Decode throughput

Fixed short counting prompt, deterministic sampling, completion-token count divided by wall time:

| Concurrency | Output per request | Runs | Aggregate output throughput |
| ---: | ---: | --- | ---: |
| 1 | 512 | 3 | 23.65 / 23.34 / 24.24 tok/s |
| 4 | 512 | 2 | 81.72 / 81.28 tok/s |

The benchmark accumulated 4,474 accepted MTP draft tokens out of 4,491 drafted tokens (`99.62%`). A deterministic completion and an OpenAI chat-completions request both returned coherent output.

Cold startup is long because the checkpoint contains 206 safetensors files and about 296,475 tensor entries. The measured target+MTP model loading phase was about 17.5 minutes, followed by compile, kernel warmup and graph capture. This is dominated by many small ModelOpt/NVFP4 tensor load operations rather than PCIe transfer alone.

## Safety

- Keep the model snapshot read-only in the container.
- Change `VLLM_API_KEY` before exposing the API.
- Keep `SAFETENSORS_FAST_GPU=0`; PLE must remain on NVMe.
- Do not enable activation INT8/FP8 for the Marlin fallback path.
- Do not delete or rewrite the original PLE safetensors files.
