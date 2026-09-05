# Qwen3.8 Flash Next NVFP4 on DGX Spark

Single DGX Spark (ARM64, GB10/SM121), TP1 × PP1, text-only serving of `RadixArk/Qwen3.8-Flash-Next-NVFP4`.

## Precision and checkpoint

Pinned revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
Architecture: `Qwen4ExpForConditionalGeneration`; quantization: ModelOpt NVFP4.
The original checkpoint is 125.91 GiB. Its 47.68-GiB FP8 E4M3 PLE table
(320,001,536 rows × 160 bytes, 128 logical shards in 10 files) remains in the
original read-only safetensors files. Only requested rows are read with native
parallel `pread`; no full table copy or residency is required.

Routed experts retain original packed NVFP4. Attention, shared experts,
embedding, LM head and MTP retain their checkpoint precision. No hybrid FP8
side-layer conversion, FP8 KV, or extra weight quantization is enabled.

## Clean-final source

This branch includes the twelve patches used by the tested `clean-final` image:
`0003`, `0007`–`0017`. See [patches/README.md](patches/README.md).
They provide PLE offload/prefetch, Mamba correctness fixes, GB10 FLA/deterministic
QSA fixes, and indexed/batched weight loading. Native PLE and licensed upstream
QSA top-k sources are included. GEMV, B12x, reference mmap, expanded prefill graph,
and other unaccepted experiments are **not** included.

## Build on Spark

The ARM64 base is pinned to:
`docker.io/vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`.
The build extracts all files in the base manifest, verifies hashes, applies all
patches in order, verifies the final manifest, and compiles native PLE and SM121a
top-k extensions. The resulting image records the Git commit in its OCI labels.

```bash
cp .env.example .env  # only for a new deployment; preserve an existing .env
TMPDIR=$HOME/app/tmp CONTAINER_ENGINE=podman \
  OUTPUT_IMAGE=localhost/vllm-qwen38-nvfp4-spark:latest \
  bash scripts/build-image.sh
```

On NixOS, put image temporary files under `/home`, not the small root filesystem.
The compose file uses explicit NVIDIA devices and read-only NixOS driver mounts
because this host has no working NVIDIA CDI setup. Adjust these for other hosts.
Set `.env` model-cache path to the directory containing `snapshots/` and choose an
API key before exposing the service. Never commit `.env`.

## Requested runtime configuration

```text
TP1 PP1, V2 runner, text-only
max_model_len=262144 (native; no YaRN or other context extension)
max_num_batched_tokens=8192
max_num_seqs=4
MTP=3
gpu_memory_utilization=0.90
KV=bfloat16, Mamba SSM=bfloat16, mamba-cache-mode=align
prefix-match-unit=32, prefix caching, chunked prefill
PLE native pread workers=16, bounded next-chunk prefetch enabled
NVFP4 expert CPU staging enabled
restart=unless-stopped
```

Batch8192 is the user's new deployment choice; historical clean-final performance
below used batch4096. Do not present that table as a new batch8192 benchmark.
The compile cache is isolated at `/root/.cache/vllm/clean-final-b8192-v1` under a
persistent host mount; experimental AOT caches must not be mixed into it.

```bash
podman-compose config
podman-compose up -d
podman logs -f qwen38-nvfp4-spark
```

## Historical clean-final results (batch4096)

Image `727b44a9c11b1ea18816a9541d8d5d6998b38b831400f17a203214f7ea4c2a0b`.
Normal GPU clocks after a power cycle; fixed inputs/seed, three-round medians,
unique cache salts and `cached_tokens=0`. SSE prefill is prompt tokens / TTFT;
pure decode excludes TTFT and the first token.

| Input | Prefill tok/s | Single-stream decode tok/s |
| --- | ---: | ---: |
| 32K website-style predictable text | 2126.68 | 42.23 |
| Technical document | 1918.74 | 32.03 |
| Code | 1914.12 | 33.72 |

- Random 128K, non-streaming one-output-token request: 68.03s, 1926.67 tok/s.
- Four concurrent requests: full-batch decode window 83.36 / 84.68 aggregate tok/s;
  makespan output throughput including TTFT/tail 60.73 / 77.31 tok/s.
- Model loading: ~805s before loader changes → ~474s (41% reduction). This excludes
  subsequent compile/warmup/graph capture. Model residency ~78.1 GiB.
- KV pool at one clean-final startup: 26.27 GiB / 959,898 tokens, 3.66×262144
  theoretical concurrency. Capacity varies with startup allocations; it is not
  the single-request context limit or a guarantee of four full-length requests.

Older ~800 tok/s prefill / ~17–24 tok/s decode measurements were collected while
GB10 was stuck at 507MHz. Recovery to ~2.4GHz followed a power cycle; that inference
speedup must not be attributed to code patches. The 2900 prefill / 45 general
decode target has **not** been achieved. A later fresh AOT-cache baseline differed
numerically from earlier cache runs; do not combine their best values or claim
cross-build bitwise model equivalence. See [SPARK-AUDIT.md](SPARK-AUDIT.md).

## Verification

`tests/test_expert_staging.py` compares original-loader parameter bytes against
staged loading on CUDA; set `ORIGINAL_ROUTED_EXPERTS` to the unmodified base file.
`tests/test_mamba_memmove.py` checks overlapping/non-overlapping copy cases.
`tests/test_prefix_cache.py` requires a healthy API and `VLLM_API_KEY`; use a fresh
`RUN_TAG`. It asserts positive output token IDs and exact cached/uncached text,
token IDs and first-token logprobs for three prompts. These are targeted tests,
not a full model quality evaluation.

Use `scripts/benchmark_vllm_decode.mjs` (Node18+, no dependencies) for reproducible
SSE measurements, with `--uncached --seed 42 --json-output <file>` and optionally
`--prompt-file`. Ensure no running/waiting requests before benchmarking. Do not
use the server's ten-second average prompt throughput as request performance.

## Operational safety

- Preserve original safetensors and read-only model mounts. Do not copy the whole PLE table.
- Swap16GiB at `/home/swapfile` is an OOM buffer, not model capacity. Track
  `pswpin/pswpout` deltas during tests. Its NixOS declaration requires the next
  authorized rebuild for persistent configuration; no reboot/rebuild is automated here.
- Rootless service requires user linger; `podman-restart.service` has been enabled
  on the test host, but full reboot recovery still needs a separate host-level test.
- Check API health, real OOM/illegal access/worker death, GPU clocks and memory
  after startup. Ordinary allocator warnings alone do not establish a fatal OOM.
- Keep reference/research images separate from production. Do not use skip-invalid
  Mamba copy guards as a correctness fix.
