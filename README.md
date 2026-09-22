# vllm-cmp170hx

Reproducible CMP 170HX deployments built from pinned upstream/model sources.

This `main` branch separates shared infrastructure from model-specific files:

```text
.
├── manifests/                  # pinned shared dependency/image metadata
├── patches/
│   ├── lmcache/                # shared LMCache patch series
│   └── vllm/                   # shared vLLM runtime fixes
├── scripts/
│   ├── apply-lmcache-patches.sh
│   ├── build-deepseek-v41-image.sh
│   ├── build-deepseek-v41-lmcache-images.sh
│   ├── build-qwen38-image.sh
│   ├── benchmark-vllm.mjs      # decode, prefill, and counting benchmarks
│   └── test-lmcache-patches.py
└── models/
    ├── deepseek-v41/
    │   ├── compose*.yml
    │   ├── manifests/          # pinned DeepSeek source/base image
    │   ├── patches/            # DeepSeek-only vLLM patches
    │   ├── scripts/            # DeepSeek-only diagnostics
    │   └── docs/
    └── qwen3.8-flash-next/
        ├── compose.yml
        ├── manifests/          # pinned Qwen source/base image
        ├── patches/            # Qwen-only PLE/NVMe implementation
        └── native/
```

## Shared versus model-specific changes

Put a change in root `patches/` when the same source patch is intended for all
models using that component. Shared official LMCache patches are listed in
`patches/lmcache/series`; shared vLLM runtime fixes are listed in
`patches/vllm/series`.

Put model implementation patches, Compose files, source pins, and diagnostics
under `models/<model>/`. DeepSeek's vLLM patch series is therefore under
`models/deepseek-v41/patches/`.

Build entry points stay in root `scripts/` so automation can call them from a
stable location even as more model directories are added.

## DeepSeek V4.1

See [`models/deepseek-v41/README.md`](models/deepseek-v41/README.md) for the
complete build, deployment, safety, and debugging guide.

Build the pinned DeepSeek image from the repository root:

```bash
CONTAINER_ENGINE=podman \
OUTPUT_IMAGE=localhost/deepseek-v41-cmp170hx:latest \
bash scripts/build-deepseek-v41-image.sh
```

Build the official patched LMCache server and inject the identical LMCache
payload into the DeepSeek client image:

```bash
DEEPSEEK_BASE_IMAGE=localhost/deepseek-v41-cmp170hx:latest \
bash scripts/build-deepseek-v41-lmcache-images.sh
```

Create a private deployment environment and start it:

```bash
cd models/deepseek-v41
cp .env.example .env
chmod 600 .env
# Set a private VLLM_API_KEY and host paths in .env.

podman compose --podman-run-args=--ipc=host \
  -f compose.lmcache.yml up -d
```

Never commit `.env`, API keys, model weights, benchmark results, or host disk
UUIDs.

## Qwen3.8 Flash Next

See [`models/qwen3.8-flash-next/README.md`](models/qwen3.8-flash-next/README.md).
The complete image is built reproducibly from the same pinned author source,
the shared vLLM and LMCache series, and Qwen's model-specific process-isolated
NVMe PLE patch:

```bash
bash scripts/build-qwen38-image.sh
```

## Benchmark client

`scripts/benchmark-vllm.mjs` requires Node.js 18+ and has no npm dependencies.
Its file header documents the three standard test modes in copyable commands:

1. ordinary short-input decode: about 16 user tokens, 500 output tokens,
   `ignore_eos`;
2. prefill: approximately N input tokens and exactly one output token;
3. deterministic counting output for near-full speculative acceptance.

Show all options:

```bash
node scripts/benchmark-vllm.mjs --help
```

The client requires `VLLM_API_KEY`, assigns a unique `cache_salt` to every
request, rejects nonzero `cached_tokens`, uses server-reported usage counts, and
records exact streamed `token_ids` when available. Decode output includes both
token/s and speculative step/s.

## Current shared LMCache baseline

The shared manifest `manifests/lmcache.env` pins:

```text
LMCache v0.5.5
source commit 05a013b29da78cf2321b9b46ec5039dde2fb0bb0
Python 3.12 / CUDA 13.0 official image payloads by digest
```

Patch rationale and validation gates are documented in
[`patches/lmcache/README.md`](patches/lmcache/README.md).

## License

Apache-2.0; see [`LICENSE`](LICENSE).
