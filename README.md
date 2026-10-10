# vllm-cmp170hx

Reproducible CMP 170HX deployments built from pinned upstream/model sources.

This `main` branch separates shared infrastructure from model-specific files:

```text
.
├── manifests/                  # pinned shared dependency/image metadata
├── monitoring/                 # standalone Prometheus + Grafana for all nodes
├── patches/
│   ├── lmcache/                # shared LMCache patch series
│   ├── router/                 # vllm-router (production-stack) patch series
│   └── vllm/                   # shared vLLM runtime fixes
├── scripts/
│   ├── apply-lmcache-patches.sh
│   ├── build-deepseek-v41-image.sh
│   ├── build-glm53-dflash2-image.sh          # GLM W4A16 + DFlash2 image
│   ├── build-lmcache-server-image.sh         # shared LMCache server image
│   ├── build-qwen38-image.sh
│   ├── build-router-image.sh   # patched vllm-router image
│   ├── benchmark-vllm.mjs      # decode, prefill, and counting benchmarks
│   ├── test-lmcache-patches.py
│   └── test-router-patches.py
└── models/
    ├── deepseek-v4.1-flash/
    │   ├── compose*.yml
    │   ├── manifests/          # pinned DeepSeek source/base image
    │   ├── patches/            # DeepSeek-only vLLM patches
    │   ├── scripts/            # DeepSeek-only diagnostics
    │   └── docs/
    ├── deepseek-v4-flash/      # PD pair on the DeepSeek V4.1 image
    │   └── compose.pd.yml
    ├── glm-5.3-flash/          # NVFP4 PD pair on the DeepSeek V4.1 image;
    │   │                       # W4A16 + DFlash2 PP4 on its own image
    │   ├── compose.pd.yml, compose.w4a16-dflash2.yml
    │   ├── manifests/          # GLM pins + SHA256 of the dflash2 stack
    │   ├── patches/dflash2/    # GLM DFlash2 vLLM series (on top of DeepSeek's)
    │   ├── native/ampere_marlin/  # compiled sm_80 Marlin decode (Apache-2.0)
    │   └── tests/
    └── qwen3.8-flash-next-nvfp4/
        ├── compose.yml
        ├── compose.pd.yml      # prefill + decode + LMCache + router
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
`models/deepseek-v4.1-flash/patches/`.

Build entry points stay in root `scripts/` under their existing names so
existing automation continues to work after the model directory rename. The
host cache and L2 paths retain their existing locations; renaming a repository
folder or container must not silently abandon persistent data.

## DeepSeek V4.1

See [`models/deepseek-v4.1-flash/README.md`](models/deepseek-v4.1-flash/README.md) for the
complete build, deployment, safety, and debugging guide.

Build the pinned DeepSeek image from the repository root:

```bash
CONTAINER_ENGINE=podman \
OUTPUT_IMAGE=localhost/vllm-backport:deepseek-v4.1-flash \
bash scripts/build-deepseek-v41-image.sh
```

The DeepSeek build produces the complete client image in one run: it reuses
`lmcache-server:latest` (building it only if absent, or when
`REBUILD_LMCACHE_IMAGE=1`) and copies that image's patched payload. There is
no separate client-payload build step.

Create a private deployment environment and start it:

```bash
cd models/deepseek-v4.1-flash
cp .env.example .env
chmod 600 .env
# Set a private VLLM_API_KEY and host paths in .env.

podman compose --podman-run-args=--ipc=host \
  -f compose.yml up -d
```

The repository model directories and Compose service/container names are
`deepseek-v4.1-flash` and `qwen3.8-flash-next-nvfp4`. On an existing host,
move each ignored `.env` into its renamed `models/<model>/` directory before
using Compose; retain its private values. Existing host cache, checkpoint and
L2 paths are intentionally unchanged. Old Podman containers keep their former
names/commands until explicitly replaced; changing Compose alone does not
upgrade or remove them.

Never commit `.env`, API keys, model weights, benchmark results, or host disk
UUIDs.

## Qwen3.8 Flash Next

See [`models/qwen3.8-flash-next-nvfp4/README.md`](models/qwen3.8-flash-next-nvfp4/README.md).
The complete image is built reproducibly from the same pinned author source,
the shared vLLM and LMCache series, and Qwen's model-specific process-isolated
NVMe PLE patch:

```bash
bash scripts/build-qwen38-image.sh
```

## GLM-5.3 Flash

See [`models/glm-5.3-flash/README.md`](models/glm-5.3-flash/README.md).
`nvidia/GLM-5.3-Flash-NVFP4` runs as a PP4 prefill/decode pair on the DeepSeek
V4.1 image: the pinned author source already implements `glm5next`, and the
image's shared vLLM series carries the NVFP4 load fix the model needs
(`patches/vllm/0007-nvfp4-marlin-scale-factor-amax.patch`).

## Monitoring

See [`monitoring/README.md`](monitoring/README.md). One Prometheus + Grafana
stack (podman compose, host network) scrapes the `/metrics` of every vLLM
engine, vllm-router and LMCache server listed in `.env`
(`VLLM_METRICS_TARGETS=name=host:port,...`) and provisions a throughput
overview (per instance and PD-deduplicated totals) plus the upstream vLLM
dashboards with `server`/`role`/`instance` variables.

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

## Upstream sources

This repository is a deployment layer: it pins upstream sources and adds only the
patches described above. The two repositories it is built from are:

| Upstream | What we take from it | Where it is pinned |
| --- | --- | --- |
| [`344303947/dsv41-flash-pp5-170hx`](https://github.com/344303947/dsv41-flash-pp5-170hx) | the CMP 170HX vLLM backport source that both model images are built from, together with its model/PP deployment recipes | `SOURCE_REPO` / `SOURCE_COMMIT` in `models/*/manifests/source.env` (commit `d63af5a4`) |
| [`vllm-project/production-stack`](https://github.com/vllm-project/production-stack) | **vllm-router**, the community-maintained single OpenAI-compatible endpoint in front of a prefill/decode pair | `manifests/router.env` (`vllm-stack-0.1.13`) |

The rest of the stack, pinned the same way:

| Upstream | What we take from it | Where it is pinned |
| --- | --- | --- |
| [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | the engine itself, through the backport source above | `models/*/manifests/source.env` |
| [`LMCache/LMCache`](https://github.com/LMCache/LMCache) | the KV cache engine; `localhost/lmcache-server` plus the shared patch series in `patches/lmcache/` | `manifests/lmcache.env` (v0.5.5, `05a013b2`) |
| [`ai-dynamo/nixl`](https://github.com/ai-dynamo/nixl) | NIXL 1.5.0 (the `nixl-cu13` wheel), the transfer engine vLLM's `NixlConnector` uses for a PD handoff; layered on as an overlay image | `scripts/build-nixl-overlay-image.sh` (`NIXL_VERSION`) |

### What we add on top of vllm-router

Each model directory carries a disaggregated deployment
(`models/<model>/compose.pd.yml`): one prefill engine, one decode engine, the
LMCache server both roles share, and a router that presents them as a single
endpoint.

We do not run upstream's router unmodified. `patches/router/` is applied on top
of the pinned release and the result ships as our own image (built by
`scripts/build-router-image.sh`, self-tested by `scripts/test-router-patches.py`):

* the prefill hop is capped in both response families. The Responses API ignores
  `max_tokens`, so without an explicit `max_output_tokens=1` the prefill engine
  generated the whole answer before the decode hop was asked to start, and the
  client waited for both generations;
* both hops' usage is merged into the standard fields the client sees
  (`cached_tokens` = `min(prefill, decode)`), with the raw per-hop numbers
  attached under `router_hops` for billing.

The router must be started with `--routing-logic disaggregated_prefill`; see the
model READMEs for the measured behaviour and the pitfalls of the alternatives.

## License

Apache-2.0; see [`LICENSE`](LICENSE).
