# vllm-cmp170hx

**The fastest open-source DeepSeek-V4.1-Flash inference stack on NVIDIA SM80 — 200–500 tok/s single-stream and up to 10K tok/s prefill (8K at 1M) on 4× 80G PCIe GPUs with unmodified official weights, built directly on official `vllm-project/vllm` and `LMCache`.**

## Models

### DeepSeek V4.1 Flash

See [`models/deepseek-v4.1-flash/README.md`](models/deepseek-v4.1-flash/README.md)
for the complete architecture overview, optimization profiles, native extensions,
deployment guide, and both 180 W & 250 W benchmarks.

#### Measured TEP4 Benchmarks (`4× 80G SM80 PCIe Gen4`, 250 W, `compose.yml` + LMCache)

`TP=4 + EP=4`, unmodified official MXFP4/FP8 weights, `nvfp4_ds_mla` KV cache
(`2,014,502` GPU KV tokens = `1.92× 1M` + LMCache `8 GiB` L1 RAM / `500 GiB`
NVMe L2), `VLLM_DSV41_OPT_PROFILE=tep4_fused`, `enable_adaptive_verification=true`,
`0` OOM across full 1M prefill and 32-concurrency stress.

**Cold Prefill (`tok/s`, `c=1`, `cached_tokens=0`)**

| Target Length | 64K | 128K | 256K | 512K | 1M |
|---|---:|---:|---:|---:|---:|
| **Prefill (`tok/s`)** | **9,928.7** | **10,027.5** | **9,789.3** | **9,242.6** | **7,992.4** |

**Decode Throughput & Step Latency (`full_batch_tok_s` / `step_ms`, `max_tokens=500`)**

| Concurrency `c` | `prose` | `code` | `counting` |
|---:|---:|---:|---:|
| **1** | **212.0** (`9.81 ms`, acc 2.07) | **491.6** (`11.17 ms`, acc 5.43) | **567.6** (`10.47 ms`, acc 5.88) |
| **2** | **353.2** (`11.88 ms`, acc 2.07) | **779.6** (`14.83 ms`, acc 5.51) | **935.3** (`13.15 ms`, acc 5.88) |
| **4** | **578.2** (`13.57 ms`, acc 1.95) | **1,173.3** (`19.35 ms`, acc 5.41) | **1,480.3** (`16.66 ms`, acc 5.88) |
| **8** | **813.3** (`18.86 ms`, acc 1.91) | **1,710.7** (`26.25 ms`, acc 5.45) | **2,095.1** (`23.54 ms`, acc 5.88) |
| **16** | **1,122.6** (`27.42 ms`, acc 1.93) | **2,343.9** (`37.18 ms`, acc 5.46) | **2,651.0** (`36.63 ms`, acc 5.88) |
| **32** | **1,436.1** (`36.51 ms`, acc 1.64) | **2,894.7** (`59.31 ms`, acc 5.42) | **3,304.5** (`57.58 ms`, acc 5.88) |

#### Validate & Build

By default (`SOURCE_MODE=vllm-branch`), `scripts/build-deepseek-v41-image.sh`
exports this repository's `vllm` branch, compiles all 6 self-contained native
CUDA/C++ extensions, bundles the patched official LMCache payload from
`localhost/lmcache-server:latest` (building it automatically if absent, or when
`REBUILD_LMCACHE_IMAGE=1`), and runs the offline test gates:

```bash
# 1. Offline validation gate (11 test suites + py_compile, no container build)
VALIDATE_ONLY=1 bash scripts/build-deepseek-v41-image.sh

# 2. Build the complete production TEP4 image
CONTAINER_ENGINE=podman \
OUTPUT_IMAGE=localhost/vllm-main-sm80:deepseek-v4.1-flash-tep4 \
bash scripts/build-deepseek-v41-image.sh

# Optional: rebuild by replaying models/deepseek-v4.1-flash/patches/ onto official upstream
# SOURCE_MODE=patch-upstream bash scripts/build-deepseek-v41-image.sh
```

#### Deploy (Default: 4× 80G SM80 TEP4 — PCIe P2P or NVLink)

`VLLM_DSV41_OPT_PROFILE=tep4_fused` automatically detects NVLink vs PCIe Gen4 P2P
at worker startup (`ampere_peer_topology()`), requiring zero code or `.env` edits
when moving between PCIe and NVLink SM80 hosts:

```bash
cd models/deepseek-v4.1-flash
cp .env.example .env
chmod 600 .env
# Set a private VLLM_API_KEY, MODEL_ROOT, cache/LMCache paths, and DSV41_GPU_0..3.

podman compose --podman-run-args=--ipc=host up -d
```

For the optional 6× CMP 170HX pipeline-parallel deployment (`PP=6`, `7,7,7,7,7,5`),
use `.env.pp6.example` and `-f compose.pp6.yml`:

```bash
cd models/deepseek-v4.1-flash
cp .env.pp6.example .env
chmod 600 .env
podman compose --podman-run-args=--ipc=host -f compose.pp6.yml up -d
```

Host paths in `.env*.example` and Compose defaults (`/srv/models`,
`/srv/vllm-cache`, `/srv/lmcache`, ...) are generic examples; set the real paths
for your host in `.env`. Variables marked `${VAR:?...}` have no default.

Never commit `.env`, API keys, model weights, benchmark results, host addresses,
or host disk UUIDs.

## Branch Architecture

This repository uses a two-branch layout directly anchored to official
`vllm-project/vllm`:

- **`vllm` branch**: Complete patched vLLM source tree (official upstream
  snapshot followed by feature-scoped SM80 / DeepSeek-V4.1 commits matching
  `models/deepseek-v4.1-flash/patches/`).
- **`main` branch** (this branch): Deployment manifests, Compose files,
  self-contained native CUDA/C++ extensions, LMCache patches, build/validation
  scripts, offline tests, monitoring, and documentation.

Exact upstream commits and Docker base image digests are pinned in
`manifests/vllm.env` and `manifests/lmcache.env`.

## Repository Layout

```text
.
├── .github/workflows/              # automated validation & GHCR/Docker Hub image builds
├── manifests/                      # pinned upstream vLLM & LMCache commits / Docker base digests
├── monitoring/                     # standalone Prometheus + Grafana stack
├── patches/
│   └── lmcache/                    # official LMCache patch series (0001..0010)
├── scripts/
│   ├── apply-deepseek-v41-patches.sh
│   ├── apply-lmcache-patches.sh
│   ├── benchmark-vllm.mjs          # decode, prefill, and counting benchmark client
│   ├── build-deepseek-v41-image.sh # builds the full DeepSeek V4.1 image (+ LMCache)
│   ├── build-lmcache-server-image.sh
│   ├── ensure-lmcache-server-image.sh
│   ├── test-lmcache-bounded-pd-groups.py
│   ├── test-lmcache-patches.py
│   └── test-vllm-common-patches.py
└── models/
    └── deepseek-v4.1-flash/
        ├── compose.yml             # default: 4x 80G SM80 TEP4 (TP=4 + EP=4 + LMCache, NVLink/PCIe auto)
        ├── .env.example            # default TEP4 environment template
        ├── compose.pp6.yml         # optional: 6x CMP 170HX (PP=6 + LMCache + CPU Engram)
        ├── .env.pp6.example        # optional PP6 environment template
        ├── manifests/              # pinned vllm branch & upstream source metadata
        ├── native/                 # 6 self-contained SM80 CUDA/C++ extensions built from source
        ├── patches/                # auditable patch archive replaying onto official vLLM
        ├── scripts/                # DeepSeek-only diagnostics
        ├── tests/                  # offline unit & encoding test suites
        └── docs/                   # design notes and historical PP6 investigation records
```

## Monitoring

See [`monitoring/README.md`](monitoring/README.md). One standalone Prometheus +
Grafana stack (`podman compose`, host network) scrapes `/metrics` from vLLM
engines, LMCache servers, and optional GPU exporters listed in `.env`
(`VLLM_METRICS_TARGETS=name=host:port,...`), provisioning a throughput overview
and upstream vLLM dashboards with `server` / `role` / `instance` selectors.

## Benchmark Client

`scripts/benchmark-vllm.mjs` requires Node.js 18+ and has no npm dependencies.
It supports three standard modes documented in its header:

1. ordinary short-input decode: ~16 user tokens, 500 output tokens, `ignore_eos`;
2. cold prefill: approximately N input tokens and 1 output token;
3. deterministic counting output for near-full speculative acceptance.

```bash
node scripts/benchmark-vllm.mjs --help
```

The client requires `VLLM_API_KEY`, assigns a unique `cache_salt` per request,
rejects nonzero `cached_tokens`, uses server-reported usage counts, and records
exact streamed `token_ids` when available. Decode output reports both token/s
and speculative step/s.

## Shared LMCache Baseline

`manifests/lmcache.env` pins the official LMCache v0.5.5 CUDA 13.0 release and
payload digests. Patch rationale and validation gates are documented in
[`patches/lmcache/README.md`](patches/lmcache/README.md).

## Upstream Sources

This repository builds directly on official upstream releases with no third-party
vLLM fork in the chain:

| Upstream | What we take from it | Where it is pinned |
| --- | --- | --- |
| [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | Official vLLM engine, maintained on our `vllm` branch as feature commits on top of upstream | `models/deepseek-v4.1-flash/manifests/source.env` |
| [`LMCache/LMCache`](https://github.com/LMCache/LMCache) | Official LMCache v0.5.5 server/payload images plus `patches/lmcache/0001..0010` | `manifests/lmcache.env` |

## License

Apache-2.0; see [`LICENSE`](LICENSE).
