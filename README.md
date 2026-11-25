# vllm-sm80

**The fastest open-source DeepSeek-V4.1-Flash inference stack on NVIDIA SM80 — 200–560 tok/s single-stream and up to 10.5K tok/s prefill (8.3K at 1M) on 4× 80G PCIe GPUs with unmodified official weights, built directly on official `vllm-project/vllm` and `LMCache`.**

## Prebuilt Images (GHCR)

One unified SM80 vLLM image serves all DeepSeek parallelism splits (auto-selecting the optimal profile via `VLLM_DSV41_OPT_PROFILE=auto` and NVLink/PCIe topology detection) while leaving non-DeepSeek models on untouched official vLLM defaults:

- **vLLM SM80**: `ghcr.io/canoziia/vllm-sm80:latest`
- **Shared LMCache Server**: `ghcr.io/canoziia/vllm-sm80/lmcache-server:latest`

## Models

### DeepSeek V4.1 Flash

See [`models/deepseek-v4.1-flash/README.md`](models/deepseek-v4.1-flash/README.md)
for build instructions, Compose deployments (4× 80G TEP4 & 6× CMP 170HX PP6),
optimization profiles, native CUDA/C++ extensions, and full 180 W & 250 W benchmarks.

#### Measured TEP4 Benchmarks (`4× 80G SM80 PCIe Gen4`, 250 W, `compose.yml` + LMCache)

`TP=4 + EP=4`, unmodified official MXFP4/FP8 weights, `nvfp4_ds_mla` KV cache
(`2,014,502` GPU KV tokens = `1.92× 1M` + LMCache `8 GiB` L1 RAM / `500 GiB`
NVMe L2), `VLLM_DSV41_OPT_PROFILE=auto` (`tep4_fused`), `enable_adaptive_verification=true`,
`0` OOM across full 1M prefill and 32-concurrency stress.

**Cold Prefill (`tok/s`, `c=1`, `cached_tokens=0`)**

| Target Length | 64K | 128K | 256K | 512K | 1M |
|---|---:|---:|---:|---:|---:|
| **Prefill (`tok/s`)** | **10,551.7** | **10,491.8** | **10,205.2** | **9,515.6** | **8,314.6** |

**Decode Throughput & Step Latency (`full_batch_tok_s` / `step_ms`, `max_tokens=500`)**

| Concurrency `c` | `prose` | `code` | `counting` |
|---:|---:|---:|---:|
| **1** | **211.9** (`9.94 ms`, acc 2.10) | **488.6** (`11.24 ms`, acc 5.43) | **563.7** (`10.54 ms`, acc 5.88) |
| **2** | **370.9** (`11.74 ms`, acc 2.18) | **777.1** (`14.36 ms`, acc 5.46) | **932.2** (`12.81 ms`, acc 5.88) |
| **4** | **564.8** (`13.46 ms`, acc 1.90) | **1,180.1** (`18.74 ms`, acc 5.45) | **1,485.5** (`16.11 ms`, acc 5.88) |
| **8** | **818.2** (`18.81 ms`, acc 1.93) | **1,770.8** (`24.88 ms`, acc 5.44) | **2,115.5** (`22.66 ms`, acc 5.88) |
| **16** | **1,133.1** (`27.80 ms`, acc 1.96) | **2,223.0** (`38.91 ms`, acc 5.45) | **2,544.0** (`37.71 ms`, acc 5.88) |
| **32** | **1,398.2** (`44.25 ms`, acc 1.94) | **2,851.3** (`61.20 ms`, acc 5.41) | **3,192.3** (`60.13 ms`, acc 5.88) |

## Branch Architecture

This repository uses a two-branch layout directly anchored to official
`vllm-project/vllm`:

- **`vllm` branch**: Complete patched vLLM source tree (official upstream
  snapshot followed by feature-scoped SM80 commits matching
  `models/deepseek-v4.1-flash/patches/`).
- **`main` branch** (this branch): Pinned upstream manifests, Compose
  deployments, self-contained native CUDA/C++ extensions, LMCache patches,
  build/validation scripts, offline tests, monitoring, and documentation.

Exact upstream commits and Docker base image digests are pinned in
[`manifests/vllm.env`](manifests/vllm.env) and
[`manifests/lmcache.env`](manifests/lmcache.env).

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
│   ├── benchmark-vllm.mjs          # decode, prefill, prose, code, and counting benchmark client
│   ├── run-standard-benchmark.sh   # reproducible 23-point README benchmark + Markdown summary
│   ├── build-deepseek-v41-image.sh # builds the unified SM80 vLLM image (+ LMCache)
│   ├── build-lmcache-server-image.sh
│   ├── ensure-lmcache-server-image.sh
│   ├── test-lmcache-bounded-pd-groups.py
│   ├── test-lmcache-patches.py
│   └── test-vllm-common-patches.py
└── models/
    └── deepseek-v4.1-flash/
        ├── README.md               # DeepSeek V4.1 Flash build, deploy, and full benchmarks
        ├── compose.yml             # default: 4x 80G SM80 TEP4 (TP=4 + EP=4 + LMCache)
        ├── .env.example            # default TEP4 environment template
        ├── compose.pp6.yml         # optional: 6x CMP 170HX (PP=6 + LMCache + CPU Engram)
        ├── .env.pp6.example        # optional PP6 environment template
        ├── native/                 # 7 self-contained SM80 CUDA/C++ extensions built from source
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
It supports `decode`, `prefill`, `prose`, `code`, and `counting` modes:

```bash
# Reproducible 23-point benchmark matching the README tables above
# (prefill 64K..1M + prose/code/counting c=1,2,4,8,16,32 at max_tokens=500):
VLLM_API_KEY=... bash scripts/run-standard-benchmark.sh

# Quick smoke subset (prefill 64K/128K + c=1,8,32):
VLLM_API_KEY=... bash scripts/run-standard-benchmark.sh --quick

# Single-mode usage:
node scripts/benchmark-vllm.mjs --help
```

The client requires `VLLM_API_KEY`, assigns a unique `cache_salt` per request,
rejects nonzero `cached_tokens`, uses server-reported usage counts, and records
exact streamed `token_ids` when available. Decode output reports both token/s
and speculative step/s.

## Upstream Sources

This repository builds directly on official upstream releases with no third-party
vLLM fork in the chain:

| Upstream | What we take from it | Where it is pinned |
| --- | --- | --- |
| [`vllm-project/vllm`](https://github.com/vllm-project/vllm) | Official vLLM engine & base image, maintained on our `vllm` branch as feature commits on top of upstream | [`manifests/vllm.env`](manifests/vllm.env) |
| [`LMCache/LMCache`](https://github.com/LMCache/LMCache) | Official LMCache v0.5.5 server/payload images plus [`patches/lmcache/`](patches/lmcache/README.md) | [`manifests/lmcache.env`](manifests/lmcache.env) |

## License

Apache-2.0; see [`LICENSE`](LICENSE).
