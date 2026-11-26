# vllm-sm80

**The fastest open-source DeepSeek-V4.1-Flash inference stack on NVIDIA SM80 — 278–719 tok/s single-stream, up to 2.0–4.3K tok/s at c=32, and up to 12.9K tok/s prefill (9.8K at 1M) on 4× 80G PCIe GPUs with unmodified official weights, built directly on official `vllm-project/vllm` and `LMCache`.**

## Prebuilt Images (GHCR)

One unified SM80 vLLM image serves all DeepSeek parallelism splits (auto-selecting the optimal profile via `VLLM_DSV41_OPT_PROFILE=auto` and NVLink/PCIe topology detection) while leaving non-DeepSeek models on untouched official vLLM defaults:

- **vLLM SM80**: `ghcr.io/canoziia/vllm-sm80:latest`
- **Shared LMCache Server**: `ghcr.io/canoziia/vllm-sm80/lmcache-server:latest`

## Models

### DeepSeek V4.1 Flash

See [`models/deepseek-v4.1-flash/README.md`](models/deepseek-v4.1-flash/README.md)
for build instructions, Compose deployments (default 4× 80G Pure TP4, optional TEP4 & 6× CMP 170HX PP6),
optimization profiles, native CUDA/C++ extensions, and full 180 W & 250 W benchmarks.

#### Measured Pure TP4 Benchmarks (`4× 80G SM80 PCIe Gen4`, 250 W, `compose.yml` + LMCache)

Pure `TP=4` (`EP=1`, `NI=576`, `E=384`, 100% load-balanced across all 384 experts with
zero All-to-All), unmodified official MXFP4/FP8 weights, `nvfp4_ds_mla` KV cache
(`2,014,502` GPU KV tokens = `1.92× 1M` + LMCache `8 GiB` L1 RAM / `500 GiB`
NVMe L2), `VLLM_DSV41_OPT_PROFILE=auto` (`tp4_fused`), `enable_adaptive_verification=true`,
`0` OOM across full 1M prefill and 32-concurrency stress.

**Cold Prefill (`tok/s`, `c=1`, `cached_tokens=0`)**

| Target Length | 64K | 128K | 256K | 512K | 1M |
|---|---:|---:|---:|---:|---:|
| **Prefill (`tok/s`)** | **12,911.9** | **12,803.8** | **12,417.1** | **11,555.2** | **9,842.6** |

**Decode Throughput & Step Latency (`full_batch_tok_s` / `step_ms`, `max_tokens=500`)**

| Concurrency `c` | `prose` | `code` | `counting` |
|---:|---:|---:|---:|
| **1** | **278.3** (`7.73 ms`, acc 2.15) | **621.9** (`8.88 ms`, acc 5.46) | **718.5** (`8.27 ms`, acc 5.88) |
| **2** | **432.4** (`8.76 ms`, acc 1.89) | **936.8** (`11.89 ms`, acc 5.47) | **1,152.5** (`10.36 ms`, acc 5.88) |
| **4** | **719.9** (`10.80 ms`, acc 1.95) | **1,403.9** (`15.66 ms`, acc 5.48) | **1,818.8** (`13.16 ms`, acc 5.88) |
| **8** | **1,098.7** (`13.77 ms`, acc 1.88) | **2,141.1** (`20.62 ms`, acc 5.46) | **2,767.0** (`17.32 ms`, acc 5.88) |
| **16** | **1,460.6** (`21.26 ms`, acc 1.94) | **2,858.0** (`30.44 ms`, acc 5.40) | **3,572.1** (`26.86 ms`, acc 5.88) |
| **32** | **1,979.7** (`26.59 ms`, acc 1.65) | **3,624.4** (`47.96 ms`, acc 5.44) | **4,310.0** (`44.53 ms`, acc 5.88) |

## Branch Architecture

This repository uses a two-branch layout directly anchored to official
`vllm-project/vllm`:

- **`vllm` branch**: Single source of truth for the patched vLLM source tree
  (official upstream snapshot followed by clean feature-scoped SM80 commits).
- **`main` branch** (this branch): Pinned upstream manifests, Compose
  deployments (Pure TP4, PD disaggregation, PP6), self-contained native
  CUDA/C++ extensions, LMCache & Router patches, build/validation scripts,
  offline tests, monitoring, and documentation.

Exact upstream commits and Docker base image digests are pinned in
[`manifests/vllm.env`](manifests/vllm.env),
[`manifests/lmcache.env`](manifests/lmcache.env), and
[`manifests/router.env`](manifests/router.env).

## Repository Layout

```text
.
├── .github/workflows/              # automated validation & GHCR/Docker Hub image builds
├── manifests/                      # pinned upstream vLLM, LMCache & Router commits / base digests
├── monitoring/                     # standalone Prometheus + Grafana stack
├── patches/
│   ├── lmcache/                    # official LMCache patch series (0001..0011)
│   └── router/                     # vLLM production-stack PD router patch series (0001..0005)
├── scripts/
│   ├── apply-lmcache-patches.sh
│   ├── benchmark-vllm.mjs          # decode, prefill, prose, code, and counting benchmark client
│   ├── run-standard-benchmark.sh   # reproducible 23-point README benchmark + Markdown summary
│   ├── build-deepseek-v41-image.sh # builds the unified SM80 vLLM image from the `vllm` branch (+ LMCache)
│   ├── build-lmcache-server-image.sh
│   ├── build-router-image.sh       # builds the patched PD router image
│   ├── ensure-lmcache-server-image.sh
│   ├── test-lmcache-bounded-pd-groups.py
│   ├── test-lmcache-patches.py
│   ├── test-router-patches.py
│   └── test-vllm-common-patches.py
└── models/
    └── deepseek-v4.1-flash/
        ├── README.md               # DeepSeek V4.1 Flash build, deploy, and full benchmarks
        ├── compose.yml             # default: 4x 80G SM80 Pure TP4 (TP=4, EP=1 + LMCache)
        ├── .env.example            # default Pure TP4 environment template
        ├── compose.pd.yml          # optional: 6x 80G SM80 PD disaggregation (TP4 Decode + TP2 YOCO Prefill)
        ├── .env.pd.example         # optional PD disaggregation environment template
        ├── compose.pp6.yml         # optional: 6x CMP 170HX (PP=6 + LMCache + CPU Engram)
        ├── .env.pp6.example        # optional PP6 environment template
        ├── native/                 # 7 self-contained SM80 CUDA/C++ extensions built from source
        ├── scripts/                # DeepSeek-only utilities (e.g. prepare-engram-shm.py)
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
