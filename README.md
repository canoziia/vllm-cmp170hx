# vllm-sm80

**The fastest open-source DeepSeek-V4.1-Flash inference stack on NVIDIA SM80 — 238–627 tok/s single-stream, up to 1.7–3.9K tok/s at c=32, and up to 12.2K tok/s prefill (9.4K at 1M) on 4× 80G PCIe GPUs with unmodified official weights, built directly on official `vllm-project/vllm` and `LMCache`.**

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
| **Prefill (`tok/s`)** | **12,191.6** | **12,045.2** | **11,587.6** | **10,714.3** | **9,367.8** |

**Decode Throughput & Step Latency (`full_batch_tok_s` / `step_ms`, `max_tokens=500`)**

| Concurrency `c` | `prose` | `code` | `counting` |
|---:|---:|---:|---:|
| **1** | **239.9** (`8.96 ms`, acc 2.15) | **548.0** (`9.89 ms`, acc 5.36) | **627.3** (`9.47 ms`, acc 5.88) |
| **2** | **396.7** (`9.98 ms`, acc 1.99) | **853.3** (`13.05 ms`, acc 5.47) | **1,038.2** (`11.50 ms`, acc 5.88) |
| **4** | **642.0** (`12.17 ms`, acc 1.96) | **1,364.8** (`16.33 ms`, acc 5.46) | **1,733.8** (`13.81 ms`, acc 5.88) |
| **8** | **1,024.2** (`15.26 ms`, acc 1.97) | **2,012.1** (`22.08 ms`, acc 5.43) | **2,559.8** (`18.73 ms`, acc 5.88) |
| **16** | **1,342.8** (`23.06 ms`, acc 1.93) | **2,851.1** (`31.28 ms`, acc 5.41) | **3,297.0** (`29.10 ms`, acc 5.88) |
| **32** | **1,741.8** (`35.50 ms`, acc 1.93) | **3,333.6** (`51.49 ms`, acc 5.40) | **3,938.5** (`48.73 ms`, acc 5.88) |

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
        ├── compose.yml             # default: 4x 80G SM80 Pure TP4 (TP=4, EP=1 + LMCache)
        ├── .env.example            # default Pure TP4 environment template
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
