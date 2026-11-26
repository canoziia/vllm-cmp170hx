# DeepSeek V4.1 Flash on SM80 GPUs (Pure TP4 80G, optional TEP4 & PP6 CMP 170HX)

Reproducible patch series, self-contained native CUDA/C++ extensions, and Podman Compose
deployments for `deepseek-ai/DeepSeek-V4.1-Flash-1017` built directly on official
`vllm-project/vllm` for SM80 GPUs:

- **Default — 4x 80G SM80 Pure TP4 (`compose.yml` / `.env.example`)**: `TP=4, EP=1, DCP=1, PP=1`
  (`NI=576`, all `384` routed experts sharded 1/4 per rank for 100% load balance across decode and
  prefill, `nvfp4_ds_mla` KV cache, `2,014,502` GPU KV tokens = `1.92x 1M` at `2.20 GB/rank` +
  LMCache 8 GiB L1 / 500 GiB L2, topology-aware NVLink / PCIe P2P collectives).
  *(Legacy `TEP4` remains supported by adding `--enable-expert-parallel`, auto-selecting `tep4_fused`.)*
- **Optional — 6x CMP 170HX PP6 (`compose.pp6.yml` / `.env.pp6.example`)**: `TP=1 x PP=6`
  (`7,7,7,7,7,5`, LMCache KV tier, CPU Engram offload).

## Repository Branch Layout & Source

This repository uses a two-branch layout (exact upstream commits and Docker base
image digests are pinned in [`../../manifests/vllm.env`](../../manifests/vllm.env)
and [`../../manifests/lmcache.env`](../../manifests/lmcache.env)):

- **`vllm` branch**: Complete patched vLLM source tree (official upstream snapshot followed
  by feature-scoped commits matching `patches/` 1:1).
- **`main` branch**: Deployment compose files (default TEP4 `compose.yml`, optional PP6
  `compose.pp6.yml`), self-contained native CUDA/C++ extensions (`native/`), build scripts
  (`scripts/build-deepseek-v41-image.sh`, consuming our `vllm` branch by default), offline
  test suites, and auditable patch archive (`patches/`).
- **Shared LMCache layer**: official LMCache v0.5.5 + `../../patches/lmcache/0001..0010`
  (client `.so` modules automatically rebuilt against the official vLLM base PyTorch ABI).
- **Prebuilt GHCR images**:
  - `ghcr.io/canoziia/vllm-sm80:latest` (unified SM80 vLLM image — works across TEP4, PP6, other DeepSeek splits, and non-DeepSeek models)
  - `ghcr.io/canoziia/vllm-sm80/lmcache-server:latest` (shared LMCache server)

### Self-Contained Native Extensions (`models/deepseek-v4.1-flash/native/`)

All seven native extensions are built from source during `scripts/build-deepseek-v41-image.sh`:

| directory | artifact | purpose |
|---|---|---|
| `native/dsv4_moe` | `vllm/_dsv4_moe_C.abi3.so` | SM80 weight-streaming MXFP4 MoE decode (`w13`/`w2`), fused SwiGLU, single-CTA deterministic align |
| `native/dsv41_dense_mxfp8` | `vllm/_dsv41_dense_mxfp8.so` | 1x-HBM dense MXFP8 decode GEMMs (`M <= 192`), split `wq_b + indexer.wq_b` epilogue, `splitk_m8` |
| `native/dsv41_indexer` | `vllm/_dsv41_indexer_C.abi3.so` | Fast CUDA prefill sparse-indexer logits + top-512 / top-2048 writer |
| `native/dsv41_swa` | `vllm/_dsv41_swa_C.abi3.so` | In-place ragged sliding-window metadata builder |
| `native/dsv41_r2` | `vllm/_dsv41_r2_C.abi3.so` | Single-kernel `fused_gate_topk_align` + `dsv4_topk_v2` (with `is_padding` & `bias_vl`), `w2_splitk_ep` (256-col + fused shared `__hadd2`), C++ Tensor-Core `woa_mxfp8_splitk`, `tp_greedy_*` local verification, `dequant_marlin_mxfp8` |
| `native/dsv41_pcie_dma` | `vllm/_dsv41_pcie_dma_C.so` | O(1) `K = 3` triple-buffered Copy-Engine Ring DMA (`ce_reduce_scatter`, `ce_all_gather`, `ce_all_reduce_inplace`) with 64-bit bidirectional PTX ACK |
| `native/dsv41_engram_cpu` | `.../common/_dsv41_engram_cpu.so` | OpenMP + AVX2/FMA packed 264 B FP8+UE8M0 host Engram gather |

## Pure TP4 Deployment (`4x 80G SM80`, Default) & Automatic NVLink / PCIe Detection

```text
TP4 (EP=1, NI=576, E=384), DCP1, PP1
DSpark K=5, TP-local greedy verification + adaptive verification (COST_MS_TP4)
max_model_len=1,048,576
max_num_batched_tokens=3072
max_num_seqs=32
KV=nvfp4_ds_mla (2,014,502 GPU KV tokens at 2.20 GB/rank = 1.92x 1M + LMCache 8 GiB L1 / 500 GiB L2)
CPU Engram offload (2x 94.4 GiB packed 264B rows), CPU Vision offload
VLLM_DSV41_OPT_PROFILE=auto   # auto-selects tp4_fused for Pure TP=4 (or tep4_fused with --enable-expert-parallel)
```

### Zero-Edit Migration Between Parallelism Splits, PCIe P2P, and NVLink

`VLLM_DSV41_OPT_PROFILE=auto` (baked into `ghcr.io/canoziia/vllm-sm80:latest` by default)
inspects the model `config.json` and launch parallelism at startup:

- **Non-DeepSeek models**: completely untouched (100% standard upstream vLLM).
- **DeepSeek-V4.1 Pure `TP=4` (`PP=1`, default)**: automatically selects `tp4_fused`
  (`NI=576` weight-streaming MXFP4 W13/W2 with GPU-side dynamic `nblk` Split-K `dynks`,
  `DeferringMarlinExperts` + 3-regime push-Lamport finalize across all `M=1..192`,
  5×-bandwidth-reduced Engram Local Partial-Dot for `M >= 160`, `fused_gate_topk_align` for
  `local_e <= 384`, and online self-calibrating `OnlineCostModel` anchored to `COST_MS_TP4`),
  and queries `ampere_peer_topology()` at worker startup via `cudaDevP2PAttrPerformanceRank` / NVML:
  - **PCIe Gen4 P2P (`"pcie"`)**: runs 3-regime push-Lamport AllReduce (`OneShot` `M=1..2`,
    `TwoShot` `M=3..15`, `Ring` `M=16..192` with Hop-0 PCIe overlap and deferred sentinel reset)
    across all decode batch sizes (`c=1..32`), single-kernel Lamport symmetric-memory exchange
    (`markov_symm.py`) across the 5 DSpark Markov steps, and Copy-Engine Ring DMA Row-Sharded
    prefill (`24.5 GB/s` P2P ring).
  - **NVLink (`"nvlink"`)**: automatically unlocks full `MAX_ROWS = 192` two-shot CuTe
    Lamport AllReduce (`ring_min_rows = 0`) and column AllGather across all decode batch sizes.
- **DeepSeek-V4.1 Legacy `TP=4 + EP=4` (`--enable-expert-parallel`)**: automatically selects `tep4_fused`.
- **DeepSeek-V4.1 `PP > 1` (e.g. 6x CMP 170HX PP6)**: automatically selects `default`
  (or `verification` when `enable_adaptive_verification: true`).
- **Other DeepSeek-V4.1 `TP > 1` splits (e.g. TP2 / TP8)**: automatically selects `tp_auto`
  (or `tp_auto_verification`).

Moving the same container image between Pure TP4 / TEP4 / PP6 or PCIe / NVLink SM80 hosts requires
**zero `.py` changes**.

- **Host PCIe Relaxed Ordering (`NVreg_EnablePCIERelaxedOrderingMode`)**: keep the default `0` on
  single-switch `PIX` hosts (where P2P already saturates PCIe Gen4 x16; enabling `1` trades
  `+1.8–2.7%` cold prefill throughput for `~0.07–0.89 ms/step` decode sync latency regression
  across 86 per-step barriers). On `PHB` / `NODE` / `SYS` topologies traversing the CPU Root
  Complex, `1` may yield larger P2P/DMA gains and is worth benchmarking locally.

### Quickstart (Prebuilt GHCR Images)

```bash
# Default: 4x 80G SM80 Pure TP4 (PCIe P2P or NVLink)
cd models/deepseek-v4.1-flash
cp .env.example .env
chmod 600 .env
# Edit VLLM_API_KEY, MODEL_ROOT, cache/LMCache paths, and DSV41_GPU_0..3 in .env

podman compose --podman-run-args=--ipc=host up -d
podman logs -f deepseek-v4.1-flash-tp4

# Optional: 6x CMP 170HX PP6 (7,7,7,7,7,5)
# cp .env.pp6.example .env
# chmod 600 .env
# podman compose -f compose.pp6.yml --podman-run-args=--ipc=host up -d
```

### Validate & Build Locally (Optional)

```bash
# 1. Offline validation gate (all test suites + py_compile, no container build):
VALIDATE_ONLY=1 bash scripts/build-deepseek-v41-image.sh

# 2. Build the unified SM80 vLLM image locally from our `vllm` branch + 6 native extensions:
CONTAINER_ENGINE=podman \
OUTPUT_IMAGE=localhost/vllm-sm80:latest \
bash scripts/build-deepseek-v41-image.sh
# (Optional: set SOURCE_MODE=patch-upstream to rebuild from official upstream + patches/)
```

Host paths in `.env*.example` and Compose defaults (`/srv/models`, `/root/app/...`) are
examples; set the real paths for your host in `.env`. Never commit `.env`, API keys,
model weights, benchmark results, host addresses, or host disk UUIDs.

## Optimisation Profiles (`vllm/dsv41_opt_profile.py`)

`VLLM_DSV41_OPT_PROFILE` applies profile defaults via `os.environ.setdefault` in
`vllm.env_override` before any other vLLM import, so individual environment variables
always take precedence:

| profile | target topology | description |
|---|---|---|
| `auto` (image default) | Any SM80 / any model | Inspects model `config.json` + `-tp`/`-pp`/`--enable-expert-parallel`: leaves non-DeepSeek models untouched; selects `tp4_fused` for Pure TP4 (default), `tep4_fused` for legacy TEP4, `default`/`verification` for PP>1, `tp_auto` for other TP>1 |
| `tp4_fused` (4×80G default) | 4x 80G Pure TP4 (PCIe or NVLink) | Pure TP4 deep-fusion profile (`EP=1, NI=576`): 3-regime push-Lamport finalize (`OneShot M=1..2`, `TwoShot M=3..15`, `Ring M=16..192`), GPU-side `dynks` W13 + occ=5 `M=4..6`, Engram Local Partial-Dot (`M>=160`), `fused_gate_topk_align` (`local_e<=384`), `moe_prefill` split-list (`NI=576`), online self-calibrating `OnlineCostModel` + `COST_MS_TP4` |
| `default` | PP6 / general SM80 | Base 1x-HBM MXFP8/MXFP4 decode & long-context prefill kernels |
| `verification` | PP6 + DSpark | Adds sub-width & mixed-`spec_k` FULL CUDA graphs, metadata replay, live-load PP cohorts |
| `tp_auto` / `tp_auto_verification` | General DeepSeek TP>1 | TP-safe SM80 optimisations (Engram NUMA interleave, FP8 `wo_a`, vision offload, cuBLAS MXFP8 prefill, topology-aware collectives) |
| `tep4_verification` | 4x 80G legacy TEP4 | Baseline SM80 adaptive-verification profile (`nvfp4_ds_mla`, CPU vision offload, `COST_MS_BASE`) |
| `tep4_fused` | 4x 80G legacy TEP4 (PCIe or NVLink) | Legacy TEP4 (`TP=4 + EP=4`) deep-fusion profile (`w2_splitk_ep` + fused shared add, `COST_MS_FUSED`) |
| `off` / `0` | Any | Disables all automatic profile overrides |

## Preserved API & Serving Contracts

- **Multimodal Vision CPU Offload (`VLLM_DSV41_OFFLOAD_VISION_EMBED=1`)**: keeps the vision
  tower on CPU RAM (`0 B` permanent GPU HBM) while preserving full image understanding and
  in-graph pure-text token embedding (`VLLM_DSV41_IN_GRAPH_TEXT_EMBED=1`).
- **Checkpoint `1017` Encoding (`deepseek_v41_encoding.py`)**: `low`/`high`/`max` = `50`/`75`/`100`
  (accepting legacy `xhigh` -> `max`), tool namespaces (`ns::name`), and preserved reasoning across
  tool-result turns (`tests/test_dsv41_encoding_1017.py`).
- **Top-Level Thinking Switch (`deepseek_thinking.py`)**: `"thinking": {"type": "enabled" | "disabled"}`
  mapped cleanly before tokenizer and reasoning parser (`tests/test_deepseek_thinking.py`).
- **Adaptive & Manual Speculative Widths**: `enable_adaptive_verification: true` uses the
  request-local measured policy (`dsv41_measured_policy.py`); per-request `"vllm_xargs": {"spec_k": 0..5}`
  overrides verification depth cleanly.
- **SSE Keep-Alive & Detailed Spec Metrics**: `--sse-keep-alive-interval` and
  `--per-request-spec-decode-metrics=detailed` supported out of the box.

## Measured Pure TP4 Benchmarks (`4x 80G SM80 PCIe Gen4`, `compose.yml` + LMCache)

Measured on the live `compose.yml` stack (`TP=4, EP=1, NI=576`, `nvfp4_ds_mla`,
`VLLM_KV_CACHE_MEMORY=2200000000` -> `2,014,502` GPU KV tokens = `1.92x 1M`,
LMCache `8 GiB` L1 RAM + `500 GiB` NVMe L2, `max_num_batched_tokens=3072`,
`VLLM_DSV41_OPT_PROFILE=auto` (`tp4_fused`), `enable_adaptive_verification=true`, `0` OOM across
full 1M prefill + 32-concurrency stress).

### Cold Prefill (`tok/s`, `c=1`, `cached_tokens=0`, async streaming to LMCache L1 + L2)

| Target Length | 180W Power Cap | 250W Power Limit | 250W vs 180W |
|---:|---:|---:|---:|
| **64K** | 10,694.4 | **12,191.6** | +14.0% |
| **128K** | 11,189.9 | **12,045.2** | +7.6% |
| **256K** | 10,692.2 | **11,587.6** | +8.4% |
| **512K** | 9,600.2 | **10,714.3** | +11.6% |
| **1M** | 8,454.6 | **9,367.8** | +10.8% |

### Decode Throughput & Step Latency (`full_batch_tok_s` / `step_ms`, `max_tokens=500`)

#### 180W Power Cap (`RO=0`, `0` OOM across full 1M prefill + 32-concurrency stress)

| Concurrency `c` | `prose` | `code` | `counting` |
|---:|---:|---:|---:|
| **1** | **249.0** (`7.77 ms`, acc 1.93; fixed `k=2/3`: `253/263` @ `7.63/8.16 ms`) | **578.8** (`9.54 ms`, acc 5.46) | **669.2** (`8.88 ms`, acc 5.88) |
| **2** | **401.5** (`9.45 ms`, acc 1.90) | **853.5** (`13.05 ms`, acc 5.47) | **1,069.8** (`11.16 ms`, acc 5.88) |
| **4** | **665.5** (`10.45 ms`, acc 1.75) | **1,362.8** (`16.60 ms`, acc 5.56) | **1,763.7** (`13.57 ms`, acc 5.88) |
| **8** | **1,087.1** (`13.77 ms`, acc 1.86) | **1,881.6** (`23.23 ms`, acc 5.33) | **2,498.0** (`19.19 ms`, acc 5.88) |
| **16** | **1,342.2** (`19.55 ms`, acc 1.64; fixed `k=1`: `1,390.1` @ `18.42 ms`) | **2,617.8** (`33.59 ms`, acc 5.48) | **3,208.3** (`29.90 ms`, acc 5.88) |
| **32** | **1,804.7** (`29.10 ms`, acc 1.64; fixed `k=1`: `1,899.3` @ `26.96 ms`) | **3,388.1** (`52.52 ms`, acc 5.39) | **3,917.5** (`49.00 ms`, acc 5.88) |

#### 250W Power Limit (`RO=0`, `0` OOM)

| Concurrency `c` | `prose` | `code` | `counting` |
|---:|---:|---:|---:|
| **1** | **239.9 (`8.96 ms`, acc 2.15)** | **548.0 (`9.89 ms`, acc 5.36)** | **627.3 (`9.47 ms`, acc 5.88)** |
| **2** | **396.7 (`9.98 ms`, acc 1.99)** | **853.3 (`13.05 ms`, acc 5.47)** | **1,038.2 (`11.50 ms`, acc 5.88)** |
| **4** | **642.0 (`12.17 ms`, acc 1.96)** | **1,364.8 (`16.33 ms`, acc 5.46)** | **1,733.8 (`13.81 ms`, acc 5.88)** |
| **8** | **1,024.2 (`15.26 ms`, acc 1.97)** | **2,012.1 (`22.08 ms`, acc 5.43)** | **2,559.8 (`18.73 ms`, acc 5.88)** |
| **16** | **1,342.8 (`23.06 ms`, acc 1.93)** | **2,851.1 (`31.28 ms`, acc 5.41)** | **3,297.0 (`29.10 ms`, acc 5.88)** |
| **32** | **1,741.8 (`35.50 ms`, acc 1.93)** | **3,333.6 (`51.49 ms`, acc 5.40)** | **3,938.5 (`48.73 ms`, acc 5.88)** |
