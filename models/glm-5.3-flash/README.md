# GLM-5.3-Flash on CMP 170HX

Two prefill/decode pairs share container names and ports; run one of them:

| compose | target | speculative | image |
|---|---|---|---|
| `compose.pd.w4a16-dflash2.yml` (**production**) | `canada-quant/GLM-5.3-Flash-W4A16-MTP` | DFlash2 | `glm-5.3-flash-w4a16-dflash2` |
| `compose.pd.yml` | `nvidia/GLM-5.3-Flash-NVFP4` | MTP-3 | `deepseek-v4.1-flash` |

`compose.w4a16-dflash2.yml` runs the W4A16 + DFlash2 engine alone (PP4, one
role), see [below](#w4a16--dflash2pp4单引擎).

## W4A16 + DFlash2 PD（生产）

prefill PP4 用 GPU1–4（:8301），decode PP4 用 GPU5–8（:8302），一个 LMCache
server（:5559，HTTP :18559），PD router 入口 **:9103**；模型名
`canada-quant/GLM-5.3-Flash-W4A16-MTP`，max-model-len 1048576。

```bash
scripts/build-glm53-dflash2-image.sh                       # engine image
REBUILD_LMCACHE_IMAGE=1 scripts/build-lmcache-server-image.sh   # when patches/lmcache changes
cd models/glm-5.3-flash
cp .env.example .env && chmod 600 .env                     # fill in VLLM_API_KEY, paths, GPUs
podman compose -f compose.pd.w4a16-dflash2.yml --podman-run-args=--ipc=host up -d
```

### 几何

| 参数 | 值 | 原因 |
|---|---|---|
| target block = LMCache chunk = retention | 5120 | KDA 状态页在 DFlash 最大深度 7 时需要 ≥4608 token；还要能被 drafter block 整除 |
| drafter block | 1024 | drafter 滑窗 2048 与 chunk 5120 都必须是 drafter block 的整数倍（LMCache 校验）；dflash2/0003 把 1024 token 的 drafter block 放在一个 5 MiB 的 MLA 页里（4 MiB 数据） |
| `--max-num-batched-tokens` | 5184 | 带 LMCache 的混合模型每步必须推进一个完整 block，再加上 draft 输入槽 |
| KV | 12 GiB/卡 | 按 1M 口径 2,691,005 tokens（2.57×）；峰值显存余量约 1.4 GiB |
| KDA 页填充 | 11.89% | |

PP 下最后一段比其他段多一个 drafter 滑窗分组，LMCache 默认会认为各段对象
分组描述不一致而拒绝注册；`LMCACHE_MERGE_BOUNDED_PD=1`（`patches/lmcache/0009`，
默认关）让各段按跨 chunk 窗口分组，描述一致。server 与两个引擎都要设。
不兼容 CacheBlend（`full_sw_kv`），也不要与别的布局共用 L2 目录。

L2：`fs_native`，容量 400 GiB，占用到 60% 开始按 LRU 回收，每次回收约 20% 的
key；`adopt_existing=true` 时重启前的文件也计入容量并能被回收
（`patches/lmcache/0006`）。这是水位策略，不是硬配额。

### 尾段状态传递（0034）

LMCache 只存完整的 5120-token chunk（KDA 状态只在 block 边界有检查点），原先 decode
要重算提示最后不满一个 chunk 的部分（最多 5119 token；107k 时 4600 token，约 1.8 s，
在 decode GPU 上）。`VLLM_GLM53_PD_TAIL=1`（profile 打开；补丁默认关；只对
LMCacheMPConnector 生效）时，prefill 在提示算完后把每个 PP 段的末尾状态（KDA 循环与
conv 状态、MLA/indexer/kpool 尾段、DFlash2 drafter 窗口、最后的 hidden/aux）作为独立
namespace 的 GPU chunk，用 LMCache 原有的 STORE/RETRIEVE＋CUDA IPC 存入（L1/L2/LRU
照常，LMCache 不改）；四段都存好后才发布一条与完整提示、cache_salt、模型配置绑定的
generation 记录（走 LMCache 已有的 engine-driven API，所以 LMCache server 要
`--supported-transfer-mode=auto`）。decode 先装入普通前缀 chunk，再把尾段装进请求私有页。

装入的 KDA 状态已经读过最后一个提示 token，所以 decode 第一步只采样：不跑 target
前向，用恢复的 hidden 和 decode 请求自己的采样参数跑 LM head、sampler 和 DFlash 提议。
target 重算 0 token；装入状态、首步 logits 与候选和本地完整 prefill 逐位一致。107k
一次实测两跳合计 23.55 s → 20.17 s。

只有一端打开、尾段或前缀缺失、多模态、LoRA、prompt logprobs、被抢占的请求、提示长度
正好是 5120 的整数倍时走原路径。适用范围：PP4/TP1/DP1、mamba align、folded aux、
DFlash2。测试：`tests/dflash2/test_pd_tail_gpu.py`（真实状态与首步逐位比较＋一轮端到端）。

### 优化开关

`VLLM_GLM53_OPT_PROFILE=default`（dflash2/0033）在每个进程启动时把下表开关设为
部署值，**已设置的变量不覆盖**，所以单独关某一项只需在 compose 里写
`VLLM_XXX: "0"`。不设 profile 时所有补丁回到各自默认（关）。

| 补丁 | 开关 | 值 |
|---|---|---|
| 0001 | `VLLM_GLM5_AUX_HIDDEN_TENSOR` | `stream_mean` |
| 0004 | `VLLM_GLM5_DFLASH_ADAPTIVE_K` / `_DEPTHS` / `_ACCEPT` | `1` / `7,5` / `0` |
| 0005 | `VLLM_GLM5_THIN_GEMM` | 1 |
| 0006 | `VLLM_GLM5_MARLIN_DECODE_CUDA` / `VLLM_GLM5_MARLIN_DECODE_VARIANT` | 1 / `orig` |
| 0007 | `VLLM_DFLASH2_FUSED_GROUPED_CONV` | 1 |
| 0008 | `VLLM_GLM5_ROUTER_ALIGN_DECODE` | 1 |
| 0009 | `VLLM_GLM5_SPARSE_MLA_MM_EXPERIMENTAL` | 1 |
| 0010 / 0023 | `VLLM_GLM5_ROUTE_V2` / `VLLM_GLM5_ROUTE_V2_GEMV` / `VLLM_GLM5_ROUTE_V2_FIRST` | 1 / `tc` / 0 |
| 0011 | `VLLM_GLM5_INDEXER_GATHER_CLAMP` | 1 |
| 0012–0014 | `VLLM_GLM5_PP_SPARSE_MLA_PREFILL` / `VLLM_GLM5_PP_MARLIN_PREFILL` / `VLLM_GLM5_PP_KDA_PREFILL` | 1 |
| 0016 | `VLLM_GLM5_PP_FOLD_DRAFT_FC` | 1 |
| 0017 | `VLLM_GLM5_DECODE_MHC_V2` | 1 |
| 0018 / 0019 | `VLLM_GLM5_PROLOGUE_FUSE` / `VLLM_GLM5_PROLOGUE_PACKED_H2D` | 1 |
| 0020 | `VLLM_GLM5_SPARSE_MLA_DEVICE_REQ_IDS` | 1 |
| 0021 | `VLLM_PP_DRAFT_TAIL_STAGE` | 2 |
| 0022 | `VLLM_GLM5_MLA_PROFILE_WS_CLAMP` | 1 |
| 0024 | `VLLM_GLM5_DECODE_IDX_GLUE` | 1 |
| 0025 / 0028 | `VLLM_GLM5_TARGET_PRENORM_FP32` / `VLLM_MHC_POST_FUSE_SQRSUM` | 1 / 0 |
| 0026 | `VLLM_GLM5_MOE_SUM_ADD` | 1 |
| 0027 | `VLLM_GLM5_SHARED_EXPERT_REORDER` | 1 |
| 0029 | `VLLM_GLM5_IDX_DUAL_GEMM` | 1 |
| 0030 / 0031 | `VLLM_PP_METADATA_CACHE` / `VLLM_PP_PACK_TENSORS` | 1 |
| 0034 | `VLLM_GLM53_PD_TAIL` | 1 |
| 0035 | `VLLM_GLM53_PREFIX_RETENTION` | 1 |

`VLLM_GLM53_PREFIX_RETENTION` 补丁默认关闭，profile 打开：在现有
hash1024/target5120/draft1024 几何下，为实际 KDA partial checkpoint
保留 drafter 连续尾窗，使首次 extension 不必先全量重算来补 shared junction。
不修改 EAGLE 标记或状态有效性规则，不新增 KDA checkpoint，不修改 LMCache；
4487-token 首轮后的正常可复用位置仍为3072，而不是4096。

0030/0031 改变 PP 段之间的传输格式，所有段必须一致（profile 在每个进程里
相同）。不在 profile 里、默认值即部署值的：`VLLM_GLM5_PP_KDA_PREFILL_MAX_TOKENS`
（16384，0032）、`VLLM_GLM5_PRENORM_BF16X3_MIN_TOKENS`（384，0028）。

### 思考开关

模型自带的模板在生成提示末尾总是放一个未闭合的 `<think>`，`enable_thinking=false`
也一样，模型照常思考，parser 却按请求标志把思考当成正文。compose 改用
`chat_template.jinja`（镜像内 `/opt/glm-5.3-flash/chat_template.jinja`，
`--chat-template`）：它与模型模板只差一处，`enable_thinking=false` 时生成提示以空的
`<think></think>` 结尾（模板对历史中没有思考的 assistant 轮次本来就用这个形式），
模型直接作答，正文在 `content`。vLLM 会把 `reasoning_effort=none` 转成
`enable_thinking=false`。其他请求（默认、`low`/`high`、`enable_thinking=true`）渲染结果
与原模板逐字相同，思考在 `message.reasoning`。

### 实测（node2，出厂设置）

| 模式 | c1 tok/s | c8 tok/s |
|---|---:|---:|
| counting | 222.7 | 720.5（KV 12 GiB 单轮 744.1） |
| prose | 65.9 | 328.6 |
| code | 155.2 | 594.2（12 GiB 单轮 595.8） |

- 107,000 token prefill：prefill 直连约 5,500 tok/s，经 PD 约 4,900 tok/s；差别
  主要是 decode 侧重算最后一个不满 chunk 的尾段（107000 mod 5120 = 4600 token）。
- 524,288 token 107.5 s，1,040,000 token 248 s，decode 从 LMCache 命中 1,039,360。
- chunk 边界处 decode 侧从 LMCache 恢复的 drafter 窗口 KV、target 末 chunk KV、
  KDA 状态、aux hidden 与本地 prefill 逐位一致；16/504 token 后缀的同一序列
  直连与 PD 接受数相同（5.735 / 5.543 draft/step）。

## NVFP4 + MTP-3 PD (`compose.pd.yml`, previous production)

Model: [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
at revision `da920bb0b9f4a06727223a349e55468e38352348`.

- `Glm5NextForConditionalGeneration` (`model_type: glm5_next`), natively
  multimodal (24-layer vision tower);
- 45 decoder layers: 34 KDA linear-attention layers and 11 DeepSeek sparse
  attention layers (3, 7, ..., 43; `index_topk 2048`), mHC (`hc_mult 4`);
- MoE: 288 routed experts, top-8, one shared expert; layers 0-2 are dense;
- one MTP layer (`num_nextn_predict_layers: 1`, layer 45, BF16);
- `max_position_embeddings: 1048576`;
- ModelOpt NVFP4 for the routed experts and dense MLPs; attention, shared
  experts, gates, embeddings, `lm_head` and the vision tower stay BF16.
  190.4 GiB on disk.

## Image

The deployment reuses `localhost/vllm-backport:deepseek-v4.1-flash`
(`scripts/build-deepseek-v41-image.sh`): the pinned upstream source
(`344303947/dsv41-flash-pp5-170hx@d63af5a4`) already registers `glm5next`, and
the image carries the shared vLLM series including
`patches/vllm/0007-nvfp4-marlin-scale-factor-amax.patch`, which this model
**requires**. Without it every PP4 rank OOMs in
`process_weights_after_loading` (the NVFP4 -> Marlin conversion computed its
scale factor with a boolean gather that asks for 6.75 GiB per MoE layer).

## Roles

| role | GPUs (NUMA) | port | PP4 partition | weights per rank |
|---|---|---|---|---|
| prefill | 1,2,3,4 (0) | 8301 | `13,12,12,8` | 44.3 / 50.3 / 50.3 / 50.1 GiB |
| decode | 5,6,7,8 (1) | 8302 | `13,12,12,8` | same |
| lmcache | sees 1..8 | 5559 (HTTP 18559) | - | - |
| router | - | **9103** | - | - |

Layer sizes from the safetensors headers: layers 0-2 0.34 GiB, layers 3-44
~4.09 GiB, the MTP layer 13.84 GiB, embeddings / `lm_head` / vision 1.18 /
1.18 / 1.05 GiB. The last rank carries the MTP layer and `lm_head`, so it
gets the fewest layers. The partition must be identical on both roles: the
shared LMCache server holds one layout.

## Why these numbers

**The unified KV block is 4480 tokens under PP.** vLLM sizes the attention
block so that one attention page is at least one recurrent-state page
(`Setting attention block size to 4480 tokens ...`). With TP4 the KDA state is
split four ways and the block is 1152 (the value in the upstream README); with
PP every rank holds the full state (64 heads x 128 x 128 fp32 per layer), so
the block is 4480. The LMCache fork requires a state checkpoint at every chunk
boundary and at least one full block per prefill step, hence on both roles:

- LMCache `--chunk-size=4480 --separate-object-groups`;
- `--prefix-cache-retention-interval=4480`;
- `--max-num-batched-tokens=4480` (the fork rejects anything smaller:
  `Mamba-hybrid models with LMCache require max_num_batched_tokens >= block_size`).

`--kv-cache-dtype=bfloat16` keeps the block at 4480; an fp8 attention page
would double it.

**KV is 4 GiB per rank.** A 4480-token warmup step needs ~5.8 GiB of
activations on the 50 GiB ranks, and the LMCache server allocates ~210 MiB per
GPU when it registers the KV cache. At 6 GiB of KV the registration OOM'd on
ranks 1 and 2 (92 MiB free) and the engine waited forever. At 4 GiB:

- KV pool **1,236,992 tokens** (1.18x of a 1M-token request) on both roles;
- CUDA graph capture 0.31-0.51 GiB per rank;
- free after startup: rank 0 ~12 GiB, ranks 1-3 2.1-2.3 GiB.

Rank 0 has room for one more layer, but moving a 4.1 GiB layer from any other
rank does not raise the minimum free memory, so the partition stays as is.

## Verified (through the router, `:9103`)

| check | result |
|---|---|
| chat | correct answer; `reasoning` and `content` split by `glm45` |
| PD handoff, fresh 31,903-token prompt | decode hop read **31,360** tokens (7 x 4480) from LMCache, computed 543 |

Decode with MTP-3 (`scripts/benchmark-vllm.mjs`; counting/prose 500 tokens,
code stops naturally):

| shape | c=1 per request | accepted / step | c=4 full batch | c=8 full batch |
|---|---|---|---|---|
| counting | 67.0 tok/s | 2.86 | 215.4 tok/s | 312.3 tok/s |
| prose | 62.4 tok/s | 2.33 | 184.4 tok/s | 276.3 tok/s |
| code | 82.2 tok/s | 3.20 | 225.3 tok/s | 345.2 tok/s |

Not yet measured: long-context prefill (262k / 1M) memory and speed, image
inputs, tool calls.

## Run

```bash
# from this directory; .env (ignored) holds the machine paths and the key:
#   GLM_MODEL_CACHE=/root/app/vllm/cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4
#   VLLM_CACHE_PREFILL=/root/app/vllm/cache/glm53f-pd-prefill
#   VLLM_CACHE_DECODE=/root/app/vllm/cache/glm53f-pd-decode
#   LMCACHE_L2_PATH=/root/app/lmcache/glm-5.3-flash
#   VLLM_API_KEY=...
podman compose -f compose.pd.yml --podman-run-args=--ipc=host up -d
podman compose -f compose.pd.yml ps
```

`--podman-run-args=--ipc=host` is required (the connector shares the KV cache
over CUDA IPC and podman-compose ignores the compose `ipc` key).

Both engines run with `--shutdown-timeout=30`. LMCache maps each worker's KV
cache over CUDA IPC, and that memory stays allocated until the worker
unregisters or the server reaps it. Reaping happens after 120 s without a
heartbeat (`worker_reap_timeout_seconds`); in practice the measured gap was
143.8 s. vLLM's default (`0`, abort) kills EngineCore the moment it receives
SIGTERM. The workers then exit when they notice their parent has gone, and
when the API server (PID 1) exits, the container's remaining processes are
killed too. In one measured `podman stop` of the decode role, PP2 lost that race and never
unregistered. Its 4.3 GiB of KV stayed allocated on GPU 7, now charged to the
LMCache process, until the server reaped it 143.8 s later. A replacement
engine started within that window fails vLLM's startup check (`Free memory on
device ... is less than desired GPU memory utilization`). With a non-zero
timeout the engine drains, all four workers unregister, and the memory is free
as soon as `podman stop` returns (8 s). A role can then be replaced on its own,
without restarting the LMCache server. After a crash or `podman kill`, wait for
the server to log `Reaped GPU instance` before starting the replacement.

## W4A16 + DFlash2（PP4，单引擎）

本节对应 `compose.w4a16-dflash2.yml`（单引擎，不使用 LMCache）与镜像
`localhost/vllm-backport:glm-5.3-flash-w4a16-dflash2`；生产 PD 用的是同一镜像。

> **许可证提醒**：DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2` 采用
> **CC BY-NC-ND 4.0**，只能用于非商业测试，不得用于商业服务，也不得分发其修改版本。
> 目标模型与本仓库的补丁不受此限制，但部署整体受 drafter 许可证约束。

### 来源与固定版本

| 项目 | 固定版本 |
|---|---|
| vLLM 源码 | `344303947/dsv41-flash-pp5-170hx@d63af5a4`，加 DeepSeek V4.1 系列、公共系列、adaptive 系列（与 `deepseek-v4.1-flash` 镜像相同） |
| 移植来源 | Morrowmake/vllm-cmp170hx `ampere-glm53` @ `3a2bf16dae8b97f5ff2c7e9bc5809d24545e6340` |
| 推荐配置来源 | Morrowmake/glm53-flash-cmp170hx-recipe @ `a242b4fc53724bc5216f416fbbfaf7bf318b9f2c` |
| 目标模型 | `canada-quant/GLM-5.3-Flash-W4A16-MTP@5723f4d0`，宿主机 `/root/app/models/GLM-5.3-Flash-W4A16-MTP` |
| drafter | `incoai/GLM-5.3-Flash-DFlash2@bf582e4e`，宿主机 `/root/app/models/GLM-5.3-Flash-DFlash2` |

固定值记录在 `manifests/source.env`。补丁、Marlin 源码与脚本的 SHA256 记录在
`manifests/patches.sha256`，应用脚本会先校验它。

### 补丁系列（`patches/dflash2/`）

34 个补丁，按 `series` 顺序依次叠加在 DeepSeek 镜像源码之上。每个补丁都是相对前一个补丁结果的
git diff。除 0015 外，所有开关默认关闭，镜像本身不改变任何默认行为；compose 用
`VLLM_GLM53_OPT_PROFILE=default`（0033）一次打开，开关表见上文“优化开关”。
详细说明（英文）以及与开发编号的对照见 `patches/dflash2/README.md`。

| # | 作用 | 开关 | GPU 实测结论 |
|---|---|---|---|
| 0001 | GLM 目标模型提供 DFlash aux hidden state，并支持 PP 转发 | `VLLM_GLM5_AUX_HIDDEN_TENSOR=stream_mean` | 必需（启动通过） |
| 0002 | kpool tail ring 按草稿深度扩容，修复被拒草稿写坏 pool key 的问题 | 常开 | 正确性修复，293 项回滚矩阵通过 |
| 0003 | drafter 的滑窗层复用 MLA KV 张量（1024 token 一块，放在一个 MLA 页内）；Mamba 检查点按 target 状态页对齐 | 常开；`--block-size=5120`、`--max-num-batched-tokens=5184` | 必需 |
| 0004 | 自适应验证深度 | `VLLM_GLM5_DFLASH_ADAPTIVE_K*` | 有效 |
| 0005 | sm_80 thin-M BF16 GEMM | `VLLM_GLM5_THIN_GEMM` | 小幅有效 |
| 0006 | 编译版 sm_80 Marlin MoE decode（`vllm._ampere_marlin_C`） | `VLLM_GLM5_MARLIN_DECODE_CUDA` | 有效，11 项 GPU 测试通过 |
| 0007 | DFlash2 融合 grouped conv | `VLLM_DFLASH2_FUSED_GROUPED_CONV` | 小幅有效 |
| 0008 | 确定性小 M MoE 对齐（route v2 的回退路径） | `VLLM_GLM5_ROUTER_ALIGN_DECODE` | 约 1% |
| 0009 | sparse MLA decode 调度 | `VLLM_GLM5_SPARSE_MLA_MM_EXPERIMENTAL` | 有效 |
| 0010 | MoE route v2（gate 模式与原路由逐位一致；tc 模式与对方 `_moe_route_kernel` 逐位一致） | `VLLM_GLM5_ROUTE_V2=1`；补丁默认 `_GEMV=gate`，**profile 用 `tc`**（见下） | 约 +9%；tc 再 +0.6%（c1 29.05→29.21 步/秒） |
| 0011 | indexer gather 工作区收紧 | `VLLM_GLM5_INDEXER_GATHER_CLAMP` | KV 池 +25%（配合 util 0.95、logits 128 MiB） |
| 0012 | PP sparse MLA prefill（Gluon 64 头） | `VLLM_GLM5_PP_SPARSE_MLA_PREFILL` | 有效 |
| 0013 | PP Marlin MoE prefill 拆块 | `VLLM_GLM5_PP_MARLIN_PREFILL` | MoE prefill 1.23–1.31×，99.9% 逐位一致 |
| 0014 | PP KDA chunk prefill | `VLLM_GLM5_PP_KDA_PREFILL` | 比 FLA 更准，1.4–1.58× |
| 0015 | indexer Triton JIT 预热 | `VLLM_GLM5_INDEXER_JIT_WARMUP`（**默认开**） | 冷启动后首个 110k 请求 28.9 s → 18.0 s |
| 0016 | drafter fc 折叠到各 PP stage | `VLLM_GLM5_PP_FOLD_DRAFT_FC` | c8 +1.5% |
| 0017 | sm_80 Gluon mHC decode v2（post + pre + RMSNorm，M ≤ 32） | `VLLM_GLM5_DECODE_MHC_V2` | GPU 数值与 CUDA graph 测试通过；须与 0025 一起开 |
| 0018 | GDN/KDA spec-decode 元数据一次 Triton 构建 | `VLLM_GLM5_PROLOGUE_FUSE` | GPU 逐位一致 |
| 0019 | prepare_inputs 三个索引数组合并为一次 HtoD | `VLLM_GLM5_PROLOGUE_PACKED_H2D` | GPU 逐位一致 |
| 0020 | sparse MLA `req_id_per_token` 在设备端生成 | `VLLM_GLM5_SPARSE_MLA_DEVICE_REQ_IDS` | GPU 逐位一致 |
| 0021 | drafter 尾段（候选 lm_head、top-k、selector）移到 PP stage 2（含审计加固） | `VLLM_PP_DRAFT_TAIL_STAGE=2` | VERIFY 33,059 行零差异；FlashInfer 双流隔离 GPU 测试通过；c8 +11% |
| 0022 | sparse MLA profile 占位缓冲收紧到 16384 行 | `VLLM_GLM5_MLA_PROFILE_WS_CLAMP` | KV 池 1.68M → 2.095M；243k 单请求与 8×114k 压力通过 |
| 0023 | route v2 tile 读取顺序；可选“router 先于 shared experts 入队” | `VLLM_GLM5_ROUTE_V2_FIRST`（**默认 0**，profile 也设 0） | kernel 部分逐位不变；`=1` 整机变慢 |
| 0024 | indexer 权重缩放并入 decode FWHT 量化 | `VLLM_GLM5_DECODE_IDX_GLUE` | GPU 逐位一致 |
| 0025 | mHC prenorm GEMM 在所有 T 走 fp32 TileLang | `VLLM_GLM5_TARGET_PRENORM_FP32` | 与 0017 一起：相同历史首块与对方逐位一致，prose 接受率恢复 |
| 0026 | routed moe_sum 与 shared expert 加法融合 | `VLLM_GLM5_MOE_SUM_ADD` | GPU 逐位 512/512 |
| 0027 | shared experts 在 routed experts 之后入队 | `VLLM_GLM5_SHARED_EXPERT_REORDER` | 逐位一致；与 0026 一起 c8 counting 752.9 |
| 0028 | T ≥ 384 的 prenorm GEMM 改用对方的 bf16x3 张量核 Triton kernel（Apache-2.0，原样复制） | `VLLM_GLM5_PRENORM_BF16X3_MIN_TOKENS`（默认 384，0 = 等同 0025）；**只在 0025 打开时生效** | T=2312 prenorm 923 → 154 μs；T ≥ 384 与对方逐位一致，T < 384 与 0025 逐位一致；冷 prefill 见下 |
| 0029 | indexer wk+weights 合成一次双输出 thin GEMM（k 列 bf16 逐位不变；head weights 以 fp32 累加值输出，替代 cast + fp32 sgemm） | `VLLM_GLM5_IDX_DUAL_GEMM`（默认 0；需 `VLLM_GLM5_THIN_GEMM=1`；**profile 打开**） | 每 MLA 层省 15.7 μs；c1 counting 28.87→29.05 步/秒，c8 727.8→742.7；接受率不变；weights 误差为 sgemm 的 1.07×/1.24×（outlier/cancel 输入的 mean），max 更低 |
| 0030 | PP 段间跳的元数据缓存：每跳先发 32 字节 CPU header，元数据 pickle 字节与上一跳相同时不再发送 payload；张量与顺序不变 | `VLLM_PP_METADATA_CACHE`（默认 0；**profile 打开**；所有 PP rank 必须一致） | 与 0031 一起见“本轮实测”；CPU 逻辑 5 项 + 真实 Gloo 160 步通过 |
| 0031 | PP 段间跳的多个张量打包为一次 NCCL P2P（mHC hidden_states + fc 折叠部分和，每跳 2 次 → 1 次；字节与 padded 行数不变） | `VLLM_PP_PACK_TENSORS`（默认 0；**profile 打开**；启动时校验所有 PP rank 一致，不一致即报错） | 两卡 NCCL 120 步 packed=False/True 逐位一致；与 0030 一起 c1 counting 29.32→29.44 步/秒，c8 739.5→752.4 |
| 0032 | PP sparse MLA / KDA prefill 放开到 2312 行以上（原限制只是验证时的 chunk 大小） | 随 0012/0014；`VLLM_GLM5_PP_KDA_PREFILL_MAX_TOKENS` 默认 16384 | 2312/5120/8192/10240 行与分块调用逐位一致（输出与 KDA 末状态） |
| 0033 | 一个 profile 开关设置系列部署值 | `VLLM_GLM53_OPT_PROFILE=default` | 已设置的单项开关优先 |
| 0034 | prefill 把提示尾段状态交给 decode，decode 首步只采样（PD） | `VLLM_GLM53_PD_TAIL`（profile 打开） | 四段状态与首步 logits/候选逐位一致；107k 两跳 23.55 → 20.17 s |
| 0035 | 首轮保留本地 partial prefix 的 drafter 尾窗 | `VLLM_GLM53_PREFIX_RETENTION`（默认关，profile 打开） | 不改 checkpoint/lookup 语义；短多轮脚本 `tests/dflash2/test_prefix_retention.py` |

**route v2 用 tc 模式（profile 中 `VLLM_GLM5_ROUTE_V2_GEMV=tc`）**：tc 的 router logits 与我们旧的 gate 路径（`_bf16_gemv_kernel`）
**不逐位一致**（归约顺序不同，第 8、9 名专家近似并列时可能翻转），但与对方的 `_moe_route_kernel` **逐位一致**。选它有两个原因：
一是与对方数值对齐，prose 接受率 1.232/轮，与对方相同（gate 为 1.218）；二是减少 SM 争用。gate 模式的 GEMV 一次铺开
288 个 8-warp CTA，shared experts（侧流）只能等它排空，结果与 routed Marlin 重叠更多；tc 是一次 80-CTA launch。
单卡 MoE 层 bench（M=8）：gate 385.5、tc 375.9、无路由 363.8 μs/层；整机 c1 counting 29.05→29.21 步/秒
（`/tmp/dcp/dflash-port/STEP-GAP-2.md` §7–8）。

0028 的代码位于 0025 的 `VLLM_GLM5_TARGET_PRENORM_FP32=1` 分支内部，0025 关闭时完全不可达，所以系列默认行为不变
（静态检查断言了这一点）。0028 的 kernel 会重写 sqrsum，因此 profile 显式设 `VLLM_MHC_POST_FUSE_SQRSUM=0`
（该变量在我们树中已存在、默认 0；设 1 会让 mhc_post 额外计算一次随后被覆盖的 sqrsum）。

0023 的 `VLLM_GLM5_ROUTE_V2_FIRST` 在开发版中随 route v2 默认开启，
整机实测 `=1` 变慢，因此正式补丁把默认值改为 0（只有设为 `1` 才启用），profile 仍显式设 `0`；
这样即使有人漏掉这项 env，也不会进入较慢的顺序。


未纳入的开发补丁：context-KV graph、旧版 mHC v2（开发 0009，已由 0017 完整移植取代）、KDA 双投影（三者均无整机收益）；mHC v1 数值
（未在 GPU 上验证）；force-file、draft trace、PP trace、相同历史（accept-same-history）overlay（仅用于诊断）。去掉这些补丁后，其余补丁只有
gather clamp 的 `envs.py` 上下文需要重新生成。最终源码与开发树逐文件比对，差异只有被排除的补丁。

### 编译版 Marlin 扩展（`native/ampere_marlin/`）

源码取自 Morrowmake `3a2bf16dae`，`decode.cu`、`decode_orig.cu`、`module.cpp` 均未修改，
许可证为 Apache-2.0（保留 `LICENSE` 与出处说明）。镜像构建时由
`install-into-vllm.sh` 编译，并安装为 `<site-packages>/vllm/_ampere_marlin_C.abi3.so`。
0006 在 `VLLM_GLM5_MARLIN_DECODE_LIB` 为空时本来就会 `import vllm._ampere_marlin_C`，
因此加载逻辑不需要修改，只更新了错误提示与 docstring 中的路径。部署时不再需要挂载 `.so`。

构建时的变通（已在镜像中实测）：镜像中的 CUDA toolkit 缺少 `cusparse.h`、`cusolverDn.h`
等头文件，这些文件位于 pip 安装的 `nvidia/cu13/include`。不能把该目录整体加入 `CPATH`，
否则会与 toolkit 的 CUDA runtime 头文件冲突（`__cudaLaunch` 宏错误）。因此脚本只把
toolkit 中缺失的头文件逐个软链到临时目录，再把这个临时目录加入 `CPATH`。
构建期自检（`native/ampere-marlin-selfcheck.py`，不需要 GPU）会用 0006 的真实加载器校验
torch/CUDA/ABI 与 4 个算子 schema，并确认所有开关保持默认值。

### 构建

```bash
cd /root/app/vllm-cmp170hx
CONTAINER_ENGINE=podman bash scripts/build-glm53-dflash2-image.sh
# -> localhost/vllm-backport:glm-5.3-flash-w4a16-dflash2
```

与 `build-deepseek-v41-image.sh` 一样，构建从 `BASE_IMAGE` 开始：拉取固定源码，执行
`scripts/apply-glm53-dflash2-patches.sh`（先校验 SHA256，再应用 DeepSeek、公共、adaptive
三个系列，然后对每个 dflash2 补丁执行 `git apply --check` 和 `git apply`，最后做静态检查），
再复制 LMCache 客户端载荷，并在镜像内编译 Marlin 扩展。不会叠加在现有镜像之上。
镜像是 DeepSeek 镜像的超集，GLM 以外的模型行为不变。

只在 CPU 上检查补丁系列（不需要容器）：
`bash models/glm-5.3-flash/tests/check-dflash2-series.sh`。
GPU 测试的运行方法见 `tests/dflash2/README.md`。

### 运行

```bash
cd models/glm-5.3-flash
# .env：VLLM_API_KEY=...，GLM_DFLASH2_CACHE=/root/app/vllm/cache/glm53f-w4a16-dflash2
# GPU 默认 5..8：GLM_GPU_0..GLM_GPU_3
mkdir -p /root/app/vllm/cache/glm53f-w4a16-dflash2/{tmp,triton,inductor}
podman compose -f compose.w4a16-dflash2.yml --podman-run-args=--ipc=host up -d
```

主要参数：V2 runner；`VLLM_PP_LAYER_PARTITION=13,11,11,10`；`--block-size=5120`；
`--max-num-batched-tokens=5184`；`--long-prefill-token-threshold=0`；KV 12 GiB；
`FULL_AND_PIECEWISE`；prefix caching；`--mamba-cache-mode=align`；DFlash2
`num_speculative_tokens=3`，自适应深度 `7,5`，`ACCEPT=0`（只按负载选深度：1 个请求验证 7、2 个 5、更多 3）；
`VLLM_GLM53_OPT_PROFILE=default`；`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`；Triton 与 Inductor 缓存放在挂载目录中。
下面“实测性能”“本轮实测”是 block 4608、batch 2312 时测得的单引擎数据。

分层的另一选择是 `GLM_PP_LAYER_PARTITION=12,12,12,9`：c8 约 +10%，但 KV 池从 1.68M
降到 1.24M token（这两个数是 0022 之前测得的）。

“实测性能”一节第一张表是 0001–0016 镜像的数据（不挂载源码）；0017–0031 见“本轮实测”，
其中包含新镜像（不挂载源码）的复测。

### 实测性能（node2，GPU5–8，PP4）

测试方法：2 轮预热，5 轮正式，取中位数，每次 500 token。“对方”指 Morrowmake 引擎在同一组卡上的结果。

新镜像 `localhost/vllm-backport:glm-5.3-flash-w4a16-dflash2`（Marlin 扩展编译进镜像，**不挂载源码**，
本 compose 原样启动）的实测，中位数 [最小–最大]：

| 负载 | 本镜像 单用户 | 对方 单用户 | 本镜像 c8 合计 | 对方 c8 合计 |
|---|---|---|---|---|
| counting | 215.2 tok/s [214.6–218.7] | 230 | 664.0 | 766 |
| code | 152.2 [141.0–154.1] | 157 | 540.7 | 610 |
| prose | 62.5 | 68 | 308.2 | 337 |

- 单用户每秒步数：counting 28.0，对方 30.0。prose 的单 prompt 吞吐受文本分叉影响很大
  （同一配置在不同运行中实测 46–62 tok/s）；8 个不同 prose prompt 的平均每轮接受数
  我们 1.384、对方 1.386，没有系统差异。
- 冷 prefill（容器重启后，先发一个 8k 预热请求）：7k token 3,675 tok/s，28k 5,161 tok/s，
  114k 5,698 tok/s（TTFT 20.0 s；对方 5,862 tok/s）。0015 之前，重启后首个 110k 请求需 28.9 s，
  原因是 indexer logits kernel 在长上下文才首次编译。
- KV 池 1,678,534 token（对方 1,917,988）。
- 卡组差异：node2 的 GPU1 最高 SM 1410 MHz、HBM 1458 MHz，其余卡为 1695/1728 MHz。
  GPU1–4 这组比 GPU5–8 单用户慢约 4%，跨卡组的数字不能直接比较；上表两边都在 GPU5–8 上测得。
- 层切分：默认 `13,11,11,10`。`12,12,12,9` 让 c8 counting 提高约 10%（在 GPU1–4 上 660 → 720），
  但 KV 池降到约 1.24M；`12,11,11,11` 更差（c8 625，KV 1.11M），因为最后一段还承担 drafter。
  c8 时开发版 PP 流水 trace 诊断显示 rank0 的 GPU 约 98% 忙、rank3 约 65%，瓶颈在 rank0。

### 本轮实测（0017–0031）

0030 + 0031（PP 元数据缓存 + 张量打包）实测：node2 GPU5–8，2 轮预热、5 轮正式取中位数；基础为 0001–0029 全开 +
tc 路由（即下方“新镜像复测”的配置），两项同时打开：

| 负载 | c1 | c8 合计 | 对方 c1 | 对方 c8 |
|---|---|---|---|---|
| counting | 226.0 tok/s（每秒步数 29.44） | 752.4 | 230.1（29.97） | 765.7 |
| code | 155.7 | 609.2 | 157.5 | 608.5 |
| prose | 66.9（每秒步数 30.03，每步输出 2.22） | 335.4 | 68.2（30.61，2.22） | 341.1 |

- 不开这两项时同一镜像：c1 counting 225.1（29.32 步/秒），c8 739.5 / 589.9 / 324.0（counting / code / prose）。
- 与对方差距：c1 每秒步数 −1.8%（counting）/−1.9%（prose），c8 counting −1.7%、code +0.1%、prose −1.7%，所有负载 ≤ 约 2%。
- prose 接受率 1.232/轮（与对方相同）；greedy 3 类×4 次×128 token，所有偏离 top-1 的位置 logprob 差 ≤ 0.75。
- GPU 单测：`test_pp_pack_hop_gpu.py --no-bench` 两卡 NCCL 120 步 packed=False/True 逐位一致。
- 上述数字是在 0001–0029 镜像树上以开发补丁加入这两项后测得。

**新镜像复测（0001–0031，本 compose 原样启动、不挂载源码，GPU5–8，同方法）**：

| 负载 | c1 | c8 合计 | 对方 c1 | 对方 c8 |
|---|---|---|---|---|
| counting | 225.6 tok/s（每秒步数 29.38，−2.0%） | 753.3 [748.6–758.8]（−1.6%） | 230.1（29.97） | 765.7 |
| code | 168.1 | 603.8（−0.8%） | 157.5 | 608.5 |
| prose | 66.8（每秒步数 29.99，−2.0%；每步输出 2.22） | 328.7（每秒步数 23.08 vs 23.54，−2.0%） | 68.2（30.61） | 341.1 |

- 冷 prefill（7k / 28k / 114k）：3,705 / 5,215 / 5,713 tok/s（对方 3,839 / 5,316 / 5,862，−3.5% / −1.9% / −2.5%）。
- KV 池 2,054,477 token（对方 1,917,988，+7%）；prose 接受率 1.232/轮，与对方相同。
- c8 prose 吞吐差 3.6%，其中步速差 2.0%，其余来自各自生成轨迹的每步接受数（1.75 vs 1.79；8 并发时两边自身均有批次相关的输出分叉）。
- c8 时各 PP 段每步 GPU 计算与对方持平（rank0 8.54/8.53 ms、rank1 7.99/7.83、rank2 9.27/9.14）。
- 正确性：同一引擎 greedy 3 类×4 次×128 token，共 3 组（36 条）。偏离 top-1 的位置中 35 条序列的最大 logprob 差 ≤0.5；
  1 条在第 6 个 token 分叉后，第 45 个 token 差 1.5（用 prefill 的 prompt_logprobs 评分，分叉后上下文不同；未复现，记录备查）。
  GPU 单测（镜像内）：0030/0031 的 CPU/gloo/两卡 NCCL 测试全部通过（gloo 测试需 `--security-opt apparmor=unconfined`）。

以下为 0030/0031 之前的结果。

> 下面第一张表在 node2 的**源码挂载开发部署**上测得（开发树按上表顺序应用，开关与本 compose 相同）；
> 紧接着的“新镜像复测”是 0001–0029 镜像、本 compose 原样启动、**不挂载源码**的结果。

新镜像复测（`localhost/vllm-backport:glm-5.3-flash-w4a16-dflash2`，0001–0029，本 compose 原样启动、不挂载源码，GPU5–8，同方法）：

| 负载 | c1 | c8 合计 | 对方 c1 | 对方 c8 |
|---|---|---|---|---|
| counting | 225.1 tok/s（每秒步数 29.32） | 739.5 [726.0–746.9] | 230.1（29.97） | 765.7 |
| code | 149.2 [143.1–160.8] | 589.9 | 157.5 | 608.5 |
| prose | 66.6（每秒步数 29.88，每步输出 2.22） | 324.0 [314.7–345.5] | 68.2（30.61，2.22） | 341.1 |

- 冷 prefill：7k 3,709、28k 5,180、114k 5,725 tok/s（对方 3,839 / 5,316 / 5,862）。
- KV 池 2,054,477 token（0021 审计修复后尾段 workspace 在 KV 定容前计入预算；对方 1,917,988，我们多 7%）。
- prose 接受数（metrics）1.232/轮，与对方相同。
- 与对方差距：c1 每秒步数 −2.2%（counting）/−2.4%（prose），c8 counting −3.4%，c8 code −3.1%，114k prefill −2.3%。
  c1 code 单 prompt 吞吐受接受率轨迹影响大（运行间 143–161），不宜单独比较。
- 剩余差距的定位：同一 kernel 单独运行时两边耗时相同；差距来自 shared expert（侧流 thin GEMM）与 routed Marlin
  并发时的 SM 争用（我们 Marlin 与侧流重叠 28.7%，对方 18.9%）。`tc` 路由已缓解一部分。
- 正确性：同一引擎 T=0 greedy 3 类×4 次×128 token，所有偏离 top-1 的位置 logprob 差距 0–0.75（近似平局）。
  GPU 单测（镜像内）：0021 尾段与 FlashInfer 隔离、0018–0020、Marlin decode、kpool 回滚矩阵 293 项、0024、0026、0029 全部通过。

开发部署结果：

GPU5–8，PP4，2 轮预热、5 轮正式取中位数，每次 500 token。我们为 0026+0027（开发 0027+0028）全开的配置，
对方为 Morrowmake 引擎同卡组结果。

| 负载 | 我们 c1 | 对方 c1 | 我们 c8 合计 | 对方 c8 合计 |
|---|---|---|---|---|
| counting | 224.8 tok/s（每秒步数 29.28） | 230.1（29.97） | 752.9 | 765.7 |
| code | 153.9 | 157.5 | 600.7 | 608.5 |
| prose | 66.6（每秒步数 29.88，每步输出 2.222） | 68.2（30.61，2.222） | 328.5 | 341.1 |

- 冷 prefill（0028，源码挂载开发部署，容器重启后先发 8k 预热请求）：

  | 长度 | 0028 之前 | 0028 之后 | 对方 |
  |---|---|---|---|
  | 7k | 3,675 tok/s | 3,675–3,788 | 3,839 |
  | 28k | — | 5,252–5,268 | 5,316 |
  | 114k | 5,527 | 5,723–5,793 | 5,862 |

  打开 0028 后 prose 接受率不变（每轮 1.232）。
- KV 池：我们 2,095,119 token（0022 之前 1,678,534），对方 1,917,988。
- 与上一版镜像相比：c1 counting 215.2 → 224.8，c8 counting 664.0 → 752.9；与对方的差距
  从 c1 约 6.5%、c8 约 13% 缩小到 c1 约 2.3%、c8 约 1.7%（counting）。
- **关键发现（prose 接受率）**：之前 prose 每步输出比对方低，原因不在 drafter 或自适应深度，而在目标模型数值：
  我们在 T=32 的验证批上，mHC prenorm GEMM 走 BF16 cuBLAS，而对方在所有 T 都走 fp32 TileLang；
  mHC decode 路径（TileLang vs 对方 Gluon v2）也不同。0017 与 0025 一起打开后，相同历史下的首块
  目标输出与对方**逐位一致**，prose 接受率恢复（每轮 1.232，每步输出 2.222，与对方相同）。

无收益、未纳入的项：

PP metadata cache（`VLLM_PP_METADATA_CACHE`）与 PP 传输张量打包（`VLLM_PP_PACK_TENSORS`）之前列在此表中，
记为“整机无收益”；那是在较早的基线上测的（当时其他 decode 瓶颈占主导，PP 通信不在关键路径上）。
在 0001–0029 完整栈 + tc 路由上两项一起打开测得有效（c1 counting 29.32→29.44 步/秒，c8 counting 739.5→752.4、
code 589.9→609.2、prose 324.0→335.4），已作为 0030、0031 纳入，从此表移除。

| 项 | 结果 |
|---|---|
| `VLLM_GLM5_ROUTE_V2_FIRST=1`（0023 的入队顺序部分） | 整机变慢；补丁保留但默认 0，compose 设 0 |
| thin GEMM 配置选择与对方相同（评估时曾用编号 0030，与现在的 0030 无关；`VLLM_GLM5_THIN_GEMM_MM_SELECT`） | 整机 c8 729.2，无收益；已从系列删除 |
| shared experts 侧流设 stream 优先级（评估时曾用编号 0031，与现在的 0031 无关；`VLLM_GLM5_SHARED_STREAM_PRIORITY`） | 单卡 MoE 层 bench：tc+high 381.3 vs tc+default 375.9 μs/层，更慢；已从系列删除 |

### 正确性证据

- 开/关投机解码对比：T=0 greedy，3 类 prompt，每类 4 次，每次 128 token。短提示词与计数在两种模式下
  各自完全稳定；所有偏离 top-1 的位置，logprob 差距都不超过 0.75，属于 BF16 近似平局。
- 0002：GPU 回滚矩阵 293 项通过。0006：11 项 GPU 测试通过。0010 gate 模式与原路由逐位一致；tc 模式的 router logits 与对方 `_moe_route_kernel` 逐位一致（`bench_route`：mm_logits_bitwise True）。
  0013：99.9% 逐位一致。0014：误差比 FLA 更小。
- 0018–0020、0023（kernel 部分）、0024、0026、0027 在 GPU 上与关闭时逐位一致；0021 用
  `VLLM_PP_DRAFT_TAIL_VERIFY=1` 对比 33,059 行草稿零差异。0017+0025 使相同历史首块与对方逐位一致。
- 0005、0006、0012、0013、0014、0016 都会改变求和顺序，与未打补丁的路径不是逐位相同。
  因此逐位比较只能在开关相同的两次运行之间进行。
- 新镜像的自稳定性（同一引擎重复 4 次）：短提示词与计数在第 19/33/42 个 token 处有运行间分叉，
  长上下文有冷/暖 prefix cache 路径分叉。所有偏离 top-1 的位置 logprob 差距为 0.125–0.75，
  分叉点本身为 0.125–0.25，均为近似平局。推测原因是每步验证行数随草稿接受情况变化，
  不同行数走不同 kernel（编译版 Marlin 只覆盖 M=4/8），舍入不同；未单独证明。
  没有观察到大差距的错误 token。

### 已知差距与未移植项

以下为 0017–0027 之前的分析；0017（mHC v2）、0024（idx glue 的 fwht/expand 部分）、0021（drafter 尾段）
已在本轮移植，剩余差距见“本轮实测”。

同卡组（GPU5–8）与对方相比：单用户 counting 低约 6.5%（每秒步数 28.0 vs 30.0），c8 低约 11–13%。
长 prefill 已基本持平。对方 decode 融合组的消融（全部关闭）让对方 c1 counting 降约 7%、c8 降约 9%，
其中 mHC 子项还会改变接受率的单 prompt 表现。下面各项都没有移植；
它们对剩余差距的贡献是推测，没有逐项测量：

- KDA v2 decode 融合（mHC v2 已由 0017 移植）；
- idx glue 的 weights、glue、cache 三项（0024 只移植了 fwht 与 expand；moesum 由 0026 移植）；
- 对方的 PP 传输包原实现（0030/0031 以自己的方式实现了元数据缓存与张量打包，保留 padded 行和每步 pickle；以下原开关未移植：`VLLM_PP_PACKED_HOP`、`VLLM_PP_HOP_NO_METADATA`、`VLLM_PP_SPLIT_DRAFT_EVENT`、
  `VLLM_PP_SPREAD_DECODES`）；
- fair prefill（`--prefill-chunk-with-decodes`、`--max-num-partial-prefills`）、drafter 显存相关开关
  （`ROPE_FIT`、`SELECTOR_SHARD`）、编译版 Marlin prefill（`VLLM_GLM5_MARLIN_PREFILL_CUDA`）；
- 不要与 PP 一起使用 `--load-format fastsafetensors`：对方在 PP 下会把 drafter 强制改回 `auto`，这个修复没有移植。
