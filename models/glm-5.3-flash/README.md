# GLM-5.3-Flash NVFP4 PD (prefill/decode disaggregated) on eight CMP 170HX GPUs

> The W4A16 + DFlash2 PP4 deployment (own image, own patch series) is described
> in [W4A16 + DFlash2](#w4a16--dflash2pp4单引擎) below.

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

本节对应 `compose.w4a16-dflash2.yml` 与镜像
`localhost/vllm-backport:glm-5.3-flash-w4a16-dflash2`。它与上面的 NVFP4 PD
部署相互独立：目标模型是 W4A16 量化的 GLM-5.3-Flash，投机解码使用 DFlash2
drafter，不使用 LMCache。

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

28 个补丁，按 `series` 顺序依次叠加在 DeepSeek 镜像源码之上。每个补丁都是相对前一个补丁结果的
git diff。除 0015 外，所有开关默认关闭，镜像本身不改变任何默认行为；compose 通过 env 打开它们
（0023 的 `VLLM_GLM5_ROUTE_V2_FIRST` 除外，见下）。
详细说明（英文）以及与开发编号的对照见 `patches/dflash2/README.md`。

| # | 作用 | 开关 | GPU 实测结论 |
|---|---|---|---|
| 0001 | GLM 目标模型提供 DFlash aux hidden state，并支持 PP 转发 | `VLLM_GLM5_AUX_HIDDEN_TENSOR=stream_mean` | 必需（启动通过） |
| 0002 | kpool tail ring 按草稿深度扩容，修复被拒草稿写坏 pool key 的问题 | 常开 | 正确性修复，293 项回滚矩阵通过 |
| 0003 | drafter 的滑窗层复用 MLA KV 张量 | 常开；需要 `--block-size=4608` | 必需 |
| 0004 | 自适应验证深度 | `VLLM_GLM5_DFLASH_ADAPTIVE_K*` | 有效 |
| 0005 | sm_80 thin-M BF16 GEMM | `VLLM_GLM5_THIN_GEMM` | 小幅有效 |
| 0006 | 编译版 sm_80 Marlin MoE decode（`vllm._ampere_marlin_C`） | `VLLM_GLM5_MARLIN_DECODE_CUDA` | 有效，11 项 GPU 测试通过 |
| 0007 | DFlash2 融合 grouped conv | `VLLM_DFLASH2_FUSED_GROUPED_CONV` | 小幅有效 |
| 0008 | 确定性小 M MoE 对齐（route v2 的回退路径） | `VLLM_GLM5_ROUTER_ALIGN_DECODE` | 约 1% |
| 0009 | sparse MLA decode 调度 | `VLLM_GLM5_SPARSE_MLA_MM_EXPERIMENTAL` | 有效 |
| 0010 | MoE route v2（gate 模式与原路由逐位一致） | `VLLM_GLM5_ROUTE_V2=1`、`_GEMV=gate` | 约 +9% |
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
| 0023 | route v2 tile 读取顺序；可选“router 先于 shared experts 入队” | `VLLM_GLM5_ROUTE_V2_FIRST`（**默认 0**，compose 也设 0） | kernel 部分逐位不变；`=1` 整机变慢 |
| 0024 | indexer 权重缩放并入 decode FWHT 量化 | `VLLM_GLM5_DECODE_IDX_GLUE_0024` | GPU 逐位一致 |
| 0025 | mHC prenorm GEMM 在所有 T 走 fp32 TileLang | `VLLM_GLM5_TARGET_PRENORM_FP32_0026` | 与 0017 一起：相同历史首块与对方逐位一致，prose 接受率恢复 |
| 0026 | routed moe_sum 与 shared expert 加法融合 | `VLLM_GLM5_MOE_SUM_ADD` | GPU 逐位 512/512 |
| 0027 | shared experts 在 routed experts 之后入队 | `VLLM_GLM5_SHARED_EXPERT_REORDER` | 逐位一致；与 0026 一起 c8 counting 752.9 |
| 0028 | T ≥ 384 的 prenorm GEMM 改用对方的 bf16x3 张量核 Triton kernel（Apache-2.0，原样复制） | `VLLM_GLM5_TARGET_PRENORM_FP32_0026B_MIN_TOKENS`（默认 384，0 = 等同 0025）；**只在 0025 打开时生效** | T=2312 prenorm 923 → 154 μs；T ≥ 384 与对方逐位一致，T < 384 与 0025 逐位一致；冷 prefill 见下 |

0028 的代码位于 0025 的 `VLLM_GLM5_TARGET_PRENORM_FP32_0026=1` 分支内部，0025 关闭时完全不可达，所以系列默认行为不变
（静态检查断言了这一点）。0028 的 kernel 会重写 sqrsum，因此 compose 显式设 `VLLM_MHC_POST_FUSE_SQRSUM=0`
（该变量在我们树中已存在、默认 0；设 1 会让 mhc_post 额外计算一次随后被覆盖的 sqrsum）。

0017–0028 的开关名沿用开发编号（如 `_0024`、`_0026` 后缀），以免已有部署改名；补丁编号与开发编号的
对照见 `patches/dflash2/README.md`。0023 的 `VLLM_GLM5_ROUTE_V2_FIRST` 在开发版中随 route v2 默认开启，
整机实测 `=1` 变慢，因此正式补丁把默认值改为 0（只有设为 `1` 才启用），compose 仍显式设 `0`；
这样即使有人漏掉这项 env，也不会进入较慢的顺序。

未纳入的开发补丁：context-KV graph、旧版 mHC v2（开发 0009，已由 0017 完整移植取代）、KDA 双投影（三者均无整机收益）；
PP metadata cache（开发 0025）与 PP 传输张量打包（开发 0029），两者整机无收益；mHC v1 数值
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
# GPU 默认使用 5..8（测试卡）。生产 prefill 使用 1..4：GLM_GPU_0=1 ... GLM_GPU_3=4
mkdir -p /root/app/vllm/cache/glm53f-w4a16-dflash2/{tmp,triton,inductor}
podman compose -f compose.w4a16-dflash2.yml up -d
```

主要参数：V2 runner；`VLLM_PP_LAYER_PARTITION=13,11,11,10`；`--block-size=4608`；
`--max-num-batched-tokens=2312`；`--long-prefill-token-threshold=0`；
`--gpu-memory-utilization=0.95`；`FULL_AND_PIECEWISE`；prefix caching；
`--mamba-cache-mode=align`；DFlash2 `num_speculative_tokens=3`，自适应深度 `7,5`，`ACCEPT=0`（只按负载选深度：1 个请求验证 7、2 个 5、更多 3）；
0005–0028 全部开启（0023 的 `VLLM_GLM5_ROUTE_V2_FIRST=0`）；`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`；Triton 与 Inductor 缓存放在挂载目录中。

分层的另一选择是 `GLM_PP_LAYER_PARTITION=12,12,12,9`：c8 约 +10%，但 KV 池从 1.68M
降到 1.24M token（这两个数是 0022 之前测得的）。

“实测性能”一节第一张表是 0001–0016 镜像的数据（不挂载源码）；0017–0028 见“本轮实测”，
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

### 本轮实测（0017–0027，源码挂载开发部署）

> 下面第一张表在 node2 的**源码挂载开发部署**上测得（开发树按上表顺序应用，开关与本 compose 相同）；
> 紧接着的“新镜像复测”是 0001–0028 镜像、本 compose 原样启动、**不挂载源码**的结果。

新镜像复测（`localhost/vllm-backport:glm-5.3-flash-w4a16-dflash2`，GPU5–8，同方法）：

| 负载 | c1 | c8 合计 | 对方 c1 | 对方 c8 |
|---|---|---|---|---|
| counting | 221.6 tok/s（每秒步数 28.87） | 727.8 [714.0–733.2] | 230.1（29.97） | 765.7 |
| code | 156.3 | 594.4 | 157.5 | 608.5 |
| prose | 65.3（每秒步数 29.47，每步输出 2.21） | 332.2 | 68.2（30.61） | 341.1 |

- 冷 prefill：7k 3,696、28k 5,104、114k 5,651 tok/s（对方 3,839 / 5,316 / 5,862）。
- KV 池 2,054,477 token（0021 审计修复后尾段 workspace 在 KV 定容前计入预算，比开发部署少约 2%；对方 1,917,988）。
- prose 接受数（metrics）1.218/轮（对方 1.232）。
- 正确性：同一引擎 T=0 greedy 3 类×4 次×128 token，所有偏离 top-1 的位置 logprob 差距 0–0.5（近似平局）。
- c8 counting 在开发部署的多次运行中为 724–753，本次 727.8；运行间波动约 ±2%，因此 c8 与对方的差距在 1.7–5% 之间。

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

| 项 | 结果 |
|---|---|
| PP metadata cache（开发 0025，`VLLM_PP_METADATA_CACHE_0025`） | 整机无收益 |
| PP 传输张量打包（开发 0029，`VLLM_PP_PACK_TENSORS_0029`） | 整机无收益 |
| `VLLM_GLM5_ROUTE_V2_FIRST=1`（0023 的入队顺序部分） | 整机变慢；补丁保留但默认 0，compose 设 0 |

### 正确性证据

- 开/关投机解码对比：T=0 greedy，3 类 prompt，每类 4 次，每次 128 token。短提示词与计数在两种模式下
  各自完全稳定；所有偏离 top-1 的位置，logprob 差距都不超过 0.75，属于 BF16 近似平局。
- 0002：GPU 回滚矩阵 293 项通过。0006：11 项 GPU 测试通过。0010 gate 模式与原路由逐位一致。
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
- PP 传输包（`VLLM_PP_PACKED_HOP`、`VLLM_PP_HOP_NO_METADATA`、`VLLM_PP_SPLIT_DRAFT_EVENT`、
  `VLLM_PP_SPREAD_DECODES`）；
- fair prefill（`--prefill-chunk-with-decodes`、`--max-num-partial-prefills`）、drafter 显存相关开关
  （`ROPE_FIT`、`SELECTOR_SHARD`）、编译版 Marlin prefill（`VLLM_GLM5_MARLIN_PREFILL_CUDA`）；
- 不要与 PP 一起使用 `--load-format fastsafetensors`：对方在 PP 下会把 drafter 强制改回 `auto`，这个修复没有移植。
