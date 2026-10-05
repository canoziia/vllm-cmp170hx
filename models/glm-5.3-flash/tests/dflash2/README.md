# GLM-5.3-Flash W4A16 + DFlash2 测试

从开发目录（dflash-port/tests）复制，只保留与生产补丁（`../../patches/dflash2/series`）对应的测试。文件名去掉了旧的开发编号；文件内部的注释与 docstring 仍使用开发编号，对照见下表与 `../../patches/dflash2/README.md`。

| 生产补丁 | 开发编号 | 文件 | 需要 | 状态 |
|---|---|---|---|---|
| 0001 | 0001 | `test_aux_hidden_states_pp.py` | torch + 打过补丁的 vllm（CPU） | 本仓库未重跑 |
| 0002 | 0002 | `test_kpool_tail_ring.py` | CPU 部分 + CUDA 部分（无 GPU 自动跳过） | 本仓库未重跑 |
| 0002 | 0002 | `test_kpool_rollback_matrix_gpu.py` | sm_80；`KPOOL_CANDIDATE` 或 `GLM_DFLASH2_TREE` 指向被测 `kpool_compress.py` | GPU 实测 293 项通过（开发阶段） |
| 0003 | 0003 | `test_kv_layout_dflash_pure.py` | 仅 python3；`GLM_DFLASH2_PRISTINE_TREE`、`GLM_DFLASH2_TREE` | 本地通过（CPU 脚本） |
| 0004 | 0004 | `test_adaptive_k_pure.py` | 仅 python3；`GLM_DFLASH2_TREE` | 本地 27/27 通过；force-file 用例改为“确认不存在” |
| 0004 | 0004 | `test_adaptive_k_container.py` | 容器 + vLLM 源码 `tests/`（CPU） | 未运行 |
| 0005 | 0005 | `test_thin_gemm_pure.py` | 仅 python3；`GLM_DFLASH2_TREE` | 本地通过 |
| 0005 | 0005 | `test_thin_gemm_container.py` | 容器；GPU 层需要 CUDA | 未确认 |
| 0006 | 0006 | `test_marlin_decode_gpu.py`、`marlin_decode_common.py`、`bench_marlin_decode.py` | sm_80 + `vllm._ampere_marlin_C` | GPU 实测 11 项通过（开发阶段，`.so` 由挂载提供） |
| 0008 | 00010 | `test_router_align_gpu.py` | sm_80 | 未确认 |
| 0009 | 0012 | `bench_sparse_mla_decode_gpu.py` | sm_80；`--mm` 需要 Morrowmake 源文件 | 仅基准 |
| 0010 | 0013 | `test_route_v2_gpu.py`、`bench_route_v2_gpu.py` | sm_80 | 未确认 |
| 0011 | 0016 | `test_indexer_gather_clamp.py` | 容器（GPU） | 未确认 |
| 0012 | 0017 | `test_pp_sparse_prefill_gpu.py` | sm_80 | 未确认 |
| 0013 | 0019 | `test_pp_marlin_prefill_gpu.py` | sm_80，`VLLM_GLM5_PP_MARLIN_PREFILL=1` | GPU：MoE prefill 1.23–1.31×，99.9% 逐位 |
| 0014 | 0020 | `test_pp_kda_prefill_gpu.py` | sm_80 | GPU：比 FLA 更准，1.4–1.58× |
| 0015 | 0022 | `test_indexer_jit_warmup_gpu.py` | sm_80 | 整机：冷启动首个 110k 请求 28.9 s → 18.0 s |
| 0016 | 0023 | `test_pp_fold_draft_fc_container.py`、`test_pp_fold_draft_fc_gpu.py` | CPU 容器 / sm_80（`--draft` 指向 drafter） | 未确认 |

| 0017 | 0017 | `test_mhc_v2_full_gpu.py` | sm_80；`OURS_V2` 默认取 `$GLM_DFLASH2_TREE/.../mhc_decode_v2.py`；`GATE=1` 测真实入口 | GPU 数值与 graph 通过（开发阶段） |
| 0018–0020 | 0018–0020 | `test_host_overhead_gpu.py` | CUDA + 打过补丁的 vllm | GPU 逐位通过（开发阶段） |
| 0021 | 0021 | `test_pp_draft_tail_gpu.py` | CUDA（thin GEMM 层需 sm_80）；`--draft` 可选 | 开发阶段（审计前版本）编写 |
| 0021 | 0021 | `test_pp_tail_topk_audit_gpu.py` | CUDA + FlashInfer（镜像内版本） | FlashInfer 双流隔离 GPU 通过（开发阶段） |
| 0023 | 0023 | `bench_route_first_gpu.py`、`fixtures/route_v2_decode_pre0023.py` | sm_80；不 import vllm，按文件路径加载 | kernel 逐位不变（开发阶段） |
| 0024 | 0024 | `bench_idx_glue_gpu.py` | sm_80；需要 `--run-gpu` | GPU 逐位通过（开发阶段） |
| 0026 | 0027 | `test_moe_sum_add_gpu.py`（第 3 项需 `vllm._ampere_marlin_C` 与 `marlin_decode_common.py`） | sm_80 | GPU 逐位 512/512（开发阶段） |
| 0028 | 0026b | `bench_prenorm_bf16x3_gpu.py`（`--mm-file` 指向对方 `ampere_prefill/mhc_prenorm.py`） | sm_80 | T=2312 923 → 154 μs；T ≥ 384 与对方逐位、T < 384 与 0025 逐位（开发阶段） |
| 0027 | 0028 | `bench_shared_reorder_gpu.py`（+ `marlin_decode_common.py`） | sm_80 + `vllm._ampere_marlin_C` | 逐位（开发阶段） |
| 0029 | — | `test_idx_dual_gemm_gpu.py`（k 列逐位、weights 误差 ≤ 1.5× sgemm、op 回退逐位、graph、双流、微基准）；纯 CPU 部分在 `test_thin_gemm_pure.py` | sm_80 | node2 通过：每 MLA 层省 15.7 μs；weights mean 1.07×/1.24×（outlier/cancel），max 更低 |
| —（诊断） | — | `trace_overlap_split.py`：把 profiler trace 中 `_thin_gemm_kernel` / `moe_dec_gemm` / `_post_update_num_computed_tokens_kernel` 每次调用按“单独运行 / 与 NCCL 重叠 / 与其他流重叠”分组统计 | 仅 python3（CPU） | 合成 trace 自测 |
| —（诊断） | — | `bench_shared_contention_gpu.py`（+ `marlin_decode_common.py`）：一个 rank 的 MoE 层 graph，路由前奏 none/gate/tc × 侧流优先级 default/high + serial，输出逐位相同，us/层与重叠比例 | sm_80 + `vllm._ampere_marlin_C` | node2 M=8：none 363.8、gate 385.5、tc 375.9、tc+high 381.3 μs/层（据此 compose 改用 tc；0031 未纳入） |

0007（融合 grouped conv）没有单独测试；它由整机 A/B 覆盖。0022（profile 占位收紧）与 0025（fp32 prenorm）
没有单元测试：0022 由 KV 池大小与 243k/8×114k 压力测试覆盖，0025 由相同历史首块与对方逐位比较覆盖
（该诊断 overlay 不在仓库内）。

“未确认”表示本仓库整理时没有对应的 GPU 运行记录，不代表失败。

## CPU（任意机器，无 torch）

```bash
bash models/glm-5.3-flash/tests/check-dflash2-series.sh            # 自动拉取固定源码
bash models/glm-5.3-flash/tests/check-dflash2-series.sh /path/to/clean-checkout
```

## 在镜像中运行（GPU）

镜像里 vllm 安装在 `/usr/local/lib/python3.12/dist-packages`，这正是测试中 `GLM_DFLASH2_TREE` 的默认值，因此不需要覆盖源码。把本目录只读挂载进去即可：

```bash
cd /root/app/vllm-cmp170hx
podman run --rm -it --device nvidia.com/gpu=5 --security-opt label=disable \
  -v "$PWD/models/glm-5.3-flash/tests/dflash2":/tests:ro \
  -v /root/app/models:/root/app/models:ro \
  -w /tests --entrypoint bash localhost/vllm-backport:glm-5.3-flash-w4a16-dflash2 -c '
    python3 -m pip install -q pytest 2>/dev/null || true
    export PYTHONDONTWRITEBYTECODE=1
    python3 -m pytest -q -s -p no:cacheprovider \
      test_kpool_tail_ring.py test_kpool_rollback_matrix_gpu.py \
      test_marlin_decode_gpu.py test_aux_hidden_states_pp.py
    VLLM_GLM5_PP_MARLIN_PREFILL=1 python3 -m pytest -q -s -p no:cacheprovider test_pp_marlin_prefill_gpu.py
    python3 -m pytest -q -s -p no:cacheprovider test_pp_kda_prefill_gpu.py test_pp_sparse_prefill_gpu.py
    python3 test_route_v2_gpu.py --quick
  '
```

- `test_marlin_decode_gpu.py` 在镜像中不再需要 `VLLM_GLM5_MARLIN_DECODE_LIB`，它会导入包内的 `vllm._ampere_marlin_C`。
- `test_pp_fold_draft_fc_gpu.py --draft /root/app/models/GLM-5.3-Flash-DFlash2` 读取真实 drafter 的 `fc.weight`。
- `test_adaptive_k_container.py` 需要 vLLM 源码树的 `tests/` 目录（镜像中没有），请在应用过补丁的源码树中运行：`cd <source> && python -m pytest -q <repo>/models/glm-5.3-flash/tests/dflash2/test_adaptive_k_container.py`。
- 容器是 `--rm` 的一次性环境，`pip install pytest` 不影响镜像。

### 0017–0028（新镜像内）

```bash
cd /root/app/vllm-cmp170hx
podman run --rm -it --device nvidia.com/gpu=5 --security-opt label=disable \
  -v "$PWD/models/glm-5.3-flash/tests/dflash2":/tests:ro \
  -v /root/app/models:/root/app/models:ro \
  -w /tests --entrypoint bash localhost/vllm-backport:glm-5.3-flash-w4a16-dflash2 -c '
    python3 -m pip install -q pytest 2>/dev/null || true
    export PYTHONDONTWRITEBYTECODE=1
    set -e
    GATE=1 BENCH=0 python3 test_mhc_v2_full_gpu.py                       # 0017
    python3 -m pytest -q -s -p no:cacheprovider test_host_overhead_gpu.py # 0018-0020
    VLLM_GLM5_THIN_GEMM=1 python3 test_pp_draft_tail_gpu.py \
        --draft /root/app/models/GLM-5.3-Flash-DFlash2                    # 0021
    VLLM_GLM5_THIN_GEMM=0 python3 test_pp_draft_tail_gpu.py
    python3 -m pytest -q -s -p no:cacheprovider test_pp_tail_topk_audit_gpu.py
    python3 bench_route_first_gpu.py --ms 1,4,8                           # 0023
    python3 bench_idx_glue_gpu.py --run-gpu                               # 0024
    python3 test_moe_sum_add_gpu.py                                       # 0026
    python3 bench_shared_reorder_gpu.py --tokens 4,8                      # 0027
    # 0028：需要对方源码，另挂载 -v <MM>:/mm:ro
    # python3 bench_prenorm_bf16x3_gpu.py --mm-file /mm/vllm/ampere_prefill/mhc_prenorm.py \
    #     --mm-tl-file /mm/vllm/model_executor/kernels/mhc/tilelang_kernels.py
  '
```

### 0029（新镜像内）

```bash
cd /root/app/vllm-cmp170hx
podman run --rm -it --device nvidia.com/gpu=5 --security-opt label=disable \
  -v "$PWD/models/glm-5.3-flash/tests/dflash2":/tests:ro \
  -w /tests --entrypoint bash localhost/vllm-backport:glm-5.3-flash-w4a16-dflash2 -c '
    export PYTHONDONTWRITEBYTECODE=1
    set -e
    python3 test_idx_dual_gemm_gpu.py --json /tmp/0029.jsonl                    # 0029
  '
```

- 对已有 profile（两边 `.pt.trace.json[.gz]`）：
  `python3 trace_overlap_split.py ours/rank0.json.gz mm/rank0.json.gz`，看 `alone` 一行两边是否相同。
- shared experts 与 Marlin 争用诊断（同一镜像内，不需要任何开关）：
  `VLLM_GLM5_THIN_GEMM=1 python3 bench_shared_contention_gpu.py --tokens 8 --trace-dir /tmp/sc`，
  然后 `python3 trace_overlap_split.py /tmp/sc/M8_gate_default.json /tmp/sc/M8_tc_default.json`。
- 0029 已在 compose 中打开（`VLLM_GLM5_IDX_DUAL_GEMM: "1"`）。
- 已评估未纳入：0030（`VLLM_GLM5_THIN_GEMM_MM_SELECT`，整机 c8 729.2，无收益）、
  0031（`VLLM_GLM5_SHARED_STREAM_PRIORITY`，bench 中 tc+high 381.3 vs tc+default 375.9 μs/层，更慢）。

- 所有脚本默认从镜像 site-packages 读取被测文件；要测另一份源码树，设 `GLM_DFLASH2_TREE=<树根>`
  并把它放进 `PYTHONPATH`。
- `test_mhc_v2_full_gpu.py`：`MM_V2=<Morrowmake>/vllm/ampere_decode/mhc_decode_v2.py` 可加入对方实现对比；
  `REAL_CKPT=/root/app/models/GLM-5.3-Flash-W4A16-MTP` 使用真实权重。
- `bench_route_first_gpu.py`：`--orig` 默认是 0023 之前的 route v2（`fixtures/`），`--new` 默认是镜像内文件，
  `--mm` 可选。
- `test_pp_draft_tail_gpu.py --no-private` 是负对照（thin GEMM 开启时允许失败）。
