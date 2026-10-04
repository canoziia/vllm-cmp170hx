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

0007（融合 grouped conv）没有单独测试；它由整机 A/B 覆盖。

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
