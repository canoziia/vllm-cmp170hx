# SPDX-License-Identifier: Apache-2.0
"""Patch 0004 (adaptive DFlash depth): tests that need torch + an installed
vLLM with patches 0001-0004 overlaid + the vLLM source ``tests/`` package
(for ``tests.v1.core.utils``). Run inside the production container from the
vLLM source root:

    cd <vllm-src> && python -m pytest -q <repo>/models/glm-5.3-flash/tests/dflash2/test_adaptive_k_container.py

CPU only (no GPU kernels). Adapted from Morrowmake
tests/v1/spec_decode/test_glm5_dflash_adaptive_k.py / _accept_depth.py.
NOT RUN YET (no torch here); expect small signature fixes on first run
(``create_scheduler`` keyword names, CudaGraphManager config stand-ins).
"""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import CompilationConfig, CUDAGraphMode, ParallelConfig
from vllm.config import SchedulerConfig, VllmConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.dynamic.adaptive_k import AdaptiveKConfig
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils

BY_LOAD = {"by_load": [7, 5, 3]}
K7_ACC = {"by_load": [7, 5, 3],
          "accept": {"costs": [1.0, 1.077, 1.154, 1.231, 1.308]}}


def _make_scheduler(adaptive_k, k=7, cls=Scheduler):
    base = create_scheduler(
        max_num_seqs=16, max_num_batched_tokens=8192, num_speculative_tokens=k
    )
    spec = base.vllm_config.speculative_config
    assert spec is not None
    if adaptive_k is not None:
        spec.adaptive_k = adaptive_k
        spec.adaptive_k_config = AdaptiveKConfig.from_dict(adaptive_k, k)
    return cls(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=base.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )


def _step_with_drafts(scheduler, requests, width=7):
    for r in requests:
        r.spec_token_ids = list(range(11, 11 + width))
    return scheduler.schedule()


@pytest.mark.parametrize("n,k", [(1, 7), (2, 5), (3, 3), (6, 3)])
def test_scheduler_verifies_by_load_and_drafts_full_block(n, k):
    s = _make_scheduler(BY_LOAD)
    reqs = create_requests(num_requests=n)
    for r in reqs:
        s.add_request(r)
    s.schedule()  # prefill
    out = _step_with_drafts(s, reqs)
    assert s.cur_num_spec_tokens == k
    assert out.num_spec_tokens_to_schedule == 7  # drafter keeps its full block
    for r in reqs:
        assert out.scheduled_spec_decode_tokens[r.request_id] == list(range(11, 11 + k))
        assert out.num_scheduled_tokens[r.request_id] == 1 + k


def test_async_scheduler_placeholders_full_width():
    s = _make_scheduler(BY_LOAD, cls=AsyncScheduler)
    reqs = create_requests(num_requests=2)
    for r in reqs:
        s.add_request(r)
    out = s.schedule()
    assert out.num_spec_tokens_to_schedule == 7
    out = s.schedule()
    if out.scheduled_spec_decode_tokens:  # V1-runner cadence (no PP)
        assert s.cur_num_spec_tokens == 5
        for r in reqs:
            assert len(out.scheduled_spec_decode_tokens[r.request_id]) == 5


def test_off_path_is_fixed_depth():
    s = _make_scheduler(None, k=3)
    assert s.adaptive_k is None
    [r] = create_requests(num_requests=1)
    s.add_request(r)
    s.schedule()
    out = _step_with_drafts(s, [r], width=3)
    assert s.cur_num_spec_tokens == 3
    assert out.num_spec_tokens_to_schedule == 3
    assert out.scheduled_spec_decode_tokens[r.request_id] == [11, 12, 13]


def test_acceptance_reaches_scheduler_and_rejections_roll_back():
    s = _make_scheduler(K7_ACC)
    [r] = create_requests(num_requests=1)
    s.add_request(r)
    s.schedule()
    # Low acceptance history -> depth 3 for this request.
    rng = random.Random(5)
    for _ in range(60):
        acc = 0
        while acc < 7 and rng.random() < 0.4:
            acc += 1
        s.adaptive_k.observe(r.request_id, 7, acc)
    _step_with_drafts(s, [r])
    assert s.cur_num_spec_tokens == 3


def test_finished_request_is_forgotten():
    s = _make_scheduler(K7_ACC)
    [r] = create_requests(num_requests=1)
    s.add_request(r)
    s.schedule()
    s.adaptive_k.observe(r.request_id, 7, 7)
    assert r.request_id in s.adaptive_k._acc_s
    s.finish_requests(r.request_id, RequestStatus.FINISHED_ABORTED)
    assert r.request_id not in s.adaptive_k._acc_s


# ----------------------------------------------------------- CUDA-graph shapes


def _graph_manager(monkeypatch, raw, k, max_num_seqs, capture_sizes, flag=True):
    monkeypatch.setattr(
        gpu_cudagraph_utils, "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    cc = CompilationConfig(cudagraph_mode="FULL_AND_PIECEWISE",
                           cudagraph_capture_sizes=capture_sizes)
    cc.max_cudagraph_capture_size = max(capture_sizes)
    cc.post_init_cudagraph_sizes()
    vc = MagicMock(spec=VllmConfig)
    vc.compilation_config = cc
    vc.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=max_num_seqs)
    vc.parallel_config = ParallelConfig()
    vc.cache_config = SimpleNamespace(use_kda_recoverssm=False)
    vc.num_speculative_tokens = k
    cfg = AdaptiveKConfig.from_dict(raw, k)
    spec = MagicMock()
    spec.uses_dynamic_speculative_decoding.return_value = False
    spec.num_speculative_tokens_per_batch_size = None
    spec.uses_adaptive_k.return_value = True
    spec.adaptive_k_draft_counts.return_value = cfg.allowed
    spec.adaptive_k_config = cfg
    vc.speculative_config = spec
    return gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vc, device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=k + 1, adaptive_k_capture=flag,
    )


def _full_decode_shapes(m):
    return sorted(
        (d.uniform_token_count, d.num_reqs)
        for d in m._capture_descs[CUDAGraphMode.FULL]
        if d.uniform_token_count
    )


SIZES = [1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 40, 48, 56, 64, 128]


def test_each_depth_captured_only_where_it_runs(monkeypatch):
    m = _graph_manager(monkeypatch, BY_LOAD, 7, 8, SIZES)
    shapes = _full_decode_shapes(m)
    assert [s for s in shapes if s[0] == 8] == [(8, 1)]       # k=7: 1 request
    assert {n for q, n in shapes if q == 6} <= {1, 2}         # k=5: <= 2
    assert max(n for q, n in shapes if q == 4) == 8           # k=3: all
    m._graphs_captured = True
    for q, n in [(8, 1), (6, 1), (6, 2)] + [(4, i) for i in range(1, 9)]:
        d = m.dispatch(num_reqs=n, num_tokens=n * q, uniform_token_count=q,
                       num_active_loras=0)
        assert d.cg_mode == CUDAGraphMode.FULL and d.uniform_token_count == q


def test_flag_off_manager_unchanged(monkeypatch):
    """The drafter's managers (adaptive_k_capture=False) keep one width."""
    m = _graph_manager(monkeypatch, BY_LOAD, 7, 8, SIZES, flag=False)
    assert {q for q, _ in _full_decode_shapes(m)} == {8}
