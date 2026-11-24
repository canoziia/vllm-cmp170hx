# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side structured draft backfill with async pipeline scheduling."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("structured", "prior_placeholders", "drafts", "expected_pending"),
    [
        (True, 0, [-1, -1, -1], True),
        (True, 0, [12, 13, 14], False),
        (True, 2, [12, 13, 14], True),
        (False, 0, [-1, -1, -1], False),
        (True, 0, [], False),
    ],
)
def test_async_structured_draft_handoff_without_prior_output(
    structured: bool,
    prior_placeholders: int,
    drafts: list[int],
    expected_pending: bool,
):
    """A worker-side -1 draft needs backfill even after all outputs settled."""
    scheduler = object.__new__(AsyncScheduler)
    request = SimpleNamespace(
        is_prefill_chunk=False,
        use_structured_output=structured,
        num_output_placeholders=prior_placeholders,
        spec_token_ids=[],
    )
    scheduler.requests = {"req": request}
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.use_v2_model_runner = False
    output = SimpleNamespace(
        num_scheduled_tokens={"req": 1 + len(drafts)},
        scheduled_spec_decode_tokens={"req": drafts},
        num_spec_tokens_to_schedule=3,
        pending_structured_output_tokens=False,
    )
    with patch.object(Scheduler, "_update_after_schedule"):
        scheduler._update_after_schedule(output)
    assert output.pending_structured_output_tokens is expected_pending
    assert request.num_output_placeholders == prior_placeholders + 1 + len(drafts)


def test_request_keyed_drafts_survive_interleaved_batches():
    """Read B's persistent draft slot even if A replaced the last-batch snapshot."""
    runner = object.__new__(GPUModelRunner)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"A": 1, "B": 0},
        draft_tokens=torch.tensor([[31, 32, 33], [11, 12, 13]]),
    )
    runner.draft_tokens_handler = Mock()
    runner.draft_tokens_handler.get_draft_tokens.return_value = SimpleNamespace(
        req_ids=["A"], draft_token_ids=[[11, 12, 13]]
    )
    result = runner.take_draft_token_ids(["B", "gone"])
    assert result.req_ids == ["B"]
    assert result.draft_token_ids == [[31, 32, 33]]
    runner.draft_tokens_handler.get_draft_tokens.assert_not_called()
    empty = runner.take_draft_token_ids(["gone"])
    assert empty.req_ids == [] and empty.draft_token_ids == []


def test_async_structured_backfill_with_empty_batch_queue():
    """Backfill before sampling when the previous result was already delivered."""
    core = object.__new__(EngineCore)
    core.batch_queue = deque(maxlen=2)
    core.batch_queue_size = 2
    core.is_ec_consumer = True
    core.is_pooling_model = False
    core.check_for_draft_tokens = True
    events: list[str] = []
    output = SimpleNamespace(
        num_scheduled_tokens={"req": 4},
        scheduled_spec_decode_tokens={"req": [-1, -1, -1]},
        total_num_scheduled_tokens=4,
        pending_structured_output_tokens=True,
    )
    core.scheduler = Mock()
    core.scheduler.has_requests.return_value = True
    core.scheduler.schedule.return_value = output

    def update_drafts(draft_ids, scheduled):
        events.append("backfill")
        assert draft_ids == [42, 43, 44]
        scheduled.scheduled_spec_decode_tokens["req"] = draft_ids

    core.scheduler.update_draft_token_ids_in_output.side_effect = update_drafts

    def grammar(scheduled):
        events.append("grammar")
        assert scheduled.scheduled_spec_decode_tokens["req"] == [42, 43, 44]
        return "constrained grammar"

    core.scheduler.get_grammar_bitmask.side_effect = grammar
    core.model_executor = Mock()
    exec_future = object()
    sample_future = object()
    core.model_executor.execute_model.return_value = exec_future

    def take_drafts(req_ids):
        events.append("take_drafts")
        assert req_ids == ["req"]
        return [42, 43, 44]

    core.model_executor.take_draft_token_ids.side_effect = take_drafts
    core.model_executor.sample_tokens.side_effect = lambda *args, **kwargs: (
        events.append("sample") or sample_future
    )
    core.log_error_detail = lambda scheduled: nullcontext()
    core._should_throttle_prefills = lambda: False

    assert core.step_with_batch_queue() == (None, True)
    assert events == ["take_drafts", "backfill", "grammar", "sample"]
    assert list(core.batch_queue) == [(sample_future, output, exec_future)]
