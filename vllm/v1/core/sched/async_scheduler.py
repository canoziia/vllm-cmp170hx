# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler


class AsyncScheduler(Scheduler):
    use_v2_model_runner: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_v2_model_runner = True
        # Only XPU can disable PP decode microbatching via VLLM_XPU_PP_MICROBATCH.
        self.decode_stagger = self.parallel_config.pipeline_parallel_size
        if current_platform.is_xpu() and not envs.VLLM_XPU_PP_MICROBATCH:
            self.decode_stagger = 1

    @property
    def pp_size(self) -> int:
        return getattr(self, "decode_stagger", 1)

    @pp_size.setter
    def pp_size(self, value: int) -> None:
        self.decode_stagger = value

    def _get_max_num_scheduled_decodes(self) -> int:
        if (
            not envs.VLLM_PP_DECODE_COHORT_BALANCE
            or not self.use_v2_model_runner
            or not self.use_pp
        ):
            return self.max_num_running_reqs
        if os.environ.get("VLLM_PP_DECODE_COHORT_LIVE_LOAD", "0") == "1":
            # The capacity-based limit is inert at low load: four requests
            # completing prefill together remain locked in a four-request
            # cadence forever. Bound established decodes by the LIVE load.
            # Do not change the pp_size feedback-ring cadence or cap prefill.
            num_running, num_waiting = self.get_request_counts()
            live = min(self.max_num_running_reqs, num_running + num_waiting)
            return max(1, (live + self.pp_size - 1) // self.pp_size)
        # ceil(max_num_seqs / pp_size)
        return (self.max_num_running_reqs + self.pp_size - 1) // self.pp_size

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        spec_token_placeholders = [-1] * scheduler_output.num_spec_tokens_to_schedule
        pending_structured_output_tokens = (
            scheduler_output.pending_structured_output_tokens
        )
        structured_output_spec_decode = getattr(
            scheduler_output, "has_structured_output_spec_decode", False
        )
        has_structured_output_requests = getattr(
            scheduler_output, "has_structured_output_requests", True
        )
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            # The scheduled draft ids may still be worker-side -1 placeholders
            # even when no previous output tokens are outstanding. In that case
            # the grammar must wait for take_draft_token_ids(): otherwise the
            # bitmask is built from -1, and fail-closed verification rejects the
            # entire draft window on every subsequent step.
            scheduled_drafts = spec_decode_tokens.get(req_id, ())
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(scheduled_drafts)
            if has_structured_output_requests and request.use_structured_output:
                pending_structured_output_tokens |= (
                    request.num_output_placeholders > 0
                    or -1 in scheduled_drafts
                )
                structured_output_spec_decode |= cur_num_spec_tokens > 0
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            if hasattr(request, "num_output_placeholder_steps"):
                request.num_output_placeholder_steps += 1
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            limit = getattr(request, "verification_draft_limit", None)
            request.spec_token_ids = (
                spec_token_placeholders
                if limit is None
                else spec_token_placeholders[:limit]
            )

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                earliest = self.current_step + self.pp_size
                if (
                    getattr(self, "use_pp", False)
                    and os.environ.get("VLLM_PP_DECODE_COHORT_LIVE_LOAD", "0") == "1"
                ):
                    phase = getattr(request, "decode_cohort_phase", None)
                    loads = [0] * self.pp_size
                    for live in self.running:
                        assigned = getattr(live, "decode_cohort_phase", None)
                        if assigned is not None:
                            loads[assigned] += 1
                    capacity = self._get_max_num_scheduled_decodes()
                    fragmented = (
                        sum(n > 0 for n in loads)
                        > (sum(loads) + capacity - 1) // capacity
                    )
                    if phase is None or (fragmented and loads[phase] < capacity):
                        if phase is not None:
                            loads[phase] -= 1

                        # Fill an existing non-full cohort before opening an
                        # empty phase. Repack partial phases after load growth
                        # too; delay the NEXT decode, never consume a feedback
                        # result before its original T+PP boundary.
                        def phase_cost(delay):
                            load = loads[(earliest + delay) % self.pp_size]
                            return (
                                load >= capacity,
                                -load if load < capacity else load,
                                delay,
                            )

                        earliest += min(range(self.pp_size), key=phase_cost)
                    request.decode_cohort_phase = earliest % self.pp_size
                request.next_decode_eligible_step = earliest

        scheduler_output.pending_structured_output_tokens = (
            pending_structured_output_tokens
        )
        if hasattr(scheduler_output, "has_structured_output_spec_decode"):
            scheduler_output.has_structured_output_spec_decode = (
                structured_output_spec_decode
            )
