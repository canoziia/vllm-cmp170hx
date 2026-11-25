# SPDX-License-Identifier: Apache-2.0
"""Fixed measured CPU policy adapter. No external source loading, capability or trace IO.

Per-request state is generation-pinned. Only valid actual-width feedback can
advance selection; scheduled widths remain retained when a fresh choice is
ineligible. Feedback recurrence is CPU-visible cadence, not GPU kernel time.
"""

import math
import os
import time
from types import SimpleNamespace
from .dsv41_measured_policy import OnlineCostModel, Policy


class FixedHistoryPolicy:
    def __init__(self, logger, decay=0.95):
        if isinstance(decay, bool) or not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("invalid history decay")
        self.logger = logger
        self.decay = decay
        self.active = {}
        self.generation = 0
        self.feedback_time = None
        self.warned = False
        self._step_seq = 0
        self._last_batch_fb = None
        self._next_cost_log = 100
        online_enabled = (
            os.environ.get("VLLM_DSV41_ONLINE_COST_MODEL", "1").strip().lower()
            not in ("0", "false", "off", "no")
        )
        self.cost_model = OnlineCostModel() if online_enabled else None
        self.logger.info(
            "DSV41 fixed measured CPU policy enabled: decay=%s; online_cost_model=%s; no hot loader",
            decay,
            online_enabled,
        )

    def make_entry(self, request):
        return self.new_entry(Policy({"decay": self.decay}))

    def new_entry(self, plugin):
        self.generation += 1
        return SimpleNamespace(
            plugin=plugin,
            k=5,
            generation=self.generation,
            previous=None,
            fresh=False,
            seq=0,
            failed=False,
        )

    def admit(self, request):
        if getattr(request, "verification_draft_limit", None) is None:
            self.active[request.request_id] = self.make_entry(request)

    def is_disabled(self, entry):
        return entry.failed

    def trace(self, event):
        pass

    def fail(self, rid, entry, exc):
        # Fail closed to fixed full5, not an unvalidated width. Manual never
        # enters this adapter. No retry until next request admission.
        entry.failed = True
        entry.k = 5
        entry.fresh = False
        if not self.warned:
            self.logger.warning(
                "DSV41 fixed policy failure (%s); full5 retained", type(exc).__name__
            )
            self.warned = True

    def retained_limits(self, requests):
        result = {
            r.request_id: self.active[r.request_id].k
            for r in requests
            if r.request_id in self.active
            and getattr(r, "verification_draft_limit", None) is None
        }
        if (
            len(result) >= 2
            and os.environ.get("VLLM_DSV41_COHORT_UNIFORM_K", "0") == "1"
        ):
            sorted_ks = sorted(result.values())
            cohort_k = sorted_ks[(len(sorted_ks) - 1) // 2]
            for rid in list(result):
                result[rid] = cohort_k
        return result

    def choose(self, requests, concurrency, residents=None):
        result = {}
        newly_trimmed = []
        for r in requests:
            entry = self.active.get(r.request_id)
            if (
                entry is None
                or getattr(r, "verification_draft_limit", None) is not None
            ):
                continue
            if not self.is_disabled(entry) and entry.fresh:
                was_trimmed = bool(getattr(entry.plugin, "trimmed", False))
                try:
                    k = entry.plugin.choose(
                        dict(
                            current_k=entry.k,
                            concurrency=concurrency,
                            cohort_size=len(requests),
                            cost_model=self.cost_model,
                        )
                    )
                    if type(k) is not int or not 1 <= k <= 5 or k > entry.k + 1:
                        raise ValueError("policy width outside allowed transition")
                    self.trace(
                        dict(
                            event="choose",
                            request=r.request_id,
                            seq=entry.seq,
                            old=entry.k,
                            next=k,
                            concurrency=concurrency,
                            diagnostics=getattr(entry.plugin, "diagnostics", {}),
                        )
                    )
                    entry.k = k
                    entry.fresh = False
                    if not was_trimmed and bool(getattr(entry.plugin, "trimmed", False)):
                        newly_trimmed.append(entry)
                except Exception as exc:
                    self.fail(r.request_id, entry, exc)
            result[r.request_id] = entry.k
        if (
            len(result) >= 2
            and os.environ.get("VLLM_DSV41_COHORT_UNIFORM_K", "0") == "1"
        ):
            sorted_ks = sorted(result.values())
            cohort_k = sorted_ks[(len(sorted_ks) - 1) // 2]
            if cohort_k == 5:
                for entry in newly_trimmed:
                    entry.plugin.trimmed = False
            for r in requests:
                rid = r.request_id
                if rid in result:
                    entry = self.active.get(rid)
                    if entry is not None and not self.is_disabled(entry):
                        entry.k = cohort_k
                    result[rid] = cohort_k
        return result

    def scheduled(self, output, residents, concurrency):
        now = time.perf_counter()
        self._step_seq += 1
        prefilling = any(r.is_prefill_chunk for r in residents)
        if self.cost_model is not None:
            num_sched = getattr(output, "num_scheduled_tokens", None)
            if num_sched:
                batch_b = len(num_sched)
                batch_m = sum(num_sched.values())
                is_pure_decode = (
                    not prefilling
                    and batch_b >= 1
                    and batch_m >= 2
                    and batch_m <= batch_b * 6
                    and len(output.scheduled_spec_decode_tokens) == batch_b
                )
                output._dsv41_batch_stamp = (
                    self._step_seq,
                    now,
                    is_pure_decode,
                    batch_b,
                    batch_m,
                )
        if not self.active:
            return
        key = tuple(sorted(r.request_id for r in residents))
        stamps = {}
        for rid, drafts in output.scheduled_spec_decode_tokens.items():
            entry = self.active.get(rid)
            if entry is not None and not self.is_disabled(entry):
                entry.seq += 1
                stamps[rid] = (
                    entry.generation,
                    now,
                    key,
                    prefilling,
                    concurrency,
                    len(drafts),
                    entry.seq,
                )
        output._dsv41_policy_stamps = stamps

    def begin_feedback(self):
        self.feedback_time = time.perf_counter()

    def _observe_batch_cost(self, output):
        if self.cost_model is None or getattr(output, "_dsv41_batch_cost_done", False):
            return
        output._dsv41_batch_cost_done = True
        stamp = getattr(output, "_dsv41_batch_stamp", None)
        if stamp is None:
            return
        step_seq, sent, is_pure_decode, batch_b, batch_m = stamp
        now = (
            self.feedback_time
            if self.feedback_time is not None
            else time.perf_counter()
        )
        prev = self._last_batch_fb
        self._last_batch_fb = (step_seq, now, is_pure_decode)
        if (
            not is_pure_decode
            or prev is None
            or not prev[2]
            or step_seq != prev[0] + 1
            or sent - prev[1] > 0.02
        ):
            return
        dt = now - prev[1]
        if not 0.0005 < dt < 0.5:
            return
        try:
            if self.cost_model.observe(batch_b, batch_m, dt):
                if self.cost_model.obs_count >= self._next_cost_log:
                    self.logger.info(
                        "DSV41 online cost model updated: %s",
                        self.cost_model.diagnostics(),
                    )
                    self._next_cost_log = (
                        500
                        if self._next_cost_log < 500
                        else self._next_cost_log + 2000
                    )
        except Exception:
            pass

    def feedback(self, output, request, drafted, accepted, output_tokens, valid):
        self._observe_batch_cost(output)
        rid = request.request_id
        entry = self.active.get(rid)
        stamp = getattr(output, "_dsv41_policy_stamps", {}).get(rid)
        if (
            entry is None
            or stamp is None
            or stamp[0] != entry.generation
            or self.is_disabled(entry)
        ):
            return
        _, sent, key, prefilling, concurrency, actual_k, seq = stamp
        now = (
            self.feedback_time
            if self.feedback_time is not None
            else time.perf_counter()
        )
        latency = now - sent
        previous = entry.previous
        entry.previous = (now, key, prefilling, actual_k)
        seconds = latency
        timing_valid = bool(
            valid
            and not prefilling
            and previous is not None
            and previous[1] == key
            and not previous[2]
            and previous[3] == actual_k
        )
        if timing_valid:
            seconds = now - previous[0]
        if not valid:
            entry.previous = None
        if not 0 < seconds < 60 or actual_k != drafted:
            timing_valid = False
        event = dict(
            request=rid,
            seq=seq,
            k=drafted,
            accepted=accepted,
            output_tokens=output_tokens,
            seconds=seconds,
            latency_s=latency,
            timing_kind="feedback_recurrence"
            if timing_valid
            else "dispatch_to_feedback",
            timing_valid=timing_valid,
            valid=valid,
            concurrency=concurrency,
            load_epoch=hash(key),
        )
        self.trace(dict(event="feedback", **event))
        if not valid:
            return
        try:
            entry.plugin.observe(event)
            entry.fresh = True
        except Exception as exc:
            self.fail(rid, entry, exc)

    def forget(self, request_id):
        self.active.pop(request_id, None)

    def observe(self, request_id, drafted, accepted):
        # Compatibility with old scheduler call; actual observations are
        # exclusively processed by feedback() with attributed timing.
        pass
