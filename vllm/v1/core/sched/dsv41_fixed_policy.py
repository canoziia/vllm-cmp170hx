# SPDX-License-Identifier: Apache-2.0
"""Fixed measured CPU policy adapter. No external source loading, capability or trace IO.

Per-request state is generation-pinned. Only valid actual-width feedback can
advance selection; scheduled widths remain retained when a fresh choice is
ineligible. Feedback recurrence is CPU-visible cadence, not GPU kernel time.
"""

import math
import time
from types import SimpleNamespace
from .dsv41_measured_policy import Policy


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
        self.logger.info(
            "DSV41 fixed measured CPU policy enabled: decay=%s; no hot loader", decay
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
        return {
            r.request_id: self.active[r.request_id].k
            for r in requests
            if r.request_id in self.active
            and getattr(r, "verification_draft_limit", None) is None
        }

    def choose(self, requests, concurrency, residents=None):
        result = {}
        for r in requests:
            entry = self.active.get(r.request_id)
            if (
                entry is None
                or getattr(r, "verification_draft_limit", None) is not None
            ):
                continue
            if not self.is_disabled(entry) and entry.fresh:
                try:
                    k = entry.plugin.choose(
                        dict(
                            current_k=entry.k,
                            concurrency=concurrency,
                            cohort_size=len(requests),
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
                except Exception as exc:
                    self.fail(r.request_id, entry, exc)
            result[r.request_id] = entry.k
        return result

    def scheduled(self, output, residents, concurrency):
        if not self.active:
            return
        now = time.perf_counter()
        key = tuple(sorted(r.request_id for r in residents))
        prefilling = any(r.is_prefill_chunk for r in residents)
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

    def feedback(self, output, request, drafted, accepted, output_tokens, valid):
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
