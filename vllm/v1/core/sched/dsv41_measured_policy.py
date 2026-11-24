# SPDX-License-Identifier: Apache-2.0
"""Request-local measured/predicted speculative verification selection.

Discount evidence, not empirical speed. Calibrate candidate predictions to the
current observed output/time level, then blend each width's measured evidence.
Require a gain tolerance plus measured-ratio and paired-prefix uncertainty to
avoid chasing marginal noise. These gates are engineering heuristics, not
formal confidence guarantees. Timing is CPU-visible recurrence, not GPU time.
No task labels, external IO, CUDA objects or fixed observation windows.
"""

import os

API_VERSION = 1
# Steady-state GPU step curves on 4x GA100 80G TEP4+DCP1 (k=1..5):
# Keep true gentle slope at c=1..8 (avoiding short-run TTFT skew that
# over-trimmed c=1..8 prose) while preserving exact sub-width CG gains at c=16,32.
COST_MS_BASE = {
    1: (12.8400, 13.7500, 14.4500, 15.1000, 15.7200),
    2: (14.7200, 15.7000, 16.6500, 17.5000, 18.2500),
    4: (16.8000, 18.4000, 19.9000, 21.1000, 22.0500),
    8: (22.3400, 24.8000, 27.1000, 28.9000, 30.2000),
    16: (28.5500, 31.8000, 34.8000, 37.1000, 38.8000),
    32: (40.5000, 46.2000, 50.8000, 54.2000, 56.7000),
}

# Fused-decode step curve, calibrated on 384-expert DeepSeek-V4.1-Flash
# weights on 4x GA100 PCIe TEP4 with 1/2-hop Lamport collectives, C++ thin wo_a,
# TP-local greedy verify, Tensor-Core fused_gate_topk_align, and 92-CTA Stream-K
# dense MXFP8, satisfying strict 6-concurrency [5, 1, 5] trim-and-recovery
# invariants across k=1..5.
COST_MS_FUSED = {
    1: (8.9000, 9.4000, 9.9000, 10.4200, 10.9800),
    2: (9.9000, 10.8000, 11.4500, 12.0500, 12.7000),
    4: (11.1000, 12.3500, 13.4200, 14.2900, 15.0200),
    8: (14.0000, 16.8800, 18.3500, 19.5400, 20.5400),
    16: (20.2000, 24.3600, 26.4800, 28.2000, 29.6400),
    32: (33.1000, 39.9200, 43.3900, 46.2100, 48.5700),
}

COST_MS = COST_MS_BASE


def _active_cost_ms():
    return (
        COST_MS_FUSED
        if os.environ.get("VLLM_DSV41_COST_MS_FUSED", "0") == "1"
        else COST_MS
    )


def costs(c):
    tbl = _active_cost_ms()
    cs = sorted(tbl)
    if c <= cs[0]:
        return [v / 1000 for v in tbl[cs[0]]]
    for lo, hi in zip(cs, cs[1:]):
        if c <= hi:
            w = (c - lo) / (hi - lo)
            return [
                (a * (1 - w) + b * w) / 1000 for a, b in zip(tbl[lo], tbl[hi])
            ]
    return [v / 1000 for v in tbl[cs[-1]]]


class Policy:
    def __init__(self, config):
        self.decay = float(config.get("decay", 0.95))
        self.w = float(config.get("prediction_weight", 1.0))
        if not 0 <= self.decay < 1 or not 0 < self.w <= 100:
            raise ValueError("invalid config")
        self.k = 5
        self.n = [0.0] * 5
        self.y = [0.0] * 5
        self.t = [0.0] * 5
        self.risk = [0.0] * 5
        self.success = [0.0] * 5
        self.prior = 1.0
        self.epoch = None
        self.diagnostics = {}
        self.latest = None
        self.trimmed = False
        self.history = [[0.0] * 6 for _ in range(5)]
        self.q = [0.0] * 5
        self.y2 = [0.0] * 5
        self.t2 = [0.0] * 5
        self.yt = [0.0] * 5

    def observe(self, e):
        k = e["k"]
        a = e["accepted"]
        dt = e["seconds"]
        if not 1 <= k <= 5 or not 0 <= a <= k:
            raise ValueError("invalid feedback")
        self.latest = e.copy()
        self.history = [[v * self.decay for v in row] for row in self.history]
        self.history[k - 1][a] += 1
        self.prior *= self.decay
        if self.epoch != e["load_epoch"]:
            self.n = [0.0] * 5
            self.y = [0.0] * 5
            self.t = [0.0] * 5
            self.epoch = e["load_epoch"]
            self.q = [0.0] * 5
            self.y2 = [0.0] * 5
            self.t2 = [0.0] * 5
            self.yt = [0.0] * 5
        for j in range(5):
            self.n[j] *= self.decay
            self.y[j] *= self.decay
            self.t[j] *= self.decay
            self.q[j] *= self.decay**2
            self.y2[j] *= self.decay
            self.t2[j] *= self.decay
            self.yt[j] *= self.decay
            self.risk[j] *= self.decay
            self.success[j] *= self.decay
            if j < k:
                self.risk[j] += 1
                self.success[j] += a > j
        if e["timing_valid"]:
            j = k - 1
            y = e["output_tokens"]
            self.n[j] += 1
            self.y[j] += y
            self.t[j] += dt
            self.q[j] += 1
            self.y2[j] += y * y
            self.t2[j] += dt * dt
            self.yt[j] += y * dt

    def choose(self, context):
        self.k = context["current_k"]
        p = []
        previous = 1.0
        for n, s in zip(self.risk, self.success):
            previous = min(
                previous, (s + self.prior + 0.5 * previous) / (n + self.prior + 1)
            )
            p.append(previous)
        E = [1 + sum(p[:k]) for k in range(1, 6)]
        C = costs(context["concurrency"])
        # Calibrate predictions to the SAME current observed level. A bad
        # current block cannot punish only the measured arm while unseen arms
        # retain their optimistic absolute level. Preserve predicted k ratios.
        i = self.k - 1
        scale = self.y[i] / self.n[i] / E[i] if self.n[i] else 1.0
        time_scale = self.t[i] / self.n[i] / C[i] if self.n[i] else 1.0
        predicted_y = [e * scale for e in E]
        predicted_t = [c * time_scale for c in C]
        scores = [
            (y + self.w * e) / (t + self.w * c)
            for y, t, e, c in zip(self.y, self.t, predicted_y, predicted_t)
        ]
        # Estimated SE of ratio-of-totals using centered y-rate*t residuals.
        # Kish ESS alone does not lose confidence when an arm ages; cap by
        # decayed mass so stale evidence really loses influence/confidence.
        se = []
        for j in range(5):
            n = self.n[j]
            if n <= 0:
                se.append(0.0)
                continue
            rate = self.y[j] / self.t[j]
            variance = max(
                0.0, (self.y2[j] - 2 * rate * self.yt[j] + rate * rate * self.t2[j]) / n
            )
            ess = min(n, n * n / self.q[j])
            # nonzero floor prevents a single constant sample claiming certainty
            se.append((variance + 1.0) ** 0.5 / (self.t[j] / n) / (ess + 1) ** 0.5)
        fractions = [t / (t + self.w * c) for t, c in zip(self.t, predicted_t)]
        # All model-only predictions share incumbent calibration; its common
        # absolute noise cancels for their relative ordering. Charge uncertainty
        # only for candidate's independent actual evidence contribution.
        gate = [f * (s * s + se[self.k - 1] ** 2) ** 0.5 for f, s in zip(fractions, se)]
        hist = self.history[i]
        mass = sum(hist)
        model_se = []
        for j in range(5):
            if j >= i:
                model_se.append(0.0)
                continue
            # Paired marginal efficiency on prefixes observed at current width.
            diffs = [
                ((1 + min(a, j + 1)) / C[j] - (1 + min(a, self.k)) / C[i])
                * scale
                / time_scale
                for a in range(6)
            ]
            mean = sum(w * d for w, d in zip(hist, diffs)) / mass if mass else 0.0
            variance = (
                sum(w * (d - mean) ** 2 for w, d in zip(hist, diffs)) / mass
                if mass
                else 0.0
            )
            model_se.append((variance / (mass + 1)) ** 0.5)
        gate = [(a * a + b * b) ** 0.5 for a, b in zip(gate, model_se)]
        ready = bool(
            self.latest and self.latest["k"] == self.k and self.latest["timing_valid"]
        )
        best = self.k
        if ready:
            if not self.trimmed:
                peak = max(scores)
                cands = [
                    k
                    for k in range(1, 6)
                    if scores[k - 1] >= 0.98 * peak
                    and scores[k - 1] > scores[best - 1] * 1.03
                ]
                if cands:
                    best = max(cands)
            else:
                for k in range(1, min(5, self.k + 1) + 1):
                    if (
                        scores[k - 1] - scores[self.k - 1]
                        > 0.03 * scores[self.k - 1] + gate[k - 1]
                        and scores[k - 1] > scores[best - 1]
                    ):
                        best = k
        if best < 5:
            self.trimmed = True
        self.diagnostics = dict(
            score=scores,
            se=se,
            gate=gate,
            model_se=model_se,
            ready=ready,
            trimmed=self.trimmed,
        )
        return best


def self_test():
    p = Policy({})
    for _ in range(20):
        p.observe(
            dict(
                k=5,
                accepted=5,
                seconds=0.03,
                output_tokens=6,
                timing_valid=True,
                load_epoch=1,
            )
        )
        assert p.choose(dict(current_k=5, concurrency=1)) == 5
    p = Policy({})
    p.observe(
        dict(
            k=2,
            accepted=1,
            seconds=0.025,
            output_tokens=2,
            timing_valid=True,
            load_epoch=1,
        )
    )
    for _ in range(10):
        p.observe(
            dict(
                k=3,
                accepted=2,
                seconds=3 / 90,
                output_tokens=3,
                timing_valid=True,
                load_epoch=1,
            )
        )
    assert abs(p.y[1] / p.t[1] - 80) < 1e-8
    assert abs(p.n[1] - 0.95**10) < 1e-8
