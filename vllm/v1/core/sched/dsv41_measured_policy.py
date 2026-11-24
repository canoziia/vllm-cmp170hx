# SPDX-License-Identifier: Apache-2.0
"""Request-local measured/predicted speculative verification selection.

Discount evidence, not empirical speed. Calibrate candidate predictions to the
current observed output/time level, then blend each width's measured evidence.
Require a gain tolerance plus measured-ratio and paired-prefix uncertainty to
avoid chasing marginal noise. These gates are engineering heuristics, not
formal confidence guarantees. Timing is CPU-visible recurrence, not GPU time.
No task labels, external IO, CUDA objects or fixed observation windows.
"""

API_VERSION = 1
# Steady-state GPU step curves on 4x GA100 80G TEP4+DCP1 (k=1..5):
# Keep true gentle slope at c=1..8 (avoiding short-run TTFT skew that
# over-trimmed c=1..8 prose) while preserving exact sub-width CG gains at c=16,32.
COST_MS = {
    1: (12.8400, 13.8000, 14.1500, 15.0500, 15.7200),
    2: (14.7200, 15.7000, 16.3000, 18.5400, 19.3100),
    4: (16.8000, 18.7000, 20.7700, 22.8000, 25.5000),
    8: (22.3400, 25.0000, 29.6000, 31.5000, 33.5000),
    16: (28.5500, 32.5200, 38.2600, 43.9900, 48.0700),
    32: (40.5000, 53.8000, 63.0000, 72.5000, 81.2200),
}


def costs(c):
    cs = sorted(COST_MS)
    if c <= cs[0]:
        return [v / 1000 for v in COST_MS[cs[0]]]
    for lo, hi in zip(cs, cs[1:]):
        if c <= hi:
            w = (c - lo) / (hi - lo)
            return [
                (a * (1 - w) + b * w) / 1000 for a, b in zip(COST_MS[lo], COST_MS[hi])
            ]
    return [v / 1000 for v in COST_MS[cs[-1]]]


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
                cold = max(k for k in range(1, 6) if scores[k - 1] >= 0.98 * peak)
                if scores[cold - 1] > scores[best - 1] * 1.03:
                    best = cold
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
