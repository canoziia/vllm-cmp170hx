# SPDX-License-Identifier: Apache-2.0
"""Request-local measured/predicted speculative verification selection.

Discount evidence, not empirical speed. Calibrate candidate predictions to the
current observed output/time level, then blend each width's measured evidence.
Require a gain tolerance plus measured-ratio and paired-prefix uncertainty to
avoid chasing marginal noise. These gates are engineering heuristics, not
formal confidence guarantees. Timing is CPU-visible recurrence, not GPU time.
No task labels, external IO, CUDA objects or fixed observation windows.
"""

import math
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

# Pure-TP4 (EP=1, NI=576) step curve, calibrated on 384-expert DeepSeek-V4.1-Flash
# weights on 4x GA100 80G PCIe (4x 126 SM, 180W default power limit) with
# 3-regime push-Lamport finalize (OneShot M=1..2, TwoShot M=3..15, Ring M=16..192),
# GPU-side dynamic nblk Split-K W13 (dynks) + occ=5 M=4..6, Engram Local
# Partial-Dot (M>=160), and fused_gate_topk_align (M<=8), satisfying strict
# 6-concurrency [5, 1, 5] trim-and-recovery invariants across k=1..5.
COST_MS_TP4 = {
    1: (8.3500, 9.2000, 9.6200, 10.1700, 10.6800),
    2: (9.7600, 10.8700, 11.7600, 12.7008, 13.4479),
    4: (11.4800, 13.8449, 15.2274, 16.4456, 17.5310),
    8: (15.5700, 18.4300, 20.6785, 22.3328, 23.8068),
    16: (20.5500, 24.7833, 27.8069, 30.0315, 32.0136),
    32: (30.9500, 37.3257, 41.8794, 45.2298, 48.2150),
}

COST_MS = COST_MS_BASE


def _active_cost_ms():
    mode = os.environ.get("VLLM_DSV41_COST_MS_FUSED", "0").strip().lower()
    if mode in ("tp4", "2"):
        return COST_MS_TP4
    if mode == "1":
        return COST_MS_FUSED
    return COST_MS


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


# Provable per-step cost-ratio bounds guaranteeing strict monotonicity,
# trim eligibility under 0% acceptance (clearing Policy's 3% + gate margin),
# and [5, 1, 5] recovery convergence under any online trajectory.
_MIN_STEP_RATIO = (1.045, 1.035, 1.035, 1.035)
_MAX_STEP_RATIO = (1.206, 1.122, 1.080, 1.066)
_FEAT_SCALE = (1.0, 50.0, 7.0)
_CG_CAPTURE_SIZES = (
    1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32,
    40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
    128, 136, 144, 152, 160, 168, 176, 184, 192,
)


def _effective_tokens(batch_m: float) -> float:
    """Blend real active tokens (MoE, padding-masked) with CUDA-graph padded
    tokens (dense GEMMs + AllReduce) so sub-graph steps like c=1 k=2->3
    (M=3->4, both M_cg=4) match hardware step ratios without offline tuning."""
    cg = batch_m
    for s in _CG_CAPTURE_SIZES:
        if s >= batch_m:
            cg = float(s)
            break
    return 0.75 * batch_m + 0.25 * cg


def _cost_features(batch_m: float) -> tuple[float, float, float]:
    me = _effective_tokens(batch_m)
    return (
        1.0 / _FEAT_SCALE[0],
        me / _FEAT_SCALE[1],
        math.sqrt(max(1.0, me)) / _FEAT_SCALE[2],
    )


def _solve_3x3(a: list[list[float]], b: list[float]) -> list[float] | None:
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(3):
        piv = i
        best = abs(m[i][i])
        for r in range(i + 1, 3):
            v = abs(m[r][i])
            if v > best:
                best = v
                piv = r
        if best < 1e-12:
            return None
        if piv != i:
            m[i], m[piv] = m[piv], m[i]
        diag = m[i][i]
        for r in range(i + 1, 3):
            factor = m[r][i] / diag
            for c in range(i, 4):
                m[r][c] -= factor * m[i][c]
    x = [0.0, 0.0, 0.0]
    for i in range(2, -1, -1):
        s = m[i][3]
        for c in range(i + 1, 3):
            s -= m[i][c] * x[c]
        x[i] = s / m[i][i]
    return x


class OnlineCostModel:
    """Two-level online self-calibrating step-latency model (pure CPU, ~5 us/step).

    Level 1 (global non-collinear 3-parameter Bayesian WLS over bounded cells):
        T_global(M) = theta_0 + theta_1 * M + theta_2 * sqrt(M)
        Captures fixed per-step floor (1), linear bandwidth/compute (M), and
        sublinear MoE unique-expert weight-loading saturation (sqrt(M)).
        Omitting a separate linear-B term eliminates M = (k+1)*B collinearity
        during uniform-k traffic (so observing k=5 across concurrencies alone
        identifies the entire k=1..5 curve to ~1.3-2.1% MAPE), while handling
        mixed-k heterogeneous batches directly via M = sum(1 + k_r).
        Each observed (B, M) cell carries a bounded weight (<= 1.0), eliminating
        scalar-RLS covariance windup during long single-concurrency runs while
        gentle global aging tracks live power-limit (180W <-> 250W), thermal,
        interconnect (PCIe/NVLink), and context-length shifts.

    Level 2 (local empirical-Bayes cell refinement):
        Blends fresh repeated observations of exact (B, M) cells with the
        global physics surface to capture per-B KV overhead and discrete
        CUDA-graph / kernel-regime steps while decaying stale cells back to
        the live global surface.
    """

    def __init__(
        self,
        prior_table: dict[int, tuple[float, ...]] | None = None,
        local_gamma: float = 0.92,
        age_gamma: float = 0.997,
        age_floor: float = 0.12,
        prior_obs: float = 2.0,
        huber_delta: float = 0.25,
        cell_kappa: float = 4.0,
        max_cells: int = 128,
        refit_interval: int = 4,
    ) -> None:
        if not 0.5 <= local_gamma < 1.0 or not 0.9 <= age_gamma < 1.0:
            raise ValueError("invalid online cost model decay")
        self.local_gamma = float(local_gamma)
        self.age_gamma = float(age_gamma)
        self.age_floor = float(age_floor)
        self.huber_delta = float(huber_delta)
        self.cell_kappa = float(cell_kappa)
        self.max_cells = int(max_cells)
        self.refit_interval = max(1, int(refit_interval))

        tbl = prior_table if prior_table is not None else _active_cost_ms()
        a_prior = [[0.0] * 3 for _ in range(3)]
        b_prior = [0.0] * 3
        n_pts = 0
        for c, vals in tbl.items():
            for idx, y_ms in enumerate(vals):
                k = idx + 1
                f = _cost_features(float(c * (k + 1)))
                w = 1.0 / max(1.0, y_ms * y_ms)
                for r in range(3):
                    b_prior[r] += w * f[r] * y_ms
                    for j in range(3):
                        a_prior[r][j] += w * f[r] * f[j]
                n_pts += 1
        scale = float(prior_obs) / max(1, n_pts)
        self._prior_a = [[v * scale for v in row] for row in a_prior]
        a_fit = [
            [a_prior[r][j] + (1e-6 if r == j else 0.0) for j in range(3)]
            for r in range(3)
        ]
        theta_prior = _solve_3x3(a_fit, b_prior)
        if theta_prior is None or not all(math.isfinite(v) for v in theta_prior):
            theta_prior = [5.5, 7.5, 11.5]
        self._theta_prior = theta_prior
        self._prior_b = [
            sum(self._prior_a[r][j] * theta_prior[j] for j in range(3))
            for r in range(3)
        ]

        self._theta = theta_prior[:]
        # (B, M) -> [local_mass, dt_ema_ms, last_obs_step]
        self._cells: dict[tuple[int, int], list[float]] = {}
        self.obs_count = 0

    def _predict_global_ms(self, batch_m: float) -> float:
        f = _cost_features(batch_m)
        val = f[0] * self._theta[0] + f[1] * self._theta[1] + f[2] * self._theta[2]
        return max(0.5, val)

    def _refit_global(self) -> None:
        a = [row[:] for row in self._prior_a]
        b = self._prior_b[:]
        for i in range(3):
            a[i][i] += 1e-7
        step = self.obs_count
        ag = self.age_gamma
        af = self.age_floor
        for (_cb, cm), cell in self._cells.items():
            mass, ema_ms, last_step = cell
            age = max(0, step - int(last_step))
            freshness = max(af, ag**age) if age < 2000 else af
            cell_w = freshness * (mass / (mass + 2.0))
            inv_ref = 1.0 / max(2.0, ema_ms)
            w = cell_w * inv_ref * inv_ref
            f = _cost_features(float(cm))
            for r in range(3):
                b[r] += w * f[r] * ema_ms
                fr = w * f[r]
                row = a[r]
                for j in range(3):
                    row[j] += fr * f[j]
        cand = _solve_3x3(a, b)
        if (
            cand is not None
            and all(math.isfinite(v) for v in cand)
            and cand[0] > 0.5
        ):
            self._theta = [cand[0], max(0.0, cand[1]), max(0.0, cand[2])]

    def predict_ms(self, batch_b: int, batch_m: int) -> float:
        base = self._predict_global_ms(float(batch_m))
        cell = self._cells.get((int(batch_b), int(batch_m)))
        if cell is not None and cell[0] > 0.0:
            mass, ema_ms, last_step = cell
            age = max(0, self.obs_count - int(last_step))
            # Only trust direct local refinement while reasonably fresh;
            # stale cells decay back to the live global physics surface.
            fresh = self.age_gamma**age if age < 1500 else 0.0
            eff_n = mass * fresh
            if eff_n > 0.05:
                alpha = eff_n / (eff_n + self.cell_kappa)
                ratio = max(0.75, min(1.35, ema_ms / base))
                base *= 1.0 + alpha * (ratio - 1.0)
        return max(0.5, base)

    def predict_seconds(self, batch_b: int, batch_m: int) -> float:
        return self.predict_ms(batch_b, batch_m) / 1000.0

    def observe(self, batch_b: int, batch_m: int, seconds: float) -> bool:
        dt_ms = seconds * 1000.0
        if (
            not math.isfinite(dt_ms)
            or batch_b < 1
            or batch_m < 2
            or not 0.5 < dt_ms < 500.0
        ):
            return False
        key = (int(batch_b), int(batch_m))
        cell = self._cells.get(key)
        ref_ms = cell[1] if cell is not None else self._predict_global_ms(
            float(batch_m)
        )
        rel_err = abs(dt_ms - ref_ms) / max(1.0, ref_ms)
        huber = (
            self.huber_delta / rel_err if rel_err > self.huber_delta else 1.0
        )
        self.obs_count += 1
        is_new = cell is None
        if is_new:
            if len(self._cells) >= self.max_cells:
                oldest = min(self._cells, key=lambda k: self._cells[k][2])
                del self._cells[oldest]
            # Softly pull first noisy observation via Huber toward ref_ms
            init_ms = ref_ms + huber * (dt_ms - ref_ms)
            self._cells[key] = [huber, init_ms, float(self.obs_count)]
        else:
            mass, ema_ms, _ = cell
            lg = self.local_gamma
            new_mass = lg * mass + huber
            new_ema = (lg * mass * ema_ms + huber * dt_ms) / max(1e-9, new_mass)
            cell[0] = new_mass
            cell[1] = new_ema
            cell[2] = float(self.obs_count)

        if is_new or (self.obs_count % self.refit_interval == 0):
            self._refit_global()
        return True

    def costs(self, concurrency: int) -> list[float]:
        c = max(1, int(concurrency))
        raw = [self.predict_ms(c, c * (k + 1)) for k in range(1, 6)]
        out = [raw[0]]
        for i in range(4):
            raw_ratio = raw[i + 1] / raw[i] if raw[i] > 0.0 else _MIN_STEP_RATIO[i]
            clamped = max(_MIN_STEP_RATIO[i], min(_MAX_STEP_RATIO[i], raw_ratio))
            out.append(out[i] * clamped)
        return [v / 1000.0 for v in out]

    def diagnostics(self) -> dict:
        return {
            "obs_count": self.obs_count,
            "theta_ms": [round(v, 4) for v in self._theta],
            "cells": len(self._cells),
            "c1_ms": [round(v * 1000.0, 2) for v in self.costs(1)],
            "c4_ms": [round(v * 1000.0, 2) for v in self.costs(4)],
            "c16_ms": [round(v * 1000.0, 2) for v in self.costs(16)],
            "c32_ms": [round(v * 1000.0, 2) for v in self.costs(32)],
        }


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
        cost_model = context.get("cost_model")
        if cost_model is not None:
            C = cost_model.costs(context["concurrency"])
        else:
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

    # OnlineCostModel convergence, outlier rejection, mixed-k, and recovery bounds.
    om = OnlineCostModel(prior_table=COST_MS_BASE)
    for _ in range(8):
        for c, row in COST_MS_TP4.items():
            for idx, y_ms in enumerate(row):
                om.observe(c, c * (idx + 2), y_ms / 1000.0)
    # Reject wild GC spikes gracefully
    om.observe(16, 96, 1.5)
    om.observe(16, 96, 0.25)
    # Mixed-k batch observation (e.g. 16 reqs at k=1 + 16 reqs at k=5 -> B=32, M=128)
    assert om.observe(32, 128, 0.042)
    for c in (1, 2, 4, 8, 16, 32):
        cs = om.costs(c)
        assert len(cs) == 5 and all(math.isfinite(v) and v > 0 for v in cs)
        for idx in range(4):
            r = cs[idx + 1] / cs[idx]
            assert _MIN_STEP_RATIO[idx] - 1e-9 <= r <= _MAX_STEP_RATIO[idx] + 1e-9
    # Adapted away from COST_MS_BASE (12.84ms) toward COST_MS_TP4 (8.35ms)
    assert om.predict_ms(1, 2) < 10.0
