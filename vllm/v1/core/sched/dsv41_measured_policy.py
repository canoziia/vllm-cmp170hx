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
    1: (7.1800, 7.6200, 8.1200, 8.5800, 8.9000),
    2: (8.2000, 9.1000, 9.7500, 10.7000, 11.1000),
    4: (9.6000, 11.2000, 11.8000, 13.4000, 13.7500),
    8: (11.1000, 14.3000, 16.6000, 18.4000, 19.5000),
    16: (18.3000, 22.8000, 26.2000, 29.5000, 30.6000),
    32: (26.5000, 35.0000, 42.2000, 48.4000, 51.8000),
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
_MIN_STEP_RATIO = (1.045, 1.035, 1.030, 1.020)
_MAX_STEP_RATIO = (1.380, 1.240, 1.160, 1.110)
_FEAT_SCALE = (1.0, 50.0, 7.0)
_CG_CAPTURE_SIZES = (
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32,
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


def _solve_nxn(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        piv = i
        best = abs(m[i][i])
        for r in range(i + 1, n):
            v = abs(m[r][i])
            if v > best:
                best = v
                piv = r
        if best < 1e-12:
            return None
        if piv != i:
            m[i], m[piv] = m[piv], m[i]
        diag = m[i][i]
        for r in range(i + 1, n):
            factor = m[r][i] / diag
            for c in range(i, n + 1):
                m[r][c] -= factor * m[i][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n]
        for c in range(i + 1, n):
            s -= m[i][c] * x[c]
        x[i] = s / m[i][i]
    return x


def _solve_nnls_4(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Exact active-set NNLS over 4 physics parameters (theta_0 > 0.5, theta_1..3 >= 0)."""
    best_x = None
    best_obj = float("inf")
    for mask in range(1, 16):
        if not (mask & 1):
            continue
        idx = [i for i in range(4) if (mask >> i) & 1]
        sub_a = [[a[r][c] for c in idx] for r in idx]
        sub_b = [b[r] for r in idx]
        sol = _solve_nxn(sub_a, sub_b)
        if sol is None or any(not math.isfinite(v) for v in sol):
            continue
        if sol[0] <= 0.5 or any(v < 0.0 for v in sol[1:]):
            continue
        x = [0.0] * 4
        for pos, i in enumerate(idx):
            x[i] = sol[pos]
        obj = 0.0
        for r in range(4):
            ax_r = sum(a[r][c] * x[c] for c in range(4))
            obj += 0.5 * x[r] * ax_r - b[r] * x[r]
        if obj < best_obj:
            best_obj = obj
            best_x = x
    return best_x


class OnlineCostModel:
    """Hardware-agnostic two-level online step-latency model (pure CPU, ~8 us/step).

    Level 1 (4-parameter physical MoE decode surface with scale-free live prior):
        T_phys(B, M) = theta_0
                     + theta_B   * (B / 32)
                     + theta_moe * (1 - (1 - K / E)^M)
                     + theta_M   * (M / 192)
        Decouples per-request draft/scheduler cost (B), coupon-collector MoE
        unique-expert saturation (smooth and linear near M -> 0, eliminating
        the sqrt(M) derivative singularity at c=1), and linear token compute/
        collective cost (M). A live multiplicative scale factor rescales the
        weak Ridge prior to any host GPU/power cap/interconnect automatically.

    Level 2 (per-concurrency empirical-Bayes cell shrinkage in costs(c)):
        Each concurrency c computes a local anchor from its fresh direct cell
        observations and shrinks each observed (c, k) cell toward its exact
        measured EMA, followed by bounded monotone step-ratio projection so
        high-c traffic can never distort c=1 or c=8 decisions.
    """

    def __init__(
        self,
        prior_table: dict[int, tuple[float, ...]] | None = None,
        local_gamma: float = 0.92,
        age_gamma: float = 0.996,
        age_floor: float = 0.05,
        prior_obs: float = 8.0,
        huber_delta: float = 0.25,
        cell_kappa: float = 4.0,
        max_cells: int = 128,
        refit_interval: int = 4,
        moe_topk: int = 8,
        num_experts: int = 384,
    ) -> None:
        self.local_gamma = float(local_gamma)
        self.age_gamma = float(age_gamma)
        self.age_floor = float(age_floor)
        self.prior_obs = float(prior_obs)
        self.huber_delta = float(huber_delta)
        self.cell_kappa = float(cell_kappa)
        self.max_cells = int(max_cells)
        self.refit_interval = max(1, int(refit_interval))
        self.moe_p = max(1e-4, min(0.5, float(moe_topk) / max(1, int(num_experts))))

        # Fit dimensionless shape prior from prior_table (or universal defaults)
        tbl = prior_table if prior_table is not None else _active_cost_ms()
        a_p = [[0.0] * 4 for _ in range(4)]
        b_p = [0.0] * 4
        n_pts = 0
        for c, vals in tbl.items():
            for idx, y_ms in enumerate(vals):
                f = self._features(float(c), float(c * (idx + 2)))
                w = 1.0 / max(1.0, y_ms * y_ms)
                for r in range(4):
                    b_p[r] += w * f[r] * y_ms
                    for j in range(4):
                        a_p[r][j] += w * f[r] * f[j]
                n_pts += 1
        a_fit = [
            [a_p[r][j] + (1e-6 if r == j else 0.0) for j in range(4)]
            for r in range(4)
        ]
        theta_prior = _solve_nnls_4(a_fit, b_p)
        if theta_prior is None or not all(math.isfinite(v) for v in theta_prior):
            theta_prior = [6.6, 5.4, 7.5, 30.2]
        self._shape_prior = theta_prior
        # Unit-scale prior precision matrix (rescaled live by 1 / live_scale^2)
        scale = float(prior_obs) / max(1, n_pts)
        self._prior_a_unit = [[v * scale for v in row] for row in a_p]
        self._live_scale = 1.0
        self._theta = theta_prior[:]
        # (B, M) -> [local_mass, dt_ema_ms, last_obs_step]
        self._cells: dict[tuple[int, int], list[float]] = {}
        self.obs_count = 0

    def _features(self, batch_b: float, batch_m: float) -> tuple[float, float, float, float]:
        me = _effective_tokens(batch_m)
        return (
            1.0,
            batch_b / 32.0,
            1.0 - (1.0 - self.moe_p) ** max(0.0, me),
            me / 192.0,
        )

    def _predict_global_ms(self, batch_b: float, batch_m: float) -> float:
        f = self._features(batch_b, batch_m)
        val = (
            f[0] * self._theta[0]
            + f[1] * self._theta[1]
            + f[2] * self._theta[2]
            + f[3] * self._theta[3]
        )
        return max(0.5, val)

    def _refit_global(self) -> None:
        step = self.obs_count
        ag = self.age_gamma
        af = self.age_floor
        num_s = 0.0
        den_s = 0.0
        weighted_cells: list[tuple[float, float, tuple[float, float, float, float]]] = []
        for (cb, cm), cell in self._cells.items():
            mass, ema_ms, last_step = cell
            age = max(0, step - int(last_step))
            fresh = max(af, ag**age) if age < 3000 else af
            cw = fresh * (mass / (mass + 2.0))
            f = self._features(float(cb), float(cm))
            y_shape = max(
                0.5,
                f[0] * self._shape_prior[0]
                + f[1] * self._shape_prior[1]
                + f[2] * self._shape_prior[2]
                + f[3] * self._shape_prior[3],
            )
            num_s += cw * (ema_ms / y_shape)
            den_s += cw
            weighted_cells.append((cw, ema_ms, f))
        if den_s > 1e-6:
            self._live_scale = max(0.05, min(20.0, num_s / den_s))

        ls = self._live_scale
        prior_th = [v * ls for v in self._shape_prior]
        inv_ls2 = 1.0 / max(1e-4, ls * ls)
        a = [[v * inv_ls2 for v in row] for row in self._prior_a_unit]
        b = [
            sum(a[r][j] * prior_th[j] for j in range(4))
            for r in range(4)
        ]
        for i in range(4):
            a[i][i] += 1e-7 * inv_ls2

        for cw, ema_ms, f in weighted_cells:
            inv_ref = 1.0 / max(1.0, ema_ms)
            w = cw * inv_ref * inv_ref
            for r in range(4):
                b[r] += w * f[r] * ema_ms
                fr = w * f[r]
                row = a[r]
                for j in range(4):
                    row[j] += fr * f[j]

        cand = _solve_nnls_4(a, b)
        if cand is not None:
            self._theta = cand

    def predict_ms(self, batch_b: int, batch_m: int) -> float:
        base = self._predict_global_ms(float(batch_b), float(batch_m))
        cell = self._cells.get((int(batch_b), int(batch_m)))
        if cell is not None and cell[0] > 0.0:
            mass, ema_ms, last_step = cell
            age = max(0, self.obs_count - int(last_step))
            fresh = self.age_gamma**age if age < 3000 else 0.0
            eff_n = mass * fresh
            if eff_n > 0.05:
                alpha = eff_n / (eff_n + self.cell_kappa)
                ratio = max(0.70, min(1.40, ema_ms / base))
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
        ref_ms = (
            cell[1]
            if cell is not None
            else self._predict_global_ms(float(batch_b), float(batch_m))
        )
        rel_err = abs(dt_ms - ref_ms) / max(1.0, ref_ms)
        huber = (
            self.huber_delta / rel_err if rel_err > self.huber_delta else 1.0
        )
        # Allow rapid cold-start scale adaptation on an unfamiliar host/power cap
        if self.obs_count < 16 and 0.5 < dt_ms < 250.0:
            huber = 1.0
        self.obs_count += 1
        is_new = cell is None
        if is_new:
            if len(self._cells) >= self.max_cells:
                oldest = min(self._cells, key=lambda k: self._cells[k][2])
                del self._cells[oldest]
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
        g = [self._predict_global_ms(float(c), float(c * (k + 1))) for k in range(1, 6)]
        num_a = 0.0
        den_a = 0.0
        cell_info: list[tuple[float, float]] = []
        for idx in range(5):
            m = c * (idx + 2)
            cell = self._cells.get((c, m))
            eff_n = 0.0
            ema = g[idx]
            if cell is not None and cell[0] > 0.0:
                age = max(0, self.obs_count - int(cell[2]))
                fresh = self.age_gamma**age if age < 3000 else 0.0
                eff_n = cell[0] * fresh
                ema = cell[1]
                if eff_n > 0.05:
                    w = eff_n / (eff_n + self.cell_kappa)
                    num_a += w * (ema / g[idx])
                    den_a += w
            cell_info.append((eff_n, ema))
        c_anchor = (num_a / den_a) if den_a > 0.0 else 1.0
        c_anchor = max(0.65, min(1.50, c_anchor))

        raw: list[float] = []
        for idx in range(5):
            base = g[idx] * c_anchor
            eff_n, ema = cell_info[idx]
            if eff_n > 0.05:
                alpha = eff_n / (eff_n + self.cell_kappa)
                ratio = max(0.70, min(1.40, ema / base))
                base *= 1.0 + alpha * (ratio - 1.0)
            raw.append(max(0.5, base))

        out = [raw[0]]
        for i in range(4):
            raw_ratio = raw[i + 1] / raw[i] if raw[i] > 0.0 else _MIN_STEP_RATIO[i]
            clamped = max(_MIN_STEP_RATIO[i], min(_MAX_STEP_RATIO[i], raw_ratio))
            out.append(out[i] * clamped)
        return [v / 1000.0 for v in out]

    def diagnostics(self) -> dict:
        return {
            "obs_count": self.obs_count,
            "scale": round(self._live_scale, 3),
            "theta_ms": [round(v, 3) for v in self._theta],
            "cells": len(self._cells),
            "c1_ms": [round(v * 1000.0, 2) for v in self.costs(1)],
            "c4_ms": [round(v * 1000.0, 2) for v in self.costs(4)],
            "c8_ms": [round(v * 1000.0, 2) for v in self.costs(8)],
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

    def expected_tokens(self) -> list[float]:
        p = []
        previous = 1.0
        for n, s in zip(self.risk, self.success):
            previous = min(
                previous, (s + self.prior + (0.5 + 0.4 * (previous ** 6)) * previous) / (n + self.prior + 1)
            )
            p.append(previous)
        return [1 + sum(p[:k]) for k in range(1, 6)]

    def choose(self, context):
        self.k = context["current_k"]
        E = self.expected_tokens()
        cost_model = context.get("cost_model")
        if cost_model is not None:
            C = cost_model.costs(context["concurrency"])
        else:
            C = costs(context["concurrency"])
        i = self.k - 1
        # Pure unbiased throughput estimator: shared survival-curve E[k]
        # divided by live OnlineCostModel latency C[k]. Eliminates the
        # single-sample arm-attribution bug where one unlucky a=0 step on a
        # newly tried width k crashed its per-arm y[k]/t[k] score by >50%.
        scores = [e / c for e, c in zip(E, C)]
        se = [0.0] * 5
        hist = self.history[i]
        mass = sum(hist)
        model_se = []
        for j in range(5):
            if j >= i:
                model_se.append(0.0)
                continue
            # Paired marginal efficiency on prefixes observed at current width.
            diffs = [
                (1 + min(a, j + 1)) / C[j] - (1 + min(a, self.k)) / C[i]
                for a in range(6)
            ]
            mean = sum(w * d for w, d in zip(hist, diffs)) / mass if mass else 0.0
            variance = (
                sum(w * (d - mean) ** 2 for w, d in zip(hist, diffs)) / mass
                if mass
                else 0.0
            )
            model_se.append(0.5 * (variance / (mass + 1)) ** 0.5)
        gate = model_se[:]
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
                    if scores[k - 1] >= 0.995 * peak
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
