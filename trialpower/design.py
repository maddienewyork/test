"""Frequentist (classical) power and sample-size calculations for one trial."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .endpoints import Binary, Survival


@dataclass
class TrialDesign:
    """A single two-arm superiority trial.

    endpoint: a Continuous / Binary / Survival endpoint.
    n_total: total patients *enrolled* (both arms).
    alpha: overall type-I error. With ``sides=2`` the test is two-sided at
        ``alpha`` (i.e. alpha/2 in the favourable direction).
    ratio: allocation treatment:control (2.0 means 2:1).
    dropout: fraction of enrolled patients who are not evaluable.
    """

    endpoint: object
    n_total: float
    alpha: float = 0.05
    sides: int = 2
    ratio: float = 1.0
    dropout: float = 0.0

    # ---- helpers -------------------------------------------------------
    @property
    def a1(self) -> float:
        """One-sided alpha in the favourable direction."""
        return self.alpha / self.sides

    def arms(self, n_total=None):
        """Evaluable patients per arm (treatment, control)."""
        n = (self.n_total if n_total is None else n_total) * (1 - self.dropout)
        return n * self.ratio / (1 + self.ratio), n / (1 + self.ratio)

    def _b(self, effect):
        return float(self.endpoint.to_benefit(effect))

    # ---- core quantities ----------------------------------------------
    def power(self, effect, n_total=None) -> float:
        """Power to show a statistically significant benefit if the true effect is ``effect``."""
        return float(self.power_benefit(self._b(effect), n_total))

    def power_benefit(self, b, n_total=None):
        nt, nc = self.arms(n_total)
        return self.endpoint.power(b, nt, nc, self.a1)

    def critical_effect(self, n_total=None) -> float:
        """Smallest *observed* effect that would be statistically significant."""
        nt, nc = self.arms(n_total)
        return float(self.endpoint.from_benefit(self.endpoint.critical_benefit(nt, nc, self.a1)))

    def events(self, n_total=None):
        if isinstance(self.endpoint, Survival):
            return float(self.endpoint.events(*self.arms(n_total)))
        return None

    def sample_size(self, effect, target_power=0.8) -> dict:
        """Enrolled N needed to reach ``target_power`` for true ``effect``."""
        b = self._b(effect)
        if not np.isfinite(b) or b <= 0:
            return {"n_total": math.inf, "n_t": math.inf, "n_c": math.inf, "events": None, "power": float("nan")}

        def f(n):
            return self.power_benefit(b, n) - target_power

        lo = 4.0 / (1 - self.dropout)
        hi = lo
        while f(hi) < 0:
            hi *= 2
            if hi > 1e8:
                return {"n_total": math.inf, "n_t": math.inf, "n_c": math.inf, "events": None, "power": float("nan")}
        n_req = optimize.brentq(f, lo, hi, xtol=1e-6) if f(lo) < 0 else lo
        # round up per arm (evaluable), then inflate for dropout
        nt_e, nc_e = self.arms(n_req)
        n_t = math.ceil(math.ceil(nt_e - 1e-9) / (1 - self.dropout) - 1e-9)
        n_c = math.ceil(math.ceil(nc_e - 1e-9) / (1 - self.dropout) - 1e-9)
        n = n_t + n_c
        return {
            "n_total": n,
            "n_t": n_t,
            "n_c": n_c,
            "events": None if self.events(n) is None else math.ceil(self.events(n_req) - 1e-9),
            "power": float(self.power_benefit(b, n)),
        }

    def mde(self, target_power=0.8, n_total=None) -> float:
        """Minimum detectable (true) effect: the effect giving ``target_power`` at this N."""
        hi = 1.0
        if isinstance(self.endpoint, Binary):
            hi = self.endpoint.max_benefit() - 1e-6
            if self.power_benefit(hi, n_total) < target_power:
                return float("nan")
        else:
            while self.power_benefit(hi, n_total) < target_power:
                hi *= 2
                if hi > 1e6:
                    return float("nan")
        b = optimize.brentq(lambda x: self.power_benefit(x, n_total) - target_power, 1e-12, hi, xtol=1e-10)
        return float(self.endpoint.from_benefit(b))

    def power_curve_n(self, effect, n_grid):
        b = self._b(effect)
        return np.array([float(self.power_benefit(b, n)) for n in n_grid])

    def power_curve_effect(self, effects, n_total=None):
        return np.asarray(self.power_benefit(self.endpoint.to_benefit(effects), n_total), dtype=float)
