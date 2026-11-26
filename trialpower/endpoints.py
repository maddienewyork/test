"""Endpoint models.

Every endpoint maps its *natural* effect measure (mean difference, risk
difference, hazard ratio) onto a common internal **benefit scale** where
``benefit > 0`` always means "treatment is better than control". All power,
prior and simulation maths works on the benefit scale; results are converted
back to natural units for display.

=============  =====================  ==============================
Endpoint       Natural effect         Benefit
=============  =====================  ==============================
Continuous     mean_T - mean_C        +/- difference
Binary         p_T - p_C              +/- risk difference
Survival       HR (T vs C)            -log(HR)  (if lower HR is better)
=============  =====================  ==============================
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

_EPS = 1e-9


def _k(nt, nc):
    """sqrt(1/nt + 1/nc): the two-sample standard-error multiplier."""
    return np.sqrt(1.0 / nt + 1.0 / nc)


@dataclass
class Continuous:
    """Continuous endpoint analysed with a two-sample t-test.

    sd: common (pooled) standard deviation of the outcome.
    higher_is_better: True if a larger treatment-minus-control difference is good.
    """

    sd: float
    higher_is_better: bool = True
    kind: str = "continuous"
    effect_label: str = "Mean difference (T - C)"

    def to_benefit(self, effect):
        e = np.asarray(effect, dtype=float)
        return e if self.higher_is_better else -e

    def from_benefit(self, b):
        b = np.asarray(b, dtype=float)
        return b if self.higher_is_better else -b

    def se(self, b, nt, nc):
        return self.sd * _k(nt, nc) * np.ones_like(np.asarray(b, dtype=float))

    def critical_benefit(self, nt, nc, a1):
        df = nt + nc - 2
        return stats.t.ppf(1 - a1, df) * self.sd * _k(nt, nc)

    def power(self, b, nt, nc, a1):
        """P(significant in the favourable direction) — exact noncentral t."""
        df = nt + nc - 2
        tcrit = stats.t.ppf(1 - a1, df)
        ncp = np.asarray(b, dtype=float) / (self.sd * _k(nt, nc))
        return stats.nct.sf(tcrit, df, ncp)

    def simulate(self, b, nt, nc, a1, rng):
        b = np.asarray(b, dtype=float)
        nt, nc = max(int(round(nt)), 2), max(int(round(nc)), 2)
        df = nt + nc - 2
        k = _k(nt, nc)
        b_hat = b + self.sd * k * rng.standard_normal(b.shape)
        s_hat = self.sd * np.sqrt(rng.chisquare(df, b.shape) / df)
        t_stat = b_hat / (s_hat * k)
        return b_hat, t_stat > stats.t.ppf(1 - a1, df)


@dataclass
class Binary:
    """Binary (responder) endpoint analysed with a pooled two-proportion z-test.

    p_control: expected control-arm response rate.
    higher_is_better: True if a higher response rate is good.
    """

    p_control: float
    higher_is_better: bool = True
    kind: str = "binary"
    effect_label: str = "Risk difference (pT - pC)"

    def to_benefit(self, effect):
        e = np.asarray(effect, dtype=float)
        return e if self.higher_is_better else -e

    def from_benefit(self, b):
        b = np.asarray(b, dtype=float)
        return b if self.higher_is_better else -b

    def p_treat(self, b):
        pt = self.p_control + self.from_benefit(b)
        return np.clip(pt, _EPS, 1 - _EPS)

    def max_benefit(self):
        return (1 - self.p_control) if self.higher_is_better else self.p_control

    def _se0(self, b, nt, nc):
        pt = self.p_treat(b)
        pbar = (nt * pt + nc * self.p_control) / (nt + nc)
        return np.sqrt(pbar * (1 - pbar) * (1 / nt + 1 / nc))

    def se(self, b, nt, nc):
        pt, pc = self.p_treat(b), self.p_control
        return np.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc)

    def critical_benefit(self, nt, nc, a1):
        # observed difference needed for significance when the pooled rate ~= p_control
        return stats.norm.ppf(1 - a1) * self._se0(0.0, nt, nc)

    def power(self, b, nt, nc, a1):
        b = np.asarray(b, dtype=float)
        z = stats.norm.ppf(1 - a1)
        b_eff = self.to_benefit(self.p_treat(b) - self.p_control)  # respects clipping
        return stats.norm.cdf((b_eff - z * self._se0(b, nt, nc)) / self.se(b, nt, nc))

    def simulate(self, b, nt, nc, a1, rng):
        b = np.asarray(b, dtype=float)
        nt, nc = max(int(round(nt)), 1), max(int(round(nc)), 1)
        xt = rng.binomial(nt, self.p_treat(b))
        xc = rng.binomial(nc, self.p_control, size=b.shape)
        pt_hat, pc_hat = xt / nt, xc / nc
        pbar = (xt + xc) / (nt + nc)
        se0 = np.sqrt(pbar * (1 - pbar) * (1 / nt + 1 / nc))
        b_hat = self.to_benefit(pt_hat - pc_hat)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se0 > 0, b_hat / se0, 0.0)
        return b_hat, z > stats.norm.ppf(1 - a1)


@dataclass
class Survival:
    """Time-to-event endpoint analysed with a log-rank test (Schoenfeld approximation).

    event_prob: probability that an evaluable patient has an event by the final
        analysis (use :func:`event_probability` to derive it from medians,
        accrual and follow-up). Events = evaluable N x event_prob.
    higher_is_better: False (default) when HR < 1 favours treatment.
    """

    event_prob: float
    higher_is_better: bool = False
    kind: str = "survival"
    effect_label: str = "Hazard ratio (T vs C)"

    def to_benefit(self, effect):
        lhr = np.log(np.asarray(effect, dtype=float))
        return lhr if self.higher_is_better else -lhr

    def from_benefit(self, b):
        b = np.asarray(b, dtype=float)
        return np.exp(b if self.higher_is_better else -b)

    def events(self, nt, nc):
        return (nt + nc) * self.event_prob

    def se(self, b, nt, nc):
        p = nt / (nt + nc)
        d = self.events(nt, nc)
        return np.ones_like(np.asarray(b, dtype=float)) / np.sqrt(d * p * (1 - p))

    def critical_benefit(self, nt, nc, a1):
        return stats.norm.ppf(1 - a1) * self.se(0.0, nt, nc)

    def power(self, b, nt, nc, a1):
        b = np.asarray(b, dtype=float)
        return stats.norm.cdf(b / self.se(b, nt, nc) - stats.norm.ppf(1 - a1))

    def simulate(self, b, nt, nc, a1, rng):
        b = np.asarray(b, dtype=float)
        se = self.se(b, nt, nc)
        b_hat = b + se * rng.standard_normal(b.shape)
        return b_hat, b_hat / se > stats.norm.ppf(1 - a1)


def event_probability(median_control, hr, accrual, min_followup, ratio=1.0):
    """Probability a patient has an event by the final analysis.

    Exponential survival, uniform accrual over ``accrual`` time units, analysis
    at ``accrual + min_followup``. Averaged over arms by allocation ``ratio``
    (treatment:control). All times in the same unit as ``median_control``.
    """
    lam_c = np.log(2) / median_control
    out = []
    for lam in (lam_c * hr, lam_c):
        if accrual <= 0:
            out.append(1 - np.exp(-lam * min_followup))
        else:
            out.append(1 - (np.exp(-lam * min_followup) - np.exp(-lam * (min_followup + accrual))) / (lam * accrual))
    w = ratio / (1 + ratio)
    return float(w * out[0] + (1 - w) * out[1])


def make_endpoint(kind: str, **kw):
    kind = kind.lower()
    if kind in ("continuous", "cont", "mean"):
        return Continuous(sd=kw["sd"], higher_is_better=kw.get("higher_is_better", True))
    if kind in ("binary", "bin", "proportion", "responder"):
        return Binary(p_control=kw["p_control"], higher_is_better=kw.get("higher_is_better", True))
    if kind in ("survival", "tte", "time-to-event"):
        return Survival(event_prob=kw["event_prob"], higher_is_better=kw.get("higher_is_better", False))
    raise ValueError(f"unknown endpoint kind: {kind!r}")
