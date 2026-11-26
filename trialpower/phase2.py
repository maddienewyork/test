"""Turning Phase II results into a prior for the Phase III treatment effect.

Phase II estimates are typically *optimistic* for Phase III: programmes only
advance when Phase II looked good (selection bias / "winner's curse"), Phase III
populations are broader, and endpoints or doses can differ. The prior below
exposes three standard levers for this:

* ``haircut`` – shrink the Phase II point estimate towards zero
  (benefit x (1 - haircut)). 10-40% is common in practice.
* ``weight`` – power-prior / effective-sample-size discount in (0, 1].
  The prior SD is ``se / sqrt(weight)``; weight=0.5 treats Phase II as if it
  had half its patients.
* ``prob_effect`` – probability the drug truly works. With probability
  ``1 - prob_effect`` the true effect is exactly zero (mixture prior). Useful
  to encode base-rate scepticism (e.g. new mechanism, surrogate endpoint).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Phase2Evidence:
    """Phase II effect estimate in natural units and its standard error on the benefit scale
    (for hazard ratios: the SE of log HR)."""

    estimate: float
    se: float
    label: str = ""

    @classmethod
    def from_means(cls, mean_t, mean_c, sd, n_t, n_c):
        return cls(mean_t - mean_c, sd * math.sqrt(1 / n_t + 1 / n_c), "means")

    @classmethod
    def from_proportions(cls, x_t, n_t, x_c, n_c):
        pt, pc = x_t / n_t, x_c / n_c
        return cls(pt - pc, math.sqrt(pt * (1 - pt) / n_t + pc * (1 - pc) / n_c), "proportions")

    @classmethod
    def from_hr_ci(cls, hr, lower, upper, level=0.95):
        z = stats.norm.ppf(0.5 + level / 2)
        return cls(hr, (math.log(upper) - math.log(lower)) / (2 * z), "hr_ci")

    @classmethod
    def from_hr_events(cls, hr, events, ratio=1.0):
        p = ratio / (1 + ratio)
        return cls(hr, 1 / math.sqrt(events * p * (1 - p)), "hr_events")

    @classmethod
    def from_config(cls, cfg: dict):
        """Build from a dict with one of the supported key sets."""
        c = dict(cfg)
        if "estimate" in c and "se" in c:
            return cls(float(c["estimate"]), float(c["se"]), "estimate_se")
        if {"mean_t", "mean_c", "sd", "n_t", "n_c"} <= c.keys():
            return cls.from_means(c["mean_t"], c["mean_c"], c["sd"], c["n_t"], c["n_c"])
        if {"x_t", "n_t", "x_c", "n_c"} <= c.keys():
            return cls.from_proportions(c["x_t"], c["n_t"], c["x_c"], c["n_c"])
        if {"hr", "lower", "upper"} <= c.keys():
            return cls.from_hr_ci(c["hr"], c["lower"], c["upper"], c.get("level", 0.95))
        if {"hr", "events"} <= c.keys():
            return cls.from_hr_events(c["hr"], c["events"], c.get("ratio", 1.0))
        raise ValueError(f"cannot interpret phase2 block: {sorted(c)}")


@dataclass
class EffectPrior:
    """Prior on the true Phase III benefit: mixture of N(mean, sd^2) (w.p. prob_effect) and 0."""

    mean_benefit: float
    sd_benefit: float
    prob_effect: float = 1.0

    @classmethod
    def from_phase2(cls, endpoint, evidence: Phase2Evidence, haircut=0.0, weight=1.0, prob_effect=1.0):
        if not 0 < weight <= 1:
            raise ValueError("weight must be in (0, 1]")
        m = float(endpoint.to_benefit(evidence.estimate)) * (1 - haircut)
        return cls(m, evidence.se / math.sqrt(weight), prob_effect)

    @classmethod
    def normal(cls, endpoint, mean_effect, sd_benefit, prob_effect=1.0):
        return cls(float(endpoint.to_benefit(mean_effect)), float(sd_benefit), prob_effect)

    @classmethod
    def fixed(cls, endpoint, effect):
        return cls(float(endpoint.to_benefit(effect)), 0.0, 1.0)

    def sample(self, n, rng):
        b = self.mean_benefit + self.sd_benefit * rng.standard_normal(n)
        if self.prob_effect < 1:
            b = np.where(rng.random(n) < self.prob_effect, b, 0.0)
        return b

    def prob_above(self, b0) -> float:
        """P(true benefit > b0)."""
        if self.sd_benefit > 0:
            p = stats.norm.sf((b0 - self.mean_benefit) / self.sd_benefit)
        else:
            p = float(self.mean_benefit > b0)
        return float(self.prob_effect * p + (1 - self.prob_effect) * (0.0 > b0))

    def quadrature(self, n_nodes=101):
        """Nodes/weights (benefit scale) for E[f(b)] under the prior."""
        if self.sd_benefit > 0:
            x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
            nodes, weights = self.mean_benefit + self.sd_benefit * x, w / w.sum()
        else:
            nodes, weights = np.array([self.mean_benefit]), np.array([1.0])
        weights = weights * self.prob_effect
        if self.prob_effect < 1:
            nodes = np.append(nodes, 0.0)
            weights = np.append(weights, 1 - self.prob_effect)
        return nodes, weights
