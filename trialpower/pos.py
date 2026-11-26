"""Probability of success (PoS / assurance) for a Phase III programme.

Classical power answers "if the true effect is exactly X, how often do we win?".
Probability of success averages that over our *uncertainty* about X (from
Phase II), which is what a portfolio / go-no-go decision actually needs:

    PoS = E_prior[ P(trial succeeds | true effect) ]

Two independent estimators are provided:

* :func:`assurance` – deterministic numerical integration (Gauss–Hermite
  quadrature) of the exact power function over the prior. Fast; use for curves.
* :func:`simulate_pos` – Monte Carlo: draw a true effect from the prior, simulate
  the Phase III trial(s) at the summary-statistic level, apply the success rule,
  repeat. Gives the full joint distribution of true vs observed effect, so you
  can also ask "how often is a win a *real* win?".

Success rule (per trial): statistically significant in the favourable direction
AND (optionally) the observed effect (on the analysis scale, i.e. after any
dropout dilution) is at least the minimum clinically meaningful effect ``mcid`` (e.g. the target-product-profile threshold). With
``n_trials > 1`` (e.g. two pivotal trials for FDA) *all* trials must succeed;
they share the same true effect, so they are correlated — PoS(2 trials) is not
PoS(1)^2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .design import TrialDesign
from .phase2 import EffectPrior


def _mcid_b(design, mcid):
    return None if mcid is None else float(design.endpoint.to_benefit(mcid))


def success_threshold(design: TrialDesign, mcid=None, n_total=None) -> float:
    """Minimum *observed* effect (natural units) that counts as a success:
    the stricter of the significance boundary and the MCID."""
    nt, nc = design.arms(n_total)
    ep = design.analysis_endpoint
    c = float(ep.critical_benefit(nt, nc, design.a1))
    m = _mcid_b(design, mcid)
    if m is not None:
        c = max(c, m)
    return float(ep.from_benefit(c))


def prob_success_given_effect(design: TrialDesign, b, mcid=None, n_total=None):
    """P(one trial succeeds | true benefit b) — vectorised over b."""
    nt, nc = design.arms(n_total)
    ep = design.analysis_endpoint
    b = design.analysis_benefit(b)
    p_sig = ep.power(b, nt, nc, design.a1)
    m = _mcid_b(design, mcid)
    if m is None or m <= float(ep.critical_benefit(nt, nc, design.a1)):
        return p_sig
    # MCID is the binding hurdle: P(observed benefit >= mcid) (normal approximation)
    return np.minimum(p_sig, stats.norm.cdf((b - m) / ep.se(b, nt, nc)))


def assurance(design: TrialDesign, prior: EffectPrior, mcid=None, n_trials=1, n_total=None) -> float:
    """PoS by numerical integration of P(success | effect)^n_trials over the prior."""
    nodes, w = prior.quadrature()
    p = prob_success_given_effect(design, nodes, mcid, n_total)
    return float(np.sum(w * p ** n_trials))


@dataclass
class PoSResult:
    n_sims: int
    n_trials: int
    pos: float  # all trials significant AND meet MCID (if given)
    pos_sig_only: float  # all trials significant (ignores MCID)
    pos_any: float  # at least one trial succeeds
    pos_assurance: float  # quadrature cross-check
    prob_true_benefit: float  # P(true effect favours treatment)
    prob_true_mcid: float | None  # P(true effect >= MCID)
    ppv_benefit: float  # P(true effect favours treatment | success)
    ppv_mcid: float | None  # P(true effect >= MCID | success)
    success_threshold: float  # min observed effect that wins (natural units)
    critical_effect: float  # min observed effect that is stat. sig (natural units)
    true_effect: np.ndarray = field(repr=False)  # natural units
    observed_effect: np.ndarray = field(repr=False)  # trial 1, natural units
    success: np.ndarray = field(repr=False)  # programme success per sim

    @property
    def mc_se(self) -> float:
        return float(np.sqrt(self.pos * (1 - self.pos) / self.n_sims))

    def summary(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "pos", "pos_sig_only", "pos_any", "pos_assurance", "prob_true_benefit",
            "prob_true_mcid", "ppv_benefit", "ppv_mcid", "success_threshold", "critical_effect")}
        d["mc_se"] = self.mc_se
        return d


def simulate_pos(design: TrialDesign, prior: EffectPrior, mcid=None, n_trials=1,
                 n_sims=100_000, seed=None, n_total=None) -> PoSResult:
    """Monte Carlo probability of success."""
    rng = np.random.default_rng(seed)
    ep = design.analysis_endpoint
    nt, nc = design.arms(n_total)
    m = _mcid_b(design, mcid)

    b_true = prior.sample(n_sims, rng)
    all_sig = np.ones(n_sims, bool)
    all_ok = np.ones(n_sims, bool)
    any_ok = np.zeros(n_sims, bool)
    b_hat_first = None
    for i in range(n_trials):
        b_hat, sig = ep.simulate(design.analysis_benefit(b_true), nt, nc, design.a1, rng)
        ok = sig if m is None else sig & (b_hat >= m)
        all_sig &= sig
        all_ok &= ok
        any_ok |= ok
        if i == 0:
            b_hat_first = b_hat

    n_ok = all_ok.sum()
    ppv_b = float((b_true[all_ok] > 0).mean()) if n_ok else float("nan")
    ppv_m = None if m is None else (float((b_true[all_ok] >= m).mean()) if n_ok else float("nan"))
    nt_, nc_ = nt, nc
    return PoSResult(
        n_sims=n_sims,
        n_trials=n_trials,
        pos=float(all_ok.mean()),
        pos_sig_only=float(all_sig.mean()),
        pos_any=float(any_ok.mean()),
        pos_assurance=assurance(design, prior, mcid, n_trials, n_total),
        prob_true_benefit=prior.prob_above(0.0),
        prob_true_mcid=None if m is None else prior.prob_above(m),
        ppv_benefit=ppv_b,
        ppv_mcid=ppv_m,
        success_threshold=success_threshold(design, mcid, n_total),
        critical_effect=float(ep.from_benefit(ep.critical_benefit(nt_, nc_, design.a1))),
        true_effect=design.endpoint.from_benefit(b_true),
        observed_effect=ep.from_benefit(b_hat_first),
        success=all_ok,
    )


def pos_curve(design: TrialDesign, prior: EffectPrior, n_grid, mcid=None, n_trials=1):
    """PoS (assurance) and classical power at the prior mean, across sample sizes.

    The gap between the two is the price of uncertainty: power -> 100% as N grows,
    but PoS plateaus at P(true effect clears the success hurdle).
    """
    b_mean = prior.mean_benefit
    rows = []
    for n in n_grid:
        rows.append({
            "n_total": float(n),
            "pos": assurance(design, prior, mcid, n_trials, n),
            "power_at_prior_mean": float(prob_success_given_effect(design, b_mean, mcid, n)) ** n_trials,
        })
    return rows
