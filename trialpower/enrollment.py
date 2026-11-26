"""Enrollment (recruitment) curve simulation.

Model: the Poisson–gamma recruitment model widely used in industry
(Anisimov & Fedorov, 2007):

* Sites activate uniformly at random between ``activation_start`` and
  ``activation_end`` (months), then ramp linearly to full speed over
  ``site_ramp`` months.
* Each site recruits as a Poisson process. Its full-speed rate is
  ``mean_rate x relative_rate``, with ``relative_rate ~ Gamma(mean 1, CV site_cv)``,
  because some sites are stars and many are duds.
* The programme-wide ``mean_rate`` is itself uncertain:
  ``Gamma(mean rate_per_site, CV rate_uncertainty)``. Planning rates are guesses,
  and this term is usually the biggest driver of late finishes.
* **Re-forecast:** given ``enrolled`` patients after ``elapsed`` months, the
  mean-rate prior is updated with the observed data, which is conjugate:
  ``Gamma(a0 + enrolled, b0 + site-months of exposure)``. The rest of the
  trial is then simulated from today.

Optionally, for time-to-event trials, an :class:`EventModel` simulates event
accrual (exponential event and dropout times) to predict when the target
number of events, and so the final analysis, is reached.

All times are in months.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@dataclass
class EnrollmentPlan:
    target_n: int  # patients to randomise
    n_sites: int
    rate_per_site: float  # randomised patients / site / month at full speed
    activation_start: float = 0.0  # month first site opens
    activation_end: float = 6.0  # month last site opens
    site_ramp: float = 0.0  # months for a new site to reach full speed
    site_cv: float = 0.8  # between-site variability of rates (0 = identical sites)
    rate_uncertainty: float = 0.2  # CV of the overall mean rate (0 = rate known exactly)
    screen_fail: float = 0.0  # fraction of screened patients who fail screening

    # ---- deterministic helpers ------------------------------------------
    def _exposure(self, u):
        """Effective full-speed months a site has contributed u months after activation."""
        u = np.maximum(np.asarray(u, dtype=float), 0.0)
        r = self.site_ramp
        if r <= 0:
            return u
        return np.where(u < r, u ** 2 / (2 * r), u - r / 2)

    def _activation_nodes(self, k=256):
        if self.n_sites == 1 or self.activation_end <= self.activation_start:
            return np.array([self.activation_start])
        return self.activation_start + (np.arange(k) + 0.5) / k * (self.activation_end - self.activation_start)

    def site_months(self, t):
        """Expected full-speed site-months of exposure accumulated by time t (all sites)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        a = self._activation_nodes()
        return self.n_sites * self._exposure(t[:, None] - a[None, :]).mean(axis=1)

    def expected_curve(self, t, rate=None):
        """Planned (average-rate) cumulative enrollment at times t, capped at target."""
        rate = self.rate_per_site if rate is None else rate
        return np.minimum(rate * self.site_months(t), self.target_n)

    def expected_completion(self, rate=None, start_count=0, start_time=0.0):
        rate = self.rate_per_site if rate is None else rate
        need = self.target_n - start_count
        if need <= 0:
            return start_time
        base = rate * self.site_months(start_time)[0]
        lo, hi = start_time, max(start_time, self.activation_end) + 1.0
        while rate * self.site_months(hi)[0] - base < need:
            hi *= 2
            if hi > 1e5:
                return math.inf
        for _ in range(60):
            mid = (lo + hi) / 2
            if rate * self.site_months(mid)[0] - base < need:
                lo = mid
            else:
                hi = mid
        return hi

    @property
    def screened_needed(self):
        return math.ceil(self.target_n / (1 - self.screen_fail)) if self.screen_fail < 1 else math.inf


@dataclass
class EventModel:
    """Exponential event and dropout model for time-to-event trials."""

    median_control: float  # months
    hr: float  # assumed true HR (treatment vs control)
    target_events: int
    ratio: float = 1.0  # treatment:control
    dropout_per_year: float = 0.05  # annual probability of loss to follow-up


@dataclass
class EnrollmentResult:
    plan: EnrollmentPlan
    grid: np.ndarray  # months
    enroll_q: dict  # quantile -> cumulative enrolled on grid
    enroll_mean: np.ndarray
    completion: np.ndarray  # months to reach target, per simulation (inf = not within horizon)
    expected: np.ndarray  # planned curve on grid (using the [posterior] mean rate)
    mean_rate: float  # mean of the (posterior) site rate used
    rate_cv: float
    elapsed: float = 0.0
    enrolled: int = 0
    events_q: dict | None = None
    events_mean: np.ndarray | None = None
    event_completion: np.ndarray | None = field(default=None, repr=False)

    @staticmethod
    def _summ(x):
        f = x[np.isfinite(x)]
        out = {"p_within_horizon": float(np.isfinite(x).mean())}
        for q in (0.1, 0.5, 0.9):
            out[f"p{int(q * 100)}"] = float(np.quantile(x, q)) if len(f) else math.inf
        out["mean"] = float(f.mean()) if len(f) else math.inf
        return out

    def summary(self) -> dict:
        d = {"lpi_" + k: v for k, v in self._summ(self.completion).items()}
        if self.event_completion is not None:
            d.update({"analysis_" + k: v for k, v in self._summ(self.event_completion).items()})
        return d

    def prob_complete_by(self, month, events=False) -> float:
        x = self.event_completion if events else self.completion
        return float((x <= month).mean())

    def monthly_table(self):
        """One row per whole month: enrolment (and event) quantiles."""
        rows = []
        for m in range(int(math.ceil(self.grid[0])), int(math.floor(self.grid[-1])) + 1):
            i = int(np.argmin(np.abs(self.grid - m)))
            row = {"month": m, "planned": float(self.expected[i]), "mean": float(self.enroll_mean[i])}
            row.update({f"p{int(q * 100)}": float(self.enroll_q[q][i]) for q in (0.1, 0.5, 0.9)})
            row["p_lpi_by_month"] = self.prob_complete_by(m)
            if self.events_q is not None:
                row.update({f"events_p{int(q * 100)}": float(self.events_q[q][i]) for q in (0.1, 0.5, 0.9)})
                row["p_analysis_by_month"] = self.prob_complete_by(m, events=True)
            rows.append(row)
        return rows


def _gamma_ab(mean, cv):
    a = 1.0 / cv ** 2
    return a, a / mean  # shape, rate


def posterior_rate(plan: EnrollmentPlan, elapsed: float, enrolled: int):
    """Update the mean site rate with observed enrolment. Returns (mean, cv)."""
    exposure = float(plan.site_months(elapsed)[0])
    if plan.rate_uncertainty <= 0:
        return plan.rate_per_site, 0.0
    a0, b0 = _gamma_ab(plan.rate_per_site, plan.rate_uncertainty)
    a, b = a0 + enrolled, b0 + exposure
    return a / b, 1 / math.sqrt(a)


def simulate_enrollment(plan: EnrollmentPlan, n_sims=2000, seed=None, elapsed=0.0, enrolled=0,
                        events: EventModel | None = None, horizon=None, dt=0.1) -> EnrollmentResult:
    """Monte Carlo enrollment (and optional event) curves.

    elapsed / enrolled: for a re-forecast of an ongoing trial (months since the
    first site could open, patients randomised so far).
    """
    rng = np.random.default_rng(seed)
    if elapsed > 0:
        mean_rate, rate_cv = posterior_rate(plan, elapsed, enrolled)
    else:
        mean_rate, rate_cv = plan.rate_per_site, plan.rate_uncertainty
    need = plan.target_n - enrolled

    if horizon is None:
        t_exp = plan.expected_completion(mean_rate, enrolled, elapsed)
        span = (t_exp - elapsed) if np.isfinite(t_exp) else 120
        horizon = elapsed + max(12.0, 3.0 * span)
        if events is not None:
            horizon += 4 * events.median_control
        horizon = min(horizon, elapsed + 600)
    grid = np.arange(elapsed, horizon + dt / 2, dt)
    nb = len(grid)

    cum_all = np.empty((n_sims, nb))
    ev_all = np.empty((n_sims, nb)) if events is not None else None
    completion = np.full(n_sims, np.inf)
    ev_completion = np.full(n_sims, np.inf) if events is not None else None

    chunk = max(1, int(4e6 // (plan.n_sites * nb)))
    for s0 in range(0, n_sims, chunk):
        k = min(chunk, n_sims - s0)
        mu = rng.gamma(*_ab_to_np(mean_rate, rate_cv), size=k) if rate_cv > 0 else np.full(k, mean_rate)
        rel = rng.gamma(1 / plan.site_cv ** 2, plan.site_cv ** 2, (k, plan.n_sites)) if plan.site_cv > 0 \
            else np.ones((k, plan.n_sites))
        if plan.n_sites == 1 or plan.activation_end <= plan.activation_start:
            a = np.full((k, plan.n_sites), plan.activation_start)
        else:
            a = rng.uniform(plan.activation_start, plan.activation_end, (k, plan.n_sites))
        # cumulative intensity on the grid, summed over sites (sum of Poissons is Poisson)
        lam = np.einsum("ks,ksb->kb", rel, plan._exposure(grid[None, None, :] - a[:, :, None])) * mu[:, None]
        inc = rng.poisson(np.diff(lam, axis=1, prepend=lam[:, :1]))
        cum = enrolled + np.cumsum(inc, axis=1)
        hit = cum >= plan.target_n
        idx = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        completion[s0:s0 + k] = np.where(idx >= 0, grid[np.maximum(idx, 0)], np.inf)
        cum_all[s0:s0 + k] = np.minimum(cum, plan.target_n)

        if events is not None:
            for j in range(k):
                ev_all[s0 + j], ev_completion[s0 + j] = _simulate_events(
                    rng, grid, inc[j], need, enrolled, elapsed, plan.activation_start, events, dt)

    res = EnrollmentResult(
        plan=plan, grid=grid,
        enroll_q={q: np.quantile(cum_all, q, axis=0) for q in QUANTILES},
        enroll_mean=cum_all.mean(axis=0), completion=completion,
        expected=np.minimum(enrolled + mean_rate * (plan.site_months(grid) - plan.site_months(elapsed)[0]),
                            plan.target_n),
        mean_rate=mean_rate, rate_cv=rate_cv, elapsed=elapsed, enrolled=enrolled,
    )
    if events is not None:
        res.events_q = {q: np.quantile(ev_all, q, axis=0) for q in QUANTILES}
        res.events_mean = ev_all.mean(axis=0)
        res.event_completion = ev_completion
    return res


def _ab_to_np(mean, cv):
    a, b = _gamma_ab(mean, cv)
    return a, 1 / b  # numpy uses (shape, scale)


def _simulate_events(rng, grid, inc, need, enrolled, elapsed, start, ev: EventModel, dt):
    counts = inc.copy()
    # stop enrolling once the target is hit
    cs = np.cumsum(counts)
    over = cs > need
    if over.any():
        i = over.argmax()
        counts[i] -= cs[i] - need
        counts[i + 1:] = 0
    t_new = np.repeat(grid, counts) - rng.uniform(0, dt, counts.sum())  # inc[b] arrived in (grid[b]-dt, grid[b]]
    t_old = rng.uniform(start, max(elapsed, start + 1e-9), enrolled) if enrolled > 0 else np.empty(0)
    t_enr = np.concatenate([t_old, np.maximum(t_new, elapsed)])
    n = len(t_enr)
    treat = rng.random(n) < ev.ratio / (1 + ev.ratio)
    lam = math.log(2) / ev.median_control * np.where(treat, ev.hr, 1.0)
    t_event = rng.exponential(1 / lam)
    if ev.dropout_per_year > 0:
        delta = -math.log(1 - ev.dropout_per_year) / 12
        observed = t_event < rng.exponential(1 / delta, n)
    else:
        observed = np.ones(n, bool)
    cal = np.sort(t_enr[observed] + t_event[observed])
    curve = np.searchsorted(cal, grid, side="right").astype(float)
    done = cal[ev.target_events - 1] if len(cal) >= ev.target_events else np.inf
    return curve, (done if done <= grid[-1] else np.inf)


def sites_needed(plan: EnrollmentPlan, deadline: float, prob=0.8, n_sims=1000, seed=1, max_sites=5000):
    """Smallest number of sites so that P(last patient in by ``deadline``) >= prob.

    Returns None if even ``max_sites`` is not enough (e.g. activation ends too late).
    """
    def p(ns):
        r = simulate_enrollment(replace(plan, n_sites=ns), n_sims, seed, horizon=deadline + 0.2)
        return r.prob_complete_by(deadline)

    if p(max_sites) < prob:
        return None
    lo, hi = 0, 1
    while p(hi) < prob:
        lo, hi = hi, min(hi * 2, max_sites)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p(mid) >= prob:
            hi = mid
        else:
            lo = mid
    return hi
