"""Frequentist (classical) power and sample-size calculations for one trial."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import optimize

from .endpoints import Binary, Continuous, Survival

#: How patients who drop out are handled in the primary analysis.
DROPOUT_HANDLING = {
    "exclude": "Excluded (complete-case): fewer evaluable patients, effect unchanged",
    "dilute": "ITT, dropouts lose the drug effect: everyone analysed, effect diluted by (1 - dropout_T)",
    "nri": "ITT, dropouts count as treatment failures (non-responder imputation, binary only)",
}


@dataclass
class TrialDesign:
    """A single two-arm superiority trial.

    endpoint: a Continuous / Binary / Survival endpoint.
    n_total: total patients *enrolled* (both arms).
    alpha: overall type-I error. With ``sides=2`` the test is two-sided at
        ``alpha`` (i.e. alpha/2 in the favourable direction).
    ratio: allocation treatment:control (2.0 means 2:1).
    dropout: fraction of enrolled patients who drop out (both arms).
    dropout_t / dropout_c: per-arm dropout; override ``dropout`` when given
        (treatment-arm dropout is often higher because of adverse events).
    dropout_handling: one of :data:`DROPOUT_HANDLING`:

        * ``"exclude"`` – dropouts are not analysed. Evaluable N per arm is
          N_arm x (1 - dropout_arm); the effect among completers is unchanged.
          For time-to-event this is censoring: fewer events.
        * ``"dilute"`` – treatment-policy / ITT: all randomised patients are
          analysed, but treatment-arm dropouts get no drug benefit, so the true
          effect is scaled by (1 - dropout_t). Binary: pT' = (1-d)pT + d*pC.
        * ``"nri"`` – binary only: dropouts in *both* arms are counted as
          failures (non-responders if higher is better).
    """

    endpoint: object
    n_total: float
    alpha: float = 0.05
    sides: int = 2
    ratio: float = 1.0
    dropout: float = 0.0
    dropout_t: float | None = None
    dropout_c: float | None = None
    dropout_handling: str = "exclude"

    def __post_init__(self):
        if self.dropout_handling not in DROPOUT_HANDLING:
            raise ValueError(f"dropout_handling must be one of {list(DROPOUT_HANDLING)}")
        if self.dropout_handling == "nri" and not isinstance(self.endpoint, Binary):
            raise ValueError("non-responder imputation ('nri') only applies to binary endpoints")

    # ---- helpers -------------------------------------------------------
    @property
    def a1(self) -> float:
        """One-sided alpha in the favourable direction."""
        return self.alpha / self.sides

    @property
    def d_t(self) -> float:
        return self.dropout if self.dropout_t is None else self.dropout_t

    @property
    def d_c(self) -> float:
        return self.dropout if self.dropout_c is None else self.dropout_c

    def enrolled_arms(self, n_total=None):
        n = self.n_total if n_total is None else n_total
        return n * self.ratio / (1 + self.ratio), n / (1 + self.ratio)

    def arms(self, n_total=None):
        """Analysed patients per arm (treatment, control)."""
        nt, nc = self.enrolled_arms(n_total)
        if self.dropout_handling == "exclude":
            return nt * (1 - self.d_t), nc * (1 - self.d_c)
        return nt, nc

    @property
    def analysis_endpoint(self):
        """Endpoint as seen by the analysis (NRI shifts the control rate)."""
        if self.dropout_handling == "nri":
            ep = self.endpoint
            return replace(ep, p_control=self._fail(ep.p_control, self.d_c))
        return self.endpoint

    def _fail(self, p, d):
        # dropouts counted as failures: non-responders if higher is better, events otherwise
        return p * (1 - d) if self.endpoint.higher_is_better else p + (1 - p) * d

    def analysis_benefit(self, b):
        """Map the true (on-treatment) benefit to the benefit the analysis estimates."""
        b = np.asarray(b, dtype=float)
        if self.dropout_handling == "dilute":
            return b * (1 - self.d_t)
        if self.dropout_handling == "nri":
            ep = self.endpoint
            pt = self._fail(ep.p_treat(b), self.d_t)
            pc = self._fail(ep.p_control, self.d_c)
            return ep.to_benefit(pt - pc)
        return b

    def _b(self, effect):
        return float(self.endpoint.to_benefit(effect))

    # ---- core quantities ----------------------------------------------
    def power(self, effect, n_total=None) -> float:
        """Power to show a statistically significant benefit if the true effect is ``effect``."""
        return float(self.power_benefit(self._b(effect), n_total))

    def power_benefit(self, b, n_total=None):
        nt, nc = self.arms(n_total)
        return self.analysis_endpoint.power(self.analysis_benefit(b), nt, nc, self.a1)

    def critical_effect(self, n_total=None) -> float:
        """Smallest *observed* effect (on the analysis scale) that would be statistically significant."""
        nt, nc = self.arms(n_total)
        ep = self.analysis_endpoint
        return float(ep.from_benefit(ep.critical_benefit(nt, nc, self.a1)))

    def events(self, n_total=None):
        if isinstance(self.endpoint, Survival):
            return float(self.endpoint.events(*self.arms(n_total)))
        return None

    def sample_size(self, effect, target_power=0.8) -> dict:
        """Enrolled N needed to reach ``target_power`` for true ``effect``."""
        inf = {"n_total": math.inf, "n_t": math.inf, "n_c": math.inf, "events": None, "power": float("nan")}
        b = self._b(effect)
        if not np.isfinite(b) or b <= 0:
            return inf

        def f(n):
            return self.power_benefit(b, n) - target_power

        lo = 4.0 / max(1e-6, 1 - max(self.d_t, self.d_c))
        hi = lo
        while f(hi) < 0:
            hi *= 2
            if hi > 1e8:
                return inf
        n_req = optimize.brentq(f, lo, hi, xtol=1e-6) if f(lo) < 0 else lo
        nt_e, nc_e = self.enrolled_arms(n_req)
        n_t, n_c = math.ceil(nt_e - 1e-9), math.ceil(nc_e - 1e-9)
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


# ---------------------------------------------------------------------------
# Power heatmaps
# ---------------------------------------------------------------------------
#: Parameters that can go on a heatmap axis.
GRID_PARAMS = {
    "effect": "True treatment effect",
    "treat_outcome": "Drug-arm outcome",
    "control_outcome": "Control-arm outcome",
    "n_total": "Total N enrolled",
    "dropout": "Dropout (both arms)",
    "dropout_t": "Dropout, drug arm",
    "alpha": "Alpha",
    "ratio": "Allocation ratio T:C",
    "sd": "Outcome SD",
    "p_control": "Control response rate",
    "event_prob": "P(event)",
}


def power_grid(design: TrialDesign, effect, x, y, control_outcome=None):
    """Power over a 2-D grid of assumptions.

    x, y: ``(param, values)`` with ``param`` from :data:`GRID_PARAMS`.
    effect: true effect (natural units) used when neither axis sets it.
    control_outcome: baseline control-arm outcome for the ``treat_outcome`` /
        ``control_outcome`` axes. Arm outcomes are: mean (continuous), response
        rate (binary), median survival time (time-to-event, HR = median_C / median_T
        under exponential survival). Defaults to 0 / p_control / 12.

    Returns an array of shape (len(y_values), len(x_values)).
    """
    ep0 = design.endpoint
    if control_outcome is None:
        control_outcome = {Continuous: 0.0, Binary: getattr(ep0, "p_control", None), Survival: 12.0}[type(ep0)]
    xp, xv = x
    yp, yv = y
    out = np.full((len(yv), len(xv)), np.nan)
    for i, yval in enumerate(yv):
        for j, xval in enumerate(xv):
            params = {xp: xval, yp: yval}
            ep_kw, des_kw = {}, {}
            eff = params.get("effect", effect)
            if "treat_outcome" in params or "control_outcome" in params:
                t = params.get("treat_outcome")
                c = params.get("control_outcome", control_outcome)
                if isinstance(ep0, Binary):
                    ep_kw["p_control"] = c
                if t is None:  # only control varies: keep the effect fixed
                    pass
                elif isinstance(ep0, Survival):
                    eff = c / t
                else:
                    eff = t - c
            for k in ("sd", "p_control", "event_prob"):
                if k in params:
                    ep_kw[k] = params[k]
            for k in ("n_total", "dropout", "dropout_t", "alpha", "ratio"):
                if k in params:
                    des_kw[k] = params[k]
            try:
                ep = replace(ep0, **ep_kw) if ep_kw else ep0
                d = replace(design, endpoint=ep, **des_kw)
                b = ep.to_benefit(eff)
                if isinstance(ep, Binary):
                    pt = ep.p_control + float(ep.from_benefit(b))
                    if not (0 < ep.p_control < 1 and 0 <= pt <= 1):
                        continue
                out[i, j] = float(d.power_benefit(b))
            except (TypeError, ValueError):
                continue
    return out
