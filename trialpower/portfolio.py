"""Evaluate many trials from a flat, spreadsheet-like spec (YAML / CSV / app table)."""
from __future__ import annotations

import math

import numpy as np

from .design import TrialDesign
from .endpoints import make_endpoint
from .phase2 import EffectPrior, Phase2Evidence
from .pos import simulate_pos

# Flat column schema shared by the YAML loader, the CLI and the app's table editor.
COLUMNS = [
    "name", "endpoint", "higher_is_better", "sd", "p_control", "event_prob",
    "alpha", "sides", "n_total", "ratio", "dropout", "dropout_t", "dropout_c", "dropout_handling",
    "phase2_estimate", "phase2_se", "haircut", "weight", "prob_effect",
    "assumed_effect", "mcid", "n_trials", "target_power",
]

DEFAULTS = {
    "alpha": 0.05, "sides": 2, "ratio": 1.0, "dropout": 0.0, "dropout_handling": "exclude",
    "haircut": 0.0, "weight": 1.0, "prob_effect": 1.0,
    "n_trials": 1, "target_power": 0.9, "assumed_effect": None, "mcid": None,
}


def _blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() == "")


def normalise(trial: dict, defaults: dict | None = None) -> dict:
    """Merge defaults and expand a nested ``phase2:`` block into phase2_estimate / phase2_se."""
    row = {**DEFAULTS, **(defaults or {})}
    row.update({k: v for k, v in trial.items() if not _blank(v)})
    if "phase2" in row and isinstance(row["phase2"], dict):
        ev = Phase2Evidence.from_config(row.pop("phase2"))
        row["phase2_estimate"], row["phase2_se"] = ev.estimate, ev.se
    if _blank(row.get("higher_is_better")):
        row["higher_is_better"] = str(row["endpoint"]).lower() not in ("survival", "tte", "time-to-event")
    return {c: (None if _blank(row.get(c)) else row.get(c)) for c in COLUMNS}


def build(row: dict):
    ep = make_endpoint(
        row["endpoint"], sd=row.get("sd"), p_control=row.get("p_control"),
        event_prob=row.get("event_prob"), higher_is_better=bool(row["higher_is_better"]),
    )
    opt = {k: (None if _blank(row.get(k)) else float(row[k])) for k in ("dropout_t", "dropout_c")}
    design = TrialDesign(ep, float(row["n_total"]), float(row["alpha"]), int(row["sides"]),
                         float(row["ratio"]), float(row["dropout"]), **opt,
                         dropout_handling=str(row.get("dropout_handling") or "exclude"))
    prior = EffectPrior.from_phase2(
        ep, Phase2Evidence(float(row["phase2_estimate"]), float(row["phase2_se"])),
        haircut=float(row["haircut"]), weight=float(row["weight"]), prob_effect=float(row["prob_effect"]))
    return design, prior


def evaluate(row: dict, n_sims=50_000, seed=1) -> dict:
    """All headline numbers for one trial."""
    design, prior = build(row)
    ep = design.endpoint
    assumed = row.get("assumed_effect")
    if _blank(assumed):
        assumed = float(ep.from_benefit(prior.mean_benefit))
    tp = float(row["target_power"])
    mcid = None if _blank(row.get("mcid")) else float(row["mcid"])
    ss = design.sample_size(assumed, tp)
    res = simulate_pos(design, prior, mcid, int(row["n_trials"]), n_sims, seed)
    return {
        "name": row.get("name") or "",
        "endpoint": ep.kind,
        "n_total": design.n_total,
        "events": design.events(),
        "phase2_effect": row["phase2_estimate"],
        "assumed_effect": assumed,
        "power": design.power(assumed),
        f"n_for_{int(round(tp * 100))}pct": ss["n_total"],
        f"mde_{int(round(tp * 100))}pct": design.mde(tp),
        "critical_effect": res.critical_effect,
        "success_threshold": res.success_threshold,
        "p_true_benefit": res.prob_true_benefit,
        "p_true_mcid": res.prob_true_mcid,
        "pos_mc": res.pos,
        "pos_mc_se": res.mc_se,
        "pos_assurance": res.pos_assurance,
        "ppv": res.ppv_mcid if res.ppv_mcid is not None else res.ppv_benefit,
    }


def load_yaml(path) -> list[dict]:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get("defaults", {})
    return [normalise(t, defaults) for t in cfg["trials"]]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (float, np.floating)):
        if math.isinf(v):
            return "inf"
        if math.isnan(v):
            return "n/a"
        return f"{v:.0f}" if abs(v) >= 100 and float(v).is_integer() else f"{v:.3g}"
    return str(v)
