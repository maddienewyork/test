"""Interactive trial power & probability-of-success tool.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from trialpower import (GRID_PARAMS, EffectPrior, Phase2Evidence, TrialDesign, event_probability, make_endpoint,
                        power_grid,
                        pos_curve, simulate_pos)
from trialpower.enrollment import EnrollmentPlan, EventModel, simulate_enrollment, sites_needed
from trialpower.portfolio import COLUMNS, evaluate, load_yaml

HERE = Path(__file__).parent
BLUE, ORANGE, GRAY = "#257de0", "#f47230", "#8a8a8a"

st.set_page_config(page_title="Trial Power & PoS", layout="wide")


def f3(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3g}"


def pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


# ----------------------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("Trial")
kind = sb.selectbox("Endpoint", ["continuous", "binary", "survival"],
                    format_func={"continuous": "Continuous (t-test)", "binary": "Binary / responder (z-test)",
                                 "survival": "Time-to-event (log-rank)"}.get)

if kind == "continuous":
    sd = sb.number_input("Outcome SD", 0.001, value=1.0, step=0.1, format="%.3f")
    hib = sb.radio("Favourable direction", ["Higher is better", "Lower is better"]) == "Higher is better"
    ep = make_endpoint(kind, sd=sd, higher_is_better=hib)
elif kind == "binary":
    pc = sb.slider("Control response rate", 0.01, 0.99, 0.30, 0.01)
    hib = sb.radio("Favourable direction", ["Higher is better", "Lower is better"]) == "Higher is better"
    ep = make_endpoint(kind, p_control=pc, higher_is_better=hib)
else:
    hib = False
    with sb.expander("Estimate event probability from medians"):
        med = st.number_input("Control median (months)", 0.1, value=12.0)
        hr_plan = st.number_input("Planning HR", 0.05, 2.0, 0.70)
        acc = st.number_input("Accrual period (months)", 0.0, value=18.0)
        fu = st.number_input("Min follow-up after last patient (months)", 0.0, value=12.0)
        est = event_probability(med, hr_plan, acc, fu)
        st.caption(f"Exponential model, uniform accrual → **{est:.3f}**")
    evp = sb.slider("P(event) by final analysis", 0.01, 1.0, 0.65, 0.01)
    ep = make_endpoint(kind, event_prob=evp)

sb.subheader("Design")
c1, c2 = sb.columns(2)
alpha = c1.number_input("Alpha", 0.0001, 0.5, 0.05, 0.005, format="%.4f")
sides = c2.selectbox("Sides", [2, 1])
n_total = sb.number_input("Total N enrolled", 4, value=400, step=10)
ratio = sb.number_input("Allocation ratio T:C", 0.1, 10.0, 1.0, 0.5)
target = sb.slider("Target power", 0.5, 0.99, 0.90, 0.01)

sb.subheader("Dropout")
per_arm = sb.checkbox("Different dropout by arm", help="Drug-arm dropout is often higher (adverse events).")
if per_arm:
    c1, c2 = sb.columns(2)
    dropout_t = c1.number_input("Drug arm", 0.0, 0.9, 0.15, 0.05)
    dropout_c = c2.number_input("Control arm", 0.0, 0.9, 0.10, 0.05)
    dropout = max(dropout_t, dropout_c)
else:
    dropout = sb.number_input("Dropout (both arms)", 0.0, 0.9, 0.10, 0.05)
    dropout_t = dropout_c = None
handling_opts = ["exclude", "dilute"] + (["nri"] if kind == "binary" else [])
handling = sb.selectbox(
    "How dropouts are analysed", handling_opts,
    format_func={"exclude": "Excluded (complete-case / censored)",
                 "dilute": "ITT: dropouts lose drug effect (dilution)",
                 "nri": "ITT: dropouts = treatment failures (NRI)"}.get,
    help="Excluded: fewer evaluable patients, effect unchanged; for time-to-event, dropouts are censored (fewer events). "
         "Dilution: everyone analysed, but drug-arm dropouts get no benefit, so effect x (1 - drug-arm dropout). "
         "NRI: dropouts in both arms counted as non-responders.")
design = TrialDesign(ep, n_total, alpha, sides, ratio, dropout, dropout_t, dropout_c, handling)

sb.subheader("Phase II evidence")
mode = sb.radio("Input", ["Summary data", "Estimate + SE"], horizontal=True)
if mode == "Estimate + SE":
    est_lbl = "Hazard ratio" if kind == "survival" else ep.effect_label
    e2 = sb.number_input(est_lbl, value=0.70 if kind == "survival" else 0.30, format="%.4f")
    se2 = sb.number_input("SE" + (" of log HR" if kind == "survival" else ""), 0.0001, value=0.15, format="%.4f")
    ev = Phase2Evidence(e2, se2)
elif kind == "continuous":
    c1, c2 = sb.columns(2)
    mt = c1.number_input("Mean T", value=0.5, format="%.3f")
    mc = c2.number_input("Mean C", value=0.0, format="%.3f")
    sd2 = sb.number_input("Phase II SD", 0.001, value=float(sd), format="%.3f")
    nt2 = c1.number_input("n T", 2, value=50)
    nc2 = c2.number_input("n C", 2, value=50)
    ev = Phase2Evidence.from_means(mt, mc, sd2, nt2, nc2)
elif kind == "binary":
    c1, c2 = sb.columns(2)
    xt = c1.number_input("Responders T", 0, value=25)
    nt2 = c2.number_input("n T", 1, value=50)
    xc = c1.number_input("Responders C", 0, value=15)
    nc2 = c2.number_input("n C", 1, value=50)
    ev = Phase2Evidence.from_proportions(xt, nt2, xc, nc2)
else:
    hr2 = sb.number_input("Phase II HR", 0.05, 3.0, 0.65)
    c1, c2 = sb.columns(2)
    lo = c1.number_input("95% CI lower", 0.01, 3.0, 0.45)
    hi = c2.number_input("95% CI upper", 0.02, 5.0, 0.94)
    ev = Phase2Evidence.from_hr_ci(hr2, lo, hi)
sb.caption(f"Phase II effect **{f3(ev.estimate)}**, SE {f3(ev.se)}")

sb.subheader("Phase II → III discounting")
haircut = sb.slider("Haircut on effect", 0.0, 0.8, 0.20, 0.05,
                    help="Shrink the Phase II benefit towards zero (winner's curse, broader population).")
weight = sb.slider("Evidence weight", 0.05, 1.0, 1.0, 0.05,
                   help="Power-prior weight: 0.5 treats Phase II as if it had half the patients (wider prior).")
prob_eff = sb.slider("P(drug truly works)", 0.05, 1.0, 1.0, 0.05,
                     help="Mixture prior: with 1 − p the true effect is exactly zero.")
prior = EffectPrior.from_phase2(ep, ev, haircut, weight, prob_eff)
prior_mean_effect = float(ep.from_benefit(prior.mean_benefit))

sb.subheader("Success definition")
use_mcid = sb.checkbox("Also require a minimum clinically meaningful effect", True)
mcid = None
if use_mcid:
    if kind == "continuous":
        default_mcid = (0.2 if hib else -0.2) * sd
    elif kind == "binary":
        default_mcid = 0.10 if hib else -0.10
    else:
        default_mcid = 0.80
    mcid = sb.number_input("Minimum effect (natural units)", value=float(default_mcid), format="%.4f")
n_trials = sb.selectbox("Pivotal trials that must all succeed", [1, 2, 3])
n_sims = sb.select_slider("Monte Carlo simulations", [10_000, 50_000, 100_000, 250_000, 500_000], 100_000)
seed = sb.number_input("Random seed", 0, value=1)

# ----------------------------------------------------------------------------- main
st.title("Clinical trial power & probability of success")
tab_pow, tab_pos, tab_enr, tab_port, tab_help = st.tabs(
    ["Power", "Probability of success", "Enrollment", "Portfolio", "Methods"])

# ---------------- Power
with tab_pow:
    override = st.checkbox("Power for a different effect than the discounted Phase II estimate")
    assumed = st.number_input("Assumed true effect", value=float(round(prior_mean_effect, 4)), format="%.4f") \
        if override else prior_mean_effect
    ss = design.sample_size(assumed, target)
    mde = design.mde(target)
    crit = design.critical_effect()
    m = st.columns(4)
    m[0].metric("Power at planned N", pct(design.power(assumed)), help=f"True effect = {f3(assumed)}")
    m[1].metric(f"N for {target:.0%} power", f"{ss['n_total']}" if np.isfinite(ss["n_total"]) else "∞",
                delta=None if not np.isfinite(ss["n_total"]) else f"{ss['n_total'] - n_total:+.0f} vs plan",
                delta_color="off")
    m[2].metric(f"MDE ({target:.0%} power)", f3(mde), help="Smallest TRUE effect with target power at planned N")
    m[3].metric("Critical observed effect", f3(crit), help="Smallest OBSERVED effect that is statistically significant")
    ev_txt = f" · **{design.events():.0f} events** expected" if kind == "survival" else ""
    dil_txt = ""
    if handling != "exclude":
        dil_txt = (f" Effect the ITT analysis actually sees: "
                   f"**{f3(float(ep.from_benefit(design.analysis_benefit(ep.to_benefit(assumed)))))}**.")
    st.caption(f"Assumed effect {f3(assumed)} ({ep.effect_label}). Analysed per arm: "
               f"{design.arms()[0]:.0f} T / {design.arms()[1]:.0f} C{ev_txt}.{dil_txt}")

    hi_n = max(n_total * 2, ss["n_total"] * 1.3 if np.isfinite(ss["n_total"]) else n_total * 2)
    ngrid = np.unique(np.linspace(10, hi_n, 120).round())
    dfn = pd.DataFrame({"Total N": ngrid, "Power": design.power_curve_n(assumed, ngrid)})
    base = alt.Chart(dfn).mark_line(color=BLUE, strokeWidth=2).encode(
        x=alt.X("Total N:Q", title="Total N enrolled"), y=alt.Y("Power:Q", title="Power", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("Total N:Q"), alt.Tooltip("Power:Q", format=".1%")])
    rule_n = alt.Chart(pd.DataFrame({"Total N": [n_total]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(x="Total N:Q")
    rule_p = alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(y=alt.Y("y:Q", title="Power"))

    b_lo = min(0.0, prior.mean_benefit - 3 * prior.sd_benefit)
    b_hi = max(design.endpoint.to_benefit(mde) * 1.6 if np.isfinite(mde) else 1, prior.mean_benefit + 3 * prior.sd_benefit)
    if kind == "binary":
        b_hi = min(b_hi, ep.max_benefit() - 1e-3)
        b_lo = max(b_lo, -(1 - ep.max_benefit()) + 1e-3)
    bgrid = np.linspace(b_lo, b_hi, 200)
    dfe = pd.DataFrame({"True effect": ep.from_benefit(bgrid), "Power": design.power_benefit(bgrid)})
    eff_chart = alt.Chart(dfe).mark_line(color=BLUE, strokeWidth=2).encode(
        x=alt.X("True effect:Q", title=f"True {ep.effect_label}", scale=alt.Scale(zero=False)),
        y=alt.Y("Power:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("True effect:Q", format=".3f"), alt.Tooltip("Power:Q", format=".1%")])
    rule_a = alt.Chart(pd.DataFrame({"True effect": [assumed]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(x="True effect:Q")

    c1, c2 = st.columns(2)
    c1.markdown("**Power vs sample size** (dashed: planned N, target power)")
    c1.altair_chart((base + rule_n + rule_p).interactive(), width="stretch")
    c2.markdown("**Power vs true effect at planned N** (dashed: assumed effect)")
    c2.altair_chart((eff_chart + rule_a).interactive(), width="stretch")

    rows = []
    for p in (0.8, 0.85, 0.9, 0.95):
        s = design.sample_size(assumed, p)
        rows.append({"Target power": f"{p:.0%}", "Total N": s["n_total"], "N treatment": s["n_t"],
                     "N control": s["n_c"], **({"Events": s["events"]} if kind == "survival" else {}),
                     f"MDE at N={n_total}": f3(design.mde(p))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # ---- heatmap
    st.markdown("#### Power heatmap")
    nuis = {"continuous": "sd", "binary": "p_control", "survival": "event_prob"}[kind]
    outcome_word = {"continuous": "mean", "binary": "response rate", "survival": "median survival"}[kind]
    presets = {
        "arms": (f"Drug-arm vs control-arm {outcome_word}", "treat_outcome", "control_outcome"),
        "n": ("Treatment effect vs total N", "effect", "n_total"),
        "dropout": ("Treatment effect vs dropout", "effect", "dropout"),
        "nuis": (f"Treatment effect vs {GRID_PARAMS[nuis].lower()}", "effect", nuis),
    }
    preset = st.selectbox("Axes", list(presets), format_func=lambda k: presets[k][0])
    _, xp, yp = presets[preset]

    # sensible default ranges around the current assumptions
    a = float(assumed)
    if kind == "survival":
        base_ctrl = st.number_input("Control-arm median survival (months)", 0.1, value=12.0) if preset == "arms" else 12.0
    elif kind == "binary":
        base_ctrl = ep.p_control
    else:
        base_ctrl = float(mc) if mode == "Summary data" else 0.0

    def default_range(p):
        if p == "effect":
            if kind == "survival":
                return float(np.exp(np.log(a) * 2)), min(1.0, float(np.exp(np.log(a) * 0.25)))
            lo, hi = sorted([a * 0.25, a * 2.0])
            if kind == "binary":
                lo, hi = max(lo, -ep.p_control + 0.01), min(hi, 0.99 - ep.p_control)
            return lo, hi
        if p == "treat_outcome":
            if kind == "survival":
                return base_ctrl, base_ctrl / a ** 2
            if kind == "binary":
                return max(0.01, ep.p_control + min(0, 2 * a)), min(0.99, ep.p_control + max(0, 2 * a))
            return tuple(sorted([base_ctrl, base_ctrl + 2 * a]))
        if p == "control_outcome":
            if kind == "survival":
                return base_ctrl * 0.6, base_ctrl * 1.4
            if kind == "binary":
                return max(0.01, ep.p_control - 0.15), min(0.99, ep.p_control + 0.15)
            return base_ctrl - abs(a), base_ctrl + abs(a)
        if p == "n_total":
            return max(10.0, n_total * 0.5), n_total * 2.0
        if p == "dropout":
            return 0.0, 0.4
        if p == "sd":
            return ep.sd * 0.6, ep.sd * 1.6
        if p == "p_control":
            return max(0.01, ep.p_control - 0.15), min(0.99, ep.p_control + 0.15)
        if p == "event_prob":
            return max(0.05, ep.event_prob - 0.3), min(1.0, ep.event_prob + 0.2)

    c = st.columns(5)
    xlo, xhi = default_range(xp)
    ylo, yhi = default_range(yp)
    xlab = "True HR" if (xp == "effect" and kind == "survival") else (
        f"True {ep.effect_label}" if xp == "effect" else GRID_PARAMS[xp])
    ylab = GRID_PARAMS[yp]
    xlo = c[0].number_input(f"{xlab}: from", value=float(round(xlo, 4)), format="%.4g", key=f"xlo_{preset}_{kind}")
    xhi = c[1].number_input("to", value=float(round(xhi, 4)), format="%.4g", key=f"xhi_{preset}_{kind}")
    ylo = c[2].number_input(f"{ylab}: from", value=float(round(ylo, 4)), format="%.4g", key=f"ylo_{preset}_{kind}")
    yhi = c[3].number_input("to", value=float(round(yhi, 4)), format="%.4g", key=f"yhi_{preset}_{kind}")
    steps = c[4].slider("Steps", 3, 15, 9)
    xv = np.linspace(xlo, xhi, steps)
    yv = np.linspace(ylo, yhi, steps)
    if yp == "n_total":
        yv = np.round(yv)
    grid = power_grid(design, assumed, (xp, list(xv)), (yp, list(yv)), control_outcome=base_ctrl)

    def lbl(v, p):
        if p == "n_total":
            return f"{v:.0f}"
        if p in ("dropout", "dropout_t") or (p in ("p_control", "event_prob")) or (kind == "binary" and p.endswith("outcome")):
            return f"{100 * v:.0f}%" if abs(100 * v - round(100 * v)) < 1e-6 else f"{100 * v:.1f}%"
        return f"{v:.3g}"

    xs = [lbl(v, xp) for v in xv]
    ys = [lbl(v, yp) for v in yv]
    hm = pd.DataFrame([{"x": xs[j], "y": ys[i], "Power": grid[i, j]} for i in range(len(yv)) for j in range(len(xv))])
    hm["label"] = hm["Power"].map(lambda p: "–" if not np.isfinite(p) else f"{100 * p:.0f}%")
    hm["meets"] = hm["Power"] >= target
    enc_x = alt.X("x:O", sort=xs, title=xlab, axis=alt.Axis(labelAngle=0))
    enc_y = alt.Y("y:O", sort=list(reversed(ys)), title=ylab)
    rect = alt.Chart(hm).mark_rect(stroke="white", strokeWidth=2).encode(
        x=enc_x, y=enc_y,
        color=alt.Color("Power:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                        legend=alt.Legend(format="%", title="Power")),
        tooltip=[alt.Tooltip("x:O", title=xlab), alt.Tooltip("y:O", title=ylab),
                 alt.Tooltip("Power:Q", format=".1%")])
    def text_layer(meets, weight):
        return alt.Chart(hm[hm["meets"] == meets]).mark_text(fontSize=12, fontWeight=weight).encode(
            x=enc_x, y=enc_y, text="label:N",
            color=alt.condition("datum.Power > 0.55", alt.value("white"), alt.value("#191919")))

    st.altair_chart((rect + text_layer(False, "normal") + text_layer(True, "bold")).properties(height=max(260, 38 * steps)), width="stretch")
    notes = [f"**Bold** cells meet the {target:.0%} target."]
    if preset == "arms":
        if kind == "survival":
            notes.append("HR = control median ÷ drug median (exponential survival); P(event) held fixed.")
        elif kind == "continuous":
            notes.append("With a common SD, power depends only on the drug − control difference, so it is constant "
                         "along the diagonals.")
        else:
            notes.append("Variance depends on both rates, so the same difference has different power at different "
                         "control rates.")
    if preset != "n":
        notes.append(f"Total N = {n_total}.")
    st.caption(" ".join(notes))

# ---------------- PoS
with tab_pos:
    res = simulate_pos(design, prior, mcid, n_trials, n_sims, int(seed))
    m = st.columns(4)
    m[0].metric("Probability of success", pct(res.pos), help=f"Monte Carlo ± {pct(1.96 * res.mc_se)} (95%). "
                f"Quadrature cross-check: {pct(res.pos_assurance)}")
    m[1].metric("Success threshold (observed)", f3(res.success_threshold),
                help="Smallest observed effect that counts as a win: max of significance boundary and the minimum "
                     "meaningful effect")
    m[2].metric("P(true effect favours drug)", pct(res.prob_true_benefit),
                help="Ceiling on PoS as N → ∞ (without a minimum-effect requirement)")
    m[3].metric("P(win is real)", pct(res.ppv_mcid if res.ppv_mcid is not None else res.ppv_benefit),
                help="Among simulated successes, share whose TRUE effect meets the minimum effect "
                     "(or favours the drug if none set)")
    extra = [f"Significant only: **{pct(res.pos_sig_only)}**", f"Quadrature: **{pct(res.pos_assurance)}**",
             f"Power at discounted point estimate: **{pct(design.power(prior_mean_effect) ** n_trials)}**"]
    if n_trials > 1:
        extra.append(f"≥1 of {n_trials} trials succeeds: **{pct(res.pos_any)}**")
    if res.prob_true_mcid is not None:
        extra.append(f"P(true effect ≥ minimum): **{pct(res.prob_true_mcid)}**")
    st.caption(" · ".join(extra))

    def hist(values, success, title):
        lo, hi = np.percentile(values, [0.5, 99.5])
        bins = np.linspace(lo, hi, 51)
        hs, _ = np.histogram(values[success], bins)
        hf, _ = np.histogram(values[~success], bins)
        n = len(values)
        rows = []
        for i in range(len(hs)):
            s_, f_ = hs[i] / n, hf[i] / n
            rows.append({"x0": bins[i], "x1": bins[i + 1], "y0": 0.0, "y1": s_, "Share": s_, "Outcome": "Success"})
            rows.append({"x0": bins[i], "x1": bins[i + 1], "y0": s_, "y1": s_ + f_, "Share": f_, "Outcome": "Failure"})
        return alt.Chart(pd.DataFrame(rows)).mark_rect(stroke="white", strokeWidth=0.5).encode(
            x=alt.X("x0:Q", title=title, scale=alt.Scale(zero=False)), x2="x1:Q",
            y=alt.Y("y0:Q", axis=alt.Axis(format="%"), title="Share of simulations"), y2="y1:Q",
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Success", "Failure"], range=[BLUE, ORANGE]),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("x0:Q", format=".3f", title="from"), alt.Tooltip("x1:Q", format=".3f", title="to"),
                     "Outcome:N", alt.Tooltip("Share:Q", format=".2%")])

    sub = slice(0, min(len(res.success), 100_000))
    te, oe, sc = res.true_effect[sub], res.observed_effect[sub], res.success[sub]
    finite = np.isfinite(te) & np.isfinite(oe)
    c1, c2 = st.columns(2)
    c1.markdown("**Where does success come from?** True effect drawn from the Phase II prior")
    c1.altair_chart(hist(te[finite], sc[finite], f"True {ep.effect_label}"), width="stretch")
    lines = pd.DataFrame({"x0": [res.success_threshold]})
    c2.markdown("**Observed Phase III effect** (dashed: success threshold)")
    c2.altair_chart(hist(oe[finite], sc[finite], f"Observed {ep.effect_label} (trial 1)")
                    + alt.Chart(lines).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(x="x0:Q"),
                    width="stretch")

    st.markdown("**PoS vs sample size** — power keeps climbing; PoS plateaus because the effect itself is uncertain")
    ngrid = np.unique(np.geomspace(20, max(n_total * 4, 200), 60).round())
    curve = pd.DataFrame(pos_curve(design, prior, ngrid, mcid, n_trials)).rename(columns={
        "n_total": "Total N", "pos": "Probability of success", "power_at_prior_mean": "Power at point estimate"})
    long = curve.melt("Total N", var_name="Measure", value_name="Probability")
    ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("Total N:Q", scale=alt.Scale(type="log", domain=[float(ngrid[0]), float(ngrid[-1])], nice=False)),
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Measure:N", scale=alt.Scale(domain=["Probability of success", "Power at point estimate"],
                                                     range=[BLUE, ORANGE]), legend=alt.Legend(orient="top")),
        tooltip=["Total N:Q", "Measure:N", alt.Tooltip("Probability:Q", format=".1%")])
    st.altair_chart((ch + rule_n).interactive(), width="stretch")

    st.markdown("**Sensitivity: PoS by haircut × P(drug works)** (quadrature, current N)")
    from trialpower import assurance
    hcs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    pes = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    grid = pd.DataFrame(
        [[assurance(design, EffectPrior.from_phase2(ep, ev, h, weight, p), mcid, n_trials) for h in hcs] for p in pes],
        index=[f"P(works) {p:.0%}" for p in pes], columns=[f"haircut {h:.0%}" for h in hcs])
    st.dataframe(grid.style.format("{:.1%}").background_gradient(cmap="Blues", vmin=0, vmax=1),
                 width="stretch")

# ---------------- Enrollment
@st.cache_data(show_spinner="Simulating enrollment…")
def _run_enrollment(plan_kw, ev_kw, n_sims, seed, elapsed, enrolled):
    ev_ = EventModel(**ev_kw) if ev_kw else None
    return simulate_enrollment(EnrollmentPlan(**plan_kw), n_sims, seed, elapsed, enrolled, ev_)


def fan_chart(res, q_dict, mean_unused, target_val, ylab, planned=None):
    ends = [x[np.isfinite(x)] for x in (res.completion, res.event_completion) if x is not None]
    ends = np.concatenate(ends) if ends else np.array([])
    t_max = np.quantile(ends, 0.99) * 1.1 if len(ends) else res.grid[-1]
    keep = res.grid <= max(t_max, res.grid[0] + 1)
    g = res.grid[keep]
    df = pd.DataFrame({"Month": g, "p10": q_dict[0.10][keep], "p90": q_dict[0.90][keep], "p25": q_dict[0.25][keep],
                       "p75": q_dict[0.75][keep], "Median": q_dict[0.50][keep]})
    band80 = alt.Chart(df).mark_area(color=BLUE, opacity=0.18).encode(
        x=alt.X("Month:Q", title="Months from first site open"), y=alt.Y("p10:Q", title=ylab), y2="p90:Q")
    band50 = alt.Chart(df).mark_area(color=BLUE, opacity=0.30).encode(x="Month:Q", y="p25:Q", y2="p75:Q")
    med = alt.Chart(df).mark_line(color=BLUE, strokeWidth=2).encode(
        x="Month:Q", y="Median:Q",
        tooltip=[alt.Tooltip("Month:Q", format=".1f"), alt.Tooltip("p10:Q", format=".0f", title="10th pct"),
                 alt.Tooltip("Median:Q", format=".0f"), alt.Tooltip("p90:Q", format=".0f", title="90th pct")])
    layers = band80 + band50 + med
    if planned is not None:
        dp = pd.DataFrame({"Month": g, "Planned": planned[keep]})
        layers += alt.Chart(dp).mark_line(color=ORANGE, strokeWidth=2, strokeDash=[6, 4]).encode(
            x="Month:Q", y="Planned:Q", tooltip=[alt.Tooltip("Month:Q", format=".1f"),
                                                 alt.Tooltip("Planned:Q", format=".0f")])
    layers += alt.Chart(pd.DataFrame({"t": [target_val]})).mark_rule(color=GRAY, strokeDash=[4, 4]).encode(y="t:Q")
    return layers.interactive()


def month_hist(x, title):
    f = x[np.isfinite(x)]
    if not len(f):
        return None
    bins = np.arange(np.floor(f.min()), np.ceil(f.max()) + 1, 0.5 if np.ptp(f) < 15 else 1.0)
    h, e = np.histogram(f, bins)
    df = pd.DataFrame({"from": e[:-1], "to": e[1:], "Share": h / len(x)})
    return alt.Chart(df).mark_rect(color=BLUE, stroke="white", strokeWidth=1).encode(
        x=alt.X("from:Q", title=title), x2="to:Q", y=alt.Y("Share:Q", axis=alt.Axis(format="%"), title="Share of simulations"),
        tooltip=[alt.Tooltip("from:Q", title="from month"), alt.Tooltip("to:Q", title="to month"),
                 alt.Tooltip("Share:Q", format=".1%")])


with tab_enr:
    st.markdown("Monte Carlo recruitment forecast (Poisson–gamma model): sites open over time and ramp up, "
                "site speeds vary, and the planned rate itself is uncertain.")
    c = st.columns(4)
    e_target = c[0].number_input("Patients to randomise", 2, value=int(n_total), step=10, key="e_target")
    e_sites = c[1].number_input("Sites", 1, value=40, key="e_sites")
    e_rate = c[2].number_input("Patients / site / month (full speed)", 0.01, value=0.8, step=0.1, format="%.2f",
                               key="e_rate")
    e_sf = c[3].number_input("Screen-fail rate", 0.0, 0.95, 0.25, 0.05, key="e_sf",
                             help="Only used to report how many patients must be screened.")
    c = st.columns(5)
    e_a0 = c[0].number_input("First site opens (month)", 0.0, value=0.0, key="e_a0")
    e_a1 = c[1].number_input("Last site opens (month)", 0.0, value=6.0, key="e_a1")
    e_ramp = c[2].number_input("Site ramp-up (months)", 0.0, value=2.0, key="e_ramp",
                               help="Months for a newly opened site to reach full speed (linear ramp).")
    e_scv = c[3].number_input("Site-to-site variability (CV)", 0.0, 3.0, 0.8, 0.1, key="e_scv",
                              help="0 = all sites identical; ~0.8–1.2 is typical (a few sites recruit most patients).")
    e_rcv = c[4].number_input("Uncertainty in planned rate (CV)", 0.0, 2.0, 0.25, 0.05, key="e_rcv",
                              help="How wrong the planning rate could be. 0.2–0.4 is realistic before first patient in.")

    with st.expander("Re-forecast an ongoing trial"):
        refc = st.checkbox("Use actual enrollment to date", key="e_refc")
        c = st.columns(2)
        e_elapsed = c[0].number_input("Months since first site could open", 0.0, value=6.0, key="e_el")
        e_enrolled = c[1].number_input("Patients randomised so far", 0, value=60, key="e_en")
        st.caption("The planned rate is updated with what has actually happened (Bayesian gamma update; the "
                   "uncertainty CV above sets how strongly the plan is believed), then the remainder is simulated.")
    if not refc:
        e_elapsed, e_enrolled = 0.0, 0

    ev_kw = None
    if kind == "survival":
        st.markdown("**Event accrual (time-to-event)**")
        c = st.columns(4)
        ev_target = c[0].number_input("Events for final analysis", 1, value=int(math.ceil(design.events())), key="e_ev")
        ev_med = c[1].number_input("Control median (months)", 0.1, value=12.0, key="e_med")
        ev_hr = c[2].number_input("Assumed HR", 0.05, 3.0, float(round(prior_mean_effect, 3)), key="e_hr")
        ev_do = c[3].number_input("Loss to follow-up per year", 0.0, 0.9, 0.05, 0.01, key="e_do")
        ev_kw = dict(median_control=ev_med, hr=ev_hr, target_events=int(ev_target), ratio=float(ratio),
                     dropout_per_year=ev_do)

    c = st.columns(3)
    e_sims = c[0].select_slider("Simulations", [500, 1000, 2000, 5000], 2000, key="e_sims")
    deadline = c[1].number_input("Deadline for last patient in (month)", 0.5, value=18.0, key="e_dl")
    plan_kw = dict(target_n=int(e_target), n_sites=int(e_sites), rate_per_site=float(e_rate),
                   activation_start=float(e_a0), activation_end=float(max(e_a0, e_a1)), site_ramp=float(e_ramp),
                   site_cv=float(e_scv), rate_uncertainty=float(e_rcv), screen_fail=float(e_sf))
    if e_enrolled >= e_target:
        st.warning("Already at or above the target; nothing left to simulate.")
    else:
        res_e = _run_enrollment(plan_kw, ev_kw, int(e_sims), int(seed), float(e_elapsed), int(e_enrolled))
        plan = res_e.plan
        se = res_e.summary()
        m = st.columns(4)
        m[0].metric("Planned last patient in", f"month {plan.expected_completion(res_e.mean_rate, e_enrolled, e_elapsed):.1f}",
                    help="Using the average rate with no randomness: the usual spreadsheet answer.")
        m[1].metric("Simulated LPI (median)", f"month {se['lpi_p50']:.1f}")
        m[2].metric("80% interval", f"{se['lpi_p10']:.1f} – {se['lpi_p90']:.1f}")
        m[3].metric(f"P(LPI by month {deadline:g})", pct(res_e.prob_complete_by(deadline)))
        cap = [f"Patients to screen: **~{plan.screened_needed:,}**"]
        if refc:
            cap.append(f"Updated rate: **{res_e.mean_rate:.2f}**/site/month (plan {e_rate:.2f}; CV now "
                       f"{res_e.rate_cv:.2f})")
        if se["lpi_p_within_horizon"] < 1:
            cap.append(f"⚠ {pct(1 - se['lpi_p_within_horizon'])} of simulations never finish within the horizon")
        st.caption(" · ".join(cap))

        st.markdown("**Cumulative enrollment**: median, 50% and 80% bands; dashed orange = plan; dashed gray = target")
        st.altair_chart(fan_chart(res_e, res_e.enroll_q, res_e.enroll_mean, e_target, "Patients randomised",
                                  res_e.expected), width="stretch")

        c1, c2 = st.columns(2)
        hchart = month_hist(res_e.completion, "Last patient in (month)")
        if hchart is not None:
            c1.markdown("**When does enrollment finish?**")
            c1.altair_chart(hchart, width="stretch")
        if ev_kw is not None:
            c2.markdown("**When is the final analysis?** (target events reached)")
            ha = month_hist(res_e.event_completion, "Final analysis (month)")
            if ha is not None:
                c2.altair_chart(ha, width="stretch")
            m = st.columns(3)
            m[0].metric("Final analysis (median)", f"month {se['analysis_p50']:.1f}")
            m[1].metric("80% interval", f"{se['analysis_p10']:.1f} – {se['analysis_p90']:.1f}")
            m[2].metric("Events expected at LPI (median)",
                        f"{np.interp(se['lpi_p50'], res_e.grid, res_e.events_q[0.5]):.0f}")
            st.markdown("**Cumulative events**: median, 50% and 80% bands; dashed gray = target events")
            st.altair_chart(fan_chart(res_e, res_e.events_q, res_e.events_mean, ev_kw["target_events"], "Events"),
                            width="stretch")

        st.markdown("**How many sites to hit the deadline?**")
        c = st.columns([1, 1, 2])
        conf = c[0].slider("With probability", 0.5, 0.95, 0.8, 0.05, key="e_conf")
        if c[1].button("Find sites needed"):
            with st.spinner("Searching…"):
                ns = sites_needed(EnrollmentPlan(**plan_kw), deadline, conf, n_sims=500, seed=int(seed))
            c[2].markdown(f"**{ns} sites** gives ≥{conf:.0%} chance of last patient in by month {deadline:g}."
                          if ns else "Not reachable: sites open too late or rate too low. Move the deadline "
                                     "or open sites sooner.")

        tbl = pd.DataFrame(res_e.monthly_table())
        with st.expander("Month-by-month table"):
            fmt_cols = {c_: "{:.0f}" for c_ in tbl.columns if c_ not in ("month",) and not c_.startswith("p_")}
            fmt_cols.update({c_: "{:.0%}" for c_ in tbl.columns if c_.startswith("p_")})
            st.dataframe(tbl.style.format(fmt_cols), hide_index=True, width="stretch")
            st.download_button("Download CSV", tbl.to_csv(index=False), "enrollment_forecast.csv", "text/csv")

# ---------------- Portfolio
with tab_port:
    st.markdown("Edit one row per trial (or upload a YAML in the `examples/trials.yaml` format). "
                "Phase II effect is in natural units; SE is of log HR for survival. Blank `assumed_effect` "
                "= discounted Phase II estimate.")
    up = st.file_uploader("Load YAML", type=["yaml", "yml"])
    if up is not None:
        tmp = io.StringIO(up.getvalue().decode())
        import yaml
        cfg = yaml.safe_load(tmp)
        from trialpower.portfolio import normalise
        rows = [normalise(t, cfg.get("defaults", {})) for t in cfg["trials"]]
    else:
        rows = load_yaml(HERE / "examples" / "trials.yaml")
    table = st.data_editor(pd.DataFrame(rows, columns=COLUMNS), num_rows="dynamic", width="stretch",
                           key=f"port_{up.name if up else 'example'}")
    psims = st.select_slider("Simulations per trial", [10_000, 50_000, 100_000], 50_000, key="psims")
    if st.button("Run portfolio", type="primary"):
        out = []
        for r in table.replace({np.nan: None}).to_dict("records"):
            if not r.get("endpoint"):
                continue
            try:
                out.append(evaluate(r, psims, int(seed)))
            except Exception as e:  # keep going on a bad row
                out.append({"name": r.get("name"), "error": str(e)})
        df = pd.DataFrame(out)
        pct_cols = [c for c in ("power", "p_true_benefit", "p_true_mcid", "pos_mc", "pos_assurance", "ppv") if c in df]
        int_cols = [c for c in df if c in ("n_total", "events") or c.startswith("n_for_")]
        st.dataframe(df.style.format({**{c: "{:.1%}" for c in pct_cols}, **{c: "{:,.0f}" for c in int_cols}},
                                     na_rep="–", precision=3),
                     width="stretch", hide_index=True)
        if "pos_mc" in df:
            ok = df.dropna(subset=["pos_mc"])
            st.markdown(f"Expected number of successful programmes: **{ok['pos_mc'].sum():.2f}** of {len(ok)}")
        st.download_button("Download CSV", df.to_csv(index=False), "portfolio_results.csv", "text/csv")

# ---------------- Methods
with tab_help:
    st.markdown((HERE / "docs" / "METHODS.md").read_text())
