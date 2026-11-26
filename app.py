"""Interactive trial power & probability-of-success tool.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from trialpower import (EffectPrior, Phase2Evidence, TrialDesign, event_probability, make_endpoint,
                        pos_curve, simulate_pos)
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
c1, c2 = sb.columns(2)
ratio = c1.number_input("Ratio T:C", 0.1, 10.0, 1.0, 0.5)
dropout = c2.number_input("Dropout", 0.0, 0.9, 0.10, 0.05)
target = sb.slider("Target power", 0.5, 0.99, 0.90, 0.01)
design = TrialDesign(ep, n_total, alpha, sides, ratio, dropout)

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
tab_pow, tab_pos, tab_port, tab_help = st.tabs(["Power", "Probability of success", "Portfolio", "Methods"])

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
    st.caption(f"Assumed effect {f3(assumed)} ({ep.effect_label}). Evaluable per arm: "
               f"{design.arms()[0]:.0f} T / {design.arms()[1]:.0f} C{ev_txt}.")

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
