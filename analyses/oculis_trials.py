"""Power, probability of success and enrollment analyses for two Oculis trials.

    python analyses/oculis_trials.py            # prints a markdown report
    python analyses/oculis_trials.py > analyses/oculis_report.md

PREDICT-1 (licaminlimab, TNFR1-genotype dry eye disease) and PIONEER-1
(privosegtor + methylprednisolone in acute optic neuritis).

IMPORTANT: inputs come from public press releases / registry summaries (Nov 2026).
Many design parameters (SDs, allocation, alpha, site counts, recruitment rates,
Phase 2 drug-vs-vehicle effects in subgroups) are NOT public; they are
ASSUMPTIONS, flagged below with `# ASSUMPTION`. Edit them and re-run.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trialpower import (Binary, Continuous, EffectPrior, Phase2Evidence, TrialDesign, assurance,  # noqa: E402
                        power_grid, simulate_pos)
from trialpower.enrollment import EnrollmentPlan, simulate_enrollment, sites_needed  # noqa: E402

SEED = 2026
SIMS = 200_000


def pct(x):
    return f"{100 * x:.0f}%"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def heat_table(grid, xs, ys, xlab, ylab):
    hdr = [f"{ylab} \\ {xlab}"] + list(xs)
    return table(hdr, [[y] + [("–" if not np.isfinite(v) else pct(v)) for v in row] for y, row in zip(ys, grid)])


# =============================================================================
# PREDICT-1 — licaminlimab, TNFR1-mediated dry eye disease
# =============================================================================
# Public: randomization complete, 176 patients; ~2/3 carry the TNFR1 variant
# (primary analysis population); primary endpoint = change from baseline to
# Day 29 in global ocular discomfort severity (vs vehicle), genotype population.
# Topline expected ~year-end 2026.
P1_N_TOTAL = 176
P1_N_PRIMARY = round(176 * 2 / 3)          # ~117 TNFR1-variant patients
P1_RATIO = 1.0                             # ASSUMPTION: 1:1 drug:vehicle
P1_ALPHA = 0.05                            # ASSUMPTION: two-sided 5%
P1_DROPOUT = 0.05                          # ASSUMPTION: 4-week trial, low dropout
P1_SD = 25.0                               # ASSUMPTION: SD of change in 0-100 VAS discomfort score
# Phase 2 drug-vs-vehicle difference in the genotype subgroup is NOT public.
# ASSUMPTION: -10 points (licaminlimab minus vehicle; negative = less discomfort),
# with the SE a genotype subgroup of ~23 patients (RELIEF) would give.
P1_PH2_EFFECT = -10.0
P1_PH2_SE = P1_SD * math.sqrt(1 / 12 + 1 / 11)
P1_HAIRCUT = 0.30                          # ASSUMPTION: small exploratory subgroup -> heavy optimism
P1_PROB_WORKS = 0.65                       # ASSUMPTION: DED symptom endpoints have high vehicle response
P1_MCID = None                             # significance only


def predict1():
    ep = Continuous(sd=P1_SD, higher_is_better=False)
    d = TrialDesign(ep, P1_N_PRIMARY, P1_ALPHA, 2, P1_RATIO, P1_DROPOUT)
    prior = EffectPrior.from_phase2(ep, Phase2Evidence(P1_PH2_EFFECT, P1_PH2_SE), P1_HAIRCUT, 1.0, P1_PROB_WORKS)
    res = simulate_pos(d, prior, P1_MCID, 1, SIMS, SEED)

    out = ["## PREDICT-1 (licaminlimab, TNFR1-genotype dry eye disease)", ""]
    out.append(f"Primary population ≈ **{P1_N_PRIMARY}** TNFR1-variant patients (2/3 of {P1_N_TOTAL} randomized), "
               f"1:1, two-sided α={P1_ALPHA}, {pct(P1_DROPOUT)} dropout, SD of change {P1_SD:g} points (0–100 scale).")
    out.append("")
    out.append("### Power")
    rows = []
    for eff in (-4, -6, -8, -10, -12, -15):
        rows.append([f"{eff:+g}", pct(d.power(eff)), d.sample_size(eff, 0.8)["n_total"],
                     d.sample_size(eff, 0.9)["n_total"]])
    out.append(table(["True difference vs vehicle", f"Power at N={P1_N_PRIMARY}", "N for 80%", "N for 90%"], rows))
    out.append("")
    out.append(f"* **MDE** at N={P1_N_PRIMARY}: 80% power → **{d.mde(0.8):.1f}** points; 90% → {d.mde(0.9):.1f}.")
    out.append(f"* **Critical observed effect** (smallest result that is significant): **{d.critical_effect():.1f}** points.")
    out.append("")
    out.append("Power by true difference × SD of change (N fixed):")
    effs = [-4.0, -6.0, -8.0, -10.0, -12.0, -15.0]
    sds = [18.0, 22.0, 25.0, 28.0, 32.0]
    g = power_grid(d, None, ("effect", effs), ("sd", sds))
    out.append(heat_table(g, [f"{e:+g}" for e in effs], [f"SD {s:g}" for s in sds], "difference", "SD"))
    out.append("")
    out.append("### Probability of success")
    out.append(f"Prior from Phase 2 (assumed {P1_PH2_EFFECT:+g} ± {P1_PH2_SE:.1f}), {pct(P1_HAIRCUT)} haircut, "
               f"P(drug truly works) {pct(P1_PROB_WORKS)}.")
    out.append("")
    out.append(table(["Measure", "Value"], [
        ["**PoS (stat. sig. win)**", f"**{pct(res.pos)}** (quadrature {pct(res.pos_assurance)})"],
        ["Power at discounted point estimate", pct(d.power(float(ep.from_benefit(prior.mean_benefit))))],
        ["P(true effect favours drug)", pct(res.prob_true_benefit)],
        ["P(a win is a real benefit)", pct(res.ppv_benefit)],
    ]))
    out.append("")
    out.append("PoS sensitivity — haircut × P(drug truly works):")
    hcs, pws = [0.0, 0.15, 0.3, 0.45], [1.0, 0.8, 0.65, 0.5]
    rows = [[f"P(works) {pct(p)}"] + [pct(assurance(d, EffectPrior.from_phase2(
        ep, Phase2Evidence(P1_PH2_EFFECT, P1_PH2_SE), h, 1.0, p))) for h in hcs] for p in pws]
    out.append(table(["", *[f"haircut {pct(h)}" for h in hcs]], rows))
    out.append("")
    out.append("PoS sensitivity — assumed Phase 2 subgroup difference (haircut and P(works) as base):")
    rows = []
    for e in (-6.0, -8.0, -10.0, -12.0, -15.0):
        pr = EffectPrior.from_phase2(ep, Phase2Evidence(e, P1_PH2_SE), P1_HAIRCUT, 1.0, P1_PROB_WORKS)
        rows.append([f"{e:+g}", pct(assurance(d, pr))])
    out.append(table(["Phase 2 difference", "PoS"], rows))
    return "\n".join(out)


# =============================================================================
# PIONEER-1 — privosegtor + IV methylprednisolone, acute optic neuritis
# =============================================================================
# Public: Phase 3, SPA agreed with FDA; primary = proportion gaining >=15 letters
# of low-contrast visual acuity (LCVA) at Month 3; vs placebo + methylprednisolone;
# ~210 patients (registry mirror, unverified); start 28 May 2026; first patient
# dosed ~Sep 2026; primary completion Dec 2027 (so LPI ~Sep 2027).
# Phase 2 ACUITY (~33-36 pts, 2 doses + placebo): ~18-letter LCVA difference at M3, p=0.004.
P2_N = 210
P2_RATIO = 1.0                       # ASSUMPTION: 1:1, single dose arm
P2_ALPHA = 0.05                      # ASSUMPTION
P2_DROPOUT = 0.10                    # ASSUMPTION
P2_HANDLING = "exclude"              # ASSUMPTION (NRI shown as sensitivity)
ACUITY_DIFF = 18.0                   # letters, public
ACUITY_SE = ACUITY_DIFF / stats.norm.isf(0.004 / 2)      # 6.25 letters, back-calculated from p=0.004
LCVA_SD = ACUITY_SE / math.sqrt(1 / 22 + 1 / 11)         # ~17 letters, implied (22 active vs 11 placebo)
PLACEBO_GAIN = 12.0                  # ASSUMPTION: mean LCVA gain on steroids alone at Month 3 (letters)
P2_HAIRCUT = 0.40                    # ASSUMPTION: tiny Phase 2, p=0.004 on ~33 pts -> large winner's curse
P2_PROB_WORKS = 0.75                 # ASSUMPTION: neuroprotection historically hard; BTD + structural support


def responder_rate(mean_gain, sd=LCVA_SD, thr=15.0):
    return float(stats.norm.sf((thr - mean_gain) / sd))


def pioneer_prior(placebo_gain=PLACEBO_GAIN, haircut=P2_HAIRCUT, prob=P2_PROB_WORKS, rng_seed=SEED):
    """Translate the ACUITY mean-LCVA difference into a prior on the >=15-letter responder difference."""
    rng = np.random.default_rng(rng_seed)
    delta = ACUITY_DIFF * (1 - haircut) + ACUITY_SE * rng.standard_normal(400_000)
    pc = responder_rate(placebo_gain)
    rd = stats.norm.sf((15 - placebo_gain - delta) / LCVA_SD) - pc
    return pc, EffectPrior(float(rd.mean()), float(rd.std()), prob)


def pioneer1():
    pc, prior = pioneer_prior()
    ep = Binary(p_control=pc)
    d = TrialDesign(ep, P2_N, P2_ALPHA, 2, P2_RATIO, P2_DROPOUT, dropout_handling=P2_HANDLING)
    res = simulate_pos(d, prior, None, 1, SIMS, SEED)
    rd_mean = prior.mean_benefit

    out = ["## PIONEER-1 (privosegtor, acute optic neuritis)", ""]
    out.append(f"N={P2_N}, 1:1, two-sided α={P2_ALPHA}, {pct(P2_DROPOUT)} dropout (excluded). Endpoint: ≥15-letter "
               f"LCVA gain at Month 3. ACUITY's 18-letter difference (p=0.004) implies SE ≈ {ACUITY_SE:.1f} and "
               f"SD of change ≈ {LCVA_SD:.0f} letters; with an assumed placebo + steroid mean gain of "
               f"{PLACEBO_GAIN:g} letters → placebo responder rate ≈ **{pct(pc)}**.")
    out.append("")
    out.append("### Power")
    rows = []
    for pt in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        e = pt - pc
        rows.append([pct(pt), f"{100 * e:+.0f} pts", pct(d.power(e)), d.sample_size(e, 0.8)["n_total"],
                     d.sample_size(e, 0.9)["n_total"]])
    out.append(table(["Drug responder rate", "Difference", f"Power at N={P2_N}", "N for 80%", "N for 90%"], rows))
    out.append("")
    out.append(f"* **MDE** at N={P2_N}: 80% power → **{100 * d.mde(0.8):+.1f}** percentage points; "
               f"90% → {100 * d.mde(0.9):+.1f}.")
    out.append(f"* **Critical observed difference**: **{100 * d.critical_effect():+.1f}** points.")
    nri = TrialDesign(ep, P2_N, P2_ALPHA, 2, P2_RATIO, P2_DROPOUT, dropout_handling="nri")
    out.append(f"* If dropouts are counted as non-responders (NRI), MDE (80%) → {100 * nri.mde(0.8):+.1f} points.")
    out.append("")
    out.append("Power heatmap — drug responder rate × placebo responder rate (N fixed):")
    pts = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    pcs = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    g = power_grid(d, None, ("treat_outcome", pts), ("control_outcome", pcs))
    out.append(heat_table(g, [pct(x) for x in pts], [pct(y) for y in pcs], "drug", "placebo"))
    out.append("")
    out.append("### Probability of success")
    out.append(f"ACUITY effect with {pct(P2_HAIRCUT)} haircut → expected responder difference "
               f"**{100 * rd_mean:+.0f} points** (prior SD {100 * prior.sd_benefit:.0f}); "
               f"P(drug truly works) {pct(P2_PROB_WORKS)}.")
    out.append("")
    out.append(table(["Measure", "Value"], [
        ["**PoS (stat. sig. win)**", f"**{pct(res.pos)}** (quadrature {pct(res.pos_assurance)})"],
        ["Power at discounted point estimate", pct(d.power(rd_mean))],
        ["P(true effect favours drug)", pct(res.prob_true_benefit)],
        ["P(a win is a real benefit)", pct(res.ppv_benefit)],
    ]))
    out.append("")
    out.append("PoS sensitivity — haircut × P(drug truly works):")
    hcs, pws = [0.2, 0.4, 0.6, 0.75], [1.0, 0.85, 0.75, 0.6]
    rows = []
    for p in pws:
        row = [f"P(works) {pct(p)}"]
        for h in hcs:
            pc_, pr = pioneer_prior(haircut=h, prob=p)
            row.append(pct(assurance(d, pr)))
        rows.append(row)
    out.append(table(["", *[f"haircut {pct(h)}" for h in hcs]], rows))
    out.append("")
    out.append("PoS sensitivity — assumed placebo + steroid mean LCVA gain:")
    rows = []
    for gain in (6.0, 9.0, 12.0, 15.0, 18.0):
        pc_, pr = pioneer_prior(placebo_gain=gain)
        dd = TrialDesign(Binary(p_control=pc_), P2_N, P2_ALPHA, 2, P2_RATIO, P2_DROPOUT)
        rows.append([f"{gain:g} letters", pct(pc_), f"{100 * pr.mean_benefit:+.0f} pts", pct(assurance(dd, pr))])
    out.append(table(["Placebo mean gain", "Placebo responders", "Expected difference", "PoS"], rows))
    return "\n".join(out)


# ---- PIONEER-1 enrollment ----------------------------------------------------
# Months measured from study start (28 May 2026). LPI implied by primary
# completion Dec 2027 minus 3-month follow-up ≈ Sep 2027 ≈ month 15.1.
E_TARGET = P2_N
E_DEADLINE = 15.1
E_SITES = 50                       # ASSUMPTION
E_ACTIVATION = (3.0, 12.0)         # ASSUMPTION: first patient ~Sep 2026 (month ~3), sites open over 9 months
E_RAMP = 2.0                       # ASSUMPTION
E_SITE_CV = 1.0                    # ASSUMPTION: acute ON is rare; recruitment concentrated in neuro-ophth centres
E_RATE_CV = 0.4                    # ASSUMPTION: no public recruitment history for this indication


def pioneer1_enrollment():
    out = ["### PIONEER-1 enrollment forecast", ""]
    out.append(f"Target {E_TARGET} randomized; {E_SITES} sites opening months {E_ACTIVATION[0]:g}–{E_ACTIVATION[1]:g} "
               f"after study start (28 May 2026), {E_RAMP:g}-month ramp-up, site CV {E_SITE_CV}, rate uncertainty "
               f"CV {E_RATE_CV}. Deadline: LPI by ~Sep 2027 (month {E_DEADLINE}) to make the Dec 2027 "
               "primary completion.")
    out.append("")
    start = np.datetime64("2026-05-28")

    def to_date(m):
        return str(start + np.timedelta64(int(round(m * 30.44)), "D"))[:7] if np.isfinite(m) else "never"

    rows = []
    for rate in (0.15, 0.25, 0.35, 0.50):
        plan = EnrollmentPlan(E_TARGET, E_SITES, rate, *E_ACTIVATION, E_RAMP, E_SITE_CV, E_RATE_CV)
        r = simulate_enrollment(plan, 3000, SEED)
        s = r.summary()
        rows.append([f"{rate:g}", to_date(plan.expected_completion()), to_date(s["lpi_p50"]),
                     f"{to_date(s['lpi_p10'])} – {to_date(s['lpi_p90'])}", pct(r.prob_complete_by(E_DEADLINE))])
    out.append(table(["Patients/site/month", "Planned LPI", "Simulated LPI (median)", "80% interval",
                      "P(LPI by Sep 2027)"], rows))
    out.append("")
    rows = []
    for rate in (0.15, 0.25, 0.35):
        plan = EnrollmentPlan(E_TARGET, E_SITES, rate, *E_ACTIVATION, E_RAMP, E_SITE_CV, E_RATE_CV)
        ns = sites_needed(plan, E_DEADLINE, 0.8, n_sims=600, seed=SEED, max_sites=1000)
        rows.append([f"{rate:g}", ns if ns else ">1000 / not reachable"])
    out.append("Sites needed for an 80% chance of LPI by Sep 2027 (same activation window):")
    out.append("")
    out.append(table(["Patients/site/month", "Sites needed"], rows))
    out.append("")
    out.append("To re-forecast with actual data: `python -m trialpower enroll --target 210 --sites <n> --rate <r> "
               "--activation 3 12 --ramp 2 --site-cv 1 --rate-cv 0.4 --elapsed 6 --enrolled <randomized so far>`.")
    return "\n".join(out)


if __name__ == "__main__":
    print("# Oculis trial analyses: PREDICT-1 and PIONEER-1\n")
    print("_Generated by `analyses/oculis_trials.py`. Inputs marked ASSUMPTION in the script are not public; "
          "treat outputs as scenario analysis, not predictions._\n")
    print(predict1())
    print()
    print(pioneer1())
    print()
    print(pioneer1_enrollment())
