### What the tool computes

**1. Classical power (Power tab).** For a two-arm superiority trial:

| Endpoint | Test | Effect | Notes |
|---|---|---|---|
| Continuous | two-sample t-test (exact noncentral *t*) | mean difference T − C | common SD |
| Binary | pooled two-proportion z-test | risk difference pT − pC | no continuity correction |
| Time-to-event | log-rank (Schoenfeld) | hazard ratio T vs C | events = evaluable N × P(event) |

Power is the probability of a *significant result in the favourable direction*
(two-sided α means α/2 in that direction). Dropout reduces evaluable N;
sample sizes are rounded up per arm then inflated for dropout.

**Dropout models** (per-arm rates allowed — drug-arm dropout is often higher):

| Handling | What happens | Typical use |
|---|---|---|
| Excluded (complete-case) | analysed N per arm = N × (1 − dropout); effect unchanged; for time-to-event dropouts are censored, so fewer events | per-protocol / completers, MMRM-style estimands |
| ITT dilution | everyone analysed, drug-arm dropouts get no drug benefit → effect × (1 − drug-arm dropout) | treatment-policy estimand, retrieved-dropout data |
| ITT non-responder imputation (binary) | dropouts in both arms counted as failures: pT·(1−dT), pC·(1−dC) | responder endpoints (e.g. IGA, ACR20) |

**Power heatmap** — power in each cell over two assumptions: drug-arm vs
control-arm outcome (means, response rates, or median survival with
HR = median_C / median_T), or treatment effect vs N, dropout, or the SD / control
rate / event probability.

* **MDE** – the smallest *true* effect with the target power at the planned N.
* **Critical observed effect** – the smallest *observed* effect that reaches p < α.
  Useful reality check: an observed effect only just past this line is a "win"
  most stakeholders won't find impressive.

**2. Phase II → Phase III prior.** Phase II gives an estimate θ̂₂ with standard
error SE₂. The prior for the true Phase III benefit is

  with probability *p*: N( θ̂₂ × (1 − haircut), SE₂² / weight ), otherwise exactly 0.

* **haircut** corrects optimism (only programmes with good Phase II advance —
  regression to the mean; broader Phase III populations). 10–40% is typical.
* **weight** (power-prior) inflates uncertainty, e.g. when the Phase II endpoint,
  dose or population differs from Phase III.
* **P(drug truly works)** adds scepticism a single Phase II can't remove
  (base rates, surrogate endpoints, novel mechanisms).

**3. Probability of success (PoS / "assurance").**
PoS = E_prior[ P(success | true effect) ]. Success per trial = statistically
significant **and** (optionally) observed effect ≥ the minimum clinically
meaningful effect. With k pivotal trials all must succeed; they share the true
effect, so PoS(2 trials) > PoS(1)².

* *Monte Carlo*: draw a true effect, simulate the trial(s) at summary-statistic
  level (normal mean + χ² variance for t-tests, binomial counts for responders,
  normal log-HR with variance 1/(d·p(1−p)) for survival), apply the rule, repeat.
* *Quadrature*: Gauss–Hermite integration of the exact power function over the
  prior — a deterministic cross-check (they should agree within MC error).

Outputs worth watching:

* **P(true effect favours drug)** — the ceiling PoS approaches as N → ∞. If PoS
  is far below power-at-the-point-estimate, the issue is effect uncertainty, not N.
* **Success threshold** — the minimum observed effect that wins.
* **P(win is real)** — among simulated wins, the share with a true effect at least
  the minimum meaningful effect.

**4. Enrollment simulation (Enrollment tab / `python -m trialpower enroll`).**
Poisson–gamma recruitment model (Anisimov & Fedorov 2007):

* sites open uniformly between the first and last activation month, then ramp
  linearly to full speed;
* each site recruits as a Poisson process at `mean rate × site factor`, with the site
  factor ~ Gamma(mean 1, CV = site variability);
* the mean rate itself ~ Gamma(planned rate, CV = rate uncertainty), which is usually
  the biggest reason trials finish late;
* **re-forecast:** with *n* patients after *t* months, the mean-rate prior is updated
  to Gamma(a₀ + n, b₀ + site-months of exposure), and the rest of the trial is simulated;
* **events (time-to-event):** each patient gets an exponential event time (control
  median, HR) and an exponential loss-to-follow-up time; the final analysis happens
  when the target number of events is reached;
* **sites needed:** the smallest number of sites giving P(last patient in by the
  deadline) ≥ the chosen probability, found by bisection over simulations.

Outputs: planned vs simulated last-patient-in (median, 80% interval), P(LPI by a
deadline), fan charts of patients and events, timing of the final analysis, and a
month-by-month table (CSV).

### Not covered (yet)
Group-sequential / interim analyses, per-country or per-site caps on enrollment, non-inferiority margins, multiplicity across
endpoints, non-proportional hazards, regulatory/commercial success beyond the
primary endpoint.
