# trialpower — clinical trial power & probability of success

Two connected tools:

1. **Power analysis.** Standard frequentist power and sample size for a two-arm superiority trial.
   You set alpha, sidedness, N, allocation ratio, dropout and target power for each trial.
   - Continuous endpoints (exact noncentral-t), binary/responder endpoints (pooled z), and time-to-event endpoints (log-rank / Schoenfeld; includes an event-probability helper that works from medians, accrual and follow-up).
   - Dropout model: one rate or per-arm rates, analysed as **excluded** (complete-case), **ITT dilution**, or **non-responder imputation** (binary).
   - **Power heatmap**: drug-arm vs control-arm outcome, or effect vs N, dropout, or SD / control rate.
   - Outputs: power, N for the target power, **MDE** (the smallest *true* effect you can detect), and the **critical observed effect** (the smallest *observed* effect that is significant).
2. **Probability of success (PoS).** Turns Phase II data into a prior on the true Phase III effect and averages power over that uncertainty.
   - Computed by **Monte Carlo** (simulates the trial outcomes) and by numerical integration (a deterministic cross-check).
   - Phase II → III levers: an effect **haircut**, an evidence **weight** (power prior), and **P(drug truly works)** (a mixture with a null effect).
   - Success = statistically significant **and** an observed effect ≥ a **minimum clinically meaningful effect**. You can optionally require **k pivotal trials** to all succeed; they are correlated through the shared true effect.
   - Outputs: PoS, the **success threshold** (the minimum observed effect that wins), P(true effect favours drug) (the PoS ceiling), P(win is real), PoS vs N, and a haircut × P(works) sensitivity grid.

3. **Enrollment simulation.** Monte Carlo recruitment forecasts using the Poisson–gamma model: site activation, ramp-up, variation between sites, and uncertainty in the planned rate.
   - Outputs: the last-patient-in distribution and P(LPI by a deadline), enrollment and event fan charts, final-analysis timing for time-to-event trials, and a month-by-month CSV.
   - **How many sites do I need**, and a **re-forecast** that updates the projection with actual enrollment to date.

See [`docs/METHODS.md`](docs/METHODS.md) for the statistics.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py                                   # interactive app
python -m trialpower run examples/trials.yaml --csv results.csv   # portfolio from YAML
python -m trialpower enroll --target 400 --sites 40 --rate 0.8 --ramp 2 --events 250   # enrollment forecast
pytest                                                 # tests
```

## Per-trial config (YAML)

```yaml
defaults: {alpha: 0.05, sides: 2, dropout: 0.1, haircut: 0.2, target_power: 0.9}
trials:
  - name: ONC-301 PFS
    endpoint: survival          # continuous | binary | survival
    event_prob: 0.70
    n_total: 450
    phase2: {hr: 0.62, lower: 0.42, upper: 0.92}
    prob_effect: 0.8
    mcid: 0.80                  # minimum clinically meaningful HR
```

The `phase2:` block accepts `{estimate, se}`, `{mean_t, mean_c, sd, n_t, n_c}`, `{x_t, n_t, x_c, n_c}`,
`{hr, lower, upper}` or `{hr, events}`. See `examples/trials.yaml`.

## Python API

```python
from trialpower import *
ep = Continuous(sd=1.1, higher_is_better=False)
design = TrialDesign(ep, n_total=400, alpha=0.05, dropout=0.1)
design.power(-0.44); design.sample_size(-0.44, 0.9); design.mde(0.9); design.critical_effect()

prior = EffectPrior.from_phase2(ep, Phase2Evidence.from_means(-0.95, -0.40, 1.1, 60, 60), haircut=0.2)
res = simulate_pos(design, prior, mcid=-0.3, n_trials=1, n_sims=100_000)
res.summary()
```

Effects are always Treatment − Control (HR: Treatment vs Control).
