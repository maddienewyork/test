import math

import numpy as np
import pytest

from trialpower import (Binary, Continuous, EffectPrior, Phase2Evidence, Survival, TrialDesign,
                        assurance, event_probability, simulate_pos)
from trialpower.portfolio import evaluate, load_yaml

# --- classical power against textbook values ---------------------------------

def test_continuous_sample_size_matches_textbook():
    # d = 0.5, two-sided alpha .05, 80% power -> 64 per arm (noncentral t, e.g. G*Power)
    d = TrialDesign(Continuous(sd=1.0), n_total=128)
    assert d.sample_size(0.5, 0.8)["n_t"] == 64
    assert d.power(0.5) == pytest.approx(0.8015, abs=1e-3)


def test_binary_sample_size_matches_textbook():
    # 0.40 vs 0.60, two-sided .05, 80% power -> 97 per arm (pooled z, no continuity correction)
    d = TrialDesign(Binary(p_control=0.40), n_total=100)
    assert d.sample_size(0.20, 0.8)["n_t"] == 97


def test_survival_events_match_schoenfeld():
    # HR 0.7, 1:1, two-sided .05, 80% -> 247 events
    d = TrialDesign(Survival(event_prob=1.0), n_total=100)
    assert d.sample_size(0.7, 0.8)["events"] == 247


def test_dropout_and_ratio():
    d = TrialDesign(Continuous(sd=1.0), n_total=100, ratio=2, dropout=0.2)
    ss = d.sample_size(0.5, 0.9)
    assert ss["n_t"] == pytest.approx(2 * ss["n_c"], abs=2)
    assert d.power(0.5, ss["n_total"]) >= 0.9


def test_mde_and_critical_effect_are_consistent():
    d = TrialDesign(Continuous(sd=1.0, higher_is_better=False), n_total=200)
    mde = d.mde(0.8)
    assert mde < 0
    assert d.power(mde) == pytest.approx(0.8, abs=1e-6)
    # an observed effect exactly at the critical value has p = alpha -> power at that true effect ~ 50%
    assert d.power(d.critical_effect()) == pytest.approx(0.5, abs=0.01)


def test_event_probability():
    # no accrual, follow-up = median, HR 1 -> 50%
    assert event_probability(12, 1.0, 0, 12) == pytest.approx(0.5)


# --- probability of success ---------------------------------------------------

def test_assurance_closed_form_continuous():
    # z-approx closed form: PoS = Phi((m - c) / sqrt(se^2 + s^2)); large n so t ~ z
    from scipy import stats
    ep = Continuous(sd=1.0)
    d = TrialDesign(ep, n_total=2000)
    prior = EffectPrior(0.1, 0.05)
    se = math.sqrt(4 / 2000)
    c = 1.959964 * se
    closed = stats.norm.cdf((0.1 - c) / math.sqrt(se**2 + 0.05**2))
    assert assurance(d, prior) == pytest.approx(closed, abs=2e-3)


def test_fixed_prior_pos_equals_power():
    d = TrialDesign(Survival(event_prob=0.6), n_total=500)
    prior = EffectPrior.fixed(d.endpoint, 0.75)
    assert assurance(d, prior) == pytest.approx(d.power(0.75), abs=1e-9)


@pytest.mark.parametrize("ep,ev", [
    (Continuous(sd=1.1, higher_is_better=False), Phase2Evidence.from_means(-0.9, -0.4, 1.1, 60, 60)),
    (Binary(p_control=0.15), Phase2Evidence.from_proportions(22, 50, 7, 50)),
    (Survival(event_prob=0.7), Phase2Evidence.from_hr_ci(0.62, 0.42, 0.92)),
])
@pytest.mark.parametrize("n_trials", [1, 2])
def test_monte_carlo_matches_quadrature(ep, ev, n_trials):
    d = TrialDesign(ep, n_total=300, dropout=0.1)
    prior = EffectPrior.from_phase2(ep, ev, haircut=0.2, weight=0.8, prob_effect=0.85)
    r = simulate_pos(d, prior, n_trials=n_trials, n_sims=200_000, seed=3)
    assert r.pos == pytest.approx(r.pos_assurance, abs=0.01)
    assert r.pos <= r.pos_any
    assert r.pos <= r.prob_true_benefit + 0.03  # can't (much) beat P(drug works)


def test_mcid_reduces_pos_and_raises_ppv():
    ep = Continuous(sd=1.0)
    d = TrialDesign(ep, n_total=1000)
    prior = EffectPrior(0.2, 0.15)
    base = simulate_pos(d, prior, n_sims=100_000, seed=1)
    strict = simulate_pos(d, prior, mcid=0.2, n_sims=100_000, seed=1)
    assert strict.pos < base.pos
    assert strict.success_threshold == pytest.approx(0.2)
    assert strict.pos == pytest.approx(strict.pos_assurance, abs=0.01)
    assert strict.ppv_benefit >= base.ppv_benefit


def test_two_trials_correlated():
    ep = Continuous(sd=1.0)
    d = TrialDesign(ep, n_total=300)
    prior = EffectPrior(0.25, 0.15)
    one = assurance(d, prior, n_trials=1)
    two = assurance(d, prior, n_trials=2)
    assert two > one**2  # positive correlation through the shared true effect


def test_example_portfolio_runs():
    rows = load_yaml("examples/trials.yaml")
    assert len(rows) == 3
    for row in rows:
        out = evaluate(row, n_sims=20_000)
        assert 0 <= out["pos_mc"] <= 1
        assert abs(out["pos_mc"] - out["pos_assurance"]) < 0.02


# --- dropout models -------------------------------------------------------------

def test_dropout_exclude_per_arm():
    d = TrialDesign(Continuous(sd=1.0), n_total=200, dropout_t=0.3, dropout_c=0.1)
    assert d.arms() == pytest.approx((70, 90))


def test_dropout_dilution_scales_effect():
    # ITT dilution: power with effect 0.5 and 20% drug-arm dropout == no-dropout power at effect 0.4
    dil = TrialDesign(Continuous(sd=1.0), n_total=200, dropout=0.2, dropout_handling="dilute")
    ref = TrialDesign(Continuous(sd=1.0), n_total=200)
    assert dil.power(0.5) == pytest.approx(ref.power(0.4))
    # dilution typically needs more patients than complete-case for the same dropout
    exc = TrialDesign(Continuous(sd=1.0), n_total=200, dropout=0.2)
    assert dil.sample_size(0.5, 0.9)["n_total"] > exc.sample_size(0.5, 0.9)["n_total"]


def test_dropout_nri_binary():
    d = TrialDesign(Binary(p_control=0.2), n_total=300, dropout_t=0.2, dropout_c=0.1, dropout_handling="nri")
    ref = TrialDesign(Binary(p_control=0.2 * 0.9), n_total=300)
    # pT 0.5 -> 0.4 after NRI, pC 0.2 -> 0.18
    assert d.power(0.3) == pytest.approx(ref.power(0.4 - 0.18))
    with pytest.raises(ValueError):
        TrialDesign(Continuous(sd=1), n_total=100, dropout_handling="nri")


def test_dropout_dilution_flows_into_pos():
    ep = Continuous(sd=1.0)
    d = TrialDesign(ep, n_total=300, dropout=0.25, dropout_handling="dilute")
    prior = EffectPrior(0.4, 0.1)
    r = simulate_pos(d, prior, n_sims=100_000, seed=2)
    assert r.pos == pytest.approx(r.pos_assurance, abs=0.01)
    assert r.pos < assurance(TrialDesign(ep, n_total=300), prior)


# --- heatmap --------------------------------------------------------------------

def test_power_grid_arm_outcomes_binary():
    from trialpower import power_grid
    d = TrialDesign(Binary(p_control=0.3), n_total=200)
    pt, pc = [0.4, 0.5, 0.6], [0.2, 0.3]
    g = power_grid(d, None, ("treat_outcome", pt), ("control_outcome", pc))
    assert g.shape == (2, 3)
    assert g[1, 2] == pytest.approx(TrialDesign(Binary(p_control=0.3), n_total=200).power(0.3))
    assert np.all(np.diff(g, axis=1) > 0)  # more drug effect -> more power


def test_power_grid_effect_vs_n_and_survival_medians():
    from trialpower import power_grid
    d = TrialDesign(Continuous(sd=1.0), n_total=100)
    g = power_grid(d, 0.5, ("effect", [0.3, 0.5]), ("n_total", [100, 200]))
    assert g[1, 1] == pytest.approx(d.power(0.5, 200))
    s = TrialDesign(Survival(event_prob=0.7), n_total=400)
    g = power_grid(s, None, ("treat_outcome", [15.0]), ("control_outcome", [10.0]))
    assert g[0, 0] == pytest.approx(s.power(10 / 15))
