import math

import numpy as np
import pytest

from trialpower.enrollment import (EnrollmentPlan, EventModel, posterior_rate, simulate_enrollment,
                                   sites_needed)


def test_deterministic_limit_matches_expected_curve():
    # tiny variability -> simulated LPI ~ deterministic completion time
    p = EnrollmentPlan(400, 40, 0.8, 0, 6, site_ramp=2, site_cv=1e-4, rate_uncertainty=1e-4)
    r = simulate_enrollment(p, 1000, seed=1)
    assert np.median(r.completion) == pytest.approx(p.expected_completion(), abs=0.5)


def test_simple_case_closed_form():
    # all sites open at 0, no ramp: N = sites * rate * t  ->  t = 400 / (40*0.5) = 20
    p = EnrollmentPlan(400, 40, 0.5, 0, 0, 0, 1e-4, 1e-4)
    assert p.expected_completion() == pytest.approx(20, rel=1e-6)


def test_uncertainty_widens_spread_not_much_the_median():
    base = EnrollmentPlan(300, 30, 1.0, 0, 6, 1, site_cv=0.8, rate_uncertainty=0.0)
    wide = EnrollmentPlan(300, 30, 1.0, 0, 6, 1, site_cv=0.8, rate_uncertainty=0.3)
    a = simulate_enrollment(base, 2000, 1).completion
    b = simulate_enrollment(wide, 2000, 1).completion
    assert np.ptp(np.quantile(b, [0.1, 0.9])) > 1.5 * np.ptp(np.quantile(a, [0.1, 0.9]))


def test_reforecast_learns_from_slow_enrollment():
    p = EnrollmentPlan(400, 40, 1.0, 0, 6, 0, 0.8, 0.3)
    mean, cv = posterior_rate(p, elapsed=10, enrolled=120)  # plan expected ~280 by month 10
    assert mean < 1.0 and cv < 0.3
    slow = simulate_enrollment(p, 1000, 1, elapsed=10, enrolled=120)
    fresh = simulate_enrollment(p, 1000, 1)
    assert np.median(slow.completion) > np.median(fresh.completion)
    assert slow.enroll_q[0.5][0] == 120


def test_curves_monotone_and_capped():
    p = EnrollmentPlan(200, 20, 1.0)
    r = simulate_enrollment(p, 500, 2)
    for q in r.enroll_q.values():
        assert np.all(np.diff(q) >= 0) and q[-1] <= 200
    assert 0 <= r.prob_complete_by(12) <= r.prob_complete_by(24) <= 1


def test_event_accrual():
    # instant enrollment, no dropout: events by time t ~ N * (1 - exp(-lambda t)) averaged over arms
    p = EnrollmentPlan(1000, 1000, 1000.0, 0, 0, 0, 1e-4, 1e-4)
    ev = EventModel(median_control=12, hr=1.0, target_events=500, dropout_per_year=0.0)
    r = simulate_enrollment(p, 200, 3, events=ev, horizon=60)
    assert np.median(r.event_completion) == pytest.approx(12, abs=0.5)  # half have an event by the median
    s = r.summary()
    assert s["analysis_p50"] > s["lpi_p50"]


def test_sites_needed():
    p = EnrollmentPlan(300, 10, 1.0, 0, 3, 1, 0.6, 0.1)
    n = sites_needed(p, deadline=12, prob=0.8, n_sims=300)
    assert n is not None
    r = simulate_enrollment(EnrollmentPlan(300, n, 1.0, 0, 3, 1, 0.6, 0.1), 2000, 9)
    assert r.prob_complete_by(12) == pytest.approx(0.8, abs=0.08)
    assert sites_needed(EnrollmentPlan(300, 10, 1.0, 0, 20), deadline=12, n_sims=100, max_sites=50) is None
