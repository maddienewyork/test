"""Command line.

    python -m trialpower run examples/trials.yaml [--csv out.csv]
    python -m trialpower enroll --target 400 --sites 40 --rate 0.8 --ramp 2 [--events 300]
"""
from __future__ import annotations

import argparse
import csv
import sys

from .portfolio import evaluate, fmt, load_yaml

_PCT = ("power", "p_true_benefit", "p_true_mcid", "pos_mc", "pos_assurance", "ppv")


def _cell(k, v):
    if k in _PCT and isinstance(v, float):
        return f"{100 * v:.1f}%"
    return fmt(v)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trialpower", description="Clinical trial power & probability of success")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="evaluate every trial in a YAML file")
    run.add_argument("config")
    run.add_argument("--sims", type=int, default=100_000, help="Monte Carlo simulations per trial")
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--csv", help="also write results to this CSV file")
    en = sub.add_parser("enroll", help="simulate an enrollment curve")
    en.add_argument("--target", type=int, required=True, help="patients to randomise")
    en.add_argument("--sites", type=int, required=True)
    en.add_argument("--rate", type=float, required=True, help="randomised patients / site / month at full speed")
    en.add_argument("--activation", type=float, nargs=2, default=(0, 6), metavar=("START", "END"),
                    help="months over which sites open (default 0 6)")
    en.add_argument("--ramp", type=float, default=0.0, help="months for a site to reach full speed")
    en.add_argument("--site-cv", type=float, default=0.8, help="between-site rate variability (CV)")
    en.add_argument("--rate-cv", type=float, default=0.2, help="uncertainty in the planned rate (CV)")
    en.add_argument("--screen-fail", type=float, default=0.0)
    en.add_argument("--elapsed", type=float, default=0.0, help="re-forecast: months elapsed so far")
    en.add_argument("--enrolled", type=int, default=0, help="re-forecast: patients randomised so far")
    en.add_argument("--events", type=int, help="time-to-event: target events for the final analysis")
    en.add_argument("--median", type=float, default=12.0, help="control median survival (months)")
    en.add_argument("--hr", type=float, default=0.7)
    en.add_argument("--dropout-per-year", type=float, default=0.05)
    en.add_argument("--sims", type=int, default=2000)
    en.add_argument("--seed", type=int, default=1)
    en.add_argument("--csv", help="write the month-by-month table to this CSV")
    args = ap.parse_args(argv)

    if args.cmd == "enroll":
        return _enroll(args)

    results = [evaluate(r, args.sims, args.seed) for r in load_yaml(args.config)]
    for r in results:
        print(f"\n== {r['name']} ({r['endpoint']}) ==")
        for k, v in r.items():
            if k not in ("name", "endpoint"):
                print(f"  {k:<20} {_cell(k, v)}")
    if args.csv:
        keys = list(dict.fromkeys(k for r in results for k in r))
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"\nwrote {args.csv}", file=sys.stderr)


def _enroll(args):
    from .enrollment import EnrollmentPlan, EventModel, simulate_enrollment

    plan = EnrollmentPlan(args.target, args.sites, args.rate, *args.activation, args.ramp,
                          args.site_cv, args.rate_cv, args.screen_fail)
    ev = None if args.events is None else EventModel(args.median, args.hr, args.events, 1.0, args.dropout_per_year)
    r = simulate_enrollment(plan, args.sims, args.seed, args.elapsed, args.enrolled, ev)
    s = r.summary()
    print(f"Planned (average-rate) last patient in: month {plan.expected_completion(r.mean_rate, args.enrolled, args.elapsed):.1f}")
    print(f"Simulated LPI: median {s['lpi_p50']:.1f}, 80% interval {s['lpi_p10']:.1f}-{s['lpi_p90']:.1f} months")
    if args.screen_fail:
        print(f"Patients to screen: ~{plan.screened_needed}")
    if args.elapsed:
        print(f"Updated site rate: {r.mean_rate:.3f}/site/month (CV {r.rate_cv:.2f})")
    if ev:
        print(f"Final analysis ({args.events} events): median {s['analysis_p50']:.1f}, "
              f"80% interval {s['analysis_p10']:.1f}-{s['analysis_p90']:.1f} months")
    rows = r.monthly_table()
    print(f"\n{'month':>5} {'planned':>8} {'p10':>6} {'p50':>6} {'p90':>6} {'P(LPI)':>7}")
    for row in rows[:: max(1, len(rows) // 30)]:
        print(f"{row['month']:>5} {row['planned']:>8.0f} {row['p10']:>6.0f} {row['p50']:>6.0f} {row['p90']:>6.0f} "
              f"{100 * row['p_lpi_by_month']:>6.0f}%")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
