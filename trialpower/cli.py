"""Command line: ``python -m trialpower run examples/trials.yaml [--csv out.csv]``."""
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
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    main()
