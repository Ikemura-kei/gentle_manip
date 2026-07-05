"""Reconcile the experiment registry (experiments.csv) with the log tree on disk.

Runs launch-time registration automatically, but you sometimes delete deprecated run dirs by
hand — this pass keeps the table honest:
  - drops table rows whose run_dir no longer exists (deleted runs);
  - back-fills rows for ID-shaped run dirs present on disk but missing from the table
    (a launch that didn't register, or a lost table).

Usage (any env):
    uv run --project envs/sim python -m gentle_manip.scripts.reconcile_experiments
    uv run --project envs/sim python -m gentle_manip.scripts.reconcile_experiments --list
"""
from __future__ import annotations

import argparse

from gentle_manip.utils.experiment_registry import TABLE_PATH, format_table, reconcile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="just print the table, don't reconcile")
    ap.add_argument("--base", default="logs", help="log root to scan (default: logs)")
    args = ap.parse_args()

    if not args.list:
        res = reconcile(base=args.base)
        print(f"reconciled {TABLE_PATH}:")
        print(f"  kept:    {res['kept']}")
        print(f"  dropped: {len(res['dropped'])} {res['dropped'] or ''}")
        print(f"  added:   {len(res['added'])} {res['added'] or ''}")
        print()
    print(format_table())


if __name__ == "__main__":
    main()
