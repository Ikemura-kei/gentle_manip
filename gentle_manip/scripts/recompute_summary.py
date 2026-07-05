"""Recompute an eval run's summary.json from its episodes.csv, using the CURRENT
metrics.aggregate() — e.g. after the stress metric became success-gated (+P90/P95). Rewrites
summary.json in place (the episodes.csv table is NOT touched); provenance meta (checkpoint,
experiment, seed, ...) is preserved from the old summary. The old all-episode stress value
survives as stress_peak_mean_all in the new summary, so nothing is lost.

Usage (any env; e.g. envs/sim):
    uv run --project envs/sim python -m gentle_manip.scripts.recompute_summary <eval_dir> [<eval_dir> ...]
where <eval_dir> contains episodes.csv + summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gentle_manip.evaluation.metrics import aggregate, write_summary

# keys aggregate() computes itself — everything else in the old summary is provenance meta.
_COMPUTED = {"n_episodes", "success_rate", "ever_success_rate", "mean_episode_reward",
             "is_soft_task", "stress_n_success", "stress_peak_mean", "stress_peak_std",
             "stress_peak_p90", "stress_peak_p95", "stress_mean_mean",
             "stress_peak_mean_all", "stress_mean_mean_all", "stress_peak_std_all"}


def _num(v):
    if v in ("", "None", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _records(csv_path: Path) -> list[dict]:
    out = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            out.append({
                "success": int(float(r["success"])) if r.get("success") not in ("", None) else 0,
                "ever_success": int(float(r["ever_success"])) if r.get("ever_success") not in ("", None) else 0,
                "episode_reward": _num(r.get("episode_reward")) or 0.0,
                "stress_peak": _num(r.get("stress_peak")),
                "stress_mean": _num(r.get("stress_mean")),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_dirs", nargs="+", type=Path)
    args = ap.parse_args()
    for d in args.eval_dirs:
        summ_p, csv_p = d / "summary.json", d / "episodes.csv"
        if not csv_p.exists():
            print(f"[skip] no episodes.csv in {d}")
            continue
        old = json.loads(summ_p.read_text()) if summ_p.exists() else {}
        meta = {k: v for k, v in old.items() if k not in _COMPUTED}   # preserve provenance
        recs = _records(csv_p)
        new = aggregate(recs, **meta)
        write_summary(new, summ_p)
        og = old.get("stress_peak_mean")
        print(f"[ok] {d}")
        print(f"     success_rate {new['success_rate']:.3f}  n={new['n_episodes']}")
        if new.get("is_soft_task"):
            print(f"     stress_peak (success-gated): mean {new['stress_peak_mean']:.0f}  "
                  f"P90 {new['stress_peak_p90']:.0f}  P95 {new['stress_peak_p95']:.0f}  "
                  f"(n_succ={new['stress_n_success']})")
            print(f"     stress_peak (all-episode)  : {new['stress_peak_mean_all']:.0f}"
                  + (f"   [old summary reported {og:.0f}]" if isinstance(og, (int, float)) else ""))


if __name__ == "__main__":
    main()
