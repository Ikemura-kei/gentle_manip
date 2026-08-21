"""Read the shelf-ablation arms out of their bench logs and print the comparison.

    uv run --project envs/sim python -m gentle_manip.scripts.report_shelf_ablation [tags...]

Each `run_grasp_bench.sh <tag>` writes `logs/grasp_bench/<tag>_eval.log`, whose last line names the
eval run directory. This resolves tag -> run dir -> summary.json so the arms can be compared by the
name they were launched under rather than by timestamp.

WHY THE STANDARD ERROR IS PRINTED. At 25 episodes per arm the sustained-stress spread is ~9 kPa on a
~27 kPa mean, so the smallest difference two arms can distinguish is ~13 %. A mean difference below
that is not a result, and reading these tables without the SE column invites exactly that mistake.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_BENCH = _REPO / "logs" / "grasp_bench"

DEFAULT_TAGS = ["shelf_t0_o0", "shelf_t0_oX", "shelf_t55_o0", "shelf_t55_oX"]
METRICS = [("stress_max_tmax", "peak"), ("stress_top20_ttop20", "sustained"),
           ("stress_top10_tmax", "top10@peak"), ("stress_mean_tmean", "bulk")]


def resolve(tag: str) -> Path | None:
    log = _BENCH / f"{tag}_eval.log"
    if not log.exists():
        return None
    hits = re.findall(r"logs/scripted_policy/[\w.-]+", log.read_text(errors="replace"))
    for h in reversed(hits):
        p = _REPO / h
        if (p / "summary.json").exists():
            return p
    return None


def stats(run: Path) -> dict:
    """Per-metric (mean, std) recomputed from episodes.csv, SUCCESS-GATED.

    Read from the CSV rather than summary.json because only the two headline metrics carry a `_std`
    there (`metrics.STRESS_COLS` want_pct flag), and a mean with no spread beside it is precisely
    what makes a noise-level difference look like a finding.
    """
    with open(run / "episodes.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("success") in ("1", "1.0", "True", "true")]
    out = {}
    for key, _ in METRICS:
        v = []
        for r in rows:
            try:
                x = float(r.get(key, ""))
            except (TypeError, ValueError):
                continue
            if not math.isnan(x):
                v.append(x)
        if v:
            a = np.asarray(v)
            out[key] = (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, a.size)
    return out


def main() -> None:
    tags = sys.argv[1:] or DEFAULT_TAGS
    rows = []
    for t in tags:
        run = resolve(t)
        if run is None:
            print(f"  {t:16s} (no completed run yet)")
            continue
        rows.append((t, json.loads((run / "summary.json").read_text()), run, stats(run)))
    if not rows:
        return

    base = {k: v[0] for k, v in rows[0][3].items()}
    base_se = {k: v[1] / math.sqrt(max(v[2], 1)) for k, v in rows[0][3].items()}

    print(f"\n{'arm':16s} {'theta':>6s} {'open':>6s} {'n':>4s} {'succ':>6s}  "
          + "  ".join(f"{lab:>22s}" for _, lab in METRICS))
    print("-" * (42 + 24 * len(METRICS)))
    for tag, s, _, st in rows:
        cells = []
        for key, _ in METRICS:
            if key not in st:
                cells.append(f"{'-':>22s}")
                continue
            m, sd, n = st[key]
            se = sd / math.sqrt(max(n, 1))
            b = base.get(key)
            delta = f"{100 * (m - b) / b:+5.1f}%" if b else "     "
            # A delta is only meaningful against the spread: flag the ones inside the noise.
            # Two independent arms -> the difference's SE is sqrt(se_a^2 + se_b^2); with comparable
            # spreads that is ~sqrt(2)*se, so use that rather than the single-arm SE.
            se_d = math.sqrt(se ** 2 + (base_se.get(key, se)) ** 2)
            sig = "" if (b is not None and abs(m - b) > 2 * se_d) else "~"
            cells.append(f"{m:8.0f}{'+-' + format(sd, '.0f'):>7s} {delta}{sig:1s}")
        print(f"{tag:16s} {s.get('shelf_deg', 0):6.0f} {s.get('shelf_open', 0) * 1e3:6.1f} "
              f"{s['n_episodes']:4d} {s['success_rate']:6.3f}  " + "  ".join(cells))

    print("\n  '~' = the difference from the first arm is inside 2 standard errors, i.e. not a result.")
    print("  peak-stress phase (a shelf that moves damage from `lift` to `hold` relocated it):")
    for tag, s, _, _st in rows:
        d = s.get("peak_stress_phase_dist") or {}
        print(f"    {tag:16s} " + "  ".join(f"{k} {v:.2f}" for k, v in sorted(d.items())))
    print("\n  runs:")
    for tag, _, run, _st in rows:
        print(f"    {tag:16s} {run.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
