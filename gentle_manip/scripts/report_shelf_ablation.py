"""Read the shelf-ablation arms out of their bench logs and print the comparison.

    uv run --project envs/sim python -m gentle_manip.scripts.report_shelf_ablation [tags...]

Each `run_grasp_bench.sh <tag>` writes `logs/grasp_bench/<tag>_eval.log`, whose last line names the
eval run directory. This resolves tag -> run dir -> summary.json so the arms can be compared by the
name they were launched under rather than by timestamp.

WHY THE COMPARISON IS PAIRED. The harness fixes the scenario seeds, so arm A's episode k and arm B's
episode k face the SAME object pose, shape, scale and material — the arms are matched samples, not
independent ones. Treating them as independent (SE = sd/sqrt(n)) puts the resolution floor at ~13 %
of the sustained-stress mean, which is wildly pessimistic: two independent 25-episode runs of the
IDENTICAL configuration were measured at 23762 and 23761 Pa, i.e. a run-to-run floor near 1 Pa.

So this reports the mean PER-EPISODE difference and its own standard error. The episode-to-episode
spread (~9 kPa) is real variation across scenarios and is deliberately differenced out; what
survives is the effect of the configuration change, which is what the arms are being compared on.
The unpaired spread is still shown, in parentheses, as a description of the scenario distribution.
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


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def by_seed(run: Path) -> dict:
    """{scenario_seed -> {metric: value}} for SUCCESSFUL episodes.

    Keyed by scenario seed so two arms can be matched episode-for-episode; an episode that failed in
    either arm is dropped from that pairing, which keeps the stress comparison success-gated on both
    sides (the same convention metrics.aggregate uses).
    """
    with open(run / "episodes.csv", newline="") as f:
        out = {}
        for r in csv.DictReader(f):
            if r.get("success") not in ("1", "1.0", "True", "true"):
                continue
            key = (r.get("scenario_seed"), r.get("env"))
            out[key] = {k: _num(r.get(k)) for k, _ in METRICS}
        return out


def stats(run: Path) -> dict:
    """Per-metric (mean, std, n) over successful episodes — the unpaired description."""
    d = by_seed(run)
    out = {}
    for key, _ in METRICS:
        v = [e[key] for e in d.values() if e.get(key) is not None]
        if v:
            a = np.asarray(v, float)
            out[key] = (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, a.size)
    return out


def paired(base: dict, arm: dict, key: str):
    """(mean % difference, its SE, n_pairs) over scenarios present and successful in BOTH arms."""
    d = [arm[k][key] - base[k][key] for k in base.keys() & arm.keys()
         if base[k].get(key) is not None and arm[k].get(key) is not None]
    b = [base[k][key] for k in base.keys() & arm.keys()
         if base[k].get(key) is not None and arm[k].get(key) is not None]
    if len(d) < 2:
        return None
    d, b = np.asarray(d, float), np.asarray(b, float)
    rel = 100.0 * d / np.mean(b)
    return float(rel.mean()), float(rel.std(ddof=1) / math.sqrt(rel.size)), rel.size


def main() -> None:
    tags = sys.argv[1:] or DEFAULT_TAGS
    rows = []
    for t in tags:
        run = resolve(t)
        if run is None:
            print(f"  {t:16s} (no completed run yet)")
            continue
        rows.append((t, json.loads((run / "summary.json").read_text()), run, stats(run),
                     by_seed(run)))
    if not rows:
        return

    base_pairs = rows[0][4]

    print(f"\n{'arm':16s} {'theta':>6s} {'open':>6s} {'n':>4s} {'succ':>6s}  "
          + "  ".join(f"{lab:>24s}" for _, lab in METRICS))
    print("-" * (42 + 26 * len(METRICS)))
    for tag, s, _, st, pairs in rows:
        cells = []
        for key, _ in METRICS:
            if key not in st:
                cells.append(f"{'-':>24s}")
                continue
            m, sd, _n = st[key]
            pr = paired(base_pairs, pairs, key)
            if pr is None:
                cells.append(f"{m:8.0f} ({sd:5.0f}){'':>10s}")
                continue
            rel, se, npair = pr
            # PAIRED: the scenarios are matched by seed, so the between-scenario spread cancels.
            sig = "" if abs(rel) > 2 * se else "~"
            cells.append(f"{m:8.0f} ({sd:5.0f}) {rel:+6.1f}+-{se:4.1f}{sig:1s}")
        print(f"{tag:16s} {s.get('shelf_deg', 0):6.0f} {s.get('shelf_open', 0) * 1e3:6.1f} "
              f"{s['n_episodes']:4d} {s['success_rate']:6.3f}  " + "  ".join(cells))

    print("\n  cells: mean (unpaired sd)  PAIRED %diff vs the first arm +- its SE")
    print("  '~' = the paired difference is inside 2 SE, i.e. not a result.")
    print("  peak-stress phase (a shelf that moves damage from `lift` to `hold` relocated it):")
    for tag, s, _, _st, _p in rows:
        d = s.get("peak_stress_phase_dist") or {}
        print(f"    {tag:16s} " + "  ".join(f"{k} {v:.2f}" for k, v in sorted(d.items())))
    print("\n  runs:")
    for tag, _, run, _st, _p in rows:
        print(f"    {tag:16s} {run.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
