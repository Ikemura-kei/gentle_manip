"""Compare canonical-eval runs on GENTLENESS + success — success-gated stress with P90/P95 and a
paired per-scenario test.

The canonical harness faces every policy with the SAME fixed scenarios (seed 0, seed_for_batch),
so a run's episodes.csv rows are matched across policies by (batch, env). This script:
  - recomputes per-policy aggregates from each episodes.csv (success-gated stress: mean / std /
    P90 / P95 over SUCCESSFUL episodes only — a failed episode never touched the object, so its
    near-zero stress must not count; see metrics.aggregate), plus all-episode stress for context;
  - does a PAIRED comparison vs a baseline: on scenarios BOTH policies succeed, Wilcoxon
    signed-rank on per-scenario peak-stress deltas (controls scene difficulty), + a McNemar-style
    success crosstab (who succeeds where).

Usage (any env with scipy; e.g. envs/sim):
    uv run --project envs/sim python -m gentle_manip.scripts.compare_evals \\
        BC=<bc_run>/eval/<dt> trpip50=<...>/eval/<dt> trpip150=<...>/eval/<dt> \\
        --baseline BC --plot logs/compare_stress.png --out logs/compare.json
Each arg is LABEL=<eval_dir> (dir with episodes.csv) or LABEL=<path/to/episodes.csv>.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> list[dict]:
    csvp = path / "episodes.csv" if path.is_dir() else path
    with open(csvp, newline="") as f:
        return list(csv.DictReader(f))


def _f(row, key):
    v = row.get(key, "")
    if v in ("", "None", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    succ = np.array([int(float(r["success"])) for r in rows], bool)
    peak_all = np.array([_f(r, "stress_peak") for r in rows if _f(r, "stress_peak") is not None])
    peak_s = np.array([_f(r, "stress_peak") for r in rows
                       if int(float(r["success"])) and _f(r, "stress_peak") is not None])
    mean_s = np.array([_f(r, "stress_mean") for r in rows
                       if int(float(r["success"])) and _f(r, "stress_mean") is not None])
    g = lambda a, q: float(np.percentile(a, q)) if a.size else float("nan")
    return {
        "n": n, "success_rate": float(succ.mean()) if n else 0.0,
        "n_success": int(succ.sum()),
        "stress_peak_mean": float(peak_s.mean()) if peak_s.size else float("nan"),
        "stress_peak_std": float(peak_s.std()) if peak_s.size else float("nan"),
        "stress_peak_p90": g(peak_s, 90), "stress_peak_p95": g(peak_s, 95),
        "stress_mean_mean": float(mean_s.mean()) if mean_s.size else float("nan"),
        "stress_peak_mean_all": float(peak_all.mean()) if peak_all.size else float("nan"),
    }


def _key(r):
    return (int(float(r["batch"])), int(float(r["env"])))


def _paired(base_rows, other_rows) -> dict:
    """Paired peak-stress delta (other - base) on scenarios BOTH succeed + success crosstab."""
    b = {_key(r): r for r in base_rows}
    o = {_key(r): r for r in other_rows}
    keys = sorted(set(b) & set(o))
    bs = np.array([int(float(b[k]["success"])) for k in keys], bool)
    os_ = np.array([int(float(o[k]["success"])) for k in keys], bool)
    both = bs & os_
    # per-scenario peak-stress delta where both succeeded and both have stress
    deltas = []
    for k in keys:
        if int(float(b[k]["success"])) and int(float(o[k]["success"])):
            pb, po = _f(b[k], "stress_peak"), _f(o[k], "stress_peak")
            if pb is not None and po is not None:
                deltas.append(po - pb)
    deltas = np.asarray(deltas, float)
    res = {
        "n_shared_scenarios": len(keys),
        "n_both_success": int(both.sum()),
        "only_base_success": int((bs & ~os_).sum()),
        "only_other_success": int((~bs & os_).sum()),
        "n_paired_stress": int(deltas.size),
        "median_peak_delta": float(np.median(deltas)) if deltas.size else float("nan"),
        "mean_peak_delta": float(deltas.mean()) if deltas.size else float("nan"),
    }
    if deltas.size >= 5 and np.any(deltas != 0):
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(deltas)
            res["wilcoxon_p"] = float(p)
        except Exception as e:
            res["wilcoxon_p"] = f"n/a ({e})"
    else:
        res["wilcoxon_p"] = "n/a (too few paired successes)"
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evals", nargs="+", help="LABEL=<eval_dir|episodes.csv> ...")
    ap.add_argument("--baseline", default=None, help="LABEL to pair others against (default: first)")
    ap.add_argument("--plot", type=Path, default=None, help="write a success-gated peak-stress box plot PNG")
    ap.add_argument("--out", type=Path, default=None, help="write the full comparison as JSON")
    args = ap.parse_args()

    runs = {}
    for a in args.evals:
        label, _, p = a.partition("=")
        runs[label] = _load(Path(p))
    base = args.baseline or next(iter(runs))

    aggs = {lab: _agg(rows) for lab, rows in runs.items()}
    print(f"\n{'policy':<12}{'n':>5}{'succ%':>8}{'peakμ(s)':>11}{'peakP90':>10}"
          f"{'peakP95':>10}{'meanμ(s)':>10}{'peakμ(all)':>12}  (s)=success-gated")
    for lab, m in aggs.items():
        print(f"{lab:<12}{m['n']:>5}{100*m['success_rate']:>7.1f}%{m['stress_peak_mean']:>11.0f}"
              f"{m['stress_peak_p90']:>10.0f}{m['stress_peak_p95']:>10.0f}"
              f"{m['stress_mean_mean']:>10.0f}{m['stress_peak_mean_all']:>12.0f}")

    print(f"\nPaired vs baseline '{base}' (peak-stress delta = other - base, on BOTH-success "
          f"scenarios; negative = other is gentler):")
    paired = {}
    for lab, rows in runs.items():
        if lab == base:
            continue
        pr = _paired(runs[base], rows)
        paired[lab] = pr
        print(f"  {lab:<12} both_succ={pr['n_both_success']:>3}  "
              f"median_Δpeak={pr['median_peak_delta']:>+8.0f}  "
              f"mean_Δpeak={pr['mean_peak_delta']:>+8.0f}  "
              f"wilcoxon_p={pr['wilcoxon_p']}  "
              f"(succ only:{base}={pr['only_base_success']} / {lab}={pr['only_other_success']})")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            data, labels = [], []
            for lab, rows in runs.items():
                ps = [_f(r, "stress_peak") for r in rows
                      if int(float(r["success"])) and _f(r, "stress_peak") is not None]
                if ps:
                    data.append(ps); labels.append(f"{lab}\n(n={len(ps)})")
            fig, ax = plt.subplots(figsize=(1.6 * len(data) + 2, 5))
            ax.boxplot(data, labels=labels, showmeans=True)
            ax.set_ylabel("peak von-Mises stress (success episodes)")
            ax.set_title("Gentleness: success-gated peak stress by policy")
            ax.axhline(40000, ls="--", c="r", lw=1, label="~40 kPa yield")
            ax.legend()
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(args.plot, dpi=120)
            print(f"\n[plot] {args.plot}")
        except Exception as e:
            print(f"[plot] skipped: {e}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"aggregates": aggs, "paired_vs_" + base: paired}, indent=2))
        print(f"[json] {args.out}")


if __name__ == "__main__":
    main()
