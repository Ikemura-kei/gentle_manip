"""Plot the RAW (non-delta) gap: each synthetic condition's predicted-action distance from S
(all-sim, treated as ground truth here), for all 12 conditions across the P/Q/R families --
no "delta from baseline" transformation, so bar height directly answers "how far off is this
setup's predicted action from the pure-sim baseline".

Conditions per family (see probe_policy_all_family_channel_isolated_gap.py / the README
glossary for what P/Q/R and "R cloud" mean):
  P-family: P (all-sim proprio + R cloud), Pp/Pq/Pg (P + one real channel added)
  Q-family: Q (all-real proprio + sim cloud), Qp/Qq/Qg (Q with one real channel swapped to sim)
  R-family: R (all-real proprio + R cloud), Rp/Rq/Rg (R with one real channel swapped to sim)

Pure post-processing of an already-computed
`all_family_channel_isolated_gap_summary.csv` (no policy re-run, no Genesis) -- reads the
per-episode mean columns that script already wrote (`<key>_S_<metric>_mean`).

Outputs (3 rows = pos/quat/gripper metric, 3 cols = P/Q/R family, shared y-limit per row):
  `raw_gap_vs_sim_summary.png`     -- grouped bar chart, mean +/- std over episodes
  `raw_gap_vs_sim_per_episode.png` -- one line per condition, x = episode index

Usage (envs/deploy -- matplotlib only):
    uv run --project envs/deploy python examples/sim2real_diagnose/plot_raw_gap_vs_sim.py \\
        dataset/real_deploy/ahaxs800_printed_mushrooms/sim2real_data_analysis/policy_all_family_channel_isolated_gap/all_family_channel_isolated_gap_summary.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


CHANNELS = [("baseline", None, "black"), ("pos", "p", "crimson"),
            ("quat", "q", "dodgerblue"), ("gripper", "g", "darkorange")]
FAMILIES = [("P", "P"), ("Q", "Q"), ("R", "R")]
METRICS = [(0, "pos_mm", "action pos diff from S (mm)"),
           (1, "quat_deg", "action rot diff from S (deg)"),
           (2, "grip_mm", "action gripper diff from S (mm)")]


def _key(fam_letter, chan_suffix):
    return f"{fam_letter}{chan_suffix or ''}_S"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <csv's dir>/../policy_raw_gap_vs_sim/")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(args.csv_path)))
    means = {k: [float(r[k]) for r in rows] for k in rows[0].keys() if k not in ("episode", "n_frames")}
    picks = [int(r["episode"]) for r in rows]
    print(f"Loaded {len(rows)} episodes from {args.csv_path}", flush=True)

    out_dir = args.out_dir or (args.csv_path.parent.parent / "policy_raw_gap_vs_sim")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── aggregate: grouped bar chart, mean +/- std over episodes ──────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for row_idx, mkey, mlabel in METRICS:
        col_data = []
        for fam_label, fam_letter in FAMILIES:
            labels, vals, errs, colors = [], [], [], []
            for chan, suffix, color in CHANNELS:
                arr = np.array(means[f"{_key(fam_letter, suffix)}_{mkey}_mean"])
                labels.append(chan); vals.append(arr.mean()); errs.append(arr.std()); colors.append(color)
            col_data.append((labels, vals, errs, colors, fam_label))
        row_lo = min(min(np.array(v) - np.array(e)) for _l, v, e, _c, _t in col_data)
        row_hi = max(max(np.array(v) + np.array(e)) for _l, v, e, _c, _t in col_data)
        row_lo = min(row_lo, 0.0); row_hi = max(row_hi, 0.0)
        pad = 0.05 * (row_hi - row_lo) if row_hi > row_lo else 1.0
        for col_idx, (labels, vals, errs, colors, fam_label) in enumerate(col_data):
            ax = axes[row_idx, col_idx]
            ax.bar(labels, vals, yerr=errs, color=colors, capsize=4)
            ax.axhline(0, color="gray", lw=0.8)
            ax.grid(alpha=0.3, axis="y")
            ax.set_ylim(row_lo - pad, row_hi + pad)
            if row_idx == 0:
                ax.set_title(f"{fam_label}-family (vs S = all-sim \"ground truth\")", fontsize=10)
        axes[row_idx, 0].set_ylabel(mlabel)
    for col_idx in range(3):
        axes[2, col_idx].set_xlabel("condition")
    fig.suptitle("raw predicted-action distance from all-sim baseline S, by condition "
                "(mean +/- std across episodes, NOT a delta)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    spath = out_dir / "raw_gap_vs_sim_summary.png"
    fig.savefig(spath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {spath}", flush=True)

    # ── per-episode: one line per condition, x = episode ──────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for row_idx, mkey, mlabel in METRICS:
        row_series = []
        for fam_label, fam_letter in FAMILIES:
            for chan, suffix, color in CHANNELS:
                row_series.append(np.array(means[f"{_key(fam_letter, suffix)}_{mkey}_mean"]))
        row_lo = min(s.min() for s in row_series); row_hi = max(s.max() for s in row_series)
        row_lo = min(row_lo, 0.0); row_hi = max(row_hi, 0.0)
        pad = 0.05 * (row_hi - row_lo) if row_hi > row_lo else 1.0
        for col_idx, (fam_label, fam_letter) in enumerate(FAMILIES):
            ax = axes[row_idx, col_idx]
            for chan, suffix, color in CHANNELS:
                arr = np.array(means[f"{_key(fam_letter, suffix)}_{mkey}_mean"])
                ax.plot(picks, arr, "o-", label=chan, color=color, lw=1.6, ms=3)
            ax.grid(alpha=0.3)
            ax.set_ylim(row_lo - pad, row_hi + pad)
            if row_idx == 0:
                ax.set_title(f"{fam_label}-family (vs S = all-sim \"ground truth\")", fontsize=10)
                ax.legend(fontsize=7)
        axes[row_idx, 0].set_ylabel(mlabel)
    for col_idx in range(3):
        axes[2, col_idx].set_xlabel("episode")
    fig.suptitle("raw predicted-action distance from all-sim baseline S, per episode "
                "(NOT a delta)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    epath = out_dir / "raw_gap_vs_sim_per_episode.png"
    fig.savefig(epath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {epath}", flush=True)

    print("\n=== overall mean raw distance from S, by condition ===")
    for fam_label, fam_letter in FAMILIES:
        for chan, suffix, _color in CHANNELS:
            key = _key(fam_letter, suffix)
            p = np.mean(means[f"{key}_pos_mm_mean"])
            q = np.mean(means[f"{key}_quat_deg_mean"])
            g = np.mean(means[f"{key}_grip_mm_mean"])
            print(f"  {key:5s} ({fam_label}-family, {chan:8s}):  pos={p:.2f}mm   rot={q:.2f}deg   grip={g:.3f}mm")


if __name__ == "__main__":
    main()
