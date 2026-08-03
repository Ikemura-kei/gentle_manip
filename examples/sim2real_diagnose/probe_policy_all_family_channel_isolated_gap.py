"""All THREE channel-isolation families side by side (P, Q, R), each showing the isolated
per-channel (pos/quat/gripper) effect on the predicted action, anchored back to S (all-sim).
Combines probe_policy_channel_isolated_gap.py (P-family, Q-family) and
probe_policy_rcloud_channel_isolated_gap.py (Q-family, R-family) into one 3-column view so
all three can be read off a single figure, with shared y-limits within each metric row so bar
heights / line amplitudes are directly comparable across families at a glance.

  P-family  baseline P = all-SIM proprio + R cloud.     Pp/Pq/Pg ADD one real channel.
  Q-family  baseline Q = all-REAL proprio + SIM cloud.  Qp/Qq/Qg REMOVE one real channel.
  R-family  baseline R = all-REAL proprio + R cloud.    Rp/Rq/Rg REMOVE one real channel.

("R cloud" = Condition R's EDITED cloud, real arm + sim-swapped mushroom, NOT the original
unedited real camera capture -- see the README glossary in this directory.)

P-family starts from nothing-real and asks "how much does adding channel X move the action";
Q-family and R-family start from everything-real (differing only in whether the cloud is held
at sim or real) and ask "how much does removing channel X move the action back". All three
columns share the same channel-color palette (pos=crimson, quat=dodgerblue,
gripper=darkorange) and, within each metric row, the same y-axis range.

Same open-loop / teacher-forced probing as every other probe in this directory: at each frame
t the policy's history buffer is fed that condition's own ground-truth sequence up to t and we
record only the immediate next predicted action -- no closed-loop rollout, no error
accumulation.

Outputs, per episode: `epNN_all_family_channel_isolated_gap.png` (delta vs time, 3 rows x 3
cols, shared y-limit per row). Aggregate: `all_family_channel_isolated_gap_summary.png` (bar
chart, same layout) and `.csv`.

Usage (envs/dppo_deploy -- needs the DPPO policy code + torch + CUDA):
    uv run --project envs/dppo_deploy python examples/sim2real_diagnose/probe_policy_all_family_channel_isolated_gap.py \\
        dataset/real_deploy/ahaxs800_printed_mushrooms/sim2real_data_analysis/hybrid_arm_real_mushroom_sim.pkl \\
        --ckpt logs/dppo/dppo-pretrain/single_lift_mushroom_rigid/cho/ahaxs/checkpoint/state_800.pt \\
        --normalization dataset/dppo/single_lift_mushroom_rigid/cho/normalization.npz
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_REPO = _THIS_DIR.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from gentle_manip.scripts.deploy_real_dppo import DPPOPolicyAdapter   # noqa: E402
from gentle_manip.actions.action_config import ActionConfig            # noqa: E402
from gentle_manip.actions.pipeline import ActionPipeline               # noqa: E402
import replay_deploy_in_sim as rds                                     # noqa: E402  (reuse quat helpers)
from probe_policy_action_diff import _to_physical, _replay_condition   # noqa: E402


# (channel, color) -- consistent across all three families/columns.
CHANNELS = [("pos", "crimson"), ("quat", "dodgerblue"), ("gripper", "darkorange")]
METRICS = [(0, "pos_mm", "action pos diff (mm)"),
           (1, "quat_deg", "action rot diff (deg)"),
           (2, "grip_mm", "action gripper diff (mm)")]
# (family, baseline key, {channel: swap key}, direction, title)
FAMILIES = [
    ("P", "P_S", {"pos": "Pp_S", "quat": "Pq_S", "gripper": "Pg_S"}, "add",
     "P-family Δ (ADDING real channel, R cloud)"),
    ("Q", "Q_S", {"pos": "Qp_S", "quat": "Qq_S", "gripper": "Qg_S"}, "remove",
     "Q-family Δ (REMOVING real channel, SIM cloud)"),
    ("R", "R_S", {"pos": "Rp_S", "quat": "Rq_S", "gripper": "Rg_S"}, "remove",
     "R-family Δ (REMOVING real channel, R cloud)"),
]
COMPARISONS = [(key, color) for fam, base, swaps, _dir, _title in FAMILIES
               for key, color in [(base, "black")] + [(swaps[c], col) for c, col in CHANNELS]]


def _family_delta(series, base_key, swap_key, direction, row_idx):
    if direction == "add":
        return series[swap_key][row_idx] - series[base_key][row_idx]
    return series[base_key][row_idx] - series[swap_key][row_idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl", type=Path)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--normalization", type=Path, required=True)
    ap.add_argument("--action-config", type=Path,
                    default=_REPO / "gentle_manip/configs/action/abs_pose_abs_gripper.yaml")
    ap.add_argument("--obs-dim", type=int, default=8)
    ap.add_argument("--cond-steps", type=int, default=2)
    ap.add_argument("--act-steps", type=int, default=4)
    ap.add_argument("--ft-denoising-steps", type=int, default=0, help="0 for a BC checkpoint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <pkl's dir>/policy_all_family_channel_isolated_gap/")
    ap.add_argument("--episodes", default="", help="comma-sep explicit episode indices (default: all)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pickle.load(open(args.pkl, "rb"))
    episodes = data["episodes"]
    print(f"Loaded {len(episodes)} episodes from {args.pkl}", flush=True)

    picks = ([int(x) for x in args.episodes.split(",")] if args.episodes.strip()
             else list(range(len(episodes))))

    out_dir = args.out_dir or (args.pkl.parent / "policy_all_family_channel_isolated_gap")
    out_dir.mkdir(parents=True, exist_ok=True)

    action_cfg = ActionConfig.from_dict(yaml.safe_load(open(args.action_config)))
    pipeline = ActionPipeline(action_cfg)
    action_dim = action_cfg.action_dim

    policy = DPPOPolicyAdapter(
        str(args.ckpt), str(args.normalization), obs_dim=args.obs_dim, action_dim=action_dim,
        cond_steps=args.cond_steps, act_steps=args.act_steps,
        ft_denoising_steps=args.ft_denoising_steps, device=args.device)

    means = {f"{key}_{m}": [] for key, _c in COMPARISONS for m in ("pos_mm", "quat_deg", "grip_mm")}
    csv_rows = []

    for ep_idx in picks:
        ep = episodes[ep_idx]
        obs = ep["observations"]
        ee_pos, ee_quat, gw = obs["ee_pos"], obs["ee_quat"], obs["gripper_width"]
        ee_pos_s, ee_quat_s, gw_s = obs["ee_pos_sim"], obs["ee_quat_sim"], obs["gripper_width_sim"]
        pc_r, pc_s = obs["point_cloud"], obs["point_cloud_sim"]
        T = ee_pos.shape[0]
        ts = np.arange(T)

        raws = {
            "S":  _replay_condition(policy, ee_pos_s, ee_quat_s, gw_s, pc_s),
            "P":  _replay_condition(policy, ee_pos_s, ee_quat_s, gw_s, pc_r),
            "Pp": _replay_condition(policy, ee_pos,   ee_quat_s, gw_s, pc_r),
            "Pq": _replay_condition(policy, ee_pos_s, ee_quat,   gw_s, pc_r),
            "Pg": _replay_condition(policy, ee_pos_s, ee_quat_s, gw,   pc_r),
            "Q":  _replay_condition(policy, ee_pos,   ee_quat,   gw,   pc_s),
            "Qp": _replay_condition(policy, ee_pos_s, ee_quat,   gw,   pc_s),
            "Qq": _replay_condition(policy, ee_pos,   ee_quat_s, gw,   pc_s),
            "Qg": _replay_condition(policy, ee_pos,   ee_quat,   gw_s, pc_s),
            "R":  _replay_condition(policy, ee_pos,   ee_quat,   gw,   pc_r),
            "Rp": _replay_condition(policy, ee_pos_s, ee_quat,   gw,   pc_r),
            "Rq": _replay_condition(policy, ee_pos,   ee_quat_s, gw,   pc_r),
            "Rg": _replay_condition(policy, ee_pos,   ee_quat,   gw_s, pc_r),
        }
        phys = {k: _to_physical(pipeline, v) for k, v in raws.items()}

        series = {}
        for key, _color in COMPARISONS:
            a, b = key.split("_")
            pos_a, quat_a, grip_a = phys[a]
            pos_b, quat_b, grip_b = phys[b]
            quat_b_aligned = rds._align_quat_sign(quat_a, quat_b)
            pos_d = np.linalg.norm(pos_a - pos_b, axis=1) * 1000
            quat_d = np.rad2deg(rds._quat_angular_diff(quat_a, quat_b_aligned))
            grip_d = np.abs(grip_a - grip_b)[:, 0] * 1000
            series[key] = (pos_d, quat_d, grip_d)
            means[f"{key}_pos_mm"].append(float(pos_d.mean()))
            means[f"{key}_quat_deg"].append(float(quat_d.mean()))
            means[f"{key}_grip_mm"].append(float(grip_d.mean()))

        row = {"episode": ep_idx, "n_frames": T}
        for key, _c in COMPARISONS:
            row[f"{key}_pos_mm_mean"] = float(series[key][0].mean())
            row[f"{key}_quat_deg_mean"] = float(series[key][1].mean())
            row[f"{key}_grip_mm_mean"] = float(series[key][2].mean())
        csv_rows.append(row)

        # ── per-episode: 3 rows (metric) x 3 cols (family), shared y-limit per row ──
        fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
        for row_idx, _mkey, mlabel in METRICS:
            row_deltas = []
            for col_idx, (fam, base, swaps, direction, title) in enumerate(FAMILIES):
                ax = axes[row_idx, col_idx]
                for chan, color in CHANNELS:
                    delta = _family_delta(series, base, swaps[chan], direction, row_idx)
                    row_deltas.append(delta)
                    ax.plot(ts, delta, label=f"{chan} channel", color=color, lw=1.8)
                ax.axhline(0, color="gray", lw=0.8, ls=":")
                ax.grid(alpha=0.3)
                if row_idx == 0:
                    ax.set_title(title, fontsize=10)
                    ax.legend(fontsize=7)
            lo = min(d.min() for d in row_deltas); hi = max(d.max() for d in row_deltas)
            lo = min(lo, 0.0); hi = max(hi, 0.0)
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            for col_idx in range(3):
                axes[row_idx, col_idx].set_ylim(lo - pad, hi + pad)
            axes[row_idx, 0].set_ylabel(mlabel)
        axes[2, 0].set_xlabel("frame t"); axes[2, 1].set_xlabel("frame t"); axes[2, 2].set_xlabel("frame t")
        fig.suptitle(f"hybrid ep {ep_idx} — P/Q/R-family isolated channel effect, delta from "
                    f"each family's own baseline (open-loop, no error accumulation)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fpath = out_dir / f"ep{ep_idx:02d}_all_family_channel_isolated_gap.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)

        bits = "  ".join(f"{key}(pos={series[key][0].mean():.1f}mm,rot={series[key][1].mean():.1f}deg,"
                         f"grip={series[key][2].mean():.2f}mm)" for key, _c in COMPARISONS)
        print(f"ep {ep_idx}: {bits}", flush=True)
        print(f"  saved {fpath}", flush=True)

    # ── aggregate: 3 rows (metric) x 3 cols (family) bar charts, shared y-limit per row ──
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for row_idx, mkey, mlabel in METRICS:
        row_vals_all = []
        col_data = []
        for col_idx, (fam, base, swaps, direction, title) in enumerate(FAMILIES):
            labels = [c for c, _col in CHANNELS]
            colors = [col for _c, col in CHANNELS]
            vals, errs = [], []
            for chan, _color in CHANNELS:
                base_arr = np.array(means[f"{base}_{mkey}"])
                swap_arr = np.array(means[f"{swaps[chan]}_{mkey}"])
                d = (swap_arr - base_arr) if direction == "add" else (base_arr - swap_arr)
                vals.append(d.mean()); errs.append(d.std())
            col_data.append((labels, vals, errs, colors, title))
            row_vals_all.append((np.array(vals), np.array(errs)))
        lo = min((v - e).min() for v, e in row_vals_all); hi = max((v + e).max() for v, e in row_vals_all)
        lo = min(lo, 0.0); hi = max(hi, 0.0)
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        for col_idx, (labels, vals, errs, colors, title) in enumerate(col_data):
            ax = axes[row_idx, col_idx]
            ax.bar(labels, vals, yerr=errs, color=colors, capsize=4)
            ax.axhline(0, color="gray", lw=0.8)
            ax.grid(alpha=0.3, axis="y")
            ax.set_ylim(lo - pad, hi + pad)
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
        axes[row_idx, 0].set_ylabel(mlabel)
    for col_idx in range(3):
        axes[2, col_idx].set_xlabel("channel swapped")
    fig.suptitle(f"mean isolated channel effect, P vs Q vs R family, over {len(picks)} "
                "episodes (error bars = std across episodes)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    spath = out_dir / "all_family_channel_isolated_gap_summary.png"
    fig.savefig(spath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {spath}", flush=True)

    cpath = out_dir / "all_family_channel_isolated_gap_summary.csv"
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"saved {cpath}", flush=True)

    # ── same 3-column figure, PLUS a 4th reference column: R_S itself (real proprio + R
    # cloud vs S), NOT decomposed by channel -- "everything is real except the small
    # mushroom-swap edit in the cloud". Lets you eyeball how much of the total combined gap
    # each isolated channel delta accounts for. Saved as a SEPARATE file; the 3-column
    # original above is untouched. ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(17, 10))
    for row_idx, mkey, mlabel in METRICS:
        col_data = []
        for fam, base, swaps, direction, title in FAMILIES:
            labels = [c for c, _col in CHANNELS]
            colors = [col for _c, col in CHANNELS]
            vals, errs = [], []
            for chan, _color in CHANNELS:
                base_arr = np.array(means[f"{base}_{mkey}"])
                swap_arr = np.array(means[f"{swaps[chan]}_{mkey}"])
                d = (swap_arr - base_arr) if direction == "add" else (base_arr - swap_arr)
                vals.append(d.mean()); errs.append(d.std())
            col_data.append((labels, vals, errs, colors, title))
        r_s_arr = np.array(means[f"R_S_{mkey}"])
        ref_val, ref_err = r_s_arr.mean(), r_s_arr.std()
        row_lo = min(min(np.array(v) - np.array(e)) for _l, v, e, _c, _t in col_data)
        row_hi = max(max(np.array(v) + np.array(e)) for _l, v, e, _c, _t in col_data)
        row_lo = min(row_lo, ref_val - ref_err, 0.0)
        row_hi = max(row_hi, ref_val + ref_err, 0.0)
        pad = 0.05 * (row_hi - row_lo) if row_hi > row_lo else 1.0
        for col_idx, (labels, vals, errs, colors, title) in enumerate(col_data):
            ax = axes[row_idx, col_idx]
            ax.bar(labels, vals, yerr=errs, color=colors, capsize=4)
            ax.axhline(0, color="gray", lw=0.8)
            ax.grid(alpha=0.3, axis="y")
            ax.set_ylim(row_lo - pad, row_hi + pad)
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
        ax4 = axes[row_idx, 3]
        ax4.bar(["R vs S"], [ref_val], yerr=[ref_err], color="tab:green", capsize=4)
        ax4.axhline(0, color="gray", lw=0.8)
        ax4.grid(alpha=0.3, axis="y")
        ax4.set_ylim(row_lo - pad, row_hi + pad)
        if row_idx == 0:
            ax4.set_title("R baseline vs S (total, not decomposed —\nreal proprio + R "
                          "cloud, only the mushroom edit differs)", fontsize=9)
        axes[row_idx, 0].set_ylabel(mlabel)
    for col_idx in range(4):
        axes[2, col_idx].set_xlabel("channel swapped" if col_idx < 3 else "")
    fig.suptitle(f"mean isolated channel effect, P vs Q vs R family, plus R-vs-S total "
                f"reference, over {len(picks)} episodes (error bars = std across episodes)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    spath4 = out_dir / "all_family_channel_isolated_gap_summary_4col.png"
    fig.savefig(spath4, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {spath4}", flush=True)

    print("\n=== overall mean action-prediction difference across all episodes ===")
    for key, _color in COMPARISONS:
        print(f"  {key}:  pos={np.mean(means[f'{key}_pos_mm']):.2f}mm"
              f"   rot={np.mean(means[f'{key}_quat_deg']):.2f}deg"
              f"   grip={np.mean(means[f'{key}_grip_mm']):.3f}mm")

    print("\n=== channel ranking, all three families ===")
    for chan, _color in CHANNELS:
        bits = []
        for fam, base, swaps, direction, _title in FAMILIES:
            row_bits = []
            for _row_idx, mkey, _mlabel in METRICS:
                unit = "mm" if mkey != "quat_deg" else "deg"
                base_v = np.mean(means[f"{base}_{mkey}"])
                swap_v = np.mean(means[f"{swaps[chan]}_{mkey}"])
                d = (swap_v - base_v) if direction == "add" else (base_v - swap_v)
                short = mkey.split("_")[0]
                row_bits.append(f"{short}_delta={d:+.3f}{unit}")
            bits.append(f"{fam}-family: " + " ".join(row_bits))
        print(f"  {chan:8s}  " + "   |  ".join(bits))


if __name__ == "__main__":
    main()
