"""A THIRD family, alongside the P-family/Q-family split in probe_policy_channel_isolated_gap.py
-- this one asks: does the pos/quat/gripper ranking found there still hold when the point cloud
is REAL instead of sim?

Recap of the two existing families (both single-channel swaps, both anchored to S = all-sim):
  P-family:  baseline P  = all-SIM proprio + R cloud;    Pp/Pq/Pg ADD one real channel.
  Q-family:  baseline Q  = all-REAL proprio + SIM cloud; Qp/Qq/Qg REMOVE one real channel.

Q-family already answers "how much does each channel matter, holding the cloud at sim". This
script adds the missing condition: holding the cloud at REAL (Condition R's point cloud, i.e.
real arm + sim-swapped mushroom) and removing one real proprioception channel at a time from
the fully-real baseline R:

  R-family:  baseline R  = all-REAL proprio + R cloud;   Rp/Rq/Rg REMOVE one real channel.
      R   real pos, real quat, real grip   + R cloud   (combined reference, both real)
      Rp  SIM pos,  real quat, real grip   + R cloud   (R w/ sim pos)
      Rq  real pos, SIM quat,  real grip   + R cloud   (R w/ sim quat)
      Rg  real pos, real quat, SIM grip    + R cloud   (R w/ sim gripper)

Q-family and R-family are plotted side by side (same layout style as the P-vs-Q plot) so the
only thing that differs between the two columns is whether the cloud is sim or real -- if the
pos>quat>gripper ranking and the near-zero cross-channel leakage look the same in both columns,
that's evidence the ranking is a property of the proprioception channels themselves, not an
artifact of which point cloud happened to be used.

Same open-loop / teacher-forced probing as every other probe in this directory: at each frame
t the policy's history buffer is fed that condition's own ground-truth sequence up to t and we
record only the immediate next predicted action -- no closed-loop rollout, no error
accumulation.

Outputs, per episode: `epNN_rcloud_channel_isolated_gap.png` (delta vs time, 3 rows x 2 cols).
Aggregate: `rcloud_channel_isolated_gap_summary.png` (bar chart) and `.csv`.

Usage (envs/dppo_deploy -- needs the DPPO policy code + torch + CUDA):
    uv run --project envs/dppo_deploy python examples/sim2real_diagnose/probe_policy_rcloud_channel_isolated_gap.py \\
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


COMPARISONS = [
    ("Q_S",  "proprioception-only gap, SIM cloud (real pos+quat+grip, sim cloud)", "tab:purple"),
    ("Qp_S", "  w/ sim pos (SIM pos, real quat+grip, sim cloud)",                  "orchid"),
    ("Qq_S", "  w/ sim quat (real pos, SIM quat, real grip, sim cloud)",           "magenta"),
    ("Qg_S", "  w/ sim gripper (real pos+quat, SIM grip, sim cloud)",              "tab:pink"),
    ("R_S",  "combined reference, REAL cloud (real pos+quat+grip, R cloud)",       "tab:green"),
    ("Rp_S", "  w/ sim pos (SIM pos, real quat+grip, R cloud)",                    "yellowgreen"),
    ("Rq_S", "  w/ sim quat (real pos, SIM quat, real grip, R cloud)",             "teal"),
    ("Rg_S", "  w/ sim gripper (real pos+quat, SIM grip, R cloud)",                "olive"),
]

# (channel, sim-cloud-family swap key, real-cloud-family swap key, color) -- colors match
# probe_policy_channel_isolated_gap.py's channel palette for cross-plot consistency.
CHANNELS = [
    ("pos",     "Qp_S", "Rp_S", "crimson"),
    ("quat",    "Qq_S", "Rq_S", "dodgerblue"),
    ("gripper", "Qg_S", "Rg_S", "darkorange"),
]
METRICS = [(0, "pos_mm", "action pos diff (mm)"),
           (1, "quat_deg", "action rot diff (deg)"),
           (2, "grip_mm", "action gripper diff (mm)")]


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
                    help="default: <pkl's dir>/policy_rcloud_channel_isolated_gap/")
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

    out_dir = args.out_dir or (args.pkl.parent / "policy_rcloud_channel_isolated_gap")
    out_dir.mkdir(parents=True, exist_ok=True)

    action_cfg = ActionConfig.from_dict(yaml.safe_load(open(args.action_config)))
    pipeline = ActionPipeline(action_cfg)
    action_dim = action_cfg.action_dim

    policy = DPPOPolicyAdapter(
        str(args.ckpt), str(args.normalization), obs_dim=args.obs_dim, action_dim=action_dim,
        cond_steps=args.cond_steps, act_steps=args.act_steps,
        ft_denoising_steps=args.ft_denoising_steps, device=args.device)

    means = {f"{key}_{m}": [] for key, *_ in COMPARISONS for m in ("pos_mm", "quat_deg", "grip_mm")}
    csv_rows = []

    for ep_idx in picks:
        ep = episodes[ep_idx]
        obs = ep["observations"]
        ee_pos, ee_quat, gw = obs["ee_pos"], obs["ee_quat"], obs["gripper_width"]
        ee_pos_s, ee_quat_s, gw_s = obs["ee_pos_sim"], obs["ee_quat_sim"], obs["gripper_width_sim"]
        pc_r, pc_s = obs["point_cloud"], obs["point_cloud_sim"]
        T = ee_pos.shape[0]
        ts = np.arange(T)

        raw_S  = _replay_condition(policy, ee_pos_s, ee_quat_s, gw_s, pc_s)
        raw_Q  = _replay_condition(policy, ee_pos,   ee_quat,   gw,   pc_s)   # all-real proprio + sim cloud
        raw_Qp = _replay_condition(policy, ee_pos_s, ee_quat,   gw,   pc_s)   # Q w/ sim pos
        raw_Qq = _replay_condition(policy, ee_pos,   ee_quat_s, gw,   pc_s)   # Q w/ sim quat
        raw_Qg = _replay_condition(policy, ee_pos,   ee_quat,   gw_s, pc_s)   # Q w/ sim grip
        raw_R  = _replay_condition(policy, ee_pos,   ee_quat,   gw,   pc_r)   # all-real proprio + R cloud
        raw_Rp = _replay_condition(policy, ee_pos_s, ee_quat,   gw,   pc_r)   # R w/ sim pos
        raw_Rq = _replay_condition(policy, ee_pos,   ee_quat_s, gw,   pc_r)   # R w/ sim quat
        raw_Rg = _replay_condition(policy, ee_pos,   ee_quat,   gw_s, pc_r)   # R w/ sim grip

        raws = {"S": raw_S, "Q": raw_Q, "Qp": raw_Qp, "Qq": raw_Qq, "Qg": raw_Qg,
                "R": raw_R, "Rp": raw_Rp, "Rq": raw_Rq, "Rg": raw_Rg}
        phys = {k: _to_physical(pipeline, v) for k, v in raws.items()}

        series = {}
        for key, _desc, color in COMPARISONS:
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
        for key, *_ in COMPARISONS:
            row[f"{key}_pos_mm_mean"] = float(series[key][0].mean())
            row[f"{key}_quat_deg_mean"] = float(series[key][1].mean())
            row[f"{key}_grip_mm_mean"] = float(series[key][2].mean())
        csv_rows.append(row)

        # delta-from-baseline (Q_S - Qx_S for the sim-cloud column, R_S - Rx_S for the
        # real-cloud column) -- same simplification as probe_policy_channel_isolated_gap.py.
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="col")
        for row_idx, _mkey, mlabel in METRICS:
            ax_q, ax_r = axes[row_idx, 0], axes[row_idx, 1]
            for chan, q_key, r_key, color in CHANNELS:
                delta_q = series["Q_S"][row_idx] - series[q_key][row_idx]
                delta_r = series["R_S"][row_idx] - series[r_key][row_idx]
                ax_q.plot(ts, delta_q, label=f"{chan} channel", color=color, lw=1.8)
                ax_r.plot(ts, delta_r, label=f"{chan} channel", color=color, lw=1.8)
            ax_q.axhline(0, color="gray", lw=0.8, ls=":")
            ax_r.axhline(0, color="gray", lw=0.8, ls=":")
            ax_q.set_ylabel(mlabel); ax_q.grid(alpha=0.3)
            ax_r.grid(alpha=0.3)
            if row_idx == 0:
                ax_q.set_title("Q-family Δ (SIM cloud, removing real channel)")
                ax_r.set_title("R-family Δ (REAL cloud, removing real channel)")
                ax_q.legend(fontsize=8); ax_r.legend(fontsize=8)
        axes[2, 0].set_xlabel("frame t"); axes[2, 1].set_xlabel("frame t")
        fig.suptitle(f"hybrid ep {ep_idx} — does point-cloud realness change the channel "
                    f"ranking? (open-loop, no error accumulation)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fpath = out_dir / f"ep{ep_idx:02d}_rcloud_channel_isolated_gap.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)

        bits = "  ".join(f"{key}(pos={series[key][0].mean():.1f}mm,rot={series[key][1].mean():.1f}deg,"
                         f"grip={series[key][2].mean():.2f}mm)" for key, *_ in COMPARISONS)
        print(f"ep {ep_idx}: {bits}", flush=True)
        print(f"  saved {fpath}", flush=True)

    # ── aggregate: grouped bar chart, same 3-metric x 2-family layout ────────────────
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for row_idx, mkey, mlabel in METRICS:
        ax_q, ax_r = axes[row_idx, 0], axes[row_idx, 1]
        labels = [c for c, *_ in CHANNELS]
        colors = [color for *_, color in CHANNELS]
        q_vals, q_errs, r_vals, r_errs = [], [], [], []
        for chan, q_key, r_key, color in CHANNELS:
            dq = np.array(means[f"Q_S_{mkey}"]) - np.array(means[f"{q_key}_{mkey}"])
            dr = np.array(means[f"R_S_{mkey}"]) - np.array(means[f"{r_key}_{mkey}"])
            q_vals.append(dq.mean()); q_errs.append(dq.std())
            r_vals.append(dr.mean()); r_errs.append(dr.std())
        ax_q.bar(labels, q_vals, yerr=q_errs, color=colors, capsize=4)
        ax_r.bar(labels, r_vals, yerr=r_errs, color=colors, capsize=4)
        ax_q.axhline(0, color="gray", lw=0.8)
        ax_r.axhline(0, color="gray", lw=0.8)
        ax_q.set_ylabel(mlabel); ax_q.grid(alpha=0.3, axis="y")
        ax_r.grid(alpha=0.3, axis="y")
        if row_idx == 0:
            ax_q.set_title("Q-family Δ (SIM cloud, removing real channel)")
            ax_r.set_title("R-family Δ (REAL cloud, removing real channel)")
        # shared y-range across the row (both columns of the SAME metric) so bar heights are
        # directly visually comparable between sim-cloud and real-cloud
        lo = min(min(np.array(q_vals) - np.array(q_errs)), min(np.array(r_vals) - np.array(r_errs)), 0.0)
        hi = max(max(np.array(q_vals) + np.array(q_errs)), max(np.array(r_vals) + np.array(r_errs)), 0.0)
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax_q.set_ylim(lo - pad, hi + pad)
        ax_r.set_ylim(lo - pad, hi + pad)
    axes[2, 0].set_xlabel("channel swapped"); axes[2, 1].set_xlabel("channel swapped")
    fig.suptitle(f"mean isolated channel effect, sim cloud vs real cloud, over {len(picks)} "
                "episodes (error bars = std across episodes)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    spath = out_dir / "rcloud_channel_isolated_gap_summary.png"
    fig.savefig(spath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {spath}", flush=True)

    cpath = out_dir / "rcloud_channel_isolated_gap_summary.csv"
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"saved {cpath}", flush=True)

    print("\n=== overall mean action-prediction difference across all episodes ===")
    for key, desc, _color in COMPARISONS:
        print(f"  {key} ({desc}):  pos={np.mean(means[f'{key}_pos_mm']):.2f}mm"
              f"   rot={np.mean(means[f'{key}_quat_deg']):.2f}deg"
              f"   grip={np.mean(means[f'{key}_grip_mm']):.3f}mm")

    print("\n=== channel ranking: sim cloud (Q-family) vs real cloud (R-family) ===")
    for chan, q_key, r_key, _color in CHANNELS:
        bits_q, bits_r = [], []
        for _row_idx, mkey, mlabel in METRICS:
            unit = "mm" if mkey != "quat_deg" else "deg"
            dq = np.mean(means[f"Q_S_{mkey}"]) - np.mean(means[f"{q_key}_{mkey}"])
            dr = np.mean(means[f"R_S_{mkey}"]) - np.mean(means[f"{r_key}_{mkey}"])
            short = mkey.split("_")[0]
            bits_q.append(f"{short}_delta={dq:+.3f}{unit}")
            bits_r.append(f"{short}_delta={dr:+.3f}{unit}")
        print(f"  {chan:8s}  Q-family (sim cloud): " + " ".join(bits_q)
              + f"   |  R-family (real cloud): " + " ".join(bits_r))


if __name__ == "__main__":
    main()
