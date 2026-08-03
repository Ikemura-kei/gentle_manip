"""Full per-PROPRIOCEPTION-CHANNEL decomposition -- extends probe_policy_gripper_isolated_gap.py's
gripper-vs-rest split to give ee_pos and ee_quat the same one-at-a-time treatment, so all three
proprioception channels (pos, quat, gripper) can be ranked head-to-head by how much they alone
move the policy's predicted action.

Two families of single-channel swaps, both anchored back to S (all-sim) so every comparison
shares the same baseline and is directly comparable on one plot:

  P-family (start from P = all-sim proprio + Condition R's point cloud; ADD one real channel)
      P   sim pos, sim quat, sim grip                  + R cloud   (point-cloud-only gap)
      Pp  REAL pos, sim quat, sim grip                 + R cloud   (P + real pos)
      Pq  sim pos, REAL quat, sim grip                 + R cloud   (P + real quat)
      Pg  sim pos, sim quat, REAL grip                 + R cloud   (P + real gripper)

  Q-family (start from Q = all-real proprio + Condition S's point cloud; REMOVE one real channel)
      Q   real pos, real quat, real grip                + sim cloud  (proprioception-only gap)
      Qp  SIM pos, real quat, real grip                  + sim cloud  (Q w/ sim pos)
      Qq  real pos, SIM quat, real grip                  + sim cloud  (Q w/ sim quat)
      Qg  real pos, real quat, SIM grip                  + sim cloud  (Q w/ sim gripper)

Reading the result: in the P-family, (Pp_S - P_S), (Pq_S - P_S), (Pg_S - P_S) each isolate how
much ADDING that one real channel moves the action away from all-sim. In the Q-family,
(Q_S - Qp_S), (Q_S - Qq_S), (Q_S - Qg_S) each isolate how much REMOVING that one real channel
(putting it back to sim) moves the action back toward all-sim. Comparing these three deltas
within each family ranks pos vs quat vs gripper by their individual contribution to the
proprioception gap -- this is the "grouped" (not per-scalar-element) version of that ranking:
pos and quat are each treated as one channel, not split into x/y/z or w/x/y/z components.

R_S (both real, reference) is included for scale. All comparisons are vs S (sim-only actions).

Same open-loop / teacher-forced probing as every other probe in this directory: at each frame
t the policy's history buffer is fed that condition's own ground-truth sequence up to t and we
record only the immediate next predicted action -- no closed-loop rollout, no error
accumulation.

Outputs, per episode: `epNN_channel_isolated_gap.png` (diff vs time, 3 rows x 9 comparisons).
Aggregate: `channel_isolated_gap_summary.png` and `channel_isolated_gap_summary.csv`.

Usage (envs/dppo_deploy -- needs the DPPO policy code + torch + CUDA):
    uv run --project envs/dppo_deploy python examples/sim2real_diagnose/probe_policy_channel_isolated_gap.py \\
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
    ("P_S",  "point-cloud-only gap (all-sim proprio, R cloud)",        "tab:orange"),
    ("Pp_S", "  + real pos (REAL pos, sim quat+grip, R cloud)",        "tab:red"),
    ("Pq_S", "  + real quat (sim pos, REAL quat, sim grip, R cloud)",  "firebrick"),
    ("Pg_S", "  + real gripper (sim pos+quat, REAL grip, R cloud)",    "darkred"),
    ("Q_S",  "proprioception-only gap (all-real proprio, sim cloud)",  "tab:purple"),
    ("Qp_S", "  w/ sim pos (SIM pos, real quat+grip, sim cloud)",      "orchid"),
    ("Qq_S", "  w/ sim quat (real pos, SIM quat, real grip, sim cloud)", "magenta"),
    ("Qg_S", "  w/ sim gripper (real pos+quat, SIM grip, sim cloud)",  "tab:pink"),
    ("R_S",  "combined arm-only gap (reference, both real)",           "tab:green"),
]
# (channel, P-family swap key, Q-family swap key, color) -- maximally distinct hues so the
# three channels never get confused with each other across any subplot.
CHANNELS = [
    ("pos",     "Pp_S", "Qp_S", "crimson"),
    ("quat",    "Pq_S", "Qq_S", "dodgerblue"),
    ("gripper", "Pg_S", "Qg_S", "darkorange"),
]
# (series-tuple index, mean-dict suffix, axis label)
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
                    help="default: <pkl's dir>/policy_channel_isolated_gap/")
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

    out_dir = args.out_dir or (args.pkl.parent / "policy_channel_isolated_gap")
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
        raw_R  = _replay_condition(policy, ee_pos,   ee_quat,   gw,   pc_r)
        raw_P  = _replay_condition(policy, ee_pos_s, ee_quat_s, gw_s, pc_r)   # all-sim proprio + R cloud
        raw_Q  = _replay_condition(policy, ee_pos,   ee_quat,   gw,   pc_s)   # all-real proprio + sim cloud
        raw_Pp = _replay_condition(policy, ee_pos,   ee_quat_s, gw_s, pc_r)   # P + real pos
        raw_Pq = _replay_condition(policy, ee_pos_s, ee_quat,   gw_s, pc_r)   # P + real quat
        raw_Pg = _replay_condition(policy, ee_pos_s, ee_quat_s, gw,   pc_r)   # P + real grip
        raw_Qp = _replay_condition(policy, ee_pos_s, ee_quat,   gw,   pc_s)   # Q w/ sim pos
        raw_Qq = _replay_condition(policy, ee_pos,   ee_quat_s, gw,   pc_s)   # Q w/ sim quat
        raw_Qg = _replay_condition(policy, ee_pos,   ee_quat,   gw_s, pc_s)   # Q w/ sim grip

        raws = {"S": raw_S, "R": raw_R, "P": raw_P, "Q": raw_Q,
                "Pp": raw_Pp, "Pq": raw_Pq, "Pg": raw_Pg,
                "Qp": raw_Qp, "Qq": raw_Qq, "Qg": raw_Qg}
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

        # Plot DELTA-FROM-BASELINE, not raw absolute diff-from-S: e.g. for the pos channel in
        # the P-family, Pp_S(t) - P_S(t) is "how much adding real pos moves the action away
        # from the all-sim baseline, on top of what the point cloud alone already did" -- this
        # collapses each family's baseline+3-variants (4 raw lines) down to 3 delta lines, and
        # a delta near 0 for a metric a channel doesn't own (e.g. gripper-channel's pos delta)
        # is a direct, uncluttered visualization of near-zero cross-channel leakage.
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="col")
        for row_idx, _mkey, mlabel in METRICS:
            ax_p, ax_q = axes[row_idx, 0], axes[row_idx, 1]
            for chan, p_key, q_key, color in CHANNELS:
                delta_p = series[p_key][row_idx] - series["P_S"][row_idx]
                delta_q = series["Q_S"][row_idx] - series[q_key][row_idx]
                ax_p.plot(ts, delta_p, label=f"{chan} channel", color=color, lw=1.8)
                ax_q.plot(ts, delta_q, label=f"{chan} channel", color=color, lw=1.8)
            ax_p.axhline(0, color="gray", lw=0.8, ls=":")
            ax_q.axhline(0, color="gray", lw=0.8, ls=":")
            ax_p.set_ylabel(mlabel); ax_p.grid(alpha=0.3)
            ax_q.grid(alpha=0.3)
            if row_idx == 0:
                ax_p.set_title("P-family Δ (adding real channel to all-sim + R cloud)")
                ax_q.set_title("Q-family Δ (removing real channel from all-real + sim cloud)")
                ax_p.legend(fontsize=8); ax_q.legend(fontsize=8)
        axes[2, 0].set_xlabel("frame t"); axes[2, 1].set_xlabel("frame t")
        fig.suptitle(f"hybrid ep {ep_idx} — per-channel isolated action gap, delta from family "
                    f"baseline (open-loop, no error accumulation)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fpath = out_dir / f"ep{ep_idx:02d}_channel_isolated_gap.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)

        bits = "  ".join(f"{key}(pos={series[key][0].mean():.1f}mm,rot={series[key][1].mean():.1f}deg,"
                         f"grip={series[key][2].mean():.2f}mm)" for key, *_ in COMPARISONS)
        print(f"ep {ep_idx}: {bits}", flush=True)
        print(f"  saved {fpath}", flush=True)

    # ── aggregate: grouped bar chart of the isolated channel effect (mean +/- std across
    # episodes), same 3-metric x 2-family layout as the per-episode delta plot so the two
    # figures read the same way -- this is the plot that actually answers "which channel
    # dominates", so it's the bar heights (not 9 overlapping per-episode lines) that carry
    # the signal. ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for row_idx, mkey, mlabel in METRICS:
        ax_p, ax_q = axes[row_idx, 0], axes[row_idx, 1]
        labels = [c for c, *_ in CHANNELS]
        colors = [color for *_, color in CHANNELS]
        p_vals, p_errs, q_vals, q_errs = [], [], [], []
        for chan, p_key, q_key, color in CHANNELS:
            dp = np.array(means[f"{p_key}_{mkey}"]) - np.array(means[f"P_S_{mkey}"])
            dq = np.array(means[f"Q_S_{mkey}"]) - np.array(means[f"{q_key}_{mkey}"])
            p_vals.append(dp.mean()); p_errs.append(dp.std())
            q_vals.append(dq.mean()); q_errs.append(dq.std())
        ax_p.bar(labels, p_vals, yerr=p_errs, color=colors, capsize=4)
        ax_q.bar(labels, q_vals, yerr=q_errs, color=colors, capsize=4)
        ax_p.axhline(0, color="gray", lw=0.8)
        ax_q.axhline(0, color="gray", lw=0.8)
        ax_p.set_ylabel(mlabel); ax_p.grid(alpha=0.3, axis="y")
        ax_q.grid(alpha=0.3, axis="y")
        if row_idx == 0:
            ax_p.set_title("P-family Δ (adding real channel)")
            ax_q.set_title("Q-family Δ (removing real channel)")
    axes[2, 0].set_xlabel("channel swapped"); axes[2, 1].set_xlabel("channel swapped")
    fig.suptitle(f"mean isolated channel effect on predicted action, over {len(picks)} episodes "
                "(error bars = std across episodes)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    spath = out_dir / "channel_isolated_gap_summary.png"
    fig.savefig(spath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {spath}", flush=True)

    cpath = out_dir / "channel_isolated_gap_summary.csv"
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"saved {cpath}", flush=True)

    print("\n=== overall mean channel-isolated action-prediction difference across all episodes ===")
    for key, desc, _color in COMPARISONS:
        print(f"  {key} ({desc}):  pos={np.mean(means[f'{key}_pos_mm']):.2f}mm"
              f"   rot={np.mean(means[f'{key}_quat_deg']):.2f}deg"
              f"   grip={np.mean(means[f'{key}_grip_mm']):.3f}mm")

    print("\n=== channel ranking (delta from each family's own baseline, all 3 metrics) ===")
    for chan, p_key, q_key, _color in CHANNELS:
        bits_p, bits_q = [], []
        for _row_idx, mkey, mlabel in METRICS:
            unit = "mm" if mkey != "quat_deg" else "deg"
            dp = np.mean(means[f"{p_key}_{mkey}"]) - np.mean(means[f"P_S_{mkey}"])
            dq = np.mean(means[f"Q_S_{mkey}"]) - np.mean(means[f"{q_key}_{mkey}"])
            short = mkey.split("_")[0]
            bits_p.append(f"{short}_delta={dp:+.3f}{unit}")
            bits_q.append(f"{short}_delta={dq:+.3f}{unit}")
        print(f"  {chan:8s}  P-family (adding real {chan}): " + " ".join(bits_p)
              + f"   |  Q-family (removing real {chan}): " + " ".join(bits_q))


if __name__ == "__main__":
    main()
