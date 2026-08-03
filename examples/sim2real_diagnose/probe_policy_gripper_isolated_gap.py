"""Per-PROPRIOCEPTION-ELEMENT decomposition, on top of probe_policy_isolated_gap.py's
point-cloud-vs-proprioception split. Isolates gripper_width specifically from the other two
proprioception channels (ee_pos, ee_quat) by building two more synthetic single-swap
conditions, each a variant of an existing condition with ONLY gripper_width flipped:

  Pg (point-cloud-only gap, WITH real gripper)
      sim ee_pos/ee_quat  +  REAL gripper_width  +  Condition R's point cloud
      -- i.e. condition P (probe_policy_isolated_gap.py) with its gripper_width swapped
      real; ee_pos/ee_quat and the point cloud are untouched from P.

  Qg (proprioception-only gap, WITH sim gripper)
      REAL ee_pos/ee_quat  +  sim gripper_width  +  Condition S's point cloud
      -- i.e. condition Q (probe_policy_isolated_gap.py) with its gripper_width swapped
      back to sim; ee_pos/ee_quat and the point cloud are untouched from Q.

Both are compared against S (all-sim), same as P and Q, so all five comparisons --
P_S, Pg_S, Q_S, Qg_S, R_S -- share the same baseline and are directly comparable on one
plot. The DELTA between P_S and Pg_S isolates "how much does adding real gripper_width
change the action, on top of an otherwise-sim-proprioception + real point cloud"; the delta
between Q_S and Qg_S isolates "how much does REMOVING real gripper_width change the action,
on top of an otherwise-real-proprioception + sim point cloud" -- together these bound
gripper_width's own contribution to the proprioception-only gap (Q_S) from two directions.

Same open-loop / teacher-forced probing as the other probes in this directory: at each frame
t the policy's history buffer is fed that condition's own ground-truth sequence up to t and
we record only the immediate next predicted action -- no closed-loop rollout, no error
accumulation.

Outputs, per episode: `epNN_gripper_isolated_gap.png` (diff vs time, 3 rows x 5 comparisons).
Aggregate: `gripper_isolated_gap_summary.png` and `gripper_isolated_gap_summary.csv`.

Usage (envs/dppo_deploy -- needs the DPPO policy code + torch + CUDA):
    uv run --project envs/dppo_deploy python examples/sim2real_diagnose/probe_policy_gripper_isolated_gap.py \\
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
    ("P_S",  "point-cloud-only gap (sim pos+quat+grip, R cloud)",              "tab:orange"),
    ("Pg_S", "  + real gripper (sim pos+quat, REAL grip, R cloud)",            "tab:red"),
    ("Q_S",  "proprioception-only gap (real pos+quat+grip, sim cloud)",        "tab:purple"),
    ("Qg_S", "  w/ sim gripper (real pos+quat, SIM grip, sim cloud)",          "tab:pink"),
    ("R_S",  "combined arm-only gap (reference, both real)",                   "tab:green"),
]


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
                    help="default: <pkl's dir>/policy_gripper_isolated_gap/")
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

    out_dir = args.out_dir or (args.pkl.parent / "policy_gripper_isolated_gap")
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
        raw_P  = _replay_condition(policy, ee_pos_s, ee_quat_s, gw_s, pc_r)   # sim proprio + R cloud
        raw_Q  = _replay_condition(policy, ee_pos,   ee_quat,   gw,   pc_s)   # real proprio + sim cloud
        raw_Pg = _replay_condition(policy, ee_pos_s, ee_quat_s, gw,   pc_r)   # sim pos+quat, REAL grip, R cloud
        raw_Qg = _replay_condition(policy, ee_pos,   ee_quat,   gw_s, pc_s)   # real pos+quat, SIM grip, sim cloud

        raws = {"S": raw_S, "R": raw_R, "P": raw_P, "Q": raw_Q, "Pg": raw_Pg, "Qg": raw_Qg}
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

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for key, desc, color in COMPARISONS:
            pos_d, quat_d, grip_d = series[key]
            ls = "--" if key in ("Pg_S", "Qg_S") else "-"
            axes[0].plot(ts, pos_d, ls, label=f"{key} — {desc}", color=color, lw=1.8)
            axes[1].plot(ts, quat_d, ls, label=f"{key} — {desc}", color=color, lw=1.8)
            axes[2].plot(ts, grip_d, ls, label=f"{key} — {desc}", color=color, lw=1.8)
        axes[0].set_ylabel("action pos diff (mm)"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=6.5)
        axes[1].set_ylabel("action rot diff (deg)"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=6.5)
        axes[2].set_ylabel("action gripper diff (mm)"); axes[2].set_xlabel("frame t")
        axes[2].grid(alpha=0.3); axes[2].legend(fontsize=6.5)
        fig.suptitle(f"hybrid ep {ep_idx} — gripper-width-isolated action gap "
                    f"(open-loop, no error accumulation)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fpath = out_dir / f"ep{ep_idx:02d}_gripper_isolated_gap.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)

        bits = "  ".join(f"{key}(pos={series[key][0].mean():.1f}mm,rot={series[key][1].mean():.1f}deg,"
                         f"grip={series[key][2].mean():.2f}mm)" for key, *_ in COMPARISONS)
        print(f"ep {ep_idx}: {bits}", flush=True)
        print(f"  saved {fpath}", flush=True)

    # ── aggregate ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for key, desc, color in COMPARISONS:
        p, q, g = means[f"{key}_pos_mm"], means[f"{key}_quat_deg"], means[f"{key}_grip_mm"]
        ls = "s--" if key in ("Pg_S", "Qg_S") else "o-"
        label = f"{key} — {desc}"
        axes[0].plot(picks, p, ls, label=label, color=color); axes[0].axhline(np.mean(p), color=color, ls=":", alpha=0.5)
        axes[1].plot(picks, q, ls, label=label, color=color); axes[1].axhline(np.mean(q), color=color, ls=":", alpha=0.5)
        axes[2].plot(picks, g, ls, label=label, color=color); axes[2].axhline(np.mean(g), color=color, ls=":", alpha=0.5)
    axes[0].set_ylabel("mean pos diff (mm)"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=6.5)
    axes[1].set_ylabel("mean rot diff (deg)"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=6.5)
    axes[2].set_ylabel("mean gripper diff (mm)"); axes[2].set_xlabel("episode")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=6.5)
    fig.suptitle("per-episode mean gripper-isolated action diff (dotted = overall mean across all episodes)")
    fig.tight_layout()
    spath = out_dir / "gripper_isolated_gap_summary.png"
    fig.savefig(spath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {spath}", flush=True)

    cpath = out_dir / "gripper_isolated_gap_summary.csv"
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"saved {cpath}", flush=True)

    print("\n=== overall mean gripper-isolated action-prediction difference across all episodes ===")
    for key, desc, _color in COMPARISONS:
        print(f"  {key} ({desc}):  pos={np.mean(means[f'{key}_pos_mm']):.2f}mm"
              f"   rot={np.mean(means[f'{key}_quat_deg']):.2f}deg"
              f"   grip={np.mean(means[f'{key}_grip_mm']):.3f}mm")


if __name__ == "__main__":
    main()
