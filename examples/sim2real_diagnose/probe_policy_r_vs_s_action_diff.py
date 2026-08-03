"""R vs S ONLY -- a focused re-run of probe_policy_action_diff.py's "arm-only gap" comparison,
dropped down to a single line/bar so it's not sharing an axis with O_S/O_R. Same open-loop /
teacher-forced probing, same two conditions:

  R  real proprioception (ee_pos/ee_quat/gripper_width) + Condition R's point cloud (real arm
     points, real mushroom stripped out and replaced with the paired sim rollout's mushroom --
     see the README glossary in this directory for what "R cloud" means)
  S  the same sim rollout, fully unedited (arm AND mushroom both sim)

R vs S isolates the arm/proprioception gap specifically: the mushroom is identical (sim) in
both conditions, so it can't be the source of any predicted-action difference.

Outputs, per episode: `epNN_r_vs_s_action_diff.png` (pos/rot/gripper diff vs time, one line).
Aggregate: `r_vs_s_action_diff_summary.png` (per-episode mean, dashed = overall mean) and
`r_vs_s_action_diff_summary.csv`.

Usage (envs/dppo_deploy -- needs the DPPO policy code + torch + CUDA):
    uv run --project envs/dppo_deploy python examples/sim2real_diagnose/probe_policy_r_vs_s_action_diff.py \\
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
                    help="default: <pkl's dir>/policy_r_vs_s_action_diff/")
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

    out_dir = args.out_dir or (args.pkl.parent / "policy_r_vs_s_action_diff")
    out_dir.mkdir(parents=True, exist_ok=True)

    action_cfg = ActionConfig.from_dict(yaml.safe_load(open(args.action_config)))
    pipeline = ActionPipeline(action_cfg)
    action_dim = action_cfg.action_dim

    policy = DPPOPolicyAdapter(
        str(args.ckpt), str(args.normalization), obs_dim=args.obs_dim, action_dim=action_dim,
        cond_steps=args.cond_steps, act_steps=args.act_steps,
        ft_denoising_steps=args.ft_denoising_steps, device=args.device)

    means = {"pos_mm": [], "quat_deg": [], "grip_mm": []}
    csv_rows = []

    for ep_idx in picks:
        ep = episodes[ep_idx]
        obs = ep["observations"]
        ee_pos, ee_quat, gw = obs["ee_pos"], obs["ee_quat"], obs["gripper_width"]
        ee_pos_s, ee_quat_s, gw_s = obs["ee_pos_sim"], obs["ee_quat_sim"], obs["gripper_width_sim"]
        pc_r, pc_s = obs["point_cloud"], obs["point_cloud_sim"]
        T = ee_pos.shape[0]
        ts = np.arange(T)

        raw_R = _replay_condition(policy, ee_pos, ee_quat, gw, pc_r)
        raw_S = _replay_condition(policy, ee_pos_s, ee_quat_s, gw_s, pc_s)

        pos_r, quat_r, grip_r = _to_physical(pipeline, raw_R)
        pos_s, quat_s, grip_s = _to_physical(pipeline, raw_S)
        quat_s_aligned = rds._align_quat_sign(quat_r, quat_s)
        pos_d = np.linalg.norm(pos_r - pos_s, axis=1) * 1000
        quat_d = np.rad2deg(rds._quat_angular_diff(quat_r, quat_s_aligned))
        grip_d = np.abs(grip_r - grip_s)[:, 0] * 1000

        means["pos_mm"].append(float(pos_d.mean()))
        means["quat_deg"].append(float(quat_d.mean()))
        means["grip_mm"].append(float(grip_d.mean()))
        csv_rows.append({"episode": ep_idx, "n_frames": T,
                         "R_S_pos_mm_mean": float(pos_d.mean()),
                         "R_S_quat_deg_mean": float(quat_d.mean()),
                         "R_S_grip_mm_mean": float(grip_d.mean())})

        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(ts, pos_d, color="tab:green", lw=1.8)
        axes[1].plot(ts, quat_d, color="tab:green", lw=1.8)
        axes[2].plot(ts, grip_d, color="tab:green", lw=1.8)
        axes[0].set_ylabel("action pos diff (mm)"); axes[0].grid(alpha=0.3)
        axes[1].set_ylabel("action rot diff (deg)"); axes[1].grid(alpha=0.3)
        axes[2].set_ylabel("action gripper diff (mm)"); axes[2].set_xlabel("frame t")
        axes[2].grid(alpha=0.3)
        fig.suptitle(f"hybrid ep {ep_idx} — R vs S action-prediction difference "
                    f"(arm-only gap, open-loop, no error accumulation)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fpath = out_dir / f"ep{ep_idx:02d}_r_vs_s_action_diff.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)

        print(f"ep {ep_idx}: R_S(pos={pos_d.mean():.1f}mm,rot={quat_d.mean():.1f}deg,"
              f"grip={grip_d.mean():.2f}mm)", flush=True)
        print(f"  saved {fpath}", flush=True)

    # ── aggregate ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(picks, means["pos_mm"], "o-", color="tab:green")
    axes[0].axhline(np.mean(means["pos_mm"]), color="tab:green", ls="--", alpha=0.6)
    axes[1].plot(picks, means["quat_deg"], "o-", color="tab:green")
    axes[1].axhline(np.mean(means["quat_deg"]), color="tab:green", ls="--", alpha=0.6)
    axes[2].plot(picks, means["grip_mm"], "o-", color="tab:green")
    axes[2].axhline(np.mean(means["grip_mm"]), color="tab:green", ls="--", alpha=0.6)
    axes[0].set_ylabel("mean pos diff (mm)"); axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("mean rot diff (deg)"); axes[1].grid(alpha=0.3)
    axes[2].set_ylabel("mean gripper diff (mm)"); axes[2].set_xlabel("episode")
    axes[2].grid(alpha=0.3)
    fig.suptitle("R vs S per-episode mean action diff (dashed = overall mean across all episodes)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    spath = out_dir / "r_vs_s_action_diff_summary.png"
    fig.savefig(spath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {spath}", flush=True)

    cpath = out_dir / "r_vs_s_action_diff_summary.csv"
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"saved {cpath}", flush=True)

    print("\n=== overall mean R vs S action-prediction difference across all episodes ===")
    print(f"  pos={np.mean(means['pos_mm']):.2f}mm   rot={np.mean(means['quat_deg']):.2f}deg"
          f"   grip={np.mean(means['grip_mm']):.3f}mm")


if __name__ == "__main__":
    main()
