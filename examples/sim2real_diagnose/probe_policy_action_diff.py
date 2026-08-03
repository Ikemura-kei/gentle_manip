"""Probe a trained DPPO policy on the paired hybrid dataset's three conditions (O, R, S) —
per-TIMESTEP action prediction difference, open-loop / teacher-forced: at each frame t, the
policy's history buffer is fed that CONDITION's own ground-truth (real or sim) observation
sequence up to t, and we record what it would predict RIGHT NOW. No closed-loop rollout, so
no error accumulation across time — this isolates "does a different observation source
change the policy's immediate action" from "how would errors compound over a real rollout"
(the latter is a separate, harder question, deliberately out of scope here).

Three comparisons (same structure as compute_hybrid_distances.py):
  O vs S — full sim2real action gap (arm AND object together)
  O vs R — edit-only sanity check (mushroom swapped, arm/proprioception unchanged)
  R vs S — arm-only action gap (object held constant/sim in both)
Plus a fourth PIPELINE sanity check: O's predicted action vs the ORIGINAL recorded action
from the real deployment (episodes here are literally open-loop replays of that same
policy's own real rollout, so O's replayed prediction should closely reproduce what was
actually commanded live — a large mismatch would mean something in this probe is wired
wrong, not a real finding).

Actions are compared in PHYSICAL units (via the SAME abs_pose_abs_gripper ActionPipeline
used everywhere else in this repo), decomposed like every other diagnostic in this
directory: position (mm), rotation (degrees, via rot6d -> quat angular diff), gripper (mm).

Outputs, per episode: `epNN_action_diff.png` (diff vs time, 3 rows x 3 comparisons).
Aggregate: `action_diff_summary.png` (per-episode mean + overall mean) and
`action_diff_summary.csv`. Sanity check printed + saved to `sanity_check.csv`.

Usage (envs/dppo_deploy — needs the DPPO policy code + torch + CUDA):
    uv run --project envs/dppo_deploy python examples/sim2real_diagnose/probe_policy_action_diff.py \\
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


def _to_physical(pipeline: ActionPipeline, raw_actions: np.ndarray):
    """(T, 10) raw actions -> physical pos (T,3), quat_wxyz (T,4), gripper (T,1)."""
    phys = pipeline.process(raw_actions)     # (T, 8) pos+quat+gripper
    return phys[:, :3], phys[:, 3:7], phys[:, 7:8]


def _replay_condition(policy: DPPOPolicyAdapter, ee_pos, ee_quat, gripper_width, point_cloud) -> np.ndarray:
    """Feed one condition's ground-truth obs sequence through the policy frame-by-frame,
    recording the IMMEDIATE next action (first of the chunk) predicted at each t from the
    history accumulated so far (open-loop / teacher-forced). Returns (T, action_dim) raw."""
    T = ee_pos.shape[0]
    action_dim = policy.action_min.shape[0]
    actions = np.zeros((T, action_dim), np.float32)

    def _obs(t):
        return {"ee_pos": ee_pos[t], "ee_quat": ee_quat[t], "gripper_width": gripper_width[t],
                "point_cloud": point_cloud[t]}

    policy.reset(_obs(0))
    actions[0] = policy.predict()[0]
    for t in range(1, T):
        policy.push(_obs(t))
        actions[t] = policy.predict()[0]
    return actions


COMPARISONS = [("O_S", "chamfer-matched full gap", "O vs S (full gap)", "tab:red"),
               ("O_R", "edit-only sanity", "O vs R (edit-only)", "tab:blue"),
               ("R_S", "arm-only gap", "R vs S (arm-only gap)", "tab:green")]


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
                    help="default: <pkl's dir>/policy_action_diff/")
    ap.add_argument("--episodes", default="", help="comma-sep explicit episode indices (default: all)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pickle.load(open(args.pkl, "rb"))
    episodes = data["episodes"]
    print(f"Loaded {len(episodes)} episodes from {args.pkl}", flush=True)
    if "point_cloud_orig" not in episodes[0]["observations"]:
        raise RuntimeError("point_cloud_orig missing -- run add_original_real_cloud.py first")

    picks = ([int(x) for x in args.episodes.split(",")] if args.episodes.strip()
             else list(range(len(episodes))))

    out_dir = args.out_dir or (args.pkl.parent / "policy_action_diff")
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
    sanity_rows = []

    for ep_idx in picks:
        ep = episodes[ep_idx]
        obs = ep["observations"]
        ee_pos, ee_quat, gw = obs["ee_pos"], obs["ee_quat"], obs["gripper_width"]
        ee_pos_s, ee_quat_s, gw_s = obs["ee_pos_sim"], obs["ee_quat_sim"], obs["gripper_width_sim"]
        pc_o, pc_r, pc_s = obs["point_cloud_orig"], obs["point_cloud"], obs["point_cloud_sim"]
        T = ee_pos.shape[0]
        ts = np.arange(T)

        raw_O = _replay_condition(policy, ee_pos, ee_quat, gw, pc_o)
        raw_R = _replay_condition(policy, ee_pos, ee_quat, gw, pc_r)
        raw_S = _replay_condition(policy, ee_pos_s, ee_quat_s, gw_s, pc_s)

        # pipeline sanity check: O's replayed prediction vs the actually-recorded action
        pos_o, quat_o, grip_o = _to_physical(pipeline, raw_O)
        pos_rec, quat_rec, grip_rec = _to_physical(pipeline, ep["actions"])
        quat_rec_aligned = rds._align_quat_sign(quat_o, quat_rec)
        sanity_pos = float(np.linalg.norm(pos_o - pos_rec, axis=1).mean() * 1000)
        sanity_quat = float(np.rad2deg(rds._quat_angular_diff(quat_o, quat_rec_aligned).mean()))
        sanity_grip = float(np.abs(grip_o - grip_rec).mean() * 1000)
        sanity_rows.append({"episode": ep_idx, "pos_mm": sanity_pos, "quat_deg": sanity_quat,
                            "grip_mm": sanity_grip})
        print(f"ep {ep_idx} [sanity: O-replay vs recorded]  pos={sanity_pos:.2f}mm "
              f"quat={sanity_quat:.2f}deg grip={sanity_grip:.2f}mm", flush=True)

        raws = {"O": raw_O, "R": raw_R, "S": raw_S}
        phys = {k: _to_physical(pipeline, v) for k, v in raws.items()}

        series = {}
        for key, _desc, label, color in COMPARISONS:
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

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        for key, _desc, label, color in COMPARISONS:
            pos_d, quat_d, grip_d = series[key]
            axes[0].plot(ts, pos_d, label=label, color=color, lw=1.8)
            axes[1].plot(ts, quat_d, label=label, color=color, lw=1.8)
            axes[2].plot(ts, grip_d, label=label, color=color, lw=1.8)
        axes[0].set_ylabel("action pos diff (mm)"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
        axes[1].set_ylabel("action rot diff (deg)"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
        axes[2].set_ylabel("action gripper diff (mm)"); axes[2].set_xlabel("frame t")
        axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
        fig.suptitle(f"hybrid ep {ep_idx} — policy action prediction difference vs time "
                    f"(open-loop, no error accumulation)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fpath = out_dir / f"ep{ep_idx:02d}_action_diff.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)

        bits = "  ".join(f"{key}(pos={series[key][0].mean():.1f}mm,rot={series[key][1].mean():.1f}deg,"
                         f"grip={series[key][2].mean():.1f}mm)" for key, *_ in COMPARISONS)
        print(f"ep {ep_idx}: {bits}", flush=True)
        print(f"  saved {fpath}", flush=True)

    # ── aggregate ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for key, _desc, label, color in COMPARISONS:
        p, q, g = means[f"{key}_pos_mm"], means[f"{key}_quat_deg"], means[f"{key}_grip_mm"]
        axes[0].plot(picks, p, "o-", label=label, color=color); axes[0].axhline(np.mean(p), color=color, ls="--", alpha=0.5)
        axes[1].plot(picks, q, "o-", label=label, color=color); axes[1].axhline(np.mean(q), color=color, ls="--", alpha=0.5)
        axes[2].plot(picks, g, "o-", label=label, color=color); axes[2].axhline(np.mean(g), color=color, ls="--", alpha=0.5)
    axes[0].set_ylabel("mean pos diff (mm)"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_ylabel("mean rot diff (deg)"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    axes[2].set_ylabel("mean gripper diff (mm)"); axes[2].set_xlabel("episode")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)
    fig.suptitle("per-episode mean action diff (dashed = overall mean across all episodes)")
    fig.tight_layout()
    spath = out_dir / "action_diff_summary.png"
    fig.savefig(spath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {spath}", flush=True)

    cpath = out_dir / "action_diff_summary.csv"
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print(f"saved {cpath}", flush=True)

    spath2 = out_dir / "sanity_check.csv"
    with open(spath2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sanity_rows[0].keys()))
        w.writeheader(); w.writerows(sanity_rows)
    print(f"saved {spath2}", flush=True)

    print("\n=== overall mean action-prediction difference across all episodes ===")
    for key, _desc, label, _color in COMPARISONS:
        print(f"  {label:25s}  pos={np.mean(means[f'{key}_pos_mm']):.2f}mm"
              f"   rot={np.mean(means[f'{key}_quat_deg']):.2f}deg"
              f"   grip={np.mean(means[f'{key}_grip_mm']):.2f}mm")
    print(f"\n=== sanity check (O-replay vs actually-recorded action), mean over episodes ===")
    print(f"  pos={np.mean([r['pos_mm'] for r in sanity_rows]):.2f}mm"
          f"  quat={np.mean([r['quat_deg'] for r in sanity_rows]):.2f}deg"
          f"  grip={np.mean([r['grip_mm'] for r in sanity_rows]):.2f}mm")


if __name__ == "__main__":
    main()
