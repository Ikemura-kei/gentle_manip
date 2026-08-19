"""Grasp-orientation visualization for a whole v3 (FEM gentleness) demo category.

Unlike grasp_pose_analysis.py (single data.pkl, v2 schema), collect_demos_synth_v3.py
writes SHARDED demos (shard_0000.pkl, shard_0001.pkl, ...) under one or more run dirs
(a resumed collection has several run dirs). This script globs every shard under the
given category directory, merges all episodes, then plots the grasp pose (position +
orientation) actually EXECUTED in each trajectory in 3D, colored by the FEM-measured
top10 von Mises stress fraction of yield (priv_stress[:, 1]) recorded at that same
step — so the plot doubles as a v3-quality visual (gentle grasps in cool colors,
harsher ones warm), not just an orientation-diversity plot.

Usage:
    uv run --project envs/sim python examples/demo_analysis/grasp_pose_analysis_v3.py \\
        dataset/demos/single_lift_mushroom_soft --out-dir dataset/demos/single_lift_mushroom_soft
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

GW_EPS = 1e-3  # gripper-width drop that signals closing start


def _find_grasp_step(ep: dict) -> int | None:
    gw = ep["observations"]["gripper_width"][:, 0]
    closing = np.where(gw < gw[0] - GW_EPS)[0]
    if len(closing) == 0:
        return None
    return max(0, int(closing[0]) - 1)


def _load_all_episodes(category_dir: Path) -> tuple[list[dict], str]:
    shard_files = sorted(glob.glob(str(category_dir / "**" / "shard_*.pkl"), recursive=True))
    if not shard_files:
        # fall back to a flat data.pkl (v2-style)
        flat = sorted(glob.glob(str(category_dir / "**" / "data.pkl"), recursive=True))
        shard_files = flat
    episodes = []
    for f in shard_files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        episodes.extend(d["episodes"])
    task_name = category_dir.name
    return episodes, task_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("category_dir", type=Path,
                    help="e.g. dataset/demos/single_lift_mushroom_soft (searches all run-dir shards under it)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--yield-stress", type=float, default=None,
                    help="Pa, for annotating the stress colorbar in kPa-of-yield terms if known")
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir else args.category_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes, task_name = _load_all_episodes(args.category_dir)
    print(f"Loaded {len(episodes)} episodes from shards under {args.category_dir}")

    grasp_pos, grasp_quat, grasp_stress_frac, grasp_stress_kpa = [], [], [], []
    skipped = 0
    for ep in episodes:
        t = _find_grasp_step(ep)
        if t is None:
            skipped += 1
            continue
        obs = ep["observations"]
        grasp_pos.append(obs["ee_pos"][t])
        grasp_quat.append(obs["ee_quat"][t])
        if "priv_stress" in obs:
            # priv_stress = [mean_frac, top10_frac] of yield (see PrivilegedConfig.stress)
            grasp_stress_frac.append(float(obs["priv_stress"][t, 1]))
        else:
            grasp_stress_frac.append(np.nan)

    print(f"  Grasp step found: {len(grasp_pos)} / {len(episodes)}  (skipped {skipped})")
    grasp_pos = np.asarray(grasp_pos, dtype=np.float64)
    grasp_quat = np.asarray(grasp_quat, dtype=np.float64)
    grasp_stress_frac = np.asarray(grasp_stress_frac, dtype=np.float64)

    quat_xyzw = np.concatenate([grasp_quat[:, 1:], grasp_quat[:, :1]], axis=1)
    mats = Rotation.from_quat(quat_xyzw).as_matrix()
    approach = mats[:, :, 2]   # EE z-axis: approach/closing direction into the object
    N = len(grasp_pos)

    # ── Figure: grasp position + approach direction, colored by FEM stress fraction ──
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    valid_stress = grasp_stress_frac[~np.isnan(grasp_stress_frac)]
    vmax = np.nanpercentile(grasp_stress_frac, 95) if len(valid_stress) else 1.0
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(0, max(vmax, 1e-6))
    colors = cmap(norm(np.nan_to_num(grasp_stress_frac, nan=0.0)))

    ax.scatter(grasp_pos[:, 0], grasp_pos[:, 1], grasp_pos[:, 2],
               c=colors, s=40, edgecolor="k", linewidth=0.3)
    arrow_len = 0.03
    ax.quiver(grasp_pos[:, 0], grasp_pos[:, 1], grasp_pos[:, 2],
              -approach[:, 0], -approach[:, 1], -approach[:, 2],  # point INTO the object
              length=arrow_len, normalize=True, color=colors, linewidth=1.6, arrow_length_ratio=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.1)
    cbar.set_label("top10 von Mises stress / yield (FEM v3, at grasp)")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"v3 FEM grasp poses across trajectories — {task_name}  (N={N})")
    plt.tight_layout()
    out1 = out_dir / "grasp_pose_v3_3d.png"
    plt.savefig(out1, dpi=150)
    print(f"Saved → {out1}")

    # rotating video
    import io
    import imageio
    FPS, DURATION = 30, 8
    N_FRAMES = FPS * DURATION
    frames = []
    print(f"  Rendering {N_FRAMES} frames …", flush=True)
    for i in range(N_FRAMES):
        ax.view_init(elev=22, azim=360.0 * i / N_FRAMES)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf)[:, :, :3])
    out_vid = out_dir / "grasp_pose_v3_3d.mp4"
    imageio.mimwrite(str(out_vid), frames, fps=FPS, quality=8)
    print(f"Saved → {out_vid}")
    plt.close(fig)

    # ── Orientation-only sphere plot (diversity, independent of position) ──
    fig2 = plt.figure(figsize=(8, 7))
    ax2 = fig2.add_subplot(111, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:25j]
    ax2.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                       color="lightgray", linewidth=0.3, alpha=0.4)
    ox = np.zeros(N)
    ax2.quiver(ox, ox, ox, approach[:, 0], approach[:, 1], approach[:, 2],
              length=1.0, normalize=True, color=colors, linewidth=1.4, arrow_length_ratio=0.15)
    ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)
    ax2.set_title(f"Approach-direction diversity (unit sphere) — {task_name}")
    cbar2 = plt.colorbar(sm, ax=ax2, shrink=0.55, pad=0.1)
    cbar2.set_label("top10 stress / yield")
    plt.tight_layout()
    out2 = out_dir / "grasp_orientation_sphere_v3.png"
    plt.savefig(out2, dpi=150)
    print(f"Saved → {out2}")
    plt.close(fig2)

    print(f"\n=== Grasp position (m), N={N} ===")
    for i, name in enumerate(["x", "y", "z"]):
        vv = grasp_pos[:, i]
        print(f"  {name}: μ={vv.mean():.3f}  σ={vv.std():.3f}  range=[{vv.min():.3f}, {vv.max():.3f}]")
    if len(valid_stress):
        print(f"\n=== FEM top10 stress/yield at grasp ===")
        print(f"  μ={valid_stress.mean():.3f}  σ={valid_stress.std():.3f}  "
              f"range=[{valid_stress.min():.3f}, {valid_stress.max():.3f}]  (>1.0 = crush)")


if __name__ == "__main__":
    main()
