"""Analyze grasp pose diversity: EE orientation at the moment gripper starts closing.

For each episode, the grasp pose is the EE pose at the last step BEFORE the gripper
begins to narrow. Orientation diversity is visualized two ways:

  grasp_orientation_sphere.png  — x/y/z axes of the EE frame plotted as points on
                                   a unit sphere (3 sub-plots: approach / closing / all)
  grasp_euler_distribution.png  — roll / pitch / yaw histograms

Usage:
    uv run --project envs/sim python examples/demo_analysis/grasp_pose_analysis.py \\
        dataset/demos/single_lift_mushroom_rigid/26-07-26-mah/data.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

GW_EPS = 1e-3   # gripper-width drop that signals closing start


def _find_grasp_step(ep: dict) -> int | None:
    """Index of the last step before gripper starts closing, or None if not found."""
    gw = ep["observations"]["gripper_width"][:, 0]
    closing = np.where(gw < gw[0] - GW_EPS)[0]
    if len(closing) == 0:
        return None
    return max(0, int(closing[0]) - 1)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_pkl", type=Path, nargs="?",
                    default=Path(__file__).resolve().parents[2]
                            / "dataset/demos/single_lift_mushroom_rigid/26-07-25-kqs/data.pkl")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: same folder as data_pkl)")
    args = ap.parse_args()
    data_pkl = args.data_pkl
    out_dir  = args.out_dir if args.out_dir else data_pkl.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    d        = pickle.load(open(data_pkl, "rb"))
    episodes = d["episodes"]
    print(f"Loaded {len(episodes)} episodes from {data_pkl.name}")

    # ── Extract grasp-moment poses ─────────────────────────────────────────────
    grasp_pos  = []
    grasp_quat = []   # wxyz convention
    skipped    = 0
    for ep in episodes:
        t = _find_grasp_step(ep)
        if t is None:
            skipped += 1
            continue
        grasp_pos.append(ep["observations"]["ee_pos"][t])
        grasp_quat.append(ep["observations"]["ee_quat"][t])

    print(f"  Grasp step found: {len(grasp_pos)} / {len(episodes)}  (skipped {skipped})")
    grasp_pos  = np.array(grasp_pos,  dtype=np.float64)   # (N, 3)
    grasp_quat = np.array(grasp_quat, dtype=np.float64)   # (N, 4) wxyz

    # scipy uses xyzw convention
    quat_xyzw = np.concatenate([grasp_quat[:, 1:], grasp_quat[:, :1]], axis=1)
    R    = Rotation.from_quat(quat_xyzw)
    mats = R.as_matrix()           # (N, 3, 3)

    x_axes = mats[:, :, 0]        # closing direction
    y_axes = mats[:, :, 1]        # lateral
    z_axes = mats[:, :, 2]        # approach direction (into object)

    eulers = R.as_euler("xyz", degrees=True)   # (N, 3) roll / pitch / yaw
    N = len(grasp_pos)
    colors = np.arange(N)          # episode index → colormap

    # ── Figure 1: 3-D quiver — approach directions from a shared origin ────────
    # All arrows start at (0,0,0) and are unit length, so they live on the unit
    # sphere. This shows pure orientation diversity without spatial clutter.
    # Each arrow is the EE z-axis (approach / forward direction) in world frame.
    cmap     = plt.get_cmap("plasma")
    norm     = plt.Normalize(0, N - 1)
    col_rgba = cmap(norm(np.arange(N)))

    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Faint unit-sphere wireframe for spatial reference
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 25j]
    ax.plot_wireframe(
        np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
        color="lightgray", linewidth=0.3, alpha=0.4,
    )

    # One arrow per episode, all from origin
    ox = np.zeros(N)
    ax.quiver(
        ox, ox, ox,
        z_axes[:, 0], z_axes[:, 1], z_axes[:, 2],
        length=1.0, normalize=True,
        color=col_rgba, linewidth=1.4, arrow_length_ratio=0.15,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="episode index", shrink=0.55, pad=0.08)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    ax.set_title(
        f"Gripper approach direction (EE z-axis) in world frame\n"
        f"{data_pkl.parent.name}  ({N} episodes)",
        fontsize=11,
    )

    plt.tight_layout()
    out1 = out_dir / "grasp_orientation_vectors.png"
    plt.savefig(out1, dpi=150)
    print(f"Saved → {out1}")

    # ── Rotating video ─────────────────────────────────────────────────────────
    import io
    import imageio

    FPS      = 30
    DURATION = 8           # seconds for one full 360° rotation
    N_FRAMES = FPS * DURATION
    ELEV     = 22          # fixed elevation angle

    print(f"  Rendering {N_FRAMES} frames …", flush=True)
    frames = []
    for i in range(N_FRAMES):
        azim = 360.0 * i / N_FRAMES
        ax.view_init(elev=ELEV, azim=azim)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        frames.append(imageio.imread(buf)[:, :, :3])

    out_vid = out_dir / "grasp_orientation_vectors.mp4"
    imageio.mimwrite(str(out_vid), frames, fps=FPS, quality=8)
    print(f"Saved → {out_vid}")

    plt.close(fig)

    # ── Figure 2: Euler angle histograms ──────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
    fig2.suptitle(
        f"Grasp Euler angles (intrinsic XYZ) — {data_pkl.parent.name}  ({N} episodes)",
        fontsize=13,
    )
    for i, (name, color) in enumerate(zip(
        ["Roll (X, °)", "Pitch (Y, °)", "Yaw (Z, °)"],
        ["#4878d0", "#ee854a", "#6acc65"],
    )):
        ax = axes2[i]
        vals = eulers[:, i]
        ax.hist(vals, bins=min(20, N // 2 + 1), color=color, edgecolor="none", alpha=0.85)
        ax.set_title(
            f"{name}\nμ={vals.mean():.1f}°   σ={vals.std():.1f}°   "
            f"[{vals.min():.1f}°, {vals.max():.1f}°]",
            fontsize=10,
        )
        ax.set_xlabel("Degrees", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out2 = out_dir / "grasp_euler_distribution.png"
    plt.savefig(out2, dpi=150)
    print(f"Saved → {out2}")
    plt.close(fig2)

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n=== Grasp orientation (Euler XYZ, degrees) ===")
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        v = eulers[:, i]
        print(f"  {name:<6}: μ={v.mean():+.1f}°  σ={v.std():.1f}°  "
              f"range=[{v.min():.1f}°, {v.max():.1f}°]")

    print(f"\n=== Grasp position (m) ===")
    for i, name in enumerate(["x", "y", "z"]):
        v = grasp_pos[:, i]
        print(f"  {name}: μ={v.mean():.3f}  σ={v.std():.3f}  "
              f"range=[{v.min():.3f}, {v.max():.3f}]")


if __name__ == "__main__":
    main()
