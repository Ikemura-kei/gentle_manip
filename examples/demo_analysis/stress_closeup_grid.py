"""Closeup video of per-particle MPM stress DURING a v3-synthesized grasp, for
several categories side by side.

Genesis exposes per-particle (not per-mesh-vertex — MPM has no mesh, only a point
cloud of material particles) von Mises stress directly via the entity state:
`obj.get_state().von_mises` -> (num_envs, n_particles), already used by
gentle_manip/rewards/stress.py and the v3 gentleness gate. This script reads it raw
every step (genesis_worker.read_state() normally drops the per-particle array and
keeps only scalar summaries, to avoid shipping it over the worker's IPC each step —
here we call get_state() directly since we're in-process).

For each category: replay a previously-collected v3 episode's exact object-DR pose
(from dr_params.csv) and its recorded end-effector trajectory (ee_pos/ee_quat/
gripper_width, which is what GenesisWorker.step() consumes as absolute per-step
targets) through a *fresh* single-env worker, reading the true particle cloud +
stress at every frame. Renders a fixed-view, zoomed-in 3D scatter colored on a
yellow (low) -> red (high) scale as a fraction of the object's yield stress, then
stacks the per-category clips into one 1xN video.

Usage:
    uv run --project envs/sim python examples/demo_analysis/stress_closeup_grid.py \\
        --categories mushroom grape kiwi raspberry --out-dir /tmp/stress_closeup
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from gentle_manip.envs.genesis_worker import GenesisWorker
from gentle_manip.experiment import Experiment
from gentle_manip.tasks.single_lift import SingleLiftTask
from gentle_manip.robot.xarm7_sim import _np

REPO = Path(__file__).resolve().parents[2]

STRESS_CMAP = LinearSegmentedColormap.from_list("gentle_stress", ["#f6e27a", "#e8892f", "#b5251f"])


def _find_episode_and_dr(category: str):
    """Prefer the main (full) collection dir; fall back to the smoketest dir."""
    candidates = [
        REPO / "dataset/demos" / f"single_lift_{category}_soft",
        REPO / "dataset/demos_smoketest_v3_all16" / f"single_lift_{category}_soft",
    ]
    for base in candidates:
        run_dirs = sorted(base.glob("*")) if base.exists() else []
        for run_dir in run_dirs:
            data_pkl = run_dir / "data.pkl"
            shard = sorted(run_dir.glob("shard_*.pkl"))
            src = data_pkl if data_pkl.exists() else (shard[0] if shard else None)
            dr_csv = run_dir / "dr_params.csv"
            if src is not None and dr_csv.exists():
                with open(src, "rb") as f:
                    d = pickle.load(f)
                if d["episodes"]:
                    return d["episodes"][0], dr_csv, run_dir
    raise FileNotFoundError(f"No v3 episode+dr_params found for category={category}")


def _read_dr_row0(dr_csv: Path) -> dict:
    import csv
    with open(dr_csv) as f:
        row = next(csv.DictReader(f))
    return {k: float(v) for k, v in row.items() if k not in ("batch", "env", "success", "flipped")}


def _replay_and_capture(category: str, n_frames_cap: int = 260):
    ep, dr_csv, run_dir = _find_episode_and_dr(category)
    dr = _read_dr_row0(dr_csv)
    print(f"[{category}] replaying from {run_dir.name}, {len(ep['actions'])} steps")

    exp = Experiment.load(f"single_lift_{category}_soft_easy")
    task = SingleLiftTask(exp.task_cfg)
    spec = task.scene_spec

    worker = GenesisWorker(spec, num_envs=1, show_viewer=False,
                           settle_steps=30, settle_max_steps=200,
                           settle_vel_thresh=0.002, render_obs_cameras=False)
    try:
        object_dxy = np.array([[dr["obj_dx"], dr["obj_dy"]]], dtype=np.float32)
        object_euler = np.deg2rad(np.array(
            [[dr["roll_deg"], dr["pitch_deg"], dr["yaw_deg"]]], dtype=np.float32))
        home_offset = np.array([[dr["home_dx"], dr["home_dy"], dr["home_dz"]]], dtype=np.float32)
        worker.reset(object_dxy=object_dxy, object_euler=object_euler, home_offset=home_offset)

        obs = ep["observations"]
        T = min(len(ep["actions"]), n_frames_cap)
        frames_pos, frames_stress = [], []
        for t in range(T):
            cur_pos  = obs["ee_pos"][t][None].astype(np.float32)
            cur_quat = obs["ee_quat"][t][None].astype(np.float32)
            cur_grip = obs["gripper_width"][t].astype(np.float32)  # already (1,) — no extra batch dim
            worker.step(cur_pos, cur_quat, cur_grip)
            st = worker.handle.objects[0].get_state()
            frames_pos.append(_np(st.pos)[0].copy())        # (n_p, 3)
            frames_stress.append(_np(st.von_mises)[0].copy())  # (n_p,)
        return frames_pos, frames_stress, task.object_yield_stress
    finally:
        worker.close()


def _render_category_video(category: str, frames_pos, frames_stress, yield_stress: float,
                           out_path: Path, fps: int = 30):
    all_pos = np.stack(frames_pos)          # (T, n_p, 3)
    all_stress = np.stack(frames_stress)    # (T, n_p)
    # Per-object bounding radius from a single representative (mid-trajectory) frame,
    # not the union across all frames -- a union inflates the box with the arm/gripper
    # sweeping through and defeats the "closeup" framing. Tight padding for a real closeup.
    ref = all_pos[len(all_pos) // 2]
    ref_center = ref.mean(axis=0)
    half_extent = max(np.percentile(np.linalg.norm(ref - ref_center, axis=-1), 98) * 1.6, 0.018)

    # Per-frame centroids, lightly smoothed (EMA) so the camera follows the object
    # as it's lifted (height, and any lateral drift) without being global-average
    # framing -- a fixed whole-trajectory center left the object drifting out of
    # frame once the lift phase raised it well above its resting position.
    raw_centers = all_pos.mean(axis=1)                    # (T, 3)
    frame_centers = np.empty_like(raw_centers)
    frame_centers[0] = raw_centers[0]
    alpha = 0.15
    for t in range(1, len(raw_centers)):
        frame_centers[t] = alpha * raw_centers[t] + (1 - alpha) * frame_centers[t - 1]

    # Adaptive color scale: normalize by this trajectory's OWN peak stress (not the
    # material yield) so the red<->yellow range is actually visible even though v3's
    # grasps stay far below yield by design (gentle) -- the metric of interest here is
    # the RELATIVE stress buildup during the grasp, not absolute proximity to damage.
    vmax = max(np.percentile(all_stress, 99.5), 1e-6)
    crush_frac = float(all_stress.max() / max(yield_stress, 1e-6))

    fig = plt.figure(figsize=(4.2, 4.2), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    import io, imageio
    writer_frames = []
    for pos, stress, center in zip(frames_pos, frames_stress, frame_centers):
        ax.cla()
        ax.set_facecolor("black")
        frac = np.clip(stress / vmax, 0, 1.0)
        colors = STRESS_CMAP(frac)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=42, linewidths=0)
        ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
        ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
        ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.view_init(elev=14, azim=50)
        ax.set_title(category.replace("_", " "), color="white", fontsize=13, pad=-6)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="black")
        buf.seek(0)
        writer_frames.append(imageio.imread(buf)[:, :, :3])
    imageio.mimwrite(str(out_path), writer_frames, fps=fps, quality=8)
    plt.close(fig)
    print(f"  -> {out_path}  ({len(writer_frames)} frames, peak stress = "
          f"{crush_frac*100:.1f}% of yield)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["mushroom", "grape", "kiwi", "raspberry"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=260)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    for cat in args.categories:
        frames_pos, frames_stress, yield_stress = _replay_and_capture(cat, args.max_frames)
        out_path = args.out_dir / f"{cat}_stress_closeup.mp4"
        _render_category_video(cat, frames_pos, frames_stress, yield_stress, out_path)
        clip_paths.append(out_path)

    print("\nDone. Clips:")
    for p in clip_paths:
        print(" ", p)


if __name__ == "__main__":
    main()
