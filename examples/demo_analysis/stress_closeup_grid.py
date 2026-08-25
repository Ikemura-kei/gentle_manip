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

# Tracking-camera params -- MUST match _render_category_video's matplotlib
# view_init(elev=14, azim=50) and its EMA-smoothed centroid tracking exactly,
# so the RGB video is the same viewpoint on the same rollout, not just "some
# real camera somewhere" (caught live 2026-08-19: the first RGB pass used the
# scene's static cam_ext -- a totally different framing from the closeup's
# custom zoomed/tracking view, and a SEPARATE replay call, so neither the
# viewpoint nor the exact rollout matched).
TRACK_ELEV_DEG = 14.0
TRACK_AZIM_DEG = 50.0
TRACK_DIST_MULT = 2.75   # ~= 1/tan(fov/2) for fov=40, so the object's half_extent
                        # fills the frame similarly to the plot's tight ±half_extent axes
TRACK_MIN_DIST_M = 0.22  # floor on camera distance -- Genesis cameras default to
                        # near=0.05m clip (third_party/genesis/genesis/vis/camera.py);
                        # half_extent*TRACK_DIST_MULT alone put small objects (mushroom
                        # half_extent~0.02m -> r~0.055m) INSIDE the near plane, so the
                        # object (and often the whole robot) rendered as pure background
                        # (caught live 2026-08-20: mushroom/tomato/raspberry RGB clips
                        # showed nothing). 0.22m keeps >=0.15m clearance from the near
                        # plane for every category's object+gripper.
TRACK_FOV = 40.0
TRACK_EMA_ALPHA = 0.15  # must match _render_category_video's alpha


def _spherical_offset(half_extent: float) -> np.ndarray:
    r = max(half_extent * TRACK_DIST_MULT, TRACK_MIN_DIST_M)
    elev = np.deg2rad(TRACK_ELEV_DEG)
    azim = np.deg2rad(TRACK_AZIM_DEG)
    return np.array([r * np.cos(elev) * np.cos(azim),
                     r * np.cos(elev) * np.sin(azim),
                     r * np.sin(elev)])


def _find_episodes_and_dr(category: str):
    """Prefer the main (full) collection dir; fall back to the smoketest dir.
    Returns ALL episodes (not just the first) plus the dr_params.csv rows, so
    the caller can try several candidates and verify each one actually lifts
    the object on replay -- dr_params.csv only logs a PREFIX of attempts (seen
    live: 20 rows for a 50-episode banana collection, since it's rewritten
    per resume-attempt rather than accumulated across the whole run), so row
    index i is not reliably episode i beyond the first several; combined with
    MPM's known rollout non-determinism, blindly replaying episode 0 against
    dr row 0 can silently reproduce a FAILED lift even though the original
    collected episode succeeded (caught live 2026-08-19 -- several of the
    first-rendered closeups showed a failed/missed grasp on replay)."""
    candidates = [
        REPO / "dataset/demos" / f"single_lift_{category}_soft",
        REPO / "dataset/demos_smoketest_v3_all16" / f"single_lift_{category}_soft",
    ]
    for base in candidates:
        run_dirs = sorted(base.glob("*")) if base.exists() else []
        for run_dir in run_dirs:
            if "rollout" in run_dir.name:
                continue   # RLDG rollout dirs -- not raw v3 synthesis output
            data_pkl = run_dir / "data.pkl"
            shard = sorted(run_dir.glob("shard_*.pkl"))
            src = data_pkl if data_pkl.exists() else (shard[0] if shard else None)
            dr_csv = run_dir / "dr_params.csv"
            if src is not None and dr_csv.exists():
                with open(src, "rb") as f:
                    d = pickle.load(f)
                if d["episodes"]:
                    return d["episodes"], dr_csv, run_dir
    raise FileNotFoundError(f"No v3 episode+dr_params found for category={category}")


def _read_dr_rows(dr_csv: Path) -> list:
    import csv
    with open(dr_csv) as f:
        rows = list(csv.DictReader(f))
    return [{k: float(v) for k, v in row.items() if k not in ("batch", "env", "success", "flipped")}
           for row in rows]


def _replay_succeeded(frames_pos, yield_stress: float, frames_stress,
                      min_rise_m: float = 0.04) -> tuple:
    """A replay counts as a successful gentle lift iff the object's centroid
    rose meaningfully (holds -> actually lifted, not a missed/slipped grasp)
    and never exceeded yield stress (holds -> gentle, matching the same
    height+no-crush success definition used everywhere else in this
    project). min_rise_m=0.04 (not the original demo's own ~0.15-0.2m rise):
    a REPLAY on a freshly-reset scene often lifts somewhat less than the
    original collection run even when the grasp genuinely holds (dr_params.csv
    row/episode misalignment means the replay's reset conditions are only an
    approximation of what the episode's actions were actually tuned for) --
    0.04m is comfortably above resting-state jitter (<1cm) while not
    rejecting genuine, visually-clear lifts. Returns (ok, rise, peak_stress)
    so callers can log the real numbers instead of just pass/fail."""
    z = np.stack(frames_pos).mean(axis=1)[:, 2]
    rise = float(z[-10:].mean() - z[0])
    peak_stress = float(np.max([s.max() for s in frames_stress]))
    return rise >= min_rise_m and peak_stress <= yield_stress, rise, peak_stress


def _replay_one(worker, ep, dr, n_frames_cap, capture_rgb: bool = False, track_cam=None):
    """capture_rgb=True (with track_cam) renders from a camera whose pose is
    updated EVERY step to match the stress plot's own tracking math exactly
    (same EMA-smoothed centroid, same elev/azim, same distance) -- computed
    causally (only past/current frames), so the RGB frames line up 1:1 in time
    with frames_pos/frames_stress from this SAME call. Called with the SAME
    (episode, dr row) pair as the search that found it; a single-env (num_envs=1)
    MPM replay of identical actions from an identical reset is deterministic in
    practice (the documented MPM non-determinism is specifically a multi-env
    parallel-atomics effect), so this reproduces the same rollout, not a
    different lift of the same category."""
    object_dxy = np.array([[dr["obj_dx"], dr["obj_dy"]]], dtype=np.float32)
    object_euler = np.deg2rad(np.array(
        [[dr["roll_deg"], dr["pitch_deg"], dr["yaw_deg"]]], dtype=np.float32))
    home_offset = np.array([[dr["home_dx"], dr["home_dy"], dr["home_dz"]]], dtype=np.float32)
    worker.reset(object_dxy=object_dxy, object_euler=object_euler, home_offset=home_offset)

    obs = ep["observations"]
    T = min(len(ep["actions"]), n_frames_cap)
    frames_pos, frames_stress, frames_rgb = [], [], []
    half_extent = None
    center_ema = None
    for t in range(T):
        cur_pos  = obs["ee_pos"][t][None].astype(np.float32)
        cur_quat = obs["ee_quat"][t][None].astype(np.float32)
        cur_grip = obs["gripper_width"][t].astype(np.float32)  # already (1,) — no extra batch dim
        worker.step(cur_pos, cur_quat, cur_grip)
        st = worker.handle.objects[0].get_state()
        pos = _np(st.pos)[0].copy()
        frames_pos.append(pos)                            # (n_p, 3)
        frames_stress.append(_np(st.von_mises)[0].copy())  # (n_p,)
        if capture_rgb and track_cam is not None:
            center = pos.mean(axis=0)
            if half_extent is None:
                # Object size doesn't change during a lift -- fine to fix from
                # frame 0 rather than needing the whole trajectory in advance.
                half_extent = max(np.percentile(np.linalg.norm(pos - center, axis=-1), 98) * 1.6,
                                  0.018)
            center_ema = center if center_ema is None else (
                TRACK_EMA_ALPHA * center + (1 - TRACK_EMA_ALPHA) * center_ema)
            cam_pos = center_ema + _spherical_offset(half_extent)
            track_cam.set_pose(pos=cam_pos, lookat=center_ema)
            frames_rgb.append(_np(track_cam.render(rgb=True, depth=False)[0]).astype(np.uint8))
    return frames_pos, frames_stress, frames_rgb


def _replay_and_capture(category: str, n_frames_cap: int = 260, max_tries: int = 40):
    """Search candidates WITHOUT rgb capture (cheaper), then re-replay the winner
    ONCE MORE with capture_rgb=True -- avoids paying the render cost on every
    rejected candidate. MPM replay is deterministic given the same actions/reset,
    so the confirmation replay reproduces the same frames_pos/frames_stress
    (only used for the printed peak-stress line, not re-validated).

    n_candidates is capped at len(dr_rows), NOT just len(episodes): dr_params.csv
    only logs a PREFIX of attempts (see _find_episodes_and_dr's docstring), so for
    episode index i >= len(dr_rows) the old `dr_rows[min(i, len(dr_rows)-1)]`
    clamp silently REUSED the last logged row -- pairing episode i's actions with
    a DIFFERENT episode's object pose/orientation. That's a guaranteed miss, not
    a fair trial. Capping at len(dr_rows) only ever tries genuinely-matched
    (episode, dr row) pairs (caught live 2026-08-20 investigating gelatin's 8/8
    rejected replays)."""
    episodes, dr_csv, run_dir = _find_episodes_and_dr(category)
    dr_rows = _read_dr_rows(dr_csv)
    n_candidates = min(len(episodes), len(dr_rows), max_tries)

    exp = Experiment.load(f"single_lift_{category}_soft_easy")
    task = SingleLiftTask(exp.task_cfg)
    spec = task.scene_spec
    # Extra camera purely for the tracking-shot RGB video -- pose is overwritten
    # every frame via set_pose(), so its spec-time pos/lookat are placeholders.
    from gentle_manip.scenes.scene_spec import CameraEntry
    spec.cameras = list(spec.cameras) + [
        CameraEntry(name="cam_track", pos=(0.5, -0.5, 0.5), lookat=(0.3, 0.0, 0.1),
                   fov=TRACK_FOV, resolution=(320, 240))]

    worker = GenesisWorker(spec, num_envs=1, show_viewer=False,
                           settle_steps=30, settle_max_steps=200,
                           settle_vel_thresh=0.002, render_obs_cameras=True)
    track_cam = worker.handle.cameras["cam_track"][0]
    try:
        # Two-tier fallback, both ranked separately by rise: `lifted` = the
        # object genuinely rose (rise >= min_rise_m) regardless of stress --
        # preferred over `fallback` (best-rise overall, may be a near-zero or
        # negative "lift") because a real lift that happens to exceed yield
        # stress (an over-squeeze) is a far more honest showcase clip than a
        # replay where the object never left the table. Plain best-rise is
        # the last resort only when NOTHING in the search ever lifted the
        # object at all (caught live 2026-08-20: mushroom's true successes
        # all replayed at 106-138% yield -- a firm/crushing grasp, not the
        # gentle grasp collected originally, but still a real lift, unlike
        # falling back on a candidate that never left the table).
        fallback = None
        lifted = None
        winner = None
        for i in range(n_candidates):
            ep = episodes[i]
            dr = dr_rows[i]
            print(f"[{category}] trying episode {i}/{n_candidates-1} from {run_dir.name}, "
                 f"{len(ep['actions'])} steps", flush=True)
            frames_pos, frames_stress, _ = _replay_one(worker, ep, dr, n_frames_cap)
            ok, rise, peak_stress = _replay_succeeded(
                frames_pos, task.object_yield_stress, frames_stress)
            print(f"[{category}]   episode {i}: rise={rise*100:.1f}cm "
                 f"peak_stress={peak_stress/task.object_yield_stress*100:.0f}%yield "
                 f"-> {'OK' if ok else 'reject'}", flush=True)
            if fallback is None or rise > fallback[2]:
                fallback = (frames_pos, frames_stress, rise, ep, dr)
            if rise >= 0.04 and (lifted is None or rise > lifted[2]):
                lifted = (frames_pos, frames_stress, rise, ep, dr)
            if ok:
                print(f"[{category}] episode {i} replayed as a successful lift -- using it",
                     flush=True)
                winner = (ep, dr)
                break
        if winner is None and lifted is not None:
            print(f"[{category}] WARNING: none of the {n_candidates} episodes replayed as a "
                 f"CLEAN (gentle) successful lift -- using the best real lift "
                 f"({lifted[2]*100:.1f}cm, may exceed yield stress) instead", flush=True)
            winner = (lifted[3], lifted[4])
        elif winner is None:
            print(f"[{category}] WARNING: none of the {n_candidates} episodes ever lifted the "
                 f"object at all -- using the best-rise candidate "
                 f"({fallback[2]*100:.1f}cm) instead", flush=True)
            winner = (fallback[3], fallback[4])

        ep, dr = winner
        frames_pos, frames_stress, frames_rgb = _replay_one(
            worker, ep, dr, n_frames_cap, capture_rgb=True, track_cam=track_cam)
        return frames_pos, frames_stress, frames_rgb, task.object_yield_stress
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


def _render_rgb_video(frames_rgb, out_path: Path, fps: int = 30):
    """Companion RGB video from a REAL Genesis tracking camera whose pose
    matches the stress plot's own view_init/EMA-centroid math frame-for-frame
    (see cam_track / _spherical_offset) -- so viewers see the actual robot
    grasp from the SAME viewpoint on the SAME rollout as the abstract
    stress-colored point cloud, not just some other camera's take on the same
    category. Fixed 2026-08-19: the first version used the scene's static
    cam_ext (a totally different, non-tracking framing) from a SEPARATE replay
    call, so neither the viewpoint nor the exact timing matched."""
    import imageio
    imageio.mimwrite(str(out_path), [f for f in frames_rgb], fps=fps, quality=8)
    print(f"  -> {out_path}  ({len(frames_rgb)} rgb frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["mushroom", "grape", "kiwi", "raspberry"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=260)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    for cat in args.categories:
        out_path = args.out_dir / f"{cat}_stress_closeup.mp4"
        rgb_path = args.out_dir / f"{cat}_rgb.mp4"
        if out_path.exists() and rgb_path.exists():
            print(f"[{cat}] already rendered -- skipping", flush=True)
            clip_paths.append(out_path)
            continue
        frames_pos, frames_stress, frames_rgb, yield_stress = _replay_and_capture(
            cat, args.max_frames)
        if not out_path.exists():
            _render_category_video(cat, frames_pos, frames_stress, yield_stress, out_path)
        if not rgb_path.exists():
            _render_rgb_video(frames_rgb, rgb_path)
        clip_paths.append(out_path)

    print("\nDone. Clips:")
    for p in clip_paths:
        print(" ", p)


if __name__ == "__main__":
    main()
