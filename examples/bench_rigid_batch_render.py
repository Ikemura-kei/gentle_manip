"""Benchmark: batched-camera render + PerceptionPipeline FPS for rigid simulation.

Loads actual training configs (task + obs) so the FPS reflects real training conditions.
Measures:
  - Full pipeline FPS  : physics + depth render + PerceptionPipeline (point cloud)
  - Physics-only FPS   : no render, no pipeline (upper-bound)

Also generates validation videos saved to --output-dir:
  - point_cloud.mp4   — processed point cloud at each step (env 0, matplotlib 3D)
  - rgb_render.mp4    — raw RGB from the scene camera (physics stability check)

sim_dt explanation
------------------
sim_dt = the duration of one policy step (one call to scene.step()).
Within each step Genesis runs `substeps` physics sub-integrations of duration
  (sim_dt / substeps).
SingleLiftTask uses sim_dt = 1/30 ≈ 33 ms (30 Hz policy).  At substeps=20 the
physics integration dt is 33/20 = 1.67 ms — conservative relative to Genesis
official examples:
  franka_cube.py : dt=10 ms, substeps=1  → physics dt 10 ms
  grasp_env.py   : ctrl_dt=10 ms, substeps=2 → physics dt 5 ms

Usage:
    MUJOCO_GL=egl uv run --project envs/sim \\
        python examples/bench_rigid_batch_render.py
    MUJOCO_GL=egl uv run --project envs/sim \\
        python examples/bench_rigid_batch_render.py \\
          --task-config gentle_manip/configs/tasks/single_lift_mushroom_rigid.yaml \\
          --obs-config gentle_manip/configs/obs/point_cloud_1cam.yaml \\
          --n-envs 32 --steps 200 --video-steps 100
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

WARM = 20  # warm-up steps excluded from timing


# ─── config loading ────────────────────────────────────────────────────────────

def _load_spec(task_config_path: str):
    """Instantiate SingleLiftTask from a task YAML and return its scene_spec."""
    import yaml
    from gentle_manip.tasks.single_lift import SingleLiftTask

    with open(task_config_path) as f:
        task_cfg = yaml.safe_load(f)
    task = SingleLiftTask(task_cfg)
    return task.scene_spec


def _load_obs_config(obs_config_path: str):
    import yaml
    from gentle_manip.perception.obs_config import ObsConfig

    with open(obs_config_path) as f:
        raw = yaml.safe_load(f)
    return ObsConfig.from_dict(raw)


# ─── benchmark ────────────────────────────────────────────────────────────────

def _fixed_arm(num_envs: int):
    ee_pos  = np.tile(np.array([0.40, 0.0, 0.30], np.float32), (num_envs, 1))
    ee_quat = np.tile(np.array([0.0, 1.0, 0.0, 0.0], np.float32), (num_envs, 1))
    grip    = np.full(num_envs, 0.04, np.float32)  # (B,) scalar width per env
    return ee_pos, ee_quat, grip


def _state_to_raw_obs(state: dict):
    from gentle_manip.envs.raw_obs import RawObs
    return RawObs(
        ee_pos=state["ee_pos"],
        ee_quat=state["ee_quat"],
        gripper_width=state["gripper_width"],
        joint_pos=state["joint_pos"],
        joint_vel=state["joint_vel"],
        depth_images=state["depth_images"],
        rgb_images={},
        camera_intrinsics=state["camera_intrinsics"],
        camera_extrinsics=state["camera_extrinsics"],
        tactile_images={},
    )


def _run_bench(worker, num_envs: int, steps: int, obs_config, *, profile: bool = False):
    """Run `steps` steps (after warm-up) and return (fps_full, fps_phys)."""
    from gentle_manip.perception.pipeline import PerceptionPipeline
    from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud
    from gentle_manip.perception.pointcloud_ops import crop_pointcloud, subsample_pointcloud

    pipeline = PerceptionPipeline(obs_config)
    ee_pos, ee_quat, grip = _fixed_arm(num_envs)

    # Warm-up (excluded from timing)
    for _ in range(WARM):
        worker.step(ee_pos, ee_quat, grip)

    # Full pipeline: physics + camera render + PerceptionPipeline
    t0 = time.perf_counter()
    for _ in range(steps):
        state = worker.step(ee_pos, ee_quat, grip)
        raw = _state_to_raw_obs(state)
        pipeline.process(raw)
    t_full = time.perf_counter() - t0

    # Physics-only: skip depth render and PerceptionPipeline
    worker.render_obs_cameras = False
    t1 = time.perf_counter()
    for _ in range(steps):
        worker.step(ee_pos, ee_quat, grip)
    t_phys = time.perf_counter() - t1
    worker.render_obs_cameras = True

    fps_full = steps / t_full
    fps_phys = steps / t_phys
    overhead_ms = 1000.0 * (t_full - t_phys) / steps

    print(f"  Full pipeline (physics + render + PC):  "
          f"{fps_full:6.1f} batch-steps/s  ({fps_full * num_envs:.0f} env-steps/s)")
    print(f"  Physics only:                           "
          f"{fps_phys:6.1f} batch-steps/s  ({fps_phys * num_envs:.0f} env-steps/s)")
    print(f"  Render + PerceptionPipeline overhead:   {overhead_ms:.2f} ms/batch-step")

    if profile and obs_config.point_cloud is not None:
        print("\n  -- pipeline stage breakdown (avg over 20 steps) --")
        pc_cfg = obs_config.point_cloud
        pixel_sample_n = pc_cfg.pixel_sample_n

        # Get one state for profiling
        state = worker.step(ee_pos, ee_quat, grip)
        raw = _state_to_raw_obs(state)
        depth = raw.depth_images[pc_cfg.cameras[0]]
        K = raw.camera_intrinsics[pc_cfg.cameras[0]]
        E = raw.camera_extrinsics[pc_cfg.cameras[0]]

        N = 20
        t_bp = t_cr = t_ss = 0.0
        for _ in range(N):
            t = time.perf_counter()
            pts, valid = depth_to_pointcloud(depth, K, E, pixel_sample_n=pixel_sample_n)
            t_bp += time.perf_counter() - t

            t = time.perf_counter()
            pts, valid = crop_pointcloud(pts, valid, pc_cfg.crop_min, pc_cfg.crop_max)
            t_cr += time.perf_counter() - t

            t = time.perf_counter()
            subsample_pointcloud(pts, valid, pc_cfg.max_points)
            t_ss += time.perf_counter() - t

        n_valid = int(valid.sum() / num_envs)
        print(f"    backproject ({depth.shape[1]}×{depth.shape[2]}={depth.shape[1]*depth.shape[2]} px"
              f"{f' → {pixel_sample_n}' if pixel_sample_n else ''}): {t_bp/N*1000:.1f} ms")
        print(f"    crop (→ {n_valid} pts/env avg):  {t_cr/N*1000:.1f} ms")
        print(f"    subsample (→ {pc_cfg.max_points}):  {t_ss/N*1000:.1f} ms")

    return fps_full, fps_phys


# ─── video generation ─────────────────────────────────────────────────────────

def _pc_frame(pc: np.ndarray, step: int, total: int, crop_min, crop_max,
              ee_pos: np.ndarray | None = None) -> np.ndarray:
    """Render a point-cloud scatter frame (H, W, 3) uint8 via Agg backend.

    Zero-padded points (subsample_pointcloud fills unused slots with (0,0,0))
    are filtered before plotting — they sit at the world origin, outside the crop
    box, and appear as a spurious blob when matplotlib ignores axis clip limits.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Remove zero-padded slots: any point where ALL coords are exactly 0.0 is padding.
    # Real in-crop points always have x>=crop_min[0]>0 and z>=crop_min[2]>0.
    valid_mask = ~np.all(pc == 0.0, axis=-1)
    pc = pc[valid_mask]

    fig = plt.figure(figsize=(7, 5))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    if len(pc):
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.8, c=pc[:, 2],
                   cmap="plasma", vmin=crop_min[2], vmax=crop_max[2])
    if ee_pos is not None:
        ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]], s=40, c="red",
                   marker="x", zorder=5, label="EE")
    ax.set_xlim(crop_min[0], crop_max[0])
    ax.set_ylim(crop_min[1], crop_max[1])
    ax.set_zlim(crop_min[2], crop_max[2])
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(f"step {step}/{total - 1}  |  {len(pc)} pts", fontsize=9)
    ax.view_init(elev=28, azim=-60)
    fig.tight_layout()
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return img


def _arm_trajectory(step: int, total: int, num_envs: int):
    """Simple reach-down trajectory: arm moves from z=0.30 down to z=0.17 and back up.

    One full cycle over `total` steps so the arm visits a range of configurations
    during the video — makes it easy to spot which point clusters belong to the arm.
    """
    # Sinusoidal z: 0.30 at step 0, dips to 0.17 at mid, back to 0.30 at end
    t = step / max(total - 1, 1)                       # [0, 1]
    z = 0.30 - 0.13 * np.sin(np.pi * t)               # 0.30 → 0.17 → 0.30
    ee_pos  = np.tile(np.array([0.40, 0.0, z], np.float32), (num_envs, 1))
    ee_quat = np.tile(np.array([0.0, 1.0, 0.0, 0.0], np.float32), (num_envs, 1))
    grip    = np.full(num_envs, 0.04, np.float32)
    return ee_pos, ee_quat, grip


def _generate_videos(worker, num_envs: int, video_steps: int, obs_config, out_dir: Path,
                     policy_hz: float = 30.0):
    """Run video_steps steps with a reach-down arm trajectory, save MP4s."""
    import imageio
    from gentle_manip.perception.pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline(obs_config)

    pc_cfg = obs_config.point_cloud
    crop_min = list(pc_cfg.crop_min) if pc_cfg else [0.2, -0.215, 0.0]
    crop_max = list(pc_cfg.crop_max) if pc_cfg else [0.71, 0.215, 0.45]

    pc_frames: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []

    for step in range(video_steps):
        ee_pos, ee_quat, grip = _arm_trajectory(step, video_steps, num_envs)
        state = worker.step(ee_pos, ee_quat, grip)
        raw = _state_to_raw_obs(state)
        obs = pipeline.process(raw)

        # Point cloud (env 0) — pass EE pos so it's marked with a red X
        if "point_cloud" in obs:
            pc = obs["point_cloud"][0]  # (max_points, 3)
            pc_frames.append(_pc_frame(pc, step, video_steps, crop_min, crop_max,
                                       ee_pos=state["ee_pos"][0]))

        # RGB render (env 0)
        rgb = worker.render_rgb(all_envs=False)
        if rgb is not None:
            rgb_frames.append(rgb)

    out_dir.mkdir(parents=True, exist_ok=True)
    if pc_frames:
        p = out_dir / "point_cloud.mp4"
        imageio.mimsave(str(p), pc_frames, fps=policy_hz)
        print(f"  Point cloud video : {p}")
    if rgb_frames:
        p = out_dir / "rgb_render.mp4"
        imageio.mimsave(str(p), rgb_frames, fps=policy_hz)
        print(f"  RGB render video  : {p}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-config",
                    default="gentle_manip/configs/tasks/single_lift_mushroom_rigid.yaml",
                    help="Task YAML (must be a SingleLiftTask config)")
    ap.add_argument("--obs-config",
                    default="gentle_manip/configs/obs/point_cloud_1cam_fast.yaml",
                    help="Observation YAML (determines PerceptionPipeline)")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200, help="Timing steps (after warm-up)")
    ap.add_argument("--video-steps", type=int, default=100,
                    help="Steps for validation videos (0 = skip)")
    ap.add_argument("--output-dir", default="logs/bench",
                    help="Directory for validation videos")
    ap.add_argument("--profile", action="store_true",
                    help="Print per-stage pipeline timing breakdown")
    ap.add_argument("--compare-slow", action="store_true",
                    help="Also run without pixel_sample_n for comparison")
    ap.add_argument("--env-spacing", type=float, default=4.0,
                    help="Distance (m) between parallel envs; 4m keeps neighbours "
                         "outside the 1m depth_max filter with margin (default: 4.0)")
    args = ap.parse_args()

    spec       = _load_spec(args.task_config)
    obs_config = _load_obs_config(args.obs_config)
    out_dir    = Path(args.output_dir)

    substeps  = int(os.environ.get("GM_SIM_SUBSTEPS") or spec.sim_substeps)
    phys_dt_ms = spec.sim_dt / substeps * 1000.0

    print(f"\n=== rigid batch-render benchmark ===")
    print(f"  task config  : {args.task_config}")
    print(f"  obs config   : {args.obs_config}")
    print(f"  n_envs={args.n_envs}  steps={args.steps}  env_spacing={args.env_spacing}m")
    print(f"\n  sim_dt  = {spec.sim_dt:.6f} s  ({1.0 / spec.sim_dt:.1f} Hz policy rate)")
    print(f"  substeps = {substeps}  →  physics integration dt = {phys_dt_ms:.3f} ms")
    print(f"  [ref] Genesis franka_cube : dt=10 ms, substeps=1 → physics dt 10.000 ms")
    print(f"  [ref] Genesis grasp_env   : dt=10 ms, substeps=2 → physics dt  5.000 ms")
    print()

    from gentle_manip.envs.genesis_worker import GenesisWorker, _init_genesis

    _init_genesis()
    worker = GenesisWorker(spec, args.n_envs, show_viewer=False, show_fps=False,
                           env_spacing=args.env_spacing)
    assert worker.handle.batch_render_cameras, (
        "Expected batched camera rendering for a rigid scene but got per-env cameras. "
        "Check that the task config has no soft objects."
    )

    fps_full, fps_phys = _run_bench(worker, args.n_envs, args.steps, obs_config,
                                    profile=args.profile)

    if getattr(args, "compare_slow", False) and obs_config.point_cloud is not None \
            and obs_config.point_cloud.pixel_sample_n is not None:
        import copy
        slow_cfg = copy.deepcopy(obs_config)
        assert slow_cfg.point_cloud is not None
        slow_cfg.point_cloud.pixel_sample_n = None
        print(f"\n  -- comparison: pixel_sample_n=None (all {640*480} pixels) --")
        _run_bench(worker, args.n_envs, args.steps, slow_cfg)

    if args.video_steps > 0:
        print(f"\nGenerating validation videos ({args.video_steps} steps → {out_dir}/)")
        _generate_videos(worker, args.n_envs, args.video_steps, obs_config, out_dir,
                         policy_hz=1.0 / spec.sim_dt)

    worker.close()

    N = args.n_envs
    print(f"\n=== summary ({N} envs) ===")
    print(f"  Full pipeline:  {fps_full:.1f} batch-steps/s  ({fps_full * N:.0f} env-steps/s)")
    print(f"  Physics only:   {fps_phys:.1f} batch-steps/s  ({fps_phys * N:.0f} env-steps/s)")
    print(f"  Videos in: {out_dir}/")


if __name__ == "__main__":
    main()
