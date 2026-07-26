"""Rigid-body grasp synthesis — Genesis viewer test.

Port of the old CMA-ES grasp synthesis to the current rigid-body sim.

What this does:
  1. Spawns a Genesis scene: mushroom (rigid) + table, 1 env, viewer ON.
  2. Resets and reads the object centroid (mushroom resting on table).
  3. Builds a signed-distance function for the mushroom mesh.
  4. Samples surface points from the left and right finger meshes.
  5. Runs CMA-ES (800 evals) to find the best 7-DOF grasp pose.
  6. Executes: teleport to pre-grasp → Cartesian approach → close gripper → lift.
  7. Holds at lift height and reports success. Viewer stays open (Ctrl-C to quit).

Soft-body evaluation and stress metrics are intentionally omitted — this is
a "did the grasp synthesis find a valid rigid-body grasp?" sanity check.

Usage:
    MUJOCO_GL=glfw uv run --project envs/sim python grasp_synthesis/run_grasp_synth.py
    MUJOCO_GL=glfw uv run --project envs/sim python grasp_synthesis/run_grasp_synth.py --maxfevals 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

# ── repo root → sys.path so gentle_manip + synth_utils are importable ────────
ROOT = Path(__file__).resolve().parent.parent
GRASP_DIR = ROOT / "grasp_synthesis"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GRASP_DIR) not in sys.path:
    sys.path.insert(0, str(GRASP_DIR))

os.environ.setdefault("MUJOCO_GL", "glfw")

from synth_utils import (              # noqa: E402  (after sys.path setup)
    build_object_sdf,
    grasp_cost,
    run_cmaes,
    sample_finger_surface,
    finger_world_pts,
    FINGER_TO_TCP_Z,
)

from gentle_manip.scenes.scene_spec import (  # noqa: E402
    CameraEntry, FixtureEntry, ObjectEntry, SceneSpec,
)
from gentle_manip.envs.genesis_worker import GenesisWorker  # noqa: E402

# ── Asset paths ───────────────────────────────────────────────────────────────
MUSHROOM_MESH = str(ROOT / "gentle_manip/assets/objects/mushroom.obj")
LEFT_FINGER   = str(ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/left_finger.STL")
RIGHT_FINGER  = str(ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/right_finger.STL")

# ── Bounding-box hint for CMA-ES search region ────────────────────────────────
OBJ_SIZE = np.array([0.05, 0.05, 0.04])   # rough mushroom AABB half-size (m)

# ── Execution timing ─────────────────────────────────────────────────────────
N_HOME_TO_PRE = 200  # Cartesian interpolation steps: home → grasp
N_GRASP     = 60    # steps to hold and close gripper
N_LIFT      = 100   # steps to raise EE by LIFT_HEIGHT
N_HOLD      = 60    # steps to hold at lift height (success evaluation window)
LIFT_HEIGHT = 0.15  # meters above grasp position
RECORD_EVERY = 1    # render one frame every N sim steps; sim is 30 Hz so 1→30fps 1:1


# ── Scene ─────────────────────────────────────────────────────────────────────

def _build_spec() -> SceneSpec:
    return SceneSpec(
        objects=[ObjectEntry(name="mushroom", object_type="rigid")],
        fixtures=[FixtureEntry(fixture_type="table")],
        cameras=[CameraEntry(
            name="cam_ext",
            pos=(0.9, 0.35, 0.55),
            lookat=(0.4, 0.0, 0.1),
            fov=50.0,
            resolution=(640, 480),
        )],
        sim_dt=1/30,      # 30 Hz: each scene.step() = 33.3 ms sim time
        sim_substeps=10,
    )


# ── Synthesis ─────────────────────────────────────────────────────────────────

def synthesize(
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
    sdf_fn,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    maxfevals: int,
) -> np.ndarray:
    """Run CMA-ES. Returns best x = [tx, ty, tz, roll, pitch, yaw, width]."""
    # XY: ±1.5× bounding-box around object centroid.
    # Z: keep TCP between FINGER_TO_TCP_Z above the object (so the finger tip reaches
    # the object) and 0.25 m above it. This avoids IK-failing poses where the TCP is
    # too low (arm over-extended) or too high (finger never reaches).
    t_lb_xy = (obj_pos[:2] - 1.5 * OBJ_SIZE[:2]).tolist()
    t_ub_xy = (obj_pos[:2] + 1.5 * OBJ_SIZE[:2]).tolist()
    tcp_z_min = float(obj_pos[2]) + FINGER_TO_TCP_Z - 0.04   # finger tip ~ object centroid
    tcp_z_max = float(obj_pos[2]) + 0.25                      # arm stays comfortably reachable
    # roll ≈ π (gripper pointing down), pitch/yaw small, width 28–88 mm
    lb = t_lb_xy + [tcp_z_min, 0.8 * np.pi, -0.25 * np.pi, -0.25 * np.pi, 0.01]
    ub = t_ub_xy + [tcp_z_max, 1.0 * np.pi,  0.25 * np.pi,  0.25 * np.pi, 0.08]
    x0 = [(l + u) / 2 for l, u in zip(lb, ub)]

    def objective(x):
        return grasp_cost(x, left_pts, right_pts, sdf_fn, obj_pos, obj_quat_wxyz)

    print(f"  Running CMA-ES ({maxfevals} evals) …")
    t0 = time.time()
    best_x, best_score = run_cmaes(objective, x0, 1.0, lb, ub, maxfevals)
    print(f"  Done in {time.time() - t0:.1f}s  |  cost = {best_score:.4f}")
    print(f"  tcp_pos = {best_x[:3].round(4)}")
    print(f"  rpy_deg = {np.degrees(best_x[3:6]).round(2)}")
    print(f"  width   = {best_x[6]*1000:.1f} mm")
    return best_x


# ── Visualization ─────────────────────────────────────────────────────────────

def _make_grasp_figure(
    best_x: np.ndarray,
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
    left_pts_world: np.ndarray,
    right_pts_world: np.ndarray,
    mesh_path: str = MUSHROOM_MESH,
    figsize: tuple = (8, 8),
    env_idx: int | None = None,
):
    """Build a matplotlib 3D figure of the grasp solution. Returns fig.
    Does not call plt.show() — caller decides whether to display or render.
    """
    import matplotlib.pyplot as plt
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tcp_pos = np.asarray(best_x[:3], np.float64)
    w = float(best_x[6])
    rpy_deg = np.degrees(best_x[3:6])

    q = np.asarray(obj_quat_wxyz, np.float64)
    obj_rot = Rot.from_quat([q[1], q[2], q[3], q[0]])
    mush_mesh = trimesh.load(mesh_path, force='mesh')
    obj_surface_local, _ = trimesh.sample.sample_surface(mush_mesh, 300, seed=0)
    obj_surface_pts = obj_rot.apply(obj_surface_local.astype(np.float64)) + obj_pos

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(*left_pts_world.T,  c='royalblue', s=20, alpha=0.8, label='L sample pts')
    ax.scatter(*right_pts_world.T, c='firebrick', s=20, alpha=0.8, label='R sample pts')
    ax.scatter(*obj_surface_pts.T, c='tan',       s=10, alpha=0.6, label='object surface')
    ax.scatter(*tcp_pos,           c='gold',  s=150, zorder=7, marker='*', label='TCP')
    ax.scatter(*obj_pos,           c='black', s=120, zorder=7, marker='*', label='object centroid')

    all_pts = np.vstack([left_pts_world, right_pts_world, obj_surface_pts,
                         tcp_pos[None], obj_pos[None]])
    ctr = (all_pts.min(0) + all_pts.max(0)) / 2
    rng = (all_pts.max(0) - all_pts.min(0)).max() / 2 + 0.02
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_box_aspect([1, 1, 1])

    gx = np.array([ctr[0] - rng, ctr[0] + rng, ctr[0] + rng, ctr[0] - rng])
    gy = np.array([ctr[1] - rng, ctr[1] - rng, ctr[1] + rng, ctr[1] + rng])
    gz = np.zeros(4)
    ground = Poly3DCollection([list(zip(gx, gy, gz))], alpha=0.12)
    ground.set_facecolor('gray'); ground.set_edgecolor('darkgray')
    ax.add_collection3d(ground)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    env_str = f'Env {env_idx} — ' if env_idx is not None else ''
    ax.set_title(
        f'{env_str}Grasp sample pts\n'
        f'TCP=[{best_x[0]:.3f}, {best_x[1]:.3f}, {best_x[2]:.3f}]  '
        f'rpy=[{rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}]°  '
        f'w={w*1e3:.0f} mm',
        fontsize=9,
    )
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    return fig


def visualize_grasp(best_x: np.ndarray, obj_pos: np.ndarray,
                    obj_quat_wxyz: np.ndarray,
                    left_pts_world: np.ndarray, right_pts_world: np.ndarray,
                    mesh_path: str = MUSHROOM_MESH) -> None:
    """Interactive 3D scatter of the grasp solution. Blocks until window is closed."""
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass
    import matplotlib.pyplot as plt

    fig = _make_grasp_figure(best_x, obj_pos, obj_quat_wxyz,
                             left_pts_world, right_pts_world, mesh_path)
    out_png = GRASP_DIR / 'grasp_pose_vis.png'
    fig.savefig(str(out_png), dpi=150)
    print(f"  Figure saved → {out_png.relative_to(ROOT)}")
    print("  Close the window to proceed to Genesis execution …")
    plt.show()


def render_grasp_vis_image(
    best_x: np.ndarray,
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
    left_pts_world: np.ndarray,
    right_pts_world: np.ndarray,
    mesh_path: str = MUSHROOM_MESH,
    height: int = 480,
    env_idx: int | None = None,
) -> np.ndarray:
    """Render the grasp vis to a (height, height, 3) uint8 numpy array. Non-blocking.

    Uses the Agg backend so it works headlessly even after TkAgg was set (via
    plt.switch_backend), and does not call plt.show().
    """
    import io
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')   # safe even if TkAgg was active

    dpi = 100
    fig = _make_grasp_figure(
        best_x, obj_pos, obj_quat_wxyz, left_pts_world, right_pts_world,
        mesh_path, figsize=(height / dpi, height / dpi), env_idx=env_idx,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    import imageio
    img = imageio.imread(buf)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]   # drop alpha
    img = img.astype(np.uint8)

    # Ensure exact (height, height) size — matplotlib sometimes produces ±1px.
    h, w = img.shape[:2]
    if h != height or w != height:
        # Crop to max(height, height) then pad if smaller.
        img = img[:min(h, height), :min(w, height)]
        pad_h = max(0, height - img.shape[0])
        pad_w = max(0, height - img.shape[1])
        if pad_h or pad_w:
            img = np.pad(img, [(0, pad_h), (0, pad_w), (0, 0)])
    return img


# ── Execution ─────────────────────────────────────────────────────────────────

def _x_to_pose(x: np.ndarray, num_envs: int):
    """Convert 7-DOF x to (pos_b, quat_wxyz_b, width) batched for num_envs."""
    pos = np.asarray(x[:3], dtype=np.float32)
    q_xyzw = Rot.from_euler('xyz', x[3:6]).as_quat()        # xyzw
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    width = float(x[6])

    pos_b  = np.tile(pos[None],      (num_envs, 1))
    quat_b = np.tile(quat_wxyz[None], (num_envs, 1))
    return pos_b, quat_b, width


def execute(worker: GenesisWorker, all_best_x: list, num_envs: int,
            out_dir: Path | None = None,
            vis_images: list | None = None) -> bool:
    """Execute home → grasp → close gripper → lift with per-env grasp poses.

    out_dir:    if given, records per-env mp4 to out_dir/env_N_{success|fail}.mp4.
    vis_images: optional list of num_envs (H, H, 3) uint8 arrays (grasp vis plots).
                When provided, each frame is composed as [sim_frame | vis_image]
                side-by-side before writing to the video.
    Returns True if any env succeeds.
    """
    # Build per-env batched arrays from individual solutions
    poses = [_x_to_pose(x, 1) for x in all_best_x]
    pos_b  = np.concatenate([p[0] for p in poses], axis=0).astype(np.float32)  # (N, 3)
    quat_b = np.concatenate([p[1] for p in poses], axis=0).astype(np.float32)  # (N, 4)
    grasp_pos = pos_b.copy()   # (N, 3) — each env's final grasp position

    width_open = np.full(num_envs, 0.08, dtype=np.float32)
    width_cls  = np.array([p[2] - 0.001 for p in poses], dtype=np.float32)     # (N,)

    # Home pose — shared across all envs
    home_pos  = np.tile(worker.robot.home_pos[None].astype(np.float32),  (num_envs, 1))
    home_quat = np.tile(worker.robot.home_quat[None].astype(np.float32), (num_envs, 1))

    # Per-env SLERP: home orientation → each env's grasp orientation
    def _wxyz_to_rot(q_wxyz):
        return Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

    home_r = _wxyz_to_rot(home_quat[0])
    slerps = [Slerp([0.0, 1.0], Rot.concatenate([home_r, _wxyz_to_rot(quat_b[i])]))
              for i in range(num_envs)]

    def _interp_quat(alpha: float) -> np.ndarray:
        rows = []
        for s in slerps:
            xyzw = s(alpha).as_quat()
            rows.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        return np.array(rows, dtype=np.float32)

    # frame buffer: list-of-lists [env_idx] -> list of (H, W_total, 3) uint8
    recording = out_dir is not None
    frame_bufs: list[list] = [[] for _ in range(num_envs)]
    _step_count = [0]

    def _sim_step(pos, quat, width):
        state = worker.step(pos, quat, width)
        if recording:
            _step_count[0] += 1
            if _step_count[0] % RECORD_EVERY == 0:
                frames = worker.render_rgb(all_envs=True)   # (N, H, W, 3)
                if frames is not None:
                    for ei in range(num_envs):
                        frame = frames[ei]  # (H, W, 3)
                        if vis_images is not None and vis_images[ei] is not None:
                            vis = vis_images[ei]  # (H_vis, H_vis, 3)
                            # Crop or zero-pad to match sim frame height (usually a no-op).
                            fh = frame.shape[0]
                            if vis.shape[0] > fh:
                                vis = vis[:fh]
                            elif vis.shape[0] < fh:
                                pad = np.zeros((fh - vis.shape[0], vis.shape[1], 3), np.uint8)
                                vis = np.concatenate([vis, pad], axis=0)
                            frame = np.concatenate([frame, vis], axis=1)
                        frame_bufs[ei].append(frame)
        return state

    print(f"  [1/3] Moving home → grasp in {N_HOME_TO_PRE} steps …")
    for i in range(N_HOME_TO_PRE):
        alpha = (i + 1) / N_HOME_TO_PRE
        cur   = home_pos + alpha * (pos_b - home_pos)
        _sim_step(cur, _interp_quat(alpha), width_open)

    for _ in range(20):
        _sim_step(pos_b, quat_b, width_open)

    widths_str = " ".join(f"{w*1000:.0f}" for w in width_cls + 0.001)
    print(f"  [2/3] Closing gripper [{widths_str}] mm in {N_GRASP} steps …")
    for _ in range(N_GRASP):
        _sim_step(pos_b, quat_b, width_cls)

    print(f"  [3/3] Lifting {LIFT_HEIGHT*100:.0f} cm in {N_LIFT} steps …")
    lift_b = grasp_pos.copy(); lift_b[:, 2] += LIFT_HEIGHT
    for i in range(N_LIFT):
        alpha = (i + 1) / N_LIFT
        cur   = pos_b + alpha * (lift_b - pos_b)
        _sim_step(cur, quat_b, width_cls)

    state = None
    print(f"  Holding for {N_HOLD} steps …")
    for _ in range(N_HOLD):
        state = _sim_step(lift_b, quat_b, width_cls)

    obj_z   = state['object_center'][:, 2]
    success = obj_z > (grasp_pos[:, 2] + LIFT_HEIGHT * 0.5)
    print(f"  obj_z per env  = {np.round(obj_z, 3)}")
    print(f"  success per env= {success.tolist()}")

    if recording and frame_bufs[0]:
        import imageio
        out_dir.mkdir(parents=True, exist_ok=True)
        # sim step = dt * substeps = 4e-3 * 10 = 40 ms → 25 fps
        for ei, frames in enumerate(frame_bufs):
            tag = 'success' if success[ei] else 'fail'
            path = out_dir / f"env{ei:02d}_{tag}.mp4"
            imageio.mimwrite(str(path), frames, fps=30, quality=8)
            print(f"  Saved {path.relative_to(Path(__file__).resolve().parent.parent)}")

    return bool(np.any(success))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-envs',    type=int,   default=5,    help='Number of parallel envs')
    parser.add_argument('--maxfevals', type=int,   default=1100,  help='CMA-ES function evaluations')
    parser.add_argument('--settle',    type=int,   default=30,   help='Settle steps after reset')
    parser.add_argument('--seed',      type=int,   default=0,    help='RNG seed for pose randomization')
    parser.add_argument('--xy-range',  type=float, default=0.04, help='Per-env xy jitter range (m)')
    parser.add_argument('--pitch-roll-range', type=float, default=45.0,
                        help='Per-env pitch/roll range (degrees); yaw is always full 360°')
    args = parser.parse_args()

    # ── 1. Build scene ────────────────────────────────────────────────────────
    print("\n=== Building Genesis scene (rigid mushroom + table, viewer ON) …")
    spec   = _build_spec()
    worker = GenesisWorker(
        spec,
        num_envs=args.n_envs,
        show_viewer=True,
        settle_steps=args.settle,
        render_obs_cameras=False,   # skip depth rendering — only need state
        env_spacing=2.5,
    )

    # ── 2. Reset + settle until still ────────────────────────────────────────
    print("\n=== Resetting with per-env pose randomization …")
    rng = np.random.default_rng(args.seed)
    n = args.n_envs
    pr = np.radians(args.pitch_roll_range)
    object_dxy   = rng.uniform(-args.xy_range, args.xy_range, size=(n, 2)).astype(np.float32)
    object_euler = np.stack([
        rng.uniform(-pr,      pr,      n),   # roll
        rng.uniform(-pr,      pr,      n),   # pitch
        rng.uniform(-np.pi,   np.pi,   n),   # yaw — full 360°
    ], axis=1).astype(np.float32)
    for i in range(n):
        print(f"  Env {i}: dxy={np.round(object_dxy[i],3)}  "
              f"rpy_deg={np.round(np.degrees(object_euler[i]),1)}")
    worker.reset(object_dxy=object_dxy, object_euler=object_euler)
    print("  Waiting for object to settle …")
    obj      = worker.handle.objects[0]
    def _t(tensor):
        t = tensor
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    for step in range(600):
        worker.handle.scene.step()
        lin = np.abs(_t(obj.get_vel())).max()
        ang = np.abs(_t(obj.get_ang())).max()
        if lin < 0.003 and ang < 0.01:
            print(f"    Still after {step+1} steps  |v|={lin:.4f} |ω|={ang:.4f}")
            break
    else:
        print(f"    Settle timeout (600 steps)")

    obj_pos_all  = _t(obj.get_pos()).astype(np.float64)   # (num_envs, 3)
    obj_quat_all = _t(obj.get_quat()).astype(np.float64)  # (num_envs, 4) wxyz
    for i in range(args.n_envs):
        print(f"  Env {i}  pos={np.round(obj_pos_all[i], 4)}  quat={np.round(obj_quat_all[i], 4)}")

    # ── 3. Build SDF + sample finger geometry ────────────────────────────────
    print("\n=== Building mushroom SDF …")
    actual_mesh = worker.handle.spec.objects[0].mesh_path or MUSHROOM_MESH
    print(f"  Mesh: {Path(actual_mesh).name}")
    sdf_fn    = build_object_sdf(actual_mesh)
    left_pts  = sample_finger_surface(LEFT_FINGER,  n=300)
    right_pts = sample_finger_surface(RIGHT_FINGER, n=300)

    d0 = sdf_fn(np.zeros((1, 3)))[0]
    print(f"  SDF at mesh origin = {d0:.4f} ({'inside ✓' if d0 < 0 else 'outside — check mesh'})")

    # ── 4. Per-env grasp synthesis ────────────────────────────────────────────
    all_best_x = []
    for i in range(args.n_envs):
        print(f"\n=== Grasp synthesis env {i} (CMA-ES) …")
        bx = synthesize(obj_pos_all[i], obj_quat_all[i], sdf_fn, left_pts, right_pts, args.maxfevals)
        all_best_x.append(bx)

    # ── 4b. Visualize ────────────────────────────────────────────────────────
    # Interactive viewer for single-env runs (blocks until window is closed).
    if args.n_envs == 1:
        lw0, rw0 = finger_world_pts(all_best_x[0], left_pts, right_pts)
        print("\n=== Visualizing grasp pose (close window to continue) …")
        visualize_grasp(all_best_x[0], obj_pos_all[0], obj_quat_all[0],
                        lw0, rw0, mesh_path=actual_mesh)

    # Render per-env static vis images for the side-by-side video overlay.
    print("\n=== Rendering per-env grasp vis images for video overlay …")
    vis_images = []
    for i in range(args.n_envs):
        lw, rw = finger_world_pts(all_best_x[i], left_pts, right_pts)
        img = render_grasp_vis_image(
            all_best_x[i], obj_pos_all[i], obj_quat_all[i],
            lw, rw, mesh_path=actual_mesh, height=480, env_idx=i,
        )
        vis_images.append(img)
        print(f"  Env {i}: vis image {img.shape}")

    # ── 5. Execute ────────────────────────────────────────────────────────────
    import datetime
    out_dir = Path("/home/kei/kei/gentle_manip/logs/grasp_synth") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n=== Executing grasp (recording → {out_dir.relative_to(Path('/home/kei/kei/gentle_manip'))}) …")
    success = execute(worker, all_best_x, args.n_envs, out_dir=out_dir, vis_images=vis_images)
    print(f"\n=== Result: {'SUCCESS ✓' if success else 'FAILED ✗'}")

    # ── 6. Hold viewer open ───────────────────────────────────────────────────
    poses   = [_x_to_pose(x, 1) for x in all_best_x]
    pos_b   = np.concatenate([p[0] for p in poses], axis=0).astype(np.float32)
    quat_b  = np.concatenate([p[1] for p in poses], axis=0).astype(np.float32)
    lift_pos = pos_b.copy(); lift_pos[:, 2] += LIFT_HEIGHT
    width_b  = np.array([p[2] for p in poses], dtype=np.float32)

    print("\nViewer is open — press Ctrl-C to exit.")
    try:
        while True:
            worker.step(lift_pos, quat_b, width_b)
    except KeyboardInterrupt:
        pass

    worker.close()


if __name__ == '__main__':
    main()
