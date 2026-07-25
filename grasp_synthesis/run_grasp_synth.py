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
from scipy.spatial.transform import Rotation as Rot

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
    FINGER_GRIP_OFF,
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
N_APPROACH  = 100   # Cartesian interpolation steps: pre-grasp → grasp position
N_GRASP     = 60    # steps to hold and close gripper
N_LIFT      = 100   # steps to raise EE by LIFT_HEIGHT
N_HOLD      = 60    # steps to hold at lift height (success evaluation window)
LIFT_HEIGHT = 0.15  # meters above grasp position
PRE_GRASP_Z = 0.08  # meters above grasp position for the start of approach


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
        sim_substeps=10,
    )


# ── Synthesis ─────────────────────────────────────────────────────────────────

def synthesize(
    obj_pos: np.ndarray,
    sdf_fn,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    maxfevals: int,
) -> np.ndarray:
    """Run CMA-ES. Returns best x = [tx, ty, tz, roll, pitch, yaw, width]."""
    # Search region: ±1.5× bounding-box around object centroid
    t_lb = (obj_pos - 1.5 * OBJ_SIZE).tolist()
    t_ub = (obj_pos + 1.5 * OBJ_SIZE).tolist()
    # roll ≈ π (gripper pointing down), pitch/yaw small, width 28–88 mm
    lb = t_lb + [0.8 * np.pi, -0.25 * np.pi, -0.25 * np.pi, 0.028]
    ub = t_ub + [1.0 * np.pi,  0.25 * np.pi,  0.25 * np.pi, 0.088]
    x0 = [(l + u) / 2 for l, u in zip(lb, ub)]

    def objective(x):
        return grasp_cost(x, left_pts, right_pts, sdf_fn, obj_pos)

    print(f"  Running CMA-ES ({maxfevals} evals) …")
    t0 = time.time()
    best_x, best_score = run_cmaes(objective, x0, 1.0, lb, ub, maxfevals)
    print(f"  Done in {time.time() - t0:.1f}s  |  cost = {best_score:.4f}")
    print(f"  tcp_pos = {best_x[:3].round(4)}")
    print(f"  rpy_deg = {np.degrees(best_x[3:6]).round(2)}")
    print(f"  width   = {best_x[6]*1000:.1f} mm")
    return best_x


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


def execute(worker: GenesisWorker, best_x: np.ndarray, num_envs: int) -> bool:
    """Execute approach → grasp → lift. Returns True if any env succeeds."""
    pos_b, quat_b, width_grasp = _x_to_pose(best_x, num_envs)
    grasp_pos = pos_b[0].copy()

    width_open = np.full(num_envs, 0.08,         dtype=np.float32)
    width_cls  = np.full(num_envs, width_grasp,  dtype=np.float32)

    # Pre-grasp: directly above the synthesized grasp position
    pre_pos = grasp_pos.copy(); pre_pos[2] += PRE_GRASP_Z
    pre_b   = np.tile(pre_pos[None], (num_envs, 1))

    print(f"  [1/4] Teleporting to pre-grasp  z={pre_pos[2]:.3f} m …")
    worker.set_ee_pose(pre_b, quat_b, settle=60)

    print(f"  [2/4] Approaching in {N_APPROACH} steps …")
    for i in range(N_APPROACH):
        alpha = (i + 1) / N_APPROACH
        cur   = pre_b + alpha * (pos_b - pre_b)
        worker.step(cur, quat_b, width_open)

    print(f"  [3/4] Closing gripper ({width_grasp*1000:.0f} mm) in {N_GRASP} steps …")
    for _ in range(N_GRASP):
        worker.step(pos_b, quat_b, width_cls)

    print(f"  [4/4] Lifting {LIFT_HEIGHT*100:.0f} cm in {N_LIFT} steps …")
    lift_pos = grasp_pos.copy(); lift_pos[2] += LIFT_HEIGHT
    lift_b   = np.tile(lift_pos[None], (num_envs, 1))
    for i in range(N_LIFT):
        alpha = (i + 1) / N_LIFT
        cur   = pos_b + alpha * (lift_b - pos_b)
        worker.step(cur, quat_b, width_cls)

    state = None
    print(f"  Holding for {N_HOLD} steps …")
    for _ in range(N_HOLD):
        state = worker.step(lift_b, quat_b, width_cls)

    obj_z   = state['object_center'][:, 2]
    success = obj_z > (grasp_pos[2] + LIFT_HEIGHT * 0.5)
    print(f"  obj_z per env = {np.round(obj_z, 3)}")
    return bool(np.any(success))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-envs',    type=int, default=1,   help='Number of parallel envs')
    parser.add_argument('--maxfevals', type=int, default=800, help='CMA-ES function evaluations')
    parser.add_argument('--settle',    type=int, default=30,  help='Settle steps after reset')
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

    # ── 2. Reset ──────────────────────────────────────────────────────────────
    print("\n=== Resetting …")
    state   = worker.reset()
    obj_pos = state['object_center'][0].copy().astype(np.float64)   # env 0
    print(f"  Object centroid: {np.round(obj_pos, 4)}")

    # ── 3. Build SDF + sample finger geometry ────────────────────────────────
    print("\n=== Building mushroom SDF …")
    sdf_fn    = build_object_sdf(MUSHROOM_MESH)
    left_pts  = sample_finger_surface(LEFT_FINGER,  n=300)
    right_pts = sample_finger_surface(RIGHT_FINGER, n=300)

    # Sanity-check: SDF at the mesh origin (should be negative → inside)
    d0 = sdf_fn(np.zeros((1, 3)))[0]
    print(f"  SDF at mesh origin = {d0:.4f} ({'inside ✓' if d0 < 0 else 'outside — check mesh'})")

    # ── 4. Synthesize grasp ───────────────────────────────────────────────────
    print("\n=== Grasp synthesis (CMA-ES) …")
    best_x = synthesize(obj_pos, sdf_fn, left_pts, right_pts, args.maxfevals)

    # ── 5. Execute ────────────────────────────────────────────────────────────
    print("\n=== Executing grasp …")
    success = execute(worker, best_x, args.n_envs)
    print(f"\n=== Result: {'SUCCESS ✓' if success else 'FAILED ✗'}")

    # ── 6. Hold viewer open ───────────────────────────────────────────────────
    pos_b, quat_b, width = _x_to_pose(best_x, args.n_envs)
    lift_pos = pos_b.copy(); lift_pos[:, 2] += LIFT_HEIGHT
    width_b  = np.full(args.n_envs, width, dtype=np.float32)

    print("\nViewer is open — press Ctrl-C to exit.")
    try:
        while True:
            worker.step(lift_pos, quat_b, width_b)
    except KeyboardInterrupt:
        pass

    worker.close()


if __name__ == '__main__':
    main()
