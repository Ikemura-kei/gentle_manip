"""Bare-minimum Genesis v1.1.x deformable-grasp sim — XArm7 grasping a soft box.

A standalone dev prototype toward the framework's sim backend (Steps 9-11, 15b).
Adapts third_party/genesis/examples/coupling/grasp_soft_cube.py to our XArm7 +
xarm7_config, with an MPM soft body. Run in the sim env:

    uv run --project envs/sim python examples/gs_sim_backend_dev.py          # headless
    uv run --project envs/sim python examples/gs_sim_backend_dev.py --vis    # viewer
"""
import os



import argparse
from pathlib import Path

import numpy as np
import genesis as gs

from gentle_manip.robot import xarm7_config as cfg

_REPO = Path(__file__).resolve().parents[1]
URDF = _REPO / "gentle_manip" / "assets" / "xarm" / "xarm7_with_gripper.urdf"

# Object resting on the ground, within reach + the MPM bounds below.
OBJ_POS = (0.50, 0.0, 0.03)
OBJ_SIZE = (0.05, 0.05, 0.05)
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # gripper pointing down (per examples)
PRE_Z, GRASP_Z = 0.33, 0.19                  # EE heights (xarm_gripper_base_link)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", action="store_true")
    args = ap.parse_args()
    
    # Genesis imports mujoco at import time; a stray/invalid MUJOCO_GL (e.g. a broken
    # shell value) makes that fail. Force a valid backend before importing genesis.
    if not args.vis and os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
        os.environ["MUJOCO_GL"] = "egl"

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1./30., substeps=75),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.20, -0.15, -0.012), upper_bound=(0.75, 0.15, 0.40),
            grid_density=280,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True, enable_collision=True, enable_self_collision=True,
            gravity=(0.0, 0.0, -9.81), box_box_detection=True, constraint_timeconst=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, -1.2, 1.4), camera_lookat=(0.45, 0.0, 0.15), camera_fov=35,
        ),
        vis_options=gs.options.VisOptions(visualize_mpm_boundary=True),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file=str(URDF), fixed=True, merge_fixed_links=True,
                       links_to_keep=cfg.LINKS_TO_KEEP, pos=(0.0, 0.0, 0.0)),
        material=gs.materials.Rigid(coup_friction=3.0),
    )
    obj = scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic(E=4e3, nu=0.3, von_mises_yield_stress=2e4),
        morph=gs.morphs.Box(size=OBJ_SIZE, pos=OBJ_POS, euler=(0, 0, 0)),
        surface=gs.surfaces.Default(vis_mode="particle"),
    )

    scene.build()

    # ── robot setup ───────────────────────────────────────────────────────────
    dof_idx = [robot.get_joint(n).dof_idx_local for n in cfg.JOINT_NAMES]
    arm_dofs, grip_dofs = dof_idx[:7], dof_idx[7:]
    robot.set_dofs_kp(np.array(cfg.KP, dtype=np.float32), dof_idx)
    robot.set_dofs_kv(np.array(cfg.KV, dtype=np.float32), dof_idx)

    ee = robot.get_link(cfg.EE_LINK)
    robot.set_dofs_position(np.array(cfg.DEFAULT_JOINT_ANGLES, dtype=np.float32), dof_idx)

    grip_open = np.array(cfg.DEFAULT_JOINT_ANGLES[7:], dtype=np.float32)
    grip_closed = np.full(len(grip_dofs), 0.60, dtype=np.float32)   # ramp target; tune

    def ik(z):
        return robot.inverse_kinematics(
            link=ee, pos=np.array([OBJ_POS[0], OBJ_POS[1], z]), quat=DOWN_QUAT)

    def hold_arm(z):
        robot.control_dofs_position(ik(z)[:7], arm_dofs)

    def _np(x):  # genesis returns CUDA torch tensors
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def report(tag, i):
        st = obj.get_state()
        pos = _np(st.pos).reshape(-1, 3)
        vm = _np(st.von_mises).reshape(-1)
        ee_z = float(_np(ee.get_pos()).reshape(-1)[2])
        print(f"  [{tag} {i}] obj_z={pos[:, 2].mean():.3f} ee_z={ee_z:.3f} "
              f"maxVonMises={float(vm.max()):.1f}")

    # ── grasp sequence ────────────────────────────────────────────────────────
    print("pre-grasp")
    for _ in range(120):
        hold_arm(PRE_Z)
        robot.control_dofs_position(grip_open, grip_dofs)
        scene.step()

    print("descend")
    for _ in range(120):
        hold_arm(GRASP_Z)
        robot.control_dofs_position(grip_open, grip_dofs)
        scene.step()

    print("close gripper")
    for i in range(120):
        hold_arm(GRASP_Z)
        alpha = min(1.0, i / 80.0)
        robot.control_dofs_position(grip_open * (1 - alpha) + grip_closed * alpha, grip_dofs)
        scene.step()
        if i % 40 == 0:
            report("close", i)

    print("lift")
    for i in range(300):
        hold_arm(GRASP_Z + 0.0006 * i)
        robot.control_dofs_position(grip_closed, grip_dofs)
        scene.step()
        if i % 50 == 0:
            report("lift", i)

    print("done")
    gs.destroy()


if __name__ == "__main__":
    main()
