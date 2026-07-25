"""Phase 5 spike (the make-or-break gate): scripted grasp of the FEM mushroom with the XArm7.

Combines Phases 1-4: arm Articulation + DifferentialIKController (cartesian, Phase 2) + parallel-jaw
drive (all 6 joints equal, Phase 3) + the welded scanned mushroom as an FEM deformable (Phase 4).
Runs a scripted state machine — approach -> descend -> close -> lift -> hold — and reports the
mushroom lift height, gripper width, and per-element von Mises. THE test: does rigid-finger <->
FEM contact stay stable (no explosion), does the gripper hold the mushroom through the lift, and
does von Mises rise sensibly with grasp force?

GUI by default:
    ./isaaclab.sh -p /workspace/gm_isaac/grasp_mushroom.py
    ./isaaclab.sh -p /workspace/gm_isaac/grasp_mushroom.py --close 0.7 --grasp-z 0.19   # tune grasp
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 5: scripted grasp of the FEM mushroom.")
parser.add_argument("--usd", default="/workspace/gm_isaac/assets/xarm7.usd", help="arm USD (Phase 1)")
parser.add_argument("--obj", default="/workspace/gm_assets/objects/mushroom.obj", help="mushroom mesh")
parser.add_argument("--youngs", type=float, default=3.0e5)
parser.add_argument("--poisson", type=float, default=0.35)
parser.add_argument("--mx", type=float, default=0.45, help="mushroom x on the ground (base frame)")
parser.add_argument("--my", type=float, default=0.0, help="mushroom y")
parser.add_argument("--approach-z", type=float, default=0.32, help="EE(base) z above the mushroom")
parser.add_argument("--grasp-z", type=float, default=0.20, help="EE(base) z with fingers around it")
parser.add_argument("--lift-z", type=float, default=0.36, help="EE(base) z after lifting")
parser.add_argument("--close", type=float, default=0.65, help="gripper joint angle to close to (rad)")
parser.add_argument("--steps", type=int, default=100000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, DeformableObject, DeformableObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import bind_physics_material, create_prim, get_current_stage
from isaaclab.utils.math import subtract_frame_transforms

ARM_HOME = {"joint1": -0.4943, "joint2": -0.0623, "joint3": 0.4846, "joint4": 1.0172,
            "joint5": 0.0340, "joint6": 1.0765, "joint7": -0.0268}
GRIPPER_OPEN = {"drive_joint": 0.0, "left_finger_joint": 0.0, "left_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0, "right_finger_joint": 0.0,
                "right_inner_knuckle_joint": 0.0}
EE_LINK = "xarm_gripper_base_link"

# scripted state machine: (name, EE-z target, gripper angle, n_steps). xy target = (mx,my) throughout.
def phases(a):
    return [("approach", a.approach_z, 0.0,     150),   # above the mushroom, open
            ("descend",  a.grasp_z,    0.0,     150),   # lower fingers around it
            ("grasp",    a.grasp_z,    a.close, 150),   # close (ramped)
            ("lift",     a.lift_z,     a.close, 150),   # raise
            ("hold",     a.lift_z,     a.close, 250)]   # hold


def arm_cfg(usd):
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd, activate_contact_sensors=False),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0),
                                                   joint_pos={**ARM_HOME, **GRIPPER_OPEN}),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-7]"], effort_limit_sim=150.0,
                                       velocity_limit_sim=10.0, stiffness=8000.0, damping=600.0),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["drive_joint", ".*_finger_joint",
                                           ".*_knuckle_joint"], stiffness=1.0e5, damping=1000.0),
        })


def spawn_mushroom(obj_path, mx, my, youngs, poisson, remesh_pitch=0.0015, smooth_iters=10):
    """Load + WELD + voxel-REMESH the scan to a clean uniform mesh (Phase 4 finding: raw-scan tets
    give a diverging, inflated stress; a uniform remesh gives stable/bounded/physical stress), then
    build the deformable with the settle settings that behaved (solver_iters=40, vel_damping=5)."""
    mesh = trimesh.load(obj_path, process=True, force="mesh"); mesh.merge_vertices()
    if remesh_pitch > 0:
        vox = mesh.voxelized(pitch=remesh_pitch).fill()
        try:
            mesh = vox.marching_cubes                            # smooth (needs scikit-image)
            mesh.apply_transform(vox.transform)                  # index coords -> world (pitch+origin)!
        except Exception:
            mesh = vox.as_boxes()                                # blocky fallback, no extra deps
        mesh.merge_vertices()
        if smooth_iters > 0:
            trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)   # de-staircase (volume-preserving)
    v = np.asarray(mesh.vertices, dtype=np.float32); f = np.asarray(mesh.faces, dtype=np.int32)
    v[:, 0] += mx - v[:, 0].mean(); v[:, 1] += my - v[:, 1].mean()   # center at (mx,my)
    v[:, 2] -= v[:, 2].min()                                         # rest on ground (touching)
    stage = get_current_stage(); mp = "/World/Mushroom/geometry/mesh"
    create_prim("/World/Mushroom", "Xform", stage=stage)
    create_prim(mp, "Mesh", stage=stage, attributes={"points": v, "faceVertexIndices": f.flatten(),
                "faceVertexCounts": np.asarray([3] * len(f)), "subdivisionScheme": "bilinear"})
    schemas.define_deformable_body_properties(
        mp, sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001,
            simulation_hexahedral_resolution=10, solver_position_iteration_count=40,
            vertex_velocity_damping=5.0), stage=stage)
    m = sim_utils.DeformableBodyMaterialCfg(youngs_modulus=youngs, poissons_ratio=poisson)
    m.func("/World/Mushroom/material", m); bind_physics_material(mp, "/World/Mushroom/material", stage=stage)


def von_mises(s):
    sxx, syy, szz = s[..., 0, 0], s[..., 1, 1], s[..., 2, 2]
    sxy, syz, szx = s[..., 0, 1], s[..., 1, 2], s[..., 0, 2]
    return torch.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                      + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2) + 1e-12)


def main():
    a = args_cli
    sim = SimulationContext(sim_utils.SimulationCfg(device=a.device, dt=1.0 / 60.0))
    sim.set_camera_view(eye=[0.9, 0.7, 0.5], target=[a.mx, a.my, 0.1])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    robot = Articulation(arm_cfg(a.usd))
    spawn_mushroom(a.obj, a.mx, a.my, a.youngs, a.poisson)
    mush = DeformableObject(cfg=DeformableObjectCfg(prim_path="/World/Mushroom", spawn=None))
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=sim.device)
    sim.reset()
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()
    dt = sim.get_physics_dt()

    arm_ids = robot.find_joints(["joint[1-7]"])[0]
    grip_ids = robot.find_joints(["drive_joint", ".*_finger_joint", ".*_knuckle_joint"])[0]
    ee_id = robot.find_bodies(EE_LINK)[0][0]
    ee_jac = ee_id - 1 if robot.is_fixed_base else ee_id
    home = robot.data.default_joint_pos.clone()
    mush_z0 = float(mush.data.root_pos_w[0, 2])
    # fixed target orientation = the home (downward) EE orientation, in base frame
    ee_pos_b, ee_quat_b = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                    robot.data.body_pos_w[:, ee_id],
                                                    robot.data.body_quat_w[:, ee_id])
    down_quat = ee_quat_b.clone()
    print(f"[phase5] mushroom tets={mush.data.sim_element_stress_w.shape[1]} z0={mush_z0:.4f} | "
          f"EE home {ee_pos_b[0].cpu().numpy().round(3)}", flush=True)

    plan = phases(a)
    bounds = [(name, z, g, n) for name, z, g, n in plan]
    steps, phase_i, phase_step = 0, 0, 0
    while simulation_app.is_running():
        name, ee_z, grip_goal, n = bounds[min(phase_i, len(bounds) - 1)]
        # gripper: ramp toward the phase's goal angle
        cur_g = float(robot.data.joint_pos[0, grip_ids].mean())
        g_cmd = cur_g + np.clip(grip_goal - cur_g, -0.01, 0.01)

        jac = robot.root_physx_view.get_jacobians()[:, ee_jac, :, arm_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                        robot.data.body_pos_w[:, ee_id],
                                                        robot.data.body_quat_w[:, ee_id])
        tgt = torch.tensor([[a.mx, a.my, ee_z]], device=sim.device)
        ik.set_command(torch.cat([tgt, down_quat], dim=-1))
        arm_des = ik.compute(ee_pos_b, ee_quat_b, jac, robot.data.joint_pos[:, arm_ids])
        robot.set_joint_position_target(arm_des, joint_ids=arm_ids)
        grip_vec = torch.full((1, len(grip_ids)), g_cmd, device=sim.device)
        robot.set_joint_position_target(grip_vec, joint_ids=grip_ids)

        robot.write_data_to_sim(); mush.write_data_to_sim()
        sim.step(render=True)
        robot.update(dt); mush.update(dt)

        if steps % 30 == 0:
            vm = von_mises(mush.data.sim_element_stress_w)
            mz = float(mush.data.root_pos_w[0, 2])
            bad = " NaN/Inf!" if not torch.isfinite(mush.data.nodal_pos_w).all() else ""
            print(f"[phase5] {name:8s} step {steps:5d}  EE_z {float(ee_pos_b[0,2]):.3f}  grip {cur_g:.3f}  "
                  f"mush_lift {mz - mush_z0:+.4f}  vM mean {float(vm.mean()):7.0f} p95 "
                  f"{float(torch.quantile(vm, 0.95)):8.0f} Pa{bad}", flush=True)
        steps += 1; phase_step += 1
        if phase_step >= n and phase_i < len(bounds) - 1:
            phase_i += 1; phase_step = 0
        if a.headless and steps >= a.steps:
            break
    print("[phase5] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
