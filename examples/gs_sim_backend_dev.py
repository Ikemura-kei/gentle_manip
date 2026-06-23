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
from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud

_REPO = Path(__file__).resolve().parents[1]
URDF = _REPO / "gentle_manip" / "assets" / "xarm" / "xarm7_with_gripper.urdf"

# Object resting on the ground, within reach + the MPM bounds below.
OBJ_POS = (0.50, 0.0, 0.03)
OBJ_SIZE = (0.04, 0.04, 0.04)
POSE_DR_XY = 0.05                            # per-env object x/y jitter (meters)
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # gripper pointing down (per examples)
PRE_Z, GRASP_Z = 0.33, 0.183                  # EE heights (xarm_gripper_base_link)

# Envs are spread this far apart for rendering only (physics is independent). Bound
# per-env cameras still catch a far neighbour's arm in the background, so we crop any
# depth past DEPTH_CROP (80% of the gap): the workspace is <1 m deep, a neighbour sits
# ~ENV_SPACING away, so the cut keeps the env's own scene and drops the neighbours.
ENV_SPACING = 2.5
DEPTH_CROP = 0.8 * ENV_SPACING
DEPTH_VIEW_SCALE = 0.5                         # shrink the live depth window (factor 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--n-envs", type=int, default=4,
                    help="parallel envs (homogeneous — same object/control in each). "
                         "GPU memory scales with n_envs * grid_density.")
    ap.add_argument("--show-depth", action="store_true",
                    help="live colored-depth window (cv2; needs a display)")
    args = ap.parse_args()
    B = args.n_envs

    # Per-env pose DR (the supported deformable heterogeneity): each env's object
    # gets a different (x, y) within reach; the grasp target follows it per env.
    rng = np.random.default_rng(0)
    obj_dxy = rng.uniform(-POSE_DR_XY, POSE_DR_XY, size=(B, 2)).astype(np.float32)
    obj_xy = np.array(OBJ_POS[:2], dtype=np.float32) + obj_dxy   # (B, 2) per-env centre
    
    # Genesis imports mujoco at import time; a stray/invalid MUJOCO_GL (e.g. a broken
    # shell value) makes that fail. Force a valid backend before importing genesis.
    if not args.vis and os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
        os.environ["MUJOCO_GL"] = "egl"

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1./30., substeps=80),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.25, -0.15, -0.012), upper_bound=(0.75, 0.15, 0.32),
            grid_density=300,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True, enable_collision=True, enable_self_collision=True,
            gravity=(0.0, 0.0, -9.81), box_box_detection=True, constraint_timeconst=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, -1.2, 1.4), camera_lookat=(0.45, 0.0, 0.15), camera_fov=35,
        ),
        # Render depth for ALL envs (each env needs its own point cloud for the
        # policy obs); the depth window only *displays* up to 6 of them.
        # rendered_envs_idx must contain every env we bind a camera to (env_idx=j).
        # env_separate_rigid stays False: a per-env BOUND camera (env_idx=j) is
        # incompatible with it (set_pose shape bug), and unnecessary here — each env
        # is offset by env_spacing, so a bound camera only frames its own env anyway.
        vis_options=gs.options.VisOptions(visualize_mpm_boundary=True,
                                          rendered_envs_idx=list(range(B))),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file=str(URDF), fixed=True, merge_fixed_links=True,
                       links_to_keep=cfg.LINKS_TO_KEEP, pos=(0.0, 0.0, 0.0)),
        material=gs.materials.Rigid(coup_friction=4.0),
    )
    obj = scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic(E=4e3, nu=0.3, von_mises_yield_stress=2e4, rho=1050),
        morph=gs.morphs.Box(size=OBJ_SIZE, pos=OBJ_POS, euler=(0, 0, 0)),
        surface=gs.surfaces.Default(vis_mode="particle"),
    )
    # Per-env depth cameras (like the real cam_ext): ONE camera bound to each env
    # via env_idx=j. Genesis applies that env's offset automatically and renders it
    # fully (arm + deformable). The single batched camera can't do per-env rigid,
    # so we use one camera per env — cost is O(n_envs) renders.
    CAM_POS, CAM_LOOKAT, CAM_FOV = (0.98910661, -0.00034108, 0.09825304), (0.0, 0.0, 0.09825304), 60
    cams = [scene.add_camera(res=(640, 480), pos=CAM_POS, lookat=CAM_LOOKAT,
                             fov=CAM_FOV, GUI=False, env_idx=j) for j in range(B)]

    # env_spacing lays the envs out in a grid for the viewer (purely visual — the
    # physics in each env is independent and identical regardless). Without it,
    # all envs render at the same origin and look like one arm.
    # center_envs_at_origin=False keeps env 0 at the world origin, so the camera
    # frames env 0 directly and its point cloud is already in the (env-0) base frame.
    scene.build(n_envs=B, env_spacing=(ENV_SPACING, ENV_SPACING), center_envs_at_origin=False)

    # ── robot setup ───────────────────────────────────────────────────────────
    def tile(x):  # (d,) → (B, d): batched control input, one row per env
        return np.tile(np.asarray(x, dtype=np.float32)[None], (B, 1))

    dof_idx = [robot.get_joint(n).dof_idx_local for n in cfg.JOINT_NAMES]
    arm_dofs, grip_dofs = dof_idx[:7], dof_idx[7:]
    # kp/kv/force-range are per-DOF (shared across envs), so NOT batched.
    robot.set_dofs_kp(np.array(cfg.KP, dtype=np.float32), dof_idx)
    robot.set_dofs_kv(np.array(cfg.KV, dtype=np.float32), dof_idx)

    ee = robot.get_link(cfg.EE_LINK)
    robot.set_dofs_position(tile(cfg.DEFAULT_JOINT_ANGLES), dof_idx)

    grip_open = np.array(cfg.DEFAULT_JOINT_ANGLES[7:], dtype=np.float32)
    grip_closed = np.full(len(grip_dofs), 0.60, dtype=np.float32)   # ramp target; tune

    def ik(z):  # batched: per-env target (each env grasps its own object) → (B, n_dofs)
        pos = np.concatenate([obj_xy, np.full((B, 1), z, dtype=np.float32)], axis=1)  # (B,3)
        return robot.inverse_kinematics(link=ee, pos=pos, quat=tile(DOWN_QUAT))

    def hold_arm(z):
        robot.control_dofs_position(ik(z)[:, :7], arm_dofs)

    def _np(x):  # genesis returns CUDA torch tensors
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    # Apply per-env pose DR: shift each env's object particles by its (dx, dy)
    # (env-local frame; envs_offset below is render-only and not in physics state).
    base = _np(obj.get_particles_pos())                         # (B, n_particles, 3)
    shift = np.zeros((B, 1, 3), dtype=np.float32)
    shift[:, 0, :2] = obj_dxy
    obj.set_particles_pos(base + shift)

    def render_depth(j):
        """env j's depth image (H, W) meters, from its bound cam_ext camera."""
        return _np(cams[j].render(rgb=False, depth=True)[1]).astype(np.float32)

    def camera_obs(j=0):
        """env j's sim-backend-style camera obs — exactly the RawObs camera fields:
        depth (1, H, W) meters, intrinsics K (3, 3), world_T_cam (4, 4). The sim
        backend loops this over all envs to build per-env point clouds."""
        depth = render_depth(j)[None]                                  # (1, H, W)
        K = np.asarray(cams[j].intrinsics, dtype=np.float32)           # (3, 3)
        world_T_cam = np.linalg.inv(np.asarray(cams[j].extrinsics, dtype=np.float32))  # (4, 4)
        return depth, K, world_T_cam

    def _colorize(d, cv2):
        valid = (d > 0.05) & (d < DEPTH_CROP)
        lo, hi = (np.percentile(d[valid], [2, 98]) if valid.any() else (0.0, 1.0))
        norm = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[~valid] = 0
        return img

    def view_depth():
        # Realtime colored-depth window (cv2 + turbo). Renders every shown env (each
        # from its cam_ext-relative pose, so per-env DR object shows) and tiles them
        # into a grid. Out-of-range pixels are black. Needs a display (--show-depth).
        import cv2
        tiles = [_colorize(render_depth(j), cv2) for j in range(min(B, 6))]  # first ≤6 envs
        cols = min(3, len(tiles))
        rows = -(-len(tiles) // cols)                                 # ceil
        h, w = tiles[0].shape[:2]
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for k, t in enumerate(tiles):
            r, c = divmod(k, cols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
        canvas = cv2.resize(canvas, None, fx=DEPTH_VIEW_SCALE, fy=DEPTH_VIEW_SCALE,
                            interpolation=cv2.INTER_AREA)
        cv2.imshow("sim depth per-env (cam_ext) [turbo]", canvas)
        cv2.waitKey(1)

    step_count = [0]

    def sim_step():
        scene.step()
        step_count[0] += 1
        if args.show_depth and step_count[0] % 3 == 0:
            view_depth()

    def capture_cloud(tag):
        # Feed depth + K + world_T_cam through the SHARED depth_to_pointcloud (same
        # code as the real backend → sim/real parity). Verified to match Genesis's
        # own render_pointcloud. PerceptionPipeline then crops/subsamples downstream.
        depth, K, world_T_cam = camera_obs()
        k0 = K[0] if K.ndim == 3 else K
        e0 = world_T_cam[0] if world_T_cam.ndim == 3 else world_T_cam
        pts, valid = depth_to_pointcloud(depth[0:1], k0, e0, depth_min=0.05, depth_max=DEPTH_CROP)
        vp = pts[0][valid[0]]
        print(f"  [cloud {tag}] {int(valid.sum())} valid pts | "
              f"x[{vp[:,0].min():.2f},{vp[:,0].max():.2f}] "
              f"y[{vp[:,1].min():.2f},{vp[:,1].max():.2f}] z[{vp[:,2].min():.2f},{vp[:,2].max():.2f}]")

    def report(tag, i):
        st = obj.get_state()
        obj_z = _np(st.pos)[..., 2].mean(axis=1)      # (B,) per-env object height
        vmax = _np(st.von_mises).max(axis=1)          # (B,) per-env max von Mises
        ee_z = _np(ee.get_pos())[:, 2]                # (B,)
        print(f"  [{tag} {i}] env0: obj_z={obj_z[0]:.3f} ee_z={ee_z[0]:.3f} "
              f"maxVonMises={vmax[0]:.1f}  |  obj_z spread {obj_z.min():.3f}..{obj_z.max():.3f}")

    # ── grasp sequence ────────────────────────────────────────────────────────
    print(f"pre-grasp  (n_envs={B})  per-env object x "
          f"{obj_xy[:, 0].min():.3f}..{obj_xy[:, 0].max():.3f}  y "
          f"{obj_xy[:, 1].min():.3f}..{obj_xy[:, 1].max():.3f}")
    for _ in range(120):
        hold_arm(PRE_Z)
        robot.control_dofs_position(tile(grip_open), grip_dofs)
        sim_step()
    capture_cloud("pre-grasp")    # object resting on the table

    print("descend")
    for _ in range(120):
        hold_arm(GRASP_Z)
        robot.control_dofs_position(tile(grip_open), grip_dofs)
        sim_step()

    print("close gripper")
    for i in range(120):
        hold_arm(GRASP_Z)
        alpha = min(1.0, i / 80.0)
        robot.control_dofs_position(tile(grip_open * (1 - alpha) + grip_closed * alpha), grip_dofs)
        sim_step()
        if i % 40 == 0:
            report("close", i)

    print("lift")
    for i in range(300):
        hold_arm(GRASP_Z + 0.0006 * i)
        robot.control_dofs_position(tile(grip_closed), grip_dofs)
        sim_step()
        if i % 50 == 0:
            report("lift", i)

    print("done")
    gs.destroy()


if __name__ == "__main__":
    main()
