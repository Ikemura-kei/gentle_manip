"""Genesis v1.1.x deformable-grasp sim with ONE batched camera for all envs.

This is a variant of gs_sim_backend_dev.py that replaces the per-env bound
cameras (`scene.add_camera(..., env_idx=j)`) with a single batched camera that
renders all parallel environments in one call.

Key insight from Genesis v1.0+:
  - `gs.sensors.DepthCamera` is a raycaster and does NOT see MPM entities.
  - `scene.add_camera()` uses the rasterizer and DOES render MPM entities.
  - With `VisOptions(env_separate_rigid=True)`, one camera returns batched
    depth of shape (n_envs, H, W).

Run in the sim env:

    uv run --project envs/sim python examples/gs_sim_backend_batched_cam_dev.py
    uv run --project envs/sim python examples/gs_sim_backend_batched_cam_dev.py --vis
    uv run --project envs/sim python examples/gs_sim_backend_batched_cam_dev.py --show-depth
"""
import os
import sys
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
POSE_DR_XY = 0.1                            # per-env object x/y jitter (meters)
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # gripper pointing down
PRE_Z, GRASP_Z = 0.33, 0.183                  # EE heights (xarm_gripper_base_link)

# With env_separate_rigid=True each env is rendered at its own local origin, so
# neighbours are not visible. We keep a small spacing only for the viewer layout.
ENV_SPACING = 2.5
DEPTH_CROP = 0.95                              # workspace depth crop (meters)
DEPTH_VIEW_SCALE = 0.5                         # shrink the live depth window

# cam_ext pose in the *env-local* frame (same relative pose for every env).
CAM_POS = np.array([0.98910661, -0.00034108, 0.09825304], dtype=np.float32)
CAM_LOOKAT = np.array([0.0, 0.0, 0.09825304], dtype=np.float32)
CAM_FOV = 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--n-envs", type=int, default=4,
                    help="parallel envs (homogeneous). GPU memory scales with n_envs * grid_density.")
    ap.add_argument("--show-depth", action="store_true",
                    help="live colored-depth window (cv2; needs a display)")
    ap.add_argument("--wrist-cam", action="store_true",
                    help="also attach a batched wrist camera (cam_wrist) to the EE")
    args = ap.parse_args()
    B = args.n_envs

    rng = np.random.default_rng(0)
    obj_dxy = rng.uniform(-POSE_DR_XY, POSE_DR_XY, size=(B, 2)).astype(np.float32)
    obj_xy = np.array(OBJ_POS[:2], dtype=np.float32) + obj_dxy

    if not args.vis and os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
        os.environ["MUJOCO_GL"] = "egl"

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1. / 30., substeps=80),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.25, -0.25, -0.012),
            upper_bound=(0.75, 0.25, 0.32),
            grid_density=300,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True, enable_collision=True, enable_self_collision=True,
            gravity=(0.0, 0.0, -9.81), box_box_detection=True, constraint_timeconst=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, -1.2, 1.4),
            camera_lookat=(0.45, 0.0, 0.15),
            camera_fov=35,
        ),
        # REQUIRED for batched per-env rendering with a single camera.
        vis_options=gs.options.VisOptions(
            env_separate_rigid=True,
            rendered_envs_idx=list(range(B)),
            visualize_mpm_boundary=True,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(URDF),
            fixed=True,
            merge_fixed_links=True,
            links_to_keep=cfg.LINKS_TO_KEEP,
            pos=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(coup_friction=4.0),
    )
    obj = scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic(
            E=4e3, nu=0.3, von_mises_yield_stress=2e4, rho=1050
        ),
        morph=gs.morphs.Box(size=OBJ_SIZE, pos=OBJ_POS, euler=(0, 0, 0)),
        surface=gs.surfaces.Default(vis_mode="particle"),
    )

    # ONE batched world-fixed camera (cam_ext). With env_separate_rigid=True this
    # renders every env at its local origin and returns depth of shape (B, H, W).
    cam_ext = scene.add_camera(
        res=(640, 480),
        pos=CAM_POS,
        lookat=CAM_LOOKAT,
        fov=CAM_FOV,
        GUI=False,
        # NOTE: do NOT pass env_idx here. env_idx binds the camera to a single env
        # and disables batched output.
    )

    ee = None  # filled after build
    cam_wrist = None
    wrist_offset_T = None

    scene.build(n_envs=B, env_spacing=(ENV_SPACING, ENV_SPACING), center_envs_at_origin=False)

    # ── robot setup ─────────────────────────────────────────────────────────
    def tile(x):
        return np.tile(np.asarray(x, dtype=np.float32)[None], (B, 1))

    dof_idx = [robot.get_joint(n).dof_idx_local for n in cfg.JOINT_NAMES]
    arm_dofs, grip_dofs = dof_idx[:7], dof_idx[7:]
    robot.set_dofs_kp(np.array(cfg.KP, dtype=np.float32), dof_idx)
    robot.set_dofs_kv(np.array(cfg.KV, dtype=np.float32), dof_idx)

    ee = robot.get_link(cfg.EE_LINK)
    robot.set_dofs_position(tile(cfg.DEFAULT_JOINT_ANGLES), dof_idx)

    if args.wrist_cam:
        # Attach a second batched camera to the EE. Because the link pose is
        # per-env, the camera transform becomes (B, 4, 4) and the rasterizer
        # automatically renders each env separately.
        wrist_offset_T = np.eye(4, dtype=np.float32)
        # Roughly in front of the gripper, looking down at the workspace.
        wrist_offset_T[:3, 3] = [0.05, 0.0, -0.08]
        cam_wrist = scene.add_camera(res=(320, 240), fov=70, GUI=False)
        cam_wrist.attach(ee, wrist_offset_T)

    grip_open = np.array(cfg.DEFAULT_JOINT_ANGLES[7:], dtype=np.float32)
    grip_closed = np.full(len(grip_dofs), 0.60, dtype=np.float32)

    def ik(z):
        pos = np.concatenate([obj_xy, np.full((B, 1), z, dtype=np.float32)], axis=1)
        return robot.inverse_kinematics(link=ee, pos=pos, quat=tile(DOWN_QUAT))

    def hold_arm(z):
        robot.control_dofs_position(ik(z)[:, :7], arm_dofs)

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    # Apply per-env pose DR.
    base = _np(obj.get_particles_pos())
    shift = np.zeros((B, 1, 3), dtype=np.float32)
    shift[:, 0, :2] = obj_dxy
    obj.set_particles_pos(base + shift)

    # ── camera helpers ──────────────────────────────────────────────────────
    def update_cameras():
        """Call move_to_attach for any attached cameras before rendering."""
        if cam_wrist is not None:
            cam_wrist.move_to_attach()

    def _strip_rigid_offsets(ctx):
        """Replicate the RasterizerCameraSensor offset-stripping hack.

        With env_separate_rigid=True, rigid geometry poses have env offsets baked
        in, but the scene-camera path does not strip them before rendering. This
        makes the arm visible only in env 0. Subtract the offsets temporarily,
        render, then restore.
        """
        if not ctx.env_separate_rigid:
            return {}
        envs_offset = ctx.scene.envs_offset
        if (envs_offset == 0).all():
            return {}
        saved = {}
        for node_uid, node in ctx.rigid_nodes.items():
            primitive = node.mesh.primitives[0]
            poses = primitive.poses
            if poses is not None and len(poses) > 1:
                saved[node_uid] = poses.copy()
                poses[:, :3, 3] -= envs_offset[ctx.rendered_envs_idx]
                buf_id = ctx._scene.get_buffer_id(node, "model")
                if buf_id >= 0:
                    ctx.jit.update_buffer(buf_id, poses.transpose((0, 2, 1)))
        return saved

    def _restore_rigid_offsets(ctx, saved):
        for node_uid, poses in saved.items():
            node = ctx.rigid_nodes[node_uid]
            node.mesh.primitives[0].poses = poses
            buf_id = ctx._scene.get_buffer_id(node, "model")
            if buf_id >= 0:
                ctx.jit.update_buffer(buf_id, poses.transpose((0, 2, 1)))

    def render_batched_depth(cam):
        """Render one camera for all envs. Returns (B, H, W) float32 depth."""
        update_cameras()
        ctx = scene.visualizer.context
        # Update visual state (MPM skinning, rigid poses, etc.) before stripping.
        ctx.update(force_render=True)
        saved = _strip_rigid_offsets(ctx)
        try:
            # Bypass cam.render() so the context is not re-updated (which would
            # restore the offsets we just stripped).
            _, depth, *_ = scene.visualizer.rasterizer.render_camera(
                cam, rgb=False, depth=True
            )
        finally:
            _restore_rigid_offsets(ctx, saved)
        return _np(depth).astype(np.float32)

    def camera_obs(cam, name):
        """Return RawObs-style fields for the given camera.

        depth:       (B, H, W)
        K:           (3, 3)   (shared across envs)
        world_T_cam: (4, 4)   (world_T_cam)

        NOTE: depth_to_pointcloud expects a single shared (4, 4) world_T_cam.
        This helper assumes the camera pose is shared across envs (true for the
        unbound cam_ext). A wrist-mounted camera has a per-env pose and needs a
        different conversion path.
        """
        depth = render_batched_depth(cam)                       # (B, H, W)
        K = np.asarray(cam.intrinsics, dtype=np.float32)        # (3, 3)
        # Genesis cam.extrinsics is cam_T_world (OpenCV convention). For a
        # batched unbound camera it is (B, 4, 4) with identical entries; invert
        # to get world_T_cam and take the shared (4, 4) matrix.
        extr = np.asarray(cam.extrinsics, dtype=np.float32)
        if extr.ndim == 3:
            extr = extr[0]
        world_T_cam = np.linalg.inv(extr)
        return depth, K, world_T_cam

    def _colorize(d, cv2):
        valid = (d > 0.05) & (d < DEPTH_CROP)
        lo, hi = (np.percentile(d[valid], [2, 98]) if valid.any() else (0.0, 1.0))
        norm = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[~valid] = 0
        return img

    def view_depth():
        import cv2
        depth = render_batched_depth(cam_ext)                    # (B, H, W)
        tiles = [_colorize(depth[j], cv2) for j in range(min(B, 6))]
        cols = min(3, len(tiles))
        rows = -(-len(tiles) // cols)
        h, w = tiles[0].shape[:2]
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for k, t in enumerate(tiles):
            r, c = divmod(k, cols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
        canvas = cv2.resize(canvas, None, fx=DEPTH_VIEW_SCALE, fy=DEPTH_VIEW_SCALE,
                            interpolation=cv2.INTER_AREA)
        cv2.imshow("batched sim depth (cam_ext) [turbo]", canvas)
        cv2.waitKey(1)

    step_count = [0]

    def sim_step():
        scene.step()
        step_count[0] += 1
        if args.show_depth and step_count[0] % 3 == 0:
            view_depth()

    def capture_cloud(tag, cam=cam_ext):
        depth, K, world_T_cam = camera_obs(cam, tag)
        # depth_to_pointcloud expects (B, H, W); K is shared.
        pts, valid = depth_to_pointcloud(
            depth, K, world_T_cam, depth_min=0.05, depth_max=DEPTH_CROP
        )
        j = 0
        vp = pts[j][valid[j]]
        print(f"  [cloud {tag} env{j}] {int(valid[j].sum())} valid pts | "
              f"x[{vp[:, 0].min():.2f},{vp[:, 0].max():.2f}] "
              f"y[{vp[:, 1].min():.2f},{vp[:, 1].max():.2f}] "
              f"z[{vp[:, 2].min():.2f},{vp[:, 2].max():.2f}]")
        if B > 1:
            print(f"  [cloud {tag}] depth shape {depth.shape}; world_T_cam shape {world_T_cam.shape}")

    def report(tag, i):
        st = obj.get_state()
        obj_z = _np(st.pos)[..., 2].mean(axis=1)
        vmax = _np(st.von_mises).max(axis=1)
        ee_z = _np(ee.get_pos())[:, 2]
        print(f"  [{tag} {i}] env0: obj_z={obj_z[0]:.3f} ee_z={ee_z[0]:.3f} "
              f"maxVonMises={vmax[0]:.1f}  |  obj_z spread {obj_z.min():.3f}..{obj_z.max():.3f}")

    # ── grasp sequence ──────────────────────────────────────────────────────
    print(f"pre-grasp  (n_envs={B})  per-env object x "
          f"{obj_xy[:, 0].min():.3f}..{obj_xy[:, 0].max():.3f}  y "
          f"{obj_xy[:, 1].min():.3f}..{obj_xy[:, 1].max():.3f}")
    for _ in range(120):
        hold_arm(PRE_Z)
        robot.control_dofs_position(tile(grip_open), grip_dofs)
        sim_step()
    capture_cloud("pre-grasp")

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

    if cam_wrist is not None:
        print("wrist-cam batched render")
        d_wrist = render_batched_depth(cam_wrist)
        print(f"  wrist depth shape: {d_wrist.shape} (expected ({B}, 240, 320))")

    print("done")
    gs.destroy()


if __name__ == "__main__":
    main()
