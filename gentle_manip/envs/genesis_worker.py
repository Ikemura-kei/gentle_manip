"""GenesisWorker — owns a built Genesis scene and drives reset/step/read.

This is the single unit that touches Genesis at run time. It runs in-process
(directly, for tests) and is also the body of the GenesisProcess subprocess. It
takes target EE poses (computed by SimBackend from the policy's delta actions),
not raw actions, and returns plain-numpy state dicts that cross the process
boundary cleanly.

The grasp logic and per-env rendering are ported verbatim from the validated
prototype examples/gs_sim_backend_dev.py.
"""
from __future__ import annotations

import os

# Genesis imports mujoco at import time; force a valid GL backend before that
# (a stray/invalid MUJOCO_GL — e.g. a broken shell value — otherwise crashes the
# import). Keep an already-valid value (e.g. 'glfw' for a viewer).
if os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
    os.environ["MUJOCO_GL"] = "egl"

from typing import Optional

import numpy as np
import genesis as gs

from gentle_manip.robot.xarm7_sim import XArm7Sim, _np
from gentle_manip.scenes.scene_builder import build_scene
from gentle_manip.scenes.scene_spec import SceneSpec


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b for batched (N,4) wxyz quats (apply b, then a — world-frame
    rotation of the base orientation by a)."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], axis=1).astype(np.float32)


class GenesisWorker:
    def __init__(
        self,
        spec: SceneSpec,
        num_envs: int,
        *,
        show_viewer: bool = False,
        settle_steps: int = 40,
        settle_max_steps: int = 200,
        settle_vel_thresh: float = 0.003,
        coup_friction: float = 4.0,
        constraint_timeconst: float = 0.01,
        noslip_iterations: int = 3,
        show_fps: bool = True,
        robot_overrides: Optional[dict] = None,
        render_obs_cameras: bool = True,
        render_rgb_obs: bool = False,
        env_spacing: float = 2.5,
    ) -> None:
        self.num_envs = int(num_envs)
        self.settle_steps = int(settle_steps)
        self.settle_max_steps = int(settle_max_steps)
        self.settle_vel_thresh = float(settle_vel_thresh)
        # When False the scene cameras are still built (so they can be RGB-rendered
        # on demand, e.g. for occasional policy-behaviour clips) but read_state does
        # NOT render their depth every step — keeps the state teacher render-free/fast.
        self.render_obs_cameras = bool(render_obs_cameras)
        # RGB AS AN OBSERVATION (2026-08-29): the same render call returns both, so capturing RGB
        # alongside depth is near-free — but it is OFF by default because every existing dataset and
        # policy is depth/point-cloud only. Needed for VLA baselines (pi0.5), which take images.
        self.render_rgb_obs = bool(render_rgb_obs)

        _init_genesis()
        self.handle = build_scene(
            spec, num_envs, show_viewer=show_viewer,
            coup_friction=coup_friction, constraint_timeconst=constraint_timeconst,
            noslip_iterations=noslip_iterations, show_fps=show_fps,
            robot_overrides=robot_overrides, env_spacing=env_spacing,
        )
        rigid_grasp = any(t == "rigid" for t in self.handle.object_types)
        self.robot = XArm7Sim(self.handle.robot, num_envs, overrides=robot_overrides,
                              rigid_grasp=rigid_grasp)

    # ── lifecycle ───────────────────────────────────────────────────────────────
    def reset(self, object_dxy: Optional[np.ndarray] = None,
              home_offset: Optional[np.ndarray] = None,
              object_euler: Optional[np.ndarray] = None) -> dict:
        """Reset to the built state, re-home the arm, (optionally) pose-DR each
        env's object, settle, and return the initial state.

        object_dxy:   (num_envs, 2) per-env (dx, dy) object jitter, or None.
        home_offset:  (num_envs, 3) per-env (dx, dy, dz) jitter on the reset home EE
                      position (sim-only DR), or None for the shared home pose.
        object_euler: (num_envs, 3) per-env (roll, pitch, yaw) radians — object spawn
                      orientation DR (about its resting center), or None. Rigid: compose
                      into set_quat; soft: rotate the particle cloud about its centroid.
        """
        self.handle.scene.reset()
        self.robot.reset_to_home(home_offset)

        # Per-env rotation (quat wxyz for rigid, matrix for soft particles) from the euler DR.
        rot_quat = rot_mat = None
        if object_euler is not None:
            from scipy.spatial.transform import Rotation as _R
            R = _R.from_euler("xyz", np.asarray(object_euler, np.float64).reshape(self.num_envs, 3))
            q = R.as_quat().astype(np.float32)                    # (N,4) xyzw
            rot_quat = np.concatenate([q[:, 3:4], q[:, :3]], axis=1)   # -> wxyz (genesis convention)
            rot_mat = R.as_matrix().astype(np.float32)            # (N,3,3)

        for obj, otype, base_particles, base_pose in zip(
            self.handle.objects, self.handle.object_types,
            self.handle.object_base_particles, self.handle.object_base_pose,
        ):
            if otype == "rigid":
                base_pos, base_quat = base_pose
                shift = np.zeros((self.num_envs, 3), dtype=np.float32)
                if object_dxy is not None:                        # (N,2) xy or (N,3) xyz offset from the spawn
                    d = np.asarray(object_dxy, dtype=np.float32).reshape(self.num_envs, -1); shift[:, :d.shape[1]] = d
                obj.set_pos(base_pos + shift, zero_velocity=True)
                bq = np.asarray(base_quat, np.float32).reshape(self.num_envs, 4)
                obj.set_quat(_quat_mul_wxyz(rot_quat, bq) if rot_quat is not None else bq,
                             zero_velocity=True)
            else:
                parts = np.asarray(base_particles, np.float32).reshape(self.num_envs, -1, 3)
                if rot_mat is not None:                           # rotate each env about its centroid
                    cen = parts.mean(axis=1, keepdims=True)
                    z0 = parts[:, :, 2].min(axis=1)               # resting height BEFORE the rotation
                    parts = np.einsum("nij,npj->npi", rot_mat, parts - cen) + cen
                    # Rotating about the CENTROID does not preserve the resting height: for an
                    # ELONGATED object a pitch/roll swings the far tip well below the centroid, so
                    # the object spawns partly UNDERGROUND. The banana is 9.5 cm long with its
                    # centroid only ~1.0 cm above the table, so a 45 deg pitch buries a tip ~2 cm
                    # (half-length 4.75 cm * sin45 = 3.4 cm drop). Compact objects (mushroom,
                    # strawberry, raspberry) barely move, which is why this went unnoticed.
                    # Re-seat each env so its lowest particle returns to its original height.
                    # Clamped to RAISE ONLY, so an object that was already resting correctly is
                    # untouched and previously-collected behaviour is preserved.
                    dz = np.maximum(z0 - parts[:, :, 2].min(axis=1), 0.0)
                    parts[:, :, 2] += dz[:, None]
                shift = np.zeros((self.num_envs, 1, 3), dtype=np.float32)
                if object_dxy is not None:                        # (N,2) xy or (N,3) xyz offset from the spawn
                    d = np.asarray(object_dxy, dtype=np.float32).reshape(self.num_envs, -1); shift[:, 0, :d.shape[1]] = d
                obj.set_particles_pos(parts + shift)

        self._settle()
        return self.read_state()

    def _settle(self) -> None:
        """Run sim steps until all rigid objects come to rest (vel < thresh) or
        settle_max_steps is reached. Soft-body objects fall through to the fixed
        settle_steps minimum (no cheap velocity proxy). The minimum settle_steps
        always runs first so the scene physically separates from the spawn pose."""
        has_rigid = any(t == "rigid" for t in self.handle.object_types)

        # Minimum fixed warmup — always run regardless of object type.
        for _ in range(self.settle_steps):
            self.handle.scene.step()

        if not has_rigid:
            return   # soft bodies: fixed steps only

        # Additional steps until all rigid objects are below vel threshold.
        rigid_objs = [obj for obj, t in zip(self.handle.objects, self.handle.object_types)
                      if t == "rigid"]
        for _ in range(self.settle_max_steps - self.settle_steps):
            # get_vel() returns (num_envs, 6) [lin_xyz, ang_xyz] or (num_envs, 3) depending
            # on genesis version — handle both; we take the max speed across all envs & objs.
            max_speed = 0.0
            for obj in rigid_objs:
                v = obj.get_vel()
                vel = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
                max_speed = max(max_speed, float(np.abs(vel).max()))
            if max_speed < self.settle_vel_thresh:
                break
            self.handle.scene.step()

    def step(self, target_pos, target_quat, target_gripper) -> dict:
        """Drive the arm/gripper to the target, advance one sim step, read state."""
        self.robot.apply_target(target_pos, target_quat, target_gripper)
        self.handle.scene.step()
        return self.read_state()

    def set_ee_pose(
        self,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        settle: int = 30,
        gripper_width: Optional[np.ndarray] = None,
    ) -> None:
        """Teleport the robot EE to (pos, quat_wxyz) via IK and settle.

        pos:       (num_envs, 3) target TCP position in world frame.
        quat_wxyz: (num_envs, 4) target TCP orientation, wxyz convention.
        settle:    number of sim steps to run after the hard-set so the arm
                   is physically at rest at the target pose.

        Used for grasp synthesis: instantly place the arm at a pre-grasp pose
        before executing the approach/close/lift motion sequence.
        """
        self.robot.set_ee_pose_hard(
            np.asarray(pos, dtype=np.float32),
            np.asarray(quat_wxyz, dtype=np.float32),
            gripper_width,
        )
        for _ in range(settle):
            self.handle.scene.step()

    def set_object_pos(self, pos: np.ndarray) -> None:
        """Hard-set the first object's position. pos: (num_envs, 3) or (3,)."""
        pos = np.asarray(pos, dtype=np.float32)
        if pos.ndim == 1:
            pos = np.tile(pos[None], (self.num_envs, 1))
        self.handle.objects[0].set_pos(pos, zero_velocity=True)

    def render_rgb(self, all_envs: bool = False):
        """RGB frame(s) uint8 from the first built camera. all_envs=False -> env-0 (H,W,3);
        all_envs=True -> ALL envs (N,H,W,3). None if no camera was built.
        Works in-process AND as the subprocess 'render' command (survives relaunch scene DR)."""
        cams = getattr(self.handle, "cameras", {})
        if not cams:
            return None
        cam_list = next(iter(cams.values()))
        if getattr(self.handle, "batch_render_cameras", False):
            frames = _batched_render(cam_list[0], self.handle.scene, rgb=True, depth=False)
            frames = _np(frames[0]).astype(np.uint8)   # (B, H, W, 3)
            return frames if all_envs else frames[0]
        # Per-env path (soft scenes) — unchanged
        if all_envs:
            return np.stack([_np(c.render(rgb=True, depth=False)[0]).astype(np.uint8)
                             for c in cam_list])   # (N, H, W, 3)
        return _np(cam_list[0].render(rgb=True, depth=False)[0]).astype(np.uint8)

    def close(self) -> None:
        try:
            gs.destroy()
        except Exception:
            pass

    # Alias so GenesisWorker is interface-compatible with GenesisProcess
    # (SimBackend can drive either: subprocess for training, in-process for a viewer).
    def stop(self) -> None:
        self.close()

    # ── state read (all numpy; picklable across the process boundary) ────────────
    def drag_object(self, env: int, vel_xyz) -> None:
        """Set every particle's velocity of the (soft) object in ONE env — a lateral drag for one frame.
        Call for a few consecutive frames for a bounded slide (the solver relaxes it afterwards)."""
        obj = self.handle.objects[0]
        n = np.asarray(self.handle.object_base_particles[0]).reshape(self.num_envs, -1, 3).shape[1]
        obj.set_particles_vel(np.tile(np.asarray(vel_xyz, np.float32), (n, 1)), envs_idx=[int(env)])

    def particle_positions(self):
        """(B, n_p, 3) CURRENT MPM particle positions of the representative object (float32).
        None for a rigid object. Dev/diagnostic accessor — not shipped in read_state()."""
        if self.handle.object_types[0] == "rigid":
            return None
        return _np(self.handle.objects[0].get_state().pos).astype(np.float32)

    def read_state(self) -> dict:
        state = self.robot.read_state()

        # WRIST CAMERA: re-pose per env from world_T_ee @ EE_T_CAM_WRIST BEFORE rendering, the
        # same transform RealBackend applies each step. Genesis cameras are static at add_camera(),
        # so without this the "wrist" view would be a fixed pose that merely looks plausible.
        _wcams = self.handle.cameras.get("cam_wrist") if self.render_obs_cameras else None
        if _wcams and "wrist_cam_T" in state:
            # wrist_cam_T is world_T_cam in the OPENCV convention (+z forward, +y down)
            # -- the convention EE_T_CAM_WRIST was calibrated in, that RealBackend uses,
            # and that depth_to_pointcloud expects. Genesis's set_pose(transform=) wants
            # the OPENGL convention (-z forward, +y up); its own `camera.extrinsics`
            # property converts back with exactly this flip (`res[..., :3, 1:3] *= -1`).
            # Passing the OpenCV matrix raw points the camera 180 deg the wrong way: it
            # renders the empty background above the table, and it STILL moves with the
            # arm -- so a "does the view change?" check does not catch it.
            _T = np.asarray(state["wrist_cam_T"], dtype=np.float32).copy()
            _T[..., :3, 1:3] *= -1.0        # OpenCV -> OpenGL (self-inverse)
            for _e, _c in enumerate(_wcams):
                _c.set_pose(transform=_T[min(_e, _T.shape[0] - 1)])

        depth_images, intrinsics, extrinsics, rgb_images = {}, {}, {}, {}
        cam_items = self.handle.cameras.items() if self.render_obs_cameras else []

        if getattr(self.handle, "batch_render_cameras", False):
            # All-rigid scene: ONE global camera per name, single GPU render call.
            # Back-project with the env-local extrinsic → point cloud lands in
            # env-local coordinates, identical to the per-env path below.
            for name, cam_list in cam_items:
                cam = cam_list[0]
                _rgb, depth = _batched_render(cam, self.handle.scene,
                                              rgb=self.render_rgb_obs, depth=True)
                if self.render_rgb_obs and _rgb is not None:
                    _r = _np(_rgb).astype(np.uint8)
                    rgb_images[name] = _r[None] if _r.ndim == 3 else _r      # (B, H, W, 3)
                d = _np(depth).astype(np.float32)
                if d.ndim == 2:
                    d = d[None]   # rasterizer squeezed B=1 → restore (1, H, W)
                depth_images[name] = d   # (B, H, W)
                intrinsics[name] = np.asarray(cam.intrinsics, dtype=np.float32)
                raw_extr = np.asarray(cam.extrinsics, dtype=np.float32)
                if raw_extr.ndim == 3:
                    raw_extr = raw_extr[0]    # batched cam may return (B, 4, 4)
                extrinsics[name] = np.linalg.inv(raw_extr).astype(np.float32)  # world_T_cam
        else:
            # Per-env path (soft/MPM scenes) — one bound camera per env.
            # env-0's K/extrinsic apply to every env's depth (shared env-local frame).
            for name, cam_list in cam_items:
                _frames = [c.render(rgb=self.render_rgb_obs, depth=True) for c in cam_list]
                depth_images[name] = np.stack(
                    [_np(f[1]).astype(np.float32) for f in _frames])          # (B, H, W)
                if self.render_rgb_obs:
                    rgb_images[name] = np.stack(
                        [_np(f[0]).astype(np.uint8) for f in _frames])        # (B, H, W, 3)
                intrinsics[name] = np.asarray(cam_list[0].intrinsics, dtype=np.float32)
                extrinsics[name] = np.linalg.inv(
                    np.asarray(cam_list[0].extrinsics, dtype=np.float32)
                ).astype(np.float32)  # world_T_cam

        # Representative object → SimFeedback fields. particle_positions are large
        # and unused by the reward components, so we don't ship them every step.
        obj = self.handle.objects[0]
        if self.handle.object_types[0] == "rigid":
            # Rigid solver has no particles/stress; object_center is the entity's
            # base-link position. von_mises_stress is None — SimBackend omits the
            # key entirely so stress-reward KeyErrors propagate, per CLAUDE.md.
            state["object_center"] = _np(obj.get_pos()).astype(np.float32)      # (B, 3)
            state["object_quat"]   = _np(obj.get_quat()).astype(np.float32)    # (B, 4) wxyz
            state["von_mises_stress"] = None
            state["contact_force"] = _gripper_object_contact_force(self.robot.robot, obj)  # (B,)
        else:
            st = obj.get_state()
            particle_pos = _np(st.pos)                                   # (B, n_p, 3)
            state["object_center"] = particle_pos.mean(axis=1).astype(np.float32)   # (B, 3)
            state["von_mises_stress"] = _np(st.von_mises).astype(np.float32)        # (B, n_p)
            # ACTUAL soft-body gripper-object contact: the MPM->finger coupling force Genesis
            # applies to the finger links (0 when nothing touches them). Replaces the old geometric
            # finger<->particle distance heuristic. The soft analogue of the rigid contact_force.
            state["contact_force"] = self.robot.gripper_coupling_force()   # (B,)
            # Orientation of a DEFORMING body: the object was placed tilted (spawn euler) then FELL and
            # settled under gravity, so the spawn euler is NOT its orientation. Recover the actual pose
            # as the best-fit RIGID rotation from the nominal particles to the current ones (Kabsch —
            # the particles keep their correspondence), i.e. the rigid part of the deformation. (B,4) wxyz.
            base = np.asarray(self.handle.object_base_particles[0],
                              np.float32).reshape(self.num_envs, -1, 3)
            state["object_quat"] = _kabsch_quat_wxyz(base, particle_pos).astype(np.float32)
            if os.environ.get("GM_CONTACT_DEBUG"):
                print(f"[contact] soft coupling force env0={state['contact_force'][0]:.6f}", flush=True)

        state["depth_images"] = depth_images
        state["rgb_images"] = rgb_images      # {} unless render_rgb_obs
        state["camera_intrinsics"] = intrinsics
        state["camera_extrinsics"] = extrinsics
        return state


def _kabsch_quat_wxyz(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Best-fit rigid rotation mapping nominal particles P → current particles Q, per env (the MPM
    particles keep their index correspondence), as wxyz quaternions. This is the overall orientation of
    a deforming soft body — used because MPM exposes no rigid quaternion and the spawn euler is invalidated
    once the object falls/settles under gravity. P, Q: (B, n, 3) → (B, 4)."""
    from scipy.spatial.transform import Rotation as _R
    Pc = P - P.mean(axis=1, keepdims=True)
    Qc = Q - Q.mean(axis=1, keepdims=True)
    H = np.einsum("bni,bnj->bij", Pc, Qc)                       # (B,3,3) cross-covariance
    U, _, Vt = np.linalg.svd(H)
    V = np.transpose(Vt, (0, 2, 1)); Ut = np.transpose(U, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ Ut))                          # reflection guard
    D = np.tile(np.eye(3), (len(P), 1, 1)); D[:, 2, 2] = d
    R = V @ D @ Ut                                              # (B,3,3), det=+1
    q = _R.from_matrix(R).as_quat()                             # (B,4) xyzw
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1)        # -> wxyz


def _gripper_object_contact_force(robot_entity, object_entity) -> np.ndarray:
    """Rigid-body grip-force surrogate: sum of contact-force MAGNITUDES between the
    whole robot entity (in practice, whichever links are actually touching — the
    gripper fingers) and the object entity, per env. Shape (B,), Newtons.

    A magnitude sum (not a signed vector sum) is deliberate: the two fingers' contact
    forces point in roughly opposite directions, so a vector sum would mostly cancel
    and only show the small residual (gravity support) — hiding the squeeze force
    that's actually of interest. This mirrors the existing von-Mises stress reward's
    use of a severity scalar rather than a signed tensor.
    """
    contacts = robot_entity.get_contacts(with_entity=object_entity)
    force = _np(contacts["force_a"])                        # (B, n_contacts, 3)
    if force.shape[-2] == 0:
        return np.zeros(force.shape[0], dtype=np.float32)
    valid = _np(contacts.get("valid_mask", np.ones(force.shape[:-1], dtype=bool)))  # (B, n_contacts)
    mag = np.linalg.norm(force, axis=-1) * valid            # (B, n_contacts)
    return mag.sum(axis=-1).astype(np.float32)              # (B,)


def _batched_render(cam, scene, *, rgb: bool, depth: bool):
    """Render a global (non-env-bound) camera for all envs in one rasterizer call.

    Genesis keeps the camera transform in env-local coordinates. With
    ``env_separate_rigid=True``, each env's rigid geometry is rendered at its
    world position (local_pos + envs_offset). We temporarily shift the camera
    translation by each env's offset so the camera "follows" the geometry, then
    restore it afterwards. This is the pattern from gs_sim_backend_batched_cam_dev.py.

    Returns ``(rgb_tensor, depth_tensor)``; either may be None if not requested.
    The depth tensor has shape ``(B, H, W)`` and the RGB tensor ``(B, H, W, 3)``.
    """
    import torch
    ctx = scene.visualizer.context
    ctx.update(force_render=True)

    orig = cam._transform.clone()
    offsets = torch.as_tensor(
        ctx.scene.envs_offset[ctx.rendered_envs_idx],
        dtype=cam._transform.dtype,
        device=cam._transform.device,
    )
    cam._transform[..., :3, 3] += offsets
    scene.visualizer.rasterizer.update_camera(cam)
    try:
        out = scene.visualizer.rasterizer.render_camera(cam, rgb=rgb, depth=depth)
    finally:
        cam._transform[:] = orig
        scene.visualizer.rasterizer.update_camera(cam)
    # render_camera returns (rgb, depth, normal, …); normalise to a 2-tuple
    rgb_out   = out[0] if rgb   else None
    depth_out = out[1] if depth else None
    return rgb_out, depth_out


def _init_genesis() -> None:
    """gs.init once per process (idempotent for in-process re-creation)."""
    if getattr(gs, "_initialized", False):
        return
    gs.init(backend=gs.gpu)
