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

        # reactive perturbation state (set per-reset via `perturb`): one-frame lateral
        # velocity impulse to the object at sim-frame `fire_frame` (per env).
        self._frame = 0
        self._perturb = None

    # ── lifecycle ───────────────────────────────────────────────────────────────
    def reset(self, object_dxy: Optional[np.ndarray] = None,
              home_offset: Optional[np.ndarray] = None,
              object_euler: Optional[np.ndarray] = None,
              perturb: Optional[dict] = None) -> dict:
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
                if object_dxy is not None:
                    shift[:, :2] = np.asarray(object_dxy, dtype=np.float32)
                obj.set_pos(base_pos + shift, zero_velocity=True)
                bq = np.asarray(base_quat, np.float32).reshape(self.num_envs, 4)
                obj.set_quat(_quat_mul_wxyz(rot_quat, bq) if rot_quat is not None else bq,
                             zero_velocity=True)
            else:
                parts = np.asarray(base_particles, np.float32).reshape(self.num_envs, -1, 3)
                if rot_mat is not None:                           # rotate each env about its centroid
                    cen = parts.mean(axis=1, keepdims=True)
                    parts = np.einsum("nij,npj->npi", rot_mat, parts - cen) + cen
                shift = np.zeros((self.num_envs, 1, 3), dtype=np.float32)
                if object_dxy is not None:
                    shift[:, 0, :2] = np.asarray(object_dxy, dtype=np.float32)
                obj.set_particles_pos(parts + shift)

        self._settle()
        # arm the reactive perturbation for this episode (frame counter starts AFTER settle)
        self._frame = 0
        self._perturb = None
        if perturb is not None:
            ff = np.asarray(perturb["fire_frame"], np.int64).reshape(self.num_envs)
            vv = np.asarray(perturb["vel"], np.float32).reshape(self.num_envs, 3)
            if np.any(ff >= 0):
                self._perturb = {"fire_frame": ff, "vel": vv}
        return self.read_state()

    @staticmethod
    def _to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else (
            x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x))

    _PERTURB_HOLD = 4   # SET the object velocity for this many consecutive frames -> a
                        # bounded "dragged away" slide (not a compounding add). `perturb_vel`
                        # is the per-frame speed; ~4 frames * dt gives a ~5-15 cm displacement.

    def _apply_perturbation(self) -> None:
        """Sustained lateral velocity drag on the object over a short window starting at
        `fire_frame` ("dragged away by a random force"). MPM soft: set every particle's
        velocity for that env; rigid: set the body velocity. The solver then relaxes it.
        Never raises (a perturb failure must not kill a rollout)."""
        p = self._perturb
        if p is None:
            return
        hits = np.nonzero((p["fire_frame"] >= 0) & (self._frame >= p["fire_frame"])
                          & (self._frame < p["fire_frame"] + self._PERTURB_HOLD))[0]
        if hits.size == 0:
            return
        for obj, otype, base_particles in zip(
            self.handle.objects, self.handle.object_types, self.handle.object_base_particles,
        ):
            n_part = 0 if otype == "rigid" else int(
                np.asarray(base_particles).reshape(self.num_envs, -1, 3).shape[1])
            for e in hits:
                v = np.asarray(p["vel"][int(e)], np.float32)          # per-frame drag velocity
                try:
                    if otype == "rigid":
                        cur = self._to_np(obj.get_vel()).astype(np.float32).reshape(self.num_envs, -1)
                        cur[int(e), :3] = v
                        obj.set_vel(cur)
                    else:
                        obj.set_particles_vel(np.tile(v, (n_part, 1)).astype(np.float32),
                                              envs_idx=[int(e)])
                    if self._frame == int(p["fire_frame"][int(e)]):   # log once per kick
                        print(f"[perturb] frame {self._frame} env {int(e)} drag v={np.round(v, 3).tolist()}", flush=True)
                except Exception as ex:
                    print(f"[perturb] env {int(e)} drag failed: {ex}", flush=True)

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
        self._frame += 1
        self._apply_perturbation()          # reactive kick (no-op unless armed + frame hits)
        self.handle.scene.step()
        return self.read_state()

    def set_ee_pose(
        self,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        settle: int = 30,
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
            if frames.ndim == 3:
                frames = frames[None]   # rasterizer squeezed B=1 -> restore (1, H, W, 3), matching read_state()'s depth fix
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
    def read_state(self) -> dict:
        state = self.robot.read_state()

        depth_images, intrinsics, extrinsics = {}, {}, {}
        cam_items = self.handle.cameras.items() if self.render_obs_cameras else []

        if getattr(self.handle, "batch_render_cameras", False):
            # All-rigid scene: ONE global camera per name, single GPU render call.
            # Back-project with the env-local extrinsic → point cloud lands in
            # env-local coordinates, identical to the per-env path below.
            for name, cam_list in cam_items:
                cam = cam_list[0]
                _, depth = _batched_render(cam, self.handle.scene, rgb=False, depth=True)
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
                depth_images[name] = np.stack(
                    [_np(c.render(rgb=False, depth=True)[1]).astype(np.float32) for c in cam_list]
                )  # (B, H, W)
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
            try:
                state["contact_force"] = _gripper_object_contact_force(self.robot.robot, obj)  # (B,)
            except RuntimeError as e:
                # Upstream Genesis bug (pinned fork, unmodified collider.py): the
                # non-zerocopy get_contacts() path gathers with an int32
                # contact_sort_idx, which torch.gather rejects ("Expected dtype
                # int64 for index"). Hit on every rigid-object step, not just this
                # object -- a genesis-side fix is out of scope for this session, so
                # degrade gracefully (contact_force is an OPTIONAL signal: the
                # collector's "firm regrasp" heuristic and an opt-in privileged obs
                # field, neither load-bearing for basic rigid pick-and-lift).
                if not getattr(self, "_contact_force_warned", False):
                    print(f"[genesis_worker] contact_force unavailable (upstream Genesis "
                         f"bug, degrading to zeros): {e}", flush=True)
                    self._contact_force_warned = True
                n_envs = _np(obj.get_pos()).shape[0]
                state["contact_force"] = np.zeros(n_envs, dtype=np.float32)
        else:
            st = obj.get_state()
            particle_pos = _np(st.pos)                                   # (B, n_p, 3)
            state["object_center"] = particle_pos.mean(axis=1).astype(np.float32)   # (B, 3)
            state["von_mises_stress"] = _np(st.von_mises).astype(np.float32)        # (B, n_p)
            state["contact_force"] = None   # rigid-only surrogate; soft bodies use von_mises_stress
            # Orientation of a DEFORMING body: the object was placed tilted (spawn euler) then FELL and
            # settled under gravity, so the spawn euler is NOT its orientation. Recover the actual pose
            # as the best-fit RIGID rotation from the nominal particles to the current ones (Kabsch —
            # the particles keep their correspondence), i.e. the rigid part of the deformation. (B,4) wxyz.
            # Ported from origin/master (d37b13f) for the v3 FEM grasp-synthesis integration, which reads
            # object_quat for soft/MPM objects -- previously only set for rigid bodies (KeyError otherwise).
            base = np.asarray(self.handle.object_base_particles[0],
                              np.float32).reshape(self.num_envs, -1, 3)
            state["object_quat"] = _kabsch_quat_wxyz(base, particle_pos).astype(np.float32)

        state["depth_images"] = depth_images
        state["camera_intrinsics"] = intrinsics
        state["camera_extrinsics"] = extrinsics
        return state


def _kabsch_quat_wxyz(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Best-fit rigid rotation mapping nominal particles P -> current particles Q, per env (the MPM
    particles keep their index correspondence), as wxyz quaternions. This is the overall orientation of
    a deforming soft body -- used because MPM exposes no rigid quaternion and the spawn euler is invalidated
    once the object falls/settles under gravity. P, Q: (B, n, 3) -> (B, 4). Ported from origin/master."""
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
