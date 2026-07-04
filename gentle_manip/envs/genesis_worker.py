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
        settle_steps: int = 30,
        coup_friction: float = 4.0,
        constraint_timeconst: float = 0.01,
        noslip_iterations: int = 3,
        show_fps: bool = True,
        robot_overrides: Optional[dict] = None,
        render_obs_cameras: bool = True,
    ) -> None:
        self.num_envs = int(num_envs)
        self.settle_steps = int(settle_steps)
        # When False the scene cameras are still built (so they can be RGB-rendered
        # on demand, e.g. for occasional policy-behaviour clips) but read_state does
        # NOT render their depth every step — keeps the state teacher render-free/fast.
        self.render_obs_cameras = bool(render_obs_cameras)

        _init_genesis()
        self.handle = build_scene(
            spec, num_envs, show_viewer=show_viewer,
            coup_friction=coup_friction, constraint_timeconst=constraint_timeconst,
            noslip_iterations=noslip_iterations, show_fps=show_fps,
            robot_overrides=robot_overrides,
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

        for _ in range(self.settle_steps):
            self.handle.scene.step()
        return self.read_state()

    def step(self, target_pos, target_quat, target_gripper) -> dict:
        """Drive the arm/gripper to the target, advance one sim step, read state."""
        self.robot.apply_target(target_pos, target_quat, target_gripper)
        self.handle.scene.step()
        return self.read_state()

    def render_rgb(self):
        """Env-0 RGB frame (H,W,3) uint8 from the first built camera, or None if no camera
        was built. Used for behaviour clips / eval video — works in-process AND as the
        subprocess 'render' command (so video survives the relaunch-based scene DR)."""
        cams = getattr(self.handle, "cameras", {})
        if not cams:
            return None
        cam = next(iter(cams.values()))[0]        # env-0 camera
        return _np(cam.render(rgb=True, depth=False)[0]).astype(np.uint8)

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
        for name, cam_list in cam_items:
            # One bound camera per env; each images its own env from the same
            # relative pose, so env-0's K/extrinsic apply to every env's depth
            # (cloud lands in the base frame — see dev script).
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
            state["von_mises_stress"] = None
        else:
            st = obj.get_state()
            particle_pos = _np(st.pos)                                   # (B, n_p, 3)
            state["object_center"] = particle_pos.mean(axis=1).astype(np.float32)   # (B, 3)
            state["von_mises_stress"] = _np(st.von_mises).astype(np.float32)        # (B, n_p)

        state["depth_images"] = depth_images
        state["camera_intrinsics"] = intrinsics
        state["camera_extrinsics"] = extrinsics
        return state


def _init_genesis() -> None:
    """gs.init once per process (idempotent for in-process re-creation)."""
    if getattr(gs, "_initialized", False):
        return
    gs.init(backend=gs.gpu)
