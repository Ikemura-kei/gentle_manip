"""Sim side of the RawObs boundary: Genesis (in a subprocess) -> RawObs.

Implements the Backend protocol PolicyEnv consumes, with num_envs = N parallel
Genesis envs. Genesis itself lives in a GenesisProcess child (memory-leak fix);
SimBackend owns the policy-facing logic: it interprets actions as deltas and
accumulates them into a target EE pose + gripper, exactly as RealBackend does
(but batched), so sim and real share action semantics. Observations always
reflect the read-back state, never the command.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.envs.genesis_process import GenesisProcess
from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.robot import xarm7_config as cfg
from gentle_manip.scenes.scene_spec import SceneSpec


class SimBackend:
    def __init__(
        self,
        spec: SceneSpec,
        num_envs: int,
        config: Optional[dict] = None,
        *,
        use_subprocess: bool = True,
        show_viewer: bool = False,
    ) -> None:
        config = config or {}
        self.num_envs = int(num_envs)

        robot_overrides = config.get("robot", {})
        sim_cfg = config.get("sim", {})
        self._pose_dr_xy = float(config.get("dr", {}).get("pose_dr_xy", 0.0))
        self._gripper_max = float(
            robot_overrides.get("default_gripper_width", cfg.DEFAULT_GRIPPER_WIDTH)
        )
        self._rng = np.random.default_rng(config.get("seed", 0))

        worker_kwargs = dict(
            settle_steps=int(sim_cfg.get("settle_steps", 30)),
            coup_friction=float(sim_cfg.get("coup_friction", 4.0)),
            robot_overrides=robot_overrides,
        )
        if use_subprocess:
            # Training path: Genesis in a child process (GPU-memory-leak fix). A
            # viewer can't run under the headless egl child, so it's ignored here.
            self.process = GenesisProcess(spec, num_envs, **worker_kwargs)
            self.process.start()
        else:
            # Debug/teleop path: Genesis in-process so the 3D viewer can open.
            # Import locally so the subprocess path never imports genesis in-parent.
            from gentle_manip.envs.genesis_worker import GenesisWorker
            self.process = GenesisWorker(spec, num_envs, show_viewer=show_viewer, **worker_kwargs)

        # Accumulated command targets (seeded on reset).
        self._target_pos = np.zeros((num_envs, 3), dtype=np.float64)
        self._target_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_envs, 1)).astype(np.float64)
        self._target_gripper = np.full(num_envs, self._gripper_max, dtype=np.float64)
        self._last_state: Optional[dict] = None

    # ── Backend protocol ──────────────────────────────────────────────────────
    def reset(self, object_dxy=None, **kwargs) -> RawObs:
        # Explicit object_dxy (num_envs, 2) places the object at a chosen offset from
        # its default pose (e.g. to match a recorded demo's cube); otherwise pose-DR.
        if object_dxy is not None:
            object_dxy = np.asarray(object_dxy, dtype=np.float32).reshape(self.num_envs, 2)
        elif self._pose_dr_xy > 0:
            object_dxy = self._rng.uniform(
                -self._pose_dr_xy, self._pose_dr_xy, size=(self.num_envs, 2)
            ).astype(np.float32)

        state = self.process.reset(object_dxy)
        self._last_state = state
        # Seed targets from the actual reset pose so the first deltas are relative
        # to where the arm really is.
        self._target_pos = state["ee_pos"].astype(np.float64).copy()
        self._target_quat = state["ee_quat"].astype(np.float64).copy()
        self._target_gripper = state["gripper_width"].astype(np.float64).copy()
        return self._build_raw_obs(state)

    def step(self, scaled_action: np.ndarray) -> RawObs:
        action = np.asarray(scaled_action, dtype=np.float64).reshape(self.num_envs, -1)
        dpos, drot, dgrip = action[:, :3], action[:, 3:6], action[:, 6]

        # Translation: accumulate then clip to the workspace box (per env).
        self._target_pos = np.clip(
            self._target_pos + dpos, cfg.EE_BOUNDS_MIN, cfg.EE_BOUNDS_MAX
        )

        # Orientation: compose the delta rotation (base-frame premultiply), batched.
        q = self._target_quat
        R_cur = Rotation.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]]))
        xyzw = (Rotation.from_rotvec(drot) * R_cur).as_quat()       # (B, 4) xyzw
        wxyz = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
        neg = wxyz[:, 0] < 0
        wxyz[neg] = -wxyz[neg]                                      # keep w >= 0
        self._target_quat = wxyz

        # Gripper: accumulate then clip to [0, open width].
        self._target_gripper = np.clip(self._target_gripper + dgrip, 0.0, self._gripper_max)

        state = self.process.step(
            self._target_pos.astype(np.float32),
            self._target_quat.astype(np.float32),
            self._target_gripper.astype(np.float32),
        )
        self._last_state = state
        return self._build_raw_obs(state)

    def get_sim_feedback(self) -> Optional[SimFeedback]:
        if self._last_state is None:
            return None
        s = self._last_state
        extra = {} if s["von_mises_stress"] is None else {"von_mises_stress": s["von_mises_stress"]}
        return SimFeedback(
            ee_pos=s["ee_pos"],
            gripper_width=s["gripper_width"],
            object_center=s["object_center"],
            extra=extra,
        )

    def close(self) -> None:
        self.process.stop()

    # ── internal ──────────────────────────────────────────────────────────────
    def _build_raw_obs(self, state: dict) -> RawObs:
        raw = RawObs(
            ee_pos=state["ee_pos"],
            ee_quat=state["ee_quat"],
            gripper_width=state["gripper_width"],
            joint_pos=state["joint_pos"],
            joint_vel=state["joint_vel"],
            depth_images=state["depth_images"],
            rgb_images={},                              # MVP: depth/point-cloud only
            camera_intrinsics=state["camera_intrinsics"],
            camera_extrinsics=state["camera_extrinsics"],
            tactile_images={},                          # sim has no tactile
        )
        raw.validate()
        return raw
