from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from gentle_manip.actions.action_config import ActionConfig


def _rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthonormalization (Zhou et al. 2019): (N, 6) -> (N, 4) wxyz quat.

    Inverse of the rot6d construction in PolicyEnv._privileged_obs (first two columns
    of the rotation matrix): a1/a2 are those two columns; b1/b2/b3 are re-orthonormalized
    so any raw (unconstrained) 6-vector projects onto a valid rotation, continuously.
    """
    a1, a2 = rot6d[:, 0:3], rot6d[:, 3:6]
    eps = 1e-8
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + eps)
    a2_perp = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = a2_perp / (np.linalg.norm(a2_perp, axis=1, keepdims=True) + eps)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=-1)                # (N, 3, 3), columns b1,b2,b3
    xyzw = Rotation.from_matrix(R).as_quat()
    wxyz = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
    neg = wxyz[:, 0] < 0
    wxyz[neg] = -wxyz[neg]                              # keep w >= 0 (sign convention)
    return wxyz.astype(np.float32)


def invert_absolute_action(pos: np.ndarray, quat: np.ndarray, gripper: np.ndarray,
                           action_config: ActionConfig) -> np.ndarray:
    """Inverse of ActionPipeline._process_absolute: given a physical (pos, quat_wxyz,
    gripper) command, compute the (N, 10) raw [-1,1] action that an absolute-mode
    ActionPipeline built from `action_config` would map back to it. No history needed
    (absolute mode has no accumulation, each step is an independent forward transform).

    Use this to record an equivalent ABSOLUTE action for a trajectory that was actually
    driven via DELTA control (e.g. teleop, which is smoother/easier to operate) — invert
    the ACTUAL resulting pose read back after each step, so the recorded action is always
    exactly consistent with what really happened (no separate accumulator to drift).

    pos/gripper: un-map the linear [pos_min,pos_max]/[gripper_min,gripper_max] scaling.
    rot6d: the first two columns of R = Rotation.from_quat(quat).as_matrix() are already
    exactly orthonormal, so Gram-Schmidt on them is a no-op -- the rot6d "inverse" is just
    those two columns directly, no search needed.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    quat = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    n = pos.shape[0]

    lo, hi = action_config.clip
    span = hi - lo
    pos_min = np.asarray(action_config.pos_min, dtype=np.float64)
    pos_max = np.asarray(action_config.pos_max, dtype=np.float64)

    t_pos = (pos - pos_min) / (pos_max - pos_min)
    a_pos = np.clip(lo + t_pos * span, lo, hi)

    t_grip = (gripper - action_config.gripper_min) / (action_config.gripper_max - action_config.gripper_min)
    a_grip = np.clip(lo + t_grip * span, lo, hi).reshape(n, 1)

    a_rot6d = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        xyzw = [quat[i, 1], quat[i, 2], quat[i, 3], quat[i, 0]]
        mat = Rotation.from_quat(xyzw).as_matrix()
        a_rot6d[i] = np.concatenate([mat[:, 0], mat[:, 1]])

    return np.concatenate([a_pos, a_rot6d, a_grip], axis=1).astype(np.float32)


class ActionPipeline:
    """
    Converts raw policy output into robot commands.

    Identical code runs in sim and real. The policy always outputs values in
    the clip range (default [-1, 1]).

    Two modes (ActionConfig.mode):
        "delta"    (default): clip then scale -> (num_envs, 7) physical DELTA
                   (dpos(3) + drot(3) + dgripper(1)) for the backend to accumulate.
        "absolute": clip, then linearly map pos/gripper into their physical ranges
                   and Gram-Schmidt the 6D rotation into a quaternion -> (num_envs, 8)
                   physical ABSOLUTE command (pos(3) + quat_wxyz(4) + gripper(1)) for
                   the backend to set directly. The two modes' outputs have different
                   widths (7 vs 8), which is how a backend distinguishes them without
                   needing the mode itself threaded through its constructor.

    Usage:
        pipeline = ActionPipeline(action_config)
        cmd      = pipeline.process(raw_action)   # (num_envs, action_dim)
        space    = pipeline.build_action_space()  # gymnasium.spaces.Box
    """

    def __init__(self, action_config: ActionConfig) -> None:
        action_config.validate()
        self.cfg = action_config
        if action_config.mode == "absolute":
            self._pos_min = np.array(action_config.pos_min, dtype=np.float32)
            self._pos_max = np.array(action_config.pos_max, dtype=np.float32)
            self._gripper_min = float(action_config.gripper_min)
            self._gripper_max = float(action_config.gripper_max)
        else:
            self._scales = np.array(action_config.scales, dtype=np.float32)  # (action_dim,)

    def process(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Args:
            raw_action: (num_envs, action_dim) float32, policy output.

        Returns:
            (num_envs, 7) physical delta, or (num_envs, 8) physical absolute pose —
            see class docstring.
        """
        clipped = np.clip(raw_action, self.cfg.clip[0], self.cfg.clip[1])
        if self.cfg.mode == "absolute":
            return self._process_absolute(clipped)
        return (clipped * self._scales).astype(np.float32)

    def _process_absolute(self, clipped: np.ndarray) -> np.ndarray:
        lo, hi = self.cfg.clip
        span = float(hi - lo)
        pos_raw, rot6d_raw, grip_raw = clipped[:, 0:3], clipped[:, 3:9], clipped[:, 9]

        t_pos = (pos_raw - lo) / span
        pos = self._pos_min + t_pos * (self._pos_max - self._pos_min)

        t_grip = (grip_raw - lo) / span
        grip = self._gripper_min + t_grip * (self._gripper_max - self._gripper_min)

        quat = _rot6d_to_quat(rot6d_raw)   # (num_envs, 4) wxyz

        return np.concatenate([pos, quat, grip[:, None]], axis=1).astype(np.float32)

    def build_action_space(self) -> Box:
        """
        Returns a Box with shape (action_dim,) in the clip range.
        Follows the gymnasium single-env convention (no num_envs dim).
        """
        n = self.cfg.action_dim
        return Box(
            low=np.full(n, self.cfg.clip[0], dtype=np.float32),
            high=np.full(n, self.cfg.clip[1], dtype=np.float32),
            dtype=np.float32,
        )
