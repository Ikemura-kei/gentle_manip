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
