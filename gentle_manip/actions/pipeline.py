from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.actions.action_config import ActionConfig

# gymnasium is imported lazily (only build_action_space needs it) so the pure-numpy inverters
# invert_absolute_action / invert_delta_action (used by convert_demos/convert_demo_to_dp3 via
# actions.derive) work in envs that don't ship gymnasium (e.g. envs/dppo).


def _euler_frame_offset(action_config: ActionConfig):
    """Fixed reference-frame Rotation for euler encoding, or None. See ActionConfig — the
    euler angles encode R_rel = R_cmd * R_off^-1 (decode: R_cmd = R_rel * R_off), moving a
    top-down grasp's roll off the +/-pi wraparound seam. MUST be applied identically in
    invert_absolute_action (encode) and ActionPipeline._process_absolute (decode)."""
    off = getattr(action_config, "euler_frame_offset_deg", None)
    if off is None:
        return None
    return Rotation.from_euler(action_config.euler_seq, off, degrees=True)


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
    gripper) command, compute the raw [-1,1] action (N,10 for rot_repr='rot6d', N,7 for
    'euler') that an absolute-mode ActionPipeline built from `action_config` would map back
    to it. No history needed (absolute mode has no accumulation — each step is an independent
    forward transform), so one recorded EE-pose trajectory can be inverted into EITHER an
    absolute action set (this) or a delta action set, deriving both from the same demos.

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

    xyzw = quat[:, [1, 2, 3, 0]]                                # wxyz -> xyzw
    if getattr(action_config, "rot_repr", "rot6d") == "euler":  # -> 3 euler angles, un-mapped to [-1,1]
        emin = np.asarray(action_config.euler_min, dtype=np.float64)
        emax = np.asarray(action_config.euler_max, dtype=np.float64)
        rot = Rotation.from_quat(xyzw)
        off = _euler_frame_offset(action_config)
        if off is not None:                                     # encode RELATIVE to the offset frame
            rot = rot * off.inv()
        angles = rot.as_euler(action_config.euler_seq, degrees=False)                        # (n,3)
        a_rot = np.clip(lo + (angles - emin) / (emax - emin) * span, lo, hi)                 # (n,3)
    else:                                                       # -> 6D rotation (first two R columns)
        a_rot = np.zeros((n, 6), dtype=np.float64)
        for i in range(n):
            mat = Rotation.from_quat(xyzw[i]).as_matrix()
            a_rot[i] = np.concatenate([mat[:, 0], mat[:, 1]])

    return np.concatenate([a_pos, a_rot, a_grip], axis=1).astype(np.float32)


def invert_delta_action(prev_pos: np.ndarray, prev_quat: np.ndarray, prev_grip: np.ndarray,
                        cur_pos: np.ndarray, cur_quat: np.ndarray, cur_grip: np.ndarray,
                        action_config: ActionConfig) -> np.ndarray:
    """Delta action (N,7) that DELTA-mode ActionPipeline + backend accumulation maps from `prev`
    pose to `cur` pose — the inverse of the backend's world-frame update R_new = from_rotvec(drot)*
    R_prev (and pos/grip += delta). Lets a recorded EE-pose trajectory be turned into a DELTA action
    set, so the delta twin of an absolute dataset is derived from the SAME demos. wxyz quats in.

    Mirrors the grasp-synth collector's _invert_actions exactly (world-frame rotvec delta), just
    shared + batched so real-teleop and sim-synth convert use one code path.
    """
    prev_pos = np.asarray(prev_pos, np.float64).reshape(-1, 3)
    cur_pos = np.asarray(cur_pos, np.float64).reshape(-1, 3)
    prev_grip = np.asarray(prev_grip, np.float64).reshape(-1)
    cur_grip = np.asarray(cur_grip, np.float64).reshape(-1)
    scales = np.asarray(action_config.scales, np.float64)
    lo, hi = action_config.clip

    a_pos = np.clip((cur_pos - prev_pos) / scales[:3], lo, hi)
    a_grip = np.clip((cur_grip - prev_grip).reshape(-1, 1) / scales[6], lo, hi)
    pq = np.asarray(prev_quat, np.float64).reshape(-1, 4)[:, [1, 2, 3, 0]]   # wxyz -> xyzw
    cq = np.asarray(cur_quat, np.float64).reshape(-1, 4)[:, [1, 2, 3, 0]]
    rotvec = (Rotation.from_quat(cq) * Rotation.from_quat(pq).inv()).as_rotvec()  # world-frame (N,3)
    a_rot = np.clip(rotvec / scales[3:6], lo, hi)
    return np.concatenate([a_pos, a_rot, a_grip], axis=1).astype(np.float32)


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
        pos_raw = clipped[:, 0:3]
        t_pos = (pos_raw - lo) / span
        pos = self._pos_min + t_pos * (self._pos_max - self._pos_min)

        if self.cfg.rot_repr == "euler":                       # 7-dim: pos3 + euler3 + grip1
            euler_raw, grip_raw = clipped[:, 3:6], clipped[:, 6]
            emin = np.asarray(self.cfg.euler_min, np.float32)
            emax = np.asarray(self.cfg.euler_max, np.float32)
            angles = emin + (euler_raw - lo) / span * (emax - emin)          # (num_envs, 3) rad
            rot = Rotation.from_euler(self.cfg.euler_seq, angles, degrees=False)
            off = _euler_frame_offset(self.cfg)
            if off is not None:                                # angles were RELATIVE to the offset frame
                rot = rot * off
            xyzw = rot.as_quat()
            quat = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)        # -> wxyz
        else:                                                  # 10-dim: pos3 + rot6d6 + grip1
            grip_raw = clipped[:, 9]
            quat = _rot6d_to_quat(clipped[:, 3:9])                            # (num_envs, 4) wxyz

        t_grip = (grip_raw - lo) / span
        grip = self._gripper_min + t_grip * (self._gripper_max - self._gripper_min)
        return np.concatenate([pos, quat, grip[:, None]], axis=1).astype(np.float32)

    def build_action_space(self) -> "gymnasium.spaces.Box":
        """
        Returns a Box with shape (action_dim,) in the clip range.
        Follows the gymnasium single-env convention (no num_envs dim).
        """
        from gymnasium.spaces import Box                 # lazy: only this method needs gymnasium
        n = self.cfg.action_dim
        return Box(
            low=np.full(n, self.cfg.clip[0], dtype=np.float32),
            high=np.full(n, self.cfg.clip[1], dtype=np.float32),
            dtype=np.float32,
        )
