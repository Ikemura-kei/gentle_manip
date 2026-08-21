"""Derive an action set for a given ActionConfig from a demo's recorded EE-pose trajectory.

One delta-teleop collection records the EE-pose trajectory (ee_pos, ee_quat/ee_rot6d, gripper_width)
in its obs; both a DELTA and an ABSOLUTE (rot6d or 7d-euler) action set are just different
parameterizations of "go to the next observed pose", so both are derivable here with the shared
inverters (invert_delta_action / invert_absolute_action). Used by BOTH the DPPO converter
(convert_demos) and the DP3 converter (convert_demo_to_dp3), so the two stay identical.
"""
from __future__ import annotations

import numpy as np


def obs_quat(o: dict, T: int) -> np.ndarray:
    """(T,4) wxyz EE quats from a demo's obs, whether it stored ee_quat or ee_rot6d."""
    if "ee_quat" in o:
        return np.asarray(o["ee_quat"], np.float64).reshape(T, 4)
    from scipy.spatial.transform import Rotation as R
    r6 = np.asarray(o["ee_rot6d"], np.float64).reshape(T, 6)
    a1 = r6[:, :3] / np.linalg.norm(r6[:, :3], axis=1, keepdims=True)
    a2 = r6[:, 3:6] - np.sum(a1 * r6[:, 3:6], axis=1, keepdims=True) * a1
    a2 /= np.linalg.norm(a2, axis=1, keepdims=True)
    M = np.stack([a1, a2, np.cross(a1, a2)], axis=2)
    return R.from_matrix(M).as_quat()[:, [3, 0, 1, 2]]


def derive_action_set(ep: dict, action_config, lookahead: int = 1) -> np.ndarray:
    """Action set (T, action_dim) for `action_config`, derived from the demo's EE-pose trajectory
    (target for step t = the observed pose `lookahead` steps ahead, held on the last step). Delta
    vs absolute (+ rot_repr) is selected by action_config. Matches what an ActionPipeline maps
    back to the pose.

    lookahead (ABSOLUTE mode only, default 1): how many steps ahead the target pose is. K=1
    (next frame) makes the target lead the current pose by only one servo step (~2-3 mm) —
    measured to be BELOW the closed-loop stability threshold for a BC'd absolute policy: the
    conditional mean of "next pose | current pose" is ~the current pose (mean pull ≈ 0.0 mm at
    every height in the derived data), so a deterministic diffusion rollout stalls at a fixed
    point just above the object (observed: zwiex/qvwdj/oppsu never command below z≈0.034, ~0%
    success). Recorded COMMANDED targets (jfhlu, 0.9 success) lead the achieved pose by 5-10 mm
    (controller lag); K≈4 restores an equivalent lead from an achieved-pose trajectory. Delta
    mode ignores lookahead (a per-step difference over K steps would just saturate the scales) —
    passing lookahead>1 with a delta config is an error to keep datasets unambiguous."""
    from gentle_manip.actions.pipeline import invert_absolute_action, invert_delta_action
    o = ep["observations"]
    T = len(ep["actions"])
    pos = np.asarray(o["ee_pos"], np.float64).reshape(T, 3)
    quat = obs_quat(o, T)
    grip = np.asarray(o["gripper_width"], np.float64).reshape(T, -1)[:, 0]
    if action_config.mode == "absolute":
        nxt = np.minimum(np.arange(T) + int(lookahead), T - 1)
        tp, tq, tg = pos[nxt], quat[nxt], grip[nxt]
        return invert_absolute_action(tp, tq, tg, action_config)            # (T, 10 rot6d | 7 euler)
    if int(lookahead) != 1:
        raise ValueError(f"lookahead={lookahead} is only valid for absolute mode; "
                         "delta derivation is defined per-step (K=1)")
    nxt = np.minimum(np.arange(T) + 1, T - 1)
    tp, tq, tg = pos[nxt], quat[nxt], grip[nxt]
    return invert_delta_action(pos, quat, grip, tp, tq, tg, action_config)  # (T, 7) delta
