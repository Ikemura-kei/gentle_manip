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


def commanded_pose_set(ep: dict, source_action_config) -> tuple | None:
    """(pos, quat, grip) COMMANDED targets decoded from the demo's own recorded ABSOLUTE actions,
    or None when the recording is delta-mode (no absolute target stream to decode)."""
    if getattr(source_action_config, "mode", None) != "absolute":
        return None
    from gentle_manip.actions.pipeline import ActionPipeline
    acts = np.asarray(ep["actions"], np.float32)
    cmd = ActionPipeline(source_action_config).process(acts)      # (T, 8) pos+quat_wxyz+grip
    return cmd[:, :3].astype(np.float64), cmd[:, 3:7].astype(np.float64), cmd[:, 7].astype(np.float64)


def derive_action_set(ep: dict, action_config, source_action_config=None) -> np.ndarray:
    """Action set (T, action_dim) for `action_config`, derived from the demo. Delta vs absolute
    (+ rot_repr) is selected by action_config; matches what an ActionPipeline maps back to the pose.

    SOURCE OF THE TARGET POSES — this choice decides whether the derived policy attenuates:

    * `source_action_config` given AND the recording is absolute-mode: re-encode the demo's own
      COMMANDED targets (decode its native actions, encode in the new representation). Exact — the
      derived actions command precisely what the demonstrator commanded.
    * fallback (delta-mode recordings, e.g. teleop): the MEASURED pose trajectory, target for step
      t = the NEXT observed pose. ⚠️ The measured pose trails the commanded target by the
      controller's tracking gap (v6 demos: 6.5mm mean, 16.5mm p95, 47mm max). A policy cloned from
      measured-pose targets commands where the arm WAS; in closed loop the controller then lags
      behind THAT, and the executed trajectory attenuates step over step. This is the mechanism
      that put every derived-7d-euler policy at 0 success while their rot6d twins (native
      commanded actions) were fine — and open-loop replay validation cannot see it.
    """
    from gentle_manip.actions.pipeline import invert_absolute_action, invert_delta_action
    T = len(ep["actions"])
    cmd = commanded_pose_set(ep, source_action_config) if source_action_config is not None else None
    if cmd is not None:
        tp, tq, tg = cmd                                   # commanded targets, step-aligned
        if action_config.mode == "absolute":
            return invert_absolute_action(tp, tq, tg, action_config)
        # delta from commanded: prev = previous commanded target (the accumulation reference)
        pp = np.vstack([tp[:1], tp[:-1]]); pq = np.vstack([tq[:1], tq[:-1]])
        pg = np.concatenate([tg[:1], tg[:-1]])
        return invert_delta_action(pp, pq, pg, tp, tq, tg, action_config)
    o = ep["observations"]
    pos = np.asarray(o["ee_pos"], np.float64).reshape(T, 3)
    quat = obs_quat(o, T)
    grip = np.asarray(o["gripper_width"], np.float64).reshape(T, -1)[:, 0]
    nxt = np.minimum(np.arange(T) + 1, T - 1)
    tp, tq, tg = pos[nxt], quat[nxt], grip[nxt]
    if action_config.mode == "absolute":
        return invert_absolute_action(tp, tq, tg, action_config)            # (T, 10 rot6d | 7 euler)
    return invert_delta_action(pos, quat, grip, tp, tq, tg, action_config)  # (T, 7) delta
