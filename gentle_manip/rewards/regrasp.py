from __future__ import annotations

import numpy as np

from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.envs.raw_obs import RawObs


class RegraspReward:
    """Dense shaping bonus for re-approaching and re-closing the gripper after a
    failed first grasp attempt -- an RL-only counterpart to BC, which has no gradient
    that can push a policy off a hover/freeze local optimum at the retry decision
    point (banana regrasp-hover fix; TIDE and ReTVL BC-side fixes both failed to
    produce genuine second-attempt regrasps, see docs/cross_category_specialist_log.md).

    Per-env FSM on the observed (ee_pos, object_center, gripper_width) trajectory:
      1. "attempted" latches once the gripper closes near the object (an attempt).
      2. "armed" latches once an attempted env's gripper reopens without success --
         this is the exact state a hovering policy gets stuck in and never leaves.
      3. While armed, reward = scale * (approach progress + closing progress) this
         step -- a dense, uncapped-cumulative-but-per-step-bounded incentive to
         actually move back down and re-close, not just receive a one-off bonus.
    Once armed, stays armed for the rest of the episode (a later successful lift is
    still rewarded here on top of the task's own lift/success bonuses -- redundant
    reward for the desired behavior is harmless, since this term only ever fires
    for POSITIVE progress).
    """

    def __init__(self, scale: float = 1.0, grasp_gate_dist: float = 0.079,
                close_width_thresh: float = 0.03, reopen_width_thresh: float = 0.07) -> None:
        self.scale = scale
        self.grasp_gate_dist = grasp_gate_dist
        self.close_width_thresh = close_width_thresh
        self.reopen_width_thresh = reopen_width_thresh
        self._attempted: np.ndarray | None = None
        self._armed: np.ndarray | None = None
        self._prev_dist: np.ndarray | None = None
        self._prev_width: np.ndarray | None = None

    def reset(self, sim_feedback: SimFeedback) -> None:
        n = sim_feedback.object_center.shape[0]
        self._attempted = np.zeros(n, dtype=bool)
        self._armed = np.zeros(n, dtype=bool)
        self._prev_dist = None
        self._prev_width = None

    def __call__(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        n = sim_feedback.object_center.shape[0]
        dist = np.linalg.norm(raw_obs.ee_pos - sim_feedback.object_center, axis=-1)
        width = np.asarray(raw_obs.gripper_width).reshape(n)

        closed_near = (dist < self.grasp_gate_dist) & (width < self.close_width_thresh)
        self._attempted = self._attempted | closed_near

        reopened = width > self.reopen_width_thresh
        self._armed = self._armed | (self._attempted & reopened)

        reward = np.zeros(n, dtype=np.float32)
        if self._prev_dist is not None:
            approach_progress = np.clip(self._prev_dist - dist, 0.0, None)
            close_progress = np.clip(self._prev_width - width, 0.0, None)
            reward = self._armed.astype(np.float32) * (approach_progress + close_progress) * self.scale

        self._prev_dist = dist
        self._prev_width = width
        return reward
