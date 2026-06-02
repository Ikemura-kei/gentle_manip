from __future__ import annotations

import numpy as np

from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.envs.raw_obs import RawObs


class LiftReward:
    """Rewards upward object movement, gated on the EE being close enough to grasp.

    reward = clip(object_z - initial_z, 0, ∞) * grasp_gate * scale

    grasp_gate is 1 when ||ee_pos - object_center|| < grasp_gate_dist, else 0.
    This prevents rewarding the object floating up without being grasped.
    """

    def __init__(self, scale: float = 1.0, grasp_gate_dist: float = 0.079) -> None:
        self.scale = scale
        self.grasp_gate_dist = grasp_gate_dist
        self._initial_z: np.ndarray | None = None

    def reset(self, sim_feedback: SimFeedback) -> None:
        self._initial_z = sim_feedback.object_center[:, 2].copy()

    def __call__(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        if self._initial_z is None:
            return np.zeros(sim_feedback.object_center.shape[0], dtype=np.float32)

        dist = np.linalg.norm(raw_obs.ee_pos - sim_feedback.object_center, axis=-1)
        grasp_gate = (dist < self.grasp_gate_dist).astype(np.float32)

        lift_progress = sim_feedback.object_center[:, 2] - self._initial_z
        return np.clip(lift_progress, 0.0, None) * grasp_gate * self.scale
