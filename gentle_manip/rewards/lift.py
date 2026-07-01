from __future__ import annotations

import numpy as np

from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.envs.raw_obs import RawObs


class LiftReward:
    """Rewards upward object movement, gated on the EE being close enough to grasp.

    reward = norm(object_z - initial_z) * grasp_gate * scale

    grasp_gate is 1 when ||ee_pos - object_center|| < grasp_gate_dist, else 0 — this
    prevents rewarding the object floating up without being grasped.

    ``lift_target`` (m): if set, the rise is normalized to [0, 1] by it —
    ``clip(progress / lift_target, 0, 1)`` — so a fully-lifted object gives ~scale (a
    real gradient, comparable to the other terms) with no runaway reward for
    over-lifting. Set it near the target lift height (e.g. success-band bottom minus the
    rest z). If None, the legacy raw-metres ``clip(progress, 0, inf)`` is used.
    """

    def __init__(self, scale: float = 1.0, grasp_gate_dist: float = 0.079,
                 lift_target: float | None = None) -> None:
        self.scale = scale
        self.grasp_gate_dist = grasp_gate_dist
        self.lift_target = lift_target
        self._initial_z: np.ndarray | None = None

    def reset(self, sim_feedback: SimFeedback) -> None:
        self._initial_z = sim_feedback.object_center[:, 2].copy()

    def __call__(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        if self._initial_z is None:
            return np.zeros(sim_feedback.object_center.shape[0], dtype=np.float32)

        dist = np.linalg.norm(raw_obs.ee_pos - sim_feedback.object_center, axis=-1)
        grasp_gate = (dist < self.grasp_gate_dist).astype(np.float32)

        lift_progress = sim_feedback.object_center[:, 2] - self._initial_z
        if self.lift_target is not None:
            lift = np.clip(lift_progress / self.lift_target, 0.0, 1.0)
        else:
            lift = np.clip(lift_progress, 0.0, None)
        return lift * grasp_gate * self.scale
