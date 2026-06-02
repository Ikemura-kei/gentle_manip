from __future__ import annotations

from typing import List

import numpy as np

from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.envs.raw_obs import RawObs


class DistToObjReward:
    """exp(-decay * ||ee_pos - object_center||) — draws EE toward object."""

    def __init__(self, scale: float = 1.0, decay: float = 20.0) -> None:
        self.scale = scale
        self.decay = decay

    def reset(self, sim_feedback: SimFeedback) -> None:
        pass

    def __call__(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        dist = np.linalg.norm(raw_obs.ee_pos - sim_feedback.object_center, axis=-1)
        return np.exp(-self.decay * dist) * self.scale


class DistToGoalReward:
    """exp(-decay * ||object_center - goal_pos||) — draws object toward a fixed goal."""

    def __init__(
        self,
        goal_pos: List[float],
        scale: float = 1.0,
        decay: float = 20.0,
    ) -> None:
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.scale = scale
        self.decay = decay

    def reset(self, sim_feedback: SimFeedback) -> None:
        pass

    def __call__(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        dist = np.linalg.norm(sim_feedback.object_center - self.goal_pos, axis=-1)
        return np.exp(-self.decay * dist) * self.scale
