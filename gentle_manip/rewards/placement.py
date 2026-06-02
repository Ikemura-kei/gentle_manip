from __future__ import annotations

import numpy as np

from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.envs.raw_obs import RawObs


class PlacementReward:
    """Penalises placing an object from too high above the target z.

    reward = -clip(object_z - (goal_z + tolerance), 0, ∞) * scale

    Pressing penalty and impact-force penalty require additional
    SimFeedback.extra keys (e.g. "contact_force") not yet defined —
    extend this class when those fields are available.
    """

    def __init__(
        self,
        goal_z: float = 0.0,
        tolerance: float = 0.02,
        scale: float = 1.0,
    ) -> None:
        self.goal_z = goal_z
        self.tolerance = tolerance
        self.scale = scale

    def reset(self, sim_feedback: SimFeedback) -> None:
        pass

    def __call__(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        height_error = np.clip(
            sim_feedback.object_center[:, 2] - (self.goal_z + self.tolerance),
            0.0,
            None,
        )
        return -height_error * self.scale
