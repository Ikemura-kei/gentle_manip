from __future__ import annotations

import numpy as np

from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.rewards.distance import DistToGoalReward, DistToObjReward
from gentle_manip.rewards.lift import LiftReward
from gentle_manip.rewards.placement import PlacementReward
from gentle_manip.rewards.stress import StressReward

_REWARD_MAP = {
    "stress": StressReward,
    "dist_to_obj": DistToObjReward,
    "dist_to_goal": DistToGoalReward,
    "lift": LiftReward,
    "placement": PlacementReward,
}


class CompositeReward:
    def __init__(self, components: list) -> None:
        self.components = components

    def reset(self, sim_feedback: SimFeedback) -> None:
        for c in self.components:
            c.reset(sim_feedback)

    def __call__(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        num_envs = sim_feedback.object_center.shape[0]
        total = np.zeros(num_envs, dtype=np.float32)
        for c in self.components:
            total = total + c(sim_feedback, raw_obs)
        return total


def build_reward_fn(config: dict) -> CompositeReward:
    """Build a CompositeReward from a YAML rewards dict.

    Example config:
        stress:    {scale: 0.001, cap: 14000.0, ...}
        dist_to_obj: {scale: 1.0, decay: 20.0}
        lift:      {scale: 1.0, grasp_gate_dist: 0.079}
    """
    components = []
    for name, params in config.items():
        if name not in _REWARD_MAP:
            raise ValueError(
                f"Unknown reward component {name!r}. Available: {sorted(_REWARD_MAP)}"
            )
        components.append(_REWARD_MAP[name](**params))
    return CompositeReward(components)
