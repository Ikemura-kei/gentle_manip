from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.rewards import CompositeReward, build_reward_fn
from gentle_manip.scenes.scene_spec import SceneSpec


class BaseTask(ABC):
    """Abstract base for all tasks.

    Subclasses must implement:
        scene_spec  — declarative scene description used by SceneBuilder
        is_success  — per-env success condition (num_envs,) bool

    compute_reward calls is_success internally and adds the success bonus,
    so callers only need compute_reward + whatever task-level reset is needed.
    """

    def __init__(self, task_cfg: dict) -> None:
        self.success_scale = float(task_cfg.get("success_scale", 2.0))
        self._reward_fn: CompositeReward = build_reward_fn(task_cfg.get("rewards", {}))

    @property
    @abstractmethod
    def scene_spec(self) -> SceneSpec:
        ...

    def reset(self, sim_feedback: SimFeedback) -> None:
        """Called at episode start.  Propagates to stateful reward components."""
        self._reward_fn.reset(sim_feedback)

    def compute_reward(
        self, sim_feedback: SimFeedback, raw_obs: RawObs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (rewards, success) both shaped (num_envs,).

        Success is computed here (task knows the condition, sim does not) and
        added as a sparse bonus on top of the shaped reward components.
        """
        success = self.is_success(sim_feedback, raw_obs)
        reward = self._reward_fn(sim_feedback, raw_obs)
        reward = reward + success.astype(np.float32) * self.success_scale
        return reward, success

    @abstractmethod
    def is_success(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        """Returns bool array (num_envs,) — True when episode goal is met."""
        ...
