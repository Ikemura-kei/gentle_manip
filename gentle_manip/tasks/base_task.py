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
        # If the task names a single object, give the stress reward that object's
        # yield stress so its penalty normalizes to the bruising threshold (the
        # YAML stays material-agnostic). Pure-python registry lookup, no genesis.
        yield_stress = None
        name = task_cfg.get("object_name")
        if name:
            try:
                from gentle_manip.assets.registry import get_object_def
                yield_stress = get_object_def(name).material.von_mises_yield_stress
            except KeyError:
                pass
        # Exposed so PolicyEnv can normalize the privileged stress obs by the same yield.
        self.object_yield_stress = yield_stress
        self._reward_fn: CompositeReward = build_reward_fn(
            task_cfg.get("rewards", {}), material_yield_stress=yield_stress
        )

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
