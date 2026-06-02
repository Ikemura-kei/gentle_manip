from __future__ import annotations

import numpy as np
import gymnasium

from typing import Any


class FlattenObsWrapper:
    """Flattens a gymnasium Dict observation space into a single float32 vector.

    Key order is alphabetically sorted for determinism.  Wraps any env with:
        num_envs: int
        observation_space: gymnasium.spaces.Dict
        action_space: gymnasium.spaces.Box
        reset(**kwargs) -> dict
        step(actions) -> (dict, np.ndarray, np.ndarray, list)
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self._keys: list[str] = sorted(env.observation_space.spaces.keys())
        flat_size = sum(
            int(np.prod(env.observation_space.spaces[k].shape))
            for k in self._keys
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
        )

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs) -> np.ndarray:
        return self._flatten(self.env.reset(**kwargs))

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        obs, rew, done, info = self.env.step(actions)
        return self._flatten(obs), rew, done, info

    def _flatten(self, obs_dict: dict) -> np.ndarray:
        parts = [
            obs_dict[k].reshape(self.env.num_envs, -1).astype(np.float32)
            for k in self._keys
        ]
        return np.concatenate(parts, axis=-1)
