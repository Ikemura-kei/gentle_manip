from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


class RslRlVecEnvWrapper:
    """Adapts a flat-obs numpy env to the RSL-RL VecEnv interface.

    Expected input env interface (e.g. FlattenObsWrapper wrapping PolicyEnv):
        num_envs: int
        observation_space: gymnasium.spaces.Box  (1-D flat)
        action_space: gymnasium.spaces.Box
        reset(**kwargs) -> np.ndarray            (num_envs, num_obs)
        step(np.ndarray)  -> (obs, rew, done, info)

    RSL-RL calls step(actions) and expects:
        (obs, privileged_obs, rewards, dones, infos)
    where obs and rewards are torch tensors on `device`.
    privileged_obs is None (not used here).
    """

    def __init__(self, env: Any, device: str = "cuda") -> None:
        if torch is None:
            raise ImportError("torch is required for RslRlVecEnvWrapper")
        self.env = env
        self.device = device

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def num_obs(self) -> int:
        return int(np.prod(self.env.observation_space.shape))

    @property
    def num_privileged_obs(self) -> Optional[int]:
        return None

    @property
    def num_actions(self) -> int:
        return int(np.prod(self.env.action_space.shape))

    def reset(self) -> tuple[torch.Tensor, None]:
        obs = self.env.reset()
        return self._t(obs), None

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, list]:
        obs, rew, done, info = self.env.step(actions.cpu().numpy())
        return (
            self._t(obs),
            None,
            self._t(rew),
            self._t(done, dtype=torch.bool),
            info,
        )

    def get_observations(self) -> tuple[torch.Tensor, None]:
        obs = self.env.reset()
        return self._t(obs), None

    def _t(self, arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor(arr, dtype=dtype, device=self.device)
