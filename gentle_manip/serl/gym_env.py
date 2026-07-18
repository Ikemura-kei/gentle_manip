"""Single-env gymnasium adapter over the genesis mushroom-teacher, for HIL-SERL.

SERL's actor drives a *single* (non-vectorized) gymnasium env: reset() -> (obs, info),
step(a) -> (obs, reward, terminated, truncated, info). Genesis needs Python 3.12 and
SERL is JAX/3.10, so they can't share an interpreter — this adapter (3.10, genesis-free)
talks to a genesis server (3.12, scripts/serl_sim_server.py -> PolicyEnv over the pure
socket in gentle_manip.envs.rpc). It squeezes the num_envs=1 batch dim, and owns the
episode boundary (the genesis server runs with NO auto-reset):
  terminated = task success (goal reached, a true terminal -> mask 0, no bootstrap)
  truncated  = horizon reached (time limit -> still bootstrap)

The obs is a dict of state + privileged fields (ee_pos/ee_quat/gripper_width +
priv_object_pos/priv_object_vel/priv_stress). Wrap with SERL's SERLObsWrapper to flatten
it into the "state" key the state-based SACAgent expects.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict

from gentle_manip.envs.rpc import SimEnvClient


class SimGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 5566,
                 max_episode_steps: int = 150, action_dim: int = 7,
                 connect_timeout: float = 240.0) -> None:
        self.client = SimEnvClient(host=host, port=port, connect_timeout=connect_timeout)
        self.max_episode_steps = int(max_episode_steps)
        self._t = 0

        obs = self._squeeze(self.client.reset())
        self.observation_space = Dict({
            k: Box(-np.inf, np.inf, v.shape, np.float32) for k, v in obs.items()
        })
        # Policy output space: normalized [-1, 1] delta-pose (6) + gripper (1).
        self.action_space = Box(-1.0, 1.0, (int(action_dim),), np.float32)

    @staticmethod
    def _squeeze(obs: dict) -> dict:
        """Drop the leading num_envs=1 dim: (1, ...) -> (...)."""
        return {k: np.asarray(v)[0].astype(np.float32) for k, v in obs.items()}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.client.reseed(int(seed))
        self._t = 0
        return self._squeeze(self.client.reset()), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(1, -1)      # (1, action_dim)
        obs, reward, _done, info = self.client.step(a)
        self._t += 1
        reward = float(np.asarray(reward).ravel()[0])
        success = bool(info[0].get("success", False))
        terminated = success                                        # true terminal
        truncated = self._t >= self.max_episode_steps               # time limit
        return self._squeeze(obs), reward, terminated, truncated, {"succeed": success}

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
