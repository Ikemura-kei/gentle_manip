from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium.spaces import Box

from gentle_manip.actions.action_config import ActionConfig


class ActionPipeline:
    """
    Converts raw policy output into scaled robot commands.

    Identical code runs in sim and real. The policy always outputs values in
    the clip range (default [-1, 1]); this pipeline clips then scales to
    physical units (meters, radians).

    Usage:
        pipeline = ActionPipeline(action_config)
        cmd      = pipeline.process(raw_action)   # (num_envs, action_dim)
        space    = pipeline.build_action_space()  # gymnasium.spaces.Box
    """

    def __init__(self, action_config: ActionConfig) -> None:
        action_config.validate()
        self.cfg = action_config
        self._scales = np.array(action_config.scales, dtype=np.float32)  # (action_dim,)

    def process(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Args:
            raw_action: (num_envs, action_dim) float32, policy output.

        Returns:
            (num_envs, action_dim) float32 scaled robot command.
        """
        clipped = np.clip(raw_action, self.cfg.clip[0], self.cfg.clip[1])
        return (clipped * self._scales).astype(np.float32)

    def build_action_space(self) -> Box:
        """
        Returns a Box with shape (action_dim,) in the clip range.
        Follows the gymnasium single-env convention (no num_envs dim).
        """
        n = self.cfg.action_dim
        return Box(
            low=np.full(n, self.cfg.clip[0], dtype=np.float32),
            high=np.full(n, self.cfg.clip[1], dtype=np.float32),
            dtype=np.float32,
        )
