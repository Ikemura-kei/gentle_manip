from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class SimFeedback:
    """Structured feedback from Genesis — sim-only, never crosses the real backend.

    All array fields carry a leading num_envs dimension to match the RawObs
    batching convention.

    Universal fields (always present regardless of object type or task):
      ee_pos, gripper_width, object_center

    Task- and object-specific data goes in `extra`.  Examples:
      extra["von_mises_stress"]   # (num_envs, n_particles) — soft bodies only
      extra["particle_positions"] # (num_envs, n_particles, 3) — soft bodies only
      extra["contact_force"]      # (num_envs,) — rigid surrogates
      extra["success"]            # (num_envs,) bool — set by task before reward call
    """

    ee_pos: np.ndarray        # (num_envs, 3)   end-effector world position
    gripper_width: np.ndarray # (num_envs,)     metres
    object_center: np.ndarray # (num_envs, 3)   representative object position

    extra: Dict[str, Any] = field(default_factory=dict)
