from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ActionConfig:
    """
    Configuration for ActionPipeline.

    scales: per-dimension multipliers applied after clipping.
            Length determines the action dimensionality.
            Default matches XArm7: 6D delta pose + 1D gripper.
    clip:   (min, max) applied to raw policy output before scaling.

    Loaded from configs/action/*.yaml via ActionConfig.from_dict().
    """
    scales: List[float] = field(default_factory=lambda: [
        0.0052, 0.0052, 0.006,   # delta x, y, z  (meters)
        0.001,  0.001,  0.001,   # delta roll, pitch, yaw  (radians)
        0.05,                    # delta gripper width  (meters)
    ])
    clip: Tuple[float, float] = (-1.0, 1.0)

    @property
    def action_dim(self) -> int:
        return len(self.scales)

    def validate(self) -> None:
        if len(self.scales) == 0:
            raise ValueError("scales must not be empty")
        if self.clip[0] >= self.clip[1]:
            raise ValueError(f"clip min must be < max, got {self.clip}")

    @classmethod
    def from_dict(cls, d: dict) -> ActionConfig:
        cfg = cls(
            scales=d.get("scales", cls.__dataclass_fields__["scales"].default_factory()),
            clip=tuple(d.get("clip", (-1.0, 1.0))),
        )
        cfg.validate()
        return cfg
