from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from gentle_manip.robot.xarm7_config import DEFAULT_GRIPPER_WIDTH, EE_BOUNDS_MAX, EE_BOUNDS_MIN


@dataclass
class ActionConfig:
    """
    Configuration for ActionPipeline.

    mode: "delta" (default) or "absolute".
        - "delta": raw policy output (7-dim: dx,dy,dz,droll,dpitch,dyaw,dgripper) is
          clipped then scaled — ActionPipeline.process() returns a physical DELTA that
          the backend accumulates onto its running target pose. This is the ORIGINAL
          (and still default) representation — every existing config/script that omits
          `mode` gets exactly this behaviour, unchanged.
        - "absolute": raw policy output is 10-dim (3 position + 6 rot6d + 1 gripper).
          ActionPipeline.process() maps position/gripper linearly into
          [pos_min,pos_max]/[gripper_min,gripper_max] and Gram-Schmidt-orthonormalizes
          the 6D rotation (Zhou et al. 2019) into a quaternion, returning an 8-dim
          ABSOLUTE (pos(3) + quat_wxyz(4) + gripper(1)) command that the backend sets
          directly (no accumulation). The two modes produce different-width outputs
          (7 vs 8) so a backend can dispatch on output width without needing to know
          the mode itself.

    scales: (delta mode only) per-dimension multipliers applied after clipping.
            Length determines the action dimensionality.
            Default matches XArm7: 6D delta pose + 1D gripper.
    clip:   (min, max) applied to raw policy output before scaling/mapping.

    pos_min / pos_max: (absolute mode only) workspace bounds (meters) the raw [-1,1]
        (or custom `clip` range) position dims are linearly mapped into. Default
        mirrors the XArm7 EE workspace (xarm7_config.EE_BOUNDS_MIN/MAX).
    gripper_min / gripper_max: (absolute mode only) physical gripper width range
        (meters) the raw gripper dim is linearly mapped into. Default mirrors
        xarm7_config.DEFAULT_GRIPPER_WIDTH.

    Loaded from configs/action/*.yaml via ActionConfig.from_dict().
    """
    mode: str = "delta"

    scales: List[float] = field(default_factory=lambda: [
        0.0052, 0.0052, 0.006,   # delta x, y, z  (meters)
        0.001,  0.001,  0.001,   # delta roll, pitch, yaw  (radians)
        0.05,                    # delta gripper width  (meters)
    ])
    clip: Tuple[float, float] = (-1.0, 1.0)

    pos_min: Tuple[float, float, float] = tuple(EE_BOUNDS_MIN)
    pos_max: Tuple[float, float, float] = tuple(EE_BOUNDS_MAX)
    gripper_min: float = 0.0
    gripper_max: float = DEFAULT_GRIPPER_WIDTH

    # Absolute-mode rotation encoding: "rot6d" (default, 10-dim action: 3 pos + 6 rot6d + 1 grip;
    # continuous, singularity-free) or "euler" (7-dim: 3 pos + 3 euler + 1 grip — the RPY convention
    # most papers use; compact but has gimbal-lock/wraparound). Both process to the SAME 8-dim
    # physical (pos+quat+gripper) command, so the backend dispatch (7=delta, 8=absolute) is unchanged.
    rot_repr: str = "rot6d"
    euler_min: Tuple[float, float, float] = (-math.pi, -math.pi, -math.pi)  # euler mode: raw [-1,1] -> [min,max] rad
    euler_max: Tuple[float, float, float] = (math.pi, math.pi, math.pi)
    euler_seq: str = "xyz"   # scipy Rotation from_euler/as_euler sequence (intrinsic xyz ~ RPY)

    @property
    def action_dim(self) -> int:
        if self.mode == "absolute":
            return 7 if self.rot_repr == "euler" else 10   # 3 pos + (3 euler | 6 rot6d) + 1 gripper
        return len(self.scales)

    def validate(self) -> None:
        if self.mode not in ("delta", "absolute"):
            raise ValueError(f"mode must be 'delta' or 'absolute', got {self.mode!r}")
        if self.clip[0] >= self.clip[1]:
            raise ValueError(f"clip min must be < max, got {self.clip}")
        if self.mode == "delta":
            if len(self.scales) == 0:
                raise ValueError("scales must not be empty")
        else:
            if len(self.pos_min) != 3 or len(self.pos_max) != 3:
                raise ValueError("pos_min/pos_max must be length 3")
            if any(lo >= hi for lo, hi in zip(self.pos_min, self.pos_max)):
                raise ValueError(f"pos_min must be < pos_max elementwise, got "
                                 f"{self.pos_min} / {self.pos_max}")
            if self.gripper_min >= self.gripper_max:
                raise ValueError(f"gripper_min must be < gripper_max, got "
                                 f"{self.gripper_min} / {self.gripper_max}")
            if self.rot_repr not in ("rot6d", "euler"):
                raise ValueError(f"rot_repr must be 'rot6d' or 'euler', got {self.rot_repr!r}")
            if self.rot_repr == "euler" and any(
                lo >= hi for lo, hi in zip(self.euler_min, self.euler_max)):
                raise ValueError(f"euler_min must be < euler_max elementwise, got "
                                 f"{self.euler_min} / {self.euler_max}")

    @classmethod
    def from_dict(cls, d: dict) -> ActionConfig:
        mode = d.get("mode", "delta")
        kwargs = dict(
            mode=mode,
            clip=tuple(d.get("clip", (-1.0, 1.0))),
        )
        if mode == "absolute":
            kwargs["pos_min"] = tuple(d.get("pos_min", cls.pos_min))
            kwargs["pos_max"] = tuple(d.get("pos_max", cls.pos_max))
            kwargs["gripper_min"] = float(d.get("gripper_min", cls.gripper_min))
            kwargs["gripper_max"] = float(d.get("gripper_max", cls.gripper_max))
            kwargs["rot_repr"] = d.get("rot_repr", "rot6d")
            kwargs["euler_min"] = tuple(d.get("euler_min", cls.euler_min))
            kwargs["euler_max"] = tuple(d.get("euler_max", cls.euler_max))
            kwargs["euler_seq"] = d.get("euler_seq", cls.euler_seq)
        else:
            kwargs["scales"] = d.get("scales", cls.__dataclass_fields__["scales"].default_factory())
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg
