"""Domain randomization config — what to randomize in the sim and over what ranges.

DR is sim-only (the real side is never randomized). Knobs split by cost:
  - per-reset (cheap): object pose — sampled every reset, no scene rebuild.
  - per-scene (expensive): object material (E/nu/rho/yield) and coupling friction —
    MPM material is GLOBAL per scene in Genesis, so changing it needs a fresh scene
    via GenesisProcess.restart. Batch episodes per material to amortize the rebuild.

SimBackend owns the RNG and calls sample_object_dxy() each reset and sample_scene()
when it (re)builds. Loaded from configs/dr/*.yaml via from_dict; presets in presets.py.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple

import numpy as np

_Range = Tuple[float, float]


@dataclass
class DRConfig:
    # ── per-reset (no rebuild) ────────────────────────────────────────────────
    object_pos_xy: float = 0.0          # half-range (m); per-env uniform object x/y jitter
    robot_init_pos_xyz: float = 0.0     # half-range (m); per-env uniform jitter on the reset home EE xyz
    object_yaw_deg: float = 0.0         # half-range (deg); per-env uniform object YAW (about world z). 180 = full
    object_pitch_roll_deg: float = 0.0  # half-range (deg); per-env uniform object PITCH & ROLL tilt (small, e.g. 15)

    # ── per-scene (rebuild via GenesisProcess.restart) ────────────────────────
    object_E: Optional[_Range] = None       # Young's modulus (Pa)
    object_nu: Optional[_Range] = None      # Poisson ratio (in (0, 0.5))
    object_rho: Optional[_Range] = None     # density (kg/m^3)
    object_yield: Optional[_Range] = None   # von Mises yield stress (Pa)
    coup_friction: Optional[_Range] = None  # rigid<->MPM coupling friction
    # ── per-scene object SIZE + SHAPE (rebuild; food varies in both) ──────────
    object_scale: Optional[_Range] = None     # uniform mesh/box scale, e.g. (0.8, 1.2)
    object_bend_deg: Optional[_Range] = None  # shape: total bend along the long axis (deg) — curvature
    object_twist_deg: Optional[_Range] = None # shape: total twist about the long axis (deg)
    object_taper: Optional[_Range] = None     # shape: end-to-end thickness change (fraction, e.g. (-0.2, 0.2))
    object_rbf: Optional[_Range] = None       # shape: organic bump magnitude (fraction of size, e.g. (0, 0.05))

    seed: int = 0

    _SCENE_FIELDS = ("object_E", "object_nu", "object_rho", "object_yield", "coup_friction",
                     "object_scale", "object_bend_deg", "object_twist_deg", "object_taper", "object_rbf")
    _SHAPE_FIELDS = ("object_bend_deg", "object_twist_deg", "object_taper", "object_rbf")

    def has_reset_dr(self) -> bool:
        return (self.object_pos_xy > 0 or self.robot_init_pos_xyz > 0
                or self.object_yaw_deg > 0 or self.object_pitch_roll_deg > 0)

    def has_scene_dr(self) -> bool:
        return any(getattr(self, f) is not None for f in self._SCENE_FIELDS)

    def has_shape_dr(self) -> bool:
        return any(getattr(self, f) is not None for f in self._SHAPE_FIELDS)

    def is_noop(self) -> bool:
        return not (self.has_reset_dr() or self.has_scene_dr())

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DRConfig":
        if not d:
            return cls()
        names = {f.name for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            k = "object_pos_xy" if k == "pose_dr_xy" else k     # back-compat alias
            if k not in names:
                continue
            kw[k] = tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else v
        return cls(**kw)

    # ── sampling ──────────────────────────────────────────────────────────────
    def sample_object_dxy(self, rng: np.random.Generator, num_envs: int) -> Optional[np.ndarray]:
        """Per-env object (dx, dy) offset from the default pose, or None if disabled."""
        if self.object_pos_xy <= 0:
            return None
        return rng.uniform(-self.object_pos_xy, self.object_pos_xy, (num_envs, 2)).astype(np.float32)

    def sample_object_euler(self, rng: np.random.Generator, num_envs: int) -> Optional[np.ndarray]:
        """Per-env object orientation as (num_envs, 3) intrinsic XYZ euler (roll, pitch, yaw)
        in RADIANS, or None if disabled. Yaw ~ U(+/- object_yaw_deg) about world z (full at
        180); pitch & roll ~ U(+/- object_pitch_roll_deg) small tilts. Applied at the object's
        resting center, so the object stays put but spawns rotated/tilted."""
        if self.object_yaw_deg <= 0 and self.object_pitch_roll_deg <= 0:
            return None
        pr = np.deg2rad(self.object_pitch_roll_deg)
        yaw = np.deg2rad(self.object_yaw_deg)
        e = np.zeros((num_envs, 3), dtype=np.float32)
        e[:, 0] = rng.uniform(-pr, pr, num_envs)    # roll  (about x)
        e[:, 1] = rng.uniform(-pr, pr, num_envs)    # pitch (about y)
        e[:, 2] = rng.uniform(-yaw, yaw, num_envs)  # yaw   (about z)
        return e

    def sample_home_offset(self, rng: np.random.Generator, num_envs: int) -> Optional[np.ndarray]:
        """Per-env (dx, dy, dz) uniform offset (+/- half-range) for the reset home EE
        pose, or None if disabled."""
        if self.robot_init_pos_xyz <= 0:
            return None
        v = self.robot_init_pos_xyz
        return rng.uniform(-v, v, (num_envs, 3)).astype(np.float32)

    def sample_scene(self, rng: np.random.Generator) -> Dict[str, float]:
        """Sample the per-scene params that are randomized (single value each — material
        is global). Keys: 'E','nu','rho','yield','coup_friction' for whatever is set."""
        out: Dict[str, float] = {}
        keymap = {"object_E": "E", "object_nu": "nu", "object_rho": "rho",
                  "object_yield": "yield", "coup_friction": "coup_friction"}
        for field_name, key in keymap.items():
            rng_pair = getattr(self, field_name)
            if rng_pair is not None:
                out[key] = float(rng.uniform(rng_pair[0], rng_pair[1]))
        return out

    def sample_shape_scale(self, rng: np.random.Generator) -> Dict[str, float]:
        """Per-scene object SIZE + SHAPE, or {} if none set. 'scale' (float); shape params for
        mesh_deform.deform_mesh — 'bend'/'twist' in RADIANS (config is deg), 'taper'/'rbf'
        fractions. Sampled once per scene build (all sub-envs share the geometry)."""
        out: Dict[str, float] = {}
        if self.object_scale is not None:
            out["scale"] = float(rng.uniform(*self.object_scale))
        if self.object_bend_deg is not None:
            out["bend"] = float(np.deg2rad(rng.uniform(*self.object_bend_deg)))
        if self.object_twist_deg is not None:
            out["twist"] = float(np.deg2rad(rng.uniform(*self.object_twist_deg)))
        if self.object_taper is not None:
            out["taper"] = float(rng.uniform(*self.object_taper))
        if self.object_rbf is not None:
            out["rbf"] = float(rng.uniform(*self.object_rbf))
        return out
