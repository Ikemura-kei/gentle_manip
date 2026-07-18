"""DR presets — sensible "mild" / "aggressive" ranges centered on a soft object.

Material ranges are absolute (Pa, kg/m^3) and centered on the tofu baseline
(E=4e3, nu=0.3, rho=1050, yield=2e4); re-center them per experiment for a stiffer
object (e.g. red_cube). Mirror these in configs/dr/*.yaml for YAML-driven runs.
"""
from __future__ import annotations

from gentle_manip.domain_randomization.dr_config import DRConfig


def mild() -> DRConfig:
    return DRConfig(
        object_pos_xy=0.03,                 # +/- 3 cm
        object_E=(3.0e3, 6.0e3),
        object_rho=(900.0, 1200.0),
        coup_friction=(3.5, 4.5),
    )


def aggressive() -> DRConfig:
    return DRConfig(
        object_pos_xy=0.06,                 # +/- 6 cm
        object_E=(1.0e3, 1.5e4),
        object_nu=(0.2, 0.45),
        object_rho=(300.0, 1500.0),
        object_yield=(1.0e4, 4.0e4),
        coup_friction=(2.5, 6.0),
        # food comes in different sizes and shapes
        object_scale=(0.8, 1.2),
        object_bend_deg=(-25.0, 25.0),
        object_twist_deg=(-20.0, 20.0),
        object_taper=(-0.15, 0.15),
        object_axis_scale=(0.9, 1.1),
    )


DR_PRESETS = {"mild": mild, "aggressive": aggressive}
