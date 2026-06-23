"""Material presets for MPM soft bodies.

Each preset is the set of MPM ElastoPlastic parameters that define how an object
deforms: Young's modulus E (Pa, stiffness), Poisson ratio nu (incompressibility),
density rho (kg/m^3, mass), and the von Mises yield stress (Pa, onset of plastic
flow). Values are calibrated in sim against the dev prototype
(examples/gs_sim_backend_dev.py) and are the defaults a registry ObjectDef
inherits; an ObjectEntry in a SceneSpec can override E/nu/rho per experiment.

These are point values, not ranges. Domain randomisation lives in
domain_randomization/ and samples around these.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    youngs_modulus: float          # E (Pa)
    poisson_ratio: float           # nu, in (0, 0.5)
    density: float                 # rho (kg/m^3)
    von_mises_yield_stress: float  # Pa — plastic flow onset


# Named presets. "tofu" is the soft, easily-bruised baseline validated in the dev
# prototype (deforms visibly under the gripper, lifts intact).
MATERIALS: dict[str, Material] = {
    "tofu":    Material(youngs_modulus=4e3, poisson_ratio=0.3, density=1050.0, von_mises_yield_stress=2e4),
    "gelatin": Material(youngs_modulus=8e3, poisson_ratio=0.35, density=1100.0, von_mises_yield_stress=3e4),
    "sponge":  Material(youngs_modulus=2e3, poisson_ratio=0.2, density=300.0, von_mises_yield_stress=1e4),
}
