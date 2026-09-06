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
    # tofu E 4e3 -> 5e4 (2026-08-25): at 4 kPa a 3cm MPM block collapses into a pile under
    # gravity at our grid resolution; 50 kPa = firm (momen) tofu, still 6x softer than the
    # mushroom. Yield 20 kPa (bruises/breaks easily) unchanged; density ~water unchanged.
    "tofu":    Material(youngs_modulus=5e4, poisson_ratio=0.3, density=1050.0, von_mises_yield_stress=2e4),
    "gelatin": Material(youngs_modulus=8e3, poisson_ratio=0.35, density=1100.0, von_mises_yield_stress=3e4),
    "sponge":  Material(youngs_modulus=2e3, poisson_ratio=0.2, density=300.0, von_mises_yield_stress=1e4),
    # Firm, near-rigid block to stand in for a real red cube (stiff + high yield so
    # it barely deforms). TODO: confirm the real cube's stiffness/mass.
    "red_cube": Material(youngs_modulus=3e4, poisson_ratio=0.3, density=1050.0, von_mises_yield_stress=8e4),
    # Real edible mushroom (Agaricus bisporus): soft viscoelastic tissue, real range
    # E 0.3-3.0 MPa, nu 0.3-0.5, yield 40-80 kPa. We use the SOFT END (E=0.3 MPa) on
    # purpose: explicit MPM is CFL-limited (substeps ~ sqrt(E)), and 0.3 MPa is ~1.7x
    # cheaper than 1 MPa while still realistic — the "Config C" chosen from the
    # examples/mushroom_soft_dev.py sweep (see CLAUDE.md "Soft-body mushroom"). yield
    # 4e4 / E 3e5 -> ~13% yield strain, so it bruises under a firm grasp (the regime
    # the gentle-manipulation stress reward targets). TODO: calibrate to a real mushroom.
    "mushroom": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1000.0, von_mises_yield_stress=4e4),
    # ── Grasp-benchmark shape objects ────────────────────────────────────────────────────────────
    # These exist to vary GEOMETRY, not material: cylinder/cube share the mushroom's stiffness,
    # density and yield so a benchmark difference is attributable to shape rather than to a
    # confounded material change. Same soft-end E for the same MPM-stability reason.
    "soft_shape": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1000.0,
                           von_mises_yield_stress=4e4),
    # Raspberry (Rubus idaeus): a drupelet aggregate, markedly more fragile than a mushroom —
    # it bruises at a light squeeze. Softer (E 0.1 MPa) with a lower yield (15 kPa) and lower
    # density (it is largely water in thin-walled drupelets, and juicier/lighter than a mushroom).
    # TODO: calibrate against a real berry; these are literature-plausible, not measured.
    "raspberry": Material(youngs_modulus=1.6e5, poisson_ratio=0.4, density=900.0,
                          von_mises_yield_stress=2.0e4),  # 2026-09-06 (user): raised from E=1e5/yield=1.5e4 — spawned berries SHATTERED (plastic flow on impact; see DEVLOG)
    # Banana (Musa, ripe, whole with peel): flesh is very soft (E ~0.1-0.5 MPa); the peel
    # stiffens the whole fruit somewhat. E 0.25 MPa keeps MPM substeps near the mushroom's
    # (substeps ~ sqrt(E)). Bruises readily -> yield 25 kPa. Density just under water.
    # TODO: calibrate against a real banana; literature-plausible, not measured.
    "banana": Material(youngs_modulus=2.5e5, poisson_ratio=0.35, density=950.0,
                       von_mises_yield_stress=2.5e4),
    # Strawberry (Fragaria): softer and far more bruise-prone than a mushroom — thin skin over
    # juicy parenchyma. E 0.15 MPa, low yield (18 kPa), density just under water.
    # TODO: calibrate against a real berry; literature-plausible, not measured.
    # Cherry tomato ~2.5 cm. Firm skin over juicy locular gel; E 0.4 MPa and yield 30 kPa are literature-plausible for a firm cherry tomato, NOT measured.
    "cherry_tomato": Material(youngs_modulus=4.0e+05, poisson_ratio=0.38, density=1000.0, von_mises_yield_stress=3.0e+04),
    # Regular tomato ~6.5 cm, 5-lobed. Softer and more bruise-prone than a cherry tomato (larger, riper). E 0.3 MPa / yield 25 kPa, literature-plausible, NOT measured.
    "tomato": Material(youngs_modulus=3.0e+05, poisson_ratio=0.4, density=1000.0, von_mises_yield_stress=2.5e+04),
    # A ~3.5 cm SEGMENT cut from the thick middle of the real banana scan. Same material as `banana`. Elongation 1.72 vs the full banana's 5.12 -- the point of this object is to test whether a COMPACT piece of banana avoids the contact-model validity limit that got the full banana parked (see DEVLOG 2026-08-27).
    "banana_chunk": Material(youngs_modulus=2.5e+05, poisson_ratio=0.35, density=950.0, von_mises_yield_stress=2.5e+04),
    # A bundle of ~7 COOKED strands, 6 cm long. Cooked pasta is very soft (dry pasta is ~GPa and not MPM-tractable): E 0.03 MPa / yield 10 kPa. Elongation 2.4 -- an intermediate case between the compact objects and the parked banana.
    "pasta_bundle": Material(youngs_modulus=1.2e+05, poisson_ratio=0.42, density=1100.0, von_mises_yield_stress=1.5e+04),
    "strawberry": Material(youngs_modulus=1.5e5, poisson_ratio=0.4, density=950.0,
                           von_mises_yield_stress=1.8e4),
}
