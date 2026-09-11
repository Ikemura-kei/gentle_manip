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

    # Donut: the mushroom's stiffness/yield, but density set to hit a 50 g object at the enlarged
    # 20 cm size (user, 2026-09-10). 89 kg/m^3 is FAR below a real doughnut (~350-450) and below
    # cork — the 20 cm ring is 564 cm^3, so any edible density puts it at 500 g+, past the 398 g
    # banana the planner already failed to grasp. This is a geometry probe (does the planner take
    # the ring?), so mass is dialled to a liftable value rather than a physical one.
    # NOTE: low density is EXPENSIVE in MPM — wave speed is sqrt(E/rho), so substeps rise 3.25x
    # against rho 1000. The task config pairs this with grid_density 140 to stay affordable.
    "donut": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=88.7, von_mises_yield_stress=4e4),

    # donut_mush_small: the 10 cm ring, 69.9 cm^3, so 30 g needs rho 429 (user, 2026-09-10) — which
    # lands INSIDE the real doughnut range (350-450 kg/m^3), unlike the 20 cm donut's 89. This is
    # the one donut here that is both the mass asked for and physically plausible.
    "donut_small": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=426.0, von_mises_yield_stress=4e4),
    # ── Grasp-benchmark shape objects ────────────────────────────────────────────────────────────
    # These exist to vary GEOMETRY, not material: cylinder/cube share the mushroom's stiffness,
    # density and yield so a benchmark difference is attributable to shape rather than to a
    # confounded material change. Same soft-end E for the same MPM-stability reason.
    "soft_shape": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1000.0,
                           von_mises_yield_stress=4e4),

    # ── Objaverse expansion: per-object density so each lands at 50-100 g (user, 2026-09-10) ──
    # soft_shape's rho 1000 made mass follow VOLUME, which for these scanned meshes spans 5-300 cm^3
    # (5 g to 300 g). E/nu/yield are soft_shape's, unchanged, so only mass differs between them.
    # Density is clamped to [150, 2500] kg/m^3; martini, smoothie and telephoto_lens are too small
    # in volume to reach 50 g without exceeding that, and stay lighter.
    "objexp_hose": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1672.4, von_mises_yield_stress=4e4),   # 49 cm^3 -> 82 g
    "objexp_lemonade": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=416.1, von_mises_yield_stress=4e4),   # 123 cm^3 -> 51 g
    "objexp_license_plate": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=858.3, von_mises_yield_stress=4e4),   # 74 cm^3 -> 64 g
    "objexp_martini": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=2500.0, von_mises_yield_stress=4e4),   # 18 cm^3 -> 45 g
    "objexp_napkin": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1326.7, von_mises_yield_stress=4e4),   # 65 cm^3 -> 87 g
    "objexp_oil_lamp": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1391.6, von_mises_yield_stress=4e4),   # 60 cm^3 -> 84 g
    "objexp_pan": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=539.4, von_mises_yield_stress=4e4),   # 175 cm^3 -> 95 g
    "objexp_pet": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=721.7, von_mises_yield_stress=4e4),   # 75 cm^3 -> 54 g
    "objexp_pillow": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1042.3, von_mises_yield_stress=4e4),   # 68 cm^3 -> 71 g
    "objexp_pocket_watch": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1849.6, von_mises_yield_stress=4e4),   # 28 cm^3 -> 51 g
    "objexp_postcard": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=210.1, von_mises_yield_stress=4e4),   # 290 cm^3 -> 61 g
    "objexp_pot": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1948.1, von_mises_yield_stress=4e4),   # 39 cm^3 -> 75 g
    "objexp_radio_receiver": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=516.9, von_mises_yield_stress=4e4),   # 99 cm^3 -> 51 g
    "objexp_saucepan": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=514.0, von_mises_yield_stress=4e4),   # 117 cm^3 -> 60 g
    "objexp_sharpener": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1845.3, von_mises_yield_stress=4e4),   # 45 cm^3 -> 82 g
    "objexp_shower_cap": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1337.2, von_mises_yield_stress=4e4),   # 58 cm^3 -> 77 g
    "objexp_shredder": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=203.1, von_mises_yield_stress=4e4),   # 300 cm^3 -> 61 g
    "objexp_skullcap": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1457.6, von_mises_yield_stress=4e4),   # 55 cm^3 -> 79 g
    "objexp_smoothie": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=2500.0, von_mises_yield_stress=4e4),   # 5 cm^3 -> 13 g
    "objexp_squid": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=764.5, von_mises_yield_stress=4e4),   # 66 cm^3 -> 50 g
    "objexp_strainer": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1460.1, von_mises_yield_stress=4e4),   # 62 cm^3 -> 90 g
    "objexp_sweet_potato": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1727.4, von_mises_yield_stress=4e4),   # 49 cm^3 -> 85 g
    "objexp_tambourine": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=458.9, von_mises_yield_stress=4e4),   # 146 cm^3 -> 67 g
    "objexp_tartan": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=956.2, von_mises_yield_stress=4e4),   # 60 cm^3 -> 58 g
    "objexp_telephoto_lens": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=2500.0, von_mises_yield_stress=4e4),   # 35 cm^3 -> 87 g
    "objexp_thimble": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1121.4, von_mises_yield_stress=4e4),   # 60 cm^3 -> 67 g
    "objexp_timer": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=416.0, von_mises_yield_stress=4e4),   # 131 cm^3 -> 55 g
    "objexp_tote_bag": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=205.9, von_mises_yield_stress=4e4),   # 266 cm^3 -> 55 g
    "objexp_volleyball": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=1794.0, von_mises_yield_stress=4e4),   # 51 cm^3 -> 92 g
    "objexp_wallet": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=433.1, von_mises_yield_stress=4e4),   # 185 cm^3 -> 80 g
    "objexp_water_faucet": Material(youngs_modulus=3e5, poisson_ratio=0.35, density=2062.1, von_mises_yield_stress=4e4),   # 44 cm^3 -> 90 g
    # Raspberry (Rubus idaeus): a drupelet aggregate, markedly more fragile than a mushroom —
    # it bruises at a light squeeze. Softer (E 0.1 MPa) with a lower yield (15 kPa) and lower
    # density (it is largely water in thin-walled drupelets, and juicier/lighter than a mushroom).
    # TODO: calibrate against a real berry; these are literature-plausible, not measured.
    "raspberry": Material(youngs_modulus=1e5, poisson_ratio=0.4, density=900.0,
                          von_mises_yield_stress=1.5e4),  # RESTORED original (2026-09-06, user): with the _smooth meshes (option C) the soft true material is the target; the shatter was thin-neck geometry + CFL, not material (DEVLOG). Cherry-level values were diagnostic only.
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
