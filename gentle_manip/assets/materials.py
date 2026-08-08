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
    # Raspberry: soft, watery drupelet cluster, notoriously more fragile than the
    # mushroom preset above (bruises well before a firm grasp closes). Typical
    # literature ranges for soft berry tissue put the apparent elastic modulus at
    # the low end of soft fruit (tens to a few hundred kPa) with a density below
    # water (air pockets between drupelets let raspberries float). nu follows the
    # other near-incompressible watery-tissue presets (tofu/gelatin/mushroom).
    # TODO: calibrate to a real raspberry (no force-sensor validation yet, same
    # caveat as the mushroom/red_cube presets above).
    "raspberry": Material(youngs_modulus=1e5, poisson_ratio=0.35, density=650.0, von_mises_yield_stress=1.5e4),

    # ── Cross-category food set (literature-researched, not lab-calibrated) ──────
    # Values below are drawn from published food-engineering compression-test
    # studies (apparent/bulk elastic modulus of the FLESH, not skin/peel tensile
    # tests, since this project models a whole homogeneous soft body). Where a
    # study reports failure/rupture STRESS directly, that's used for yield;
    # otherwise yield = E * 0.15 (a yield-STRAIN heuristic consistent with the
    # mushroom preset above's own implicit ~13.3% strain, and with the two cases
    # here that DO have a directly measured yield strain: blueberry 18.5%,
    # strawberry 17.7%). nu defaults to 0.35 (high-water-content plant tissue,
    # matches tofu/gelatin/mushroom) unless a study gives a specific value.
    # Same "TODO: calibrate against a real instance" caveat as mushroom/raspberry
    # above applies to all of these -- literature-informed, not lab-verified for
    # THIS project's specific meshes.
    #
    # apple: quasi-static flesh E 1-5 MPa (cultivar-dependent; ~9 MPa under impact
    # loading), nu 0.25-0.35 (Golden Delicious studies). Density ~790 kg/m^3
    # (apples float).
    "apple": Material(youngs_modulus=3.0e6, poisson_ratio=0.30, density=790.0, von_mises_yield_stress=4.5e5),
    # pear: E=0.36 MPa (two independent multi-fruit comparative studies agree).
    "pear": Material(youngs_modulus=3.6e5, poisson_ratio=0.35, density=1020.0, von_mises_yield_stress=5.4e4),
    # tomato: MESOCARP (flesh) E 0.73-0.85 MPa -- NOT the much stiffer exocarp/peel
    # (4.6-9.6 MPa), which isn't representative of the bulk homogeneous body.
    "tomato": Material(youngs_modulus=8.0e5, poisson_ratio=0.35, density=970.0, von_mises_yield_stress=1.2e5),
    # peach: E=0.89 MPa (multi-fruit comparative study).
    "peach": Material(youngs_modulus=8.9e5, poisson_ratio=0.35, density=960.0, von_mises_yield_stress=1.3e5),
    # grape: no direct elastic-modulus citation found (only rupture FORCE at a
    # given probe geometry, not convertible to a material-intrinsic stress
    # without the study's contact area) -- ESTIMATED as a soft, thin-skinned,
    # high-turgor-pressure fruit, below tomato/pear on the same literature scale.
    # nu bumped to 0.40 (very high water content / turgor -> near-incompressible).
    "grape": Material(youngs_modulus=2.0e5, poisson_ratio=0.40, density=1080.0, von_mises_yield_stress=3.0e4),
    # strawberry: inner-tissue E=0.53 MPa AND a directly measured failure stress
    # of 0.093 MPa (17.7% failure strain) -- both used directly, not the heuristic.
    "strawberry": Material(youngs_modulus=5.3e5, poisson_ratio=0.35, density=850.0, von_mises_yield_stress=9.3e4),
    # blueberry: E=0.339 MPa, yield strain=0.185 (both directly cited; yield
    # stress = E * strain). Density ~730 kg/m^3 (blueberries are float-sorted by
    # ripeness -- most float, i.e. density below water).
    "blueberry": Material(youngs_modulus=3.39e5, poisson_ratio=0.35, density=730.0, von_mises_yield_stress=6.27e4),
    # blackberry: no direct citation found -- ESTIMATED by botanical analogy to
    # the raspberry preset above (same genus Rubus, same aggregate-drupelet
    # structure), so intentionally identical to raspberry rather than invented.
    "blackberry": Material(youngs_modulus=1.0e5, poisson_ratio=0.35, density=850.0, von_mises_yield_stress=1.5e4),
    # kiwi: edible-stage compression strength cited in the 0.06-0.36 MPa range;
    # a separate ripening study shows E dropping ~3.0 -> 0.3 MPa as it ripens.
    # Used a representative ripe/edible-stage value.
    "kiwi": Material(youngs_modulus=4.0e5, poisson_ratio=0.35, density=1030.0, von_mises_yield_stress=6.0e4),
    # fig: no direct citation found -- ESTIMATED (very soft ripe jam-like texture
    # softer than tomato/pear, placed near blueberry on the literature scale).
    "fig": Material(youngs_modulus=3.0e5, poisson_ratio=0.35, density=1030.0, von_mises_yield_stress=4.5e4),
    # cherry: E=0.40 MPa (whole-fruit BULK compression study -- distinct from
    # the much stiffer cherry SKIN tension modulus of 160-250 MPa reported
    # elsewhere, which characterizes the thin skin's tensile behavior, not the
    # bulk flesh this project's homogeneous continuum model represents).
    "cherry": Material(youngs_modulus=4.0e5, poisson_ratio=0.35, density=1060.0, von_mises_yield_stress=6.0e4),
    # avocado: ripe E=0.16 MPa (drops from 2.29 MPa unripe during ripening --
    # using the ripe/ready-to-eat value since that's the gentle-handling regime
    # this project's reward targets).
    "avocado": Material(youngs_modulus=1.6e5, poisson_ratio=0.35, density=950.0, von_mises_yield_stress=2.4e4),
    # banana: cited range 0.088-2.72 MPa depending on ripeness/measurement method
    # (unripe/green much stiffer than ripe/yellow); used a representative ripe
    # value in the middle of the two independent studies' ripe-stage numbers.
    "banana": Material(youngs_modulus=3.0e5, poisson_ratio=0.35, density=980.0, von_mises_yield_stress=4.5e4),
    # eggplant: E 0.42-1.47 MPa across fruit region/ripeness (upper section
    # stiffer than lower); used the range's midpoint.
    "eggplant": Material(youngs_modulus=7.0e5, poisson_ratio=0.35, density=980.0, von_mises_yield_stress=1.05e5),
    # mandarin_orange: whole citrus fruit tissue E=0.365-0.370 MPa (orange study;
    # mandarin is the same citrus flesh structure).
    "mandarin_orange": Material(youngs_modulus=3.7e5, poisson_ratio=0.35, density=960.0, von_mises_yield_stress=5.5e4),
    # egg: SPECIAL CASE, not a literature-measured homogeneous modulus -- a real
    # egg is a rigid brittle shell (true shell material E ~10 GPa, strength
    # 8.5-30.9 MPa) around a LIQUID interior, fundamentally not a homogeneous
    # continuum like every other preset here, so no "whole-egg E" exists in the
    # literature to cite. Deliberately modeled as moderately stiff (resists
    # deformation, unlike a mushroom's gradual soft yielding) but with a
    # DELIBERATELY LOW yield stress (cracks under light grip force, matching
    # real fragility) -- a documented simplification, not a citation.
    "egg": Material(youngs_modulus=2.0e6, poisson_ratio=0.30, density=1035.0, von_mises_yield_stress=8.0e3),
}
