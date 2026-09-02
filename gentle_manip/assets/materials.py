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
    # Updated with a literature citation (was an uncalibrated 4e3 Pa placeholder):
    # compression-ball testing reports tofu stiffness 56.7+-14.1 kPa. nu/density
    # kept at their prior (already-reasonable) values -- no citation found more
    # specific than the E measurement. yield = E * 0.15 (same heuristic as the
    # cross-category food presets below, consistent with mushroom's own ~13.3%
    # implicit yield strain).
    "tofu":    Material(youngs_modulus=5.67e4, poisson_ratio=0.3, density=1050.0, von_mises_yield_stress=8.5e3),
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
    # E lowered to the bottom of that band (0.73 -> 0.50 MPa) so the 2 cm MPM body is
    # CFL-stable at grid 300: at 8.0e5 it exploded on contact ("flashes away"), NaN'd
    # the sim worker mid-eval, and the blown-up centroid tripped a false lifted-clear
    # success. 5.0e5 tracks cherry's 4.0e5 (identical mesh), which is stable.
    "tomato": Material(youngs_modulus=5.0e5, poisson_ratio=0.35, density=970.0, von_mises_yield_stress=1.2e5),
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
    # egg (BOILED, not raw): fragile-food-25 campaign (2026-08-13) list item is
    # "boiled egg", a genuinely homogeneous coagulated-protein solid (unlike raw
    # egg's shell+liquid special case above) -- so this is a NEW preset, not a
    # reuse of "egg". No direct boiled-egg-white compression-E citation was
    # sought (out of scope for this quick pass); ESTIMATED by placing it between
    # tofu (5.67e4, directly cited) and mushroom (3e5) on this file's own soft-
    # solid scale -- a coagulated egg-white gel is qualitatively firmer than
    # tofu but still well within MPM-friendly stiffness (the raw-egg preset's
    # 2e6 Pa would need a substep count ~2.6x mushroom's Config C baseline for
    # CFL stability, an avoidable cost for a food item that isn't actually that
    # stiff once cooked). nu/density kept close to the raw-egg preset (density
    # barely changes on boiling); yield = E * 0.15 heuristic (no citation).
    "egg_boiled": Material(youngs_modulus=1.5e5, poisson_ratio=0.35, density=1030.0, von_mises_yield_stress=2.25e4),

    # ── Kitchen/protein items (raw vs. cooked -- same mesh, different material;
    # cooking denatures proteins and dramatically changes stiffness, not gross
    # geometry, so this mirrors real cooking rather than needing separate scans).
    #
    # fish (raw, cod): E=22 kPa DIRECTLY CITED (Instron compression). nu bumped to
    # 0.40 (high water content, low connective tissue per the same literature).
    # yield strain 10% (not the usual 15%) -- raw fish is notoriously prone to
    # flaking/tearing apart, a lower yield-onset is the more honest choice here.
    "fish_raw": Material(youngs_modulus=2.2e4, poisson_ratio=0.40, density=1060.0, von_mises_yield_stress=2.2e3),
    # fish (cooked): no direct per-species cooked-E citation found. ESTIMATED via
    # a ~4x stiffening factor from the actin-denaturation literature (raw->cooked
    # meat stiffening is well documented as an order-of-magnitude-scale effect,
    # e.g. 6->60 kPa i.e. 10x in one cited generic-muscle study) -- used a MORE
    # CONSERVATIVE 4x than that generic figure since fish has much less
    # connective tissue than red meat (the dominant collagen-contraction
    # mechanism is weaker), so its cooked stiffening should be smaller than
    # beef's. Flagged as an estimate, not a direct citation, unlike fish_raw.
    "fish_cooked": Material(youngs_modulus=9.0e4, poisson_ratio=0.38, density=1090.0, von_mises_yield_stress=1.35e4),
    # beef (raw): E=2 kPa DIRECTLY CITED (raw skeletal muscle, "soft, compliant"
    # per the same source). nu=0.45 (very high water content, near-incompressible
    # uncooked muscle). yield strain 10% (very low yield -- raw muscle tears
    # easily, matching real handling fragility).
    "beef_raw": Material(youngs_modulus=2.0e3, poisson_ratio=0.45, density=1060.0, von_mises_yield_stress=2.0e2),
    # beef (cooked): E=150 kPa, within the DIRECTLY CITED 70-300 kPa range for
    # cooked/non-marinaded steak (apparent elastic modulus, compression). Picked
    # the range's midpoint. Higher collagen cross-linking + moisture loss vs.
    # fish_cooked -> denser, and nu drops (less free water, more structured).
    "beef_cooked": Material(youngs_modulus=1.5e5, poisson_ratio=0.35, density=1090.0, von_mises_yield_stress=2.25e4),

    # ── Second cross-category batch: chicken/shrimp/scallop/watermelon/mozzarella/
    # dumpling/pasta (literature-researched, not lab-calibrated -- same caveat as
    # the produce set above: informed by published compression-test studies where
    # available, but NOT force-sensor-validated for this project's specific meshes).
    # yield = E * 0.15 heuristic unless a study gives a direct failure stress/strain.
    #
    # chicken_breast (raw): no direct compressive-E-in-Pa citation found despite a
    # thorough search of poultry texture-analysis literature (studies report
    # compression FORCE in N or a texturometer-software "Young's modulus" in N/s,
    # neither convertible to Pa without the untabulated probe geometry). ESTIMATED
    # via a directly-cited PROXY: a chicken-breast thermal-treatment kinetics study
    # reports a raw/room-temperature dynamic storage modulus G'=13.5+-1.3 kPa
    # (small-strain oscillatory rheology); converted to an apparent compressive E
    # via the near-incompressible-tissue relation E ~= 3*G' (nu~0.5) -> ~40 kPa.
    # This is a real cited measurement run through an idealized conversion, not a
    # directly-measured compression E -- flagged as estimated, not directly cited.
    # nu=0.40 (raw high-water lean muscle, same category as fish_raw). Density
    # ~1050 kg/m^3 (general raw-poultry food-composition reference).
    "chicken_breast_raw": Material(youngs_modulus=4.0e4, poisson_ratio=0.40, density=1050.0, von_mises_yield_stress=6.0e3),
    # shrimp (raw, peeled): no direct compressive-E citation found -- shrimp texture
    # studies (TPA, high-pressure processing, storage-quality papers) report
    # compression/shear FORCE (N) and TPA hardness/springiness, never a tabulated
    # stress-strain modulus. ESTIMATED BY ANALOGY to fish_raw (same "raw seafood
    # muscle, high water, low connective tissue" category; fish_raw=22 kPa
    # DIRECTLY CITED elsewhere in this file) -- shrimp muscle lacks the fish myotome
    # flake-plane structure so nudged slightly stiffer. nu=0.40 (matches fish_raw).
    # Density 1060 kg/m^3, within the DIRECTLY CITED apparent-density range for
    # fresh seafood, 1042-1093 kg/m^3 at 20C (Rahman 1994, J. Food Process Eng.).
    "shrimp_raw": Material(youngs_modulus=3.0e4, poisson_ratio=0.40, density=1060.0, von_mises_yield_stress=4.5e3),
    # scallop (raw adductor muscle): no direct compressive-E citation found --
    # scallop adductor-muscle papers (boiling, ultra-high-pressure processing)
    # report TPA hardness/chewiness/shear-force trends under processing, never a
    # baseline raw stress-strain modulus. ESTIMATED BY ANALOGY: adductor muscle is
    # a single dense, uniformly-packed fast/slow fiber bundle (structurally
    # different from -- and by every qualitative description firmer than -- fish
    # or shrimp muscle), so placed above shrimp_raw, roughly at the tofu preset's
    # firmness. nu=0.40 (raw seafood muscle category, ~78-80% water per retail
    # scallop composition studies). Density 1070 kg/m^3, within the same DIRECTLY
    # CITED fresh-seafood apparent-density range used for shrimp_raw above.
    "scallop_raw": Material(youngs_modulus=5.0e4, poisson_ratio=0.40, density=1070.0, von_mises_yield_stress=7.5e3),
    # watermelon (ripe flesh, not rind): E=0.536 MPa DIRECTLY CITED (Crimson Sweet
    # cultivar, quasi-static compression / nonlinear-FEA bruising study -- red
    # FLESH only, distinct from the much stiffer white/green rind at 0.9-4.9 MPa
    # in the same study, which is not representative of the homogeneous body
    # modeled here). Failure stress 27 kPa DIRECTLY CITED, same cultivar/study
    # (Charleston Gray gives 37 kPa -- used Crimson Sweet's own 27 kPa to stay
    # internally consistent with the E citation rather than mixing cultivars).
    # nu=0.40 (very high water content, ~92%, near-incompressible -- same bump as
    # grape). Density 950 kg/m^3 DIRECTLY CITED (buoyancy measurement, ~0.94 g/cm^3).
    "watermelon": Material(youngs_modulus=5.36e5, poisson_ratio=0.40, density=950.0, von_mises_yield_stress=2.7e4),
    # mozzarella (fresh): E=175 kPa DIRECTLY CITED (deformation modulus, uniaxial
    # compression of 2 cm cheese cubes, isotropic w.r.t. compression direction --
    # Fogaca et al. 2017, J. Texture Studies, "Influence of compression parameters
    # on mechanical behavior of mozzarella cheese"). No direct failure stress/
    # strain found (the same literature reports a "73% degree of elasticity"
    # recovery metric, not a rupture point) -- yield = E * 0.15 heuristic. nu=0.35
    # (moist but structured fat/protein matrix, not free-water muscle tissue --
    # same category as tofu/gelatin, not bumped to 0.40 like the raw-seafood
    # entries above). Density 1060 kg/m^3 (general fresh high-moisture mozzarella
    # food-density reference; excludes an anomalous "low sodium" outlier figure
    # found in the same search).
    "mozzarella": Material(youngs_modulus=1.75e5, poisson_ratio=0.35, density=1060.0, von_mises_yield_stress=2.625e4),
    # dumpling (cooked, dough-dominated whole-object approximation): no
    # "dumpling"-specific compression-E citation found. ESTIMATED, extrapolated
    # from a DIRECTLY CITED generic wheat-flour-dough elastic-modulus range of
    # 10-42 kPa (texturometer compression study) -- used a representative value
    # near that range's upper-middle, nudged up slightly for the starch
    # gelatinization stiffening that occurs on boiling/steaming (the raw-dough
    # citation predates cooking). Filling (ground meat/vegetable) is comparable
    # or softer, so dough-dominated is a reasonable whole-body approximation. nu=
    # 0.35 (moist starchy matrix, same category as gelatin/tofu). Density 1050
    # kg/m^3, within the DIRECTLY CITED range for wet (unleavened/uncompressed)
    # wheat dough, 975-1100 kg/m^3 (composite-dough thermophysical-properties
    # literature) -- NOT bread-crumb density (baked+leavened, much lower/airier,
    # not representative of a boiled/steamed dumpling wrapper).
    "dumpling_cooked": Material(youngs_modulus=3.0e4, poisson_ratio=0.35, density=1050.0, von_mises_yield_stress=4.5e3),
    # pasta (cooked, treated as one homogeneous bundle/mass of noodles): E DIRECTLY
    # CITED at order-of-magnitude ~10^2 kPa for the fully-hydrated/saturated
    # cooked state (Phys. of Fluids 2022, "Swelling, softening, and elastocapillary
    # adhesion of cooked pasta" -- tracks E dropping ~5 orders of magnitude from
    # dry E0=2.17 GPa through a glassy-to-rubbery cooking transition to the
    # saturated plateau); picked a representative point mid-way through that
    # cited order of magnitude. yield = E * 0.15 heuristic (no failure strain
    # reported). nu=0.35 (moist starch gel matrix, gelatin/tofu category).
    # Density 600 kg/m^3 DIRECTLY CITED (boiled/drained pasta bulk-density food
    # reference) -- deliberately the BULK figure (air gaps between strands), not
    # a single noodle's material density (~1050-1100 kg/m^3), since this preset
    # approximates the whole BUNDLE as one homogeneous body (same air-pocket
    # reasoning as the raspberry preset above).
    "pasta_cooked": Material(youngs_modulus=1.5e5, poisson_ratio=0.35, density=600.0, von_mises_yield_stress=2.25e4),
}
