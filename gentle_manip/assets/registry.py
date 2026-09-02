"""Object registry: name -> ObjectDef (geometry + default material).

OBJECT_MAP is the single source of truth for what each object name spawns. Objects
are either a primitive MPM/rigid box (mesh_path=None) or a scanned mesh (mesh_path
set -> Genesis loads gs.morphs.Mesh). Scanned meshes live in assets/objects/ and are
stored in METERS, so no scale conversion is needed at the call site.

SceneBuilder reads OBJECT_MAP to turn a SceneSpec ObjectEntry (which may override
E/nu/rho/scale/pose) into Genesis morph + material calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from gentle_manip.assets.materials import MATERIALS, Material

_OBJ_DIR = Path(__file__).resolve().parent / "objects"


@dataclass(frozen=True)
class ObjectDef:
    name: str
    material: Material
    object_type: str = "soft"                       # "soft" (MPM) | "rigid"
    size: Tuple[float, float, float] = (0.04, 0.04, 0.04)   # primitive box extents (m)
    default_pos: Tuple[float, float, float] = (0.50, 0.0, 0.03)  # resting pose (m)
    mesh_path: Optional[str] = None                 # reserved; None => primitive box
    # ── Cross-category DR (see domain_randomization/dr_config.py) ────────────────
    # Per-category defaults used when a DRConfig activates `object_category_pool`
    # but leaves its own absolute field unset — lets ONE DRConfig work across many
    # categories while every category still gets ranges suited to its own geometry
    # and material, instead of hardcoding 10-20 absolute range sets in presets.py.
    # Never used unless a category pool is active; single-category experiments are
    # unaffected. Keys mirror DRConfig's shape/material field names, minus the
    # `object_` prefix (e.g. "bend_deg", "taper", "E", "yield").
    shape_dr_ranges: Optional[Dict[str, Tuple[float, float]]] = None      # bend_deg/twist_deg/taper/rbf/axis_scale/scale
    material_dr_mult: Optional[Dict[str, Tuple[float, float]]] = None    # E/nu/rho/yield, as MULTIPLIERS on this object's own nominal (not absolute Pa)
    # ── Per-category MPM stability overrides ──────────────────────────────────────
    # SceneSpec.sim_substeps/mpm_grid_density are scene-wide constants tuned for
    # ONE object (mushroom's "Config C") — a genuinely different category can need
    # a different value for a numerically stable sim, independent of DR (this is
    # NOT randomized, just a fixed per-category correction). Discovered empirically:
    # a real photogrammetry-scan mesh (pear, ~54k faces) needs more substeps than
    # the shared default (220) to avoid NaN contact forces; a very small object
    # (blueberry, ~1cm) needs a finer MPM grid than the shared default (250) or
    # particle sampling errors out. None = use SceneSpec's existing default.
    sim_substeps_override: Optional[int] = None
    mpm_grid_density_override: Optional[float] = None


# The validated baseline matches examples/gs_sim_backend_dev.py: a 4 cm tofu cube
# resting at (0.50, 0, 0.03), within reach and the MPM bounds.
OBJECT_MAP: dict[str, ObjectDef] = {
    "tofu":    ObjectDef("tofu", MATERIALS["tofu"]),
    "gelatin": ObjectDef("gelatin", MATERIALS["gelatin"]),
    "sponge":  ObjectDef("sponge", MATERIALS["sponge"]),
    # 3 cm cube matching the real red cube the DP3 policy was trained on; rests on
    # the table (half-extent 0.015 -> center z ~0.015 after settling).
    "red_cube": ObjectDef("red_cube", MATERIALS["red_cube"],
                          size=(0.03, 0.03, 0.03), default_pos=(0.47, 0.011, 0.02)),
    # Gripper-width calibration cubes (use as rigid): known sizes spread across x so
    # all four fit in one scene; spawn z = half-extent so they rest on the table.
    "cal_cube_6": ObjectDef("cal_cube_6", MATERIALS["red_cube"], size=(0.06, 0.06, 0.06), default_pos=(0.34, 0.0, 0.035)),
    "cal_cube_5": ObjectDef("cal_cube_5", MATERIALS["red_cube"], size=(0.05, 0.05, 0.05), default_pos=(0.42, 0.0, 0.030)),
    "cal_cube_4": ObjectDef("cal_cube_4", MATERIALS["red_cube"], size=(0.04, 0.04, 0.04), default_pos=(0.50, 0.0, 0.025)),
    "cal_cube_3": ObjectDef("cal_cube_3", MATERIALS["red_cube"], size=(0.03, 0.03, 0.03), default_pos=(0.58, 0.0, 0.020)),
    # Real scanned edible mushroom (assets/objects/mushroom.obj, in meters: ~3.3 x 3.2
    # x 3.5 cm). Soft MPM body; rests on the table (mesh bottom at z=-0.0148, so
    # default_pos z=0.016 puts it just above the surface). size is informational only
    # (ignored for meshes). Material is the soft-end "Config C" (see materials.py).
    "mushroom": ObjectDef(
        "mushroom", MATERIALS["mushroom"], object_type="soft",
        size=(0.033, 0.032, 0.035), default_pos=(0.47, 0.0, 0.016),
        mesh_path=str(_OBJ_DIR / "mushroom.obj"),
        # Matches configs/dr/food_shape.yaml's tuned ranges (kept on the soft side of
        # nominal E for MPM stability at the tuned substeps — see materials.py).
        shape_dr_ranges={"bend_deg": (-25.0, 25.0), "twist_deg": (-20.0, 20.0),
                         "taper": (-0.15, 0.15), "axis_scale": (0.9, 1.1),
                         "scale": (0.8, 1.12)},
        material_dr_mult={"E": (0.667, 1.0), "nu": (0.914, 1.086), "rho": (0.9, 1.1)},
    ),
    # Second mesh-based, cross-category food object (Stage 1 triage: validated via
    # gentle_manip.scripts.repair_and_validate_mesh — tet-meshable, gripper-feasible,
    # ~1.5cm isotropic). Roughly spherical drupelet cluster, unlike the mushroom's
    # cap+stem asymmetry, so its shape DR favors organic surface bumps (rbf) over
    # bend/twist (which assume an elongated long axis).
    "raspberry": ObjectDef(
        "raspberry", MATERIALS["raspberry"], object_type="soft",
        size=(0.0154, 0.0154, 0.0146), default_pos=(0.47, 0.0, 0.008),
        mesh_path=str(_OBJ_DIR / "raspberry.stl"),
        shape_dr_ranges={"bend_deg": (-8.0, 8.0), "twist_deg": (-10.0, 10.0),
                         "taper": (-0.1, 0.1), "rbf": (0.0, 0.08),
                         "axis_scale": (0.9, 1.1), "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
    ),

    # ── Cross-category food set continued (sourced from Objaverse-LVIS, CC-BY/CC0
    # licensed, permissive-license-filtered; each validated via
    # gentle_manip.scripts.repair_and_validate_mesh — tet-meshable + gripper-
    # feasible + sane volume). default_pos.z is set to roughly half the mesh's
    # max extent (a safe "spawn above the table" height) since these meshes'
    # native pivot isn't guaranteed centered like mushroom.obj's; the settle
    # step lets each one fall into its natural resting pose regardless.
    #
    # apple: near-round, minor natural asymmetry -> small bend/twist, mild
    # surface bumps, moderate scale range (real apples span ~6-9cm).
    "apple": ObjectDef(
        "apple", MATERIALS["apple"], object_type="soft",
        size=(0.065, 0.060, 0.065), default_pos=(0.47, 0.0, 0.033),
        mesh_path=str(_OBJ_DIR / "apple.obj"),
        shape_dr_ranges={"bend_deg": (-6.0, 6.0), "twist_deg": (-8.0, 8.0),
                         "taper": (-0.08, 0.08), "rbf": (0.0, 0.04),
                         "axis_scale": (0.9, 1.1), "scale": (0.85, 1.2)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
    ),
    # pear: round nashi-type scan (not the elongated western teardrop shape);
    # moderate bend for natural lopsidedness, more taper range than apple. Real
    # photogrammetry scan (~54k faces, much denser than the other harvested
    # meshes) -- empirically NaN'd during settling at the shared 220-substep
    # default; 400 is confirmed stable (see harvest verification notes).
    "pear": ObjectDef(
        "pear", MATERIALS["pear"], object_type="soft",
        size=(0.062, 0.052, 0.065), default_pos=(0.47, 0.0, 0.0325),
        mesh_path=str(_OBJ_DIR / "pear.obj"),
        shape_dr_ranges={"bend_deg": (-10.0, 10.0), "twist_deg": (-8.0, 8.0),
                         "taper": (-0.12, 0.12), "rbf": (0.0, 0.03),
                         "axis_scale": (0.9, 1.1), "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
        sim_substeps_override=400,
    ),
    # grape: tiny, near-spherical -- minimal shape DR needed, but real grapes
    # vary a lot in size (small seedless to large table grapes), so a wide
    # scale range.
    "grape": ObjectDef(
        "grape", MATERIALS["grape"], object_type="soft",
        size=(0.019, 0.020, 0.020), default_pos=(0.47, 0.0, 0.010),
        mesh_path=str(_OBJ_DIR / "grape.obj"),
        shape_dr_ranges={"bend_deg": (-5.0, 5.0), "twist_deg": (-5.0, 5.0),
                         "taper": (-0.05, 0.05), "rbf": (0.0, 0.02),
                         "axis_scale": (0.92, 1.08), "scale": (0.8, 1.25)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
    ),
    # kiwi: elongated oval (unlike the round berries above) -- more bend/twist/
    # taper range, matching its natural long-axis asymmetry.
    "kiwi": ObjectDef(
        "kiwi", MATERIALS["kiwi"], object_type="soft",
        size=(0.043, 0.046, 0.060), default_pos=(0.47, 0.0, 0.030),
        mesh_path=str(_OBJ_DIR / "kiwi.obj"),
        shape_dr_ranges={"bend_deg": (-8.0, 8.0), "twist_deg": (-10.0, 10.0),
                         "taper": (-0.10, 0.10), "rbf": (0.0, 0.03),
                         "axis_scale": (0.9, 1.1), "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
        # E=4e5 is 1.33x mushroom's Config C nominal (substeps ~ sqrt(E)) -- a
        # modest bump over the 220 baseline; fragile-food-25 campaign (2026-08-13),
        # not yet empirically confirmed via a soft-MPM smoke test (kiwi was only
        # ever run RIGID earlier this session).
        sim_substeps_override=280,
    ),
    # cherry: tiny and near-spherical, same shape-DR profile as grape.
    "cherry": ObjectDef(
        "cherry", MATERIALS["cherry"], object_type="soft",
        size=(0.020, 0.017, 0.020), default_pos=(0.47, 0.0, 0.0098),
        mesh_path=str(_OBJ_DIR / "cherry.obj"),
        shape_dr_ranges={"bend_deg": (-5.0, 5.0), "twist_deg": (-5.0, 5.0),
                         "taper": (-0.05, 0.05), "rbf": (0.0, 0.02),
                         "axis_scale": (0.92, 1.08), "scale": (0.85, 1.2)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
    ),
    # blueberry: tiny near-spherical berry, same profile as raspberry/grape but
    # scaled down further; real blueberries vary a fair amount in size cluster
    # to cluster, so a wide scale range. Its small volume (~0.13 cm^3) undersamples
    # MPM particles at the shared 250 grid_density default (crashes with an
    # internal "kth out of bounds" particle-sampling error); 500 is confirmed
    # stable (see harvest verification notes).
    "blueberry": ObjectDef(
        "blueberry", MATERIALS["blueberry"], object_type="soft",
        size=(0.0088, 0.013, 0.009), default_pos=(0.47, 0.0, 0.0045),
        mesh_path=str(_OBJ_DIR / "blueberry.obj"),
        shape_dr_ranges={"bend_deg": (-5.0, 5.0), "twist_deg": (-5.0, 5.0),
                         "taper": (-0.05, 0.05), "rbf": (0.0, 0.03),
                         "axis_scale": (0.9, 1.1), "scale": (0.8, 1.25)},
        mpm_grid_density_override=500,
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
    ),
    # egg: distinctive, fairly CONSISTENT ellipsoid shape (unlike organic fruit,
    # real eggs don't bend/twist/bump) -- shape DR is deliberately narrow, mostly
    # just a taper range (rounder vs. more pointed eggs) and modest scale.
    "egg": ObjectDef(
        "egg", MATERIALS["egg"], object_type="soft",
        size=(0.0446, 0.058, 0.0451), default_pos=(0.47, 0.0, 0.0225),
        mesh_path=str(_OBJ_DIR / "egg.obj"),
        shape_dr_ranges={"bend_deg": (-2.0, 2.0), "twist_deg": (-2.0, 2.0),
                         "taper": (-0.08, 0.08), "axis_scale": (0.95, 1.05),
                         "scale": (0.92, 1.08)},
        material_dr_mult={"E": (0.7, 1.3), "nu": (0.95, 1.05), "rho": (0.95, 1.05),
                          "yield": (0.5, 1.5)},
    ),
    # egg_boiled: fragile-food-25 campaign's "boiled egg" list item -- uses the
    # SAME mesh as raw "egg" but the much softer, MPM-friendlier egg_boiled
    # material (see materials.py -- the raw-egg preset's 2e6 Pa shell-stiffness
    # would need ~2.6x mushroom's substep count for CFL stability, an avoidable
    # cost for a cooked, genuinely-homogeneous-solid food item).
    "egg_boiled": ObjectDef(
        "egg_boiled", MATERIALS["egg_boiled"], object_type="soft",
        size=(0.0446, 0.058, 0.0451), default_pos=(0.47, 0.0, 0.0225),
        mesh_path=str(_OBJ_DIR / "egg.obj"),
        shape_dr_ranges={"bend_deg": (-2.0, 2.0), "twist_deg": (-2.0, 2.0),
                         "taper": (-0.08, 0.08), "axis_scale": (0.95, 1.05),
                         "scale": (0.92, 1.08)},
        material_dr_mult={"E": (0.7, 1.3), "nu": (0.95, 1.05), "rho": (0.95, 1.05),
                          "yield": (0.5, 1.5)},
    ),
    # avocado: larger, elongated pear-like shape -- more bend/taper range than
    # the round berries, matching its natural asymmetry (narrower neck, wider
    # base).
    "avocado": ObjectDef(
        "avocado", MATERIALS["avocado"], object_type="soft",
        size=(0.062, 0.095, 0.0614), default_pos=(0.47, 0.0, 0.0307),
        mesh_path=str(_OBJ_DIR / "avocado.obj"),
        shape_dr_ranges={"bend_deg": (-8.0, 8.0), "twist_deg": (-6.0, 6.0),
                         "taper": (-0.12, 0.12), "rbf": (0.0, 0.02),
                         "axis_scale": (0.9, 1.1), "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.4, 2.0), "nu": (0.95, 1.05), "rho": (0.9, 1.1),
                          "yield": (0.4, 1.6)},
    ),

    # ── Kitchen/protein items: fish and beef registered TWICE (raw + cooked),
    # sharing one mesh_path -- cooking changes material, not gross geometry.
    # fish: elongated fillet-piece shape (~7 x 2.4 x 2.1cm).
    "fish_raw": ObjectDef(
        "fish_raw", MATERIALS["fish_raw"], object_type="soft",
        size=(0.07, 0.024, 0.0212), default_pos=(0.47, 0.0, 0.0106),
        mesh_path=str(_OBJ_DIR / "fish.obj"),
        shape_dr_ranges={"bend_deg": (-10.0, 10.0), "twist_deg": (-8.0, 8.0),
                         "taper": (-0.1, 0.1), "rbf": (0.0, 0.02),
                         "axis_scale": (0.9, 1.1), "scale": (0.8, 1.2)},
        material_dr_mult={"E": (0.5, 1.6), "nu": (0.97, 1.03), "rho": (0.9, 1.1),
                          "yield": (0.4, 1.6)},
    ),
    "fish_cooked": ObjectDef(
        "fish_cooked", MATERIALS["fish_cooked"], object_type="soft",
        size=(0.07, 0.024, 0.0212), default_pos=(0.47, 0.0, 0.0106),
        mesh_path=str(_OBJ_DIR / "fish.obj"),
        shape_dr_ranges={"bend_deg": (-8.0, 8.0), "twist_deg": (-6.0, 6.0),
                         "taper": (-0.1, 0.1), "rbf": (0.0, 0.02),
                         "axis_scale": (0.9, 1.1), "scale": (0.8, 1.2)},
        material_dr_mult={"E": (0.5, 1.6), "nu": (0.97, 1.03), "rho": (0.9, 1.1),
                          "yield": (0.4, 1.6)},
    ),

    # beef: diced stew-style chunk. Reasonably block-shaped (unlike fish's elongated
    # fillet), so -- after repeated Objaverse sourcing attempts (beef_(food),
    # steak_(food), patty_(food) all failed tet-meshability QA) -- registered as a
    # primitive box (mesh_path=None, mirrors tofu) rather than blocking demo
    # collection on further mesh hunting. No shape_dr_ranges (mesh_deform doesn't
    # apply to a box, same as tofu); scale-only shape DR via the DR yaml's
    # object_scale field.
    "beef_raw":    ObjectDef("beef_raw",    MATERIALS["beef_raw"],    size=(0.035, 0.03, 0.025)),
    "beef_cooked": ObjectDef("beef_cooked", MATERIALS["beef_cooked"], size=(0.035, 0.03, 0.025)),

    # ── Fragile-food 25-category campaign (2026-08-13): new meshes generated by
    # gentle_manip/scripts/generate_fragile25_meshes.py (procedural primitives +
    # mesh_deform, validated via repair_and_validate_mesh.py), materials from
    # materials.py (7 newly researched, see that file's citations). Every new
    # mesh deliberately avoids a flat/thin resting profile per user direction --
    # built tall/chunky enough for the parallel-jaw gripper.
    #
    # shiitake: distinct cap+stem silhouette from button mushroom (broader,
    # flatter cap) but the SAME fungal tissue, so reuses mushroom's material
    # and shape-DR profile rather than a new one.
    "shiitake": ObjectDef(
        "shiitake", MATERIALS["mushroom"], object_type="soft",
        size=(0.0525, 0.0373, 0.035), default_pos=(0.47, 0.0, 0.029),
        mesh_path=str(_OBJ_DIR / "shiitake.obj"),
        shape_dr_ranges={"bend_deg": (-20.0, 20.0), "twist_deg": (-15.0, 15.0),
                         "taper": (-0.12, 0.12), "axis_scale": (0.9, 1.1),
                         "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.667, 1.0), "nu": (0.914, 1.086), "rho": (0.9, 1.1)},
    ),
    # strawberry: cone/heart taper-to-a-point silhouette.
    "strawberry": ObjectDef(
        "strawberry", MATERIALS["strawberry"], object_type="soft",
        size=(0.032, 0.032, 0.035), default_pos=(0.47, 0.0, 0.019),
        mesh_path=str(_OBJ_DIR / "strawberry.obj"),
        shape_dr_ranges={"bend_deg": (-6.0, 6.0), "twist_deg": (-5.0, 5.0),
                         "taper": (-0.1, 0.1), "rbf": (0.0, 0.03),
                         "axis_scale": (0.9, 1.1), "scale": (0.85, 1.2)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
        sim_substeps_override=300,   # E=5.3e5 is 1.77x mushroom's nominal -- see kiwi's note
    ),
    # blackberry: same near-spherical drupelet-cluster profile as raspberry
    # (same genus Rubus, matches materials.py's by-analogy material too).
    "blackberry": ObjectDef(
        "blackberry", MATERIALS["blackberry"], object_type="soft",
        size=(0.022, 0.022, 0.022), default_pos=(0.47, 0.0, 0.012),
        mesh_path=str(_OBJ_DIR / "blackberry.obj"),
        shape_dr_ranges={"bend_deg": (-8.0, 8.0), "twist_deg": (-10.0, 10.0),
                         "taper": (-0.1, 0.1), "rbf": (0.0, 0.08),
                         "axis_scale": (0.9, 1.1), "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
    ),
    # peach: registered mesh is a THICK wedge (peach_slice.obj), not a whole
    # peach -- user's "ripe peach slice" list item, built tall rather than a
    # thin disc per the avoid-flat-profiles direction.
    "peach": ObjectDef(
        "peach", MATERIALS["peach"], object_type="soft",
        size=(0.03, 0.039, 0.06), default_pos=(0.47, 0.0, 0.033),
        mesh_path=str(_OBJ_DIR / "peach_slice.obj"),
        shape_dr_ranges={"bend_deg": (-8.0, 8.0), "twist_deg": (-6.0, 6.0),
                         "taper": (-0.1, 0.1), "rbf": (0.0, 0.02),
                         "axis_scale": (0.9, 1.1), "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
        sim_substeps_override=400,   # E=8.9e5 is 2.97x mushroom's nominal -- see kiwi's note
    ),
    # banana: registered mesh is a capsule PIECE/chunk (banana_piece.obj), not
    # a thin round coin-slice -- user's "banana piece" list item.
    "banana": ObjectDef(
        "banana", MATERIALS["banana"], object_type="soft",
        size=(0.0318, 0.0318, 0.057), default_pos=(0.47, 0.0, 0.031),
        mesh_path=str(_OBJ_DIR / "banana_piece.obj"),
        shape_dr_ranges={"bend_deg": (-10.0, 10.0), "twist_deg": (-5.0, 5.0),
                         "taper": (-0.08, 0.08), "axis_scale": (0.9, 1.1),
                         "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
    ),
    # banana_lying: banana_piece.obj rotated so the LONG axis is horizontal (Y) --
    # a banana chunk resting on its side, how it actually sits on a table. The
    # upright "banana" baton (long axis vertical) is unstable for rigid physics
    # (topples during settle) and impossible to grasp cleanly top-down (finger
    # reach below TCP is ~7cm > the baton height). Used by the rigid-banana
    # diverse-start regrasp surrogate. mesh origin is at the base (min-z=0).
    "banana_lying": ObjectDef(
        "banana_lying", MATERIALS["banana"], object_type="soft",
        size=(0.0318, 0.057, 0.0318), default_pos=(0.47, 0.0, 0.004),
        mesh_path=str(_OBJ_DIR / "banana_piece_lying.obj"),
        shape_dr_ranges={"bend_deg": (-10.0, 10.0), "twist_deg": (-5.0, 5.0),
                         "taper": (-0.08, 0.08), "axis_scale": (0.9, 1.1),
                         "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},
    ),
    # tomato (cherry tomato): reuses cherry.obj's geometry (round, cherry-tomato-
    # scale) paired with the tomato material -- shape-DR profile copied verbatim
    # from the cherry entry since it's the identical mesh.
    "tomato": ObjectDef(
        "tomato", MATERIALS["tomato"], object_type="soft",
        size=(0.020, 0.017, 0.020), default_pos=(0.47, 0.0, 0.0098),
        mesh_path=str(_OBJ_DIR / "cherry.obj"),
        shape_dr_ranges={"bend_deg": (-5.0, 5.0), "twist_deg": (-5.0, 5.0),
                         "taper": (-0.05, 0.05), "rbf": (0.0, 0.02),
                         "axis_scale": (0.92, 1.08), "scale": (0.85, 1.2)},
        material_dr_mult={"E": (0.6, 1.3), "nu": (0.95, 1.05), "rho": (0.85, 1.15),
                          "yield": (0.6, 1.4)},   # E top capped 1.5->1.3 (max 6.5e5) for MPM stability at 2 cm
        sim_substeps_override=480,   # E=5.0e5 (was 8.0e5, which exploded); 480 keeps margin at grid 300
    ),
    # chicken_breast: thick capsule lobe (deliberately chunkier than the real
    # flat fillet shape, per avoid-flat-profiles direction).
    "chicken_breast": ObjectDef(
        "chicken_breast", MATERIALS["chicken_breast_raw"], object_type="soft",
        size=(0.0497, 0.0497, 0.078), default_pos=(0.47, 0.0, 0.043),
        mesh_path=str(_OBJ_DIR / "chicken_breast.obj"),
        shape_dr_ranges={"bend_deg": (-8.0, 8.0), "twist_deg": (-6.0, 6.0),
                         "taper": (-0.1, 0.1), "rbf": (0.0, 0.03),
                         "axis_scale": (0.9, 1.1), "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.5, 1.6), "nu": (0.97, 1.03), "rho": (0.9, 1.1),
                          "yield": (0.4, 1.6)},
    ),
    # shrimp: bent-capsule PRIMITIVE APPROXIMATION of the curved body (no CAD
    # scan sourced -- see generate_fragile25_meshes.py's docstring).
    "shrimp": ObjectDef(
        "shrimp", MATERIALS["shrimp_raw"], object_type="soft",
        size=(0.0208, 0.0193, 0.0498), default_pos=(0.47, 0.0, 0.027),
        mesh_path=str(_OBJ_DIR / "shrimp.obj"),
        shape_dr_ranges={"bend_deg": (-15.0, 15.0), "twist_deg": (-10.0, 10.0),
                         "taper": (-0.1, 0.1), "axis_scale": (0.9, 1.1),
                         "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.5, 1.6), "nu": (0.97, 1.03), "rho": (0.9, 1.1),
                          "yield": (0.4, 1.6)},
    ),
    # scallop: short disc, kept TALLER than the real (fairly flat) shape per
    # avoid-flat-profiles direction.
    "scallop": ObjectDef(
        "scallop", MATERIALS["scallop_raw"], object_type="soft",
        size=(0.038, 0.038, 0.026), default_pos=(0.47, 0.0, 0.021),
        mesh_path=str(_OBJ_DIR / "scallop.obj"),
        shape_dr_ranges={"bend_deg": (-5.0, 5.0), "twist_deg": (-5.0, 5.0),
                         "taper": (-0.08, 0.08), "axis_scale": (0.9, 1.1),
                         "scale": (0.85, 1.2)},
        material_dr_mult={"E": (0.5, 1.6), "nu": (0.97, 1.03), "rho": (0.9, 1.1),
                          "yield": (0.4, 1.6)},
    ),
    # watermelon (cube of flesh, not whole fruit): primitive box like tofu/beef
    # -- no shape_dr_ranges (mesh_deform doesn't apply to a box).
    # E=5.36e5 is 1.79x mushroom's nominal -- see kiwi's note above.
    "watermelon": ObjectDef("watermelon", MATERIALS["watermelon"],
                            size=(0.04, 0.04, 0.045), default_pos=(0.47, 0.0, 0.0225),
                            sim_substeps_override=300),
    # cheese (mozzarella cube): primitive box like tofu/beef.
    "cheese": ObjectDef("cheese", MATERIALS["mozzarella"],
                        size=(0.035, 0.035, 0.04), default_pos=(0.47, 0.0, 0.02)),
    # dumpling: bent-capsule folded/crescent silhouette.
    "dumpling": ObjectDef(
        "dumpling", MATERIALS["dumpling_cooked"], object_type="soft",
        size=(0.036, 0.0352, 0.0576), default_pos=(0.47, 0.0, 0.032),
        mesh_path=str(_OBJ_DIR / "dumpling.obj"),
        shape_dr_ranges={"bend_deg": (-10.0, 10.0), "twist_deg": (-8.0, 8.0),
                         "taper": (-0.1, 0.1), "axis_scale": (0.9, 1.1),
                         "scale": (0.85, 1.15)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.9, 1.1),
                          "yield": (0.5, 1.5)},
    ),
    # pasta_bundle: irregular lumpy coil (rbf-bumped sphere).
    "pasta_bundle": ObjectDef(
        "pasta_bundle", MATERIALS["pasta_cooked"], object_type="soft",
        size=(0.057, 0.0469, 0.0442), default_pos=(0.47, 0.0, 0.031),
        mesh_path=str(_OBJ_DIR / "pasta_bundle.obj"),
        shape_dr_ranges={"bend_deg": (-5.0, 5.0), "twist_deg": (-5.0, 5.0),
                         "taper": (-0.05, 0.05), "rbf": (0.0, 0.05),
                         "axis_scale": (0.85, 1.15), "scale": (0.85, 1.2)},
        material_dr_mult={"E": (0.6, 1.5), "nu": (0.95, 1.05), "rho": (0.9, 1.1),
                          "yield": (0.5, 1.5)},
    ),
}


def get_object_def(name: str) -> ObjectDef:
    if name not in OBJECT_MAP:
        raise KeyError(f"Unknown object {name!r}; known objects: {sorted(OBJECT_MAP)}")
    return OBJECT_MAP[name]
