"""One-time generator for the fragile-food 25-category campaign's new base
meshes (gentle_manip, 2026-08-13). Per user direction: primitives with
literature-matched material (this script handles GEOMETRY only; materials.py
is separate), reserve real CAD sourcing only for genuinely special shapes,
and explicitly AVOID flat/thin profiles -- every shape here is built tall
enough for a parallel-jaw gripper to bite in its natural resting pose.

Each object is a trimesh primitive (icosphere/capsule/cone/cylinder) possibly
passed through mesh_deform.py's bend/taper/axis_scale ONCE at registration
time (not per-episode DR -- that's layered on top later via each object's own
shape_dr_ranges in registry.py) to get a distinctive nominal silhouette, then
exported to gentle_manip/assets/objects/<name>.obj.

Run once (envs/sim, needs trimesh):
    uv run --project envs/sim python -m gentle_manip.scripts.generate_fragile25_meshes
Then validate every output with repair_and_validate_mesh.py before registering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
OBJ_DIR = REPO / "gentle_manip" / "assets" / "objects"


def _export(mesh: trimesh.Trimesh, name: str) -> Path:
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    out = OBJ_DIR / f"{name}.obj"
    mesh.export(str(out))
    print(f"[generate] {name}: extents={mesh.extents.round(4).tolist()} "
         f"volume={mesh.volume:.2e} watertight={mesh.is_watertight} -> {out}")
    return out


def _recenter(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center at the AABB midpoint (matches the existing scanned meshes' convention
    -- scene_builder/DR code assumes the mesh's own origin is roughly its center)."""
    mesh.vertices -= mesh.bounds.mean(axis=0)
    return mesh


def make_strawberry() -> trimesh.Trimesh:
    """Cone, apex down -- a strawberry's natural tapered-to-a-point silhouette.
    ~3.2cm diameter at the shoulder, ~3.5cm tall standing on its tip."""
    m = trimesh.creation.cone(radius=0.016, height=0.035, sections=24)
    return _recenter(m)


def make_blackberry() -> trimesh.Trimesh:
    """Near-spherical drupelet cluster, same profile as the existing raspberry
    entry (same genus Rubus) -- rbf-bump shape DR (registry.py) adds the
    drupelet texture per-episode; the nominal is a plain sphere."""
    m = trimesh.creation.icosphere(radius=0.011, subdivisions=3)
    return _recenter(m)


def make_peach_slice() -> trimesh.Trimesh:
    """A THICK half cut from a whole peach (not a thin disc slice) -- user
    direction: avoid flat/thin profiles. A single axis-aligned plane cut keeps
    the tetrahedralizer's cap simple (a two-angled-plane wedge self-intersected
    at the seam); standing on the flat cut face still gives ~3cm of real
    height, narrowed slightly on one axis via axis_scale for a less-round,
    more slice-like silhouette."""
    from gentle_manip.assets import mesh_deform
    whole = trimesh.creation.icosphere(radius=0.030, subdivisions=3)   # ~6cm whole peach
    half = whole.slice_plane(plane_origin=[0, 0, 0], plane_normal=[1, 0, 0], cap=True)
    rng = np.random.default_rng(20260813)
    out = mesh_deform.deform_mesh(half, {"axis_scale": 0.65, "axis_scale_ax": 1}, rng)
    return _recenter(out)


def make_banana_piece() -> trimesh.Trimesh:
    """A ~5cm capsule segment (banana PIECE, not a thin round coin-slice) --
    resting on its cylindrical side gives ~3.2cm height (its diameter),
    graspable across that axis."""
    m = trimesh.creation.capsule(radius=0.016, height=0.025, count=(16, 16))
    return _recenter(m)


def make_shiitake() -> trimesh.Trimesh:
    """Distinct from the button-mushroom nominal: broader, flatter cap +
    thinner stem. Built from the EXISTING (already tet-meshable, already
    working) mushroom.obj via a one-time strong deform rather than a fresh
    primitive -- reuses proven geometry instead of risking a new degenerate
    mesh."""
    from gentle_manip.assets import mesh_deform
    src = OBJ_DIR / "mushroom.obj"
    m = trimesh.load(str(src), process=False, force="mesh")
    rng = np.random.default_rng(20260813)
    out = mesh_deform.deform_mesh(
        m, {"taper": 0.35, "axis_scale": 1.35, "axis_scale_ax": 0, "bend": 0.15}, rng)
    return _recenter(out)


def make_chicken_breast() -> trimesh.Trimesh:
    """Thick lobe -- a chunky capsule, not the real flat/thin fillet shape
    (user direction: avoid flat profiles for graspability)."""
    m = trimesh.creation.capsule(radius=0.025, height=0.028, count=(16, 16))
    return _recenter(m)


def make_shrimp() -> trimesh.Trimesh:
    """Curved/crescent body -- a thin capsule bent into a C via mesh_deform's
    bend() (captures the essential curved-body shrimp silhouette without
    needing a sourced scan). PRIMITIVE APPROXIMATION, not real CAD -- flagged
    per user direction as the fallback path when a real scan isn't sourced."""
    from gentle_manip.assets import mesh_deform
    m = trimesh.creation.capsule(radius=0.009, height=0.032, count=(16, 16))
    rng = np.random.default_rng(20260813)
    out = mesh_deform.deform_mesh(m, {"bend": 1.4, "taper": 0.25}, rng)
    return _recenter(out)


def make_scallop() -> trimesh.Trimesh:
    """Disc-shaped adductor muscle -- kept deliberately TALLER than the real
    (fairly flat) scallop shape per the avoid-flat-profiles direction."""
    m = trimesh.creation.cylinder(radius=0.019, height=0.026, sections=32)
    return _recenter(m)


def make_dumpling() -> trimesh.Trimesh:
    """Folded crescent/pouch -- a fat capsule bent into a half-moon."""
    from gentle_manip.assets import mesh_deform
    m = trimesh.creation.capsule(radius=0.017, height=0.022, count=(16, 16))
    rng = np.random.default_rng(20260813)
    out = mesh_deform.deform_mesh(m, {"bend": 1.1, "taper": 0.15}, rng)
    return _recenter(out)


def make_pasta_bundle() -> trimesh.Trimesh:
    """Irregular lumpy coil -- an icosphere with heavy one-time rbf bumps,
    lightly squashed (not flattened -- kept tall enough to grasp)."""
    from gentle_manip.assets import mesh_deform
    m = trimesh.creation.icosphere(radius=0.026, subdivisions=3)
    rng = np.random.default_rng(20260813)
    out = mesh_deform.deform_mesh(
        m, {"axis_scale": 0.85, "axis_scale_ax": 2, "rbf": 0.22, "rbf_n": 6}, rng)
    return _recenter(out)


GENERATORS = {
    "strawberry":     make_strawberry,
    "blackberry":     make_blackberry,
    "peach_slice":    make_peach_slice,
    "banana_piece":   make_banana_piece,
    "shiitake":       make_shiitake,
    "chicken_breast": make_chicken_breast,
    "shrimp":         make_shrimp,
    "scallop":        make_scallop,
    "dumpling":       make_dumpling,
    "pasta_bundle":   make_pasta_bundle,
}


def main() -> None:
    for name, fn in GENERATORS.items():
        mesh = fn()
        _export(mesh, name)


if __name__ == "__main__":
    main()
