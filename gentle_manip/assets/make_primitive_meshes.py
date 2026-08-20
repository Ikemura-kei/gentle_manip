"""Generate the analytic benchmark objects (cylinder, cube) as meshes in `assets/objects/`.

The grasp benchmark needs shape variety beyond the scanned mushroom: a smooth cylinder (curved,
no antipodal faces) and a sharp cube (flat antipodal faces) bracket the geometry axis, so a grasp
objective that only works on organic blobs is exposed.

Repo conventions this follows:
  * meshes are stored in METERS, so the registry uses scale=1.0 (assets/registry.py docstring);
  * the mesh origin is its CENTROID (matching mushroom.obj), so `default_pos.z` = |min z| + ~1 mm
    of clearance -- a soft body must spawn RESTING, since one dropped from height blows up in MPM;
  * watertight, since Genesis volume-samples MPM particles from the surface.

    uv run --project envs/sim python gentle_manip/assets/make_primitive_meshes.py

⚠️ The CUBE must be used with `prepare=False` in the FEM path: `prepare_mesh`'s watertight voxel
remesh ROUNDS SHARP EDGES (see demo_finger_grasp.py), which would erase the exact property the cube
is in the benchmark to test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

OBJ_DIR = Path(__file__).resolve().parent / "objects"


def _finalize(m: trimesh.Trimesh, name: str) -> trimesh.Trimesh:
    """Centre the mesh on its CENTROID and assert the invariants the sim relies on.

    Centroid-origin (not base-origin) is the existing convention — `mushroom.obj` sits at
    z in [-0.0148, +0.0199] — and it matters for two reasons:
      * `default_pos` positions the mesh ORIGIN, so a base-origin mesh spawned at
        default_pos.z = half-height would float a full half-height above the table and DROP;
        a soft body dropped from height blows up in MPM.
      * pose DR rotates the object about its origin, so a centroid origin keeps a randomized
        orientation centred on the object instead of swinging it about its base.
    The caller derives default_pos.z as |min z| + ~1 mm of clearance.
    """
    m.apply_translation(-m.centroid)
    trimesh.repair.fix_normals(m)
    assert m.is_watertight, f"{name}: not watertight — Genesis MPM sampling needs a closed surface"
    assert m.volume > 0, f"{name}: non-positive volume"
    assert np.allclose(m.centroid, 0.0, atol=1e-9), f"{name}: centroid is not at the origin"
    return m


def make_cylinder(radius=0.025, height=0.04, sections=64) -> trimesh.Trimesh:
    """Upright cylinder. Default r=2.5cm x h=4cm => 5cm diameter, comfortably inside the gripper's
    ~7.9cm usable opening while still being the widest object in the benchmark set."""
    return _finalize(trimesh.creation.cylinder(radius=radius, height=height, sections=sections),
                     "cylinder")


def make_cube(side=0.04) -> trimesh.Trimesh:
    """Sharp-edged cube. Default 4cm. Subdivided so the FEM tet mesher has enough surface samples
    to resolve contact on a face, while the edges stay exactly sharp."""
    m = trimesh.creation.box(extents=(side, side, side))
    m = m.subdivide().subdivide()            # denser faces, identical geometry (edges stay sharp)
    return _finalize(m, "cube")


def _report(m: trimesh.Trimesh, name: str, path: Path) -> None:
    ext = m.extents
    print(f"  {name:10s} -> {path.name}")
    print(f"     extents {np.round(ext, 4)} m   volume {m.volume * 1e6:8.2f} cm^3   "
          f"verts {len(m.vertices):5d} faces {len(m.faces):5d}")
    # default_pos positions the mesh ORIGIN (the centroid), so the spawn height that puts the
    # base ~1 mm above the table is |min z| + 0.001.
    print(f"     z-range [{m.bounds[0][2]:+.5f}, {m.bounds[1][2]:+.5f}]  "
          f"=> registry default_pos z = {abs(m.bounds[0][2]) + 0.001:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cylinder-radius", type=float, default=0.025)
    ap.add_argument("--cylinder-height", type=float, default=0.04)
    ap.add_argument("--cube-side", type=float, default=0.04)
    ap.add_argument("--out-dir", type=Path, default=OBJ_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[make_primitive_meshes] -> {args.out_dir}")
    for name, mesh in (("cylinder", make_cylinder(args.cylinder_radius, args.cylinder_height)),
                       ("cube4", make_cube(args.cube_side))):
        dst = args.out_dir / f"{name}.obj"
        if dst.exists() and not args.overwrite:
            print(f"  {name}: {dst.name} exists, skipping (--overwrite to replace)")
            continue
        mesh.export(str(dst))
        _report(mesh, name, dst)


if __name__ == "__main__":
    main()
