"""Mesh QA gate for the object-asset library (cross-category Stage 1).

Validates a candidate object mesh against the ACTUAL downstream consumers, not a
generic watertightness check: `trimesh.is_watertight` is too strict (the existing,
already-working `mushroom.obj` fails it) and too permissive in a different way (it
says nothing about tet-meshability, which is what `grasp_synthesis/smgrasp` and
Genesis's MPM voxelizer actually need). So the gate is:

  1. tet-meshable — tetgen (via `smgrasp.geometry.tetrahedralize`, the same function
     the FEM grasp-quality metric depends on) must succeed. If it fails on the raw
     mesh, retry once after a trimesh repair pass (fill_holes/fix_normals/
     fix_winding) and, only if that repair changes the mesh, save the repaired copy.
  2. gripper-feasible — the mesh's smallest extent must leave room for the XArm7
     gripper to close around it (DEFAULT_GRIPPER_WIDTH minus a safety margin).
  3. sane volume/scale — positive volume, extents in a plausible food-object range
     (a few mm to ~20 cm); catches degenerate or wildly mis-scaled source files.

`is_watertight` is reported for information only and never gates pass/fail.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.repair_and_validate_mesh \
        gentle_manip/assets/objects/raspberry.stl [more meshes...]
    uv run --project envs/sim python -m gentle_manip.scripts.repair_and_validate_mesh \
        gentle_manip/assets/objects/ --repair-dir /tmp/repaired
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_GRASP_DIR = _REPO / "grasp_synthesis"
for _p in (str(_REPO), str(_GRASP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import trimesh  # noqa: E402

from gentle_manip.robot import xarm7_config as robot_cfg  # noqa: E402
from smgrasp.geometry import tetrahedralize  # noqa: E402

MIN_EXTENT_M = 0.005    # 5 mm — below this, likely a bad-scale/degenerate mesh
MAX_EXTENT_M = 0.20     # 20 cm — above this, unlikely to be a single-hand food item
GRIPPER_MARGIN_M = 0.01  # safety margin subtracted from the gripper's open width


@dataclass
class MeshReport:
    path: Path
    ok: bool
    is_watertight: bool          # informational only, does not gate `ok`
    tet_meshable: bool
    gripper_feasible: bool
    volume_cm3: float
    extents_cm: tuple
    n_verts: int
    n_tets: int
    repaired: bool
    reason: str = ""


def _try_tetrahedralize(mesh: trimesh.Trimesh):
    verts, tets = tetrahedralize(mesh)
    return verts, tets


def validate_mesh(path: Path, gripper_stroke: float = robot_cfg.DEFAULT_GRIPPER_WIDTH,
                  margin: float = GRIPPER_MARGIN_M,
                  repair_dir: Optional[Path] = None) -> MeshReport:
    mesh = trimesh.load(str(path), process=False, force="mesh")
    is_watertight = bool(mesh.is_watertight)
    repaired = False

    try:
        verts, tets = _try_tetrahedralize(mesh)
    except Exception:
        # Repair pass: fill holes, fix normals/winding, retry once.
        mesh.fill_holes()
        mesh.fix_normals()
        try:
            verts, tets = _try_tetrahedralize(mesh)
            repaired = True
        except Exception as e:
            extents_cm = tuple((mesh.extents * 100).round(3).tolist())
            return MeshReport(path, False, is_watertight, False, False,
                              float(mesh.volume * 1e6), extents_cm,
                              len(mesh.vertices), 0, False,
                              reason=f"tetrahedralization failed even after repair: {e}")

    extents = mesh.extents
    extents_cm = tuple((extents * 100).round(3).tolist())
    volume_cm3 = float(mesh.volume * 1e6)

    gripper_feasible = bool(extents.min() < (gripper_stroke - margin))
    size_ok = bool((extents.min() >= MIN_EXTENT_M) and (extents.max() <= MAX_EXTENT_M))
    volume_ok = volume_cm3 > 0

    ok = gripper_feasible and size_ok and volume_ok
    reasons = []
    if not gripper_feasible:
        reasons.append(f"min extent {extents.min()*100:.2f}cm too large for gripper "
                       f"stroke {gripper_stroke*100:.1f}cm (margin {margin*100:.1f}cm)")
    if not size_ok:
        reasons.append(f"extents {extents_cm} outside plausible food range "
                       f"[{MIN_EXTENT_M*100:.1f}, {MAX_EXTENT_M*100:.1f}] cm")
    if not volume_ok:
        reasons.append("non-positive volume")

    if repaired and repair_dir is not None:
        repair_dir.mkdir(parents=True, exist_ok=True)
        out_path = repair_dir / path.name
        mesh.export(str(out_path))
        reasons.append(f"repaired copy written to {out_path}")

    return MeshReport(path, ok, is_watertight, True, gripper_feasible, volume_cm3,
                      extents_cm, len(verts), len(tets), repaired, reason="; ".join(reasons))


def _iter_mesh_paths(inputs: list[Path]) -> list[Path]:
    exts = {".obj", ".stl", ".ply"}
    paths: list[Path] = []
    for p in inputs:
        if p.is_dir():
            paths.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in exts))
        else:
            paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meshes", type=Path, nargs="+",
                    help="mesh file(s) or a directory of meshes to validate")
    ap.add_argument("--gripper-stroke", type=float, default=robot_cfg.DEFAULT_GRIPPER_WIDTH,
                    help=f"max gripper open width in meters (default {robot_cfg.DEFAULT_GRIPPER_WIDTH})")
    ap.add_argument("--margin", type=float, default=GRIPPER_MARGIN_M,
                    help=f"safety margin subtracted from gripper stroke (default {GRIPPER_MARGIN_M})")
    ap.add_argument("--repair-dir", type=Path, default=None,
                    help="if a mesh needs repair to become tet-meshable, write the repaired "
                         "copy here (original files are never modified in place)")
    args = ap.parse_args()

    paths = _iter_mesh_paths(args.meshes)
    if not paths:
        print("No mesh files found.")
        sys.exit(1)

    all_ok = True
    for p in paths:
        r = validate_mesh(p, args.gripper_stroke, args.margin, args.repair_dir)
        all_ok &= r.ok
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {p.name:28s} watertight={r.is_watertight!s:6s} "
             f"tet_meshable={r.tet_meshable!s:6s} repaired={r.repaired!s:6s} "
             f"vol={r.volume_cm3:8.3f}cm3 extents_cm={r.extents_cm} "
             f"verts={r.n_verts} tets={r.n_tets}")
        if r.reason:
            print(f"         {r.reason}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
