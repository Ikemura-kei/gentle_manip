"""Stage 2 gate: the FEM-readiness check from docs/final/adding_new_objects.md §2.

For a prepped, registry-ready mesh, checks:
  - meta["direct_tet"] is True (no voxel-remesh dilation fallback)
  - the FEM tet-mesh extents match the source mesh extents (ratio ~= 1.00, no silent dilation)
  - tet count stays <= TET_COUNT_MULT * target_tets (else tetgen refinement is exploding on a
    sharp/degenerate surface -> collection-time risk of a tetgen hang, per the 2026-09-07 lesson)

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.fem_gate_check \
        gentle_manip/assets/objects/<name>.obj
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(REPO_ROOT / "grasp_synthesis")):
    if p not in sys.path:
        sys.path.insert(0, p)

TARGET_TETS = 1500
TET_COUNT_MULT = 3.0
EXTENT_RATIO_TOL = 0.03   # +-3% counts as "no dilation"


def check(mesh_path: str, target_tets: int = TARGET_TETS) -> dict:
    import numpy as np
    import trimesh
    from smgrasp.finger_grasp_final import build_grasp_fem

    raw = trimesh.load(mesh_path, force="mesh")
    raw_ext = raw.extents
    t0 = time.time()
    obj, pad_geo, meta = build_grasp_fem(mesh_path, target_tets=target_tets)
    dt = time.time() - t0

    fem_ext = obj.verts.max(0) - obj.verts.min(0)
    ratio = fem_ext / np.maximum(raw_ext, 1e-9)
    ratio_max_dev = float(np.max(np.abs(ratio - 1.0)))

    ok_direct = bool(meta["direct_tet"])
    ok_ratio = ratio_max_dev <= EXTENT_RATIO_TOL
    ok_tetcount = meta["tets"] <= TET_COUNT_MULT * target_tets
    passed = ok_direct and ok_ratio and ok_tetcount

    return {
        "mesh_path": mesh_path, "build_time_s": round(dt, 2), "direct_tet": ok_direct,
        "tets": meta["tets"], "tet_cap": int(TET_COUNT_MULT * target_tets), "ok_tetcount": ok_tetcount,
        "raw_extents_mm": [round(float(x) * 1000, 1) for x in raw_ext],
        "fem_extents_mm": [round(float(x) * 1000, 1) for x in fem_ext],
        "extent_ratio_max_dev": round(ratio_max_dev, 4), "ok_ratio": ok_ratio,
        "ndof": meta["ndof"], "passed": passed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh_path")
    ap.add_argument("--target-tets", type=int, default=TARGET_TETS)
    args = ap.parse_args()
    r = check(args.mesh_path, target_tets=args.target_tets)
    for k, v in r.items():
        print(f"  {k}: {v}")
    print("\nGATE " + ("PASS" if r["passed"] else "FAIL"))
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
