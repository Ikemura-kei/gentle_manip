"""Generate 6 mushroom mesh variants for 3D printing.

Grid: 3 scales (0.8, 1.0, 1.12) × 2 bend values (0°, 20°).
Exported as STL in millimetres (slicer-ready). OBJ copies also written for reference.

Usage:
    uv run --project envs/sim python examples/mushroom_meshes/generate.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent.parent
MESH_SRC = ROOT / "gentle_manip/assets/objects/mushroom.obj"
OUT_DIR = Path(__file__).resolve().parent

# DR bounds for reference (food_shape.yaml)
#   object_scale:    [0.8, 1.12]
#   object_bend_deg: [-25, 25]
# (scale, bend_deg) — all within DR bounds: scale [0.8,1.12], bend [-25,25] deg
SCALE_BEND_PAIRS = [
    (0.80,  20.0),   # smallest, forward arc
    (1.00,   0.0),   # nominal, straight
    (1.12, -18.0),   # largest, backward arc
    (0.85, -22.0),   # small, strong backward
    (0.95,  13.0),   # just-under-nominal, mild forward
    (1.08,  24.0),   # near-upper, near-max forward
]


def _axes(verts: np.ndarray):
    ext = verts.max(0) - verts.min(0)
    order = np.argsort(ext)[::-1]
    return int(order[0]), int(order[1]), int(order[2])


def apply_bend(verts: np.ndarray, beta_rad: float) -> np.ndarray:
    if abs(beta_rad) < 1e-6:
        return verts
    L, P, _ = _axes(verts)
    v = verts.copy()
    s = v[:, L] - v[:, L].mean()
    length = v[:, L].max() - v[:, L].min()
    kappa = beta_rad / length
    phi = kappa * s
    R = 1.0 / kappa
    p = v[:, P] - v[:, P].mean()
    v[:, L] = R * np.sin(phi) - p * np.sin(phi) + v[:, L].mean()
    v[:, P] = R * (1.0 - np.cos(phi)) + p * np.cos(phi) + v[:, P].mean() - R
    v[:, L] -= v[:, L].mean() - verts[:, L].mean()
    v[:, P] -= v[:, P].mean() - verts[:, P].mean()
    return v


def main():
    nominal = trimesh.load(str(MESH_SRC), process=False, force="mesh")
    print(f"Loaded {MESH_SRC.name}: {len(nominal.vertices)} verts, "
          f"AABB {((nominal.vertices.max(0)-nominal.vertices.min(0))*1000).round(1)} mm")

    for scale, bend_deg in SCALE_BEND_PAIRS:
        bend_rad = np.deg2rad(bend_deg)
        tag = f"scale{scale:.2f}_bend{bend_deg:.0f}deg"

        verts = nominal.vertices.copy() * scale
        verts = apply_bend(verts, bend_rad)

        mesh = trimesh.Trimesh(vertices=verts, faces=nominal.faces, process=False)

        # Validate
        assert mesh.volume > 0, f"Degenerate mesh for {tag}"

        ext_mm = (mesh.vertices.max(0) - mesh.vertices.min(0)) * 1000
        print(f"  {tag}: AABB {ext_mm.round(1)} mm  vol={mesh.volume*1e6:.2f} cm³")

        # Export STL in mm (multiply by 1000 so slicers see millimetres)
        mesh_mm = trimesh.Trimesh(vertices=verts * 1000, faces=nominal.faces, process=False)
        stl_path = OUT_DIR / f"mushroom_{tag}.stl"
        mesh_mm.export(str(stl_path))
        print(f"    → {stl_path.name}")

    print(f"\nDone — {len(SCALE_BEND_PAIRS)} meshes written to {OUT_DIR}")


if __name__ == "__main__":
    main()
