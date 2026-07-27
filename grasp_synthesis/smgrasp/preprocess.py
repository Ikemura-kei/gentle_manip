"""Mesh conditioning for FEM — turn arbitrary (scanned, non-watertight) meshes into a
clean, watertight, coarse-enough surface that TetGen can tetrahedralize quickly.

Scanned meshes (e.g. mushroom.obj: 25k faces, 76k UNMERGED verts, not watertight) hang or
explode TetGen. The fix is a robust remesh:
  1. merge duplicate vertices,
  2. if not watertight -> solid voxelization + marching cubes (guarantees watertight) + Taubin
     smoothing (undo the voxel blockiness),
  3. quadric-decimate to a target face count (controls boundary-constraint density ⇒ tet count),
  4. final cleanup (merge, fix normals, fill any residual holes).

`tet_switches` then picks TetGen's max-tet-volume so the tet count lands near a target — the
main knob for keeping the downstream FEM + SDP tractable (Q_SM is a bulk quantity and is
mesh-resolution-robust, so a few-thousand-tet mesh is plenty; see M8).
"""
from __future__ import annotations

from typing import Union

import numpy as np
import trimesh

from .geometry import load_mesh

MeshLike = Union[str, "trimesh.Trimesh"]


def prepare_mesh(mesh_or_path: MeshLike, *, voxel_div: int = 32, smooth_iters: int = 6,
                 decimate_to: int = 0, force_remesh: bool = False) -> trimesh.Trimesh:
    """Return a clean, watertight surface ready for tetrahedralization.

    Resolution is controlled by `voxel_div` (voxels across the longest extent) — this is the
    reliable knob: the marching-cubes output is manifold and non-self-intersecting, so TetGen
    accepts it. Post-hoc `simplify_quadric_decimation` (decimate_to>0, opt-in) is AVOIDED by
    default because it can introduce self-intersections that TetGen rejects; prefer a coarser
    voxel_div to reduce the mesh. The downstream tet COUNT is set separately by TetGen's max-tet
    volume (see tet_switches), not by the surface face count.
    """
    m = load_mesh(mesh_or_path).copy()
    m.merge_vertices()                                         # scans ship unmerged duplicate verts

    if force_remesh or not m.is_watertight:
        pitch = float(m.extents.max()) / max(voxel_div, 4)
        vox = m.voxelized(pitch=pitch).fill()                 # solid voxelization
        m = vox.marching_cubes                                # watertight surface (VOXEL-INDEX coords)
        m.apply_transform(vox.transform)                      # map back to WORLD coords (index->world)
        if smooth_iters:
            trimesh.smoothing.filter_taubin(m, iterations=smooth_iters)   # de-blockify
        m.merge_vertices()
        m.fix_normals()

    if decimate_to and len(m.faces) > decimate_to:            # opt-in; may break TetGen (see above)
        m = m.simplify_quadric_decimation(face_count=decimate_to)
        m.merge_vertices()
        m.fix_normals()
        if not m.is_watertight:
            m.fill_holes()
    return m


def crop_mesh(mesh_or_path: MeshLike, *, axis: int = 1, keep_frac: float = 0.6,
              keep: str = "above", cap: bool = True) -> trimesh.Trimesh:
    """Keep the top (`keep='above'`) or bottom `keep_frac` of the mesh along `axis`, capping the
    cut so the result stays watertight. Lets us focus the tet budget on a region of interest
    (e.g. the bunny head+ears) — finer resolution there without paying for the whole object.

    Requires `shapely` + `mapbox_earcut` for the cap triangulation. If the source surface is clean
    enough the cropped+capped mesh tetrahedralizes DIRECTLY (no voxel remesh), preserving detail."""
    m = load_mesh(mesh_or_path).copy()
    m.merge_vertices()
    lo, hi = float(m.bounds[0][axis]), float(m.bounds[1][axis])
    normal = np.zeros(3)
    if keep == "above":
        normal[axis] = 1.0
        level = hi - keep_frac * (hi - lo)
    else:
        normal[axis] = -1.0
        level = lo + keep_frac * (hi - lo)
    origin = np.zeros(3)
    origin[axis] = level
    cut = m.slice_plane(origin, normal, cap=cap)
    cut.merge_vertices()
    cut.fix_normals()
    return cut


def tet_switches(mesh: trimesh.Trimesh, *, target_tets: int = 5000, quality: float = 1.4) -> str:
    """TetGen switches whose max-tet-volume `a` aims for ~target_tets (actual count runs a few×
    higher). `quality` is the radius-edge ratio bound (lower = better-shaped, more tets)."""
    a = max(float(mesh.volume), 1e-30) / max(target_tets, 1)
    return f"pq{quality}a{a:.6e}"
