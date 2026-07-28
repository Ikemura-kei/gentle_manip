"""M10 — grasp pose → ContactSet (grasp_synthesis/CLAUDE.md §4).

`sample_contacts` turns a parallel-jaw grasp (a center, a closing axis, and the pad footprint) into
contact points on the OBJECT surface with object-surface normals (into the material) — the input the
Q_SM metric consumes. Each of the two pads casts a small grid of rays inward from outside the object;
the first surface hit on each side is a contact. The friction-cone normal always comes from the
object, never the finger. Contacts are returned in the object's COM frame (pass `com`) so they line
up with the recentered ElasticObject.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import trimesh

from .geometry import load_mesh
from .types import ContactSet

MeshLike = Union[str, "trimesh.Trimesh"]


def _basis(axis: np.ndarray):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(a, t); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return a, u, v


def _pad_offsets(pad_half: float, n: int) -> np.ndarray:
    m = max(1, int(np.ceil(np.sqrt(n))))
    g = np.linspace(-pad_half, pad_half, m)
    gu, gv = np.meshgrid(g, g)
    return np.stack([gu.ravel(), gv.ravel()], axis=-1)[:n]


def sample_contacts(obj_mesh: MeshLike, center, closing_axis, *, pad_half: float = 0.015,
                    mu: float = 0.5, n_per_patch: int = 6, com: Optional[np.ndarray] = None
                    ) -> Optional[ContactSet]:
    """Parallel-jaw grasp -> ContactSet, or None if < 2 contacts / a degenerate (single-side) grasp.

    center / closing_axis are in the object mesh frame; the two pads sit on opposite sides along the
    closing axis. `pad_half` is the half-width of the square finger footprint. Rays are cast from far
    outside toward the object so the result is robust to the exact pad standoff / gripper width."""
    mesh = load_mesh(obj_mesh)
    a, u, v = _basis(closing_axis)
    center = np.asarray(center, float)
    R = 2.0 * float(mesh.extents.max())
    offs = _pad_offsets(pad_half, n_per_patch)

    pts, nrm, sides = [], [], []
    for sign in (+1.0, -1.0):
        origins = center + sign * a * R + offs[:, 0:1] * u[None] + offs[:, 1:2] * v[None]
        dirs = np.tile(-sign * a, (len(origins), 1))
        locs, ridx, tidx = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        for loc, ti in zip(locs, tidx):
            pts.append(loc)
            nrm.append(-mesh.face_normals[ti])              # normal INTO the material
            sides.append(sign)

    if len(pts) < 2 or len(set(sides)) < 2:                 # need both jaws to touch
        return None
    pts = np.asarray(pts, float)
    if com is not None:
        pts = pts - np.asarray(com, float)                  # object mesh frame -> COM frame
    return ContactSet(points=pts, normals=np.asarray(nrm, float), mu=mu)
