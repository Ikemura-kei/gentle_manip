"""M1 — mesh loading, tetrahedralization, and geometry moments.

All formulas downstream assume the origin is at the COM (grasp_synthesis/CLAUDE.md §6.3),
so build_elastic_object recenters the tet mesh and asserts ‖COM‖ ≈ 0 afterward.

Per-tet closed forms (exact, independent of the tetrahedralization since they integrate
over a partition of Ω):
    V_t   = |det(v1-v0, v2-v0, v3-v0)| / 6
    ∫_t x dV      = V_t * centroid_t          (centroid = mean of the 4 verts)
    ∫_t x xᵀ dV   = V_t/20 * ( Psum Psumᵀ + Σ_a p_a p_aᵀ ),  Psum = Σ_a p_a
derived from ∫_t λ_a λ_b dV = V_t (1+δ_ab)/20 with x = Σ_a λ_a p_a.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import trimesh

from .types import ElasticObject, MetricConfig

MeshLike = Union[str, "trimesh.Trimesh"]


def load_mesh(mesh_or_path: MeshLike) -> trimesh.Trimesh:
    if isinstance(mesh_or_path, trimesh.Trimesh):
        return mesh_or_path
    return trimesh.load(str(mesh_or_path), process=False, force="mesh")


def tetrahedralize(mesh: trimesh.Trimesh, *, order: int = 1, mindihedral: float = 10.0,
                   minratio: float = 1.5, switches: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Surface mesh -> (nodes (N,3), tets (M,4)) via TetGen. Needs a watertight surface."""
    import tetgen

    tg = tetgen.TetGen(np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
    if switches is not None:
        tg.tetrahedralize(switches=switches)
    else:
        tg.tetrahedralize(order=order, mindihedral=mindihedral, minratio=minratio)
    nodes = np.asarray(tg.node, dtype=float)
    elems = np.asarray(tg.elem, dtype=np.int64)[:, :4]     # order=1 -> 4 nodes/tet
    return nodes, elems


def tet_volumes(verts: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p = verts[tets]                                         # (M, 4, 3)
    return np.abs(np.einsum("ij,ij->i",
                            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                            p[:, 3] - p[:, 0])) / 6.0


def tet_centroids(verts: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return verts[tets].mean(axis=1)


def geometry_moments(verts: np.ndarray, tets: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (volume, com, second_moment S = ∫ x xᵀ dx) about the ORIGIN of `verts`."""
    p = verts[tets]                                         # (M, 4, 3)
    vol = tet_volumes(verts, tets)                          # (M,)
    V = float(vol.sum())
    cent = p.mean(axis=1)                                   # (M, 3)
    com = (vol[:, None] * cent).sum(0) / V                  # ∫x dV / V
    Psum = p.sum(axis=1)                                    # (M, 3)  Σ_a p_a
    outer_self = np.einsum("mai,maj->mij", p, p)            # (M,3,3) Σ_a p_a p_aᵀ
    S_tet = (vol / 20.0)[:, None, None] * (
        np.einsum("mi,mj->mij", Psum, Psum) + outer_self)   # (M,3,3)
    S = S_tet.sum(0)
    return V, com, 0.5 * (S + S.T)                          # symmetrize (kill float asymmetry)


_Q2_A, _Q2_B = 0.5854101966249685, 0.1381966011250105     # 4-pt degree-2-exact tet rule


def tet_quadrature(verts: np.ndarray, tets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Degree-2-exact tetra quadrature: points (M*4, 3), weights (M*4,) summing to |Ω|.
    Exact for polynomials up to degree 2 — enough to integrate a linear body force AND
    the quadratic x × g torque integrand independently of the closed-form maps (M2 test)."""
    p = verts[tets]                                        # (M, 4, 3)
    bary = np.full((4, 4), _Q2_B)
    np.fill_diagonal(bary, _Q2_A)                          # (q, a) barycentric of the 4 q-points
    pts = np.einsum("qa,mad->mqd", bary, p).reshape(-1, 3)
    w = np.repeat(tet_volumes(verts, tets) / 4.0, 4)
    return pts, w


def build_elastic_object(mesh_or_path: MeshLike, cfg: Optional[MetricConfig] = None,
                         **tet_kwargs) -> ElasticObject:
    """M1: tetrahedralize, recenter to COM, compute volume + second moment.
    (M4 later fills the affine stress maps A, B on the returned object.)"""
    cfg = cfg or MetricConfig()
    mesh = load_mesh(mesh_or_path)
    verts, tets = tetrahedralize(mesh, **tet_kwargs)

    V, com, _ = geometry_moments(verts, tets)
    verts_c = verts - com                                  # recenter so ∫ x dx = 0 (§6.3)
    V2, com2, S = geometry_moments(verts_c, tets)
    scale = float(np.cbrt(max(V, 1e-30)))                  # characteristic length
    assert np.linalg.norm(com2) < 1e-8 * max(scale, 1.0), f"recenter failed: ‖COM‖={np.linalg.norm(com2):.2e}"

    return ElasticObject(
        verts=verts_c, tets=tets, volume=V2, com=com, second_moment=S, nu=cfg.nu,
        elem_centroids=tet_centroids(verts_c, tets),
    )
