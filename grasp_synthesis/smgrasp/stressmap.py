"""M4 — affine per-element stress maps: σ(x_j) = A_j w + B_j f.

Compose the body-force map P (M2) and the FEM inertia-relief solve + stress recovery (M3):

    A_j = Σ_j ∘ solve_free ∘ (body-load basis Lb) ∘ P        # (6 Voigt) × (6 wrench)
    B_j = Σ_j ∘ solve_free ∘ (contact-load basis Lc)         # (6 Voigt) × (3N contacts)

Because solve_free and the stress operator Σ are LINEAR, A w + B f equals a single direct
FEM solve of the combined load — and since rigid-body motion carries zero strain, the stress
is unaffected by the inertia-relief multipliers (they only fix the rigid part of u). That is
the M4 superposition invariant. A is per-object (precomputed once); B is per contact set
(rebuilt per grasp — §9.3), so only its contact columns are re-solved.

Load bases (origin at COM):
  Lb (3n, 12): body load from the 12-dim (g0, vec G) basis.
      g0_d column:  ∫_Ω N_i e_d dx  = (Σ_{e∋i} V_e/4) e_d
      G_ck column:  ∫_Ω N_i e_c x_k dx,  ∫_e N_i x_k dx = V_e/20 (Psum_e,k + p_{i,k})
  Lc (3n, 3N): each contact force is distributed to its containing tet's nodes via the
      tet's shape functions (barycentric weights).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .fem import FEM


def body_load_basis(verts: np.ndarray, tets: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Lb (3n, 12): nodal load per unit of the (g0(3), vecG(9)) body-force basis (row-major vecG)."""
    n = len(verts)
    Lb = np.zeros((3 * n, 12))
    p = verts[tets]                                            # (M,4,3)
    Psum = p.sum(axis=1)                                       # (M,3)
    for a in range(4):                                         # local node a -> global gi
        gi = tets[:, a]
        # g0 columns 0,1,2:  add V_e/4 to dof (3 gi + d), column d
        w0 = vol / 4.0                                         # (M,)
        for d in range(3):
            np.add.at(Lb[:, d], 3 * gi + d, w0)
        # G columns 3 + 3c + k:  add V_e/20 (Psum_k + p_{a,k}) to dof (3 gi + c)
        coef = (vol[:, None] / 20.0) * (Psum + p[:, a, :])    # (M,3)  over k
        for c in range(3):
            for k in range(3):
                np.add.at(Lb[:, 3 + 3 * c + k], 3 * gi + c, coef[:, k])
    return Lb


def locate_points_in_tets(verts: np.ndarray, tets: np.ndarray, points: np.ndarray,
                          tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """For each point, the index of a containing tet and its barycentric weights (N,4).
    Points must lie inside/on the mesh. Raises if a point is not located."""
    p = verts[tets]                                           # (M,4,3)
    E = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=-1)
    invE = np.linalg.inv(E)                                   # (M,3,3)
    pts = np.asarray(points, float).reshape(-1, 3)
    diff = pts[:, None, :] - p[None, :, 0, :]                 # (N,M,3)
    b123 = np.einsum("mij,nmj->nmi", invE, diff)             # (N,M,3)
    b0 = 1.0 - b123.sum(-1)                                   # (N,M)
    bary = np.concatenate([b0[..., None], b123], axis=-1)     # (N,M,4)
    inside = (bary >= -tol).all(-1)                           # (N,M)
    tet_idx = np.full(len(pts), -1, np.int64)
    out_bary = np.zeros((len(pts), 4))
    for i in range(len(pts)):
        cand = np.where(inside[i])[0]
        if len(cand) == 0:                                    # snap: least-negative bary
            cand = [int(np.argmax(bary[i].min(axis=1)))]
        m = int(cand[0])
        tet_idx[i] = m
        out_bary[i] = np.clip(bary[i, m], 0.0, 1.0)
        out_bary[i] /= out_bary[i].sum()
    return tet_idx, out_bary


def contact_load_basis(verts: np.ndarray, tets: np.ndarray, points: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Lc (3n, 3N) and the per-contact containing-tet indices (for §6.2 masking)."""
    tet_idx, bary = locate_points_in_tets(verts, tets, points)
    N = len(tet_idx)
    Lc = np.zeros((3 * len(verts), 3 * N))
    for c in range(N):
        nodes = tets[tet_idx[c]]                              # (4,)
        for a in range(4):
            for d in range(3):
                Lc[3 * nodes[a] + d, 3 * c + d] += bary[c, a]
    return Lc, tet_idx


def body_stress_map(fem: FEM, P: np.ndarray, Lb: np.ndarray) -> np.ndarray:
    """A (M, 6, 6): per-element Voigt stress per unit wrench."""
    U, _ = fem.solve_free(Lb @ P)                             # (ndof, 6)
    M = len(fem.tets)
    A = np.zeros((M, 6, 6))
    for w in range(6):
        A[:, :, w] = fem.element_stress(U[:, w])
    return A


def contact_stress_map(fem: FEM, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """B (M, 6, 3N) per-element Voigt stress per unit contact-force component, + tet indices."""
    Lc, tet_idx = contact_load_basis(fem.verts, fem.tets, points)
    U, _ = fem.solve_free(Lc)                                 # (ndof, 3N)
    M = len(fem.tets)
    B = np.zeros((M, 6, U.shape[1]))
    for j in range(U.shape[1]):
        B[:, :, j] = fem.element_stress(U[:, j])
    return B, tet_idx
