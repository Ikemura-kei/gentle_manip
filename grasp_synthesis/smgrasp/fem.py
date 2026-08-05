"""M3 — linear-elastic FEM: stiffness, free-body (inertia-relief) solve, stress recovery.

Constant-strain linear tetrahedra. Normalized units E = 1; Lamé from ν:
    μ = 1 / (2(1+ν)),   λ = ν / ((1+ν)(1−2ν)).

The object is FLOATING, so K has a 6-D rigid-body null space R (3 translations + 3
rotations). A plain solve is ill-posed; use the bordered / inertia-relief system,
factored once (grasp_synthesis/CLAUDE.md §3.3, §6.1):

    [ K   R ] [u]   [b]
    [ Rᵀ  0 ] [α] = [0]

Every load column is then one back-substitution. For a wrench-balanced combined load the
multipliers α vanish and the recovered stress is physical; never interpret one unbalanced
column's displacement on its own.

Voigt order: [σxx, σyy, σzz, σxy, σyz, σzx]; strains use engineering shear (γ = 2ε).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def lame(nu: float, E: float = 1.0) -> Tuple[float, float]:
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def elasticity_matrix(nu: float, E: float = 1.0) -> np.ndarray:
    """Isotropic Voigt C (6×6), engineering-shear convention."""
    lam, mu = lame(nu, E)
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def shape_gradients(verts: np.ndarray, tets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """∇N per node for every tet: grad (M,4,3), and tet volumes vol (M,)."""
    p = verts[tets]                                                # (M,4,3)
    E = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=-1)  # (M,3,3) cols=edges
    vol = np.abs(np.linalg.det(E)) / 6.0
    invE = np.linalg.inv(E)                                        # rows -> ∇λ1,∇λ2,∇λ3
    grad = np.empty((len(tets), 4, 3))
    grad[:, 1:, :] = invE
    grad[:, 0, :] = -invE.sum(axis=1)
    return grad, vol


def _B_matrices(grad: np.ndarray) -> np.ndarray:
    """Strain-displacement B (M,6,12) from node shape-function gradients grad (M,4,3)."""
    M = grad.shape[0]
    B = np.zeros((M, 6, 12))
    for n in range(4):
        bx, by, bz = grad[:, n, 0], grad[:, n, 1], grad[:, n, 2]
        c = 3 * n
        B[:, 0, c + 0] = bx
        B[:, 1, c + 1] = by
        B[:, 2, c + 2] = bz
        B[:, 3, c + 0] = by; B[:, 3, c + 1] = bx
        B[:, 4, c + 1] = bz; B[:, 4, c + 2] = by
        B[:, 5, c + 0] = bz; B[:, 5, c + 2] = bx
    return B


class FEM:
    """Assembles K, the rigid modes, and the bordered inertia-relief factorization."""

    def __init__(self, verts: np.ndarray, tets: np.ndarray, nu: float = 0.33):
        self.verts = np.asarray(verts, float)
        self.tets = np.asarray(tets, np.int64)
        self.nu = nu
        self.n = len(self.verts)
        self.ndof = 3 * self.n
        self.C = elasticity_matrix(nu)
        self.grad, self.vol = shape_gradients(self.verts, self.tets)
        self.B = _B_matrices(self.grad)                            # (M,6,12)
        self.K = self._assemble_stiffness()
        self.R = self._rigid_modes()                              # (ndof, 6), orthonormal
        self._factor = None                                       # lazy bordered factorization

    # ── assembly ──────────────────────────────────────────────────────────────
    def _assemble_stiffness(self) -> sparse.csc_matrix:
        Ke = self.vol[:, None, None] * np.einsum("mki,kl,mlj->mij", self.B, self.C, self.B)  # (M,12,12)
        dof = (3 * self.tets[:, :, None] + np.arange(3)[None, None, :]).reshape(len(self.tets), 12)
        rows = np.broadcast_to(dof[:, :, None], Ke.shape).reshape(-1)
        cols = np.broadcast_to(dof[:, None, :], Ke.shape).reshape(-1)
        K = sparse.coo_matrix((Ke.reshape(-1), (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()
        return 0.5 * (K + K.T)

    def _rigid_modes(self) -> np.ndarray:
        x = self.verts
        R = np.zeros((self.ndof, 6))
        for d in range(3):                                        # 3 translations
            R[d::3, d] = 1.0
        # 3 infinitesimal rotations: displacement at node = a × x
        for a in range(3):
            axis = np.zeros(3); axis[a] = 1.0
            disp = np.cross(axis[None, :], x)                     # (n,3)
            R[:, 3 + a] = disp.reshape(-1)
        Q, _ = np.linalg.qr(R)                                    # orthonormalize for conditioning
        return Q

    # ── free-body (inertia-relief) solve ───────────────────────────────────────
    def _ensure_factor(self):
        if self._factor is None:
            n, m = self.ndof, self.R.shape[1]
            Rs = sparse.csc_matrix(self.R)
            top = sparse.hstack([self.K, Rs])
            bot = sparse.hstack([Rs.T, sparse.csc_matrix((m, m))])
            self._factor = splu(sparse.vstack([top, bot]).tocsc())

    def solve_free(self, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inertia-relief solve for load(s) b (ndof,) or (ndof, k). Returns (u, alpha)."""
        self._ensure_factor()
        b = np.asarray(b, float)
        single = b.ndim == 1
        B = b.reshape(self.ndof, -1)
        m = self.R.shape[1]
        rhs = np.vstack([B, np.zeros((m, B.shape[1]))])
        sol = self._factor.solve(rhs)
        u, alpha = sol[: self.ndof], sol[self.ndof:]
        return (u.ravel(), alpha.ravel()) if single else (u, alpha)

    def solve_pinned(self, fixed_dofs: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Standard Dirichlet solve with `fixed_dofs` held at zero (for the pinned-bar test)."""
        free = np.setdiff1d(np.arange(self.ndof), np.asarray(fixed_dofs))
        Kff = self.K[free][:, free]
        u = np.zeros(self.ndof)
        u[free] = sparse.linalg.spsolve(Kff.tocsc(), np.asarray(b, float)[free])
        return u

    def solve_dirichlet(self, fixed_dofs: np.ndarray, u_fixed: np.ndarray,
                        b: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Dirichlet solve with `fixed_dofs` PRESCRIBED to `u_fixed` (non-zero allowed) — the position/
        width-controlled contact: push the pad-contact nodes in by the commanded indentation and solve.
        Fixing both jaws anchors the object (no rigid null space), so this is a plain reduced solve.
        Returns (u, reaction): reaction is non-zero only on fixed_dofs = the force the pad must exert to
        impose the indentation (i.e. the grip force). Displacement u is E-INDEPENDENT (K∝E cancels), so
        stress σ=element_stress(u) and reaction both scale linearly with E — evaluate at E=1 and scale."""
        fixed = np.asarray(fixed_dofs, np.int64)
        u_fix = np.asarray(u_fixed, float)
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        rhs = np.zeros(self.ndof) if b is None else np.asarray(b, float).copy()
        u = np.zeros(self.ndof); u[fixed] = u_fix
        Kfc = self.K[free][:, fixed]
        u[free] = sparse.linalg.spsolve(self.K[free][:, free].tocsc(), rhs[free] - Kfc @ u_fix)
        reaction = np.zeros(self.ndof)
        reaction[fixed] = np.asarray(self.K[fixed] @ u).ravel() - rhs[fixed]
        return u, reaction

    def solve_constrained(self, C: sparse.spmatrix, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Minimum-elastic-energy displacement subject to the LINEAR constraints `C u = g`, with the
        rigid-body null space bordered out. Use for NORMAL-ONLY pad contact: each row of C constrains
        one contact node's displacement along the closing axis (`aᵀuᵢ = indentᵢ`), leaving the two
        tangential components free so the soft object can bulge under the pad (real compression, no
        clamped-flat edge singularity). Solves the KKT system
            [ K    Cᵀ   R ] [u]     [0]
            [ C    0    0 ] [λ]  =  [g]
            [ Rᵀ   0    0 ] [β]     [0]
        Returns (u, λ): λ is the per-constraint reaction (the normal contact force at each node = the
        grip). R (the 6 rigid modes) pins the tangential/rotational null freedoms left by C."""
        n, m = self.ndof, self.R.shape[1]
        Csp = sparse.csc_matrix(C); nc = Csp.shape[0]
        Rs = sparse.csc_matrix(self.R)
        top = sparse.hstack([self.K, Csp.T, Rs])
        mid = sparse.hstack([Csp, sparse.csc_matrix((nc, nc)), sparse.csc_matrix((nc, m))])
        bot = sparse.hstack([Rs.T, sparse.csc_matrix((m, nc)), sparse.csc_matrix((m, m))])
        A = sparse.vstack([top, mid, bot]).tocsc()
        rhs = np.concatenate([np.zeros(n), np.asarray(g, float), np.zeros(m)])
        sol = splu(A).solve(rhs)
        return sol[:n], sol[n:n + nc]

    def solve_constrained_fast(self, C: sparse.spmatrix, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Same result as `solve_constrained` (min-energy displacement s.t. C u = g, rigid modes pinned)
        but REUSING the cached inertia-relief factor of [K R; Rᵀ 0] (built once) instead of
        refactorizing the full bordered KKT every call — the grasp-planner speedup.

        With G = the inertia-relief solve operator (u-block of the cached factor, so M[G f; β]=[f;0]),
        stationarity K u + Cᵀλ + Rβ = 0 gives u = −G Cᵀ λ; imposing C u = g:
            W = G Cᵀ         (nc back-substitutions through the ONE cached factor)
            S = C W          (nc×nc dense Schur matrix, SPD for independent non-rigid constraints)
            λ = solve(S, −g),   u = −W λ.
        Cost per grasp: one multi-RHS back-substitution + a small dense solve — no refactorization."""
        Csp = sparse.csc_matrix(C)
        W, _ = self.solve_free(np.asarray(Csp.T.todense()))      # (ndof, nc) = G Cᵀ, reuses cache
        S = np.asarray((Csp @ W))                                # (nc, nc)
        lam = np.linalg.solve(S, -np.asarray(g, float))
        return -(W @ lam), lam

    # ── stress recovery ────────────────────────────────────────────────────────
    def element_stress(self, u: np.ndarray) -> np.ndarray:
        """Per-element Voigt stress σ (M,6) from a global displacement u (ndof,)."""
        dof = (3 * self.tets[:, :, None] + np.arange(3)[None, None, :]).reshape(len(self.tets), 12)
        ue = u[dof]                                               # (M,12)
        eps = np.einsum("mij,mj->mi", self.B, ue)                # (M,6) engineering strain
        return eps @ self.C.T                                     # (M,6) Voigt stress


def voigt_to_tensor(sig: np.ndarray) -> np.ndarray:
    """(…,6) Voigt -> (…,3,3) symmetric stress tensor."""
    s = np.asarray(sig, float)
    xx, yy, zz, xy, yz, zx = [s[..., i] for i in range(6)]
    T = np.stack([xx, xy, zx, xy, yy, yz, zx, yz, zz], axis=-1)
    return T.reshape(*s.shape[:-1], 3, 3)
