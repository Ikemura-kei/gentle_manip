"""Linear-FEM static grasp metric: the MINIMUM grip force that holds the object against gravity,
and the peak internal stress that grip induces. Genesis-free, ~ms per grasp.

Rationale (see grasp_synthesis/CLAUDE.md §11): lifting needs the grasp to resist only ONE wrench —
gravity — which two pads on soft food can do WITHOUT force closure. So we drop the Q_SM force-closure
machinery (the PSD stress-LMIs + convex-hull loop) and keep just the friction-cone force balance,
plus the FEM stress map B (contact force -> element stress) we already built. One small SOCP + a
matmul:

    min   Σ (nᵢ·fᵢ)                        # gentlest squeeze
    s.t.  G f = -w_gravity                 # contacts balance gravity  (FEASIBLE  <=> can hold it)
          ‖(I-nnᵀ)fᵢ‖ ≤ μ(nᵢ·fᵢ),  nᵢ·fᵢ≥0  # Coulomb friction cones
    then  σ = B f  ->  peak von Mises        # gentleness (lower = gentler)

Units: the FEM uses E=1. Stress under a PRESCRIBED FORCE is E-INDEPENDENT for a homogeneous linear-
elastic body (equilibrium fixes it; only Poisson ν enters), so σ = B f with E=1 IS the real stress
(Pa) for a real hold force f (N) — directly comparable to the object's yield stress.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse

from .metric import wrench_map
from .stressmap import contact_stress_map


def von_mises(sigma_voigt: np.ndarray) -> np.ndarray:
    """(…,6) Voigt (xx,yy,zz,xy,yz,zx) -> (…,) von Mises scalar."""
    s = np.asarray(sigma_voigt, float)
    xx, yy, zz, xy, yz, zx = (s[..., i] for i in range(6))
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
                   + 3.0 * (xy ** 2 + yz ** 2 + zx ** 2))


def hold_forces(points: np.ndarray, normals: np.ndarray, mu: float, w_ext: np.ndarray):
    """Min-grip contact forces that balance the external wrench w_ext (the grasp must produce
    G f = -w_ext), inside the friction cones. Returns (f (N,3) or None if infeasible, grip, status).
    grip = Σ nᵢ·fᵢ (total squeeze). Direct Clarabel (no cvxpy), like metric._support_clarabel."""
    import clarabel

    pts = np.asarray(points, float).reshape(-1, 3)
    nrm = np.asarray(normals, float).reshape(-1, 3)
    N = len(pts)
    n = 3 * N
    I3 = np.eye(3)

    A_blocks, b_parts, cones = [], [], []
    # (1) equilibrium  G f = -w_ext   ->  ZeroCone(6)
    A_blocks.append(wrench_map(pts)); b_parts.append(-np.asarray(w_ext, float).reshape(6))
    cones.append(clarabel.ZeroConeT(6))
    # (2) friction cones  s = (μ nᵢ·fᵢ ; (I-nnᵀ)fᵢ) ∈ SOC(4)  ->  A = -Mᵢ, b = 0
    for i in range(N):
        Mi = np.zeros((4, n)); cols = slice(3 * i, 3 * i + 3)
        Mi[0, cols] = mu * nrm[i]
        Mi[1:4, cols] = I3 - np.outer(nrm[i], nrm[i])
        A_blocks.append(-Mi); b_parts.append(np.zeros(4)); cones.append(clarabel.SecondOrderConeT(4))

    q = nrm.reshape(-1).copy()                                  # min Σ nᵢ·fᵢ = min qᵀf
    settings = clarabel.DefaultSettings(); settings.verbose = False
    sol = clarabel.DefaultSolver(sparse.csc_matrix((n, n)), q,
                                 sparse.csc_matrix(np.vstack(A_blocks)),
                                 np.concatenate(b_parts), cones, settings).solve()
    st = str(sol.status)
    if st not in ("Solved", "AlmostSolved"):
        return None, np.inf, st.lower()
    f = np.asarray(sol.x).reshape(N, 3)
    return f, float(nrm.reshape(-1) @ sol.x), "optimal" if st == "Solved" else "optimal_inaccurate"


def grasp_stress(obj, contacts, *, mass: float, g: float = 9.81,
                 gravity_dir=(0.0, 0.0, -1.0), accel: float = 0.0, mask_contact: bool = True,
                 B: Optional[np.ndarray] = None, tet_idx: Optional[np.ndarray] = None) -> dict:
    """Linear-FEM static grasp metric for `contacts` holding an object of `mass` kg against gravity
    (optionally + a vertical lift acceleration `accel` m/s²). Returns:
        holdable      bool   — the friction-cone force balance is feasible (can hold it)
        grip          float  — minimum total squeeze Σ nᵢ·fᵢ (N)
        stress_peak   float  — peak element von Mises (real Pa; E-independent, see module docstring)
        stress_top10  float  — mean of the top-10% most-stressed elements (robust to a single spike)
        stress_mean   float
        f             (N,3)  — the min-grip contact forces (None if not holdable)
    """
    pts = np.asarray(contacts.points, float)
    if B is None:
        B, tet_idx = contact_stress_map(obj.fem, pts)
    w_ext = np.zeros(6)
    w_ext[:3] = mass * (g + accel) * np.asarray(gravity_dir, float)   # gravity wrench at COM (=origin)

    f, grip, status = hold_forces(pts, contacts.normals, float(contacts.mu), w_ext)
    if f is None:
        return dict(holdable=False, grip=np.inf, stress_peak=np.inf, stress_top10=np.inf,
                    stress_mean=np.inf, f=None, status=status)

    sig = np.einsum("mij,j->mi", B, f.reshape(-1))               # (M,6) element Voigt stress
    vm = von_mises(sig)
    if mask_contact and tet_idx is not None:                     # drop the point-load singularity (§6.2)
        keep = np.ones(len(vm), bool); keep[np.unique(tet_idx)] = False
        vm = vm[keep]
    k = max(1, int(0.1 * len(vm)))
    return dict(holdable=True, grip=grip, stress_peak=float(vm.max()),
                stress_top10=float(np.sort(vm)[-k:].mean()), stress_mean=float(vm.mean()),
                f=f, sigma=sig, status=status)                   # sigma: full (M,6) field for rendering
