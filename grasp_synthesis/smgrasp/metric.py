"""M5 — the support-point problem (SDP + SOCP), grasp_synthesis/CLAUDE.md §3.5.

For a unit wrench direction d, find the farthest resistible wrench along √W d:

    maximize   (√W d)ᵀ w
    over       w ∈ ℝ⁶,  f ∈ ℝ^{3N}
    s.t.       w + G(x) f = 0                                  # wrench balance (Eq. 1)
               ‖(I − nᵢnᵢᵀ) fᵢ‖ ≤ μ (nᵢᵀ fᵢ)                    # friction cones (SOC)
               −I ⪯ (A_j w + B_j f) ⪯ I   ∀ j ∈ S             # stress ≤ σ_max = 1 (two-sided LMI)

G(x) maps contact forces to their net wrench (force ; torque). The stress LMIs are what bound
the problem; S is the working element set (all elements, minus the contact-adjacent ones §6.2,
or an active subset — see support_point_active). Solved with cvxpy + Clarabel (SOCP+SDP).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import sqrtm

from .stressmap import contact_stress_map
from .types import MetricConfig


def skew(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])


def wrench_map(points: np.ndarray) -> np.ndarray:
    """G (6, 3N): net wrench of the contact forces, G @ f = (Σ fᵢ ; Σ xᵢ × fᵢ)."""
    pts = np.asarray(points, float).reshape(-1, 3)
    G = np.zeros((6, 3 * len(pts)))
    for i, x in enumerate(pts):
        G[0:3, 3 * i:3 * i + 3] = np.eye(3)
        G[3:6, 3 * i:3 * i + 3] = skew(x)
    return G


def sample_sphere(n: int, dim: int = 6, seed: int = 0) -> np.ndarray:
    """n approximately-uniform unit directions on S^{dim-1}."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def q_sm(obj, contacts, cfg: Optional[MetricConfig] = None, *, n_dirs: Optional[int] = None,
         eps: Optional[float] = None, max_iter: int = 60, elements: Optional[np.ndarray] = None,
         return_details: bool = False):
    """M6 — the stress-minimization grasp metric Q_SM (grasp_synthesis/CLAUDE.md §3.5).

    Q_SM = radius of the largest origin-centred √W-ball contained in the convex hull of the
    resistible wrenches. Compute support points w(d) for sampled directions, hull them in the
    y = √W w metric (so the ball is Euclidean), take the min origin→facet distance, then refine
    incrementally (Zheng): push a new support point along the closest facet's outward normal and
    re-hull until the inradius converges. Q_SM > 0 ⇔ force closure.
    """
    from scipy.spatial import ConvexHull

    cfg = cfg or MetricConfig()
    n_dirs = n_dirs or cfg.n_dirs
    eps = eps if eps is not None else cfg.eps
    sqrtW = np.real(sqrtm(cfg.wrench_metric()))

    B, tet_idx = contact_stress_map(obj.fem, np.asarray(contacts.points, float))
    kw = dict(cfg=cfg, B=B, tet_idx=tet_idx, elements=elements)

    def support_y(d):                                          # support point in y = √W w space
        r = support_point(obj, contacts, d, **kw)
        return None if r["w"] is None else sqrtW @ r["w"]

    ys = [y for d in sample_sphere(n_dirs) if (y := support_y(d)) is not None]
    ys = list(np.asarray(ys))
    if len(ys) < 7:                                            # need >= dim+1 points for a 6D hull
        return (-np.inf, {"reason": "insufficient support points"}) if return_details else -np.inf

    # Incremental refinement (Zheng): repeatedly take the facet CLOSEST to the origin and push a
    # new support point along its outward normal. Q = signed origin→facet distance grows monotonically
    # and converges to the true inradius (>0 iff the grasp is force closure) once the closest facet can
    # no longer be extended. This must continue even while Q<0 (a sparse initial hull may exclude the
    # origin before enough support points have been added).
    Q = -np.inf
    for it in range(max_iter):
        try:
            hull = ConvexHull(np.asarray(ys))
        except Exception:                                     # degenerate hull -> lower-dim W (no closure)
            return (-np.inf, {"reason": "degenerate hull", "iters": it}) if return_details else -np.inf
        A, b = hull.equations[:, :-1], hull.equations[:, -1]  # A·y + b <= 0 inside
        nn = np.linalg.norm(A, axis=1)
        dist = -b / nn                                        # signed origin→facet distances
        k = int(np.argmin(dist))                              # facet closest to the origin
        Q = float(dist[k])
        dk = A[k] / nn[k]                                     # unit outward normal
        y = support_y(dk)
        if y is None:
            break
        if float(dk @ y) <= Q + eps:                          # facet k is a true boundary of W -> done
            break
        ys.append(y)                                          # else extend the hull along dk
    return (Q, {"iters": it, "n_points": len(ys)}) if return_details else Q


def support_point(obj, contacts, d, cfg: Optional[MetricConfig] = None, *,
                  B: Optional[np.ndarray] = None, tet_idx: Optional[np.ndarray] = None,
                  elements: Optional[np.ndarray] = None, solver: Optional[str] = None) -> dict:
    """Solve the support-point SDP for direction d. Returns dict(status, value, w, f, B, tet_idx).

    B / tet_idx (the per-contact stress map) are computed once if not supplied — pass them back in
    across many directions (same contacts) to avoid recomputing. `elements` restricts the stress
    constraints to a working set (active set); default = all elements minus contact-adjacent (§6.2)."""
    import cvxpy as cp

    cfg = cfg or MetricConfig()
    pts = np.asarray(contacts.points, float)
    normals = np.asarray(contacts.normals, float)
    mu = float(contacts.mu)
    N = len(pts)

    if B is None:
        B, tet_idx = contact_stress_map(obj.fem, pts)
    A = obj.A
    if elements is None:
        elements = np.arange(A.shape[0])
    if cfg.mask_contact_elems and tet_idx is not None:
        elements = np.setdiff1d(elements, np.unique(tet_idx))     # drop contact-adjacent (§6.2)

    obj_vec = np.real(sqrtm(cfg.wrench_metric())) @ np.asarray(d, float).reshape(6)
    G = wrench_map(pts)
    I3 = np.eye(3)

    w = cp.Variable(6)
    f = cp.Variable(3 * N)
    cons = [w + G @ f == 0]
    for i in range(N):
        ni, fi = normals[i], f[3 * i:3 * i + 3]
        cons.append(cp.SOC(mu * (ni @ fi), (I3 - np.outer(ni, ni)) @ fi))
    for j in elements:
        L = A[j] @ w + B[j] @ f                                    # (6,) Voigt stress
        sig = cp.bmat([[L[0], L[3], L[5]], [L[3], L[1], L[4]], [L[5], L[4], L[2]]])
        cons += [sig + I3 >> 0, I3 - sig >> 0]                     # −I ⪯ σ ⪯ I

    prob = cp.Problem(cp.Maximize(obj_vec @ w), cons)
    prob.solve(solver=solver or cp.CLARABEL)
    return {
        "status": prob.status,
        "value": prob.value,
        "w": None if w.value is None else np.asarray(w.value),
        "f": None if f.value is None else np.asarray(f.value).reshape(N, 3),
        "B": B,
        "tet_idx": tet_idx,
    }
