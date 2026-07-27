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


def support_point_active(obj, contacts, d, cfg: Optional[MetricConfig] = None, *,
                         B: np.ndarray, tet_idx: Optional[np.ndarray] = None,
                         n_seed: int = 40, max_iter: int = 40, tol: float = 1e-3,
                         solver: Optional[str] = None) -> dict:
    """M5b — support_point with the progressive active set (§5.4 / Algorithm 3).

    Most stress LMIs are inactive at the optimum, so instead of constraining all M elements we
    solve with a small working set, then repeatedly add the single most-stressed excluded element
    and re-solve until none violates the cap. Returns the SAME optimum as the full solve (the
    dropped constraints were slack) but with |working set| ≪ M — the tractability lever for meshes
    with thousands of tets. The initial set is a spatial spread (by centroid) so the first solve is
    bounded in every force direction."""
    from .fem import voigt_to_tensor

    cfg = cfg or MetricConfig()
    A = obj.A
    alle = np.arange(A.shape[0])
    if cfg.mask_contact_elems and tet_idx is not None:
        alle = np.setdiff1d(alle, np.unique(tet_idx))

    cent = obj.elem_centroids[alle]                           # spread the seed over the volume
    order = np.lexsort((cent[:, 2], cent[:, 1], cent[:, 0]))
    seed = alle[order[np.linspace(0, len(order) - 1, min(len(order), n_seed)).astype(int)]]
    S = set(int(j) for j in seed)

    res, it = None, 0
    for it in range(max_iter):
        res = support_point(obj, contacts, d, cfg, B=B, tet_idx=tet_idx,
                            elements=np.array(sorted(S)), solver=solver)
        if res["w"] is None:
            break
        L = (np.einsum("mij,j->mi", A[alle], res["w"])
             + np.einsum("mij,j->mi", B[alle], res["f"].reshape(-1)))
        smax = np.abs(np.linalg.eigvalsh(voigt_to_tensor(L))).max(axis=1)   # per-element max |σ|
        k = int(np.argmax(smax))
        if smax[k] <= 1.0 + tol or int(alle[k]) in S:         # nothing violates the cap -> optimal
            break
        S.add(int(alle[k]))
    if res is not None:
        res["active"], res["active_iters"] = sorted(S), it
    return res


def sample_sphere(n: int, dim: int = 6, seed: int = 0) -> np.ndarray:
    """n approximately-uniform unit directions on S^{dim-1}."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def q_sm(obj, contacts, cfg: Optional[MetricConfig] = None, *, n_dirs: Optional[int] = None,
         eps: Optional[float] = None, max_iter: int = 60, elements: Optional[np.ndarray] = None,
         stress_cap: bool = True, fn_limit: float = 1.0, active: bool = True,
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

    B, tet_idx = (contact_stress_map(obj.fem, np.asarray(contacts.points, float))
                  if stress_cap else (None, None))            # Q1 mode needs no stress map
    kw = dict(cfg=cfg, B=B, tet_idx=tet_idx, elements=elements,
              stress_cap=stress_cap, fn_limit=fn_limit)

    def support_y(d):                                          # support point in y = √W w space
        if stress_cap and active:
            r = support_point_active(obj, contacts, d, cfg, B=B, tet_idx=tet_idx)
        else:
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


def q1(obj, contacts, cfg: Optional[MetricConfig] = None, **kw):
    """Ferrari–Canny Q1 — the shape-BLIND force-closure metric: the same outer loop as q_sm but with
    the per-element stress LMIs replaced by a per-contact normal-force cap (nᵢ·fᵢ ≤ 1). Used in M7 to
    show that Q1 is symmetric where Q_SM is shape-aware."""
    return q_sm(obj, contacts, cfg, stress_cap=False, **kw)


def support_point(obj, contacts, d, cfg: Optional[MetricConfig] = None, *,
                  B: Optional[np.ndarray] = None, tet_idx: Optional[np.ndarray] = None,
                  elements: Optional[np.ndarray] = None, solver: Optional[str] = None,
                  stress_cap: bool = True, fn_limit: float = 1.0) -> dict:
    """Solve the support-point program for direction d. Returns dict(status, value, w, f, B, tet_idx).

    stress_cap=True (Q_SM): bound the wrench by the two-sided stress LMIs −I ⪯ A_j w + B_j f ⪯ I.
    stress_cap=False (Q1 / Ferrari–Canny): drop the stress LMIs and instead cap each contact's normal
    force nᵢ·fᵢ ≤ fn_limit — the shape-BLIND force-closure support point, for comparison (M7).

    B / tet_idx (the per-contact stress map, stress_cap only) are computed once if not supplied —
    pass them back in across directions to avoid recomputing. `elements` restricts the stress LMIs to
    a working set (active set); default = all elements minus contact-adjacent (§6.2)."""
    import cvxpy as cp

    cfg = cfg or MetricConfig()
    pts = np.asarray(contacts.points, float)
    normals = np.asarray(contacts.normals, float)
    mu = float(contacts.mu)
    N = len(pts)

    if stress_cap:
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
    if stress_cap:
        for j in elements:
            L = A[j] @ w + B[j] @ f                                # (6,) Voigt stress
            sig = cp.bmat([[L[0], L[3], L[5]], [L[3], L[1], L[4]], [L[5], L[4], L[2]]])
            cons += [sig + I3 >> 0, I3 - sig >> 0]                 # −I ⪯ σ ⪯ I
    else:
        for i in range(N):                                        # Q1 force normalization
            cons.append(normals[i] @ f[3 * i:3 * i + 3] <= fn_limit)

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
