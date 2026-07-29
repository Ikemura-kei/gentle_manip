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


# Voigt (xx,yy,zz,xy,yz,zx) -> Clarabel PSD-triangle svec (column-major upper triangle, off-diagonals
# scaled by √2): [S11, √2 S12, S22, √2 S13, √2 S23, S33]. σ = [[L0,L3,L5],[L3,L1,L4],[L5,L4,L2]].
_S2 = np.sqrt(2.0)
_VOIGT_TO_SVEC = np.array([
    [1, 0, 0, 0, 0, 0],          # S11 = L0
    [0, 0, 0, _S2, 0, 0],        # √2 S12 = √2 L3
    [0, 1, 0, 0, 0, 0],          # S22 = L1
    [0, 0, 0, 0, 0, _S2],        # √2 S13 = √2 L5
    [0, 0, 0, 0, _S2, 0],        # √2 S23 = √2 L4
    [0, 0, 1, 0, 0, 0],          # S33 = L2
], float)
_SVEC_I = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])   # svec of the 3x3 identity


def _support_clarabel(obj, contacts, d, cfg, *, B, tet_idx, elements, stress_cap, fn_limit, sqrtW):
    """Direct Clarabel solve of the support-point program — hand-assembled conic data, NO cvxpy
    canonicalization (which the profile showed to be ~96% of the cost). Same optimum as
    support_point's cvxpy path. Variables x = [w(6); f(3N)]; minimize -(√W d)ᵀw.

    Cones, in order: ZeroCone(6) wrench balance; per-contact SecondOrderCone(4) friction;
    then either PSDTriangleCone(3)×2 per active element (stress LMIs) or NonnegativeCone(N) (Q1)."""
    import clarabel
    from scipy import sparse

    pts = np.asarray(contacts.points, float)
    normals = np.asarray(contacts.normals, float)
    mu = float(contacts.mu)
    N = len(pts)
    n = 6 + 3 * N
    I3 = np.eye(3)

    A_blocks, b_parts, cones = [], [], []

    # (1) wrench balance  w + G f = 0  ->  [I6 | G] x + s = 0, s ∈ Zero(6)
    eq = np.zeros((6, n)); eq[:, :6] = np.eye(6); eq[:, 6:] = wrench_map(pts)
    A_blocks.append(eq); b_parts.append(np.zeros(6)); cones.append(clarabel.ZeroConeT(6))

    # (2) friction cones: s = (μ nᵢ·fᵢ ; (I-nnᵀ)fᵢ) ∈ SOC(4)  ->  A = -Mᵢ, b = 0
    for i in range(N):
        ni = normals[i]
        Mi = np.zeros((4, n)); cols = slice(6 + 3 * i, 6 + 3 * i + 3)
        Mi[0, cols] = mu * ni
        Mi[1:4, cols] = I3 - np.outer(ni, ni)
        A_blocks.append(-Mi); b_parts.append(np.zeros(4)); cones.append(clarabel.SecondOrderConeT(4))

    # (3) stress LMIs  -I ⪯ σ ⪯ I  (svec form), two PSD blocks per active element
    if stress_cap:
        A = obj.A
        for j in elements:
            Lmap = np.empty((6, n)); Lmap[:, :6] = A[j]; Lmap[:, 6:] = B[j]  # L = A_j w + B_j f
            P = _VOIGT_TO_SVEC @ Lmap                                        # svec(σ) is linear in x
            A_blocks.append(-P); b_parts.append(_SVEC_I.copy())              # svec(σ+I) ⪰ 0
            cones.append(clarabel.PSDTriangleConeT(3))
            A_blocks.append(P); b_parts.append(_SVEC_I.copy())              # svec(I-σ) ⪰ 0
            cones.append(clarabel.PSDTriangleConeT(3))
    else:                                                                    # Q1: nᵢ·fᵢ ≤ fn_limit
        nn = np.zeros((N, n))
        for i in range(N):
            nn[i, 6 + 3 * i:6 + 3 * i + 3] = normals[i]
        A_blocks.append(nn); b_parts.append(np.full(N, fn_limit)); cones.append(clarabel.NonnegativeConeT(N))

    Amat = sparse.csc_matrix(np.vstack(A_blocks))
    bvec = np.concatenate(b_parts)
    q = np.concatenate([-(sqrtW @ np.asarray(d, float).reshape(6)), np.zeros(3 * N)])
    Pmat = sparse.csc_matrix((n, n))

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    sol = clarabel.DefaultSolver(Pmat, q, Amat, bvec, cones, settings).solve()
    st = str(sol.status)
    # normalize Clarabel status -> the cvxpy vocabulary the callers/tests use
    status = {"Solved": "optimal", "AlmostSolved": "optimal_inaccurate"}.get(st, st.lower())
    ok = st in ("Solved", "AlmostSolved")                # AlmostSolved == cvxpy's optimal_inaccurate (usable)
    x = np.asarray(sol.x) if ok else None
    return {
        "status": status,
        "value": None if x is None else float(-q[:6] @ x[:6]),
        "w": None if x is None else x[:6].copy(),
        "f": None if x is None else x[6:].reshape(N, 3),
        "B": B,
        "tet_idx": tet_idx,
    }


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
    n_dirs = max(n_dirs or cfg.n_dirs, 7)                      # need >= dim+1 points for a 6-D hull
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
    a working set (active set); default = all elements minus contact-adjacent (§6.2).

    Backend: `solver=None`/"clarabel" uses the hand-assembled direct-Clarabel path (default, fast —
    no cvxpy canonicalization); any other value (e.g. "cvxpy") forces the cvxpy build (kept for
    validation / other solvers)."""
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

    sqrtW = np.real(sqrtm(cfg.wrench_metric()))
    if solver in (None, "clarabel", "CLARABEL"):
        return _support_clarabel(obj, contacts, d, cfg, B=B, tet_idx=tet_idx, elements=elements,
                                 stress_cap=stress_cap, fn_limit=fn_limit, sqrtW=sqrtW)

    import cvxpy as cp
    obj_vec = sqrtW @ np.asarray(d, float).reshape(6)
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
    prob.solve(solver=cp.CLARABEL)     # `solver` here only selects the backend (cvxpy); cvxpy uses Clarabel
    return {
        "status": prob.status,
        "value": prob.value,
        "w": None if w.value is None else np.asarray(w.value),
        "f": None if f.value is None else np.asarray(f.value).reshape(N, 3),
        "B": B,
        "tet_idx": tet_idx,
    }
