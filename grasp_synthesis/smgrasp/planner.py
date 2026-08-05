"""M9 — stochastic (CMA-ES) grasp planner maximizing Q_SM (grasp_synthesis/CLAUDE.md §9).

Searches a parallel-jaw grasp (center + closing-axis direction) to maximize the stress-minimization
metric. The ElasticObject (the expensive FEM precompute) is built ONCE; each CMA-ES evaluation only
samples contacts for the candidate pose and calls q_sm (which reuses the factorized FEM and the
active set, so a single eval is ~Q1 cost — see §9.3). Infeasible poses (a jaw misses, or < 3
contacts) get a fixed penalty so the search is pulled onto the contact manifold.

This is the drop-in replacement for the hand-tuned SDF grasp-quality objective: keep your feasibility
penalties (penetration, table, reach) and add `- w_qsm * Q_SM(contacts(x))`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .contact import sample_contacts
from .metric import q_sm
from .types import MetricConfig


def _axis(theta: float, phi: float) -> np.ndarray:
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def grasp_contacts(mesh, x, *, com=None, pad_half=0.015, mu=0.5, n_per_patch=6):
    """5-DOF grasp vector x = [cx, cy, cz, theta, phi] -> ContactSet (or None)."""
    return sample_contacts(mesh, x[:3], _axis(x[3], x[4]), pad_half=pad_half, mu=mu,
                           n_per_patch=n_per_patch, com=com)


def plan_width_grasp(obj, mesh, *, E=None, density=None, mu=None, dr=None, hold_frac: float = 1.0,
                     w_align=None, w_peak=None, x0=None, sigma: float = 0.3, maxfevals: int = 200,
                     n_starts: int = 4, pad_half: float = 0.015, g: float = 9.81, accel: float = 0.0,
                     gravity_dir=(0, 0, -1.0), max_indent: float = 0.01, seed: int = 0,
                     refine: bool = True, n_refine: int = 10, refine_scan: int = 25,
                     verbose: bool = False, record_history: bool = False) -> dict:
    """Width-controlled gentle-grasp synthesis (grasp_synthesis/CLAUDE.md §11). CMA-ES over the 6-DOF
    candidate x = [cx, cy, cz, theta, phi, width], MAXIMIZING the gentleness score. Degenerate /
    no-contact / not-holdable candidates skip the FEM via the cheap pre-filter (BIG_NEG). Multi-start
    over diverse closing axes. Returns dict(x, score, ..., evals[, history]).

    `dr` (a list of (E, mass, μ) samples from `width_grasp.make_dr`) → DR mode: score = −mean stress if
    the grasp holds in ≥ `hold_frac` of the samples (best OVERALL pose). Otherwise nominal mode with the
    fixed (E, density, μ)."""
    import cma

    from .width_grasp import score_candidate, score_candidate_dr, is_real_grasp, _perp_basis
    from .viz import boundary_faces
    # Work in the RECENTERED (COM-at-origin) frame — the same frame `indent_from_width` uses on
    # obj.verts. Seeding from mesh.bounds (original frame) misses entirely when the COM ≠ origin
    # (e.g. a cropped, offset mesh) → all candidates miss → CMA-ES sees a flat objective and quits.
    lo, hi = obj.verts.min(0), obj.verts.max(0)
    center0 = np.zeros(3)                                          # the COM (origin) — robust seed
    ext = float((hi - lo).max())

    best = {"score": -np.inf, "x": None, "res": None}
    history, feasible, n_eval = [], [], [0]
    tag = {"round": 1, "pose": None}                             # current stage, for history labelling
    aln = {}                                                      # weight overrides (None -> module defaults)
    if w_align is not None: aln["w_align"] = w_align
    if w_peak is not None: aln["w_peak"] = w_peak

    def _score(x):                                                # evaluate one candidate -> res dict
        if dr is not None:
            return score_candidate_dr(obj, x[:3], _axis(x[3], x[4]), float(x[5]), pad_half=pad_half,
                                      dr=dr, g=g, accel=accel, max_indent=max_indent,
                                      hold_frac=hold_frac, **aln)
        return score_candidate(obj, x[:3], _axis(x[3], x[4]), float(x[5]), pad_half=pad_half, E=E,
                               density=density, mu=mu, g=g, accel=accel, gravity_dir=gravity_dir,
                               max_indent=max_indent, **aln)

    def _consider(x, res):                                        # track global best + feasible pool
        if is_real_grasp(res["score"]):
            feasible.append((np.asarray(x, float).copy(), res["score"]))
            if res["score"] > best["score"]:
                best.update(score=res["score"], x=np.asarray(x, float).copy(), res=res)

    def cost(x):                                                  # cma MINIMIZES -> return −score
        n_eval[0] += 1
        res = _score(x)
        _consider(x, res)
        if record_history and res["status"] == "ok" and is_real_grasp(res["score"]):
            history.append({"eval": n_eval[0], "score": res["score"], "x": np.asarray(x, float).copy(),
                            "res": res, "round": tag["round"], "pose": tag["pose"]})
        return -res["score"]

    bounds = [[lo[0], lo[1], lo[2], 0.0, 0.0, 0.005],
              [hi[0], hi[1], hi[2], np.pi, 2 * np.pi, 1.1 * ext]]
    # Seed each restart at a SHALLOW, near-optimal width for its axis (just past first contact along
    # that axis), not a fixed deep width — else CMA-ES wanders off the axis into tilted local optima
    # and misses the axis-aligned (e.g. face) grasp, which is a stable but narrow optimum.
    bverts = obj.verts[np.unique(boundary_faces(obj.tets)[0])]

    def _seed_width(th, ph, indent=1.5e-3):
        a, u1, u2 = _perp_basis(_axis(th, ph))
        d = bverts - center0
        proj = d @ a; foot = (np.abs(d @ u1) < pad_half) & (np.abs(d @ u2) < pad_half)   # SQUARE pad
        L, R = foot & (proj < 0), foot & (proj > 0)
        if L.sum() < 3 or R.sum() < 3:
            return 0.9 * ext
        cross = proj[R].max() - proj[L].min()                     # width at first contact along axis
        return float(np.clip(cross - 2 * indent, 0.005, 1.1 * ext))

    starts = [np.asarray(x0, float)] if x0 is not None else []
    for th, ph in _axis_seeds(max(n_starts - len(starts), 0), seed):
        starts.append(np.array([*center0, th, ph, _seed_width(th, ph)]))   # start near-closed (0.9·extent)
    per = max(maxfevals // max(len(starts), 1), 20)
    for i, s0 in enumerate(starts):
        cost(s0)
        es = cma.CMAEvolutionStrategy(list(s0), sigma,
                                      {"maxfevals": per, "bounds": bounds, "seed": seed + i,
                                       "verbose": -9 if not verbose else 1})
        es.optimize(cost)

    # ── ROUND 2: for each candidate POSE, do a deterministic 1-D WIDTH scan (reliable — a pose's stress
    # is only minimal at the right width). Candidate poses = the top DISTINCT round-1 poses PLUS the
    # canonical axis seeds (center=COM, axis=x/y/z/diagonals): a good centred face pose is only good at
    # its best width, so round 1 never ranks it high enough to be picked — scanning the seeds directly
    # guarantees the flush centred grasp is evaluated. (A width scan beats a 6-DOF local polish here: it
    # optimises width precisely instead of spreading the budget over all dims.)
    if refine and feasible:
        picked = _distinct_poses(feasible, n_refine, axis_deg=25.0, center_thr=0.12 * ext)
        seed_poses = [(center0.copy(), th, ph) for th, ph in _axis_seeds(n_starts, seed)]
        tag["round"] = 2
        for pi, (c, th, ph) in enumerate(picked + seed_poses):
            tag["pose"] = pi
            cross = _seed_width(th, ph, indent=0.0)               # width at first contact along this axis
            for w in np.linspace(0.4 * cross, cross, refine_scan):
                x2 = np.array([*c, th, ph, w]); res = _score(x2); n_eval[0] += 1
                _consider(x2, res)
                if record_history and res["status"] == "ok" and is_real_grasp(res["score"]):
                    history.append({"eval": n_eval[0], "score": res["score"], "x": x2.copy(),
                                    "res": res, "round": 2, "pose": pi})

    r = best["res"] or {}
    out = {"x": best["x"], "score": best["score"], "evals": n_eval[0], "n_starts": len(starts),
           "stress_top10": r.get("stress_top10", r.get("stress_mean")),   # DR -> mean stress
           "grip": r.get("grip", r.get("grip_mean")), "hold_frac": r.get("hold_frac"),
           "align": r.get("align")}
    if record_history:
        out["history"] = history
    return out


def _distinct_poses(feasible, n, *, axis_deg, center_thr):
    """Top-scoring but spatially DISTINCT poses (center, theta, phi) from the round-1 feasible pool for
    round-2 width refinement — two poses count as the same if their centers are within `center_thr` AND
    their closing axes within `axis_deg`. Returns up to `n` (center, theta, phi) tuples."""
    cos_thr = np.cos(np.radians(axis_deg))
    picked = []
    for x, _ in sorted(feasible, key=lambda f: -f[1]):
        c, ax = x[:3], _axis(x[3], x[4])
        if all(not (np.linalg.norm(c - pc) < center_thr and abs(ax @ _axis(pth, pph)) > cos_thr)
               for pc, pth, pph in picked):
            picked.append((c, float(x[3]), float(x[4])))
            if len(picked) >= n:
                break
    return picked


def _axis_seeds(n_starts: int, seed: int):
    """Diverse closing-axis (theta, phi) seeds for multi-start. Canonical axes first (x, y, z=vertical,
    diagonals) so distinct grasp basins — e.g. the gentle vertical stem↔cap grasp (theta≈0) that a
    single horizontal start misses — are each seeded; then seeded-random directions for the remainder."""
    canon = [(np.pi / 2, 0.0), (np.pi / 2, np.pi / 2), (0.0, 0.0), (np.pi / 3, np.pi / 4),
             (np.pi / 2, np.pi / 4), (np.pi / 3, 1.25 * np.pi), (np.pi / 4, 0.0), (2 * np.pi / 3, 0.0)]
    seeds = list(canon[:n_starts])
    if n_starts > len(canon):
        rng = np.random.default_rng(seed)
        seeds += [(rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)) for _ in range(n_starts - len(canon))]
    return seeds


def plan_lift_grasp(obj, mesh, *, mass, x0=None, sigma: float = 0.3, maxfevals: int = 200,
                    n_starts: int = 4, pad_half: float = 0.015, mu: float = 0.5, n_per_patch: int = 6,
                    g: float = 9.81, gravity_dir=(0.0, 0.0, -1.0), accel: float = 0.0,
                    stress_key: str = "stress_top10", no_contact_penalty: float = 1e12,
                    no_hold_penalty: float = 1e9, seed: int = 0, verbose: bool = False,
                    record_history: bool = False) -> dict:
    """Task-based (Genesis-free) grasp synthesis: MINIMIZE the linear-FEM stress of the gentlest grip
    that still holds the object against gravity (grasp_synthesis/CLAUDE.md §11). Unlike `plan_grasp`
    (which maximizes force-closure Q_SM), this drops force closure — a lift only needs to resist ONE
    wrench — so it works on soft food where Q_SM degenerates to ~0.

    MULTI-START: runs `n_starts` CMA-ES restarts from diverse closing-axis seeds (splitting the eval
    budget) and keeps the global best — a single start misses distinct grasp basins. Penalty ladder
    pulls the search onto the feasible manifold: no/too-few contacts >> can't-hold-gravity >>
    (feasible) stress. Returns dict(x, stress, grip, contacts, evals[, history])."""
    import cma

    from .geometry import load_mesh
    from .lift_stress import grasp_stress
    mesh = load_mesh(mesh)
    lo, hi = mesh.bounds
    com = obj.com
    center = 0.5 * (lo + hi)

    best = {"stress": np.inf, "x": None, "cs": None, "grip": np.inf}
    history, n_eval = [], [0]

    def cost(x):
        n_eval[0] += 1
        cs = grasp_contacts(mesh, x, com=com, pad_half=pad_half, mu=mu, n_per_patch=n_per_patch)
        if cs is None or cs.n_contacts < 3:
            return no_contact_penalty
        r = grasp_stress(obj, cs, mass=mass, g=g, gravity_dir=gravity_dir, accel=accel)
        if not r["holdable"]:
            return no_hold_penalty
        s = float(r[stress_key])
        if s < best["stress"]:
            best.update(stress=s, x=np.asarray(x, float).copy(), cs=cs, grip=r["grip"])
        if record_history:
            history.append({"eval": n_eval[0], "stress": s, "stress_best": best["stress"],
                            "grip": r["grip"], "x": np.asarray(x, float).copy(), "contacts": cs})
        return s

    bounds = [[lo[0], lo[1], lo[2], 0.0, 0.0], [hi[0], hi[1], hi[2], np.pi, 2 * np.pi]]
    # build the start list: explicit x0 (if given) + diverse-axis seeds to fill n_starts
    starts = [np.asarray(x0, float)] if x0 is not None else []
    for th, ph in _axis_seeds(max(n_starts - len(starts), 0), seed):
        starts.append(np.array([*center, th, ph]))
    per = max(maxfevals // max(len(starts), 1), 20)
    for i, s0 in enumerate(starts):
        cost(s0)                                            # seed so best is never empty if s0 holds
        es = cma.CMAEvolutionStrategy(list(s0), sigma,
                                      {"maxfevals": per, "bounds": bounds, "seed": seed + i,
                                       "verbose": -9 if not verbose else 1})
        es.optimize(cost)

    out = {"x": best["x"], "stress": best["stress"], "grip": best["grip"], "contacts": best["cs"],
           "evals": n_eval[0], "n_starts": len(starts)}
    if record_history:
        out["history"] = history
    return out


def plan_grasp(obj, mesh, cfg: Optional[MetricConfig] = None, *, x0=None, sigma: float = 0.3,
               maxfevals: int = 200, pad_half: float = 0.015, mu: float = 0.5, n_per_patch: int = 6,
               n_dirs: int = 12, penalty: float = 10.0, seed: int = 0, verbose: bool = False,
               record_history: bool = False) -> dict:
    """Maximize Q_SM over parallel-jaw poses with CMA-ES. Returns dict(x, q_sm, contacts, evals).
    record_history=True also returns `history`: one entry per FEASIBLE evaluation
    {eval, q, q_best, x, contacts} — for visualizing the search."""
    import cma

    from .geometry import load_mesh
    mesh = load_mesh(mesh)
    lo, hi = mesh.bounds
    com = obj.com

    best = {"q": -np.inf, "x": None, "cs": None}               # track the best FEASIBLE grasp directly
    history, n_eval = [], [0]

    def neg_qsm(x):
        n_eval[0] += 1
        cs = grasp_contacts(mesh, x, com=com, pad_half=pad_half, mu=mu, n_per_patch=n_per_patch)
        if cs is None or cs.n_contacts < 3:
            return penalty
        q = q_sm(obj, cs, cfg, n_dirs=n_dirs)
        if not np.isfinite(q):
            return penalty
        if q > best["q"]:
            best.update(q=float(q), x=np.asarray(x, float).copy(), cs=cs)
        if record_history:
            history.append({"eval": n_eval[0], "q": float(q), "q_best": best["q"],
                            "x": np.asarray(x, float).copy(), "contacts": cs})
        return -q

    if x0 is None:
        x0 = np.array([*(0.5 * (lo + hi)), np.pi / 2, 0.0])   # object centre, horizontal closing axis
    bounds = [[lo[0], lo[1], lo[2], 0.0, 0.0], [hi[0], hi[1], hi[2], np.pi, 2 * np.pi]]
    neg_qsm(np.asarray(x0, float))                            # seed with x0 so best is never empty if x0 is valid
    es = cma.CMAEvolutionStrategy(list(x0), sigma,
                                  {"maxfevals": maxfevals, "bounds": bounds, "seed": seed,
                                   "verbose": -9 if not verbose else 1})
    es.optimize(neg_qsm)
    out = {"x": best["x"], "q_sm": best["q"], "contacts": best["cs"],
           "evals": int(es.result.evaluations)}
    if record_history:
        out["history"] = history
    return out
