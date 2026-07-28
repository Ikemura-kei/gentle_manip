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


def plan_grasp(obj, mesh, cfg: Optional[MetricConfig] = None, *, x0=None, sigma: float = 0.3,
               maxfevals: int = 200, pad_half: float = 0.015, mu: float = 0.5, n_per_patch: int = 6,
               n_dirs: int = 12, penalty: float = 10.0, seed: int = 0, verbose: bool = False) -> dict:
    """Maximize Q_SM over parallel-jaw poses with CMA-ES. Returns dict(x, q_sm, contacts, evals)."""
    import cma

    from .geometry import load_mesh
    mesh = load_mesh(mesh)
    lo, hi = mesh.bounds
    com = obj.com

    best = {"q": -np.inf, "x": None, "cs": None}               # track the best FEASIBLE grasp directly

    def neg_qsm(x):
        cs = grasp_contacts(mesh, x, com=com, pad_half=pad_half, mu=mu, n_per_patch=n_per_patch)
        if cs is None or cs.n_contacts < 3:
            return penalty
        q = q_sm(obj, cs, cfg, n_dirs=n_dirs)
        if not np.isfinite(q):
            return penalty
        if q > best["q"]:
            best.update(q=float(q), x=np.asarray(x, float).copy(), cs=cs)
        return -q

    if x0 is None:
        x0 = np.array([*(0.5 * (lo + hi)), np.pi / 2, 0.0])   # object centre, horizontal closing axis
    bounds = [[lo[0], lo[1], lo[2], 0.0, 0.0], [hi[0], hi[1], hi[2], np.pi, 2 * np.pi]]
    neg_qsm(np.asarray(x0, float))                            # seed with x0 so best is never empty if x0 is valid
    es = cma.CMAEvolutionStrategy(list(x0), sigma,
                                  {"maxfevals": maxfevals, "bounds": bounds, "seed": seed,
                                   "verbose": -9 if not verbose else 1})
    es.optimize(neg_qsm)
    return {"x": best["x"], "q_sm": best["q"], "contacts": best["cs"],
            "evals": int(es.result.evaluations)}
