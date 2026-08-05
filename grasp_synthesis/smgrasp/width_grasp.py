"""Width/position-controlled parallel-jaw grasp metric (grasp_synthesis/CLAUDE.md §11).

Our real gripper is POSITION-controlled (commanded width, not force), so we model the grasp as two
flat rigid pads (a cube proxy now; the real finger STL later) that close in to indent the SOFT object
by a commanded depth `delta` per jaw. The object surface conforms to the flat rigid pad (soft object +
rigid finger), so the pad-contact nodes get a PRESCRIBED inward displacement — a Dirichlet FEM solve
(fem.solve_dirichlet). Outputs: the induced stress field (spread, real deformation) and the reaction
GRIP force (an output, not an input, under position control).

Key scaling (position control): the prescribed displacement makes deformation `u` E-INDEPENDENT, and
both stress σ and grip reaction scale LINEARLY with Young's modulus E. So we solve once at E=1 and get
any (E, mass, μ) by scalar ops — cheap DR over the physical params (§11: nominal + randomize).

    stress(E)   = E · σ₁            grip(E) = E · F₁
    holdable    = 2·μ·grip ≥ m·g    (two pads, friction carries gravity)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse

from .viz import boundary_faces, von_mises

# Opt-in GPU FEM solver (torch/CUDA dense factor, ~30× faster than the scipy sparse back-sub — the
# profiled 97% hot path). Default False = the committed baseline (`solve_constrained_fast`, CPU sparse)
# so results/behaviour are unchanged unless explicitly enabled; set True (or `use_gpu_solve(True)`) to
# accelerate. Falls back to sparse for meshes too large for a dense factor (ndof > GPU_MAX_NDOF).
USE_GPU_SOLVE = False
GPU_MAX_NDOF = 16000


def use_gpu_solve(on: bool = True):
    """Enable/disable the GPU FEM solver globally (see USE_GPU_SOLVE)."""
    global USE_GPU_SOLVE
    USE_GPU_SOLVE = bool(on)


def _perp_basis(axis: np.ndarray):
    a = np.asarray(axis, float); a = a / (np.linalg.norm(a) + 1e-12)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0.0, 1, 0])
    u1 = np.cross(a, t); u1 /= np.linalg.norm(u1)
    u2 = np.cross(a, u1)
    return a, u1, u2


def boundary_normals(obj):
    """Outward unit surface normal at every tet-mesh node (0 for interior nodes), cached on `obj`.
    Used by the alignment term — a good grasp presses PERPENDICULAR to the surface (closing axis ∥ the
    contact normals), which favours flush face/through-centre grasps over grazing/edge contact."""
    vn = getattr(obj, "_bnormals", None)
    if vn is not None:
        return vn
    tri, parent = boundary_faces(obj.tets)
    v = obj.verts
    fn = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    fc = v[tri].mean(1); tetc = v[obj.tets[parent]].mean(1)      # orient OUTWARD (away from tet centroid)
    fn *= np.sign(((fc - tetc) * fn).sum(1))[:, None]
    vn = np.zeros_like(v)
    np.add.at(vn, tri.ravel(), np.repeat(fn, 3, axis=0))
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)
    obj._bnormals = vn
    return vn


def grasp_alignment(obj, axis, nodes):
    """Mean |closing-axis · surface-normal| over the contact `nodes` ∈ [0,1]: 1 = pad presses flush
    (perpendicular to the surface), →0 = grazing/edge contact. The stability signal for the objective."""
    vn = boundary_normals(obj)[nodes]
    a = np.asarray(axis, float); a /= np.linalg.norm(a) + 1e-12
    return float(np.mean(np.abs(vn @ a)))


def indent_from_width(obj, center, axis, *, pad_half: float, width: float,
                      max_indent: float = 0.01, min_nodes: int = 3):
    """Position control: pads close to gripper WIDTH `width` (inner faces at center ± width/2 along the
    axis). Returns (delta_left, delta_right, status) — the per-jaw CENTRE indentation and one of:
      'ok'         — both jaws indent the object (a real squeeze),
      'no_contact' — a jaw misses (width ≥ the object cross-section, or the footprint misses),
      'degenerate' — a jaw is buried > max_indent (⟺ the pad still penetrates the nominal mesh at
                     width + 2·max_indent) → skip, it's beyond the valid (small-strain) regime.
    Cheap (nominal mesh only, no FEM) — the pre-filter before an FEM evaluation."""
    a, u1, u2 = _perp_basis(axis)
    d = obj.verts[np.unique(boundary_faces(obj.tets)[0])] - np.asarray(center, float)
    proj = d @ a
    foot = (np.abs(d @ u1) < pad_half) & (np.abs(d @ u2) < pad_half)   # SQUARE pad footprint
    left, right = foot & (proj < 0), foot & (proj > 0)
    # `dist` (metres) = signed distance OUTSIDE the valid contact band, for a SHAPED search penalty
    # (0 when ok) — a flat penalty breaks CMA-ES on flat-faced objects (see score_candidate).
    if left.sum() < min_nodes or right.sum() < min_nodes:
        return 0.0, 0.0, "no_contact", 0.02                       # footprint misses a jaw (fixed dist)
    dl_raw = -width / 2.0 - proj[left].min()                      # >0 = indent, <0 = gap (pad short)
    dr_raw = proj[right].max() - width / 2.0
    dl, dr = max(dl_raw, 0.0), max(dr_raw, 0.0)
    if dl <= 0.0 or dr <= 0.0:                                    # a jaw doesn't reach -> NARROW to contact
        return dl, dr, "no_contact", float(max(-dl_raw, -dr_raw, 0.0))
    if dl > max_indent or dr > max_indent:                        # pad buried -> WIDEN to exit
        return dl, dr, "degenerate", float(max(dl - max_indent, dr - max_indent))
    return dl, dr, "ok", 0.0


def indent_contacts(obj, center, axis, *, pad_half: float, delta: float = None,
                    delta_left: float = None, delta_right: float = None, min_nodes: int = 3):
    """Two FLAT pads with slightly ROUNDED EDGES (a real finger pad), footprint radius `pad_half` ⟂
    the closing axis, indent the object by `delta_left`/`delta_right` (or a shared `delta`). The
    prescribed indentation is UNIFORM (= δ, a FLAT indent — correct for a flat face) over the interior
    and tapers to 0 only across a thin edge fillet, so there's no sharp flat-punch edge singularity. (An
    earlier fully-parabolic profile made even a flat cube face indent as a dome — wrong; the flat
    interior fixes it.) Each contact node is constrained ONLY along the closing axis (`aᵀuᵢ = ±dᵢ`),
    tangential free, so the soft object bulges under the pad. Returns dict(nodes, g, left_mask, axis),
    or None if a jaw grips < min_nodes boundary nodes."""
    dL = delta_left if delta_left is not None else delta
    dR = delta_right if delta_right is not None else delta
    fillet = 0.12 * pad_half                                       # THIN rounded-edge width (real pad fillet)
    a, u1, u2 = _perp_basis(axis)
    v = obj.verts
    bidx = np.unique(boundary_faces(obj.tets)[0])                  # boundary node indices
    d = v[bidx] - np.asarray(center, float)
    proj = d @ a
    p1, p2 = d @ u1, d @ u2                                       # SQUARE pad footprint (rectangular pad)
    edge = pad_half - np.maximum(np.abs(p1), np.abs(p2))          # distance to the square pad boundary
    foot = edge > 0

    nodes_all, g_all, left_mask = [], [], []
    for name, side, dd in (("left", proj < 0, dL), ("right", proj > 0, dR)):
        sel = foot & side
        nodes, ed, pr = bidx[sel], edge[sel], proj[sel]
        if len(nodes) < min_nodes:
            return None
        # PUSH EACH NODE TO A COMMON FLAT PLANE (= the pad face), not by a fixed δ·taper — the latter
        # domes a flat face and never flattens a curved one. plane = outermost proj ± δ; per-node
        # displacement (plane − proj) flattens flat AND curved faces alike. Taper ONLY the thin rim
        # (distance to the SQUARE edge) so the indent + its stress ring are SQUARE, like a real pad.
        taper = np.clip(ed / fillet, 0.0, 1.0); taper = taper * taper * (3.0 - 2.0 * taper)
        plane = (pr.min() + dd) if name == "left" else (pr.max() - dd)
        g = (plane - pr) * taper                                  # aᵀu: +a push (left, g>0) / −a (right, g<0)
        hit = (g > 1e-12) if name == "left" else (g < -1e-12)     # only nodes the pad actually reaches
        cn, cg = nodes[hit], g[hit]
        if len(cn) < min_nodes:
            return None
        nodes_all.append(cn); g_all.append(cg); left_mask.append(np.full(len(cn), name == "left"))
    return {"nodes": np.concatenate(nodes_all), "g": np.concatenate(g_all),
            "left_mask": np.concatenate(left_mask), "axis": a}


def width_grasp_stress(obj, center, axis, *, pad_half: float, delta: float = None,
                       delta_left: float = None, delta_right: float = None,
                       mask_contact: bool = True) -> dict:
    """Solve the width-controlled indentation grasp at E=1 (normal-only rounded-pad contact). Returns
    the E-independent primitives (deformation `u`, per-jaw normal grip `F1`, stress field `sigma1`,
    `top10_1`) — scale by E for a real stiffness (see module docstring). dict(valid, ...)."""
    bc = indent_contacts(obj, center, axis, pad_half=pad_half, delta=delta,
                         delta_left=delta_left, delta_right=delta_right)
    if bc is None:
        return {"valid": False}
    nodes, a = bc["nodes"], bc["axis"]
    nc = len(nodes)
    rows = np.repeat(np.arange(nc), 3)                           # C: one row per node, `a` in its 3 dofs
    cols = (3 * nodes[:, None] + np.arange(3)).reshape(-1)
    C = sparse.csr_matrix((np.tile(a, nc), (rows, cols)), shape=(nc, obj.fem.ndof))
    if USE_GPU_SOLVE and obj.fem.ndof <= GPU_MAX_NDOF:           # opt-in GPU dense solve (~30×)
        u, lam = obj.fem.solve_constrained_gpu(C, bc["g"])
    else:
        u, lam = obj.fem.solve_constrained_fast(C, bc["g"])      # reuses the cached factor (§11.6)
    sigma1 = obj.fem.element_stress(u)                           # (M,6) at E=1
    vm = von_mises(sigma1)
    F1 = float(abs(lam[bc["left_mask"]].sum()))                 # left-jaw normal grip at E=1 (= Σλ)

    if mask_contact:
        keep = ~np.isin(obj.tets, nodes).any(axis=1)            # drop tets touching a pad node
        vmk = vm[keep] if keep.any() else vm
    else:
        vmk = vm
    k = max(1, int(0.1 * len(vmk)))
    return {"valid": True, "sigma1": sigma1, "u": u, "F1": F1,
            "top10_1": float(np.sort(vmk)[-k:].mean()), "peak_1": float(vmk.max()),
            "mean_1": float(vmk.mean()), "hi_1": float(np.percentile(vm, 98)),   # UNMASKED p98 (contact-aware)
            "n_contacts": nc, "nodes": nodes, "axis": a}


def evaluate_grasp(obj, center, axis, *, pad_half: float, delta: float, E: float, density: float,
                   mu: float, g: float = 9.81, accel: float = 0.0, gravity_dir=(0, 0, -1.0),
                   prim: Optional[dict] = None) -> dict:
    """Single-sample width-controlled grasp evaluation for physical params (E, density, μ). Reuses the
    E=1 primitives `prim` (from width_grasp_stress) if given — so a DR sweep over (E, mass, μ) is just
    scalar ops after ONE solve. Returns dict(valid, holdable, stress_top10 (Pa), grip (N), ...)."""
    if prim is None:
        prim = width_grasp_stress(obj, center, axis, pad_half=pad_half, delta=delta)
    if not prim["valid"]:
        return {"valid": False, "holdable": False, "stress_top10": np.inf}
    mass = density * obj.volume
    grip = E * prim["F1"]                                          # real per-jaw normal force (N)
    holdable = (2.0 * mu * grip) >= mass * (g + accel)            # two pads, friction carries gravity
    return {"valid": True, "holdable": bool(holdable), "stress_top10": E * prim["top10_1"],
            "stress_peak": E * prim["peak_1"], "grip": grip, "mass": mass, "prim": prim}


BIG_NEG = -1e12          # degenerate / infeasible candidate score (planner MAXIMIZES)

# A SHAPED infeasibility penalty — a FLAT penalty makes CMA-ES's ranking degenerate on flat-faced
# objects (most samples miss the narrow valid width-band → identical values → the search drifts and
# terminates early, returning a poor grasp). Instead penalties DECREASE with distance to the valid
# contact band so the search has a gradient to follow. All penalties stay far below any real grasp
# score (−stress, which is > −PEN_BASE for any physical stress), and a HOLDABLE grasp always wins.
PEN_BASE = 1e8           # penalty floor (>> any physical stress in Pa)
PEN_SLOPE = 1e9          # per-metre shaping toward the contact band

# Alignment (stability) weight, in Pa per unit misalignment (§11.7): score = −stress − W_ALIGN·(1−align)
# where align = mean |closing-axis · surface-normal| over the contacts (1 = flush/perpendicular, →0 =
# grazing/edge). Big enough that a flush face grasp beats a lower-stress tilted/edge grasp; calibrated
# so it doesn't over-penalize genuinely curved contact on soft/organic objects.
W_ALIGN = 3e4

# Peak-aware stress weight (§11.7): score also subtracts W_PEAK · (unmasked p98 stress), so a grasp
# with a big contact/edge stress SPIKE (e.g. a corner grasp) is penalized even when its masked-bulk
# top10 looks low. Weight is relative to the masked top10 (both in Pa). 0.3 improved determinism (made
# the mushroom identical across seeds) and penalises concentrated grasps without hurting the good ones.
W_PEAK = 0.3


def _shaped_penalty(status: str, dist: float) -> float:
    """Infeasible-candidate score: not-holdable (contacting) > no-contact/degenerate (farther = worse)."""
    if status == "not_holdable":
        return -PEN_BASE                                          # contacting but can't hold (best of the bad)
    return -(PEN_BASE + dist * PEN_SLOPE)                         # miss/degenerate: worse the farther out


def is_real_grasp(score: float) -> bool:
    """True iff `score` is a real holdable grasp (−stress) rather than a shaped penalty."""
    return score > -PEN_BASE


def score_candidate(obj, center, axis, width, *, pad_half: float, E: float, density: float, mu: float,
                    g: float = 9.81, accel: float = 0.0, gravity_dir=(0, 0, -1.0),
                    max_indent: float = 0.01, w_align: float = W_ALIGN, w_peak: float = W_PEAK) -> dict:
    """Score one width-controlled grasp candidate for the planner (higher = gentler; MAXIMIZED).
    Cheap degeneracy filter FIRST (no FEM): pads must indent BOTH jaws and not be buried > max_indent
    (⟺ the pad penetrating the nominal mesh at width + 2·max_indent). Only 'ok' candidates get an FEM
    solve. score = −stress_top10 (Pa) − w_align·(1−alignment) for a holdable grasp — the alignment term
    (stability, §11.7) penalizes grazing/edge contact so a flush face/through-centre grasp wins; a
    SHAPED penalty (`_shaped_penalty`) otherwise. Returns dict(score, status, holdable, stress_top10,
    grip, align, delta_left, delta_right)."""
    dl, dr, status, dist = indent_from_width(obj, center, axis, pad_half=pad_half, width=width,
                                             max_indent=max_indent)
    if status != "ok":
        return {"score": _shaped_penalty(status, dist), "status": status, "holdable": False,
                "delta_left": dl, "delta_right": dr, "stress_top10": np.inf}
    prim = width_grasp_stress(obj, center, axis, pad_half=pad_half, delta_left=dl, delta_right=dr)
    if not prim["valid"]:
        return {"score": _shaped_penalty("no_contact", 0.02), "status": "no_contact",
                "holdable": False, "stress_top10": np.inf}
    r = evaluate_grasp(obj, center, axis, pad_half=pad_half, delta=None, E=E, density=density, mu=mu,
                       g=g, accel=accel, gravity_dir=gravity_dir, prim=prim)
    if not r["holdable"]:                                         # contacting but grip too low -> nudge
        return {"score": _shaped_penalty("not_holdable", 0.0), "status": "ok", "holdable": False,
                "stress_top10": r["stress_top10"], "grip": r["grip"], "prim": prim}
    align = grasp_alignment(obj, axis, prim["nodes"])            # 1 = flush/perpendicular, ->0 = edge
    score = -r["stress_top10"] - w_align * (1.0 - align) - w_peak * E * prim["hi_1"]
    return {"score": score, "status": "ok", "holdable": True, "stress_top10": r["stress_top10"],
            "grip": r["grip"], "align": align, "delta_left": dl, "delta_right": dr, "prim": prim}


def make_dr(n, *, E_range, mass_range, mu_range, seed=0):
    """A FIXED set of `n` (E, mass, μ) samples for domain randomization — fixed so the planner's
    objective stays deterministic across candidates. E in Pa, mass in kg, μ dimensionless."""
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(*E_range)), float(rng.uniform(*mass_range)), float(rng.uniform(*mu_range)))
            for _ in range(n)]


def score_candidate_dr(obj, center, axis, width, *, pad_half: float, dr, g: float = 9.81,
                       accel: float = 0.0, max_indent: float = 0.01, hold_frac: float = 1.0,
                       w_align: float = W_ALIGN, w_peak: float = W_PEAK) -> dict:
    """DR version of `score_candidate`: score a candidate for robustness over the (E, mass, μ) samples
    `dr` (find the best OVERALL pose, §11). ONE FEM solve per pose — the E-independent primitives
    (`sigma1`, `F1`) scale by each sample's E, so per-sample stress `E·top10_1` and holdability
    `2·μ·E·F1 ≥ mass·g` are scalar ops. score = −mean stress if the grasp holds in ≥ `hold_frac` of the
    samples (robust), else a SHAPED penalty (points toward the valid band). Degenerate/no-contact
    filtered first (no FEM)."""
    dl, drr, status, dist = indent_from_width(obj, center, axis, pad_half=pad_half, width=width,
                                              max_indent=max_indent)
    if status != "ok":
        return {"score": _shaped_penalty(status, dist), "status": status, "hold_frac": 0.0,
                "stress_mean": np.inf}
    prim = width_grasp_stress(obj, center, axis, pad_half=pad_half, delta_left=dl, delta_right=drr)
    if not prim["valid"]:
        return {"score": _shaped_penalty("no_contact", 0.02), "status": "no_contact",
                "hold_frac": 0.0, "stress_mean": np.inf}
    top1, F1 = prim["top10_1"], prim["F1"]
    stresses = np.array([E * top1 for E, _, _ in dr])
    holds = np.array([(2.0 * mu * E * F1) >= m * (g + accel) for E, m, mu in dr])
    hf = float(holds.mean())
    smean = float(stresses.mean())
    Es = np.array([E for E, _, _ in dr])
    # holds enough -> −mean stress − alignment penalty; else nudge toward more grip (hold shortfall)
    if hf >= hold_frac:
        align = grasp_alignment(obj, axis, prim["nodes"])
        score = -smean - w_align * (1.0 - align) - w_peak * float(np.mean(Es)) * prim["hi_1"]
    else:
        align, score = 0.0, -PEN_BASE * (1.0 + (hold_frac - hf))
    return {"score": score, "status": "ok", "hold_frac": hf, "align": align,
            "stress_mean": smean, "grip_mean": float(np.mean(Es) * F1),
            "delta_left": dl, "delta_right": drr, "prim": prim}
