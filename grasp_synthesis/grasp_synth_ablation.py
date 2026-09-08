"""Standardized grasp-synthesis ablation — ONE frozen executor, N pose generators.

Answers `docs/paper/synthesis_experiments.md` E1/E2 under the CURRENT frozen recipe: does
FEM-awareness in the *selection objective* buy gentleness, against gentleness-blind classical
and learned planners?

    OBJ=tofu N_EPISODES=16 METHOD=rigid WIDTH_MODE=extent5 bash grasp_synthesis/run_ablation.sh

**Nothing outside grasp pose generation changes.** This file does not edit — and must never
edit — `collect_demos_synth_v4.py` (frozen collector v4.2), `smgrasp/finger_grasp_final.py`
(frozen planner) or `baseline_synth.py` (the 2026-08-31 E1 baselines). It imports the frozen
collector as a module and rebinds exactly one name at runtime, `finger_grasp_final.synthesize_grasp`,
so the executor FSM, phase timings, hold tail, disturbance draw, domain randomization, stress
recording, video capture, dr_params.csv / stats.yaml schema and the config snapshot are the SAME
code objects the training data was collected with.

    --method ours   applies NO patch at all: the run IS `collect_demos_synth_v4.py`.
                    Bit-identity is by construction, not by discipline.

Methods (`--method`)
    ours       frozen v4.2 FEM-surrogate CMA-ES planner (`finger_grasp_final.plan_finger_grasp`)
    naive      top-down at the settled object centre, uniform-random yaw          [E1]
    antipodal  surface-pair sampling, Nguyen friction-cone margin ranking         [E1]
    rigid      4000-sample antipodal sweep -> top-K re-ranked by the FULL geometric
               scorer (align, pad area, COM lever, holdability) — every term ours
               uses EXCEPT stress. This is the "antipodal + rigid quality" baseline. [E1]
    sdf        v2's hand-tuned SDF geometric cost, 7-DOF CMA-ES (`synth_utils.grasp_cost`) [E2]
    gpd        GPD (ten Pas et al., IJRR 2017), external planner + CNN scoring     [E1]
    gn1b       GraspNet-baseline (Fang et al., CVPR 2020), single-view cloud       [E1]
    cgn        Contact-GraspNet (NVIDIA), single-view cloud — available, not in the paper set

Width (`--width-mode`)
    extent5    DEFAULT. width = object cross-section along the grasp's closing axis, measured in
               the slab the pads actually touch, MINUS `--width-squeeze` (default 5 mm), for EVERY
               method except `ours`. Uniform across baselines, so the comparison isolates the pose.
    extent     the same cross-section with NO squeeze (`--width-squeeze 0`): the jaws close to
               exactly the object's extent. The answer to "subtracting 5 mm handicaps the
               baselines" — report it alongside extent5 and let the reviewer pick.
    native     each generator's own width convention (GPD aperture, GraspNet width, antipodal pair
               distance, ...); generators without one fall back to extent5.
    fem        our FEM surrogate chooses the width on the baseline's pose: the frozen planner's
               own width-refine stage (a 1-D scan of REFINE_SCAN widths over +-REFINE_HALF, scored
               by `score_finger_grasp_batch`, widest holdable wins). Factorizes pose choice from
               width choice: `gpd/fem` vs `gpd/extent5` isolates what our surrogate adds.

Rotation box (`--rot-bound`, default `tier2`)
    Every generator that accepts a pose constraint is given the frozen planner's TIER-2 (most
    generous) rotation bound, so all methods search the SAME feasible set of orientations:
    roll pi +- 40 deg, pitch +- 30 deg, yaw +- 80 deg (base 30/20/60 plus the tier >= 1 relaxation
    of +10 deg roll/pitch and yaw +-80; tiers 1 and 2 share the rotation box, they differ only in
    the yield gate). Applied as: `yaw_max_deg = 80` to naive / antipodal / rigid / gpd / gn1b / cgn
    (all top-down or externally-oriented, so yaw is the constraint they expose), and the full
    roll/pitch/yaw box to `sdf`, whose CMA-ES bounds this file constructs. `--rot-bound native`
    leaves each generator's own default (yaw +-90 deg for the E1 baselines, v2's own box for sdf).

Bookkeeping parity: whatever generator produced the pose, the returned metrics (stress_top10,
grip, align, pressure, min_pad_area, width_face, tilt_deg) are measured by the FROZEN scorer
`finger_grasp_final.score_finger_grasp`, so every column of dr_params.csv means the same thing
across methods. Baseline rows carry `synth_tier = -1` (they have no tier ladder); `ours` keeps
its real tier.

Table height: the board rig's surface is z = 0.0138 m, and `--table-z` is threaded to every
generator (all of them clear the table from it), so NO constant pose offset is needed or applied.
`--pose-z-offset` exists as an escape hatch if runtime verification shows a generator ignoring it.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── our flags, stripped before the frozen collector's parser sees argv ────────────────────────
METHODS = ("ours", "naive", "antipodal", "rigid", "sdf", "gpd", "gn1b", "cgn")
WIDTH_MODES = ("extent5", "extent", "native", "fem")

_argv = sys.argv[1:]


def _take(flag: str, default, cast=str):
    """Pull `--flag value` (or a bare `--flag` for bools) out of _argv."""
    global _argv
    if flag not in _argv:
        return default
    k = _argv.index(flag)
    if isinstance(default, bool):
        del _argv[k:k + 1]
        return True
    val = _argv[k + 1]
    del _argv[k:k + 2]
    return cast(val)


METHOD = _take("--method", "ours")
WIDTH_MODE = _take("--width-mode", "extent5")
WIDTH_SQUEEZE = _take("--width-squeeze", 0.005, float)   # extent5: cross-section MINUS this
POSE_Z_OFFSET = _take("--pose-z-offset", 0.0, float)     # escape hatch; see module docstring
BASELINE_OCC = _take("--baseline-occ", False)            # forward our camera-azimuth bound (confound check)
FEM_WIDTH_SCAN = _take("--fem-width-scan", 0, int)       # 0 = the frozen REFINE_SCAN
FEM_WIDTH_HALF = _take("--fem-width-half", 0.0, float)   # 0 = the frozen REFINE_HALF
ROT_BOUND = _take("--rot-bound", "tier2")                # tier2 | native; see module docstring
if METHOD not in METHODS:
    sys.exit(f"--method must be one of {METHODS}, got {METHOD!r}")
if WIDTH_MODE not in WIDTH_MODES:
    sys.exit(f"--width-mode must be one of {WIDTH_MODES}, got {WIDTH_MODE!r}")
if ROT_BOUND not in ("tier2", "native"):
    sys.exit(f"--rot-bound must be tier2|native, got {ROT_BOUND!r}")

import collect_demos_synth_v4 as C                       # noqa: E402  FROZEN collector — never edited
from smgrasp import finger_grasp_final as FG             # noqa: E402  FROZEN planner/scorer

# ── method == ours: no patch whatsoever ───────────────────────────────────────────────────────
if METHOD == "ours":
    if WIDTH_MODE != "extent5" or POSE_Z_OFFSET or BASELINE_OCC or ROT_BOUND != "tier2":
        sys.exit("--method ours runs the frozen planner unmodified: no width-mode / rot-bound / "
                 "pose-offset / occ flags (it applies its own tier ladder)")
    print("[ablation] method=ours — NO patch applied; this run is collect_demos_synth_v4.py verbatim",
          flush=True)
    sys.argv = ["grasp_synth_ablation.py"] + _argv
    C.main()
    raise SystemExit(0)

import baseline_synth                                    # noqa: E402  E1 baselines — never edited
import synth_utils                                       # noqa: E402  v2 SDF cost — never edited
from smgrasp.viz import boundary_faces                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEFT_FINGER = str(ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/left_finger.STL")
RIGHT_FINGER = str(ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/right_finger.STL")

# The frozen planner's TIER-2 rotation box, read from the frozen constants + the tier >= 1 relaxation
# (finger_grasp_final.plan_finger_grasp: roll/pitch bounds widen by 10 deg, yaw becomes +-80 deg).
# Tier 1 and tier 2 share this box; they differ only in the yield gate, which is not a pose constraint.
TIER_RELAX_DEG = 10.0
TIER2_YAW_DEG = 80.0
TIER2_ROLL_DEG = FG.ROLL_MAX_DEG + TIER_RELAX_DEG        # 40
TIER2_PITCH_DEG = FG.PITCH_MAX_DEG + TIER_RELAX_DEG      # 30
YAW_MAX_DEG = TIER2_YAW_DEG if ROT_BOUND == "tier2" else None


# ── shared helpers (frozen geometry, so every method is measured identically) ──────────────────

def _obj_sdf(obj):
    """The frozen planner's object SDF, cached on the FEM object exactly as it caches it."""
    sdf = getattr(obj, "_obj_sdf", None)
    if sdf is None:
        sdf = obj._obj_sdf = FG.build_object_sdf(obj)
    return sdf


def _score_kw(obj, com, quat, pad_geo, kw):
    return dict(obj_com=np.asarray(com, float), obj_quat_wxyz=quat, pad_geo=pad_geo,
                E=kw.get("E", 3e5), density=kw.get("density", 1000.0), mu=kw.get("mu", 0.7),
                table_z=kw.get("table_z", 0.0), obj_sdf=_obj_sdf(obj),
                yield_stress=kw.get("yield_stress"))


def _surface_mesh_path(obj):
    """The FEM object's boundary surface as an .obj on disk, cached on the object.

    Every method is handed the SAME geometry this way: the tet-mesh surface the frozen planner
    scores, not the original scan (which may differ after the collector's mesh deformation DR).
    """
    p = getattr(obj, "_ablation_surface_obj", None)
    if p is not None and os.path.exists(p):
        return p
    import trimesh
    faces, _ = boundary_faces(np.asarray(obj.tets))
    m = trimesh.Trimesh(vertices=np.asarray(obj.verts, float), faces=np.asarray(faces), process=False)
    fd, p = tempfile.mkstemp(suffix=".obj", prefix="ablation_surface_")
    os.close(fd)
    m.export(p)
    obj._ablation_surface_obj = p
    return p


def _closing_axis_world(x):
    """World closing axis (TCP y) of a 7-DOF grasp — the direction the jaws travel."""
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_euler("xyz", np.asarray(x[3:6], float)).apply([0.0, 1.0, 0.0])


def _extent_along_closing_axis(obj, x, com, quat):
    """Object cross-section along the grasp's closing axis, restricted to the slab the pads touch.

    Delegates to `baseline_synth._local_xsec` (the convention the 2026-08-31 E1 runs used for
    learned planners) and falls back to the full projected span when the slab is too sparse.
    """
    from scipy.spatial.transform import Rotation as Rot
    R = Rot.from_euler("xyz", np.asarray(x[3:6], float))
    closing_w, approach_w = R.apply([0.0, 1.0, 0.0]), R.apply([0.0, 0.0, 1.0])
    pad_centre_w = np.asarray(x[:3], float) + approach_w * FG._z_off(float(x[6]))
    xsec = baseline_synth._local_xsec(obj, np.asarray(quat, float), np.asarray(com, float),
                                      pad_centre_w, closing_w, approach_w)
    if xsec is None:                                        # sparse slab: whole-object span instead
        q = np.asarray(quat, float)
        ax = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv().apply(closing_w)
        proj = obj.verts @ (ax / (np.linalg.norm(ax) + 1e-12))
        xsec = float(proj.max() - proj.min())
    return float(xsec)


def _apply_width(obj, x, com, quat, pad_geo, kw):
    """Set the grasp width per --width-mode. Returns (x, note) with x[6] replaced."""
    x = np.asarray(x, float).copy()
    if WIDTH_MODE == "native":
        return x, f"native {1e3 * x[6]:.1f}mm"
    if WIDTH_MODE in ("extent5", "extent"):
        squeeze = WIDTH_SQUEEZE if WIDTH_MODE == "extent5" else 0.0
        xsec = _extent_along_closing_axis(obj, x, com, quat)
        x[6] = float(np.clip(xsec - squeeze, FG.WIDTH_MIN, FG.WIDTH_MAX))
        return x, f"extent {1e3 * xsec:.1f}mm - {1e3 * squeeze:.1f}mm -> {1e3 * x[6]:.1f}mm"
    # fem: the frozen planner's own width-refine stage, applied to this pose
    n_scan = FEM_WIDTH_SCAN or FG.REFINE_SCAN
    half = FEM_WIDTH_HALF or FG.REFINE_HALF
    start = float(np.clip(x[6], FG.WIDTH_MIN, FG.WIDTH_MAX))
    widths = np.clip(start + np.linspace(-half, half, n_scan), FG.WIDTH_MIN, FG.WIDTH_MAX)
    X = []
    for w in widths:
        x2 = x.copy()
        x2[6] = float(w)
        X.append(x2)
    R = FG.score_finger_grasp_batch(obj, X, **_score_kw(obj, com, quat, pad_geo, kw))
    ok = [(x2, r["score"]) for x2, r in zip(X, R) if FG.is_real_grasp(r["score"])]
    if not ok:
        return x, f"fem refine found no holdable width around {1e3 * start:.1f}mm — kept it"
    bx, bs = max(ok, key=lambda t: t[1])
    return bx, f"fem refine {1e3 * start:.1f}mm -> {1e3 * bx[6]:.1f}mm ({len(ok)}/{n_scan} holdable)"


# ── the SDF (v2) pose generator, wired to the same FEM surface every other method sees ─────────
_FINGER_PTS = [None]


def sdf_planner(obj, pad_geo, obj_com, obj_quat_wxyz, *, E=3e5, density=1000.0, mu=0.7,
                table_z=0.0, seed=0, yaw_max_deg=None, maxfevals=1145, **_):
    """v2's synthesis: hand-tuned SDF geometric cost, 7-DOF CMA-ES (`synth_utils.grasp_cost`).

    Same cost function, finger sampling and bound construction v2 used (`collect_demos_synth_v2`
    `_synth_worker` / `_synth_bounds`), run in-process on the FEM boundary surface.
    """
    if _FINGER_PTS[0] is None:
        _FINGER_PTS[0] = (synth_utils.sample_finger_surface(LEFT_FINGER, n=300),
                          synth_utils.sample_finger_surface(RIGHT_FINGER, n=300))
    left_pts, right_pts = _FINGER_PTS[0]
    sdf_fn = synth_utils.build_object_sdf(_surface_mesh_path(obj))
    com = np.asarray(obj_com, float)
    half = 0.5 * (np.asarray(obj.verts, float).max(0) - np.asarray(obj.verts, float).min(0))
    lb = (com[:2] - 1.5 * half[:2]).tolist() + [
        float(com[2]) + synth_utils.FINGER_TO_TCP_Z - 0.04, -np.pi, -0.12 * np.pi, -0.49 * np.pi, 0.01]
    ub = (com[:2] + 1.5 * half[:2]).tolist() + [
        float(com[2]) + 0.25, np.pi, 0.12 * np.pi, 0.49 * np.pi, 0.08]
    if ROT_BOUND == "tier2":       # the frozen planner's tier-2 box replaces v2's own rotation bounds
        r, p, y = np.radians([TIER2_ROLL_DEG, TIER2_PITCH_DEG, TIER2_YAW_DEG])
        lb[3], ub[3] = np.pi - r, np.pi + r
        lb[4], ub[4] = -p, p
        lb[5], ub[5] = -y, y
    x0 = [(lo + hi) / 2 for lo, hi in zip(lb, ub)]
    best_x, score = synth_utils.run_cmaes(
        lambda x: synth_utils.grasp_cost(x, left_pts, right_pts, sdf_fn, com, obj_quat_wxyz),
        x0, 1.0, lb, ub, maxfevals, seed=int(seed))
    return {"x": np.asarray(best_x, float), "sdf_score": float(score)}


POSE_GENERATORS = {
    "naive": baseline_synth.naive_topdown,
    "antipodal": baseline_synth.antipodal,
    "rigid": baseline_synth.rigid_planner,
    "sdf": sdf_planner,
    "gpd": baseline_synth.gpd_planner,
    "gn1b": baseline_synth.gn1b_planner,
    "cgn": baseline_synth.cgn_planner,
}


# ── the single substituted name ───────────────────────────────────────────────────────────────
_call = [0]


def synthesize_grasp(obj, pad_geo, obj_com, obj_quat_wxyz, **kw):
    """Drop-in for `finger_grasp_final.synthesize_grasp`: baseline pose + frozen-scorer metrics.

    Returns the dict the frozen collector consumes. `x = None` on failure, so the collector's own
    fallback path (default top-down, episode never saved) runs exactly as it does for `ours`.
    """
    _call[0] += 1
    gen = POSE_GENERATORS[METHOD]
    occ = ({"cam_pos": kw.get("cam_pos"), "cam_azimuth_max_deg": kw.get("cam_azimuth_max_deg")}
           if BASELINE_OCC else {})
    try:
        r = gen(obj, pad_geo, obj_com, obj_quat_wxyz,
                E=kw.get("E", 3e5), density=kw.get("density", 1000.0), mu=kw.get("mu", 0.7),
                table_z=kw.get("table_z", 0.0), seed=int(kw.get("seed", 0)) + _call[0],
                yaw_max_deg=YAW_MAX_DEG, **occ)
    except Exception as e:                                   # a baseline failing is data, not a crash
        print(f"    [ablation] {METHOD} raised {type(e).__name__}: {e} -> synthesis failure", flush=True)
        return {"x": None, "stress_top10": None, "tier": -1, "status": f"error:{type(e).__name__}"}
    x = r.get("x") if isinstance(r, dict) else None
    if x is None:
        print(f"    [ablation] {METHOD}: no pose -> synthesis failure", flush=True)
        return {"x": None, "stress_top10": None, "tier": -1, "status": "no_pose"}

    x = np.asarray(x, float).copy()
    if POSE_Z_OFFSET:
        x[2] += POSE_Z_OFFSET
    x, wnote = _apply_width(obj, x, obj_com, obj_quat_wxyz, pad_geo, kw)

    res = FG.score_finger_grasp(obj, x, **_score_kw(obj, obj_com, obj_quat_wxyz, pad_geo, kw))
    print(f"    [ablation] {METHOD}/{WIDTH_MODE}: w={1e3 * x[6]:.1f}mm "
          f"yaw={np.degrees(x[5]):+.1f}deg | {wnote} | scorer status={res.get('status')} "
          f"holdable={res.get('holdable')}", flush=True)

    # The scorer RECORDS an opinion; it must never block a grasp. A gentleness-blind baseline that picks a
    # non-compressing width (`no_contact`, routine in `extent` mode) must still be EXECUTED — the
    # MPM decides whether it lifts, not our surrogate. So every field the frozen collector formats
    # or rounds is made finite here: `stress_top10 = None`/inf would either trip v4's
    # synthesis-failure branch (substituting ITS default grasp, measuring v4 instead of the
    # baseline) or raise OverflowError in `synth_stats_row`'s round(). 0.0 is the physically
    # correct surrogate stress for jaws that never compress the object; `status` carries the
    # scorer's real verdict into the CSV.
    def _num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return f if np.isfinite(f) else 0.0

    return {"x": x, "score": _num(res.get("score")), "evals": 0,
            "stress_top10": _num(res.get("stress_top10")), "grip": _num(res.get("grip")),
            "align": _num(res.get("align")), "pressure": _num(res.get("pressure")),
            "min_pad_area": _num(res.get("min_pad_area")), "width_face": res.get("width_face"),
            "tilt_deg": _num(res.get("tilt_deg")), "twist": _num(res.get("twist")),
            "status": res.get("status"), "tier": -1}


FG.synthesize_grasp = synthesize_grasp       # the ONE substitution (C.fg IS this module object)
print(f"[ablation] method={METHOD} width-mode={WIDTH_MODE} rot-bound={ROT_BOUND}"
      f"{'' if ROT_BOUND == 'native' else f' (roll pi+-{TIER2_ROLL_DEG:.0f}, pitch +-{TIER2_PITCH_DEG:.0f}, yaw +-{TIER2_YAW_DEG:.0f} deg)'} "
      f"| squeeze {1e3 * WIDTH_SQUEEZE:.1f}mm, z-offset {1e3 * POSE_Z_OFFSET:+.1f}mm, occ={BASELINE_OCC} "
      f"— frozen executor collect_demos_synth_v4.py, unmodified", flush=True)
sys.argv = ["grasp_synth_ablation.py"] + _argv
C.main()
