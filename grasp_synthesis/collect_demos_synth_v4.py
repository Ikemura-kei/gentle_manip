"""v4 — SURROGATE-SELECTED EXECUTED WIDTH (fork of v3; v3 stays untouched as the fallback).

The single change vs v3: the executor's closure constants (2.5 mm width_cls baseline,
material-aware extra_close, firm base) are REPLACED by a width chosen by the FEM surrogate
itself. After synthesis, the same pose is scanned over commanded width (the refine-round
primitive); c_y = the closure at which PREDICTED stress crosses the object's yield; the
commanded closure is lambda * c_y with ONE global gain (--closure-gain, default 1.28,
identified once on the mushroom: measured-good closure 6.4 mm / predicted c_y 5.0 mm).
Measured basis (2026-08-30, docs/fem_surrogate_status.md section 5.1'): the surrogate's
yield-closure ordering is rank-perfect across mushroom/raspberry/cherry_tomato/banana_chunk
with a stable ~0.5-0.6 conservative bias, so one gain transfers. Zero per-object constants.
The weak-grasp firm CHECK is kept as a fallback (extra = 0.5 * commanded closure, capped).
"""
from __future__ import annotations

import argparse
import csv
import os
import dataclasses
import tempfile
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

ROOT = Path(__file__).resolve().parent.parent
GRASP_DIR = ROOT / "grasp_synthesis"
for _p in (str(ROOT), str(GRASP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# headless by default; MUJOCO_GL=glfw in the shell to get the viewer
os.environ.setdefault("MUJOCO_GL", "egl")

from synth_utils import FINGER_TO_TCP_Z  # noqa: E402
from util import (  # noqa: E402  general helpers, nothing synthesis/execution-specific
    _np, _git_commit, _make_run_dir, _write_shard, _merge_shards,
)
from smgrasp import finger_grasp_final as fg  # noqa: E402  (TRIMMED module, 2026-09-04: the
# provably-inert w_occ occlusion term and its ray machinery removed. finger_grasp.py is kept
# UNCHANGED for v2/v3/v5, eval_grasp_synth and test_grasp_v4, which still audit `occ`.)
from smgrasp import finger_viz          # noqa: E402  (grasp-pose viz paired with each execution video)
from live_seed_viz import StageViewer  # noqa: E402  (--dev-viz: standalone step-through window)
from gentle_manip.actions.action_config import ActionConfig
from gentle_manip.experiment import Experiment
from gentle_manip.tasks.single_lift import SingleLiftTask
from gentle_manip.envs.genesis_worker import GenesisWorker
from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.perception.pipeline import PerceptionPipeline
from gentle_manip.perception.obs_config import CONTACT_FORCE_THRESH_N
from gentle_manip.domain_randomization.dr_config import DRConfig


# ── Constants (keep in sync with run_grasp_synth.py) ─────────────────────────

N_HOME_TO_PRE = 98          # home → grasp pose interpolation steps
N_SETTLE      = 1           # hold at grasp pose before closing
N_GRASP       = 37           # gripper close steps
N_LIFT        = 66          # lift steps
N_HOLD        = 12           # hold at lift height (success eval window)
LIFT_HEIGHT   = 0.2         # metres above grasp position

# ── Action inversion ──────────────────────────────────────────────────────────

def _invert_actions(
    prev_pos:  np.ndarray,  # (N, 3) previous commanded pos
    cur_pos:   np.ndarray,  # (N, 3) current commanded pos
    prev_quat: np.ndarray,  # (N, 4) wxyz previous commanded quat
    cur_quat:  np.ndarray,  # (N, 4) wxyz current commanded quat
    prev_grip: np.ndarray,  # (N,) previous commanded gripper width
    cur_grip:  np.ndarray,  # (N,) current commanded gripper width
    scales:    np.ndarray,  # (7,) action scales from ActionConfig
) -> np.ndarray:
    """Compute (N, 7) float32 normalized actions from consecutive absolute targets.

    Mirrors SimBackend.step() exactly:
      pos:  new = clip(prev + dpos)          → dpos = new − prev (pre-clip)
      rot:  new = Rot.from_rotvec(drot)*Rprev → drot = (Rnew * Rprev.inv()).as_rotvec()
      grip: new = clip(prev + dgrip)         → dgrip = new − prev (pre-clip)
    All components clipped to [-1, 1] after dividing by their scale.
    """
    n = cur_pos.shape[0]

    # Position delta
    a_pos = np.clip((cur_pos - prev_pos) / scales[:3], -1.0, 1.0)

    # Rotation delta: world-frame premultiply in rotvec representation
    a_rot = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        def _xyzw(wxyz): return [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
        R_prev = Rot.from_quat(_xyzw(prev_quat[i]))
        R_cur  = Rot.from_quat(_xyzw(cur_quat[i]))
        rotvec = (R_cur * R_prev.inv()).as_rotvec()
        a_rot[i] = np.clip(rotvec / scales[3:6], -1.0, 1.0)

    # Gripper delta
    a_grip = np.clip((cur_grip - prev_grip).reshape(n, 1) / scales[6], -1.0, 1.0)

    return np.concatenate([a_pos, a_rot, a_grip], axis=1).astype(np.float32)


def _invert_actions_absolute(
    cur_pos:  np.ndarray,  # (N, 3) current commanded absolute pos
    cur_quat: np.ndarray,  # (N, 4) wxyz current commanded absolute quat
    cur_grip: np.ndarray,  # (N,) current commanded absolute gripper width
    action_config: ActionConfig,
) -> np.ndarray:
    """Compute (N, 10) normalized absolute actions that ActionPipeline (mode="absolute")
    would map back to (cur_pos, cur_quat, cur_grip). No `prev_*` needed — absolute mode
    has no accumulation history, each step is an independent forward transform.

    Mirrors ActionPipeline._process_absolute's inverse exactly:
      pos/gripper: un-map the linear [pos_min,pos_max]/[gripper_min,gripper_max] scaling.
      rot6d: the first two columns of R = Rotation.from_quat(cur_quat).as_matrix() are
        already exactly orthonormal, so Gram-Schmidt on them is a no-op — the rot6d
        "inverse" is just those two columns directly, no optimization/search needed.
    """
    n = cur_pos.shape[0]
    lo, hi = action_config.clip
    span = hi - lo
    pos_min = np.asarray(action_config.pos_min, dtype=np.float64)
    pos_max = np.asarray(action_config.pos_max, dtype=np.float64)

    t_pos = (cur_pos - pos_min) / (pos_max - pos_min)
    a_pos = np.clip(lo + t_pos * span, lo, hi)

    t_grip = (cur_grip - action_config.gripper_min) / (action_config.gripper_max - action_config.gripper_min)
    a_grip = np.clip(lo + t_grip * span, lo, hi).reshape(n, 1)

    a_rot6d = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        xyzw = [cur_quat[i, 1], cur_quat[i, 2], cur_quat[i, 3], cur_quat[i, 0]]
        mat = Rot.from_quat(xyzw).as_matrix()
        a_rot6d[i] = np.concatenate([mat[:, 0], mat[:, 1]])

    return np.concatenate([a_pos, a_rot6d, a_grip], axis=1).astype(np.float32)


# ── RawObs builder ────────────────────────────────────────────────────────────

def _state_to_raw_obs(state: dict) -> RawObs:
    """Build RawObs from genesis_worker.read_state() / step() output."""
    return RawObs(
        ee_pos=state["ee_pos"],
        ee_quat=state["ee_quat"],
        gripper_width=state["gripper_width"],
        joint_pos=state.get("joint_pos"),
        joint_vel=state.get("joint_vel"),
        depth_images=state["depth_images"],
        rgb_images={},
        camera_intrinsics=state["camera_intrinsics"],
        camera_extrinsics=state["camera_extrinsics"],
        tactile_images={},
    )


# ── Scene-level DR (object SIZE + SHAPE) ──────────────────────────────────────

def _apply_scene_dr(nominal_spec, dr_cfg, rng, deform_dir, mesh_cycle=None):
    """Per-scene SIZE + SHAPE DR for the object — mirrors
    SimBackend._apply_scene_dr, but BAKES the uniform scale into the exported mesh
    (not the ObjectEntry.scale field) so the CMA-ES SDF, which loads the mesh file
    directly, matches the geometry Genesis actually simulates.

    Always deforms from the NOMINAL mesh (idempotent — no chaining across rebuilds).
    Returns (new_spec, scene_dr) with scene_dr = {"scale", "bend_deg"} for
    priv_object_dr_params. If the DR config declares no shape/scale fields, returns
    the nominal spec unchanged (scene DR is then a no-op).
    """
    import trimesh
    from gentle_manip.assets import mesh_deform
    from gentle_manip.assets.registry import get_object_def

    o = nominal_spec.objects[0]
    nominal_scale = float(o.scale or 1.0)
    shp = dr_cfg.sample_shape_scale(rng)                     # {} if no shape/scale fields set
    # MATERIAL DR. `sample_scene` existed and this collector NEVER CALLED IT, so every demo ever
    # collected here used the registry's nominal E/nu/rho and the object_E/nu/rho/coup_friction
    # ranges in all the DR configs were INERT (verified 2026-08-28). Sample and bake them now.
    # NOTE `object_yield` still cannot be randomized: ObjectEntry has no yield field, so the yield
    # always comes from the registry material (pre-existing limitation, see CLAUDE.md).
    mat = dr_cfg.sample_scene(rng)
    if mesh_cycle is not None and dr_cfg.object_mesh_pool:
        # DETERMINISTIC round-robin over the pool instead of a uniform draw. Random sampling only
        # covers the pool in expectation: an 8-episode smoke test rebuilds the scene once or twice,
        # so it sees 1-2 of 4-5 meshes and a broken variant can sit unnoticed. Cycling guarantees
        # every mesh is exercised once per full pass. See --mesh-cycle.
        pool = list(dr_cfg.object_mesh_pool)
        variant = pool[mesh_cycle[0] % len(pool)]
        mesh_cycle[0] += 1
    else:
        variant = dr_cfg.sample_mesh_variant(rng)            # base-mesh pool pick (or None);
                                                             # same cadence as size/shape DR
    if not shp and variant is None and not mat:
        return nominal_spec, {"scale": nominal_scale, "bend_deg": 0.0, "mesh_variant": o.name}

    if variant is not None:                                  # pool pick replaces the base mesh;
        nominal_mesh = get_object_def(variant).mesh_path     # shape DR deforms FROM the pick
    else:
        nominal_mesh = o.mesh_path or get_object_def(o.name).mesh_path
    mesh = trimesh.load(str(nominal_mesh), process=False, force="mesh")
    shape = {k: shp[k] for k in ("bend", "twist", "taper", "rbf", "axis_scale", "axis_scale_ax")
             if k in shp}
    if shape:
        mesh = mesh_deform.deform_mesh(mesh, shape, rng)     # bend/twist/taper/axis_scale (radians)
    mesh.apply_scale(nominal_scale * float(shp.get("scale", 1.0)))   # bake uniform scale in
    dst = Path(deform_dir) / f"{Path(nominal_mesh).stem}_dr_{rng.integers(1_000_000):06d}.obj"
    mesh.export(str(dst))

    new_obj  = dataclasses.replace(o, mesh_path=str(dst), scale=1.0,
                                   **({"youngs_modulus": mat["E"]} if "E" in mat else {}),
                                   **({"poisson_ratio": mat["nu"]} if "nu" in mat else {}),
                                   **({"density": mat["rho"]} if "rho" in mat else {}))
    new_spec = dataclasses.replace(nominal_spec, objects=[new_obj, *nominal_spec.objects[1:]])
    scene_dr = {"scale": float(shp.get("scale", 1.0)),
                "bend_deg": float(np.rad2deg(shp.get("bend", 0.0))),
                "mesh_variant": variant if variant is not None else o.name,
                # every remaining DR draw, recorded so the frozen dataset is reproducible
                "twist_deg": float(np.rad2deg(shp.get("twist", 0.0))),
                "taper": float(shp.get("taper", 0.0)),
                "rbf": float(shp.get("rbf", 0.0)),
                "axis_scale": float(shp.get("axis_scale", 1.0)),
                "axis_scale_ax": int(shp.get("axis_scale_ax", -1)),
                "mat_E": float(mat.get("E", 0.0)), "mat_nu": float(mat.get("nu", 0.0)),
                "mat_rho": float(mat.get("rho", 0.0)),
                "coup_friction": float(mat.get("coup_friction", 4.0))}
    return new_spec, scene_dr


# ── Privileged obs (sim-only state-teacher fields) ────────────────────────────

def _privileged_obs_batch(object_center, object_quat, dr_vec, priv_cfg, contact_force=None,
                          von_mises=None, yield_stress=None) -> dict:
    """Sim-only privileged fields from raw worker state — mirrors
    PolicyEnv._privileged_obs exactly, but sourced from GenesisWorker state
    (object_center + object_quat + contact_force) since this collector bypasses PolicyEnv.

    object_center: (N, 3); object_quat: (N, 4) wxyz; dr_vec: (2,) [scale, bend_deg];
    contact_force: (N,) — rigid: get_contacts Newtons; soft: MPM->finger coupling force.
    Returns a dict of (N, ...) arrays for whichever priv fields the config enables.
    """
    out = {}
    # STRESS was declared in the obs configs but NEVER mirrored here, so a config asking for
    # `privileged: stress: true` was SILENTLY IGNORED by this collector (PolicyEnv honours it;
    # this file bypasses PolicyEnv and must mirror it — see the CLAUDE.md note). Emitting it now:
    # (N,2) = [mean, top10] / yield, the same normalization PolicyEnv uses.
    if getattr(priv_cfg, "stress", False) and von_mises is not None and yield_stress:
        vm = np.asarray(von_mises, np.float32)
        out["priv_stress"] = np.stack([vm.mean(axis=1) / float(yield_stress),
                                       _stress_top10(vm) / float(yield_stress)], axis=1
                                      ).astype(np.float32)
    oc = np.asarray(object_center, dtype=np.float32)                    # (N, 3)
    if priv_cfg.object_pos:
        out["priv_object_pos"] = oc
    if getattr(priv_cfg, "object_quat", False) or priv_cfg.object_rot6d:
        wxyz = np.asarray(object_quat, dtype=np.float32)               # (N, 4)
        if getattr(priv_cfg, "object_quat", False):
            out["priv_object_quat"] = wxyz
        if priv_cfg.object_rot6d:
            xyzw = np.concatenate([wxyz[:, 1:], wxyz[:, :1]], axis=1)
            mat  = Rot.from_quat(xyzw).as_matrix()                     # (N, 3, 3)
            out["priv_object_rot6d"] = np.concatenate(
                [mat[:, :, 0], mat[:, :, 1]], axis=-1                   # first two cols (Zhou 2019)
            ).astype(np.float32)
    if priv_cfg.object_dr_params:
        out["priv_object_dr_params"] = np.tile(
            np.asarray(dr_vec, np.float32)[None], (oc.shape[0], 1))     # (N, 2)
    if getattr(priv_cfg, "contact_force", False):
        out["priv_contact_force"] = np.asarray(contact_force, np.float32)[:, None]  # (N, 1)
    if getattr(priv_cfg, "contact", False):
        # ACTUAL binary gripper-object contact from the physics contact force (rigid: get_contacts
        # Newtons; soft: MPM->finger coupling force) — exactly 0 with nothing touching.
        contact = (np.asarray(contact_force, np.float32) > CONTACT_FORCE_THRESH_N)
        out["priv_contact"] = contact.astype(np.float32)[:, None]          # (N, 1)
    return out


# ── Trajectory conversion ─────────────────────────────────────────────────────

def _x_to_targets(x: np.ndarray, num_envs: int):
    """7-DOF best_x → (pos_b, quat_wxyz_b, width) batched for num_envs."""
    pos = np.asarray(x[:3], np.float32)
    q_xyzw = Rot.from_euler("xyz", x[3:6]).as_quat()
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], np.float32)
    width = float(x[6])
    return (np.tile(pos[None], (num_envs, 1)),
            np.tile(quat_wxyz[None], (num_envs, 1)),
            width)


# ── Per-env phase FSM ─────────────────────────────────────────────────────────
# Ordered phases every env progresses through independently. Duration is currently
# the same fixed constant for every env (so, with no retry logic yet, all envs still
# finish at the same global timestep — this refactor changes HOW the per-step target
# is computed, not the observable trajectory). The per-env (phase_idx, phase_step)
# state is what a later retry/robustness pass would rewind (e.g. jump phase_idx back
# to "approach" or "grasp" for an env whose object slipped), independent of every
# other env's progress.
GRASP_VOXEL_DIV = 14              # FEM remesh resolution (frozen v4.1; was --grasp-voxel-div, never passed)
GRASP_TARGET_TETS = 1500          # FEM target tet count (ndof stays <~5k) (frozen v4.1; was --grasp-target-tets, never passed)
GRASP_MU = 0.7                    # pad-object friction coefficient (frozen v4.1; was --grasp-mu, never passed)
N_FIRM = 8   # "firm" phase duration (steps) — gradual extra close IF triggered, else a no-op hold

PHASES = [
    ("approach", N_HOME_TO_PRE),   # home → pre-grasp pose (slerp + lerp)
    ("settle",   N_SETTLE),        # hold at grasp pose, gripper still open
    ("grasp",    N_GRASP),         # close gripper (gradual)
    ("firm",     N_FIRM),          # robustness idea #1: extra squeeze IF the grip came out weak
    ("lift",     N_LIFT),          # lift to LIFT_HEIGHT above the grasp point
    ("hold",     N_HOLD),          # hold at lift height (success eval window)
]
N_PHASES  = len(PHASES)
_GRASP_IDX = [name for name, _ in PHASES].index("grasp")   # boundary the firm-check fires at

# ── Robustness idea #1: force-based grasp firming ─────────────────────────────
# At the moment an env finishes "grasp" (checked ONCE per env, at the grasp->firm
# boundary), read its just-measured grip force. If the grasp came out weak
# (< FIRM_FORCE_THRESH_N) the env closes an extra min(0.5 * commanded closure, 2mm)
# over the "firm" phase before lifting; otherwise "firm" is a no-op hold that
# _trim_long_holds collapses. Bounded to fire once per env (a one-way FSM check, not
# a retry loop). This collector is SOFT-ONLY, and for an MPM body the force is the
# MPM->finger COUPLING force (XArm7Sim.gripper_coupling_force), which the worker
# always populates — so there is exactly one firm path.

CLOSURE_SCAN_STEP_M = 0.0005   # width-scan resolution (0.5 mm)
CLOSURE_SCAN_MAX_M  = 0.012    # scan depth; c_y beyond this is clipped (very soft objects)
CLOSURE_CMD_MIN_M   = 0.0008   # never command less than this (contact would be marginal)
CLOSURE_CMD_MAX_M   = 0.008    # hard safety cap
SCAN_METRIC         = "p98"    # closure-scan yield-crossing stress metric (unmasked 98th pct)
CLOSURE_GAIN        = 4.92     # commanded closure = CLOSURE_GAIN * c_y (identified once on the mushroom, p98)


def surrogate_closure(fem_obj, pad_geo, best_x, obj_com, obj_quat, E, yld, density, mu, table_z,
                      metric="masked"):
    """c_y: closure (m, beyond the synthesized width) at which the surrogate's predicted stress
    first crosses yield. v4.2: the DEFAULT crossing is back on the contact-MASKED top10 (bulk
    damage onset). v4.1's unmasked-p98 crossing was an over-correction bundled with the real fix
    (the pen_tol relaxation): measured 2026-08-30, on soft/low-yield objects the unmasked contact
    concentration exceeds yield AT THE PLAN WIDTH ITSELF (raspberry, strawberry: c_y = 0.0 mm),
    collapsing commands to the clip minimum — strawberry success 100 % -> 46 %. The masked curve
    is smooth and non-degenerate under the relaxed scan (verified a genuine stress crossing, not
    a geometric artifact, at the identification pose). `metric="p98"` retains the v4.1 behaviour
    for comparison.

    v4.1 also fixes two scan terminations that made v4.0's c_y geometric instead of
    stress-based (measured: raspberry c_y clustered at 3.0-3.5 mm = pad depth ~1.5 mm + dw/2
    crossing the search's 3 mm gross-clipping SDF tolerance):
    - pen_tol is RELAXED to 5 cm for scan calls only. Deep indents are legitimate here (a soft
      object may take 8 mm below yield); validity is bounded by the 10 mm max_indent
      (`degenerate`) instead. The 3 mm filter still protects the CMA search unchanged.
    - statuses are handled by MEANING: `no_contact` -> keep scanning (not touching yet);
      `degenerate`/`table` -> stop at the validity edge; `ok` -> test the crossing."""
    from smgrasp import finger_grasp as fg
    x0 = np.asarray(best_x, float)
    prev_dw, prev_s = 0.0, None
    for dw in np.arange(0.0, CLOSURE_SCAN_MAX_M + 1e-9, CLOSURE_SCAN_STEP_M):
        x = x0.copy(); x[6] = max(0.004, float(x0[6]) - dw)
        sc = fg.score_finger_grasp(fem_obj, x, obj_com=np.asarray(obj_com, float),
                                   obj_quat_wxyz=np.asarray(obj_quat, float), pad_geo=pad_geo,
                                   E=E, density=density, mu=mu, table_z=table_z,
                                   pen_tol=0.05, w_press=0.05, w_peak=0.3, area_min=0.0)
        st = sc.get("status")
        if st == "no_contact":
            prev_s = None
            continue                              # pads not touching yet — deepen
        if st in ("degenerate", "table", "penetrate"):
            return float(max(dw, CLOSURE_SCAN_STEP_M))   # validity edge before yield
        s98 = sc.get("stress_top10" if metric == "masked" else "stress_p98")
        if s98 is None or not np.isfinite(s98):
            return float(max(dw, CLOSURE_SCAN_STEP_M))
        if s98 >= yld:
            # LINEAR INTERPOLATION of the crossing between the bracketing samples: the command is
            # gain * c_y with gain ~5, so raw 0.5 mm quantization would amplify to ~2.5 mm of
            # command error. p98 is smooth in closure, so the interpolated crossing is accurate.
            if prev_s is not None and s98 > prev_s:
                frac = (yld - prev_s) / (s98 - prev_s)
                return float(max(prev_dw + frac * (dw - prev_dw), 1e-4))
            return float(max(dw, CLOSURE_SCAN_STEP_M))
        prev_dw, prev_s = dw, s98
    return float(CLOSURE_SCAN_MAX_M)


FIRM_FORCE_THRESH_N  = 1.0     # below this gripper->object contact force (N) the grasp is
                               # judged weak and gets the extra firm close. For a SOFT body
                               # this is the MPM->finger coupling force, not a rigid contact.


def _stress_top10(vm: np.ndarray) -> np.ndarray:
    """(N,) top-10%-mean von Mises per env from (N, n_p) particle stress — the same
    bulk-deformation signal the reward/metric use (robust to particle count)."""
    vm = np.asarray(vm, np.float32)
    k = max(1, int(round(0.10 * vm.shape[1])))
    return np.partition(vm, -k, axis=1)[:, -k:].mean(axis=1)

# ── Post-processing: trim long held-command runs ──────────────────────────────
HELD_RUN_MAX  = 8   # runs longer than this get trimmed
HELD_RUN_KEEP = 4   # ...down to this many frames (keep the first N, discard the rest)
HELD_RUN_EPS  = 1e-5


def _trim_long_holds(act_list, *parallel_lists, max_run=HELD_RUN_MAX,
                     keep=HELD_RUN_KEEP, eps=HELD_RUN_EPS):
    """Collapse runs of MORE THAN `max_run` consecutive near-identical actions down
    to `keep` frames (keep the first `keep` of the run, discard the rest). The SAME
    kept-index selection is applied to every list in `parallel_lists` (obs/rewards/
    frames), so everything stays aligned after trimming.

    Only meaningful for ABSOLUTE-mode actions: a held absolute target repeats the
    EXACT SAME command every frame it's held, so a long hold is many redundant
    identical frames. Delta-mode actions are already ~0 while held (representing
    "no movement", not a literal repeated command) — that's a different situation
    this trim should NOT touch, so callers must gate this on action_config.mode.

    A parallel list whose length doesn't match `act_list` (e.g. `frame_bufs[i]` is
    `[]` when --record-video wasn't passed) is passed through UNCHANGED rather than
    index-trimmed — it isn't step-aligned with the actions in the first place.
    """
    T = len(act_list)
    if T == 0:
        return (act_list,) + parallel_lists
    acts = np.stack(act_list)
    keep_mask = np.ones(T, dtype=bool)

    run_start = 0
    for t in range(1, T + 1):
        same = t < T and np.linalg.norm(acts[t] - acts[run_start]) < eps
        if not same:
            run_len = t - run_start
            if run_len > max_run:
                keep_mask[run_start + keep: t] = False   # drop the tail of this run
            run_start = t

    idx = np.where(keep_mask)[0]
    trimmed_acts     = [act_list[i] for i in idx]
    trimmed_parallel = tuple(
        ([lst[i] for i in idx] if len(lst) == T else lst) for lst in parallel_lists
    )
    return (trimmed_acts,) + trimmed_parallel


def execute_and_collect(
    worker:       GenesisWorker,
    all_best_x:   List[np.ndarray],
    init_obs_batch: dict,          # batched obs from reset, keyed by obs name → (N, ...)
    perception:   PerceptionPipeline,
    action_config: ActionConfig,   # mode="delta" uses .scales; mode="absolute" uses
                                   # pos_min/pos_max/gripper_min/gripper_max/clip
    record_video: bool = False,
    priv_cfg=None,                 # PrivilegedConfig or None — sim-only state-teacher fields
    dr_vec=None,                   # (2,) [scale, bend_deg] for priv_object_dr_params
    yield_stress=None,             # Pa — normalizer for priv_stress (None = stress not emitted)
    closure_cmd=None,              # (N,) m — v4 surrogate-selected closure (replaces all constants)
) -> Tuple[List[List[dict]], List[List[np.ndarray]], List[List[float]], np.ndarray, List[List]]:
    """Execute the scripted grasp trajectory with DECOUPLED per-env phase control;
    record (obs, action, reward) per env.

    Every env independently tracks its own (phase_idx, phase_step) through PHASES.
    Each timestep, every env's target (pos, quat, gripper) is computed from ITS OWN
    phase state (not a single shared alpha/phase for the whole batch) — command
    sending is still one batched worker.step() call per timestep (Genesis requires
    this), only the per-env ROW going into that call now varies independently.

    An env that has finished all its phases (DONE) is frozen at its hold target
    (so it doesn't disturb the sim while other envs keep going) and STOPS being
    recorded — its obs/action buffers simply stop growing, so episode length can
    now differ per env (obs_bufs[i]/act_bufs[i] length = however many steps env i
    was actually active for). The loop runs until every env reaches DONE.

    The scripted trajectory itself (positions/quats/gripper widths, all physical
    units) is IDENTICAL regardless of action_config.mode — only how each step's
    *recorded action* is derived (delta-inverted vs absolute-inverted) differs.

    Returns:
        obs_bufs:    list[N] of obs-dict lists (T_i steps per env, unbatched)
        act_bufs:    list[N] of float32 action arrays (T_i steps each; 7-dim delta or
                     10-dim absolute, matching action_config.action_dim)
        rew_bufs:    list[N] of float rewards (0.0 throughout)
        success:     (N,) bool — True if final object z > grasp_z + 0.5*LIFT_HEIGHT
        frame_bufs:  list[N] of (H,W,3) uint8 frame lists; empty lists if record_video=False
    """
    scales = (np.asarray(action_config.scales, dtype=np.float64)
              if action_config.mode != "absolute" else None)
    num_envs = worker.num_envs
    poses    = [_x_to_targets(x, 1) for x in all_best_x]
    pos_b    = np.concatenate([p[0] for p in poses], axis=0).astype(np.float32)  # (N, 3)
    quat_b   = np.concatenate([p[1] for p in poses], axis=0).astype(np.float32)  # (N, 4)
    grasp_pos = pos_b.copy()
    lift_b   = grasp_pos.copy(); lift_b[:, 2] += LIFT_HEIGHT

    width_open = np.full(num_envs, 0.08, np.float32)
    # Commanded width = planned width − the surrogate-selected closure. No 2.5 mm baseline and
    # no unconditional firm in v4: the surrogate's sigma(width) curve sets the squeeze.
    _cc = (np.full(num_envs, 0.002, np.float32) if closure_cmd is None
           else np.asarray(closure_cmd, np.float32).reshape(num_envs))
    width_cls  = np.array([max(0.004, p[2] - _cc[k]) for k, p in enumerate(poses)], np.float32)
    # Mutable — "firm" phase tightens this once per env (idea #1); "lift"/"hold"/
    # a frozen (DONE) env all read the FINAL width, which is width_cls unless firmed.
    grip_target = width_cls.copy()
    # Per-env firm close distance (m). v4 firms NOTHING unconditionally (base 0): a grasp is
    # firmed only if it comes out weak at the grasp->firm boundary. (v3's material-aware
    # base+weak scaling is retired — the surrogate already targets the stress level.)
    _firm_base = 0.0            # v4: no unconditional firm — width already targets the stress level
    # "weak" is judged by a stress RISE, and 2000 Pa is 5 % of the mushroom's 40 kPa yield but
    # 6.7 % of the cherry tomato's 30 kPa — so the same absolute bar means different things.
    # Express it as that same 5 % of the object's own yield (mushroom unchanged).
    firm_close = np.full(num_envs, _firm_base, np.float32)

    home_pos  = np.tile(worker.robot.home_pos[None].astype(np.float32),  (num_envs, 1))
    home_quat = np.tile(worker.robot.home_quat[None].astype(np.float32), (num_envs, 1))

    def _wxyz_to_rot(q): return Rot.from_quat([q[1], q[2], q[3], q[0]])
    home_r = _wxyz_to_rot(home_quat[0])
    slerps = [Slerp([0., 1.], Rot.concatenate([home_r, _wxyz_to_rot(quat_b[i])]))
              for i in range(num_envs)]


    def _env_target(i: int, phase_idx: int, phase_step: int):
        """(pos, quat_wxyz, grip) for env i at its OWN (phase_idx, phase_step)."""
        name, dur = PHASES[phase_idx]
        if name == "approach":
            alpha = (phase_step + 1) / dur
            pos = home_pos[i] + alpha * (pos_b[i] - home_pos[i])
            xyzw = slerps[i](alpha).as_quat()
            quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], np.float32)
            grip = width_open[i]
        elif name == "settle":
            pos, quat, grip = pos_b[i], quat_b[i], width_open[i]
        elif name == "grasp":
            alpha = (phase_step + 1) / dur
            pos, quat = pos_b[i], quat_b[i]
            grip = width_open[i] + alpha * (width_cls[i] - width_open[i])
        elif name == "firm":
            # Close by this env's firm_close (base for a firm grasp; base+extra for a weak one).
            # SOFT always passes through firm (never skipped) so the base grip margin is preserved;
            # RIGID only reaches here when its grip was weak (strong rigid grips skip firm).
            pos, quat = pos_b[i], quat_b[i]
            alpha = (phase_step + 1) / dur
            grip_target[i] = max(0.0, width_cls[i] - alpha * firm_close[i])
            grip = grip_target[i]
        elif name == "lift":
            alpha = (phase_step + 1) / dur
            pos = pos_b[i] + alpha * (lift_b[i] - pos_b[i])
            quat, grip = quat_b[i], grip_target[i]
        else:  # "hold"
            pos, quat, grip = lift_b[i], quat_b[i], grip_target[i]
        return pos, quat, grip

    def _frozen_target(i: int):
        """Command for a DONE env — hold steady so it doesn't disturb the sim
        while other envs keep progressing (identical to its final hold target)."""
        return lift_b[i], quat_b[i], grip_target[i]

    # Per-env buffers — lengths WILL differ across envs once retry logic diverges
    # phase durations; for now (no retry) every env still finishes at the same step.
    obs_bufs:   List[List[dict]]       = [[] for _ in range(num_envs)]
    act_bufs:   List[List[np.ndarray]] = [[] for _ in range(num_envs)]
    rew_bufs:   List[List[float]]      = [[] for _ in range(num_envs)]
    frame_bufs: List[List[np.ndarray]] = [[] for _ in range(num_envs)]

    # Split initial batched obs into per-env dicts
    cur_obs_list = [{k: init_obs_batch[k][i] for k in init_obs_batch}
                    for i in range(num_envs)]

    # Commanded targets (initialised to home; updated each step for action inversion)
    prev_pos  = home_pos.copy()
    prev_quat = home_quat.copy()
    prev_grip = width_open.copy()

    # Per-env FSM state
    phase_idx  = np.zeros(num_envs, dtype=np.int64)
    phase_step = np.zeros(num_envs, dtype=np.int64)

    # Soft firm-check baseline: top10 von-Mises of the SETTLED object (gripper still
    # at home, no contact) — captured on the first step, used as the "no grasp" floor
    # the grasp->firm rise is measured against. None until the first soft state.

    def _step(cur_pos, cur_quat, cur_grip, record_mask):
        # Invert scripted target → normalized action (delta needs prev_*; absolute
        # is a stateless per-step transform of the current target alone).
        if action_config.mode == "absolute":
            actions = _invert_actions_absolute(cur_pos, cur_quat, cur_grip, action_config)  # (N, 10)
        else:
            actions = _invert_actions(prev_pos, cur_pos, prev_quat, cur_quat,
                                      prev_grip, cur_grip, scales)  # (N, 7)

        # Advance sim (depth rendered because render_obs_cameras=True)
        state = worker.step(cur_pos, cur_quat, cur_grip)

        # RGB frames (optional) — only for envs still being recorded, so per-env
        # video length matches its recorded episode length.
        if record_video:
            frames = worker.render_rgb(all_envs=True)   # (N, H, W, 3) uint8
            if frames is not None:
                for i in range(num_envs):
                    if record_mask[i]:
                        frame_bufs[i].append(frames[i])

        # Build obs from next state (+ sim-only privileged fields from the object state)
        raw_next = _state_to_raw_obs(state)
        next_obs_batch = perception.process(raw_next)
        if priv_cfg is not None:
            oq = state.get("object_quat")                          # soft MPM: no rigid quat in the step
            if oq is None:                                         # state → identity placeholder (the
                oq = np.tile(np.array([1., 0, 0, 0], np.float32), (num_envs, 1))  # deployable student
            next_obs_batch.update(_privileged_obs_batch(           # uses point_cloud, not this)
                state["object_center"], oq, dr_vec, priv_cfg,
                contact_force=state.get("contact_force"),
                von_mises=state.get("von_mises_stress"), yield_stress=yield_stress))
        next_obs_list = [{k: next_obs_batch[k][i] for k in next_obs_batch}
                         for i in range(num_envs)]

        # Record (obs_t, action_t, reward_t=0) — ONLY for envs still active this step.
        # A DONE env's buffers simply stop growing here.
        for i in range(num_envs):
            if record_mask[i]:
                obs_bufs[i].append(cur_obs_list[i])
                act_bufs[i].append(actions[i])
                rew_bufs[i].append(0.0)

        prev_pos[:]  = cur_pos
        prev_quat[:] = cur_quat
        prev_grip[:] = cur_grip
        return next_obs_list, state

    # ── Main loop: every env advances through PHASES independently ──

    while np.any(phase_idx < N_PHASES):
        active = phase_idx < N_PHASES   # (N,) bool — envs still progressing this step

        cur_pos_arr  = np.zeros((num_envs, 3), np.float32)
        cur_quat_arr = np.zeros((num_envs, 4), np.float32)
        cur_grip_arr = np.zeros(num_envs, np.float32)
        for i in range(num_envs):
            if active[i]:
                pos, quat, grip = _env_target(i, int(phase_idx[i]), int(phase_step[i]))
            else:
                pos, quat, grip = _frozen_target(i)
            cur_pos_arr[i], cur_quat_arr[i], cur_grip_arr[i] = pos, quat, grip

        cur_obs_list, state = _step(cur_pos_arr, cur_quat_arr, cur_grip_arr, record_mask=active)


        # Advance phase state for envs that were active this step.
        # TODO(retry): this is where a robustness pass would check per-env
        # success/failure (e.g. at the "lift"/"hold" -> DONE boundary) and, on
        # failure, rewind phase_idx[i] to an earlier phase instead of advancing to
        # DONE — independent of every other env's state.
        phase_step[active] += 1
        # phase_idx may already be N_PHASES (done envs) — clip before indexing PHASES;
        # those entries are masked out by `active` anyway so the clipped value is unused.
        durations = np.array([PHASES[min(int(p), N_PHASES - 1)][1] for p in phase_idx])
        rolled_over = active & (phase_step >= durations)

        # Idea #1: force-based grasp firming. Checked ONCE per env, exactly at the
        # grasp->firm boundary. A weak grip (contact force < FIRM_FORCE_THRESH_N) closes
        # an extra min(0.5 * commanded closure, 2mm); every other env passes through
        # "firm" as a no-op hold that _trim_long_holds collapses afterwards. (Soft never
        # SKIPS the phase — an earlier skip-straight-to-lift path was removed.)
        advance = np.ones(num_envs, dtype=np.int64)   # normal: one phase forward
        leaving_grasp = rolled_over & (phase_idx == _GRASP_IDX)
        if np.any(leaving_grasp):
            # Gripper->object contact force (N). For a soft body this is the MPM->finger COUPLING
            # force (XArm7Sim.gripper_coupling_force), which genesis_worker always populates — so
            # this is the only firm path. (It used to have a von-Mises `else` for soft, written when
            # contact_force was rigid-only; that branch became unreachable and never fired in any
            # collected run — every [firm] line reads "weak grip force N".)
            cf = state.get("contact_force")
            if cf is not None:                        # only a WEAK grip closes further
                for i in np.where(leaving_grasp)[0]:
                    if cf[i] < FIRM_FORCE_THRESH_N:
                        firm_close[i] = min(0.5 * float(_cc[i]), 0.002)
                        print(f"    [firm] env {i}: weak grip force {cf[i]:.2f}N < "
                              f"{FIRM_FORCE_THRESH_N}N -> closing {firm_close[i]*1000:.1f}mm")
                    # else: no extra close, but still pass through the phase

        phase_idx[rolled_over]  += advance[rolled_over]
        phase_step[rolled_over]  = 0

    # Success check from final object height: the particle-mean centre from the last step's state
    # (MPMEntity has no get_pos; this collector is soft-only, see the object_type guard in main).
    obj_z = np.asarray(state["object_center"])[:, 2]
    success = obj_z > (grasp_pos[:, 2] + LIFT_HEIGHT * 0.5)

    # Post-process: trim long held-command runs (absolute mode only — see
    # _trim_long_holds docstring for why delta mode is excluded).
    if action_config.mode == "absolute":
        for i in range(num_envs):
            act_bufs[i], obs_bufs[i], rew_bufs[i], frame_bufs[i] = _trim_long_holds(
                act_bufs[i], obs_bufs[i], rew_bufs[i], frame_bufs[i])

    return obs_bufs, act_bufs, rew_bufs, success, frame_bufs


# ── Output helpers ────────────────────────────────────────────────────────────




# ── Main ──────────────────────────────────────────────────────────────────────






def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--experiment", required=True,
                   help="experiment name under configs/experiments/ "
                        "(e.g. single_lift_mushroom_soft_armfocus_stress) — source of task, obs, "
                        "action, and DR config")
    p.add_argument("--task-name",  default=None,
                   help="override output dataset name (default: experiment's task field)")
    p.add_argument("--out-dir",    type=Path, default=Path("dataset") / "demos")
    p.add_argument("--shard-size", type=int, default=5,
                   help="episodes per shard file (merged into data.pkl at end)")
    p.add_argument("--description", type=str, default="",
                   help="free-text annotation saved in config.yaml")

    p.add_argument("--n-episodes", type=int, default=50,
                   help="total successful episodes to collect")
    p.add_argument("--n-envs",     type=int, default=10,
                   help="parallel envs per batch")
    # ── v3 FEM gentleness synthesis ──
    # E / density / yield DEFAULT TO THE OBJECT'S OWN MATERIAL (resolved from the registry after the
    # experiment loads). They used to default to 3e5 / 1000 — the MUSHROOM's values — for every
    # object, so every non-mushroom collection planned grasps with the wrong material. That is not
    # cosmetic: the FEM is linear in E (sigma = E*sigma_1, F = E*F_1), so BOTH the predicted stress
    # AND the grip force were wrong per object. On the raspberry (true E 1e5) the planner believed
    # it had 3x the grip it actually had, and reported 24.8 kPa where the true figure is ~8.3 kPa.
    # Pass a value explicitly to override.
    p.add_argument("--table-z",           type=float, default=0.0,  help="table surface height (world z, m)")
    # ── Grasp-pose DIVERSITY (ON by default — these defaults broaden the demo distribution to match v2's
    # coverage: pitch σ≈14°, continuous yaw, at ~85% collect success. Set the knobs to 0/None for the old
    # single-argmax, concentrated behaviour (v3 otherwise pins pitch~0 + snaps yaw to a few gentle axes).
    p.add_argument("--scene-dr-every", type=int, default=1,
                   help="re-randomize object SIZE+SHAPE every N batches by rebuilding the worker "
                        "(needs shape/scale fields in the experiment DR config; 0 = off, nominal "
                        "geometry). Geometry is shared across a batch's envs (batched build).")
    p.add_argument("--seed",       type=int, default=0,
                   help="RNG seed for pose DR")
    p.add_argument("--record-video", nargs="?", type=int, const=10**9, default=0,
                   help="record per-episode mp4 videos + grasp-pose PNGs to <out-dir>/videos/ (slower). "
                        "Bare `--record-video` = ALL episodes; `--record-video N` = only the FIRST N saved "
                        "episodes (rendering stops after N -> no extra cost/disk on a long run). Off by default.")
    p.add_argument("--n-grasp", type=int, default=N_GRASP,
                   help=f"gripper-close steps (the 'grasp' phase length); default {N_GRASP}. A shorter "
                        "close reaches the target width sooner (less dwell before the lift).")
    p.add_argument("--dev-viz", action="store_true",
                   help="STANDALONE step-through window (not Genesis) for ENV 0 of each batch: each synthesis "
                        "stage is drawn and BLOCKS until you press q (GM_DEV_VIZ_AUTOADVANCE=<s> auto-plays).")
    p.add_argument("--dev-viewer", action="store_true",
                   help="DEV: force --n-envs 1 and open the Genesis viewer window so the scripted "
                        "grasp can be watched live. Slow and interactive — never use for collection.")
    args = p.parse_args()
    if getattr(args, "dev_viewer", False) and args.n_envs != 1:
        print(f"[dev-viewer] forcing --n-envs {args.n_envs} -> 1 (one viewer window, one env)")
        args.n_envs = 1

    # PHASES is module-level (execute_and_collect reads the globals directly); only the grasp
    # (close) length is a CLI knob.
    global PHASES, N_PHASES, _GRASP_IDX
    PHASES = [
        ("approach", N_HOME_TO_PRE),
        ("settle",   N_SETTLE),
        ("grasp",    args.n_grasp),
        ("firm",     N_FIRM),
        ("lift",     N_LIFT),
        ("hold",     N_HOLD),
    ]
    N_PHASES   = len(PHASES)
    _GRASP_IDX = [name for name, _ in PHASES].index("grasp")

    # ── Load everything from the experiment config (same as training / eval) ──
    exp        = Experiment.load(args.experiment)
    task       = SingleLiftTask(exp.task_cfg)
    # Grasp material comes from the OBJECT's registry entry — the experiment is the single source
    # of truth, so there are no CLI overrides for it.
    from gentle_manip.assets.registry import get_object_def as _god
    # SOFT-ONLY COLLECTOR (2026-09-04). Rigid support was dropped: the firm decision reads the
    # MPM->finger coupling force, the success check reads the particle-mean centre, and the FEM
    # gentleness metric is meaningless for a body that does not deform. A rigid object would run
    # but produce grasps selected by a metric that cannot apply, so fail loudly instead.
    _otype = str(exp.task_cfg.get("object_type", "soft")).lower()
    if _otype != "soft":
        raise SystemExit(
            f"collect_demos_synth_v4 is SOFT-ONLY, but experiment {args.experiment!r} has "
            f"object_type={_otype!r} (task {exp.task_cfg.get('object_name')!r}). Rigid collection "
            f"was removed on 2026-09-04 — use an MPM/soft object, or collect_demos_synth_v3.py.")
    _mat = _god(exp.task_cfg["object_name"]).material
    _mat_E     = float(_mat.youngs_modulus)
    _mat_rho   = float(_mat.density)
    _mat_yield = float(_mat.von_mises_yield_stress)
    _fem_nu = fg.FEM_NU                 # one FEM Poisson ratio for every object (planner constant)
    print(f"  grasp material ({exp.task_cfg['object_name']}): E={_mat_E:.3g} Pa  "
          f"rho={_mat_rho:.0f}  yield={_mat_yield:.3g} Pa  (FEM nu={_fem_nu})")
    spec       = task.scene_spec
    obs_config = exp.collection_obs()
    priv_cfg   = obs_config.privileged        # sim-only state-teacher fields (None if not requested)
    action_config = exp.action_config
    dr_cfg     = DRConfig.from_dict(exp.dr)
    task_name  = args.task_name or exp._raw.get("task", args.experiment)
    rate_hz    = 1.0 / spec.sim_dt

    # Settle params: task config → CLI override.
    settle_steps     = int(exp.task_cfg.get("settle_steps",     30))
    settle_max_steps = int(exp.task_cfg.get("settle_max_steps", 200))
    settle_vel_thresh = float(exp.task_cfg.get("settle_vel_thresh", 0.002))

    perception = PerceptionPipeline(obs_config)

    collection_config = {
        "task_name":   task_name,
        "description": args.description,
        "source":      "cmaes_synth",
        "git_commit":  _git_commit(),
        "experiment":  args.experiment,
        "control":     {"n_envs": args.n_envs,
                        "n_episodes": args.n_episodes, "scene_dr_every": args.scene_dr_every,
                        "seed": args.seed, "n_home_to_pre": N_HOME_TO_PRE,
                        "n_grasp": args.n_grasp, "n_lift": N_LIFT, "n_firm": N_FIRM,
                        "n_settle": N_SETTLE,
                        "held_run_max": HELD_RUN_MAX, "held_run_keep": HELD_RUN_KEEP,
                        "scan_metric": SCAN_METRIC,
                        "closure_gain": CLOSURE_GAIN,
                        "grasp_nu": _fem_nu, "mat_E": _mat_E, "mat_rho": _mat_rho, "mat_yield": _mat_yield,
                        },
        "dr": exp.dr,
    }

    print(f"\n=== collect_demos_synth  experiment={args.experiment}"
          f" — target {args.n_episodes} episodes, {args.n_envs} envs/batch")

    rng = np.random.default_rng(args.seed)   # DR RNG (pose + scene) — must precede the first build
    # Separate stream (distinct offset so it never shares draws with `rng` above, keeping
    # DR reproducibility untouched) — makes CMA-ES's own search reproducible from --seed too
    # (previously every _synth_worker call used run_cmaes's hardcoded default seed=2567, so
    # ALL envs/batches shared the identical internal search sequence regardless of --seed).
    cma_seed_rng = np.random.default_rng(args.seed + 1_000_000)

    # ── Build scene + worker (with per-scene SIZE+SHAPE DR) ──
    # Scene DR re-randomizes object geometry by REBUILDING the worker every N batches (GenesisWorker
    # has no in-process geometry re-randomize; a fresh build is the only path). Verified memory-stable
    # across rebuilds (gs.destroy reclaims each build). Geometry is shared across a batch's envs.
    nominal_spec   = spec
    do_scene_dr    = args.scene_dr_every > 0 and dr_cfg.has_scene_dr()
    deform_dir     = tempfile.mkdtemp(prefix="gm_synth_deform_") if do_scene_dr else None

    print(f"  closure scan: metric={SCAN_METRIC} gain={CLOSURE_GAIN}")
    _mesh_cycle = None    # round-robin cursor (see --mesh-cycle);
                                                     # must precede the first _make_worker() call
    def _make_worker():
        """Build a GenesisWorker; if scene DR is on, on a freshly deformed+scaled mesh.
        Returns (worker, scene_dr_dict, actual_mesh_path)."""
        if do_scene_dr:
            spec_dr, sdr = _apply_scene_dr(nominal_spec, dr_cfg, rng, deform_dir,
                                           mesh_cycle=_mesh_cycle)
        else:
            spec_dr, sdr = nominal_spec, {"scale": float(nominal_spec.objects[0].scale or 1.0),
                                          "bend_deg": 0.0}
        w = GenesisWorker(spec_dr, num_envs=args.n_envs, show_viewer=bool(args.dev_viewer),
                          settle_steps=settle_steps, settle_max_steps=settle_max_steps,
                          settle_vel_thresh=settle_vel_thresh, render_obs_cameras=True,
                          coup_friction=float(sdr.get("coup_friction", 4.0)))
        _mp = w.handle.spec.objects[0].mesh_path
        if not _mp:
            # Silently falling back to the mushroom mesh would synthesize grasps for the WRONG
            # object and look like a bad recipe rather than a bad config. Every object in the
            # registry carries mesh_path; a missing one is a config error, so say so.
            raise RuntimeError(
                f"object {w.handle.spec.objects[0].name!r} has no mesh_path — the scene spec is "
                f"incomplete. Fix the object's registry entry / task config; the collector will "
                f"NOT fall back to a default mesh.")
        return w, sdr, _mp

    worker, scene_dr, actual_mesh = _make_worker()
    if do_scene_dr:
        print(f"  scene DR ON (every {args.scene_dr_every} batch(es)) — deformed meshes → {deform_dir}")

    # ── Everything below is CPU-only (FEM metric + CMA-ES; the GPU solve is opt-in) ──

    # Process pool reused across all batches (N workers, one per env).
    print(f"  Mesh: {Path(actual_mesh).name}")   # v3 synthesizes in-process (no worker pool)

    # ── Output dir + config snapshot ──
    run_dir  = _make_run_dir(args.out_dir, task_name)
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(collection_config, f, sort_keys=False)
    print(f"  Config → {cfg_path.resolve()}")
    print(f"  Data   → {run_dir.resolve()}/data.pkl  (shards flushed every {args.shard_size} ep)")

    # Per-env-per-batch DR + grasp log (CSV, alongside the data). One row per env each batch.
    dr_csv = open(run_dir / "dr_params.csv", "w", newline="")
    dr_writer = csv.writer(dr_csv)
    dr_writer.writerow(["batch", "env", "success", "obj_dx", "obj_dy",
                        "roll_deg", "pitch_deg", "yaw_deg", "flipped",
                        "home_dx", "home_dy", "home_dz", "scene_scale", "scene_bend_deg",
                        "mesh_variant", "closure_cmd_mm",
                        "twist_deg", "taper", "rbf", "axis_scale", "axis_scale_ax",
                        "mat_E", "mat_nu", "mat_rho", "coup_friction",
                        "stress_Pa", "grip_N", "align", "pressure_Pa", "min_pad_mm2", "width_mm", "tilt_deg",
                        # dataset_idx: 0-based index into data.pkl["episodes"], or -1 if this
                        # attempt was NOT saved. Ported from v3 (2026-08-30). Without it the
                        # CSV<->dataset join is UNDERIVABLE: v4 logs every attempt, and a
                        # `success=1` row still may not be saved (n_episodes cap, or the
                        # fallback-grasp drop), so "the successes in order" is wrong.
                        "dataset_idx",
                        # scan_metric: guardrail #3 in docs/collection_v4_handoff.md.
                        "scan_metric"])

    total_saved  = 0
    total_failed = 0
    total_fallback_dropped = 0   # succeeded-by-crushing fallback demos, dropped
    batch_idx   = 0
    shard_buf:  List[dict] = []
    shard_idx   = 0
    fem_mesh: Optional[str] = None            # v3: cache the FEM (obj+pad_geo) keyed on actual_mesh —
    fem_obj = fem_pad_geo = fem_meta = None   #     rebuild only when a scene-DR relaunch changes the mesh
    t0 = time.time()
    consec_batch_aborts = 0        # unstable-scene batch discards (see execute_and_collect guard)

    while total_saved < args.n_episodes:
        batch_idx += 1
        n = args.n_envs

        # ── Scene DR: rebuild the worker with fresh object SIZE+SHAPE every N batches ──
        if do_scene_dr and batch_idx > 1 and (batch_idx - 1) % args.scene_dr_every == 0:
            worker.close()
            worker, scene_dr, actual_mesh = _make_worker()

        print(f"\n── Batch {batch_idx}  [{total_saved}/{args.n_episodes} saved]"
              + (f"  scale={scene_dr['scale']:.3f} bend={scene_dr['bend_deg']:+.1f}°"
                   f" mesh={scene_dr.get('mesh_variant', '-')}"
                 if do_scene_dr else "") + " ──")

        # ── Reset with per-env pose DR (ranges from experiment DR config) ──
        # Bounded resilience: a rare scene draw (pose × scale × shape) can NaN the solver at
        # settle ("Invalid constraint forces" — seen 1300574, 1643324, both nominal-recipe-
        # reachable). Rebuild the scene with a FRESH DR draw and retry instead of dying;
        # persistent failures still raise after the retry budget.
        for _settle_try in range(4):
            object_dxy   = dr_cfg.sample_object_dxy(rng, n)
            object_euler = dr_cfg.sample_object_euler(rng, n)
            home_offset  = dr_cfg.sample_home_offset(rng, n)   # per-env arm-home jitter (sim-only DR)
            try:
                worker.reset(object_dxy=object_dxy, object_euler=object_euler, home_offset=home_offset)

                # Extra settling. Rigid objects roll → settle until velocity is small; soft MPM objects have no
                # rigid get_vel/get_ang (and are placed resting, not dropped) → a brief fixed settle suffices.
                obj = worker.handle.objects[0]
                if hasattr(obj, "get_vel"):
                    for _ in range(600):
                        worker.handle.scene.step()
                        if np.abs(_np(obj.get_vel())).max() < 0.003 and np.abs(_np(obj.get_ang())).max() < 0.01:
                            break
                else:                                          # soft MPM
                    for _ in range(30):
                        worker.handle.scene.step()
                break
            except Exception as e:
                if _settle_try == 3:
                    raise
                print(f"  !! reset/settle blew up ({type(e).__name__}: {e}) — rebuilding scene "
                      f"with a fresh DR draw (retry {_settle_try + 1}/3)")
                try:
                    worker.close()
                except Exception:
                    pass
                worker, scene_dr, actual_mesh = _make_worker()
                print(f"   retry scene: scale={scene_dr['scale']:.3f} bend={scene_dr['bend_deg']:+.1f}°"
                      f" mesh={scene_dr.get('mesh_variant', '-')}")

        # Read initial state (depth rendered; this is obs_0 for every env)
        init_state     = worker.read_state()
        raw_init       = _state_to_raw_obs(init_state)
        init_obs_batch = perception.process(raw_init)

        # Object pose (SETTLED) for synthesis + priv obs, straight from read_state: rigid = get_quat,
        # soft = Kabsch best-fit rotation of the settled particle cloud (NOT the spawn euler, which the
        # object leaves once it falls/settles under gravity). Same key for both.
        obj_pos_all  = init_state["object_center"].astype(np.float64)  # (N, 3)
        obj_quat_all = np.asarray(init_state["object_quat"], np.float64)  # (N, 4) wxyz

        # Episode scene-DR vector [scale, bend_deg] for priv_object_dr_params (mirrors SimBackend).
        dr_vec = np.array([float(scene_dr.get("scale", 1.0)),
                           float(scene_dr.get("bend_deg", 0.0))], dtype=np.float32)
        if priv_cfg is not None:
            init_obs_batch.update(_privileged_obs_batch(
                obj_pos_all, obj_quat_all, dr_vec, priv_cfg,
                von_mises=init_state.get("von_mises_stress"), yield_stress=_mat_yield,
                contact_force=init_state.get("contact_force")))

        # ── Per-env FEM gentleness grasp synthesis (v3) ──
        # Build the FEM ElasticObject ONCE for this batch's ACTUAL (DR shape+size) mesh — all envs share
        # it (scene-DR varies per relaunch, not per sub-env), so the expensive factorization is reused;
        # rebuild only when actual_mesh changes (a scene-DR relaunch). Then plan per-env settled pose.
        if actual_mesh != fem_mesh:
            fem_obj, fem_pad_geo, fem_meta = fg.build_grasp_fem(
                actual_mesh, voxel_div=GRASP_VOXEL_DIV, target_tets=GRASP_TARGET_TETS,
                use_gpu=True, nu=_fem_nu)
            fem_mesh = actual_mesh
            print(f"  FEM: {fem_meta['tets']} tets, ndof={fem_meta['ndof']}, gpu={fem_meta['gpu']}")
            _viz_pad_geo = fem_pad_geo          # pad placement is known only once the FEM is built
        all_best_x = []
        closure_cmd: list = []          # v4: per-env surrogate-selected closure (m)
        synth_failed: set[int] = set()      # envs running the fallback grasp (never saved)
        all_grasp  = []                                          # per-env synthesis dict (for the grasp-pose viz)
        for i in range(n):
            cma_seed = int(cma_seed_rng.integers(1, 2**31 - 1))
            live_viz = (StageViewer(fem_obj, obj_pos_all[i], obj_quat_all[i], _viz_pad_geo,
                                    label=f"batch {batch_idx} env {i} — {args.experiment}")
                        if (args.dev_viz and i == 0) else None)    # step-through window: env 0 only
            r = fg.synthesize_grasp(fem_obj, fem_pad_geo, obj_pos_all[i], obj_quat_all[i],
                                    E=_mat_E, density=_mat_rho, mu=GRASP_MU,
                                    table_z=args.table_z, seed=cma_seed,
                                    yield_stress=_mat_yield,
                                    record_history=bool(args.dev_viz),
                                    stage_cb=(live_viz.on_stage if live_viz is not None else None))
            best_x = r["x"]
            if best_x is None or r.get("stress_top10") is None:       # extremely rare: still nothing ->
                # default straight-down grasp at the object xy so the FSM never sees None (this episode may
                # not lift -> simply won't be saved, but the batch completes). tcp sits low (FINGER_TO_TCP_Z).
                best_x = np.array([obj_pos_all[i][0], obj_pos_all[i][1],
                                   float(obj_pos_all[i][2]) + FINGER_TO_TCP_Z, np.pi, 0.0, 0.0, 0.045])
                r = {"x": best_x, "stress_top10": None, "grip": None, "align": None,
                     "pressure": None, "min_pad_area": None, "width_face": None}
                print(f"  Env {i}: SYNTH FAILED -> default top-down grasp (fallback)")
                synth_failed.add(i)
            # v4: SURROGATE-SELECTED closure — scan sigma(width) at the chosen pose, take the
            # predicted yield-crossing c_y, command gain * c_y. Uses the DR-drawn E when the scene
            # sampled one (the scan must match the simulated material; yield is not DR'd).
            if r.get("stress_top10") is not None:
                _E_scan = float(scene_dr.get("mat_E") or 0.0) or float(_mat_E)
                _cy = surrogate_closure(fem_obj, fem_pad_geo, best_x,
                                        obj_pos_all[i], obj_quat_all[i],
                                        _E_scan, float(_mat_yield),
                                        _mat_rho, GRASP_MU, args.table_z,
                                        metric=SCAN_METRIC)
                _cc = float(np.clip(CLOSURE_GAIN * _cy, CLOSURE_CMD_MIN_M, CLOSURE_CMD_MAX_M))
                print(f"  Env {i}: c_y={_cy*1000:.1f}mm -> commanded closure {_cc*1000:.1f}mm")
            else:
                _cc = 0.002                      # fallback grasp: episodes are dropped anyway
            closure_cmd.append(_cc)
            all_best_x.append(best_x); all_grasp.append(r)
            if r.get("stress_top10") is not None:
                print(f"  Env {i}: stress={r['stress_top10']:.0f}Pa grip={r['grip']:.3f}N align={r['align']:.3f}"
                      f"  tcp={best_x[:3].round(4)}  w={best_x[6]*1e3:.1f} mm")

        def _save_grasp_pose(vid_dir, stem, i):
            """Render the METRIC's predicted grasp (pose + expected stress/grip/align) next to env i's
            execution video — captures this batch's FEM + per-env grasp. Non-fatal on failure."""
            try:
                g = all_grasp[i]
                finger_viz.render_grasp_pose(
                    fem_obj, fem_pad_geo, all_best_x[i], obj_pos_all[i], obj_quat_all[i], args.table_z,
                    str(Path(vid_dir) / f"{stem}_grasp.png"), E=_mat_E,
                    stress=g.get("stress_top10"), grip=g.get("grip"), align=g.get("align"),
                    width_face=g.get("width_face"), label=stem)
            except Exception as e:  # viz must never break a collection run
                print(f"    (grasp viz failed: {e})")

        # ── Execute scripted trajectory + collect data ──
        # Record video only while under the first-N cap (args.record_video = N, or 10**9 for "all");
        # once N saved, stop RENDERING (no per-step RGB cost/disk for the rest of the run).
        rec_this_batch = args.record_video > 0 and total_saved < args.record_video
        print(f"  Executing …")
        try:
            obs_bufs, act_bufs, rew_bufs, success, frame_bufs = execute_and_collect(
                worker, all_best_x, init_obs_batch, perception, action_config,
                record_video=rec_this_batch, priv_cfg=priv_cfg, dr_vec=dr_vec,
                yield_stress=_mat_yield, closure_cmd=closure_cmd)
            consec_batch_aborts = 0
        except Exception as e:
            # Some scene draws are systematically unstable (solver NaN mid-episode even after a
            # clean settle — e.g. mushroom3 @ scale 1.49, jobs 1643324/1647974, GPU nondeterminism
            # only moves WHERE it blows). Discard the batch, rebuild with a fresh DR draw, continue.
            consec_batch_aborts += 1
            print(f"  !! batch aborted mid-execution ({type(e).__name__}: {e}) — discarding and "
                  f"rebuilding with a fresh scene draw ({consec_batch_aborts} consecutive abort(s))")
            if consec_batch_aborts >= 5:
                raise
            try:
                worker.close()
            except Exception:
                pass
            worker, scene_dr, actual_mesh = _make_worker()
            continue
        print(f"  Success: {success.tolist()}")

        # ── Log per-env DR + grasp params for this batch (CSV row per env) ──
        # Rows are BUFFERED, not written yet: dataset_idx is only known after the save loop.
        _dr_rows = []
        _DS_IDX_COL = -2                     # dataset_idx is the 2nd-to-last column
        eul_deg = np.degrees(object_euler) if object_euler is not None else np.zeros((n, 3))
        for i in range(n):
            g = all_grasp[i]
            roll, pitch, yaw = eul_deg[i]
            flipped = int(abs(roll) > 140 or abs(pitch) > 140)                # a big-flip sample
            ho = home_offset[i] if home_offset is not None else (0.0, 0.0, 0.0)
            odxy = object_dxy[i] if object_dxy is not None else (0.0, 0.0)
            _dr_rows.append([batch_idx, i, int(bool(success[i])),
                                round(float(odxy[0]), 5), round(float(odxy[1]), 5),
                                round(float(roll), 1), round(float(pitch), 1), round(float(yaw), 1), flipped,
                                round(float(ho[0]), 5), round(float(ho[1]), 5), round(float(ho[2]), 5),
                                round(float(scene_dr.get("scale", 1.0)), 4),
                                round(float(scene_dr.get("bend_deg", 0.0)), 2),
                                scene_dr.get("mesh_variant", ""),
                                round(1000 * float(closure_cmd[i]), 2),
                                round(float(scene_dr.get("twist_deg", 0.0)), 2),
                                round(float(scene_dr.get("taper", 0.0)), 4),
                                round(float(scene_dr.get("rbf", 0.0)), 4),
                                round(float(scene_dr.get("axis_scale", 1.0)), 4),
                                int(scene_dr.get("axis_scale_ax", -1)),
                                round(float(scene_dr.get("mat_E", 0.0)), 1),
                                round(float(scene_dr.get("mat_nu", 0.0)), 4),
                                round(float(scene_dr.get("mat_rho", 0.0)), 1),
                                round(float(scene_dr.get("coup_friction", 4.0)), 3),
                                round(float(g.get("stress_top10") or 0), 1), round(float(g.get("grip") or 0), 4),
                                round(float(g.get("align") or 0), 4), round(float(g.get("pressure") or 0), 1),
                                round(float((g.get("min_pad_area") or 0) * 1e6), 2),
                                round(float(g["x"][6] * 1e3), 2),
                                round(float(g.get("tilt_deg") or 0), 1),
                                None,                      # dataset_idx, stamped in the save loop
                                SCAN_METRIC])

        # ── Package and shard successful (or all) episodes ──
        for i in range(n):
            # Always save failure video (if recording) before skipping demo data.
            if not success[i]:
                total_failed += 1
                if rec_this_batch and frame_bufs[i]:
                    vid_dir = run_dir / "videos_failed"
                    vid_dir.mkdir(exist_ok=True)
                    vid_path = vid_dir / f"fail{total_failed:04d}_b{batch_idx}_env{i}.mp4"
                    imageio.mimwrite(str(vid_path), frame_bufs[i], fps=round(rate_hz), quality=8)
                    _save_grasp_pose(vid_dir, vid_path.stem, i)
                    print(f"    fail video → {vid_path.name}")
                continue

            if i in synth_failed:
                # Fallback (no synthesized grasp): drop the demo even though it "succeeded" — it
                # succeeded by crushing. The failure video above is still written for inspection.
                total_fallback_dropped += 1
                print(f"    ep skipped: env {i} used the FALLBACK grasp (synthesis failed)")
                continue

            obs_list = obs_bufs[i]
            if not obs_list:
                continue
            keys    = obs_list[0].keys()
            episode = {
                "observations": {k: np.stack([o[k] for o in obs_list]) for k in keys},
                "actions":      np.stack(act_bufs[i]),
                "rewards":      np.asarray(rew_bufs[i], np.float32),
            }
            shard_buf.append(episode)
            _dr_rows[i][_DS_IDX_COL] = total_saved       # 0-based index into data.pkl["episodes"]
            total_saved += 1
            print(f"    ep {total_saved}: env {i}  {'✓' if success[i] else '✗'}  "
                  f"T={episode['actions'].shape[0]}")

            if frame_bufs[i] and total_saved <= args.record_video:   # first-N cap (precise)
                vid_dir = run_dir / "videos"
                vid_dir.mkdir(exist_ok=True)
                vid_path = vid_dir / f"ep{total_saved:04d}_env{i}_success.mp4"
                imageio.mimwrite(str(vid_path), frame_bufs[i], fps=round(rate_hz), quality=8)
                _save_grasp_pose(vid_dir, vid_path.stem, i)
                print(f"    video → {vid_path.name}")

            if len(shard_buf) >= args.shard_size:
                sp = _write_shard(run_dir, shard_buf, task_name, shard_idx, rate_hz)
                print(f"  Shard {shard_idx} → {sp.name}")
                shard_idx += 1
                shard_buf = []

            if total_saved >= args.n_episodes:
                break

        # Written HERE, after the save loop, so dataset_idx reflects what was ACTUALLY saved.
        for _r in _dr_rows:
            if _r[_DS_IDX_COL] is None:
                _r[_DS_IDX_COL] = -1
            dr_writer.writerow(_r)
        dr_csv.flush()

    # ── Flush + merge ──
    if shard_buf:
        _write_shard(run_dir, shard_buf, task_name, shard_idx, rate_hz)

    dr_csv.close()
    data_path = _merge_shards(run_dir)
    elapsed   = time.time() - t0

    total_attempts = total_saved + total_failed
    success_rate   = total_saved / total_attempts if total_attempts > 0 else 0.0

    print(f"\n=== Done ===")
    print(f"  Episodes saved   : {total_saved}")
    print(f"  Episodes failed  : {total_failed}")
    print(f"  Total attempts   : {total_attempts}")
    print(f"  Success rate     : {success_rate*100:.1f}%")
    if total_fallback_dropped:
        print(f"  Fallback dropped : {total_fallback_dropped}  (synthesis failed -> crushing demo, not saved)")
    print(f"  Elapsed          : {elapsed/60:.1f} min")
    print(f"  Data             : {data_path}")

    stats = {
        "episodes_saved":  total_saved,
        "episodes_failed": total_failed,
        "episodes_fallback_dropped": total_fallback_dropped,
        "total_attempts":  total_attempts,
        "success_rate":    round(success_rate, 4),
        "elapsed_min":     round(elapsed / 60, 2),
    }
    stats_path = run_dir / "stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False)
    print(f"  Stats            : {stats_path}")

    worker.close()


if __name__ == "__main__":
    main()
