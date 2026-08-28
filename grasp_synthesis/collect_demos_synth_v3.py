"""Autonomous sim demo collection via FEM GENTLENESS grasp synthesis (v3).

Fork of collect_demos_synth_v2.py — IDENTICAL pipeline (reset with pose DR → per-env synthesis →
scripted execution → record (obs, action, reward) in demos/record.py format) EXCEPT the grasp is
synthesized by the width-controlled FEM gentleness metric (`smgrasp.finger_grasp`) instead of the
geometric SDF cost. The metric minimizes indentation stress of the gentlest grip that still holds,
using the real finger geometry + a table/finger-penetration filter, over the SAME 7-DOF TCP grasp
`[tx,ty,tz,roll,pitch,yaw,width]` the execution FSM drives — so this is a drop-in synthesis swap.

The FEM ElasticObject is built ONCE per batch from the sim's ACTUAL (shape+size DR) mesh and reused
across all envs (they share geometry); it rebuilds only on a scene-DR relaunch. v2 is the untouched
baseline.  Config comes from --experiment (task / obs / action / DR).

Usage:
    uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v3.py \\
        --experiment single_lift_mushroom_rigid \\
        --n-episodes 50 --n-envs 5 [--grasp-gpu]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import pickle
import dataclasses
import random
import string
import subprocess
import tempfile
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from synth_utils import (  # noqa: E402
    sample_finger_surface,
    FINGER_TO_TCP_Z,
)
from smgrasp import finger_grasp as fg  # noqa: E402  (v3: FEM gentleness synthesis replaces the SDF cost)
from smgrasp import finger_viz          # noqa: E402  (grasp-pose viz paired with each execution video)
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
OBJ_SIZE      = np.array([0.05, 0.05, 0.04])   # rough mushroom AABB half-size

MUSHROOM_MESH = str(ROOT / "gentle_manip/assets/objects/mushroom.obj")
LEFT_FINGER   = str(ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/left_finger.STL")
RIGHT_FINGER  = str(ROOT / "gentle_manip/assets/xarm/xarm_gripper/meshes/right_finger.STL")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _np(tensor) -> np.ndarray:
    """CUDA tensor or array → numpy (safe for both)."""
    return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synth_bounds(obj_pos: np.ndarray):
    """Compute CMA-ES search bounds from object position."""
    t_lb_xy = (obj_pos[:2] - 1.5 * OBJ_SIZE[:2]).tolist()
    t_ub_xy = (obj_pos[:2] + 1.5 * OBJ_SIZE[:2]).tolist()
    tcp_z_min = float(obj_pos[2]) + FINGER_TO_TCP_Z - 0.04
    tcp_z_max = float(obj_pos[2]) + 0.25
    lb = t_lb_xy + [tcp_z_min, -1.0 * np.pi, -0.12 * np.pi, -0.49 * np.pi, 0.01]
    ub = t_ub_xy + [tcp_z_max,  1.0 * np.pi,  0.12 * np.pi,  0.49 * np.pi, 0.08]
    return lb, ub


def _synth_worker(payload: tuple) -> tuple:
    """Module-level CMA-ES worker — runs in a subprocess, builds its own SDF.

    Defined at module level so ProcessPoolExecutor can pickle it.  Each worker
    process (forked before CUDA init) is CPU-only; trimesh BVH is safe because
    no two workers share memory.

    payload: (mesh_path, obj_pos, obj_quat_wxyz, left_pts, right_pts, maxfevals, lb, ub,
              log_dir, seed)
    returns: (best_x (7,), score float)
    """
    (mesh_path, obj_pos, obj_quat_wxyz, left_pts, right_pts, maxfevals, lb, ub,
     log_dir, seed) = payload
    from synth_utils import build_object_sdf, grasp_cost, run_cmaes  # local import: safe in subprocess
    sdf_fn = build_object_sdf(mesh_path)
    x0 = [(lo + hi) / 2 for lo, hi in zip(lb, ub)]

    def objective(x):
        return grasp_cost(x, left_pts, right_pts, sdf_fn, obj_pos, obj_quat_wxyz)

    best_x, score = run_cmaes(objective, x0, 1.0, lb, ub, maxfevals, seed=seed, log_dir=log_dir)
    return best_x, score


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
    """Per-scene SIZE + SHAPE DR for the (rigid) object — mirrors
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

# ── Robustness idea #1: force/stress-based grasp firming ──────────────────────
# At the moment an env finishes "grasp" (checked ONCE, per env, at the grasp->firm
# boundary), read its just-measured grip signal. If the grasp came out weak, close
# an EXTRA FIRM_EXTRA_CLOSE_M over the "firm" phase before lifting; otherwise skip
# "firm" entirely. Bounded to fire once per env (a one-way linear FSM check, not a
# retry loop). The signal is object-type dependent:
#   RIGID: gripper->object contact FORCE (N). Weak ⟺ force < FIRM_FORCE_THRESH_N.
#   SOFT:  von-Mises stress top-10% RISE over the settled-rest baseline (Pa) — a
#          missed/grazing grasp barely perturbs the body, so its stress stays near
#          rest. Weak ⟺ rise < FIRM_STRESS_THRESH_PA. (Force is None for MPM.)
FIRM_FORCE_THRESH_N  = 1.0     # rigid: below this measured contact force -> needs firming
FIRM_STRESS_THRESH_PA = 2000.0 # soft: below this top10 von-Mises rise (Pa) -> grasp came out WEAK
FIRM_EXTRA_CLOSE_M   = 0.002   # BASE firm close (m, 2.0mm) — applied to EVERY soft grasp. This is the
                               # unconditional grip margin the old soft path gave all grasps; dropping
                               # it (skip-firm) cost ~15% success (skip-firm fails 39% vs firm 24%).
FIRM_WEAK_EXTRA_CLOSE_M = 0.0025  # soft: ADDITIONAL close (m, 2.5mm) on top of the base when the grasp
                               # came out weak (stress rise < FIRM_STRESS_THRESH_PA, or rigid force <
                               # FIRM_FORCE_THRESH_N) — the "squeeze more only when NOT grasped"
                               # robustness. Soft NEVER skips firm; weak firms base+extra = 4.5mm.


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
    extra_close: float = 0.0,      # squeeze this many meters TIGHTER than the synthesized width (all grasps)
    approach_xy_finish=None,       # v3.2: (lo, hi) per-env xy-progress finish fraction; None = straight line
    approach_speed=None,           # v3.3: m/step — per-env approach DURATION = dist/speed (constant
                                   # speed like real teleop; None = fixed shared duration)
    approach_rng=None,             # rng for the per-env f_i draw (reproducibility)
    trim_max_run: int = HELD_RUN_MAX,   # v3.2: held-run trim knobs (end-of-episode stop supervision)
    trim_keep: int = HELD_RUN_KEEP,
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
    # Base close = synthesized width - 2.5mm; extra_close squeezes TIGHTER still (firmer grip, all grasps).
    width_cls  = np.array([max(0.0, p[2] - 0.0025 - extra_close) for p in poses], np.float32)
    # Mutable — "firm" phase tightens this once per env (idea #1); "lift"/"hold"/
    # a frozen (DONE) env all read the FINAL width, which is width_cls unless firmed.
    grip_target = width_cls.copy()
    # Per-env firm close distance (m). SOFT firms EVERY grasp by the base amount (the grip
    # margin the old path gave all grasps); a weak grasp (low stress rise) gets set to a LARGER
    # value at the grasp->firm boundary. RIGID leaves this at base and skips firm when already firm.
    firm_close = np.full(num_envs, FIRM_EXTRA_CLOSE_M, np.float32)
    _has_firm  = any(n == "firm" for n, _ in PHASES)   # --n-firm 0 drops the phase: skip the check too

    home_pos  = np.tile(worker.robot.home_pos[None].astype(np.float32),  (num_envs, 1))
    home_quat = np.tile(worker.robot.home_quat[None].astype(np.float32), (num_envs, 1))

    def _wxyz_to_rot(q): return Rot.from_quat([q[1], q[2], q[3], q[0]])
    home_r = _wxyz_to_rot(home_quat[0])
    slerps = [Slerp([0., 1.], Rot.concatenate([home_r, _wxyz_to_rot(quat_b[i])]))
              for i in range(num_envs)]

    # v3.2 real-style approach: per-env xy-progress finish fraction f_i. The real teleop
    # approach converges xy EARLY (median: xy aligned at 60% of the pre-close time, still
    # ~84mm above the grasp) while z descends ~linearly throughout — captured here as
    # per-AXIS progress profiles over the same phase (CONTINUOUS: no via-point, no corner,
    # the trajectory never stops; xy eases in with a smoothstep while z keeps moving).
    # v3.3 approach speed compensation: with a FIXED duration, per-episode speed is
    # proportional to spawn distance (measured corr 0.91 in sim vs 0.29 in real — humans
    # move at a ~constant preferred speed and just take longer for farther objects).
    # Per-env duration = distance / speed reproduces that, and makes the commanded lead
    # (the BC action magnitude) a near-deterministic function of the current state.
    _APPR_IDX = 0    # "approach" is always the first phase

    def _profile_path_len(xy_len, z_len, f):
        """Arc length of the per-axis approach profile (xy smoothstep to f, z linear).
        The path is LONGER than the straight 3D distance (xy finishes early -> curved);
        duration must be path/speed, not dist/speed, for truly constant per-step speed."""
        al = np.linspace(0.0, 1.0, 201)
        x = np.minimum(al / f, 1.0)
        s = x * x * (3.0 - 2.0 * x)
        dsxy = np.gradient(s, al)
        return float(np.trapezoid(np.sqrt((xy_len * dsxy) ** 2 + z_len ** 2), al))

    if approach_xy_finish is not None:
        _rng = approach_rng if approach_rng is not None else np.random.default_rng(0)
        xy_finish = _rng.uniform(approach_xy_finish[0], approach_xy_finish[1], num_envs)
    else:
        xy_finish = None

    xy_len = np.linalg.norm(pos_b[:, :2] - home_pos[:, :2], axis=1)
    z_len = np.abs(pos_b[:, 2] - home_pos[:, 2])
    if approach_speed is not None:
        if xy_finish is not None:
            path = np.array([_profile_path_len(xy_len[i], z_len[i], xy_finish[i])
                             for i in range(num_envs)])
        else:
            path = np.linalg.norm(pos_b - home_pos, axis=1)
        appr_dur = np.clip(np.round(path / float(approach_speed)), 40, 130).astype(np.int64)
    else:
        appr_dur = np.full(num_envs, int(dict(PHASES)["approach"]), np.int64)

    if xy_finish is not None:
        # Speed guard: smoothstep peak xy speed = 1.5 * xy_len / (f * dur). Floor f so the
        # peak stays <= XY_V_MAX (real p95 band + deploy rate clamp). Raising f only ever
        # SHORTENS the path -> per-step speed after the guard is <= the target, never above.
        XY_V_MAX = 0.0032   # m/step
        f_min = 1.5 * xy_len / (XY_V_MAX * appr_dur.astype(np.float64))
        xy_finish = np.minimum(np.maximum(xy_finish, f_min), 1.0)

    def _env_target(i: int, phase_idx: int, phase_step: int):
        """(pos, quat_wxyz, grip) for env i at its OWN (phase_idx, phase_step)."""
        name, dur = PHASES[phase_idx]
        if name == "approach":
            dur = int(appr_dur[i])                    # per-env duration (v3.3 speed compensation)
            alpha = (phase_step + 1) / dur
            if xy_finish is None:
                pos = home_pos[i] + alpha * (pos_b[i] - home_pos[i])
            else:
                x = min(alpha / xy_finish[i], 1.0)
                s_xy = x * x * (3.0 - 2.0 * x)          # smoothstep: zero xy-velocity only at ITS arrival
                pos = np.empty(3, np.float32)
                pos[:2] = home_pos[i, :2] + s_xy * (pos_b[i, :2] - home_pos[i, :2])
                pos[2] = home_pos[i, 2] + alpha * (pos_b[i, 2] - home_pos[i, 2])   # z: linear, never pauses
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
    rest_stress = None

    def _step(cur_pos, cur_quat, cur_grip, record_mask):
        nonlocal prev_pos, prev_quat, prev_grip

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

        # Capture the settled-rest stress baseline on the first step (soft only).
        if rest_stress is None:
            vm0 = state.get("von_mises_stress")
            rest_stress = (_stress_top10(vm0) if vm0 is not None
                           else np.zeros(num_envs, np.float32))

        # Advance phase state for envs that were active this step.
        # TODO(retry): this is where a robustness pass would check per-env
        # success/failure (e.g. at the "lift"/"hold" -> DONE boundary) and, on
        # failure, rewind phase_idx[i] to an earlier phase instead of advancing to
        # DONE — independent of every other env's state.
        phase_step[active] += 1
        # phase_idx may already be N_PHASES (done envs) — clip before indexing PHASES;
        # those entries are masked out by `active` anyway so the clipped value is unused.
        durations = np.array([PHASES[min(int(p), N_PHASES - 1)][1] for p in phase_idx])
        in_appr = (phase_idx == _APPR_IDX)
        durations[in_appr] = appr_dur[in_appr]        # per-env approach duration (v3.3)
        rolled_over = active & (phase_step >= durations)

        # Idea #1: force-based grasp firming. Check ONCE, exactly at the moment an
        # env finishes "grasp" (about to enter "firm") — read this env's just-measured
        # contact force. If it's already fine, SKIP "firm" entirely (jump straight to
        # "lift", +2 instead of +1) rather than stepping through a no-op hold and
        # relying on the trim pass to clean it up afterward — no artificial stops.
        advance = np.ones(num_envs, dtype=np.int64)   # normal: one phase forward
        leaving_grasp = rolled_over & (phase_idx == _GRASP_IDX)
        if _has_firm and np.any(leaving_grasp):
            cf = state.get("contact_force")
            if cf is not None:                        # RIGID: contact force (N). ALWAYS firm base;
                for i in np.where(leaving_grasp)[0]:  # weak grip (force < thresh) firms base+extra.
                    if cf[i] < FIRM_FORCE_THRESH_N:
                        firm_close[i] = FIRM_EXTRA_CLOSE_M + FIRM_WEAK_EXTRA_CLOSE_M
                        print(f"    [firm] env {i}: weak grip force {cf[i]:.2f}N < "
                              f"{FIRM_FORCE_THRESH_N}N -> closing {firm_close[i]*1000:.1f}mm (base+extra)")
                    # else: base firm close (never skip)
            else:                                     # SOFT: von-Mises stress rise (Pa)
                # Soft ALWAYS firms by the base amount (never skip — dropping it cost ~15% success:
                # skip-firm fails 39% vs firm 24%). A WEAK grasp (low stress rise) firms MORE.
                vm = state.get("von_mises_stress")
                if vm is not None:
                    cur = _stress_top10(vm)
                    for i in np.where(leaving_grasp)[0]:
                        rise = float(cur[i] - rest_stress[i])
                        if rise < FIRM_STRESS_THRESH_PA:
                            firm_close[i] = FIRM_EXTRA_CLOSE_M + FIRM_WEAK_EXTRA_CLOSE_M
                            print(f"    [firm] env {i}: weak stress rise {rise:.0f}Pa < "
                                  f"{FIRM_STRESS_THRESH_PA:.0f}Pa -> closing {firm_close[i]*1000:.1f}mm (base+extra)")
                        # else: firm_close stays at base -> still firms, just not extra

        phase_idx[rolled_over]  += advance[rolled_over]
        phase_step[rolled_over]  = 0

    # Success check from final object height. Rigid: get_pos; soft MPM: the particle-mean centre from
    # the last step's state (MPMEntity has no get_pos).
    _o = worker.handle.objects[0]
    obj_z   = _np(_o.get_pos())[:, 2] if hasattr(_o, "get_pos") else np.asarray(state["object_center"])[:, 2]
    success = obj_z > (grasp_pos[:, 2] + LIFT_HEIGHT * 0.5)

    # Post-process: trim long held-command runs (absolute mode only — see
    # _trim_long_holds docstring for why delta mode is excluded).
    if action_config.mode == "absolute":
        for i in range(num_envs):
            act_bufs[i], obs_bufs[i], rew_bufs[i], frame_bufs[i] = _trim_long_holds(
                act_bufs[i], obs_bufs[i], rew_bufs[i], frame_bufs[i],
                max_run=trim_max_run, keep=trim_keep)

    return obs_bufs, act_bufs, rew_bufs, success, frame_bufs


# ── Output helpers ────────────────────────────────────────────────────────────

def _make_run_dir(out_dir: Path, task_name: str) -> Path:
    """Create dated run dir matching demos/record.py naming convention."""
    base = out_dir / task_name
    base.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now().strftime("%y-%m-%d")
    for _ in range(10000):
        sfx = "".join(random.choices(string.ascii_lowercase, k=3))
        cand = base / f"{date}-{sfx}"
        if not cand.exists():
            cand.mkdir()
            return cand
    raise RuntimeError(f"could not create run dir under {base}")


def _write_shard(run_dir: Path, episodes: List[dict],
                 task: str, idx: int, rate_hz: float) -> Path:
    first = episodes[0]
    payload = {
        "meta": {
            "task": task,
            "obs_keys": sorted(first["observations"].keys()),
            "action_dim": int(first["actions"].shape[1]),
            "rate_hz": rate_hz,
        },
        "episodes": episodes,
    }
    path = run_dir / f"shard_{idx:04d}.pkl"
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)
    return path


def _merge_shards(run_dir: Path) -> Optional[Path]:
    shards = sorted(run_dir.glob("shard_*.pkl"))
    if not shards:
        return None
    all_eps: List[dict] = []
    meta: Optional[dict] = None
    for p in shards:
        with open(p, "rb") as f:
            d = pickle.load(f)
        if meta is None:
            meta = dict(d["meta"])
        all_eps.extend(d["episodes"])
    meta["n_episodes"] = len(all_eps)
    meta["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out = run_dir / "data.pkl"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"meta": meta, "episodes": all_eps}, f)
    os.replace(tmp, out)
    for p in shards:
        p.unlink()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def _extra_close_arg(args, obj_def):
    """'auto' -> squeeze scaled by object size (see --grasp-extra-close); a number passes through."""
    if str(args.grasp_extra_close).lower() != "auto":
        return float(args.grasp_extra_close)
    # BASE 2 mm, not the historical 5 mm. The 5 mm was tuned for grip reliability BEFORE we knew
    # what it did to the material: measured 2026-08-28 on the mushroom under full DR, 4.8 mm put
    # the peak von Mises at ~1.10x yield with ~0 % of demos sub-yield, while 2 mm gave median
    # 0.56x with 83 % sub-yield AND slightly BETTER demonstrator success (86 % vs 80 %). Past
    # yield the MPM saturates, so the demos were all in the regime where the gentleness objective
    # carries no information (see docs/grasp_synthesis_model.md 9c).
    smallest = float(min(obj_def.size))
    return float(np.clip(0.002 * (smallest / 0.033), 0.001, 0.003))


def _yaw_max_arg(args, obj_def):
    """'auto' -> size-scaled hard yaw bound (see --grasp-yaw-max-deg); a number passes through."""
    if args.grasp_yaw_max_deg is None:
        return None
    if str(args.grasp_yaw_max_deg).lower() != "auto":
        return float(args.grasp_yaw_max_deg)
    # LARGEST extent, not smallest: what matters is how much of the object's SILHOUETTE a finger
    # can hide. A 60 mm pasta bundle is only 25 mm thick but presents a large silhouette from most
    # angles, so sizing off the thickness gave it the TIGHTEST bound (30 deg) and cost it half its
    # yield (50 % -> ~17 %) — measured 2026-08-27. The cherry tomato is small in EVERY dimension,
    # which is why it is the one that vanishes.
    largest = float(max(obj_def.size))
    t = (largest - 0.025) / (0.065 - 0.025)
    return float(np.clip(30.0 + 45.0 * t, 30.0, 75.0))


def _area_min_arg(args, scene_dr):
    """'auto' passes through (the planner picks the floor from its own feasible pool); a number is
    mm2 and gets the scene-DR scale^2 applied, as before."""
    if str(args.grasp_area_min_mm2).lower() == "auto":
        return "auto"
    return float(args.grasp_area_min_mm2) * 1e-6 * float(scene_dr["scale"]) ** 2


def _width_max_arg(args, scene_dr):
    """'auto' passes straight through (the planner derives it from the mesh, which is already the
    DR-deformed one); a numeric value is mm and gets the scene-DR scale applied, like area_min."""
    if str(args.grasp_width_max_mm).lower() == "auto":
        return "auto"
    return float(args.grasp_width_max_mm) * 1e-3 * float(scene_dr["scale"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--experiment", required=True,
                   help="experiment name under configs/experiments/ "
                        "(e.g. single_lift_mushroom_rigid) — source of task, obs, "
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
    p.add_argument("--n-envs",     type=int, default=5,
                   help="parallel envs per batch")
    p.add_argument("--maxfevals",  type=int, default=1145,
                   help="CMA-ES function evaluations per env per batch")
    # ── v3 FEM gentleness synthesis ──
    # E / density / yield DEFAULT TO THE OBJECT'S OWN MATERIAL (resolved from the registry after the
    # experiment loads). They used to default to 3e5 / 1000 — the MUSHROOM's values — for every
    # object, so every non-mushroom collection planned grasps with the wrong material. That is not
    # cosmetic: the FEM is linear in E (sigma = E*sigma_1, F = E*F_1), so BOTH the predicted stress
    # AND the grip force were wrong per object. On the raspberry (true E 1e5) the planner believed
    # it had 3x the grip it actually had, and reported 24.8 kPa where the true figure is ~8.3 kPa.
    # Pass a value explicitly to override.
    p.add_argument("--grasp-E",           type=float, default=None,  help="object Young's modulus (Pa); default = the object's material")
    p.add_argument("--grasp-density",     type=float, default=None, help="object density (kg/m^3); default = the object's material")
    p.add_argument("--mesh-cycle", action="store_true",
                   help="Walk the DR object_mesh_pool in ORDER (round-robin), one mesh per scene "
                        "rebuild, instead of sampling it uniformly at random. Guarantees every mesh "
                        "in the pool is exercised — a uniform draw only covers the pool in "
                        "expectation, and a short smoke run rebuilds the scene only once or twice, "
                        "so it can miss most of the pool (and any broken variant in it). Use for "
                        "smoke tests and coverage checks; leave OFF for real collections, where "
                        "random sampling is the correct DR.")
    p.add_argument("--grasp-nu", default=None,
                   help="Poisson ratio for the grasp FEM. Default None keeps the HISTORICAL 0.33 "
                        "(MetricConfig's 'copper' default), which every collection before "
                        "2026-08-27 used regardless of the object. Pass 'auto' for the object's own "
                        "material nu (0.30-0.42 across our objects). Unlike E, nu CANNOT be "
                        "rescaled post-hoc, so switching it changes results and invalidates "
                        "comparisons against runs made with 0.33.")
    p.add_argument("--grasp-yield",       type=float, default=None,
                   help="object von Mises yield stress (Pa); default = the object's material. Used by "
                        "--grasp-area-min-mm2 auto to keep the selected grasp UNDER yield: maximising "
                        "contact area alone over-squeezes small soft objects (measured on the "
                        "raspberry, whose yield is 15 kPa).")
    p.add_argument("--grasp-mu",          type=float, default=0.7,  help="pad-object friction coefficient")
    p.add_argument("--grasp-accel",       type=float, default=9.81,
                   help="lift-acceleration safety margin (m/s^2): holdability needs 2*mu*grip >= m*(g+accel), "
                        "so a positive value picks a FIRMER holdable width that survives the dynamic lift "
                        "(0 = gentlest quasi-static grasp, which tends to slip during the lift)")
    p.add_argument("--table-z",           type=float, default=0.0,  help="table surface height (world z, m)")
    p.add_argument("--grasp-voxel-div",   type=int,   default=14,   help="FEM remesh resolution (keep ndof<~5k)")
    p.add_argument("--grasp-target-tets", type=int,   default=1500, help="FEM target tet count")
    p.add_argument("--grasp-n-starts",    type=int,   default=6,    help="CMA multi-start count")
    p.add_argument("--grasp-width-max-mm", type=str, default=None,
                   help="Cap the synthesized grasp WIDTH (mm, scaled by the scene DR scale like "
                        "--grasp-area-min-mm2). Default None = the gripper max, 79 mm. Set this for "
                        "ELONGATED objects: with the full range available CMA grasps along the LONG "
                        "axis, pressing the two ends together rather than closing across the body, "
                        "because an end-to-end grasp presents MORE pad contact and both the area "
                        "floor and w_press reward that. Measured on the banana (run 26-08-26-tfi, "
                        "local cross-section ~17 mm): widths 42-79 mm, median 76.6, 4 of 5 spanning "
                        "the crescent, none lifting. With --grasp-width-max-mm 40 --grasp-area-min-mm2 "
                        "10 the same 6 poses gave widths 25-40 mm, align 0.69 -> 0.87 and peak stress "
                        "16.1 kPa (under the banana's 25 kPa yield). Pass 'auto' to derive the cap "
                        "from the object itself: 2.3 x the median LOCAL cross-section perpendicular "
                        "to the long axis. That descriptor is INERT for compact objects (mushroom "
                        "65.6, strawberry 70.2, raspberry 30.4 mm -- all ~2x above any width they "
                        "plan) and BINDS only on elongated ones (banana 41.2 mm, matching the 40 mm "
                        "hand-tuned here). Note the bbox would rank the banana LARGEST/easiest, the "
                        "opposite of the truth.")
    p.add_argument("--keep-synth-failures", action="store_true",
                   help="SAVE episodes whose grasp synthesis failed and fell back to the default "
                        "top-down grasp. Off by default because those demos are actively HARMFUL: "
                        "the fallback closes to a fixed 45 mm regardless of the object, and the "
                        "old comment's assumption that such an episode 'may not lift -> simply "
                        "won't be saved' is FALSE for a wide object -- on the banana (70 mm across "
                        "the crescent) it CRUSHES the object and lifts it, so it was saved as a "
                        "success. Measured 2026-08-26: 5 of 8 saved banana episodes were crushing "
                        "fallbacks. A demonstrator that crushes teaches a policy that crushes.")
    p.add_argument("--grasp-escalate", type=int, default=2,
                   help="On synthesis failure, retry with DOUBLED search budget this many times "
                        "(n_starts and maxfevals both x2, x4, ...). Objects whose feasible set is "
                        "small need more search, not different search: the banana's holdable "
                        "fraction is 0.6%% of candidates vs the mushroom's 3.7%%, and at 4x budget "
                        "its feasibility went 2/8 -> 8/8 (measured 2026-08-26). Escalating on "
                        "DEMAND beats a per-object budget constant or a shape heuristic: it costs "
                        "nothing when the base budget already succeeds (so mushroom/strawberry/"
                        "raspberry synthesis is unchanged), and it needs no shape descriptor -- a "
                        "bbox one would be actively wrong here, since the banana's bbox reads as a "
                        "large easy object while its graspable local width is only 17 mm. 0 = off.")
    p.add_argument("--grasp-gpu",         action="store_true",
                   help="use the GPU FEM solver (default CPU, so the metric doesn't compete with the sim GPU)")
    # ── Grasp-pose DIVERSITY (ON by default — these defaults broaden the demo distribution to match v2's
    # coverage: pitch σ≈14°, continuous yaw, at ~85% collect success. Set the knobs to 0/None for the old
    # single-argmax, concentrated behaviour (v3 otherwise pins pitch~0 + snaps yaw to a few gentle axes).
    p.add_argument("--grasp-diversity-tol", type=float, default=0.3,
                   help="sample among feasible grasps within this FRACTION of the best gentleness score "
                        "(0=off/argmax; 0.3 default = accept up to 30%% worse score -> diverse but still gentle)")
    p.add_argument("--grasp-jitter-deg",  type=float, default=20.0,
                   help="max +/- random perturbation (deg) on the selected grasp's roll/pitch/yaw, re-verified "
                        "to still hold within the diversity tolerance (needs --grasp-diversity-tol>0 for headroom)")
    p.add_argument("--grasp-jitter-pos",  type=float, default=0.003,
                   help="max +/- random position perturbation (m) applied with --grasp-jitter-deg")
    p.add_argument("--grasp-align",       type=float, default=2000.0,
                   help="alignment weight w_align (metric default 3e4; 2000 here). LOWER lets TILTED grasps "
                        "stay near-optimal -> the diversity sampler/jitter can broaden PITCH (w_align=3e4 pins "
                        "pitch~0). Pass 30000 to restore the strict flush-grasp metric.")
    p.add_argument("--grasp-pitch-seed-deg", type=float, default=25.0,
                   help="jitter the CMA multi-start PITCH seed by +/- this (deg). Every start otherwise "
                        "seeds pitch 0, so even with a low --grasp-align CMA rarely explores tilt; seeding "
                        "tilted starts broadens the demo pitch toward v2 (complement of the yaw seed-smear)")
    p.add_argument("--scene-dr-every", type=int, default=1,
                   help="re-randomize object SIZE+SHAPE every N batches by rebuilding the worker "
                        "(needs shape/scale fields in the experiment DR config; 0 = off, nominal "
                        "geometry). Geometry is shared across a batch's envs (batched build).")
    p.add_argument("--settle",            type=int,   default=None,
                   help="override settle_steps from task config")
    p.add_argument("--settle-max",        type=int,   default=None,
                   help="override settle_max_steps from task config")
    p.add_argument("--settle-vel-thresh", type=float, default=None,
                   help="override settle_vel_thresh from task config (m/s)")
    p.add_argument("--seed",       type=int, default=0,
                   help="RNG seed for pose DR")
    p.add_argument("--keep-failures", action="store_true",
                   help="also save episodes where the grasp failed (default: success only)")
    p.add_argument("--record-video", nargs="?", type=int, const=10**9, default=0,
                   help="record per-episode mp4 videos + grasp-pose PNGs to <out-dir>/videos/ (slower). "
                        "Bare `--record-video` = ALL episodes; `--record-video N` = only the FIRST N saved "
                        "episodes (rendering stops after N -> no extra cost/disk on a long run). Off by default.")
    p.add_argument("--n-home-to-pre", type=int, default=N_HOME_TO_PRE,
                   help="home -> pre-grasp interpolation steps (the 'approach' phase length); "
                        f"default matches the module constant ({N_HOME_TO_PRE}). Recorded into "
                        "config.yaml's control.n_home_to_pre so a non-default run is clearly marked.")
    p.add_argument("--n-grasp", type=int, default=N_GRASP,
                   help=f"gripper-close steps (the 'grasp' phase length); default {N_GRASP}. A shorter "
                        "close reaches the target width sooner (less dwell before the lift).")
    p.add_argument("--n-settle", type=int, default=N_SETTLE,
                   help=f"hold-at-grasp-pose steps before closing; default {N_SETTLE}. Item-2 finding: "
                        "real teleop hovers ~6 steps at the grasp pose before closing (scripted ~2).")
    p.add_argument("--approach-xy-finish", type=float, nargs=2, default=None,
                   metavar=("F_LO", "F_HI"),
                   help="v3.2 real-style approach: per-env xy-progress finish fraction sampled "
                        "uniform [F_LO, F_HI] (real median 0.60; recipe 0.45 0.75). xy converges "
                        "early (smoothstep) while z descends linearly — continuous, no stopping. "
                        "Default None = straight-line approach (bit-identical to v3/v3.1).")
    p.add_argument("--grasp-area-min-mm2", type=str, default="0",
                   help="v3.3 anti-stem/pinch HARD floor: reject grasps whose WORST pad grips less "
                        "than this many mm^2 of object surface (v4 anti-pinch floor in the FEM "
                        "planner; auto-scaled by scene scale^2). Measured: stem grasp 8 mm^2 vs "
                        "cap grasp 49 mm^2; suggest 15. 0 (default) = off.")
    p.add_argument("--grasp-w-press", type=float, default=0.0,
                   help="v3.3 soft worst-pad PRESSURE penalty (score -= w_press * grip/min_pad_area) "
                        "— the smooth gradient companion of the area floor (stem grasp pressure "
                        "114 kPa vs cap 37 kPa). Suggest ~0.05 (score is in Pa; pressure ~1e5). "
                        "0 (default) = off.")
    p.add_argument("--approach-speed", type=float, default=None,
                   help="v3.3 speed compensation: approach duration per env = distance/speed "
                        "(m/step; real teleop ~0.0024), clipped [40,130] steps. Fixes the fixed-"
                        "duration artifact where speed is proportional to spawn distance (sim "
                        "corr 0.91 vs real 0.29). Default None = fixed --n-home-to-pre steps.")
    p.add_argument("--held-run-max", type=int, default=HELD_RUN_MAX,
                   help=f"trim held-command runs LONGER than this (default {HELD_RUN_MAX}); "
                        "v3.2 recipe 12 (with --held-run-keep 10) preserves ~10 stop frames at "
                        "the episode end — the 'stop at lift height' supervision the 4-frame "
                        "floor was starving (fleli hold deficit).")
    p.add_argument("--held-run-keep", type=int, default=HELD_RUN_KEEP,
                   help=f"frames kept from a trimmed run (default {HELD_RUN_KEEP}); v3.2 recipe 10.")
    p.add_argument("--cam-azimuth-max-deg", type=float, default=None,
                   help="item-5 occlusion bound: shaped penalty on grasp yaw beyond this azimuth from "
                        "the camera-perpendicular direction (deg; None = off). Passed to the FEM "
                        "planner with the task camera's position; also centres the CMA seed fan. "
                        "45 validated in the v5c profile (fully-hidden episodes 24%% -> 4%%).")
    p.add_argument("--n-lift", type=int, default=N_LIFT,
                   help=f"lift-phase steps; default {N_LIFT}.")
    p.add_argument("--n-firm", type=int, default=N_FIRM,
                   help=f"'firm' phase steps (post-grasp extra squeeze idea #1); default {N_FIRM}. "
                        "0 = NO firm phase at all — the grasp goes straight to lift at width_cls "
                        "(matches the pre-firm v1 collector, e.g. the cho dataset).")
    p.add_argument("--grasp-medial-seeds", action="store_true",
                   help="seed the CMA search along the object's MEDIAL AXIS (deep-interior points, "
                        "spread by farthest-point, each closing perpendicular to the local tangent "
                        "and sized by the LOCAL cross-section) instead of putting every start at the "
                        "COM with a global-extent width. Required for elongated/non-convex objects "
                        "(a banana's COM sits in a thin band: pads there bury or miss, and since all "
                        "starts share that xy, more starts/evals cannot help). Off by default -> "
                        "convex objects keep bit-identical behaviour.")
    p.add_argument("--grasp-yaw-max-deg", type=str, default=None,
                   help="bound the TOOL yaw about the gripper's HOME orientation (yaw 0), in deg, "
                        "at CMA time — box + seed clip, folded for the parallel-jaw 180-deg "
                        "symmetry. cam_azimuth_max_deg bounds the fan about the CAMERA, which still "
                        "permitted ~90 deg home-frame yaw and real-rig occlusion. Suggest 55. "
                        "None (default) = unchanged.")
    p.add_argument("--grasp-w-peak", type=float, default=None,
                   help="peak-aware stress weight: score -= w_peak * E * UNMASKED p98 stress. The "
                        "masked top10 objective HIDES contact spikes (corner/edge grasps score low "
                        "bulk stress while spiking locally, sect 11.7); metric default 0.3, but the "
                        "legacy collector path forwards 0 — pass 0.3 to opt in. None (default) = "
                        "legacy off.")
    p.add_argument("--grasp-w-tilt", type=float, default=None,
                   help="approach-TILT penalty: score -= w_tilt * (1 - cos(approach axis, straight down)). "
                        "Targets tilted-gripper edge contact that align CANNOT see (align measures the "
                        "closing axis vs surface normal; the approach pitch is orthogonal to it) and the "
                        "rounded-pad FEM underprices. ~1.5e5 makes a 15 deg tilt cost ~5 kPa-equivalents. "
                        "None (default) = legacy off.")
    p.add_argument("--grasp-w-area", type=float, default=None,
                   help="whole-grasp contact-area REWARD (score += w_area * area) — continuous "
                        "flush-contact promotion beyond the --grasp-area-min-mm2 floor. None "
                        "(default) = legacy off.")
    p.add_argument("--grasp-extra-close", type=str, default="0.0",
                   help="squeeze FURTHER IN than the synthesized width by this many meters (tighter grip) "
                        "for EVERY grasp — e.g. 0.005 = close 5mm tighter. 0 (default) = no change. Use to "
                        "make grasps firmer (a too-gentle grip -> premature lift / slip before secured). "
                        "Pass 'auto' to scale it with the object: a FIXED squeeze is 15%% of the 33 mm "
                        "mushroom it was tuned on but 24%% of a 21 mm cherry tomato and 34%% of a 15 mm "
                        "raspberry, i.e. the same knob over-squeezes small objects. auto = "
                        "5 mm * (smallest extent / 33 mm), clipped to [2, 6] mm, so the mushroom is "
                        "unchanged (4.8 mm) and small objects get proportionally less. NOTE the "
                        "separate FIRM_EXTRA_CLOSE_M (2.5 mm) is still a constant and is NOT scaled.")
    args = p.parse_args()

    # Approach-phase length is configurable so a dataset can be collected with a shorter/longer
    # home->pre-grasp interpolation to study its effect on downstream policy learning; PHASES is
    # module-level (execute_and_collect reads the globals directly), so rebuild it from the CLI value.
    global PHASES, N_PHASES, _GRASP_IDX
    PHASES = [
        ("approach", args.n_home_to_pre),
        ("settle",   args.n_settle),
        ("grasp",    args.n_grasp),
    ]
    if args.n_firm > 0:                    # --n-firm 0 drops the firm phase entirely (cho/v1 behaviour)
        PHASES.append(("firm", args.n_firm))
    PHASES += [
        ("lift",     args.n_lift),
        ("hold",     N_HOLD),
    ]
    N_PHASES   = len(PHASES)
    _GRASP_IDX = [name for name, _ in PHASES].index("grasp")

    # ── Load everything from the experiment config (same as training / eval) ──
    exp        = Experiment.load(args.experiment)
    task       = SingleLiftTask(exp.task_cfg)
    # Resolve the grasp material from the OBJECT (see --grasp-E) unless explicitly overridden.
    from gentle_manip.assets.registry import get_object_def as _god
    _mat = _god(exp.task_cfg["object_name"]).material
    if args.grasp_E is None:       args.grasp_E = float(_mat.youngs_modulus)
    if args.grasp_density is None: args.grasp_density = float(_mat.density)
    if args.grasp_yield is None:   args.grasp_yield = float(_mat.von_mises_yield_stress)
    _fem_nu = float(_mat.poisson_ratio) if str(args.grasp_nu).lower() == "auto" else (
        None if args.grasp_nu is None else float(args.grasp_nu))
    print(f"  grasp material ({exp.task_cfg['object_name']}): E={args.grasp_E:.3g} Pa  "
          f"rho={args.grasp_density:.0f}  yield={args.grasp_yield:.3g} Pa")
    args.grasp_extra_close = _extra_close_arg(args, _god(exp.task_cfg["object_name"]))
    print(f"  grasp extra-close: {1000*args.grasp_extra_close:.1f} mm")
    _yaw = _yaw_max_arg(args, _god(exp.task_cfg["object_name"]))
    if _yaw is not None:
        print(f"  grasp yaw bound: {_yaw:.1f} deg (largest extent "
              f"{1000*max(_god(exp.task_cfg['object_name']).size):.1f} mm)")
    spec       = task.scene_spec
    # item-5 occlusion bound: the task camera's world position, only when the knob is on
    # (None keeps the planner call byte-identical to the baseline recipe).
    cam_pos = (np.asarray(spec.cameras[0].pos, float)
               if args.cam_azimuth_max_deg is not None else None)
    obs_config = exp.collection_obs()
    priv_cfg   = obs_config.privileged        # sim-only state-teacher fields (None if not requested)
    action_config = exp.action_config
    dr_cfg     = DRConfig.from_dict(exp.dr)
    task_name  = args.task_name or exp._raw.get("task", args.experiment)
    rate_hz    = 1.0 / spec.sim_dt

    # Settle params: task config → CLI override.
    settle_steps     = args.settle           or int(exp.task_cfg.get("settle_steps",     30))
    settle_max_steps = args.settle_max       or int(exp.task_cfg.get("settle_max_steps", 200))
    settle_vel_thresh = args.settle_vel_thresh or float(exp.task_cfg.get("settle_vel_thresh", 0.002))

    perception = PerceptionPipeline(obs_config)

    collection_config = {
        "task_name":   task_name,
        "description": args.description,
        "source":      "cmaes_synth",
        "git_commit":  _git_commit(),
        "experiment":  args.experiment,
        "control":     {"n_envs": args.n_envs, "maxfevals": args.maxfevals,
                        "n_episodes": args.n_episodes, "scene_dr_every": args.scene_dr_every,
                        "seed": args.seed, "n_home_to_pre": args.n_home_to_pre,
                        "n_grasp": args.n_grasp, "n_lift": args.n_lift, "n_firm": args.n_firm,
                        "n_settle": args.n_settle,
                        "cam_azimuth_max_deg": args.cam_azimuth_max_deg,
                        "approach_xy_finish": list(args.approach_xy_finish) if args.approach_xy_finish else None,
                        "approach_speed": args.approach_speed,
                        "held_run_max": args.held_run_max, "held_run_keep": args.held_run_keep,
                        "grasp_jitter_deg": args.grasp_jitter_deg,
                        "grasp_area_min_mm2": args.grasp_area_min_mm2,
                        "grasp_medial_seeds": bool(args.grasp_medial_seeds),
                        "grasp_escalate": int(args.grasp_escalate),
                        "grasp_width_max_mm": args.grasp_width_max_mm,
                        "grasp_yaw_max_deg": args.grasp_yaw_max_deg,
                        "grasp_yaw_max_deg_resolved": _yaw,
                        "grasp_nu": _fem_nu,
                        "grasp_w_press": args.grasp_w_press,
                        "grasp_extra_close": args.grasp_extra_close},
        "dr": exp.dr,
    }

    print(f"\n=== collect_demos_synth  experiment={args.experiment}"
          f" — target {args.n_episodes} episodes, {args.n_envs} envs/batch")
    if args.n_home_to_pre != N_HOME_TO_PRE:
        print(f"  *** NON-DEFAULT n_home_to_pre={args.n_home_to_pre} "
              f"(module default is {N_HOME_TO_PRE}) — recorded in config.yaml ***")

    rng = np.random.default_rng(args.seed)   # DR RNG (pose + scene) — must precede the first build
    # Separate stream (distinct offset so it never shares draws with `rng` above, keeping
    # DR reproducibility untouched) — makes CMA-ES's own search reproducible from --seed too
    # (previously every _synth_worker call used run_cmaes's hardcoded default seed=2567, so
    # ALL envs/batches shared the identical internal search sequence regardless of --seed).
    cma_seed_rng = np.random.default_rng(args.seed + 1_000_000)
    approach_rng = np.random.default_rng(args.seed + 2_000_000)   # v3.2 per-env xy-finish draws

    # ── Build scene + worker (with per-scene SIZE+SHAPE DR) ──
    # Scene DR re-randomizes object geometry by REBUILDING the worker every N batches (GenesisWorker
    # has no in-process geometry re-randomize; a fresh build is the only path). Verified memory-stable
    # across rebuilds (gs.destroy reclaims each build). Geometry is shared across a batch's envs.
    nominal_spec   = spec
    do_scene_dr    = args.scene_dr_every > 0 and dr_cfg.has_scene_dr()
    deform_dir     = tempfile.mkdtemp(prefix="gm_synth_deform_") if do_scene_dr else None

    _mesh_cycle = [0] if args.mesh_cycle else None    # round-robin cursor (see --mesh-cycle);
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
        w = GenesisWorker(spec_dr, num_envs=args.n_envs, show_viewer=False,
                          settle_steps=settle_steps, settle_max_steps=settle_max_steps,
                          settle_vel_thresh=settle_vel_thresh, render_obs_cameras=True,
                          coup_friction=float(sdr.get("coup_friction", 4.0)))
        return w, sdr, (w.handle.spec.objects[0].mesh_path or MUSHROOM_MESH)

    worker, scene_dr, actual_mesh = _make_worker()
    if do_scene_dr:
        print(f"  scene DR ON (every {args.scene_dr_every} batch(es)) — deformed meshes → {deform_dir}")

    # ── Everything below is CPU-only (trimesh BVH + scipy CMA-ES) ──
    # Mesh path and finger geometry are gathered from Genesis, then handed off to
    # subprocess workers that each build their own SDF — no CUDA involved.
    left_pts  = sample_finger_surface(LEFT_FINGER,  n=300)
    right_pts = sample_finger_surface(RIGHT_FINGER, n=300)

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
                        "mesh_variant",
                        "twist_deg", "taper", "rbf", "axis_scale", "axis_scale_ax",
                        "mat_E", "mat_nu", "mat_rho", "coup_friction",
                        "stress_Pa", "grip_N", "align", "pressure_Pa", "min_pad_mm2", "width_mm", "tilt_deg"])

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
                von_mises=init_state.get("von_mises_stress"), yield_stress=args.grasp_yield,
                contact_force=init_state.get("contact_force")))

        # ── Per-env FEM gentleness grasp synthesis (v3) ──
        # Build the FEM ElasticObject ONCE for this batch's ACTUAL (DR shape+size) mesh — all envs share
        # it (scene-DR varies per relaunch, not per sub-env), so the expensive factorization is reused;
        # rebuild only when actual_mesh changes (a scene-DR relaunch). Then plan per-env settled pose.
        if actual_mesh != fem_mesh:
            fem_obj, fem_pad_geo, fem_meta = fg.build_grasp_fem(
                actual_mesh, voxel_div=args.grasp_voxel_div, target_tets=args.grasp_target_tets,
                use_gpu=args.grasp_gpu, nu=_fem_nu)
            fem_mesh = actual_mesh
            print(f"  FEM: {fem_meta['tets']} tets, ndof={fem_meta['ndof']}, gpu={fem_meta['gpu']}")
        all_best_x = []
        synth_failed: set[int] = set()      # envs running the fallback grasp (see --keep-synth-failures)
        all_grasp  = []                                          # per-env synthesis dict (for the grasp-pose viz)
        for i in range(n):
            cma_seed = int(cma_seed_rng.integers(1, 2**31 - 1))
            r = fg.synthesize_grasp(fem_obj, fem_pad_geo, obj_pos_all[i], obj_quat_all[i],
                                    E=args.grasp_E, density=args.grasp_density, mu=args.grasp_mu,
                                    table_z=args.table_z, maxfevals=args.maxfevals,
                                    n_starts=args.grasp_n_starts, seed=cma_seed, accel=args.grasp_accel,
                                    diversity_tol=args.grasp_diversity_tol, jitter_deg=args.grasp_jitter_deg,
                                    jitter_pos=args.grasp_jitter_pos, w_align=args.grasp_align,
                                    pitch_seed_deg=args.grasp_pitch_seed_deg,
                                    cam_pos=cam_pos, cam_azimuth_max_deg=args.cam_azimuth_max_deg,
                                    area_min=_area_min_arg(args, scene_dr),
                                    yield_stress=args.grasp_yield,
                                    w_press=(args.grasp_w_press or None),
                                    medial_seeds=int(args.grasp_medial_seeds),
                                    **({"width_max": _width_max_arg(args, scene_dr)} if args.grasp_width_max_mm else {}),
                                    **({"w_peak": args.grasp_w_peak} if args.grasp_w_peak is not None else {}),
                                    **({"w_area": args.grasp_w_area} if args.grasp_w_area is not None else {}),
                                    **({"w_tilt": args.grasp_w_tilt} if args.grasp_w_tilt is not None else {}),
                                    **({"yaw_max_deg": _yaw} if _yaw is not None else {}))
            if r.get("x") is None or r.get("stress_top10") is None:   # diversity found no feasible grasp;
                print(f"  Env {i}: no feasible diverse grasp -> retry WITHOUT diversity")  # retry reliably
                r = fg.synthesize_grasp(fem_obj, fem_pad_geo, obj_pos_all[i], obj_quat_all[i],
                                        E=args.grasp_E, density=args.grasp_density, mu=args.grasp_mu,
                                        table_z=args.table_z, maxfevals=args.maxfevals,
                                        n_starts=args.grasp_n_starts, seed=cma_seed + 7, accel=args.grasp_accel,
                                        cam_pos=cam_pos, cam_azimuth_max_deg=args.cam_azimuth_max_deg,
                                        area_min=_area_min_arg(args, scene_dr),
                                        yield_stress=args.grasp_yield,
                                        w_press=(args.grasp_w_press or None),
                                        medial_seeds=int(args.grasp_medial_seeds),
                                        **({"width_max": _width_max_arg(args, scene_dr)} if args.grasp_width_max_mm else {}),
                                        **({"w_peak": args.grasp_w_peak} if args.grasp_w_peak is not None else {}),
                                        **({"w_area": args.grasp_w_area} if args.grasp_w_area is not None else {}),
                                    **({"w_tilt": args.grasp_w_tilt} if args.grasp_w_tilt is not None else {}),
                                    **({"yaw_max_deg": _yaw} if _yaw is not None else {}))
            # BUDGET ESCALATION: the retry above only drops diversity; if synthesis is still empty
            # the feasible set is simply too small for this budget to land in. Double it and look
            # again (see --grasp-escalate). Runs only on the failure path, so a run whose grasps all
            # succeed at the base budget is bit-identical to before this existed.
            for _esc in range(1, int(args.grasp_escalate) + 1):
                if r.get("x") is not None and r.get("stress_top10") is not None:
                    break
                mult = 2 ** _esc
                print(f"  Env {i}: still no feasible grasp -> escalating budget x{mult} "
                      f"({args.grasp_n_starts * mult} starts, {args.maxfevals * mult} fevals)")
                r = fg.synthesize_grasp(fem_obj, fem_pad_geo, obj_pos_all[i], obj_quat_all[i],
                                        E=args.grasp_E, density=args.grasp_density, mu=args.grasp_mu,
                                        table_z=args.table_z, maxfevals=args.maxfevals * mult,
                                        n_starts=args.grasp_n_starts * mult,
                                        seed=cma_seed + 13 * _esc, accel=args.grasp_accel,
                                        cam_pos=cam_pos, cam_azimuth_max_deg=args.cam_azimuth_max_deg,
                                        area_min=_area_min_arg(args, scene_dr),
                                        yield_stress=args.grasp_yield,
                                        w_press=(args.grasp_w_press or None),
                                        medial_seeds=int(args.grasp_medial_seeds),
                                        **({"width_max": _width_max_arg(args, scene_dr)} if args.grasp_width_max_mm else {}),
                                        **({"w_peak": args.grasp_w_peak} if args.grasp_w_peak is not None else {}),
                                        **({"w_area": args.grasp_w_area} if args.grasp_w_area is not None else {}),
                                        **({"w_tilt": args.grasp_w_tilt} if args.grasp_w_tilt is not None else {}),
                                        **({"yaw_max_deg": _yaw} if _yaw is not None else {}))
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
                    str(Path(vid_dir) / f"{stem}_grasp.png"), E=args.grasp_E,
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
                extra_close=args.grasp_extra_close,
                approach_xy_finish=args.approach_xy_finish, approach_rng=approach_rng,
            approach_speed=args.approach_speed,
                trim_max_run=args.held_run_max, trim_keep=args.held_run_keep,
                yield_stress=args.grasp_yield)
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
        eul_deg = np.degrees(object_euler) if object_euler is not None else np.zeros((n, 3))
        for i in range(n):
            g = all_grasp[i]
            roll, pitch, yaw = eul_deg[i]
            flipped = int(abs(roll) > 140 or abs(pitch) > 140)                # a big-flip sample
            ho = home_offset[i] if home_offset is not None else (0.0, 0.0, 0.0)
            odxy = object_dxy[i] if object_dxy is not None else (0.0, 0.0)
            dr_writer.writerow([batch_idx, i, int(bool(success[i])),
                                round(float(odxy[0]), 5), round(float(odxy[1]), 5),
                                round(float(roll), 1), round(float(pitch), 1), round(float(yaw), 1), flipped,
                                round(float(ho[0]), 5), round(float(ho[1]), 5), round(float(ho[2]), 5),
                                round(float(scene_dr.get("scale", 1.0)), 4),
                                round(float(scene_dr.get("bend_deg", 0.0)), 2),
                                scene_dr.get("mesh_variant", ""),
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
                                round(float(g.get("tilt_deg") or 0), 1)])
        dr_csv.flush()

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
                if not args.keep_failures:
                    continue

            if i in synth_failed and not args.keep_synth_failures:
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
