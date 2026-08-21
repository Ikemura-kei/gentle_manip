"""Autonomous sim demo collection via FEM GENTLENESS grasp synthesis (v4).

Fork of collect_demos_synth_v3.py (v1/v2/v3 remain FROZEN baselines). Same pipeline — reset with
pose DR -> per-env synthesis -> scripted execution -> record (obs, action, reward) in
demos/record.py format — with four changes, all aimed at producing demonstrations a policy can
actually imitate:

1. TRAJECTORY: a pre-grasp standoff decomposition instead of one lerp+slerp straight from home to
   the grasp pose. `approach_xy` travels at the HOME (top-down) orientation, `align` rotates in
   place where nothing can collide, `descend` moves in a straight line along the grasp's own
   approach axis (collision-free by construction, and swept-checked against the object SDF).
2. MINIMUM JERK: every phase is time-scaled by 10a^3-15a^4+6a^5 (Flash & Hogan 1985) instead of
   linearly, giving a bell-shaped velocity profile and C2 phase junctions. Because the recorded
   action is derived from consecutive TARGETS, this smooths the action stream the policy trains on.
3. GRIPPER PRESHAPE: approach at ~1.4x the grasp width rather than fully open (human reach-to-grasp
   aperture), which also cuts swept volume and camera occlusion during the descent.
4. OBJECTIVE: the v4 geometry priors are available (w_com anti-stem, w_tilt anti-side-grasp, w_occ
   anti-occlusion, area_min anti-pinch, tightened roll bounds), all default-off so this collector
   reproduces v3's grasps until they are explicitly enabled.

Structural cleanups over v3: the mutable module-level PHASES globals are gone (a PhaseSchedule is
built in main() and passed down), the trajectory lives in the shared `grasp_traj.GraspTrajectory`
(so the benchmark and the collector can no longer drift apart), and the mushroom-specific
OBJ_SIZE / MUSHROOM_MESH / LIFT_HEIGHT constants are resolved from the experiment.

Usage:
    uv run --project envs/sim python grasp_synthesis/collect_demos_synth_v4.py \\
        --experiment single_lift_mushroom_soft_abs_action \\
        --n-episodes 500 --n-envs 8 [--grasp-gpu]
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
from smgrasp import finger_grasp as fg  # noqa: E402  (FEM gentleness synthesis)
from smgrasp import finger_viz          # noqa: E402  (grasp-pose viz paired with each execution video)
from grasp_profiles import GRASP_PROFILES  # noqa: E402  (shared with the benchmark)
from grasp_traj import (bound_scaled_schedule,
                        GraspTrajectory, PhaseSchedule,  # noqa: E402  (shared with the benchmark)
                        SCHEDULE_V3, SCHEDULE_V4, SCHEDULE_V4_BLEND)
from gentle_manip.actions.action_config import ActionConfig
from gentle_manip.experiment import Experiment
from gentle_manip.tasks.single_lift import SingleLiftTask
from gentle_manip.envs.genesis_worker import GenesisWorker
from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.perception.pipeline import PerceptionPipeline
from gentle_manip.perception.obs_config import CONTACT_FORCE_THRESH_N
from gentle_manip.domain_randomization.dr_config import DRConfig


# ── Constants (keep in sync with run_grasp_synth.py) ─────────────────────────

N_HOME_TO_PRE = 98          # v3 schedule: home → grasp pose interpolation steps
N_SETTLE      = 1           # hold at grasp pose before closing
N_GRASP       = 37           # gripper close steps
N_LIFT        = 70          # lift steps (66 -> 70: dz needs 68 steps to fit the 5.5mm/step
                            # rate bound at min-jerk peak = 1.875x mean over the 200mm lift)
N_HOLD        = 12           # hold at lift height (success eval window)
LIFT_HEIGHT   = 0.2         # metres above grasp position (default; --lift-height overrides)
# v4 schedule: the approach budget is split across travel / rotate-in-place / straight descent.
# They sum to N_HOME_TO_PRE so the total episode length is unchanged from v3 by default.
N_APPROACH_XY = 55
N_ALIGN       = 25
N_DESCEND     = 18
STANDOFF_M    = 0.05        # pre-grasp standoff along the approach axis
PRESHAPE_FACT = 1.4         # approach aperture = this x grasp width (0 disables -> fully open)
OBJ_SIZE      = np.array([0.05, 0.05, 0.04])   # fallback AABB half-size (SDF search box; v1/v2 path)

DEFAULT_MESH = str(ROOT / "gentle_manip/assets/objects/mushroom.obj")
MUSHROOM_MESH = DEFAULT_MESH               # back-compat alias (eval_grasp_synth imports the v3 name)
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

def _apply_scene_dr(nominal_spec, dr_cfg, rng, deform_dir):
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
    if not shp:
        return nominal_spec, {"scale": nominal_scale, "bend_deg": 0.0}

    nominal_mesh = o.mesh_path or get_object_def(o.name).mesh_path
    mesh = trimesh.load(str(nominal_mesh), process=False, force="mesh")
    shape = {k: shp[k] for k in ("bend", "twist", "taper", "rbf", "axis_scale", "axis_scale_ax")
             if k in shp}
    if shape:
        mesh = mesh_deform.deform_mesh(mesh, shape, rng)     # bend/twist/taper/axis_scale (radians)
    mesh.apply_scale(nominal_scale * float(shp.get("scale", 1.0)))   # bake uniform scale in
    dst = Path(deform_dir) / f"{Path(nominal_mesh).stem}_dr_{rng.integers(1_000_000):06d}.obj"
    mesh.export(str(dst))

    new_obj  = dataclasses.replace(o, mesh_path=str(dst), scale=1.0)
    new_spec = dataclasses.replace(nominal_spec, objects=[new_obj, *nominal_spec.objects[1:]])
    scene_dr = {"scale": float(shp.get("scale", 1.0)),
                "bend_deg": float(np.rad2deg(shp.get("bend", 0.0)))}
    return new_spec, scene_dr


# ── Privileged obs (sim-only state-teacher fields) ────────────────────────────

def _privileged_obs_batch(object_center, object_quat, dr_vec, priv_cfg, contact_force=None) -> dict:
    """Sim-only privileged fields from raw worker state — mirrors
    PolicyEnv._privileged_obs exactly, but sourced from GenesisWorker state
    (object_center + object_quat + contact_force) since this collector bypasses PolicyEnv.

    object_center: (N, 3); object_quat: (N, 4) wxyz; dr_vec: (2,) [scale, bend_deg];
    contact_force: (N,) — rigid: get_contacts Newtons; soft: MPM->finger coupling force.
    Returns a dict of (N, ...) arrays for whichever priv fields the config enables.
    """
    out = {}
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

# NOTE: unlike v3 these are NOT module-level mutable globals rebuilt inside main(). The schedule is
# constructed there and threaded through explicitly, so the collector and the benchmark can never
# silently disagree about the phase layout (which is precisely how they drifted apart before).


def build_schedule(args) -> PhaseSchedule:
    """Phase list for this run: the v4 standoff decomposition, or v3's single approach."""
    if args.traj == "v3":
        phases = [("approach", args.n_home_to_pre)]
    elif args.traj == "v4split":
        phases = [("approach_xy", args.n_approach_xy),
                  ("align", args.n_align),
                  ("descend", args.n_descend)]
    else:                                   # "v4" (default): ONE blended Bezier reach
        phases = [("reach", args.n_home_to_pre)]
    phases += [("settle", args.n_settle), ("grasp", args.n_grasp)]
    if args.n_firm > 0:
        phases.append(("firm", args.n_firm))
    phases += [("lift", args.n_lift), ("hold", args.n_hold)]
    return PhaseSchedule(tuple(phases))

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
# ── Robustness idea #2: lift-failure detection + regrasp (--retry-max) ────────
# Checked ONCE per attempt per env, partway into the lift: the EE has provably risen (its target is
# a deterministic function of the phase step), so if the OBJECT has not risen with it, the grasp
# slipped. Recovery re-seeds the approach from the current pose (traj.begin_retry) and rewinds the
# env's phase index to 0 — an in-place regrasp against the SAME synthesized pose, on the assumption
# the object has not moved far. No CMA replan: that costs real wall-clock per env and the object is
# usually still where it was.
#
# INDEPENDENT of the shelf. This is the fallback path: if the shelf trajectory turns out too hard
# for BC to clone, v4 + retry is still a better dataset than v4 alone, at no cost in trajectory
# difficulty.
# Measured (grasp_synthesis/retry_window_probe.py, forced slip at each fraction, 5 envs):
#   frac  object risen  recovered
#   0.10       0.1 mm      0/5      -- releasing before the lift starts; the regrasp misses
#   0.15       2.0 mm      5/5
#   0.20       8.3 mm      5/5
#   0.25      15.6 mm      5/5      <- default: a genuine lift attempt, still a short drop
#   0.30      25.3 mm      5/5
#   0.45      81.4 mm      0/3      -- the object bounces out from under the planned pose
# 0.45 was the original guess and is outside the working window on BOTH counts: too late to
# recover, and it wastes most of a lift before deciding. 0.25 is in the middle of the plateau.
RETRY_CHECK_FRAC   = 0.25      # fraction into `lift` at which the slip check fires
RETRY_MIN_RISE_M   = 0.008     # object must have risen at least this much by then, else it slipped
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
    extra_close: float = 0.0,      # squeeze this many meters TIGHTER than the synthesized width (all grasps)
    schedule: PhaseSchedule = SCHEDULE_V3,   # phase layout (v4 adds approach_xy/align/descend)
    lift_height: float = LIFT_HEIGHT,
    standoff: float = STANDOFF_M,            # v4: pre-grasp offset along the approach axis
    use_minjerk: bool = True,                # v4: min-jerk time scaling in every phase
    preshape_factor: float = PRESHAPE_FACT,  # v4: approach aperture multiple (0 = fully open)
    shelf_kw: dict = None,                   # v4.1: shelf-lift geometry (empty/None = off)
    retry_max: int = 0,                      # v4.1: regrasp attempts on a detected slip (0 = off)
    width_open=None,                         # v4.1: scalar or per-env initial aperture (m)
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
        success:     (N,) bool — True if final object z > grasp_z + 0.5*lift_height
        frame_bufs:  list[N] of (H,W,3) uint8 frame lists; empty lists if record_video=False
    """
    scales = (np.asarray(action_config.scales, dtype=np.float64)
              if action_config.mode != "absolute" else None)
    num_envs = worker.num_envs

    home_pos  = np.tile(worker.robot.home_pos[None].astype(np.float32),  (num_envs, 1))
    home_quat = np.tile(worker.robot.home_quat[None].astype(np.float32), (num_envs, 1))

    # The shared trajectory engine — same object the benchmark drives, so collection and evaluation
    # cannot diverge. It owns the standoff geometry, min-jerk time scaling, preshape, and the
    # stateful grip_target that the "firm" phase writes and lift/hold read back.
    # ONE kwargs dict for both the duration scaling and the real construction — passing them
    # separately is how the two would drift apart.
    traj_kw = dict(lift_height=lift_height, extra_close=extra_close,
                   firm_close=FIRM_EXTRA_CLOSE_M, standoff=standoff,
                   use_minjerk=use_minjerk, preshape_factor=preshape_factor,
                   **({} if width_open is None else {"width_open": width_open}),
                   **(shelf_kw or {}))
    _rate_lim = (np.asarray(action_config.rate_limit, np.float64)
                 if getattr(action_config, "rate_limit", None) is not None
                 and action_config.mode == "absolute" else None)
    if _rate_lim is not None:
        # Lengthen phases so the min-jerk trajectory FITS the rate bound by construction (the
        # per-step clamp below then never engages and the executed motion keeps its bell profile).
        scaled = bound_scaled_schedule(schedule, all_best_x, home_pos, home_quat, _rate_lim,
                                       **traj_kw)
        if scaled.phases != schedule.phases:
            chg = [f"{scaled.name(i)} {schedule.duration(i)}->{scaled.duration(i)}"
                   for i in range(schedule.n_phases)
                   if scaled.duration(i) != schedule.duration(i)]
            print(f"    [rate-limit] schedule lengthened to fit bounds: {', '.join(chg)}")
        schedule = scaled
    traj = GraspTrajectory(schedule, all_best_x, home_pos, home_quat, **traj_kw)
    pos_b, quat_b = traj.pos_b, traj.quat_b
    grasp_pos = pos_b.copy()
    _has_firm = schedule.has("firm")       # --n-firm 0 drops the phase: skip the check too
    _grasp_idx = schedule.index("grasp")
    n_phases = schedule.n_phases

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
    prev_grip = traj.width_open.copy()   # delta-mode action inversion starts from the open width

    # Per-env FSM state
    phase_idx  = np.zeros(num_envs, dtype=np.int64)
    phase_step = np.zeros(num_envs, dtype=np.int64)

    # Retry state. `_lift_idx` is where the slip check lives; `obj_z_grasp` is the object's height
    # at the moment the grasp closed, which the check measures the rise against.
    _lift_idx = schedule.index("lift")
    _retry_check_step = max(1, int(RETRY_CHECK_FRAC * schedule.duration(_lift_idx)))
    n_retry = np.zeros(num_envs, dtype=np.int64)
    checked = np.zeros(num_envs, dtype=bool)     # slip check fires once per attempt
    # The soft firm check measures a stress RISE over a settled-rest baseline. That baseline is
    # captured once, globally, at step 0 — but after a retry the object has been squeezed, possibly
    # dropped and re-settled, so the old baseline describes a state that no longer exists. Mark it
    # stale per env and re-capture when the re-approach ends.
    rest_stale = np.zeros(num_envs, dtype=bool)
    obj_z_grasp = np.full(num_envs, np.nan)
    # Global step cap. The `while np.any(phase_idx < n_phases)` loop has no bound of its own, so an
    # unbounded rewind would hang the collection forever on one bad env. One full pass through the
    # schedule per allowed attempt, plus slack.
    steps_per_pass = sum(schedule.duration(p) for p in range(n_phases))
    max_total_steps = steps_per_pass * (1 + int(retry_max)) + 20
    n_steps = 0

    # Soft firm-check baseline: top10 von-Mises of the SETTLED object (gripper still
    # at home, no contact) — captured on the first step, used as the "no grasp" floor
    # the grasp->firm rise is measured against. None until the first soft state.
    rest_stress = None

    # Rate-limit audit: how often the per-step clamp engaged. bound_scaled_schedule above should
    # make it a no-op; if it engages often, the scaling missed something and the executed motion is
    # riding the clamp instead of being min-jerk.
    rate_audit = {"steps": 0, "clamped": 0, "worst_ratio": 0.0}

    def _step(cur_pos, cur_quat, cur_grip, record_mask):
        nonlocal prev_pos, prev_quat, prev_grip

        if _rate_lim is not None:
            # Clamp the commanded target against the PREVIOUS commanded target BEFORE both the
            # sim step and the action inversion, so the recorded action IS the clamped command
            # and dataset == execution. Same primitive the backends run at deploy time.
            from gentle_manip.actions.pipeline import clamp_absolute_target, invert_delta_action
            c_pos, c_quat, c_grip = clamp_absolute_target(
                prev_pos, prev_quat, prev_grip, cur_pos, cur_quat, cur_grip, _rate_lim)
            moved = (np.max(np.abs(c_pos - cur_pos)) > 1e-9
                     or np.max(np.abs(c_grip - cur_grip)) > 1e-9
                     or np.max(np.abs(np.abs(np.sum(c_quat * cur_quat, axis=1)) - 1.0)) > 1e-9)
            rate_audit["steps"] += 1
            if moved:
                rate_audit["clamped"] += 1
            cur_pos, cur_quat, cur_grip = (c_pos.astype(np.float32), c_quat.astype(np.float32),
                                           c_grip.astype(np.float32))

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
                contact_force=state.get("contact_force")))
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

    # ── Main loop: every env advances through the schedule independently ──
    while np.any(phase_idx < n_phases):
        n_steps += 1
        if n_steps > max_total_steps:
            print(f"    [retry] global step cap {max_total_steps} hit with "
                  f"{int(np.sum(phase_idx < n_phases))} env(s) unfinished -> forcing DONE")
            phase_idx[:] = n_phases
            break
        active = phase_idx < n_phases   # (N,) bool — envs still progressing this step

        cur_pos_arr  = np.zeros((num_envs, 3), np.float32)
        cur_quat_arr = np.zeros((num_envs, 4), np.float32)
        cur_grip_arr = np.zeros(num_envs, np.float32)
        for i in range(num_envs):
            if active[i]:
                pos, quat, grip = traj.target(i, int(phase_idx[i]), int(phase_step[i]))
            else:
                pos, quat, grip = traj.frozen_target(i)
            cur_pos_arr[i], cur_quat_arr[i], cur_grip_arr[i] = pos, quat, grip

        cur_obs_list, state = _step(cur_pos_arr, cur_quat_arr, cur_grip_arr, record_mask=active)

        # Capture the settled-rest stress baseline on the first step (soft only).
        if rest_stress is None:
            vm0 = state.get("von_mises_stress")
            rest_stress = (_stress_top10(vm0) if vm0 is not None
                           else np.zeros(num_envs, np.float32))

        # ── Idea #2: lift-failure detection -> in-place regrasp (--retry-max) ──
        # Fires once per attempt, partway into `lift`. The EE's rise is guaranteed by construction
        # (its target is a pure function of the phase step), so "object has not risen" IS a slip.
        if retry_max > 0:
            oc = state.get("object_center")
            if oc is not None:
                oz = np.asarray(oc)[:, 2]
                slipped = (active & (phase_idx == _lift_idx) & (phase_step >= _retry_check_step)
                           & ~checked & np.isfinite(obj_z_grasp))
                for i in np.where(slipped)[0]:
                    checked[i] = True
                    rise = float(oz[i] - obj_z_grasp[i])
                    if rise >= RETRY_MIN_RISE_M or n_retry[i] >= retry_max:
                        if rise < RETRY_MIN_RISE_M:
                            print(f"    [retry] env {i}: slip (rise {rise*1e3:.1f}mm) but attempt "
                                  f"cap {retry_max} reached -> continuing to a failed lift")
                        continue
                    n_retry[i] += 1
                    # Re-seed the approach from HERE, then rewind to phase 0. The failed attempt
                    # stays in the recorded buffers on purpose: a policy trained only on clean
                    # successes has never seen what to do after a slip.
                    traj.begin_retry(i, cur_pos_arr[i], cur_quat_arr[i], cur_grip_arr[i])
                    phase_idx[i] = 0
                    phase_step[i] = -1          # the += 1 below lands it on 0
                    checked[i] = False
                    rest_stale[i] = True        # re-captured when the re-approach ends (gripper
                    obj_z_grasp[i] = np.nan     # open, not yet touching) — see below
                    print(f"    [retry] env {i}: object rose only {rise*1e3:.1f}mm during lift "
                          f"-> regrasp attempt {int(n_retry[i])}/{retry_max}")

        # Advance phase state for envs that were active this step.
        phase_step[active] += 1
        # phase_idx may already be n_phases (done envs) — clip before indexing the schedule;
        # those entries are masked out by `active` anyway so the clipped value is unused.
        durations = np.array([schedule.duration(min(int(p), n_phases - 1)) for p in phase_idx])
        rolled_over = active & (phase_step >= durations)

        # Idea #1: force-based grasp firming. Check ONCE, exactly at the moment an
        # env finishes "grasp" (about to enter "firm") — read this env's just-measured
        # contact force. If it's already fine, SKIP "firm" entirely (jump straight to
        # "lift", +2 instead of +1) rather than stepping through a no-op hold and
        # relying on the trim pass to clean it up afterward — no artificial stops.
        advance = np.ones(num_envs, dtype=np.int64)   # normal: one phase forward
        leaving_grasp = rolled_over & (phase_idx == _grasp_idx)
        if retry_max > 0:
            # The object's height at the moment the grasp closes — the reference the slip check
            # measures the lift rise against. Per attempt, so a retry re-captures it.
            oc = state.get("object_center")
            if oc is not None:
                obj_z_grasp[leaving_grasp] = np.asarray(oc)[leaving_grasp, 2]
            # Re-capture the rest-stress baseline as the re-approach ends: the gripper is at the
            # grasp pose but still open at the preshape width, so nothing is touching yet.
            reapproached = rolled_over & (phase_idx == 0) & rest_stale
            if np.any(reapproached):
                vm = state.get("von_mises_stress")
                if vm is not None:
                    cur_rest = _stress_top10(vm)
                    rest_stress[reapproached] = cur_rest[reapproached]
                rest_stale[reapproached] = False
        if _has_firm and np.any(leaving_grasp):
            cf = state.get("contact_force")
            if cf is not None:                        # RIGID: contact force (N). ALWAYS firm base;
                for i in np.where(leaving_grasp)[0]:  # weak grip (force < thresh) firms base+extra.
                    if cf[i] < FIRM_FORCE_THRESH_N:
                        traj.set_firm_close(i, FIRM_EXTRA_CLOSE_M + FIRM_WEAK_EXTRA_CLOSE_M)
                        print(f"    [firm] env {i}: weak grip force {cf[i]:.2f}N < "
                              f"{FIRM_FORCE_THRESH_N}N -> closing "
                              f"{traj.firm_close[i]*1000:.1f}mm (base+extra)")
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
                            traj.set_firm_close(i, FIRM_EXTRA_CLOSE_M + FIRM_WEAK_EXTRA_CLOSE_M)
                            print(f"    [firm] env {i}: weak stress rise {rise:.0f}Pa < "
                                  f"{FIRM_STRESS_THRESH_PA:.0f}Pa -> closing "
                                  f"{traj.firm_close[i]*1000:.1f}mm (base+extra)")
                        # else: firm_close stays at base -> still firms, just not extra

        phase_idx[rolled_over]  += advance[rolled_over]
        phase_step[rolled_over]  = 0

    if _rate_lim is not None and rate_audit["steps"]:
        frac = rate_audit["clamped"] / rate_audit["steps"]
        msg = (f"    [rate-limit] clamp engaged on {rate_audit['clamped']}/{rate_audit['steps']} "
               f"steps ({100 * frac:.1f}%)")
        if frac > 0.02:
            msg += "  *** >2%: bound_scaled_schedule missed something — motion is riding the clamp ***"
        print(msg)

    if retry_max > 0 and np.any(n_retry > 0):
        print(f"    [retry] {int(np.sum(n_retry > 0))}/{num_envs} env(s) regrasped "
              f"({int(n_retry.sum())} attempts total) in {n_steps} steps")

    # Success check from final object height. Rigid: get_pos; soft MPM: the particle-mean centre from
    # the last step's state (MPMEntity has no get_pos).
    _o = worker.handle.objects[0]
    obj_z   = _np(_o.get_pos())[:, 2] if hasattr(_o, "get_pos") else np.asarray(state["object_center"])[:, 2]
    success = obj_z > (grasp_pos[:, 2] + lift_height * 0.5)

    # Post-process: trim long held-command runs (absolute mode only — see
    # _trim_long_holds docstring for why delta mode is excluded).
    if action_config.mode == "absolute":
        for i in range(num_envs):
            act_bufs[i], obs_bufs[i], rew_bufs[i], frame_bufs[i] = _trim_long_holds(
                act_bufs[i], obs_bufs[i], rew_bufs[i], frame_bufs[i])

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
    p.add_argument("--grasp-E",           type=float, default=3e5,  help="object Young's modulus (Pa)")
    p.add_argument("--grasp-density",     type=float, default=1000.0, help="object density (kg/m^3)")
    p.add_argument("--grasp-mu",          type=float, default=0.7,  help="pad-object friction coefficient")
    p.add_argument("--grasp-accel",       type=float, default=9.81,
                   help="lift-acceleration safety margin (m/s^2): holdability needs 2*mu*grip >= m*(g+accel), "
                        "so a positive value picks a FIRMER holdable width that survives the dynamic lift "
                        "(0 = gentlest quasi-static grasp, which tends to slip during the lift)")
    p.add_argument("--table-z",           type=float, default=0.0,  help="table surface height (world z, m)")
    p.add_argument("--grasp-voxel-div",   type=int,   default=14,   help="FEM remesh resolution (keep ndof<~5k)")
    p.add_argument("--grasp-target-tets", type=int,   default=1500, help="FEM target tet count")
    p.add_argument("--grasp-n-starts",    type=int,   default=6,    help="CMA multi-start count")
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
    p.add_argument("--n-lift", type=int, default=N_LIFT,
                   help=f"lift-phase steps; default {N_LIFT}.")
    p.add_argument("--n-firm", type=int, default=N_FIRM,
                   help=f"'firm' phase steps (post-grasp extra squeeze idea #1); default {N_FIRM}. "
                        "0 = NO firm phase at all — the grasp goes straight to lift at width_cls "
                        "(matches the pre-firm v1 collector, e.g. the cho dataset).")
    p.add_argument("--n-settle", type=int, default=N_SETTLE,
                   help=f"steps held at the grasp pose before closing; default {N_SETTLE}.")
    p.add_argument("--n-hold", type=int, default=N_HOLD,
                   help=f"steps held at lift height (the success window); default {N_HOLD}.")
    # ── v4 trajectory ─────────────────────────────────────────────────────────
    p.add_argument("--traj", choices=["v3", "v4", "v4split"], default="v4",
                   help="v4 (DEFAULT) = one BLENDED reach: a quadratic Bezier home -> grasp with the "
                        "pre-grasp standoff as its control point, so the fingers arrive exactly along "
                        "their own approach axis WITHOUT stopping mid-reach. v4split = the explicit "
                        "approach_xy -> align -> descend decomposition (measured ~5.7x worse "
                        "dimensionless jerk: stopping at the standoff is what costs it). v3 = the "
                        "original single lerp+slerp. Target-trajectory metrics (njerk / vpeaks): "
                        "v3 linear 11935/10, v3+minjerk 277/2, v4split ~1580/3, v4 blend 287/2.")
    p.add_argument("--n-approach-xy", type=int, default=N_APPROACH_XY,
                   help=f"v4 travel-phase steps; default {N_APPROACH_XY}. The three v4 approach "
                        f"phases sum to {N_APPROACH_XY + N_ALIGN + N_DESCEND} so total episode "
                        f"length matches v3's approach ({N_HOME_TO_PRE}).")
    p.add_argument("--n-align", type=int, default=N_ALIGN,
                   help=f"v4 rotate-in-place steps at the standoff; default {N_ALIGN}.")
    p.add_argument("--n-descend", type=int, default=N_DESCEND,
                   help=f"v4 straight-line descent steps; default {N_DESCEND}.")
    p.add_argument("--standoff", type=float, default=STANDOFF_M,
                   help=f"v4 pre-grasp standoff distance along the approach axis (m); default "
                        f"{STANDOFF_M}. Escalated automatically if the descent would clip the object.")
    p.add_argument("--preshape-factor", type=float, default=PRESHAPE_FACT,
                   help=f"approach aperture = this x the grasp width (default {PRESHAPE_FACT}, ~human "
                        "reach-to-grasp). 0 = hold fully open like v3. A narrower approach also cuts "
                        "swept volume and camera occlusion during the descent.")
    p.add_argument("--no-minjerk", action="store_true",
                   help="use LINEAR time scaling (v3 behaviour) instead of minimum-jerk. Min-jerk "
                        "gives a bell-shaped velocity profile with zero velocity AND acceleration at "
                        "every phase boundary; since actions are derived from consecutive targets, it "
                        "smooths the action stream the policy is trained on.")
    p.add_argument("--lift-height", type=float, default=LIFT_HEIGHT,
                   help=f"metres to lift above the grasp point; default {LIFT_HEIGHT}.")
    # ── v4.1 shelf lift ───────────────────────────────────────────────────────
    p.add_argument("--shelf-deg", type=float, default=0.0,
                   help="rotate the gripper by this much DURING the lift so one finger ends up "
                        "beneath the other and acts as a floor, carrying the object's weight as a "
                        "normal force instead of by friction. 0 (default) = off. Required grip "
                        "follows (mg/2)*max(cos/mu, sin), minimised at arctan(1/mu) = 55deg for "
                        "mu=0.7 (0.57x); 90deg is WORSE (0.70x). ABSOLUTE ACTION ONLY.")
    p.add_argument("--shelf-open", type=float, default=0.0,
                   help="metres of gripper width to release once the shelf exists (after the "
                        "rotation completes). This is where the stress reduction actually comes "
                        "from -- at a fixed width the rotation ADDS normal load (first order in von "
                        "Mises) while removing shear (second order), so rotation ALONE can be a "
                        "regression. ~0.0025 recommended.")
    p.add_argument("--shelf-sign", default="auto",
                   help="which finger becomes the floor: +1 / -1 / auto. Both make a shelf; the "
                        "sign picks which way the gripper BODY swings (~146mm at full rotation). "
                        "'auto' swings it away from the camera.")
    p.add_argument("--shelf-frac", type=float, nargs=2, default=(0.10, 0.60),
                   help="lift-progress window over which the rotation ramps")
    p.add_argument("--shelf-open-frac", type=float, nargs=2, default=(0.60, 1.00),
                   help="lift-progress window for the width release (must FOLLOW --shelf-frac)")
    p.add_argument("--init-width-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="v4.1 robustness: sample each env's INITIAL gripper aperture (m) from this "
                        "range instead of the fixed open width, e.g. --init-width-range 0.05 0.08. "
                        "Collection knob, not a benchmark one -- it widens the start distribution a "
                        "cloned policy has seen, and does not change the synthesized grasp.")
    p.add_argument("--retry-max", type=int, default=0,
                   help="v4.1: regrasp attempts when the object is detected NOT to have risen with "
                        "the gripper partway into the lift (a slip). 0 (default) = off, "
                        "bit-identical to v4. INDEPENDENT of --shelf-deg: this is the robustness "
                        "fallback if the shelf trajectory turns out too hard to clone. The failed "
                        "attempt STAYS in the recorded demo on purpose -- a policy trained only on "
                        "clean successes has never seen what to do after a slip.")
    p.add_argument("--grasp-profile", choices=sorted(GRASP_PROFILES), default=None,
                   help="named objective from grasp_profiles.py — the SAME resolution the benchmark "
                        "uses, so a collection and its benchmark cannot disagree about what a "
                        "profile means. None (default) keeps the legacy flag-driven behaviour; "
                        "individual --grasp-* flags still override on top of a profile.")
    p.add_argument("--grasp-execute-offset", type=float, default=None,
                   help="score each candidate at the width the executor ACTUALLY commands "
                        "(synthesized width minus this; the collector closes base 2.5mm + firm "
                        "2mm = 4.5mm tighter than the scored width). v4's operating-point fix — "
                        "0.0045 in the v4fix/v5 profiles. None = profile value or historical 0.")
    p.add_argument("--grasp-cam-azimuth-max", type=float, default=None,
                   help="v5 occlusion bound: cap the closing-axis azimuth to the camera ray (deg). "
                        "0 = axis perpendicular to the ray (no occlusion), 90 = along it (a finger "
                        "between camera and object). Structural (shaped penalty at every ladder "
                        "rung + seeds centred on the perpendicular), because the soft w_occ weight "
                        "is measurably inert. None = off; 45 is the v5 profile's value.")
    p.add_argument("--no-descend-check", action="store_true",
                   help="skip the swept collision check on the straight descent (which escalates the "
                        "standoff, then falls back to a top-down grasp, if the fingers would clip)")
    # ── v4 objective priors — ALL default-off, so v4 reproduces v3's grasps until enabled ──────
    p.add_argument("--grasp-peak", type=float, default=None,
                   help="w_peak, the unmasked-p98 CONCENTRATED-contact penalty (anti-pinch). The "
                        "metric default is 0.3; it has been inert in every run to date because the "
                        "old sentinel forwarded 0.0. Pass 0.3 to enable it.")
    p.add_argument("--grasp-com", type=float, default=0.0,
                   help="w_com: penalize the HORIZONTAL lever arm from the pad centre to the object "
                        "COM (anti-stem — a stem grasp holds far from the mass, so the body swings "
                        "out during the lift). 0 = off.")
    p.add_argument("--grasp-tilt", type=float, default=0.0,
                   help="w_tilt: penalize approach-axis deviation from straight down (anti-side-grasp). "
                        "0 = off. NOTE occlusion is driven mainly by YAW, not tilt — see --grasp-occ.")
    p.add_argument("--grasp-occ", type=float, default=0.0,
                   help="w_occ: penalize the fraction of the camera's view of the object that the "
                        "fingers block. 0 = off (the value is still MEASURED for the audit).")
    p.add_argument("--grasp-area-min", type=float, default=0.0,
                   help="area_min (m^2): hard floor on the WORST pad's contact area; below it a "
                        "candidate is rejected as a pinch. 0 = off. (2e-5 = 20 mm^2.)")
    p.add_argument("--grasp-roll-max-deg", type=float, default=None,
                   help="half-width of the roll search band about top-down (degrees). The historical "
                        "default of 90 admits a FULLY HORIZONTAL tool axis, i.e. pure side grasps; "
                        "15-30 excludes them structurally rather than by penalty alone.")
    p.add_argument("--grasp-extra-close", type=float, default=0.0,
                   help="squeeze FURTHER IN than the synthesized width by this many meters (tighter grip) "
                        "for EVERY grasp — e.g. 0.005 = close 5mm tighter. 0 (default) = no change. Use to "
                        "make grasps firmer (a too-gentle grip -> premature lift / slip before secured).")
    args = p.parse_args()

    # Phase layout is built HERE and passed explicitly into execute_and_collect — no module-level
    # mutable globals (v3 rebuilt PHASES/N_PHASES/_GRASP_IDX via `global`, which the benchmark then
    # read back, so the two could silently disagree about the schedule).
    schedule = build_schedule(args)
    print(f"[v4] schedule: {' -> '.join(f'{n}({d})' for n, d in schedule.phases)}"
          f"  | minjerk={not args.no_minjerk} standoff={args.standoff}m "
          f"preshape={args.preshape_factor}x")

    # The v4 objective priors, resolved once and applied to BOTH the primary synthesis and the
    # no-diversity retry (otherwise a retried env would be scored by a different objective).
    obj_kw = dict(w_com=args.grasp_com, w_tilt=args.grasp_tilt, w_occ=args.grasp_occ,
                  area_min=args.grasp_area_min,
                  cam_azimuth_max_deg=args.grasp_cam_azimuth_max)  # cam_pos added below, once the task is loaded
    if args.grasp_profile is not None:
        # Profile first, explicit CLI flags override. The profile also carries the DIVERSITY
        # settings (diversity_tol/jitter/pitch seeds), which for profiles are part of the named
        # objective — but those are separate argparse args here, so apply them explicitly.
        prof = dict(GRASP_PROFILES[args.grasp_profile])
        for k in ("diversity_tol", "jitter_deg", "jitter_pos", "pitch_seed_deg", "w_align"):
            if k in prof:
                setattr(args, {"diversity_tol": "grasp_diversity_tol", "jitter_deg": "grasp_jitter_deg",
                               "jitter_pos": "grasp_jitter_pos", "pitch_seed_deg": "grasp_pitch_seed_deg",
                               "w_align": "grasp_align"}[k], prof.pop(k))
        merged = dict(prof)
        for k, v in obj_kw.items():                       # explicit CLI values win over the profile
            if v not in (None, 0.0):
                merged[k] = v
        obj_kw = merged
        print(f"[v4] objective profile '{args.grasp_profile}' -> "
              f"{ {k: v for k, v in obj_kw.items() if k != 'cam_pos'} }")
    if args.grasp_execute_offset is not None:
        obj_kw["execute_offset"] = args.grasp_execute_offset
    if args.grasp_peak is not None:
        obj_kw["w_peak"] = args.grasp_peak
    if args.grasp_roll_max_deg is not None:
        obj_kw["roll_max"] = np.radians(args.grasp_roll_max_deg)
    if any(v for v in (args.grasp_com, args.grasp_tilt, args.grasp_occ, args.grasp_area_min)) \
            or args.grasp_peak is not None or args.grasp_roll_max_deg is not None:
        print(f"[v4] objective priors ACTIVE: { {k: v for k, v in obj_kw.items() if v} }")

    # ── v4.1 shelf lift ───────────────────────────────────────────────────────
    # ABSOLUTE ACTION ONLY. The rotation is ~39x the delta rotation scale (0.001 rad/step), so a
    # DERIVED delta action would silently clip: the sim executes the trajectory correctly while the
    # recorded dataset no longer reproduces it. That is the same class of silent corruption as the
    # euler-seam bug, so it is a hard error rather than a warning.
    shelf_sign = args.shelf_sign if args.shelf_sign == "auto" else float(args.shelf_sign)
    shelf_kw = dict(shelf_deg=args.shelf_deg, shelf_open=args.shelf_open, shelf_sign=shelf_sign,
                    shelf_frac=tuple(args.shelf_frac), shelf_open_frac=tuple(args.shelf_open_frac))
    if args.shelf_deg > 0:
        if exp.action_config.mode != "absolute":
            raise SystemExit(
                f"--shelf-deg {args.shelf_deg} requires an ABSOLUTE action config; this experiment "
                f"uses mode={exp.action_config.mode!r}. The shelf rotation exceeds the delta "
                f"rotation scale by ~39x, so the recorded delta actions would clip and NOT "
                f"reproduce the trajectory. Use an abs_pose_* action, or --shelf-deg 0.")
        if args.shelf_open_frac[0] < args.shelf_frac[1]:
            print(f"  *** WARNING: width release starts at s={args.shelf_open_frac[0]} but the "
                  f"rotation only finishes at s={args.shelf_frac[1]} — releasing before the shelf "
                  f"exists drops the object ***")
        print(f"[v4.1] SHELF LIFT: {args.shelf_deg:.0f}deg, release {args.shelf_open*1e3:.1f}mm, "
              f"sign={args.shelf_sign}, rot s{tuple(args.shelf_frac)} open s{tuple(args.shelf_open_frac)}")

    # ── Load everything from the experiment config (same as training / eval) ──
    exp        = Experiment.load(args.experiment)
    task       = SingleLiftTask(exp.task_cfg)
    spec       = task.scene_spec
    # Camera pose from the task, so occlusion is MEASURED for the audit even when w_occ == 0
    # (otherwise that column reads 0.0, which looks like "no occlusion" but means "not computed").
    obj_kw["cam_pos"] = tuple(spec.cameras[0].pos) if spec.cameras else None
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
                        "n_settle": args.n_settle, "n_hold": args.n_hold,
                        "grasp_extra_close": args.grasp_extra_close,
                        # v4 trajectory — no config snapshot = not reproducible (repo rule #7)
                        "traj": args.traj, "n_approach_xy": args.n_approach_xy,
                        "n_align": args.n_align, "n_descend": args.n_descend,
                        "standoff": args.standoff, "preshape_factor": args.preshape_factor,
                        "minjerk": not args.no_minjerk, "lift_height": args.lift_height,
                        "descend_check": not args.no_descend_check,
                        "schedule": [list(p) for p in schedule.phases],
                        # v4 objective priors (0 / None = inert, i.e. v3-equivalent scoring)
                        "w_com": args.grasp_com, "w_tilt": args.grasp_tilt,
                        "w_occ": args.grasp_occ, "w_peak": args.grasp_peak,
                        "area_min": args.grasp_area_min,
                        "roll_max_deg": args.grasp_roll_max_deg,
                        "cam_azimuth_max_deg": args.grasp_cam_azimuth_max,
                        "grasp_profile": args.grasp_profile,
                        "execute_offset": obj_kw.get("execute_offset"),
                        # v4.1 shelf lift
                        "shelf_deg": args.shelf_deg, "shelf_open": args.shelf_open,
                        "shelf_sign": args.shelf_sign,
                        "shelf_frac": list(args.shelf_frac),
                        "shelf_open_frac": list(args.shelf_open_frac),
                        "retry_max": args.retry_max,
                        "init_width_range": (list(args.init_width_range)
                                             if args.init_width_range else None)},
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

    # ── Build scene + worker (with per-scene SIZE+SHAPE DR) ──
    # Scene DR re-randomizes object geometry by REBUILDING the worker every N batches (GenesisWorker
    # has no in-process geometry re-randomize; a fresh build is the only path). Verified memory-stable
    # across rebuilds (gs.destroy reclaims each build). Geometry is shared across a batch's envs.
    nominal_spec   = spec
    do_scene_dr    = args.scene_dr_every > 0 and dr_cfg.has_scene_dr()
    deform_dir     = tempfile.mkdtemp(prefix="gm_synth_deform_") if do_scene_dr else None

    def _make_worker():
        """Build a GenesisWorker; if scene DR is on, on a freshly deformed+scaled mesh.
        Returns (worker, scene_dr_dict, actual_mesh_path)."""
        if do_scene_dr:
            spec_dr, sdr = _apply_scene_dr(nominal_spec, dr_cfg, rng, deform_dir)
        else:
            spec_dr, sdr = nominal_spec, {"scale": float(nominal_spec.objects[0].scale or 1.0),
                                          "bend_deg": 0.0}
        w = GenesisWorker(spec_dr, num_envs=args.n_envs, show_viewer=False,
                          settle_steps=settle_steps, settle_max_steps=settle_max_steps,
                          settle_vel_thresh=settle_vel_thresh, render_obs_cameras=True)
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
                        "stress_Pa", "grip_N", "align", "pressure_Pa", "min_pad_mm2", "width_mm",
                        # v4 grasp-quality audit — the columns that make stem/pinch/side grasps
                        # countable in the COLLECTED set, not just in the benchmark
                        "tilt_deg", "com_lever_mm", "occ_pred", "standoff_m", "descend_pen_mm"])

    total_saved  = 0
    total_failed = 0
    batch_idx   = 0
    shard_buf:  List[dict] = []
    shard_idx   = 0
    fem_mesh: Optional[str] = None            # v3: cache the FEM (obj+pad_geo) keyed on actual_mesh —
    fem_obj = fem_pad_geo = fem_meta = None   #     rebuild only when a scene-DR relaunch changes the mesh
    fem_sdf = None                            #     penetration SDF for the descend check (per FEM)
    t0 = time.time()

    while total_saved < args.n_episodes:
        batch_idx += 1
        n = args.n_envs

        # ── Scene DR: rebuild the worker with fresh object SIZE+SHAPE every N batches ──
        if do_scene_dr and batch_idx > 1 and (batch_idx - 1) % args.scene_dr_every == 0:
            worker.close()
            worker, scene_dr, actual_mesh = _make_worker()

        print(f"\n── Batch {batch_idx}  [{total_saved}/{args.n_episodes} saved]"
              + (f"  scale={scene_dr['scale']:.3f} bend={scene_dr['bend_deg']:+.1f}°"
                 if do_scene_dr else "") + " ──")

        # ── Reset with per-env pose DR (ranges from experiment DR config) ──
        object_dxy   = dr_cfg.sample_object_dxy(rng, n)
        object_euler = dr_cfg.sample_object_euler(rng, n)
        home_offset  = dr_cfg.sample_home_offset(rng, n)   # per-env arm-home jitter (sim-only DR)
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
                contact_force=init_state.get("contact_force")))

        # ── Per-env FEM gentleness grasp synthesis (v3) ──
        # Build the FEM ElasticObject ONCE for this batch's ACTUAL (DR shape+size) mesh — all envs share
        # it (scene-DR varies per relaunch, not per sub-env), so the expensive factorization is reused;
        # rebuild only when actual_mesh changes (a scene-DR relaunch). Then plan per-env settled pose.
        if actual_mesh != fem_mesh:
            fem_obj, fem_pad_geo, fem_meta = fg.build_grasp_fem(
                actual_mesh, voxel_div=args.grasp_voxel_div, target_tets=args.grasp_target_tets,
                use_gpu=args.grasp_gpu)
            fem_mesh = actual_mesh
            fem_sdf = None            # invalidate: the SDF belongs to the mesh that just changed
            print(f"  FEM: {fem_meta['tets']} tets, ndof={fem_meta['ndof']}, gpu={fem_meta['gpu']}")
        all_best_x = []
        all_grasp  = []                                          # per-env synthesis dict (for the grasp-pose viz)
        all_standoff = []                                        # per-env descent standoff (may escalate)
        home_pos_probe = worker.robot.home_pos[None].astype(np.float32)
        home_quat_probe = worker.robot.home_quat[None].astype(np.float32)
        for i in range(n):
            cma_seed = int(cma_seed_rng.integers(1, 2**31 - 1))
            r = fg.synthesize_grasp(fem_obj, fem_pad_geo, obj_pos_all[i], obj_quat_all[i],
                                    E=args.grasp_E, density=args.grasp_density, mu=args.grasp_mu,
                                    table_z=args.table_z, maxfevals=args.maxfevals,
                                    n_starts=args.grasp_n_starts, seed=cma_seed, accel=args.grasp_accel,
                                    diversity_tol=args.grasp_diversity_tol, jitter_deg=args.grasp_jitter_deg,
                                    jitter_pos=args.grasp_jitter_pos, w_align=args.grasp_align,
                                    pitch_seed_deg=args.grasp_pitch_seed_deg, **obj_kw)
            if r.get("x") is None or r.get("stress_top10") is None:   # diversity found no feasible grasp;
                print(f"  Env {i}: no feasible diverse grasp -> retry WITHOUT diversity")  # retry reliably
                r = fg.synthesize_grasp(fem_obj, fem_pad_geo, obj_pos_all[i], obj_quat_all[i],
                                        E=args.grasp_E, density=args.grasp_density, mu=args.grasp_mu,
                                        table_z=args.table_z, maxfevals=args.maxfevals,
                                        n_starts=args.grasp_n_starts, seed=cma_seed + 7,
                                        accel=args.grasp_accel, **obj_kw)
            best_x = r["x"]
            if best_x is None or r.get("stress_top10") is None:       # extremely rare: still nothing ->
                # default straight-down grasp at the object xy so the FSM never sees None (this episode may
                # not lift -> simply won't be saved, but the batch completes). tcp sits low (FINGER_TO_TCP_Z).
                best_x = np.array([obj_pos_all[i][0], obj_pos_all[i][1],
                                   float(obj_pos_all[i][2]) + FINGER_TO_TCP_Z, np.pi, 0.0, 0.0, 0.045])
                r = {"x": best_x, "stress_top10": None, "grip": None, "align": None,
                     "pressure": None, "min_pad_area": None, "width_face": None}
                print(f"  Env {i}: SYNTH FAILED -> default top-down grasp (fallback)")
            # ── v4: verify the STRAIGHT DESCENT is clear ──────────────────────
            # The descend phase is collision-free by construction only if the standoff itself is
            # reachable in a straight line. On an overhanging cap (or after aggressive shape DR) the
            # fingers can clip on the way down, so sweep the segment against the same finger/object
            # penetration test the per-candidate filter uses, and escalate the standoff if needed.
            # A high escalation rate means roll_max is too loose — worth watching in the log.
            descend_standoff, descend_pen = args.standoff, 0.0
            if args.traj == "v4" and not args.no_descend_check and best_x is not None:
                if fem_sdf is None:
                    fem_sdf = fg.build_object_sdf(fem_obj)
                for cand in (args.standoff, args.standoff * 1.2, args.standoff * 1.6):
                    # Check the path we will ACTUALLY command, not a straight standoff->grasp
                    # chord: with the blended (Bezier) reach the two differ, and the chord
                    # under-reports how close the executed path gets (measured 0.8mm vs 2.0mm of
                    # finger/object overlap on the same grasp).
                    probe = GraspTrajectory(schedule, [best_x],
                                            home_pos_probe, home_quat_probe,
                                            lift_height=args.lift_height, standoff=cand,
                                            use_minjerk=not args.no_minjerk,
                                            preshape_factor=args.preshape_factor)
                    ai = next((k for k in ("reach", "descend", "approach_xy", "approach")
                               if schedule.has(k)), None)
                    poses = []
                    if ai is not None:
                        pi_ = schedule.index(ai)
                        for st in range(0, schedule.duration(pi_), 4):
                            pp, qq, gg = probe.target(0, pi_, st)
                            rr = Rot.from_quat(np.asarray(qq, float)[[1, 2, 3, 0]]).as_euler("xyz")
                            poses.append(np.concatenate([pp, rr, [gg]]))
                    ok, pen = (fg.path_clearance(poses, fem_pad_geo, fem_sdf, obj_pos_all[i],
                                                 obj_quat_all[i]) if poses else
                               fg.descend_clearance(best_x, fem_pad_geo, fem_sdf,
                                                    obj_pos_all[i], obj_quat_all[i], d=cand))
                    descend_standoff, descend_pen = cand, pen
                    if ok:
                        break
                else:
                    print(f"  Env {i}: descent still clips by {descend_pen*1e3:.1f}mm at "
                          f"{descend_standoff*1e2:.0f}cm standoff -> keeping it (grasp may fail)")
                if descend_standoff != args.standoff:
                    print(f"  Env {i}: descent clipped -> standoff escalated to "
                          f"{descend_standoff*1e2:.0f}cm")
            r["descend_standoff"] = descend_standoff
            r["descend_pen_mm"] = descend_pen * 1e3
            all_standoff.append(descend_standoff)

            all_best_x.append(best_x); all_grasp.append(r)
            if r.get("stress_top10") is not None:
                print(f"  Env {i}: stress={r['stress_top10']:.0f}Pa grip={r['grip']:.3f}N align={r['align']:.3f}"
                      f"  tilt={r.get('tilt_deg') or 0:.1f}deg lever={(r.get('com_lever') or 0)*1e3:.1f}mm"
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
        obs_bufs, act_bufs, rew_bufs, success, frame_bufs = execute_and_collect(
            worker, all_best_x, init_obs_batch, perception, action_config,
            record_video=rec_this_batch, priv_cfg=priv_cfg, dr_vec=dr_vec,
            extra_close=args.grasp_extra_close,
            schedule=schedule, lift_height=args.lift_height, standoff=all_standoff,
            use_minjerk=not args.no_minjerk, preshape_factor=args.preshape_factor,
            retry_max=args.retry_max, width_open=(
                None if not args.init_width_range else
                # Per-env, resampled every batch: the aperture the reach STARTS from. Only the
                # start varies -- the preshape and the closed width are still derived from the
                # synthesized grasp, so this widens the demo distribution without changing the
                # grasp itself.
                np.random.default_rng(args.seed + batch_idx).uniform(
                    args.init_width_range[0], args.init_width_range[1], size=n)),
            shelf_kw=dict(shelf_kw, cam_pos=obj_kw.get("cam_pos"),
                          # TCP -> pad-centre offset along tool z (~25mm). The shelf pivots about
                          # the PAD CENTRE, not the TCP: rotating about the TCP swings the object
                          # on a 25mm arc, enough to drop obj_z out of the success band.
                          shelf_pivot_z=float(fem_pad_geo["z_center"] + fg._z_off(float(np.mean(
                              [float(np.asarray(x, float)[6]) for x in all_best_x]))))
                          if (args.shelf_deg > 0 and fem_pad_geo is not None) else 0.0),
        )
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
                                round(float(g.get("stress_top10") or 0), 1), round(float(g.get("grip") or 0), 4),
                                round(float(g.get("align") or 0), 4), round(float(g.get("pressure") or 0), 1),
                                round(float((g.get("min_pad_area") or 0) * 1e6), 2),
                                round(float(g["x"][6] * 1e3), 2),
                                round(float(g.get("tilt_deg") or 0), 2),
                                round(float((g.get("com_lever") or 0) * 1e3), 2),
                                ("" if g.get("occ") is None else round(float(g["occ"]), 4)),
                                round(float(g.get("descend_standoff") or 0), 4),
                                round(float(g.get("descend_pen_mm") or 0), 2)])
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
    print(f"  Elapsed          : {elapsed/60:.1f} min")
    print(f"  Data             : {data_path}")

    stats = {
        "episodes_saved":  total_saved,
        "episodes_failed": total_failed,
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
