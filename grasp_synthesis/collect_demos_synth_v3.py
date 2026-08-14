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
    contact_force: (N,) or None (rigid-only; state["contact_force"] from read_state()/step()).
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
    width_cls  = np.array([p[2] - 0.0025 for p in poses], np.float32)
    # Mutable — "firm" phase tightens this once per env (idea #1); "lift"/"hold"/
    # a frozen (DONE) env all read the FINAL width, which is width_cls unless firmed.
    grip_target = width_cls.copy()
    # Per-env firm close distance (m). SOFT firms EVERY grasp by the base amount (the grip
    # margin the old path gave all grasps); a weak grasp (low stress rise) gets set to a LARGER
    # value at the grasp->firm boundary. RIGID leaves this at base and skips firm when already firm.
    firm_close = np.full(num_envs, FIRM_EXTRA_CLOSE_M, np.float32)

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
        rolled_over = active & (phase_step >= durations)

        # Idea #1: force-based grasp firming. Check ONCE, exactly at the moment an
        # env finishes "grasp" (about to enter "firm") — read this env's just-measured
        # contact force. If it's already fine, SKIP "firm" entirely (jump straight to
        # "lift", +2 instead of +1) rather than stepping through a no-op hold and
        # relying on the trim pass to clean it up afterward — no artificial stops.
        advance = np.ones(num_envs, dtype=np.int64)   # normal: one phase forward
        leaving_grasp = rolled_over & (phase_idx == _GRASP_IDX)
        if np.any(leaving_grasp):
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
    args = p.parse_args()

    # ── Load everything from the experiment config (same as training / eval) ──
    exp        = Experiment.load(args.experiment)
    task       = SingleLiftTask(exp.task_cfg)
    spec       = task.scene_spec
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
                        "seed": args.seed},
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
                        "stress_Pa", "grip_N", "align", "pressure_Pa", "min_pad_mm2", "width_mm"])

    total_saved  = 0
    total_failed = 0
    batch_idx   = 0
    shard_buf:  List[dict] = []
    shard_idx   = 0
    fem_mesh: Optional[str] = None            # v3: cache the FEM (obj+pad_geo) keyed on actual_mesh —
    fem_obj = fem_pad_geo = fem_meta = None   #     rebuild only when a scene-DR relaunch changes the mesh
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
            print(f"  FEM: {fem_meta['tets']} tets, ndof={fem_meta['ndof']}, gpu={fem_meta['gpu']}")
        all_best_x = []
        all_grasp  = []                                          # per-env synthesis dict (for the grasp-pose viz)
        for i in range(n):
            cma_seed = int(cma_seed_rng.integers(1, 2**31 - 1))
            r = fg.synthesize_grasp(fem_obj, fem_pad_geo, obj_pos_all[i], obj_quat_all[i],
                                    E=args.grasp_E, density=args.grasp_density, mu=args.grasp_mu,
                                    table_z=args.table_z, maxfevals=args.maxfevals,
                                    n_starts=args.grasp_n_starts, seed=cma_seed, accel=args.grasp_accel,
                                    diversity_tol=args.grasp_diversity_tol, jitter_deg=args.grasp_jitter_deg,
                                    jitter_pos=args.grasp_jitter_pos, w_align=args.grasp_align,
                                    pitch_seed_deg=args.grasp_pitch_seed_deg)
            if r.get("x") is None or r.get("stress_top10") is None:   # diversity found no feasible grasp;
                print(f"  Env {i}: no feasible diverse grasp -> retry WITHOUT diversity")  # retry reliably
                r = fg.synthesize_grasp(fem_obj, fem_pad_geo, obj_pos_all[i], obj_quat_all[i],
                                        E=args.grasp_E, density=args.grasp_density, mu=args.grasp_mu,
                                        table_z=args.table_z, maxfevals=args.maxfevals,
                                        n_starts=args.grasp_n_starts, seed=cma_seed + 7, accel=args.grasp_accel)
            best_x = r["x"]
            if best_x is None or r.get("stress_top10") is None:       # extremely rare: still nothing ->
                # default straight-down grasp at the object xy so the FSM never sees None (this episode may
                # not lift -> simply won't be saved, but the batch completes). tcp sits low (FINGER_TO_TCP_Z).
                best_x = np.array([obj_pos_all[i][0], obj_pos_all[i][1],
                                   float(obj_pos_all[i][2]) + FINGER_TO_TCP_Z, np.pi, 0.0, 0.0, 0.045])
                r = {"x": best_x, "stress_top10": None, "grip": None, "align": None,
                     "pressure": None, "min_pad_area": None, "width_face": None}
                print(f"  Env {i}: SYNTH FAILED -> default top-down grasp (fallback)")
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
        obs_bufs, act_bufs, rew_bufs, success, frame_bufs = execute_and_collect(
            worker, all_best_x, init_obs_batch, perception, action_config,
            record_video=rec_this_batch, priv_cfg=priv_cfg, dr_vec=dr_vec,
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
                                round(float(g["x"][6] * 1e3), 2)])
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
