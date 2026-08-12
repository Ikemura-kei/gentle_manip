"""Autonomous sim demo collection via CMA-ES grasp synthesis.

Collects demonstrations by: reset with pose DR → per-env CMA-ES synthesis →
scripted execution, recording (obs, action, reward) tuples in exactly the
same format as demos/record.py.  Drop-in data source for DP3 training.

Config comes entirely from --experiment (task / obs / action / DR), mirroring
how the training server and eval harness are configured.

Usage:
    uv run --project envs/sim python grasp_synthesis/collect_demos_synth.py \\
        --experiment single_lift_mushroom_rigid \\
        --n-episodes 50 --n-envs 5
"""
from __future__ import annotations

import argparse
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

from synth_utils import (  # noqa: E402  (build_object_sdf/grasp_cost/run_cmaes imported inside _synth_worker)
    sample_finger_surface,
    FINGER_TO_TCP_Z,
)
from gentle_manip.actions.action_config import ActionConfig
from gentle_manip.experiment import Experiment
from gentle_manip.tasks.single_lift import SingleLiftTask
from gentle_manip.envs.genesis_worker import GenesisWorker
from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.perception.pipeline import PerceptionPipeline
from gentle_manip.domain_randomization.dr_config import DRConfig


# ── Constants (keep in sync with run_grasp_synth.py) ─────────────────────────

N_HOME_TO_PRE = 87          # home → grasp pose interpolation steps
N_SETTLE      = 1           # hold at grasp pose before closing
N_GRASP       = 39           # gripper close steps
N_LIFT        = 70          # lift steps
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

def _mesh_half_extent(mesh_path: str) -> np.ndarray:
    """(3,) AABB half-extent of the ACTUAL on-disk mesh CMA-ES will search against.

    gentle_manip extension (2026-08-13, 25-category size range): OBJ_SIZE was a
    fixed mushroom-scale (~3-4cm) constant, too tight for a 10-15cm chunk (beef,
    watermelon) and unnecessarily loose for a ~2-3cm shrimp/blueberry. Falls back
    to OBJ_SIZE on any load failure rather than crashing the whole batch.
    """
    try:
        import trimesh
        mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
        return np.asarray(mesh.extents, dtype=np.float64) / 2.0
    except Exception as e:
        print(f"  [warn] _mesh_half_extent({mesh_path}) failed ({e}); falling back to OBJ_SIZE")
        return OBJ_SIZE


def _synth_bounds(obj_pos: np.ndarray, obj_half_size: np.ndarray = OBJ_SIZE):
    """Compute CMA-ES search bounds from object position and its actual mesh size."""
    t_lb_xy = (obj_pos[:2] - 1.5 * obj_half_size[:2]).tolist()
    t_ub_xy = (obj_pos[:2] + 1.5 * obj_half_size[:2]).tolist()
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
    if nominal_mesh is None:
        # Primitive box (e.g. tofu) -- no mesh to deform/export, but `scale` still
        # applies directly to the box's own size (scene_builder.py: size = s *
        # entry.scale), so apply just that and skip the mesh pipeline entirely.
        new_scale = nominal_scale * float(shp.get("scale", 1.0))
        new_obj = dataclasses.replace(o, scale=new_scale)
        new_spec = dataclasses.replace(nominal_spec, objects=[new_obj, *nominal_spec.objects[1:]])
        scene_dr = {"scale": float(shp.get("scale", 1.0)), "bend_deg": 0.0}
        return new_spec, scene_dr

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


def _resolve_actual_mesh(spec_dr, deform_dir: Optional[str]) -> str:
    """Real, on-disk mesh path for the CMA-ES SDF (build_object_sdf needs a file).

    A mesh-based object already has one (`o.mesh_path`, possibly the DR-deformed
    export from `_apply_scene_dr`). A primitive-box object (e.g. tofu) has
    `mesh_path=None` by design -- there is no mesh to fall back on, so write a box
    .obj matching the ACTUAL simulated size (registry nominal size * the object's
    current scale, which already carries any scene-DR scale factor). Previously
    this fell back to a hardcoded mushroom mesh, silently making CMA-ES search for
    a grasp on the wrong geometry entirely for every box-primitive category.
    """
    o = spec_dr.objects[0]
    if o.mesh_path is not None:
        return o.mesh_path
    import trimesh
    from gentle_manip.assets.registry import get_object_def
    size = tuple(s * float(o.scale or 1.0) for s in get_object_def(o.name).size)
    box = trimesh.creation.box(extents=size)
    out_dir = deform_dir or tempfile.gettempdir()
    dst = Path(out_dir) / f"{o.name}_box_{size[0]:.4f}x{size[1]:.4f}x{size[2]:.4f}.obj"
    box.export(str(dst))
    return str(dst)


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
_SETTLE_IDX = [name for name, _ in PHASES].index("settle")  # regrasp rewinds here (idea #2)
_GRASP_IDX = [name for name, _ in PHASES].index("grasp")   # boundary the firm-check fires at
_LIFT_IDX  = [name for name, _ in PHASES].index("lift")    # disturbance-injection phase (idea #3)

# ── Robustness idea #2: lift-phase failure detection + regrasp ────────────────
# Checked ONCE per env, exactly at the moment it finishes "lift" (about to enter
# "hold") — read this env's just-measured object height. If the object didn't
# actually rise with the gripper (a genuine physical slip — either an
# under-firm grasp or idea #3's disturbance kick knocking it loose), REWIND
# phase_idx back to "settle" (reopens at the original grasp pose, gripper still
# open) instead of advancing to "hold", so the env re-runs settle->grasp->firm->
# lift for real: a genuine grasp-drop-retry example, not a fabricated one. Uses
# the SAME height fraction as the final success check for consistency. Bounded
# to MAX_REGRASP_RETRIES per env (default 1) so one persistently-failing env
# can't stall the batch or produce a runaway-length episode; a "regrasp in
# place" at the original CMA-ES pose is the cheap approximation (the object has
# usually only dropped a little, still under/near the fingers) — a full replan
# is deliberately NOT attempted here (see CLAUDE.md's retry brainstorm §2/§3).
REGRASP_HEIGHT_FRAC   = 0.5   # matches the final success check's fraction of LIFT_HEIGHT
MAX_REGRASP_RETRIES   = 1

# ── Robustness idea #1: force-based grasp firming ─────────────────────────────
# At the moment an env finishes "grasp" (checked ONCE, per env, at the grasp->firm
# boundary), read its just-measured contact force. If the grip came out weak (< 1N —
# CMA-ES's SDF-based cost is a geometric proxy, not a physics guarantee of a firm
# grip), close an EXTRA FIRM_EXTRA_CLOSE_M over the "firm" phase before lifting.
# Bounded to fire once per env (a one-way linear FSM check, not a retry loop).
FIRM_FORCE_THRESH_N  = 1.0     # below this measured contact force -> needs firming
FIRM_EXTRA_CLOSE_M   = 0.0025  # additional close distance (meters) if triggered
FIRM_MAX_CLOSE_FRACTION = 0.15 # ...but never more than this fraction of the grasp's
                               # own commanded width -- 2.5mm is negligible on a large
                               # object but can exceed a tiny object's own size (see
                               # collection diagnostic notes: blueberry ~9mm min-extent,
                               # 2.5mm unconditional extra close was ~28% of that).

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
    disturbance_prob: float = 0.0, # idea #3: DART-style disturbance injection (off by default)
    disturbance_max_m: float = 0.02,
    disturbance_rng=None,          # np.random.Generator; required if disturbance_prob > 0
    disturbance_phases: tuple = ("lift",),  # which phase(s) get an independent kick draw each
    enable_regrasp: bool = False,  # idea #2: lift-phase failure detection + regrasp (off by default)
    max_regrasp_retries: int = MAX_REGRASP_RETRIES,
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

    Idea #3 — DART-style disturbance injection (Laskey et al. 2017): our demos are
    100% clean scripted trajectories with zero recovery examples, a known BC
    brittleness source (small closed-loop errors compound because the policy has
    never seen how to correct them). For each phase named in `disturbance_phases`,
    with probability `disturbance_prob` one env gets an INDEPENDENT ONE-STEP random
    positional kick (magnitude up to `disturbance_max_m`, uniform direction)
    injected into the COMMANDED position at a random step during that phase. Every
    subsequent step still targets the ORIGINAL unperturbed plan (`_env_target` is
    untouched), so the recorded action at the following step is a genuine
    corrective delta back toward the intended trajectory — unlike post-hoc noise
    augmentation (which doesn't reflect real closed-loop-induced error and is known
    not to fix compounding error on its own), this bakes an actual recovery example
    into the demo itself.

    Which phase(s) to disturb matters: the original default (`("lift",)`) targets
    the failure mode "grasp succeeded, then dropped during/after lift." A teacher-
    forced diagnostic (comparing the TRAINED model's predicted actions against real
    held-out demo actions at matching states) found the model predicts gripper-
    closing direction/magnitude correctly ~90% of the time when SHOWN a real
    training-distribution state — i.e. the model itself is competent; the actual
    closed-loop failure (policy hovers near the object without closing, then closes
    empty while already retreating) is compounding POSITION error during
    approach/grasp carrying the robot to a state outside the training distribution
    right at the critical closing moment. `disturbance_phases=("grasp",)` (or
    `("approach","grasp","lift")` for full coverage) targets THIS failure mode
    directly — the demonstrator still targets the true grasp pose, so the recorded
    recovery example is "arrived slightly off target, corrected into alignment,
    then closed," which is exactly the missing skill.

    Idea #2 — lift-phase failure detection + regrasp: checked once per env, exactly
    when it finishes "lift". If the object's measured height didn't reach
    `REGRASP_HEIGHT_FRAC` of LIFT_HEIGHT above the grasp point (a genuine physical
    slip, not injected), rewind that env's phase back to "settle" (reopens at the
    original grasp pose) and let it re-run settle→grasp→firm→lift for real, up to
    `max_regrasp_retries` times. Complements idea #3: a disturbance-induced slip is
    exactly the kind of failure this can catch and recover from, but it also fires
    on ordinary weak-grasp drops with no disturbance involved.

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

    # Idea #3 setup: per-env, per-disturbed-phase draw (which envs, which step within
    # that phase, what offset) — precomputed once so the main loop only needs a cheap
    # dict lookup keyed by the CURRENT phase_idx. Each named phase gets an INDEPENDENT
    # bernoulli(disturbance_prob) draw, so an env can be disturbed in more than one
    # phase (or none) in the same episode.
    _phase_idx_by_name = {name: idx for idx, (name, _) in enumerate(PHASES)}
    _phase_dur_by_name = {name: dur for name, dur in PHASES}
    disturb_draws = {}   # phase_idx -> (mask, step, offset), each (N,) / (N,) / (N,3)
    if disturbance_prob > 0:
        assert disturbance_rng is not None, "disturbance_rng required when disturbance_prob > 0"
        for _phase_name in disturbance_phases:
            p_idx = _phase_idx_by_name[_phase_name]
            p_dur = _phase_dur_by_name[_phase_name]
            mask = disturbance_rng.random(num_envs) < disturbance_prob
            step = disturbance_rng.integers(0, p_dur, size=num_envs)
            _dirs = disturbance_rng.normal(size=(num_envs, 3))
            _dirs /= np.linalg.norm(_dirs, axis=1, keepdims=True) + 1e-9
            offset = (_dirs * disturbance_rng.uniform(0.0, disturbance_max_m, size=num_envs)[:, None]
                     ).astype(np.float32)
            disturb_draws[p_idx] = (mask, step, offset)

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
            # Idea #1: reaching this phase AT ALL means the grasp->firm check (main
            # loop) found this env's grip too weak — envs that were already fine skip
            # "firm" entirely (phase_idx jumps straight to "lift", no artificial no-op
            # steps to generate-then-trim). So here it's always a real extra close.
            #
            # Cap the extra close to a FRACTION of this env's own grasp width, not a
            # fixed 2.5mm for every object: 2.5mm is negligible on a mushroom/apple
            # but is 20-30% of a blueberry's ~9mm size -- fixed-magnitude closing
            # after an already-computed grasp risks crushing/ejecting small objects.
            # This matters MORE than usual right now because contact_force always
            # reads 0 (a pinned Genesis bug, see genesis_worker.py), so the grasp->
            # firm check below can never observe "already fine" and skip firm -- every
            # rigid grasp goes through this phase, every time.
            pos, quat = pos_b[i], quat_b[i]
            alpha = (phase_step + 1) / dur
            close_amount = min(FIRM_EXTRA_CLOSE_M, FIRM_MAX_CLOSE_FRACTION * width_cls[i])
            grip_target[i] = max(0.0, width_cls[i] - alpha * close_amount)
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
    regrasp_count = np.zeros(num_envs, dtype=np.int64)   # idea #2: retries used so far

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
            next_obs_batch.update(_privileged_obs_batch(
                state["object_center"], state.get("object_quat"), dr_vec, priv_cfg,
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
                # Idea #3: one-step positional kick at a random point during whichever
                # phase(s) this env drew for. _env_target is untouched (still returns the
                # ORIGINAL plan), so every step AFTER this one keeps targeting the
                # unperturbed trajectory — the next recorded action is therefore a genuine
                # corrective delta back toward the intended path, not just noise with no
                # recovery signal.
                _draw = disturb_draws.get(int(phase_idx[i]))
                if _draw is not None:
                    mask, step, offset = _draw
                    if mask[i] and phase_step[i] == step[i]:
                        pos = pos + offset[i]
            else:
                pos, quat, grip = _frozen_target(i)
            cur_pos_arr[i], cur_quat_arr[i], cur_grip_arr[i] = pos, quat, grip

        cur_obs_list, state = _step(cur_pos_arr, cur_quat_arr, cur_grip_arr, record_mask=active)

        # Advance phase state for envs that were active this step. Idea #2 (below)
        # is exactly the robustness pass this used to be a TODO for: on a detected
        # lift-phase failure, rewind phase_idx[i] to an earlier phase instead of
        # advancing to "hold" — independent of every other env's state.
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
            if cf is not None:
                for i in np.where(leaving_grasp)[0]:
                    if cf[i] < FIRM_FORCE_THRESH_N:
                        print(f"    [firm] env {i}: grip force {cf[i]:.2f}N < "
                              f"{FIRM_FORCE_THRESH_N}N -> closing {FIRM_EXTRA_CLOSE_M*1000:.1f}mm more")
                    else:
                        advance[i] = 2   # grip already fine -> skip "firm", straight to "lift"

        # Idea #2: lift-phase failure detection + regrasp. Check ONCE, exactly at
        # the moment an env finishes "lift" (about to enter "hold") — read this
        # env's just-measured object height. If it didn't rise far enough (a
        # genuine slip — under-firm grasp, or idea #3's disturbance knocking it
        # loose), REWIND to "settle" (reopens at the original grasp pose) instead
        # of advancing to "hold", bounded to max_regrasp_retries per env.
        leaving_lift = rolled_over & (phase_idx == _LIFT_IDX)
        if enable_regrasp and np.any(leaving_lift):
            obj_z_now = state["object_center"][:, 2]
            for i in np.where(leaving_lift)[0]:
                expected_z = grasp_pos[i, 2] + LIFT_HEIGHT * REGRASP_HEIGHT_FRAC
                if obj_z_now[i] < expected_z and regrasp_count[i] < max_regrasp_retries:
                    regrasp_count[i] += 1
                    print(f"    [regrasp] env {i}: object z={obj_z_now[i]:.4f} < "
                          f"expected {expected_z:.4f} after lift -> retry "
                          f"{regrasp_count[i]}/{max_regrasp_retries}, rewinding to settle")
                    advance[i] = _SETTLE_IDX - _LIFT_IDX   # negative -> rewind phase_idx
                    grip_target[i] = width_cls[i]          # reset for a clean re-firm check
                    # KNOWN ISSUE (deprioritized — specialist SR is the current focus):
                    # "settle" commands pos_b[i] directly with no interpolation from
                    # where the env actually is (still near lift height at this point),
                    # so the PD controller sees a large one-step target jump and drives
                    # hard toward it — visually a fast, uncontrolled-looking motion that
                    # can clip the table. A real fix needs a short interpolated
                    # re-approach from the CURRENT ee position (e.g. reuse the
                    # "approach" phase's slerp/lerp machinery with a per-env dynamic
                    # start pose instead of a fixed home_pos) rather than a hard target
                    # snap. Left as-is for now; revisit only if regrasp data quality
                    # becomes the priority again.

        phase_idx[rolled_over]  += advance[rolled_over]
        phase_step[rolled_over]  = 0

    # Success check from final object position. `state["object_center"]` (already
    # used for the privileged obs / regrasp checks above) works for BOTH rigid and
    # MPM (soft-body) entities -- gentle_manip extension (2026-08-12) switched off
    # `worker.handle.objects[0].get_pos()` (rigid-only; MPM has no `get_pos`).
    obj_z   = state["object_center"][:, 2]
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
    """Merge shard_*.pkl into data.pkl. If a data.pkl ALREADY exists (resume of a
    previously-completed run being topped up further -- gentle_manip 25-category
    speed pass, 2026-08-13), its episodes are folded in first so resuming never
    loses prior work, regardless of whether the previous invocation crashed before
    or after its own merge."""
    shards = sorted(run_dir.glob("shard_*.pkl"))
    prior_path = run_dir / "data.pkl"
    if not shards and not prior_path.exists():
        return None
    all_eps: List[dict] = []
    meta: Optional[dict] = None
    if prior_path.exists():
        with open(prior_path, "rb") as f:
            d = pickle.load(f)
        meta = dict(d["meta"])
        all_eps.extend(d["episodes"])
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


def _count_existing_episodes(run_dir: Path) -> Tuple[int, int]:
    """(episodes already saved, next free shard index) for --resume-dir.

    Sums data.pkl's episode count (if a prior complete merge exists) plus every
    leftover shard_*.pkl's episode count (if the previous invocation was killed
    before its final merge) -- covers both crash points without needing to run
    an actual merge just to count."""
    total = 0
    data_path = run_dir / "data.pkl"
    if data_path.exists():
        with open(data_path, "rb") as f:
            d = pickle.load(f)
        total += len(d["episodes"])
    max_idx = -1
    for p in sorted(run_dir.glob("shard_*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f)
        total += len(d["episodes"])
        idx = int(p.stem.split("_")[1])
        max_idx = max(max_idx, idx)
    return total, max_idx + 1


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
    p.add_argument("--resume-dir", type=Path, default=None,
                   help="crash recovery (gentle_manip 25-category speed pass, 2026-08-13): "
                        "continue an existing run dir instead of creating a fresh one. Counts "
                        "episodes already in data.pkl (if merged) plus any leftover un-merged "
                        "shard_*.pkl (if the previous invocation was killed before merging), "
                        "and only collects the remainder toward --n-episodes. Must be an "
                        "existing directory previously created by this script.")
    p.add_argument("--description", type=str, default="",
                   help="free-text annotation saved in config.yaml")

    p.add_argument("--n-episodes", type=int, default=50,
                   help="total successful episodes to collect")
    p.add_argument("--n-envs",     type=int, default=5,
                   help="parallel envs per batch")
    p.add_argument("--maxfevals",  type=int, default=1145,
                   help="CMA-ES function evaluations per env per batch")
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
    p.add_argument("--record-video", action="store_true",
                   help="write per-episode mp4 videos to <out-dir>/videos/ (slower)")
    p.add_argument("--disturbance-prob", type=float, default=0.0,
                   help="DART-style recovery-demo injection (idea #3, off by default): "
                        "probability a given env gets a one-step positional kick during "
                        "'lift', so the demo contains a genuine corrective action back "
                        "toward the plan instead of only ever-clean trajectories. "
                        "See execute_and_collect's docstring.")
    p.add_argument("--disturbance-max-m", type=float, default=0.02,
                   help="max magnitude (m) of the idea #3 positional kick")
    p.add_argument("--disturbance-phase", nargs="+", default=["lift"],
                   choices=["approach", "settle", "grasp", "firm", "lift", "hold"],
                   help="which phase(s) get an independent idea #3 disturbance draw "
                        "(default: lift only, the original behavior). Each phase is "
                        "an independent bernoulli(disturbance_prob) draw per env.")
    p.add_argument("--enable-regrasp-retry", action="store_true",
                   help="lift-phase failure detection + regrasp (idea #2, off by default): "
                        "if an env's object didn't rise far enough by the end of 'lift', "
                        "rewind to 'settle' and re-run settle->grasp->firm->lift for a "
                        "genuine grasp-drop-retry example. See execute_and_collect's docstring.")
    p.add_argument("--max-regrasp-retries", type=int, default=MAX_REGRASP_RETRIES,
                   help="cap on regrasp attempts per env (bounds episode length)")
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
    # Separate stream again for idea #3's disturbance draws (which env/step/offset).
    disturbance_rng = np.random.default_rng(args.seed + 3_000_000)

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
        return w, sdr, _resolve_actual_mesh(spec_dr, deform_dir)

    worker, scene_dr, actual_mesh = _make_worker()
    obj_half_size = _mesh_half_extent(actual_mesh)
    if do_scene_dr:
        print(f"  scene DR ON (every {args.scene_dr_every} batch(es)) — deformed meshes → {deform_dir}")

    # ── Everything below is CPU-only (trimesh BVH + scipy CMA-ES) ──
    # Mesh path and finger geometry are gathered from Genesis, then handed off to
    # subprocess workers that each build their own SDF — no CUDA involved.
    left_pts  = sample_finger_surface(LEFT_FINGER,  n=300)
    right_pts = sample_finger_surface(RIGHT_FINGER, n=300)

    # Process pool reused across all batches (N workers, one per env).
    executor = ProcessPoolExecutor(max_workers=args.n_envs)
    print(f"  Mesh: {Path(actual_mesh).name}")

    # ── Output dir + config snapshot ──
    if args.resume_dir is not None:
        if not args.resume_dir.is_dir():
            raise SystemExit(f"--resume-dir {args.resume_dir} does not exist")
        run_dir = args.resume_dir
        total_saved, shard_idx = _count_existing_episodes(run_dir)
        print(f"  RESUME → {run_dir.resolve()}  ({total_saved} episode(s) already saved, "
              f"continuing at shard {shard_idx})")
    else:
        run_dir     = _make_run_dir(args.out_dir, task_name)
        total_saved = 0
        shard_idx   = 0
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(collection_config, f, sort_keys=False)
    print(f"  Config → {cfg_path.resolve()}")
    print(f"  Data   → {run_dir.resolve()}/data.pkl  (shards flushed every {args.shard_size} ep)")

    total_failed = 0
    batch_idx   = 0
    shard_buf:  List[dict] = []
    t0 = time.time()

    while total_saved < args.n_episodes:
        batch_idx += 1
        n = args.n_envs

        # ── Scene DR: rebuild the worker with fresh object SIZE+SHAPE every N batches ──
        if do_scene_dr and batch_idx > 1 and (batch_idx - 1) % args.scene_dr_every == 0:
            worker.close()
            worker, scene_dr, actual_mesh = _make_worker()
            obj_half_size = _mesh_half_extent(actual_mesh)

        print(f"\n── Batch {batch_idx}  [{total_saved}/{args.n_episodes} saved]"
              + (f"  scale={scene_dr['scale']:.3f} bend={scene_dr['bend_deg']:+.1f}°"
                 if do_scene_dr else "") + " ──")

        # ── Reset with per-env pose DR (ranges from experiment DR config) ──
        object_dxy   = dr_cfg.sample_object_dxy(rng, n)
        object_euler = dr_cfg.sample_object_euler(rng, n)
        home_offset  = dr_cfg.sample_home_offset(rng, n)   # per-env arm-home jitter (sim-only DR)
        worker.reset(object_dxy=object_dxy, object_euler=object_euler, home_offset=home_offset)

        # Extra settling until object velocity is small. MPM (soft-body) entities have
        # no single rigid-body velocity (particle-based), but DO expose per-particle
        # velocities via get_particles_vel() -- gentle_manip extension (2026-08-13,
        # 25-category speed pass): use max-abs particle velocity as the MPM settling
        # signal instead of burning the full fixed budget unconditionally every batch.
        # min_steps guards against a spurious near-zero reading before the drop/impact
        # has even happened. Threshold is looser than rigid's 0.003 m/s -- MPM carries
        # small residual per-particle jitter even at rest (thermal-like elastic noise),
        # so a rigid-tight threshold would rarely trip and silently degrade to the old
        # fixed-600 behavior.
        obj = worker.handle.objects[0]
        mpm_settle = hasattr(obj, "get_particles_vel") and not hasattr(obj, "get_vel")
        min_steps = 150 if mpm_settle else 0
        for _step_i in range(600):
            worker.handle.scene.step()
            if mpm_settle:
                if _step_i + 1 < min_steps:
                    continue
                pvel = np.abs(_np(obj.get_particles_vel())).max()
                if pvel < 0.01:
                    break
                continue
            if not hasattr(obj, "get_vel"):
                continue    # neither rigid nor MPM velocity API -- run the full budget
            lin = np.abs(_np(obj.get_vel())).max()
            ang = np.abs(_np(obj.get_ang())).max()
            if lin < 0.003 and ang < 0.01:
                break

        # Read initial state (depth rendered; this is obs_0 for every env)
        init_state     = worker.read_state()
        raw_init       = _state_to_raw_obs(init_state)
        init_obs_batch = perception.process(raw_init)
        # Episode scene-DR vector [scale, bend_deg] for priv_object_dr_params (mirrors
        # SimBackend._episode_dr_vec). scene_dr may re-randomize this per batch (below).
        dr_vec = np.array([float(scene_dr.get("scale", 1.0)),
                           float(scene_dr.get("bend_deg", 0.0))], dtype=np.float32)
        if priv_cfg is not None:
            init_obs_batch.update(_privileged_obs_batch(
                init_state["object_center"], init_state.get("object_quat"), dr_vec, priv_cfg,
                contact_force=init_state.get("contact_force")))

        obj_pos_all  = init_state["object_center"].astype(np.float64)  # (N, 3)
        if hasattr(obj, "get_quat"):
            obj_quat_all = _np(obj.get_quat()).astype(np.float64)       # (N, 4) wxyz
        else:
            # MPM (soft-body): no live rigid-orientation query -- gentle_manip
            # extension (2026-08-12). The particles are generated already rotated
            # by `object_euler` (sampled above, fed into worker.reset()), so that
            # sampled euler IS the object's orientation -- there's no separate
            # "settled" rigid transform to drift from it the way a rigid body has.
            if object_euler is not None:
                q_xyzw = Rot.from_euler("xyz", object_euler).as_quat()  # (N, 4) xyzw
                obj_quat_all = np.concatenate(
                    [q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1).astype(np.float64)
            else:
                obj_quat_all = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

        # ── Per-env CMA-ES grasp synthesis (parallel — one subprocess per env) ──
        payloads = []
        for i in range(n):
            lb, ub = _synth_bounds(obj_pos_all[i], obj_half_size)
            cma_seed = int(cma_seed_rng.integers(1, 2**31 - 1))
            payloads.append((actual_mesh, obj_pos_all[i], obj_quat_all[i],
                             left_pts, right_pts, args.maxfevals, lb, ub,
                             str(run_dir / "cmaes_logs"), cma_seed))
        futures = [executor.submit(_synth_worker, p) for p in payloads]
        all_best_x = []
        for i, fut in enumerate(futures):
            best_x, score = fut.result()
            all_best_x.append(best_x)
            print(f"  Env {i}: cost={score:.4f}  tcp={best_x[:3].round(4)}"
                  f"  w={best_x[6]*1e3:.1f} mm")

        # ── Execute scripted trajectory + collect data ──
        print(f"  Executing …")
        obs_bufs, act_bufs, rew_bufs, success, frame_bufs = execute_and_collect(
            worker, all_best_x, init_obs_batch, perception, action_config,
            record_video=args.record_video, priv_cfg=priv_cfg, dr_vec=dr_vec,
            disturbance_prob=args.disturbance_prob, disturbance_max_m=args.disturbance_max_m,
            disturbance_rng=disturbance_rng, disturbance_phases=tuple(args.disturbance_phase),
            enable_regrasp=args.enable_regrasp_retry, max_regrasp_retries=args.max_regrasp_retries,
        )
        print(f"  Success: {success.tolist()}")

        # ── Package and shard successful (or all) episodes ──
        for i in range(n):
            # Always save failure video (if recording) before skipping demo data.
            if not success[i]:
                total_failed += 1
                if args.record_video and frame_bufs[i]:
                    vid_dir = run_dir / "videos_failed"
                    vid_dir.mkdir(exist_ok=True)
                    vid_path = vid_dir / f"fail{total_failed:04d}_b{batch_idx}_env{i}.mp4"
                    imageio.mimwrite(str(vid_path), frame_bufs[i], fps=round(rate_hz), quality=8)
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

            if args.record_video and frame_bufs[i]:
                vid_dir = run_dir / "videos"
                vid_dir.mkdir(exist_ok=True)
                # success[i] is True on the normal path; this branch is also reached
                # for a KEPT FAILURE (--keep-failures, success[i]=False fell through
                # instead of `continue`-ing above) — label the suffix accordingly so
                # the filename doesn't lie about the outcome (a real bug: every
                # --keep-failures episode was previously named "..._success.mp4"
                # regardless of its actual success[i] flag).
                outcome = "success" if success[i] else "keptfail"
                vid_path = vid_dir / f"ep{total_saved:04d}_env{i}_{outcome}.mp4"
                imageio.mimwrite(str(vid_path), frame_bufs[i], fps=round(rate_hz), quality=8)
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

    executor.shutdown(wait=False)
    worker.close()


if __name__ == "__main__":
    main()
