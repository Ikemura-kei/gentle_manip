"""v4 collector — scripted grasp execution from the FEM grasp planner.

Per env: plan a grasp (smgrasp/finger_grasp_final.plan_finger_grasp), then execute a fixed
five-phase trajectory — approach (constant speed: duration = distance / APPROACH_SPEED),
settle, close to the PLANNED width, lift, hold — recording (obs, action) through the shared
PerceptionPipeline / ActionPipeline inversion. Open loop: the planner's width refine sets the
indentation; there is no surrogate closure scan and no force-triggered firm squeeze (both
removed 2026-09-05 to test whether the planner width alone lifts; see git history for them).
"""
from __future__ import annotations

import argparse
import atexit
import csv
import os
import dataclasses
import shutil
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

APPROACH_SPEED = 0.0024     # m/step — approach duration = |home→grasp| / speed (constant speed,
                            # like real teleop; v3.3 recipe value), clipped to the range below
APPROACH_DUR_MIN, APPROACH_DUR_MAX = 12, 130   # steps (low floor: a short non-home start keeps the speed)
START_MARGIN_M = 0.005       # sampled start TCP stays inside the action box shrunk by this
START_CLEAR_M  = 0.03        # lowest finger point >= object top + this at the start
START_W_MIN    = 0.02        # non-home starts: gripper width ~ U(this, open) -> re-opened at the start
GRIP_SPEED     = 0.0022      # m/step — gripper speed at a non-home start re-open (= real teleop close speed: 2.20 mm/step, 141 eps, 7 objects, 2026-09-01 set)
START_AIR_DZ   = 0.10        # in_air starts no higher than home + this
# Mid-approach OBJECT drag + scripted re-target (colleague's reactive collector, 2026-09-02/05):
DRAG_SPEED   = (0.12, 0.38)  # m/s lateral particle velocity, random direction
DRAG_HOLD    = 4             # steps the velocity is set -> a bounded slide (~2-5 cm)
DRAG_STEP    = (12, 45)      # recorded-step window in which the drag starts (approach only)
RX_SETTLE    = 16            # steps to let the object stop before re-targeting
RX_MIN_DISP  = 0.012         # m; a smaller xy displacement is not re-targeted
STANDOFF_MIN, STANDOFF_MAX = 0.04, 0.10   # m along the grasp's approach axis: the pre-grasp standoff sits at the
                             # start pose's OWN axial distance clamped to this range (no up-then-down for a start
                             # already near the axis); the final leg runs straight along the axis (open fingers
                             # straddle the object, no diagonal into it)
N_SETTLE      = 1           # hold at grasp pose before closing
N_DWELL       = 2            # hold at the final close width before lifting (MPM contact settles)
EXEC_EXTRA_CLOSE = 0.0008    # m — execute the PLANNED width minus this (total width; 19.0 mm planned -> 18.2 mm)
N_LIFT        = 66          # lift steps
N_HOLD        = 20           # hold at lift height, FROZEN 2026-09-06 (user): 20 steps = 5 action chunks of 'arrived,
                             # stay closed'. 60 (24 % of frames) overwrote re-open-on-empty-grasp; 12 trimmed to 4
                             # _trim_long_holds -> demos ended at the TOP of the lift (median 1 frame after max
                             # height): the policy had no data for 'stay closed at height' and re-opened mid-air
                             # (sim teaser 8/17 lift-then-release; real deploys). The trailing run is now KEPT.
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

PHASES = [
    ("approach", -1),              # home → grasp pose (lerp + slerp); duration is PER-ENV
                                   #   (distance / APPROACH_SPEED), set in execute_and_collect
    ("settle",   N_SETTLE),        # hold at grasp pose, gripper still open
    ("grasp",    -1),              # close to the PLANNED width; duration PER-ENV = (open - width) / GRIP_SPEED
    ("dwell",    N_DWELL),         # hold the final width before lifting
    ("lift",     N_LIFT),          # lift to LIFT_HEIGHT above the grasp point
    ("hold",     N_HOLD),          # hold at lift height (success eval window)
]
N_PHASES  = len(PHASES)
_GRASP_PHASE = [n for n, _ in PHASES].index("grasp")

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
                     keep=HELD_RUN_KEEP, eps=HELD_RUN_EPS, keep_tail=True):
    """Collapse runs of MORE THAN `max_run` consecutive near-identical actions down
    to `keep` frames (keep the first `keep` of the run, discard the rest). The SAME
    kept-index selection is applied to every list in `parallel_lists`.

    keep_tail=True (default since 2026-09-06): the run that ENDS THE EPISODE (the final
    hold at lift height) is never trimmed — it is the only supervision for "you have
    arrived: keep commanding this pose with the gripper closed". Trimming it to 4 frames
    left the policy with no post-arrival data (hold-tail deficit). Mid-episode runs (the
    grasp dwell, a stalled approach) are still collapsed as before.
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
            if run_len > max_run and not (keep_tail and t == T):
                keep_mask[run_start + keep: t] = False   # drop the tail of this run
            run_start = t

    idx = np.where(keep_mask)[0]
    trimmed_acts     = [act_list[i] for i in idx]
    trimmed_parallel = tuple(
        ([lst[i] for i in idx] if len(lst) == T else lst) for lst in parallel_lists
    )
    return (trimmed_acts,) + trimmed_parallel


def _sample_starts(rng, dr_cfg, home_p, home_q, grasp_p, grasp_q, obj_top, lo, hi, pad_geo):
    """Per-env start from dr_cfg.start_modes -> (pos (N,3), quat wxyz (N,4), width (N,), mode names).
    Non-home starts are clipped into [lo, hi], lifted so the fingers clear the object, part-closed."""
    names = list(dr_cfg.start_modes); w = np.array([dr_cfg.start_modes[k] for k in names], float)
    w = w / w.sum() if w.sum() > 0 else np.array([float(k == "home") for k in names])
    force = os.environ.get("GM_START_MODE")                       # dev: force one mode
    P, Q, W, modes = home_p.copy(), home_q.copy(), np.full(len(home_p), 0.08), []
    for i in range(len(P)):
        m = force or names[rng.choice(len(names), p=w)]
        rh, rg = Rot.from_quat(np.roll(home_q[i], -1)), Rot.from_quat(np.roll(grasp_q[i], -1))
        sl = Slerp([0.0, 1.0], Rot.concatenate([rh, rg]))
        if m == "in_air":
            p = rng.uniform(lo, np.minimum(hi, [hi[0], hi[1], home_p[i, 2] + START_AIR_DZ]))
            r = Rot.from_euler("xyz", np.radians(rng.uniform(-1, 1, 3) * [15, 10, 30])) * rh
        elif m == "above_object":
            p = np.r_[grasp_p[i, :2] + rng.uniform(-0.015, 0.015, 2), grasp_p[i, 2] + rng.uniform(0.04, 0.12)]
            r = sl(rng.uniform())
        elif m == "mid_approach":
            f = rng.uniform(0.3, 0.8); p = home_p[i] + f * (grasp_p[i] - home_p[i])
            p[:2] += rng.uniform(-0.01, 0.01, 2); r = sl(f)
        else:
            modes.append("home"); continue
        zmin = fg.finger_min_world_z(np.r_[p, r.as_euler("xyz"), 0.08], pad_geo)
        p[2] += max(0.0, obj_top[i] + START_CLEAR_M - zmin)
        P[i], Q[i], W[i] = np.clip(p, lo, hi), np.roll(r.as_quat(), 1), rng.uniform(START_W_MIN, 0.08); modes.append(m)
    return P, Q, W, modes


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
    start_pos=None,                # (N,3)/(N,4) per-env start pose (DR start_modes); None = home
    start_quat=None,
    start_grip=None,               # (N,) start gripper width; re-opened to 80 mm over START_REOPEN steps
    drags=None,                    # {env: (start_step, vel(3,))}: mid-approach object drag, then re-target
    obj_center0=None,              # (N,3) object centres at plan time (re-target measures displacement from these)
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
    # Commanded close width = the PLANNED width minus a small fixed safety margin (the planner's
    # FEM width refine already sets the indentation); no surrogate closure, no firm squeeze.
    width_plan = np.array([p[2] for p in poses], np.float32)
    width_cls  = np.maximum(0.004, width_plan - EXEC_EXTRA_CLOSE).astype(np.float32)

    home_pos  = (np.asarray(start_pos, np.float32) if start_pos is not None
                 else np.tile(worker.robot.home_pos[None].astype(np.float32), (num_envs, 1)))
    home_quat = (np.asarray(start_quat, np.float32) if start_quat is not None
                 else np.tile(worker.robot.home_quat[None].astype(np.float32), (num_envs, 1)))
    grip0 = (np.asarray(start_grip, np.float32) if start_grip is not None else np.full(num_envs, 0.08, np.float32))

    def _wxyz_to_rot(q): return Rot.from_quat([q[1], q[2], q[3], q[0]])
    slerps = [Slerp([0., 1.], Rot.concatenate([_wxyz_to_rot(home_quat[i]), _wxyz_to_rot(quat_b[i])]))
              for i in range(num_envs)]                      # per-env start orientation (start modes differ per env)
    # Per-env approach duration = straight-line distance / APPROACH_SPEED, so every demo
    # approaches at the same speed (a fixed duration made speed grow with spawn distance).
    reopen = np.ceil((width_open - grip0) / GRIP_SPEED).astype(np.int64)             # steps to re-open (0 at home)
    grasp_dur = np.maximum(np.ceil((width_open - width_cls) / GRIP_SPEED), 1).astype(np.int64)   # close at GRIP_SPEED
    # Two-leg approach: start -> pre-grasp standoff (STANDOFF along the grasp's approach axis, tool +z),
    # then straight along that axis into the grasp. `via` holds the standoff until it is reached.
    adir = Rot.from_quat(np.roll(quat_b, -1, axis=1)).apply([0.0, 0.0, 1.0]).astype(np.float32)   # (N,3)

    def _standoff(i, from_pos):
        """Pre-grasp point on the approach axis at the start's own axial distance, clamped [MIN, MAX]."""
        d = float(np.clip(-(from_pos - pos_b[i]) @ adir[i], STANDOFF_MIN, STANDOFF_MAX))
        return (pos_b[i] - d * adir[i]).astype(np.float32)
    via = [_standoff(i, home_pos[i]) for i in range(num_envs)]
    appr_dur = np.clip(np.maximum(np.round(np.linalg.norm(np.stack(via) - home_pos, axis=1) / APPROACH_SPEED), reopen),
                       APPROACH_DUR_MIN, APPROACH_DUR_MAX).astype(np.int64)      # leg 1 >= re-open


    def _env_target(i: int, phase_idx: int, phase_step: int):
        """(pos, quat_wxyz, grip) for env i at its OWN (phase_idx, phase_step)."""
        name, dur = PHASES[phase_idx]
        if name == "approach":
            alpha = (phase_step + 1) / int(appr_dur[i])
            goal = pos_b[i] if via[i] is None else via[i]        # via = hover above a re-targeted grasp
            pos = home_pos[i] + alpha * (goal - home_pos[i])
            xyzw = slerps[i](alpha).as_quat()
            quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], np.float32)
            grip = grip0[i] + min(1.0, (phase_step + 1) / max(int(reopen[i]), 1)) * (width_open[i] - grip0[i])
        elif name == "settle":
            pos, quat, grip = pos_b[i], quat_b[i], width_open[i]
        elif name == "grasp":
            alpha = (phase_step + 1) / int(grasp_dur[i])
            pos, quat = pos_b[i], quat_b[i]
            grip = width_open[i] + alpha * (width_cls[i] - width_open[i])
        elif name == "dwell":
            pos, quat, grip = pos_b[i], quat_b[i], width_cls[i]
        elif name == "lift":
            alpha = (phase_step + 1) / dur
            pos = pos_b[i] + alpha * (lift_b[i] - pos_b[i])
            quat, grip = quat_b[i], width_cls[i]
        else:  # "hold"
            pos, quat, grip = lift_b[i], quat_b[i], width_cls[i]
        return pos, quat, grip

    def _frozen_target(i: int):
        """Command for a DONE env — hold steady so it doesn't disturb the sim
        while other envs keep progressing (identical to its final hold target)."""
        return lift_b[i], quat_b[i], width_cls[i]

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
    prev_grip = grip0.copy()

    # Per-env FSM state
    phase_idx  = np.zeros(num_envs, dtype=np.int64)
    phase_step = np.zeros(num_envs, dtype=np.int64)
    ever_lifted = np.zeros(num_envs, bool); stress_max = np.zeros(num_envs)     # episode metrics
    drags = dict(drags or {}); retargeted = np.zeros(num_envs, bool); it = 0
    obj_ref = None if obj_center0 is None else np.asarray(obj_center0, np.float32).copy()

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
        for i, (t_fire, vel) in drags.items():                  # object drag: DRAG_HOLD steps, approach only
            if active[i] and phase_idx[i] == 0 and t_fire <= it < t_fire + DRAG_HOLD:
                worker.drag_object(i, vel)

        cur_obs_list, state = _step(cur_pos_arr, cur_quat_arr, cur_grip_arr, record_mask=active)
        it += 1
        # re-target: once the object has stopped sliding, shift the grasp by its xy displacement and
        # re-approach from the current pose via a hover above the new grasp (one drag per env)
        for i in [k for k, (t_fire, _) in drags.items() if active[k] and it >= t_fire + DRAG_HOLD + RX_SETTLE]:
            del drags[i]
            disp = np.asarray(state["object_center"][i], np.float32) - obj_ref[i]; disp[2] = 0.0
            if np.hypot(disp[0], disp[1]) < RX_MIN_DISP:
                continue
            pos_b[i] += disp; grasp_pos[i] += disp; lift_b[i] += disp; obj_ref[i] += disp
            home_pos[i], home_quat[i], grip0[i] = cur_pos_arr[i], cur_quat_arr[i], width_open[i]
            via[i] = _standoff(i, home_pos[i])                                    # standoff of the NEW grasp
            slerps[i] = Slerp([0., 1.], Rot.concatenate([_wxyz_to_rot(home_quat[i]), _wxyz_to_rot(quat_b[i])]))
            appr_dur[i] = int(np.clip(np.round(np.linalg.norm(via[i] - home_pos[i]) / APPROACH_SPEED), APPROACH_DUR_MIN, APPROACH_DUR_MAX))
            phase_idx[i] = 0; phase_step[i] = -1; retargeted[i] = True   # -1: incremented below
            print(f"    [drag] env {i}: object moved {np.round(1e3*disp[:2],1)} mm -> re-target via the new standoff")
        ever_lifted |= np.asarray(state["object_center"])[:, 2] > grasp_pos[:, 2] + LIFT_HEIGHT * 0.5
        if state.get("von_mises_stress") is not None:
            stress_max = np.maximum(stress_max, _stress_top10(state["von_mises_stress"]))

        # DEBUG: planned vs commanded vs measured width while closing (+ coupling force).
        _in_grasp = active & ((phase_idx == _GRASP_PHASE) | (phase_idx == _GRASP_PHASE + 1))
        if np.any(_in_grasp):
            _gw = np.asarray(state["gripper_width"], np.float32)
            _cf = state.get("contact_force")
            for i in np.where(_in_grasp)[0]:
                _ph = PHASES[int(phase_idx[i])]
                _du = int(grasp_dur[i]) if phase_idx[i] == _GRASP_PHASE else _ph[1]
                print(f"    [{_ph[0]}] env {i} step {int(phase_step[i]) + 1:2d}/{_du}: "
                      f"planned {width_plan[i]*1000:5.1f} mm  target {width_cls[i]*1000:5.1f} mm  cmd {cur_grip_arr[i]*1000:5.1f} mm  "
                      f"measured {_gw[i]*1000:5.1f} mm"
                      + (f"  force {float(_cf[i]):.2f} N" if _cf is not None else ""))

        # Advance phase state for envs that were active this step.
        # TODO(retry): this is where a robustness pass would check per-env
        # success/failure (e.g. at the "lift"/"hold" -> DONE boundary) and, on
        # failure, rewind phase_idx[i] to an earlier phase instead of advancing to
        # DONE — independent of every other env's state.
        phase_step[active] += 1
        # phase_idx may already be N_PHASES (done envs) — clip before indexing PHASES;
        # those entries are masked out by `active` anyway so the clipped value is unused.
        durations = np.array([PHASES[min(int(p), N_PHASES - 1)][1] for p in phase_idx])
        durations[phase_idx == 0] = appr_dur[phase_idx == 0]     # per-env approach length
        durations[phase_idx == _GRASP_PHASE] = grasp_dur[phase_idx == _GRASP_PHASE]   # per-env close length
        rolled_over = active & (phase_step >= durations)

        phase_idx[rolled_over]  += 1
        phase_step[rolled_over]  = 0
        for i in np.nonzero(rolled_over & (phase_idx == 1))[0]:          # standoff reached -> final leg along the axis
            if via[i] is not None:
                home_pos[i] = via[i]; via[i] = None; phase_idx[i] = 0
                grip0[i] = width_open[i]                                       # re-open finished in leg 1: hold open
                home_quat[i] = quat_b[i]                                        # orientation finished in leg 1
                slerps[i] = Slerp([0., 1.], Rot.concatenate([_wxyz_to_rot(quat_b[i]), _wxyz_to_rot(quat_b[i])]))
                appr_dur[i] = int(np.clip(np.round(np.linalg.norm(pos_b[i] - home_pos[i]) / APPROACH_SPEED), APPROACH_DUR_MIN, APPROACH_DUR_MAX))

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

    metrics = {"ever": ever_lifted, "retargeted": retargeted,
               "stress_frac": stress_max / float(yield_stress) if yield_stress else np.full(num_envs, np.nan)}
    return obs_bufs, act_bufs, rew_bufs, success, frame_bufs, metrics


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
                        "seed": args.seed,
                        "grip_speed": GRIP_SPEED, "approach_speed": APPROACH_SPEED, "n_dwell": N_DWELL, "exec_extra_close": EXEC_EXTRA_CLOSE, "n_lift": N_LIFT, "n_settle": N_SETTLE,
                        "start_modes": dict(dr_cfg.start_modes), "disturbance_prob": dr_cfg.disturbance_prob,
                        "held_run_max": HELD_RUN_MAX, "held_run_keep": HELD_RUN_KEEP,
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
    start_rng = np.random.default_rng(args.seed + 3_000_000)     # start modes + drags (own stream)

    # ── Build scene + worker (with per-scene SIZE+SHAPE DR) ──
    # Scene DR re-randomizes object geometry by REBUILDING the worker every N batches (GenesisWorker
    # has no in-process geometry re-randomize; a fresh build is the only path). Verified memory-stable
    # across rebuilds (gs.destroy reclaims each build). Geometry is shared across a batch's envs.
    nominal_spec   = spec
    do_scene_dr    = args.scene_dr_every > 0 and dr_cfg.has_scene_dr()
    deform_dir     = tempfile.mkdtemp(prefix="gm_synth_deform_") if do_scene_dr else None
    if deform_dir:                     # one deformed .obj per relaunch — never leave them behind
        atexit.register(shutil.rmtree, deform_dir, ignore_errors=True)

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
    dr_writer.writerow(["batch", "env", "success", "ever_success", "stress_max_frac", "obj_dx", "obj_dy",
                        "roll_deg", "pitch_deg", "yaw_deg", "flipped",
                        "home_dx", "home_dy", "home_dz", "start_mode", "start_x", "start_y", "start_z", "start_w", "drag", "retarget",
                        "scene_scale", "scene_bend_deg",
                        "mesh_variant",
                        "twist_deg", "taper", "rbf", "axis_scale", "axis_scale_ax",
                        "mat_E", "mat_nu", "mat_rho", "coup_friction",
                        "stress_Pa", "grip_N", "align", "pressure_Pa", "min_pad_mm2", "width_mm", "tilt_deg",
                        # dataset_idx: 0-based index into data.pkl["episodes"], or -1 if this
                        # attempt was NOT saved. Ported from v3 (2026-08-30). Without it the
                        # CSV<->dataset join is UNDERIVABLE: v4 logs every attempt, and a
                        # `success=1` row still may not be saved (n_episodes cap, or the
                        # fallback-grasp drop), so "the successes in order" is wrong.
                        "dataset_idx"])

    total_saved  = 0
    t_synth = t_exec = t_fem = 0.0            # profiling: synthesis / execution / FEM-build wall time
    all_ever, all_sfrac = [], []              # per-attempt metrics for stats.yaml
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

        # Object pose (SETTLED) for synthesis + priv obs, straight from read_state: rigid = get_quat,
        # soft = Kabsch best-fit rotation of the settled particle cloud (NOT the spawn euler, which the
        # object leaves once it falls/settles under gravity). Same key for both.
        obj_pos_all  = init_state["object_center"].astype(np.float64)  # (N, 3)
        obj_quat_all = np.asarray(init_state["object_quat"], np.float64)  # (N, 4) wxyz

        # Episode scene-DR vector [scale, bend_deg] for priv_object_dr_params (mirrors SimBackend).
        dr_vec = np.array([float(scene_dr.get("scale", 1.0)),
                           float(scene_dr.get("bend_deg", 0.0))], dtype=np.float32)
        def _initial_obs(state):                     # obs_0 for every env (re-read after a start teleport)
            ob = perception.process(_state_to_raw_obs(state))
            if priv_cfg is not None:
                ob.update(_privileged_obs_batch(
                    obj_pos_all, obj_quat_all, dr_vec, priv_cfg,
                    von_mises=state.get("von_mises_stress"), yield_stress=_mat_yield,
                    contact_force=state.get("contact_force")))
            return ob
        init_obs_batch = _initial_obs(init_state)

        # ── Per-env FEM gentleness grasp synthesis (v3) ──
        # Build the FEM ElasticObject ONCE for this batch's ACTUAL (DR shape+size) mesh — all envs share
        # it (scene-DR varies per relaunch, not per sub-env), so the expensive factorization is reused;
        # rebuild only when actual_mesh changes (a scene-DR relaunch). Then plan per-env settled pose.
        _tf = time.perf_counter()
        if actual_mesh != fem_mesh:
            fem_obj, fem_pad_geo, fem_meta = fg.build_grasp_fem(
                actual_mesh, voxel_div=GRASP_VOXEL_DIV, target_tets=GRASP_TARGET_TETS,
                use_gpu=True, nu=_fem_nu)
            fem_mesh = actual_mesh
            t_fem += time.perf_counter() - _tf
            print(f"  FEM: {fem_meta['tets']} tets, ndof={fem_meta['ndof']}, gpu={fem_meta['gpu']}")
            _viz_pad_geo = fem_pad_geo          # pad placement is known only once the FEM is built
        _ts = time.perf_counter()
        all_best_x = []
        synth_failed: set[int] = set()      # envs running the fallback grasp (never saved)
        all_grasp  = []                                          # per-env synthesis dict (for the grasp-pose viz)
        for i in range(n):
            cma_seed = int(cma_seed_rng.integers(1, 2**31 - 1))
            live_viz = (StageViewer(fem_obj, obj_pos_all[i], obj_quat_all[i], _viz_pad_geo,
                                    label=f"batch {batch_idx} env {i} — {args.experiment}")
                        if (args.dev_viz and i == 0) else None)    # step-through window: env 0 only
            r = fg.synthesize_grasp(fem_obj, fem_pad_geo, obj_pos_all[i], obj_quat_all[i],
                                    E=_mat_E, density=_mat_rho, mu=GRASP_MU,
                                    table_z=args.table_z, tcp_z_min=float(action_config.pos_min[2]),
                                    seed=cma_seed,
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
            # DEV: where is the MPM body relative to the FEM the planner used? Project the settled
            # particles and the FEM surface onto the planned closing axis; compare both spans with the
            # commanded width (TCP) and the inner-face gap (width + finger offsets).
            _parts = worker.particle_positions()
            if _parts is not None:
                _Rx = Rot.from_euler("xyz", np.asarray(best_x[3:6], float))
                _ax = _Rx.apply([0.0, 1.0, 0.0])                                 # closing axis, world
                _Rq = Rot.from_quat(np.roll(np.asarray(obj_quat_all[i], float), -1))   # wxyz -> xyzw
                _Vw = obj_pos_all[i] + _Rq.apply(fem_obj.verts)                  # FEM surface, world
                _pp = (_parts[i].astype(float) - obj_pos_all[i]) @ _ax
                _pf = (_Vw - obj_pos_all[i]) @ _ax
                _span_p, _span_f = float(_pp.max() - _pp.min()), float(_pf.max() - _pf.min())
                _wf = float(best_x[6]) + fem_pad_geo["eps_left"] + fem_pad_geo["eps_right"]
                _rows = [f"planned width (TCP) {1e3*best_x[6]:.1f} mm   inner-face gap {1e3*_wf:.1f} mm",
                         f"FEM span along closing axis      {1e3*_span_f:.1f} mm  -> indent/side {0.5e3*(_span_f-_wf):+.2f} mm",
                         f"particle span along closing axis {1e3*_span_p:.1f} mm  -> indent/side {0.5e3*(_span_p-_wf):+.2f} mm",
                         f"FEM - particles: {1e3*(_span_f-_span_p):+.2f} mm total ({0.5e3*(_span_f-_span_p):+.2f}/side)"]
                print(f"  [proj] env {i}: " + " | ".join(_rows))
                if live_viz is not None:
                    live_viz.show_particles(best_x, _parts[i], _rows)
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

        t_synth += time.perf_counter() - _ts
        # ── Start condition (dr.start_modes) + disturbance draw (dr.disturbance_prob) ──
        _tg = [_x_to_targets(x, 1) for x in all_best_x]
        _hp = np.tile(worker.robot.home_pos[None], (n, 1)).astype(np.float64)
        _hq = np.tile(worker.robot.home_quat[None], (n, 1)).astype(np.float64)
        start_pos, start_quat, start_grip, start_mode = _sample_starts(
            start_rng, dr_cfg, _hp, _hq, np.concatenate([t[0] for t in _tg]), np.concatenate([t[1] for t in _tg]),
            worker.particle_positions()[:, :, 2].max(1),
            np.asarray(action_config.pos_min) + START_MARGIN_M, np.asarray(action_config.pos_max) - START_MARGIN_M,
            fem_pad_geo)
        moved = np.array([m_ != "home" for m_ in start_mode])
        if moved.any():
            cur_p, cur_q = init_state["ee_pos"].astype(np.float64), init_state["ee_quat"].astype(np.float64)
            for _ in range(2):                                       # 2nd pass: IK misses fall back to home
                worker.set_ee_pose(np.where(moved[:, None], start_pos, cur_p).astype(np.float32),
                                   np.where(moved[:, None], start_quat, cur_q).astype(np.float32),
                                   gripper_width=np.where(moved, start_grip, 0.08).astype(np.float32))
                init_state = worker.read_state()
                miss = [i for i in np.nonzero(moved)[0]
                        if np.linalg.norm(init_state["ee_pos"][i] - start_pos[i]) > 0.005]
                if not miss:
                    break
                for i in miss:
                    print(f"  [start] env {i}: IK missed {start_mode[i]} by "
                          f"{1e3*np.linalg.norm(init_state['ee_pos'][i] - start_pos[i]):.1f} mm -> home")
                    start_mode[i], moved[i], start_pos[i], start_quat[i], start_grip[i] = "home", False, _hp[i], _hq[i], 0.08
            init_obs_batch = _initial_obs(init_state)
        drags = {}
        for i in range(n):
            if start_mode[i] == "above_object":          # never together (user, 2026-09-06): disturbance_prob is
                continue                                  # the probability CONDITIONAL on the start not being above_object
            if start_rng.random() < dr_cfg.disturbance_prob or os.environ.get("GM_DISTURB"):
                th, sp = start_rng.uniform(0, 2 * np.pi), start_rng.uniform(*DRAG_SPEED)
                drags[i] = (int(start_rng.integers(DRAG_STEP[0], DRAG_STEP[1] + 1)), np.array([np.cos(th) * sp, np.sin(th) * sp, 0.0], np.float32))
        drags0 = dict(drags)
        for i in range(n):
            print(f"  [start] env {i}: {start_mode[i]:12s} tcp={np.round(start_pos[i], 3)} w={1e3*start_grip[i]:.0f}mm"
                  + (f"  drag @ step {drags[i][0]} {np.round(drags[i][1][:2], 2)} m/s" if i in drags else ""))

        # ── Execute scripted trajectory + collect data ──
        # Record video only while under the first-N cap (args.record_video = N, or 10**9 for "all");
        # once N saved, stop RENDERING (no per-step RGB cost/disk for the rest of the run).
        rec_this_batch = args.record_video > 0 and total_saved < args.record_video
        print(f"  Executing …")
        try:
            _te = time.perf_counter()
            obs_bufs, act_bufs, rew_bufs, success, frame_bufs, metrics = execute_and_collect(
                worker, all_best_x, init_obs_batch, perception, action_config,
                record_video=rec_this_batch, priv_cfg=priv_cfg, dr_vec=dr_vec,
                yield_stress=_mat_yield, start_pos=start_pos, start_quat=start_quat, start_grip=start_grip,
                drags=drags, obj_center0=obj_pos_all)
            t_exec += time.perf_counter() - _te
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
        _DS_IDX_COL = -1                     # dataset_idx is the last column
        eul_deg = np.degrees(object_euler) if object_euler is not None else np.zeros((n, 3))
        for i in range(n):
            g = all_grasp[i]
            roll, pitch, yaw = eul_deg[i]
            flipped = int(abs(roll) > 140 or abs(pitch) > 140)                # a big-flip sample
            ho = home_offset[i] if home_offset is not None else (0.0, 0.0, 0.0)
            odxy = object_dxy[i] if object_dxy is not None else (0.0, 0.0)
            all_ever.append(bool(metrics["ever"][i])); all_sfrac.append(float(metrics["stress_frac"][i]))
            _dr_rows.append([batch_idx, i, int(bool(success[i])), int(bool(metrics["ever"][i])),
                                round(float(metrics["stress_frac"][i]), 3),
                                round(float(odxy[0]), 5), round(float(odxy[1]), 5),
                                round(float(roll), 1), round(float(pitch), 1), round(float(yaw), 1), flipped,
                                round(float(ho[0]), 5), round(float(ho[1]), 5), round(float(ho[2]), 5),
                                start_mode[i], *[round(float(v), 4) for v in start_pos[i]], round(float(start_grip[i]), 4),
                                int(i in drags0), int(bool(metrics["retargeted"][i])),
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
                                round(float(g.get("tilt_deg") or 0), 1),
                                None,                      # dataset_idx, stamped in the save loop
                                ])

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
    _sf = np.asarray(all_sfrac, float); _n = max(len(all_ever), 1)
    print(f"  Ever lifted      : {100*sum(all_ever)/_n:.1f}%   sub-yield: {100*np.mean(_sf < 1.0) if len(_sf) else 0:.1f}%"
          f"   max stress/yield mean {np.nanmean(_sf) if len(_sf) else 0:.2f} max {np.nanmax(_sf) if len(_sf) else 0:.2f}")
    print(f"  Time             : synth {t_synth:.0f}s  exec {t_exec:.0f}s  fem {t_fem:.0f}s  -> {elapsed/max(total_saved,1):.0f} s / saved episode")
    print(f"  Data             : {data_path}")

    stats = {
        "episodes_saved":  total_saved,
        "episodes_failed": total_failed,
        "episodes_fallback_dropped": total_fallback_dropped,
        "total_attempts":  total_attempts,
        "success_rate":    round(success_rate, 4),
        "ever_success_rate": round(sum(all_ever) / _n, 4),
        "sub_yield_frac":  round(float(np.mean(_sf < 1.0)) if len(_sf) else 0.0, 4),
        "stress_max_frac_mean": round(float(np.nanmean(_sf)) if len(_sf) else 0.0, 4),
        "stress_max_frac_max":  round(float(np.nanmax(_sf)) if len(_sf) else 0.0, 4),
        "elapsed_min":     round(elapsed / 60, 2),
        "synth_s_per_attempt": round(t_synth / _n, 2),
        "exec_s_per_attempt":  round(t_exec / _n, 2),
        "fem_build_s":     round(t_fem, 2),
        "sec_per_saved_episode": round(elapsed / max(total_saved, 1), 2),
    }
    stats_path = run_dir / "stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False)
    print(f"  Stats            : {stats_path}")

    worker.close()


if __name__ == "__main__":
    main()
