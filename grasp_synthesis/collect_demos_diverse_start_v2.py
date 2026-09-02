"""OmniReset-inspired diverse-start demo collection -- v2 (2026-08-28).

Fork of collect_demos_diverse_start.py. IDENTICAL core idea (single-attempt,
always-successful, top-down grasp+lift demos whose STARTING configuration densely
covers post-failure states, so a BC policy recovers from a bad first attempt on
its own -- no explicit retry FSM). v1 stays untouched as the baseline.

v2 adds, per the user's requirements for the rigid-banana surrogate campaign:

  1. RGB VIDEO of every collected episode (`--record-video [N]`). Frames are
     rendered ONLY during the RECORDED steps (never the unrecorded pre-roll), one
     frame per physics step, and the SAME early-success trim indices are applied
     to the frame buffer -- so a video shows EXACTLY the trajectory that lands in
     data.pkl, with NO padded / duplicated / frozen frames. -> <run>/videos/
     epNNNN_envM.mp4

  2. CONTINUOUS end-effector start coverage (`--start-modes` = family weights,
     default "sweep:0.62,above:0.15,ground:0.12,air:0.11"). The dominant "sweep"
     family draws t~U(0,1) and starts the EE that fraction of the way from home to
     the grasp target (lateral + orientation jitter grow with t) -- so a run
     densely covers the ENTIRE home->object corridor, not a few fixed poses. Three
     smaller off-corridor families cover post-failure states off the direct path:
       home        -- the classic fixed home start (no pre-roll)
       near_object -- just off the real grasp point (small random 3D offset),
                      gripper open: "aimed here, object moved / missed"
       near_ground -- low, near table height, laterally offset from the object:
                      "descended to the wrong spot"
       mid_air     -- a random point in a workspace box around home: "wandered off"
     The RECORDED approach then redirects from that pose to the true grasp target
     -- one continuous successful trajectory, taught as normal BC data.

  3. TOP-DOWN grasp clamp (`--top-down`, on by default): the CMA-ES search roll
     bound is tightened from +-0.49*pi to +-0.16*pi and pitch to +-0.10*pi so
     every synthesized grasp is a genuine top-down grip (the user's "all
     successful grasped from top down").

  4. The fragile rigid-only extra-settle loop from v1 (called obj.get_ang(),
     which does not exist on every genesis entity version) is dropped -- the
     worker's own reset() settling (settle_steps / settle_max_steps /
     settle_vel_thresh from the task cfg) already handles rigid + soft.

Usage:
    env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl \
      uv run --project envs/sim_arrhenius --no-sync python \
        grasp_synthesis/collect_demos_diverse_start_v2.py \
          --experiment single_lift_banana_rigid_diverse \
          --n-episodes 500 --n-envs 8 --record-video --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

ROOT = Path(__file__).resolve().parent.parent
GRASP_DIR = ROOT / "grasp_synthesis"
for _p in (str(ROOT), str(GRASP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import collect_demos_synth as v1  # noqa: E402  reuse constants + helpers unchanged
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from gentle_manip.actions.action_config import ActionConfig  # noqa: E402
from gentle_manip.envs.genesis_worker import GenesisWorker  # noqa: E402
from gentle_manip.experiment import Experiment  # noqa: E402
from gentle_manip.perception.pipeline import PerceptionPipeline  # noqa: E402
from gentle_manip.domain_randomization.dr_config import DRConfig  # noqa: E402
from gentle_manip.tasks.single_lift import SingleLiftTask  # noqa: E402
from gentle_manip.robot import xarm7_config as XC  # noqa: E402

try:
    import imageio.v2 as imageio  # noqa: E402
except Exception:  # pragma: no cover
    import imageio  # noqa: E402

# ── Early-success-termination constants (mirrors v1 / v3 FAST_SUCCESS_* convention) ──
SUCCESS_HEIGHT     = 0.10
SUCCESS_HOLD_STEPS = 10
TRIM_MARGIN_STEPS  = 5

# ── Pre-roll ──
N_PREROLL_STEPS = 60           # unrecorded home -> start-pose steps (all non-home modes)

# EE start pose is sampled CONTINUOUSLY, not from a handful of discrete poses.
# The dominant family "sweep" places the EE at a UNIFORM RANDOM fraction t~U(0,1)
# of the way from home to the grasp target -- so across a run the start pose covers
# the whole home->object corridor densely, with lateral + orientation jitter that
# GROWS toward the object (a near-object start is noisier than a home start). Three
# smaller off-corridor families cover post-failure states that are NOT on the direct
# path: "above" (over the object, wrong height/rotation), "ground" (descended low to
# the wrong spot), "air" (wandered off). --start-modes sets the family weights;
# legacy keys (home/near_object/mid_approach) fold into "sweep".
#   "strict_home"  -- EE starts EXACTLY at the fixed home pose, full approach
#     recorded (no pre-roll). This is the NON-REGRASPABLE BASELINE distribution
#     (`--start-modes strict_home:1.0`): every demo is one clean home->grasp with
#     no diverse start coverage, so a policy trained on it has never seen a
#     post-failure state.
START_FAMILIES = ("sweep", "above", "ground", "air", "failed", "strict_home")
_LEGACY_FAMILY = {"home": "sweep", "near_object": "sweep", "mid_approach": "sweep",
                  "above_object": "above", "near_ground": "ground", "mid_air": "air", "failed_grasp": "failed"}
DEFAULT_START_MODES = "sweep:0.44,failed:0.30,above:0.10,ground:0.09,air:0.07"


def _wxyz_to_rot(q): return Rot.from_quat([q[1], q[2], q[3], q[0]])


def _rot_to_wxyz(r: Rot) -> np.ndarray:
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z], np.float32)


def _clamp_ws(pos: np.ndarray) -> np.ndarray:
    lo = np.asarray(XC.EE_BOUNDS_MIN, np.float32)
    hi = np.asarray(XC.EE_BOUNDS_MAX, np.float32)
    return np.minimum(np.maximum(pos, lo + 1e-3), hi - 1e-3)


def _synth_bounds_topdown(obj_pos: np.ndarray, top_down: bool, obj_size=None):
    """v1._synth_bounds with the roll/pitch bound tightened for a top-down grip, and
    (when obj_size is given: the drawn object's full AABB extents in m) the xy search
    box + close-width bound rescaled to the ACTUAL object. v1's hardcoded OBJ_SIZE is
    mushroom-scale (~10 cm box), which makes CMA-ES search a +-7 cm box for a 2 cm
    fruit and return a straddle width wider than the object (no contact)."""
    lb, ub = v1._synth_bounds(obj_pos)
    if obj_size is not None:
        s = np.asarray(obj_size, float)
        hx, hy = 0.5 * s[0], 0.5 * s[1]
        lb[0], ub[0] = float(obj_pos[0] - 2.0 * hx), float(obj_pos[0] + 2.0 * hx)
        lb[1], ub[1] = float(obj_pos[1] - 2.0 * hy), float(obj_pos[1] + 2.0 * hy)
        # close width in [3 mm, 1.25 x largest-lateral-dim] -- covers a firm grip
        # without letting CMA park at a no-contact straddle.
        wmax = float(min(0.08, 1.25 * max(s[0], s[1])))
        lb[6], ub[6] = 0.003, max(0.012, wmax)
    if top_down:
        # x = [tx, ty, tz, roll, pitch, yaw, width]. ONLY clamp pitch (approach tilt
        # away from straight-down); roll is left at the full default range because the
        # actual downward TCP orientation in this "xyz" euler convention sits near
        # roll = +-pi, NOT near 0 -- clamping roll to a small band (the original bug)
        # forced a sideways grip and CMA-ES could not find contact (cost 42 vs 0.001).
        # The sky/ground penalties in grasp_cost already keep the approach top-ish.
        lb[4], ub[4] = -0.18 * np.pi, 0.18 * np.pi
    return lb, ub


def _parse_start_modes(spec: str) -> Tuple[np.ndarray, np.ndarray]:
    """"sweep:0.62,above:0.15,..." -> (families, probs). Legacy per-pose keys
    (home/near_object/mid_approach/above_object/near_ground/mid_air) are accepted
    and summed into their family."""
    w = {fam: 0.0 for fam in START_FAMILIES}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, val = tok.partition(":")
        name = name.strip()
        fam = name if name in w else _LEGACY_FAMILY.get(name)
        if fam is None:
            raise ValueError(f"unknown start family {name!r}; valid: {START_FAMILIES} "
                             f"(or legacy {list(_LEGACY_FAMILY)})")
        w[fam] += float(val) if val else 1.0
    probs = np.array([w[fam] for fam in START_FAMILIES], np.float64)
    if probs.sum() <= 0:
        raise ValueError("start-modes weights sum to 0")
    return np.array(START_FAMILIES), probs / probs.sum()


def _sample_start(fam: str, home_p, home_q, grasp_p, grasp_q, obj_p, rng):
    """(start_pos(3), start_quat wxyz(4), n_recorded_approach:int, label:str,
    recover_from(3) or None). recover_from is set ONLY for the "failed" family:
    a pose AT the object with the gripper CLOSED (a just-missed first grasp) that
    the recorded trajectory OPENS + backs away from before the real approach --
    the explicit reopen-and-retry demonstration a regrasp policy needs.
    Gripper open (except "failed"), orientation ~ grasp orientation."""
    N = v1.N_HOME_TO_PRE

    def _slerp(qa, qb, t):
        return _rot_to_wxyz(Slerp([0., 1.], Rot.concatenate(
            [_wxyz_to_rot(qa), _wxyz_to_rot(qb)]))(float(np.clip(t, 0, 1))))

    if fam == "strict_home":
        # non-regraspable baseline: start exactly at home, record the full approach,
        # no pre-roll. Label "home" folds into the existing no-pre-roll path.
        return (np.asarray(home_p, np.float32), np.asarray(home_q, np.float32),
                N, "home", None)

    if fam == "sweep":
        t = float(rng.random())                       # UNIFORM along home->object
        lat = (0.008 + 0.050 * t)                      # jitter grows toward the object
        off = rng.normal(0, 1, 3).astype(np.float32) * np.array([lat, lat, 0.5 * lat], np.float32)
        p = home_p + t * (grasp_p - home_p) + off
        base = _wxyz_to_rot(_slerp(home_q, grasp_q, t))
        jit = Rot.from_rotvec(rng.normal(0, 1, 3).astype(np.float32) * (0.04 + 0.22 * t))
        q = _rot_to_wxyz(jit * base)
        n_appr = int(np.clip(round(N * (1.0 - 0.72 * t)), 24, N))
        label = "home" if t < 0.15 else ("near_object" if t > 0.72 else "mid_approach")
        return _clamp_ws(p), q, n_appr, label, None

    if fam == "above":
        h = rng.uniform(0.03, 0.20)
        xy = rng.normal(0, 1, 2).astype(np.float32) * 0.02
        p = np.array([grasp_p[0] + xy[0], grasp_p[1] + xy[1], grasp_p[2] + h], np.float32)
        tilt = Rot.from_rotvec(np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4),
                                         rng.uniform(-np.pi, np.pi)], np.float32))
        q = _rot_to_wxyz(tilt * _wxyz_to_rot(grasp_q))
        return _clamp_ws(p), q, max(28, N // 2), "above_object", None

    if fam == "ground":
        ang = rng.uniform(0, 2 * np.pi); r = rng.uniform(0.05, 0.16)
        p = grasp_p + np.array([r * np.cos(ang), r * np.sin(ang),
                                rng.uniform(-0.04, 0.02)], np.float32)
        p[2] = min(p[2], grasp_p[2] + 0.03)
        return _clamp_ws(p), _rot_to_wxyz(_wxyz_to_rot(grasp_q)), max(40, N // 2), "near_ground", None

    # air: random point in a box around home
    off = np.array([rng.uniform(-0.11, 0.11), rng.uniform(-0.13, 0.13),
                    rng.uniform(-0.09, 0.07)], np.float32)
    dq = Rot.from_rotvec(rng.normal(0, 0.18, 3))
    if fam == "failed":
        # recover_from = a just-missed grasp: at the object (small random xy/z error),
        # gripper CLOSED. start_pos = a hover ~4-8cm up & slightly toward the true
        # grasp, gripper OPEN -- where the recorded "recover" phase ends before the
        # normal approach takes over.
        err = rng.normal(0, 1, 3).astype(np.float32) * np.array([0.015, 0.015, 0.008], np.float32)
        recover_from = _clamp_ws(grasp_p + err)
        up = np.array([rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02),
                       rng.uniform(0.04, 0.09)], np.float32)
        start = _clamp_ws(grasp_p + up)
        return start, _rot_to_wxyz(_wxyz_to_rot(grasp_q)), max(28, N // 2), "failed_grasp", recover_from
    return _clamp_ws(home_p + off), _rot_to_wxyz(dq * _wxyz_to_rot(home_q)), N, "mid_air", None



def execute_and_collect_diverse_v2(
    worker: GenesisWorker,
    all_best_x: List[np.ndarray],
    init_obs_batch: dict,
    perception: PerceptionPipeline,
    action_config: ActionConfig,
    modes: List[str],                # (N,) per-env start mode
    rng: np.random.Generator,
    priv_cfg=None,
    dr_vec=None,
    record_video: bool = False,
    object_type: str = "rigid",
    yield_stress: Optional[float] = None,   # soft only: reject episodes whose top10 von Mises exceeds this
    crush_frac: float = 1.25,
    task_name_hint: Optional[str] = None,
    reactive: Optional[dict] = None,        # {"prob","speed":(lo,hi),"hold":int,"frame":(lo,hi)} or None:
                                            # random mid-approach object drag + scripted re-target
):
    """Per-env phase FSM (like collect_demos_synth_v3): every env advances through
    approach -> settle -> grasp -> lift -> hold independently, so a mode with a
    short recorded approach (near_object) simply starts closing sooner instead of
    hovering at the grasp pose while a long-approach env catches up. An env that
    finishes all phases stops being recorded (its buffers stop growing) -- it is
    held steady in the batched worker.step but NOT recorded, so there are NO
    frozen / padded frames in either the demo or the video.

    Returns: obs_bufs, act_bufs, rew_bufs, success, frame_bufs, n_preroll_steps
    """
    scales = (np.asarray(action_config.scales, dtype=np.float64)
              if action_config.mode != "absolute" else None)
    num_envs = worker.num_envs
    poses    = [v1._x_to_targets(x, 1) for x in all_best_x]
    pos_b    = np.concatenate([p[0] for p in poses], axis=0).astype(np.float32)
    quat_b   = np.concatenate([p[1] for p in poses], axis=0).astype(np.float32)
    grasp_pos = pos_b.copy()
    lift_b   = grasp_pos.copy(); lift_b[:, 2] += v1.LIFT_HEIGHT

    width_open = np.full(num_envs, 0.08, np.float32)
    try:
        from gentle_manip.assets.registry import get_object_def
        _short = float(min(get_object_def(task_name_hint).size[:2])) if task_name_hint else 0.06
    except Exception:
        _short = 0.06
    # CMA-ES routinely returns a straddle width WIDER than the object (SDF cost 0, no
    # contact). Clamp the close width to the object short axis (+2mm rigid) so a
    # straddle still grips; progressive lift-firming does the rest. (These are the
    # banana-proof values -- do not tighten: an over-tight close ejects a coarse-grid
    # soft body during the grasp.)
    _margin = 0.0     if object_type == "soft" else 0.0025
    _floor  = (min(0.014, 0.45 * _short) if object_type == "soft" else min(0.020, 0.6 * _short))
    _wcap   = min(0.075, _short + 0.002)
    width_cls  = np.clip(np.array([p[2] - _margin for p in poses], np.float32), _floor, _wcap)

    home_pos  = np.tile(worker.robot.home_pos[None].astype(np.float32),  (num_envs, 1))
    home_quat = np.tile(worker.robot.home_quat[None].astype(np.float32), (num_envs, 1))

    obj_center = init_obs_batch.get("priv_object_pos")
    if obj_center is None:
        obj_center = grasp_pos.copy()
    obj_center = np.asarray(obj_center, np.float32).reshape(num_envs, 3)

    # ── Per-env start pose + recorded-approach length ──
    start_pos  = np.zeros((num_envs, 3), np.float32)
    start_quat = np.zeros((num_envs, 4), np.float32)
    n_appr = np.zeros(num_envs, np.int64)
    dur_recover = np.zeros(num_envs, np.int64)
    recover_from = start_pos.copy()          # only meaningful where dur_recover>0
    start_grip = width_open.copy()           # gripper width at the first RECORDED step
    labels = []
    for i in range(num_envs):
        sp, sq, na, lab, rf = _sample_start(modes[i], home_pos[i], home_quat[i],
                                            pos_b[i], quat_b[i], obj_center[i], rng)
        start_pos[i], start_quat[i] = sp, sq
        n_appr[i] = na
        labels.append(lab)
        if rf is not None:                   # "failed" family: closed-gripper start + recover phase
            recover_from[i] = rf
            dur_recover[i]  = 34
            start_grip[i]   = width_cls[i]   # gripper CLOSED (just missed the grasp)
    # per-env label (home/mid_approach/near_object/above_object/near_ground/mid_air) for
    # video filenames + the episode "start_mode" tag; the collector's caller reads it back.
    modes[:] = labels
    any_preroll = any(l != "home" for l in labels)
    n_preroll_steps = N_PREROLL_STEPS if any_preroll else 0

    def _lerp(a, b, alpha):
        return a + alpha[:, None] * (b - a)

    def _slerp_pair(qa, qb, t):
        return _rot_to_wxyz(Slerp([0., 1.], Rot.concatenate(
            [_wxyz_to_rot(qa), _wxyz_to_rot(qb)]))(float(np.clip(t, 0, 1))))

    # ── Unrecorded pre-roll: home -> (recover_from if failed else start_pose) ──
    failed = dur_recover > 0
    pre_target = np.where(failed[:, None], recover_from, start_pos)
    move = np.array([m != "home" for m in modes])   # modes now holds per-env labels
    for j in range(n_preroll_steps):
        a = np.where(move, np.clip((j + 1) / N_PREROLL_STEPS, 0.0, 1.0), 0.0)
        qb = np.stack([_slerp_pair(home_quat[i], start_quat[i], a[i]) for i in range(num_envs)])
        # failed-grasp envs close the gripper over the final 40% of the pre-roll
        gclose = np.clip((a - 0.6) / 0.4, 0.0, 1.0)
        grip = np.where(failed, width_open + gclose * (width_cls - width_open), width_open)
        worker.step(_lerp(home_pos, pre_target, a), qb.astype(np.float32), grip.astype(np.float32))

    # ── Recorded buffers ──
    obs_bufs:   List[List[dict]]       = [[] for _ in range(num_envs)]
    act_bufs:   List[List[np.ndarray]] = [[] for _ in range(num_envs)]
    rew_bufs:   List[List[float]]      = [[] for _ in range(num_envs)]
    frame_bufs: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
    height_bufs: List[List[float]]     = [[] for _ in range(num_envs)]
    crushed = np.zeros(num_envs, bool)   # soft: episode ever exceeded crush_frac * yield

    cur_obs_list = [{k: init_obs_batch[k][i] for k in init_obs_batch} for i in range(num_envs)]
    prev_pos  = start_pos.copy()
    prev_quat = start_quat.copy()
    prev_grip = start_grip.copy()

    rec_slerps = [Slerp([0., 1.], Rot.concatenate(
        [_wxyz_to_rot(start_quat[i]), _wxyz_to_rot(quat_b[i])])) for i in range(num_envs)]

    # PHASES: (name, per-env duration array)
    dur_settle = np.full(num_envs, v1.N_SETTLE, np.int64)
    dur_grasp  = np.full(num_envs, v1.N_GRASP,  np.int64)
    dur_lift   = np.full(num_envs, v1.N_LIFT,   np.int64)
    dur_hold   = np.full(num_envs, v1.N_HOLD,   np.int64)
    PHASE_DUR  = [dur_recover, n_appr, dur_settle, dur_grasp, dur_lift, dur_hold]
    N_PHASES   = len(PHASE_DUR)

    def _env_target(i, ph, st):
        dur = int(PHASE_DUR[ph][i])
        if ph == 0:      # recover (failed-grasp only; dur=0 otherwise): OPEN gripper +
            a = min((st + 1) / max(dur, 1), 1.0)       # back away from the missed grasp
            pos = recover_from[i] + a * (start_pos[i] - recover_from[i])
            quat = quat_b[i]
            grip = width_cls[i] + a * (width_open[i] - width_cls[i])
        elif ph == 1:    # approach
            a = min((st + 1) / max(dur, 1), 1.0)
            pos = start_pos[i] + a * (pos_b[i] - start_pos[i])
            quat = _rot_to_wxyz(rec_slerps[i](float(a)))
            grip = width_open[i]
        elif ph == 2:    # settle
            pos, quat, grip = pos_b[i], quat_b[i], width_open[i]
        elif ph == 3:    # grasp
            a = (st + 1) / max(dur, 1)
            pos, quat = pos_b[i], quat_b[i]
            grip = width_open[i] + a * (width_cls[i] - width_open[i])
        elif ph == 4:    # lift
            a = (st + 1) / max(dur, 1)
            pos = pos_b[i] + a * (lift_b[i] - pos_b[i])
            quat = quat_b[i]
            firm = 0.0015 if object_type == "soft" else 0.0   # progressive grip firming
            grip = max(_floor, width_cls[i] - firm * a)
        else:            # hold
            firm = 0.0015 if object_type == "soft" else 0.0
            pos, quat, grip = lift_b[i], quat_b[i], max(_floor, width_cls[i] - firm)
        return pos, quat, grip

    phase_idx  = np.zeros(num_envs, np.int64)
    phase_step = np.zeros(num_envs, np.int64)

    # ── REACTIVE: a random lateral drag on the object mid-approach, then re-target the
    #    scripted grasp to the object's new position and re-enter the approach phase.
    #    `reactive` = {"prob","speed":(lo,hi),"hold":int,"frame":(lo,hi)} or None.
    rx_vel = np.zeros((num_envs, 3), np.float32)   # per-frame drag velocity (0 => no drag)
    rx_fire = np.full(num_envs, -1, np.int64)      # recorded-loop iter to start the drag
    rx_hold = 4
    if reactive:
        rx_hold = int(reactive.get("hold", 4))
        lo_s, hi_s = reactive["speed"]
        lo_f, hi_f = reactive.get("frame", (10, 40))
        for i in range(num_envs):
            if labels[i] == "failed_grasp" or rng.random() >= reactive["prob"]:
                continue
            th = rng.uniform(0, 2 * np.pi); sp = rng.uniform(lo_s, hi_s)
            rx_vel[i] = (np.cos(th) * sp, np.sin(th) * sp, 0.0)
            rx_fire[i] = int(rng.integers(int(lo_f), int(hi_f) + 1))
    rx_retargets = np.zeros(num_envs, np.int64)
    obj_at_plan = obj_center.copy()
    it = 0

    while np.any(phase_idx < N_PHASES):
        active = phase_idx < N_PHASES
        cp = np.zeros((num_envs, 3), np.float32)
        cq = np.zeros((num_envs, 4), np.float32)
        cg = np.zeros(num_envs, np.float32)
        for i in range(num_envs):
            if active[i]:
                cp[i], cq[i], cg[i] = _env_target(i, int(phase_idx[i]), int(phase_step[i]))
            else:
                cp[i], cq[i], cg[i] = lift_b[i], quat_b[i], width_cls[i]

        # REACTIVE drag: hold the object velocity for rx_hold frames from rx_fire (only
        # while still approaching -- phase 1/2). Applied BEFORE worker.step so it takes
        # this frame.
        if reactive and object_type != "rigid":
            dragging = active & (rx_fire >= 0) & (it >= rx_fire) & (it < rx_fire + rx_hold) \
                       & np.isin(phase_idx, (1, 2))
            for i in np.nonzero(dragging)[0]:
                try:
                    obj0 = worker.handle.objects[0]
                    npart = int(np.asarray(worker.handle.object_base_particles[0]).reshape(num_envs, -1, 3).shape[1])
                    obj0.set_particles_vel(np.tile(rx_vel[i], (npart, 1)).astype(np.float32),
                                           envs_idx=[int(i)])
                except Exception as ex:
                    print(f"  [reactive] env {int(i)} drag failed: {ex}", flush=True)

        if action_config.mode == "absolute":
            actions = v1._invert_actions_absolute(cp, cq, cg, action_config)
        else:
            actions = v1._invert_actions(prev_pos, cp, prev_quat, cq, prev_grip, cg, scales)
        state = worker.step(cp, cq, cg)

        if record_video:
            fr = worker.render_rgb(all_envs=True)
            if fr is not None:
                for i in range(num_envs):
                    if active[i]:
                        frame_bufs[i].append(np.asarray(fr[i], np.uint8))

        raw_next = v1._state_to_raw_obs(state)
        next_obs_batch = perception.process(raw_next)
        if priv_cfg is not None:
            oq = state.get("object_quat")
            if oq is None:
                oq = np.tile(np.array([1., 0, 0, 0], np.float32), (num_envs, 1))
            next_obs_batch.update(v1._privileged_obs_batch(
                state["object_center"], oq, dr_vec, priv_cfg,
                contact_force=state.get("contact_force")))
        next_obs_list = [{k: next_obs_batch[k][i] for k in next_obs_batch} for i in range(num_envs)]
        vm = state.get("von_mises_stress")
        if yield_stress is not None and vm is not None:
            vm = np.asarray(vm, np.float64)
            k = max(1, int(round(0.10 * vm.shape[1])))
            top10 = np.partition(vm, -k, axis=1)[:, -k:].mean(axis=1)
            crushed |= (top10 > crush_frac * yield_stress)
        obj_z = state["object_center"][:, 2]
        for i in range(num_envs):
            if active[i]:
                obs_bufs[i].append(cur_obs_list[i])
                act_bufs[i].append(actions[i])
                rew_bufs[i].append(0.0)
                height_bufs[i].append(float(obj_z[i] - grasp_pos[i, 2]))
        cur_obs_list = next_obs_list
        prev_pos[:], prev_quat[:], prev_grip[:] = cp, cq, cg

        # REACTIVE re-target: WAIT for the object to stop sliding after the drag
        # (rx_hold + RX_SETTLE frames), then shift the scripted grasp/lift targets by
        # the object's FINAL displacement and re-enter the approach phase from the
        # current EE pose. Reading mid-slide (the old bug) re-targeted to a spot the
        # object had already left -> grasp missed.
        RX_SETTLE = 16
        if reactive:
            done_drag = (rx_fire >= 0) & (it >= rx_fire + rx_hold + RX_SETTLE) & (rx_retargets < 2)
            for i in np.nonzero(done_drag & active)[0]:
                obj_now = np.asarray(state["object_center"][i], np.float32)
                disp = obj_now - obj_at_plan[i]
                if float(np.hypot(disp[0], disp[1])) < 0.012:      # not enough to matter
                    rx_fire[i] = -1
                    continue
                d3 = np.array([disp[0], disp[1], 0.0], np.float32)
                pos_b[i] += d3; grasp_pos[i] += d3; lift_b[i] += d3
                obj_center[i] = obj_now; obj_at_plan[i] = obj_now
                # re-approach STARTS from a safe hover above the object's NEW spot (lift
                # straight up from wherever the arm is, then come down on the new pose) --
                # a direct lateral lerp at grasp height knocks/misses the moved object.
                ee_now = np.asarray(state["ee_pos"][i], np.float32)
                start_pos[i] = np.array([ee_now[0], ee_now[1],
                                         max(float(ee_now[2]), float(pos_b[i][2]) + 0.11)], np.float32)
                rec_slerps[i] = Slerp([0., 1.], Rot.concatenate(
                    [_wxyz_to_rot(quat_b[i]), _wxyz_to_rot(quat_b[i])]))
                n_appr[i] = max(44, v1.N_HOME_TO_PRE // 2)         # PHASE_DUR[1] shares this array
                phase_idx[i] = 1; phase_step[i] = 0
                rx_retargets[i] += 1
                rx_fire[i] = -1                                    # one drag per env (retry loop caps re-targets)
                print(f"  [reactive] env {int(i)} re-target: object moved "
                      f"{np.round(disp[:2], 3).tolist()} m -> re-approach", flush=True)

        it += 1
        phase_step[active] += 1
        dur_now = np.array([int(PHASE_DUR[min(int(p), N_PHASES - 1)][i])
                            for i, p in enumerate(phase_idx)])
        roll = active & (phase_step >= dur_now)
        phase_idx[roll] += 1
        phase_step[roll] = 0

    obj_z_final = np.array([grasp_pos[i, 2] + (height_bufs[i][-1] if height_bufs[i] else -1.0)
                            for i in range(num_envs)])
    success = (obj_z_final > (grasp_pos[:, 2] + v1.LIFT_HEIGHT * 0.5)) & ~crushed
    if crushed.any():
        print(f"  [crush] {int(crushed.sum())} env(s) exceeded {crush_frac:.2f}x yield -> rejected")

    # ── Early-success trim (same cut on obs/act/rew AND frames) ──
    for i in range(num_envs):
        h = np.asarray(height_bufs[i]) if height_bufs[i] else np.array([])
        run = 0
        cut_at = None
        for t in range(len(h)):
            run = run + 1 if h[t] >= SUCCESS_HEIGHT else 0
            if run >= SUCCESS_HOLD_STEPS:
                cut_at = min(t + TRIM_MARGIN_STEPS + 1, len(h))
                break
        if cut_at is not None and cut_at < len(obs_bufs[i]):
            obs_bufs[i]  = obs_bufs[i][:cut_at]
            act_bufs[i]  = act_bufs[i][:cut_at]
            rew_bufs[i]  = rew_bufs[i][:cut_at]
            if frame_bufs[i]:
                frame_bufs[i] = frame_bufs[i][:cut_at]

    return obs_bufs, act_bufs, rew_bufs, success, frame_bufs, n_preroll_steps, (rx_retargets > 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True)
    p.add_argument("--task-name", default=None)
    p.add_argument("--out-dir", type=Path, default=Path("dataset") / "demos")
    p.add_argument("--shard-size", type=int, default=5)
    p.add_argument("--description", type=str, default="")
    p.add_argument("--n-episodes", type=int, default=500)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--maxfevals", type=int, default=901)
    p.add_argument("--scene-dr-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument("--start-modes", type=str, default=DEFAULT_START_MODES,
                   help="comma list mode:weight -- per-episode EE start-pose distribution")
    p.add_argument("--top-down", dest="top_down", action="store_true", default=False,
                   help="clamp CMA-ES approach PITCH so grasps stay top-ish (default OFF; the\n                         sky/ground penalties in grasp_cost already bias top-down)")
    p.add_argument("--no-top-down", dest="top_down", action="store_false")
    p.add_argument("--crush-frac", type=float, default=1.15,
                   help="soft only: reject an episode whose top-10%% von Mises ever exceeds\n"
                        "                         this multiple of the object's yield stress (default 1.15)")
    p.add_argument("--record-video", dest="record_video", nargs="?", type=int,
                   const=10**9, default=0,
                   help="record an RGB clip per saved episode; N = first N only")
    p.add_argument("--reactive", action="store_true",
                   help="collect reactive-recovery demos: random mid-approach object drag + scripted re-target")
    p.add_argument("--reactive-prob", type=float, default=0.6)
    p.add_argument("--reactive-speed", type=float, nargs=2, default=[0.12, 0.38])
    p.add_argument("--reactive-frame", type=int, nargs=2, default=[12, 45],
                   help="recorded-loop iter window to start the drag (during approach)")
    p.add_argument("--per-cat-target", type=int, default=0,
                   help="cross-category: once a category has this many demos (this run + "
                        "--cat-have preseed), DROP it from the pool so the rest get the "
                        "budget. 0 = off (draw uniformly to --n-episodes).")
    p.add_argument("--cat-have", type=str, default="",
                   help="comma list cat:count of demos ALREADY collected elsewhere, "
                        "pre-seeded into the per-cat-target accounting (e.g. "
                        "'mushroom:160,kiwi:166').")
    args = p.parse_args()

    modes_arr, modes_p = _parse_start_modes(args.start_modes)

    exp        = Experiment.load(args.experiment)
    task       = SingleLiftTask(exp.task_cfg)
    spec       = task.scene_spec
    obs_config = exp.collection_obs()
    priv_cfg   = obs_config.privileged
    action_config = exp.action_config
    dr_cfg     = DRConfig.from_dict(exp.dr)
    task_name  = args.task_name or exp._raw.get("task", args.experiment)
    rate_hz    = 1.0 / spec.sim_dt

    settle_steps      = int(exp.task_cfg.get("settle_steps",     30))
    settle_max_steps  = int(exp.task_cfg.get("settle_max_steps", 200))
    settle_vel_thresh = float(exp.task_cfg.get("settle_vel_thresh", 0.002))

    perception = PerceptionPipeline(obs_config)
    _yield = float(task.object_yield_stress) if getattr(task, "object_yield_stress", None) else None
    if _yield:
        print(f"  soft crush gate: yield={_yield:.0f} Pa (reject episodes > {args.crush_frac:.2f}x)")

    collection_config = {
        "task_name": task_name, "description": args.description,
        "source": "cmaes_synth_diverse_start_v2", "git_commit": v1._git_commit(),
        "experiment": args.experiment,
        "control": {"n_envs": args.n_envs, "maxfevals": args.maxfevals,
                    "n_episodes": args.n_episodes, "scene_dr_every": args.scene_dr_every,
                    "seed": args.seed, "start_modes": args.start_modes,
                    "top_down": bool(args.top_down),
                    "success_height": SUCCESS_HEIGHT, "success_hold_steps": SUCCESS_HOLD_STEPS,
                    "n_preroll_steps": N_PREROLL_STEPS},
        "dr": exp.dr,
    }

    print(f"\n=== collect_demos_diverse_start_v2  experiment={args.experiment}"
          f" -- target {args.n_episodes} eps, {args.n_envs} envs/batch"
          f"  start-modes={dict(zip(modes_arr.tolist(), modes_p.round(3).tolist()))}"
          f"  top_down={args.top_down}  record_video={'off' if not args.record_video else args.record_video}")

    rng = np.random.default_rng(args.seed)

    nominal_spec = spec
    do_scene_dr  = args.scene_dr_every > 0 and dr_cfg.has_scene_dr()
    xcat_pool    = list(dr_cfg.object_category_pool or ())    # MUTABLE: categories that hit
                                                             # --per-cat-target are dropped so
                                                             # the rest get the freed budget
    per_cat_target = args.per_cat_target or 0
    reactive_cfg = ({"prob": float(args.reactive_prob),
                     "speed": tuple(args.reactive_speed), "hold": 4,
                     "frame": tuple(args.reactive_frame)} if args.reactive else None)
    if reactive_cfg: print(f"  [reactive] {reactive_cfg}")
    cat_saved: Dict[str, int] = {c: 0 for c in xcat_pool}
    for tok in filter(None, (t.strip() for t in args.cat_have.split(","))):
        k, _, v = tok.partition(":")
        if k in cat_saved:
            cat_saved[k] = int(v)
    if per_cat_target:
        for c in list(xcat_pool):
            if cat_saved.get(c, 0) >= per_cat_target:
                xcat_pool.remove(c)
                print(f"  [pool] {c} already at {cat_saved[c]} >= {per_cat_target} -> excluded")
    if xcat_pool:
        print(f"  CROSS-CATEGORY pool ({len(xcat_pool)}): {list(xcat_pool)}"
              + (f"  per-cat target {per_cat_target}  have {cat_saved}" if per_cat_target else ""))
    if not xcat_pool and per_cat_target:
        print("  [pool] all categories already at target -> nothing to collect"); return
    import tempfile
    deform_dir = tempfile.mkdtemp(prefix="gm_synth_deform_dsv2_") if (do_scene_dr or xcat_pool) else None

    def _with_registry_object(base_spec, cat_name):
        """Rebuild base_spec.objects[0] from the registry entry for cat_name so a
        fresh scene spawns a different object. Material/E/nu/rho reset to None so
        scene_builder re-resolves from the new category's ObjectDef."""
        import dataclasses
        from gentle_manip.assets.registry import get_object_def
        d = get_object_def(cat_name)
        o = base_spec.objects[0]
        no = dataclasses.replace(o, name=cat_name, object_type=d.object_type,
                                 mesh_path=d.mesh_path, scale=1.0, spawn_z=None,
                                 youngs_modulus=None, poisson_ratio=None, density=None)
        return dataclasses.replace(base_spec, objects=[no, *base_spec.objects[1:]])

    def _yield_for(cat_name):
        from gentle_manip.assets.registry import get_object_def
        try:
            return float(get_object_def(cat_name).material.von_mises_yield_stress)
        except Exception:
            return _yield

    def _mesh_ok(path):
        """Reject a deformed mesh the CMA-ES SDF build would choke on (degenerate
        AABB -> trimesh 'Bounds must be (n, dimension*2)!')."""
        if not path:
            return True
        try:
            import trimesh, numpy as _np
            m = trimesh.load(str(path), process=False, force="mesh")
            b = _np.asarray(m.bounds, dtype=float)
            return (b.shape == (2, 3) and _np.all(_np.isfinite(b))
                    and _np.all(b[1] - b[0] > 1e-4) and len(m.faces) >= 8)
        except Exception:
            return False

    def _make_worker():
        base_spec = nominal_spec
        cat_name  = task.object_name
        if xcat_pool:
            cat_name  = str(rng.choice(xcat_pool))
            base_spec = _with_registry_object(nominal_spec, cat_name)
        sdr = {"scale": float(base_spec.objects[0].scale or 1.0), "bend_deg": 0.0}
        spec_dr = base_spec
        if do_scene_dr:
            for _try in range(3):
                spec_dr, sdr = v1._apply_scene_dr(base_spec, dr_cfg, rng, deform_dir)
                if _mesh_ok(spec_dr.objects[0].mesh_path):
                    break
                print(f"  [dr] degenerate deformed mesh (try {_try+1}) -> retry", flush=True)
            else:
                spec_dr, sdr = base_spec, {"scale": float(base_spec.objects[0].scale or 1.0), "bend_deg": 0.0}
                print("  [dr] falling back to nominal (no shape deform) for this scene", flush=True)
        w = GenesisWorker(spec_dr, num_envs=args.n_envs, show_viewer=False,
                          settle_steps=settle_steps, settle_max_steps=settle_max_steps,
                          settle_vel_thresh=settle_vel_thresh, render_obs_cameras=True)
        yld = _yield_for(cat_name) if (task.object_type == "soft") else None
        if xcat_pool:
            print(f"  scene object -> {cat_name}  (yield={yld})")
        return w, sdr, (w.handle.spec.objects[0].mesh_path or v1.MUSHROOM_MESH), cat_name, yld

    worker, scene_dr, actual_mesh, scene_cat, scene_yield = _make_worker()
    if do_scene_dr:
        print(f"  scene DR ON (every {args.scene_dr_every} batch) -> {deform_dir}")

    left_pts  = v1.sample_finger_surface(v1.LEFT_FINGER,  n=300)
    right_pts = v1.sample_finger_surface(v1.RIGHT_FINGER, n=300)
    executor = ProcessPoolExecutor(max_workers=args.n_envs)
    print(f"  Mesh: {Path(actual_mesh).name}")

    run_dir  = v1._make_run_dir(args.out_dir, task_name)
    vid_dir  = run_dir / "videos"
    if args.record_video:
        vid_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(collection_config, f, sort_keys=False)
    print(f"  Run dir -> {run_dir.resolve()}")

    total_saved = total_failed = 0
    batch_idx = shard_idx = 0
    shard_buf: List[dict] = []
    t0 = time.time()

    consec_fail = 0
    while total_saved < args.n_episodes:
        if per_cat_target and not xcat_pool:
            print("  [pool] every category reached its per-cat target -> stopping", flush=True)
            break
        batch_idx += 1
        n = args.n_envs
        try:
            if (do_scene_dr or xcat_pool) and batch_idx > 1 and (batch_idx - 1) % args.scene_dr_every == 0:
                worker.close()
                worker, scene_dr, actual_mesh, scene_cat, scene_yield = _make_worker()

            print(f"\n-- Batch {batch_idx}  [{total_saved}/{args.n_episodes} saved]"
                  + (f"  scale={scene_dr['scale']:.3f} bend={scene_dr['bend_deg']:+.1f}" if do_scene_dr else ""))

            object_dxy   = dr_cfg.sample_object_dxy(rng, n)
            object_euler = dr_cfg.sample_object_euler(rng, n)
            home_offset  = dr_cfg.sample_home_offset(rng, n)
            worker.reset(object_dxy=object_dxy, object_euler=object_euler, home_offset=home_offset)

            init_state = worker.read_state()
            raw_init = v1._state_to_raw_obs(init_state)
            init_obs_batch = perception.process(raw_init)
            dr_vec = np.array([float(scene_dr.get("scale", 1.0)), float(scene_dr.get("bend_deg", 0.0))], np.float32)
            if priv_cfg is not None:
                oq = init_state.get("object_quat")
                if oq is None:
                    oq = np.tile(np.array([1., 0, 0, 0], np.float32), (n, 1))
                init_obs_batch.update(v1._privileged_obs_batch(
                    init_state["object_center"], oq, dr_vec, priv_cfg,
                    contact_force=init_state.get("contact_force")))

            obj_pos_all  = init_state["object_center"].astype(np.float64)
            obj_quat_all = init_state["object_quat"].astype(np.float64)

            # actual AABB extents of the drawn+scaled object -> size-scaled CMA bounds
            try:
                from gentle_manip.assets.registry import get_object_def
                _osz = np.asarray(get_object_def(scene_cat).size, float) * float(scene_dr.get("scale", 1.0))
            except Exception:
                _osz = None

            payloads = []
            for i in range(n):
                lb, ub = _synth_bounds_topdown(obj_pos_all[i], args.top_down, obj_size=_osz)
                payloads.append((actual_mesh, obj_pos_all[i], obj_quat_all[i],
                                 left_pts, right_pts, args.maxfevals, lb, ub, str(run_dir / "cmaes_logs")))
            futures = [executor.submit(v1._synth_worker, pl) for pl in payloads]
            all_best_x = []
            for i, fut in enumerate(futures):
                best_x, score = fut.result()
                all_best_x.append(best_x)
                print(f"  Env {i}: cost={score:.3f}  tcp={best_x[:3].round(3)}  w={best_x[6]*1e3:.1f}mm")

            modes = list(np.asarray(modes_arr)[rng.choice(len(modes_arr), size=n, p=modes_p)])  # families; overwritten in-place with per-env labels
            print(f"  start modes: {modes}")

            want_video = bool(args.record_video) and total_saved < args.record_video
            obs_bufs, act_bufs, rew_bufs, success, frame_bufs, _npr, rx_flags = execute_and_collect_diverse_v2(
                worker, all_best_x, init_obs_batch, perception, action_config,
                modes, rng, priv_cfg=priv_cfg, dr_vec=dr_vec, record_video=want_video,
                object_type=task.object_type,
                yield_stress=(scene_yield if xcat_pool else _yield),
                crush_frac=args.crush_frac,
                task_name_hint=(scene_cat if xcat_pool else task.object_name),
                reactive=reactive_cfg,
            )
            print(f"  success: {success.tolist()}")

            for i in range(n):
                if not success[i]:
                    total_failed += 1
                    if not args.keep_failures:
                        continue
                obs_list = obs_bufs[i]
                if not obs_list:
                    continue
                keys = obs_list[0].keys()
                episode = {
                    "observations": {k: np.stack([o[k] for o in obs_list]) for k in keys},
                    "actions": np.stack(act_bufs[i]),
                    "rewards": np.asarray(rew_bufs[i], np.float32),
                    # a re-targeted episode is a reactive-recovery demo (object dragged
                    # mid-approach -> arm redirected); tag it so the stager/rebalancer can
                    # control the reactive share of the training mix.
                    "start_mode": ("reactive_recover" if bool(rx_flags[i]) else modes[i]),
                }
                shard_buf.append(episode)
                total_saved += 1
                ep_id = total_saved
                if per_cat_target and xcat_pool:
                    cat_saved[scene_cat] = cat_saved.get(scene_cat, 0) + 1
                    if cat_saved[scene_cat] >= per_cat_target and scene_cat in xcat_pool:
                        xcat_pool.remove(scene_cat)
                        print(f"  [pool] {scene_cat} reached {cat_saved[scene_cat]} -> "
                              f"DROPPED. remaining: {xcat_pool}", flush=True)
                if want_video and frame_bufs[i]:
                    vp = vid_dir / f"ep{ep_id:04d}_env{i}_{modes[i]}_{'ok' if success[i] else 'fail'}.mp4"
                    try:
                        imageio.mimwrite(str(vp), frame_bufs[i], fps=30, quality=8,
                                         macro_block_size=1)
                    except Exception as e:
                        print(f"    [video] ep{ep_id} write failed: {e}")
                print(f"    ep {ep_id}: env {i}  {'OK' if success[i] else 'X'}  "
                      f"T={episode['actions'].shape[0]}  mode={modes[i]}"
                      + (f"  vid" if (want_video and frame_bufs[i]) else ""))

                if len(shard_buf) >= args.shard_size:
                    sp = v1._write_shard(run_dir, shard_buf, task_name, shard_idx, rate_hz)
                    print(f"  shard {shard_idx} -> {sp.name}")
                    shard_idx += 1
                    shard_buf = []
                if total_saved >= args.n_episodes:
                    break
        except Exception as e:
            import traceback
            consec_fail += 1
            print(f"  [batch {batch_idx}] FAILED ({type(e).__name__}: {e}) -- "
                  f"skipping, rebuilding worker ({consec_fail} consecutive)", flush=True)
            traceback.print_exc()
            try:
                worker.close()
            except Exception:
                pass
            if consec_fail >= 12:
                print("  [collect] 12 consecutive batch failures -- aborting", flush=True)
                break
            worker, scene_dr, actual_mesh, scene_cat, scene_yield = _make_worker()
            continue
        consec_fail = 0

    if shard_buf:
        v1._write_shard(run_dir, shard_buf, task_name, shard_idx, rate_hz)

    data_path = v1._merge_shards(run_dir)
    elapsed = time.time() - t0
    attempts = total_saved + total_failed
    sr = total_saved / attempts if attempts else 0.0
    print(f"\n=== Done ===\n  saved={total_saved}  failed={total_failed}  SR={sr*100:.1f}%"
          f"  elapsed={elapsed/60:.1f}min\n  data={data_path}")
    with open(run_dir / "stats.yaml", "w") as f:
        yaml.dump({"episodes_saved": total_saved, "episodes_failed": total_failed,
                   "total_attempts": attempts, "success_rate": round(sr, 4),
                   "elapsed_min": round(elapsed / 60, 2)}, f, default_flow_style=False)
    executor.shutdown(wait=False)
    worker.close()


if __name__ == "__main__":
    main()
