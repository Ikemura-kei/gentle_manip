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


def _synth_bounds_topdown(obj_pos: np.ndarray, top_down: bool):
    """v1._synth_bounds with the roll/pitch bound tightened for a top-down grip."""
    lb, ub = v1._synth_bounds(obj_pos)
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
    _margin = 0.0     if object_type == "soft" else 0.0025
    _floor  = 0.014   if object_type == "soft" else 0.020
    # cap at the object's short cross-section (+2mm) so a too-wide CMA straddle still grips
    try:
        from gentle_manip.assets.registry import get_object_def
        _short = float(min(get_object_def(task_name_hint).size[:2])) if task_name_hint else 0.06
    except Exception:
        _short = 0.06
    _wcap = min(0.075, _short + 0.002)
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

    return obs_bufs, act_bufs, rew_bufs, success, frame_bufs, n_preroll_steps


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
    xcat_pool    = tuple(dr_cfg.object_category_pool or ())   # cross-category: draw object per scene
    if xcat_pool:
        print(f"  CROSS-CATEGORY pool ({len(xcat_pool)}): {list(xcat_pool)}")
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

    def _make_worker():
        base_spec = nominal_spec
        cat_name  = task.object_name
        if xcat_pool:
            cat_name  = str(rng.choice(xcat_pool))
            base_spec = _with_registry_object(nominal_spec, cat_name)
        if do_scene_dr:
            spec_dr, sdr = v1._apply_scene_dr(base_spec, dr_cfg, rng, deform_dir)
        else:
            spec_dr, sdr = base_spec, {"scale": float(base_spec.objects[0].scale or 1.0), "bend_deg": 0.0}
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

            payloads = []
            for i in range(n):
                lb, ub = _synth_bounds_topdown(obj_pos_all[i], args.top_down)
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
            obs_bufs, act_bufs, rew_bufs, success, frame_bufs, _npr = execute_and_collect_diverse_v2(
                worker, all_best_x, init_obs_batch, perception, action_config,
                modes, rng, priv_cfg=priv_cfg, dr_vec=dr_vec, record_video=want_video,
                object_type=task.object_type,
                yield_stress=(scene_yield if xcat_pool else _yield),
                crush_frac=args.crush_frac,
                task_name_hint=(scene_cat if xcat_pool else task.object_name),
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
                    "start_mode": modes[i],
                }
                shard_buf.append(episode)
                total_saved += 1
                ep_id = total_saved
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
