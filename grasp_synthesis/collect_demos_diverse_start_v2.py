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

  2. RICH end-effector start diversity (`--start-modes`, default
     "home:0.25,near_object:0.30,near_ground:0.25,mid_air:0.20"). Instead of the
     binary home/near-object of v1, each episode samples a start MODE that places
     the EE (via an UNRECORDED pre-roll from home) at a diverse pose that looks
     like where a failed first attempt would leave you:
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

START_MODES = ("home", "near_object", "near_ground", "mid_air")

# per-mode: number of RECORDED approach steps (start-pose -> grasp target).
# home keeps v1's full N_HOME_TO_PRE; the diverse modes start closer / need a
# shorter, sharper redirect (which is the regrasp-approach behaviour we want).
RECORDED_APPROACH_STEPS = {
    "home":        v1.N_HOME_TO_PRE,
    "near_object": max(30, v1.N_HOME_TO_PRE // 3),
    "near_ground": max(40, v1.N_HOME_TO_PRE // 2),
    "mid_air":     v1.N_HOME_TO_PRE,
}


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
        # x = [tx, ty, tz, roll, pitch, yaw, width]; indices 3=roll, 4=pitch
        lb[3], ub[3] = -0.16 * np.pi, 0.16 * np.pi
        lb[4], ub[4] = -0.10 * np.pi, 0.10 * np.pi
    return lb, ub


def _parse_start_modes(spec: str) -> Tuple[np.ndarray, np.ndarray]:
    """"home:0.25,near_object:0.30,..." -> (modes, probs) both aligned to START_MODES."""
    weights = {m: 0.0 for m in START_MODES}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, w = tok.partition(":")
        name = name.strip()
        if name not in weights:
            raise ValueError(f"unknown start mode {name!r}; valid: {START_MODES}")
        weights[name] = float(w) if w else 1.0
    probs = np.array([weights[m] for m in START_MODES], np.float64)
    if probs.sum() <= 0:
        raise ValueError("start-modes weights sum to 0")
    return np.array(START_MODES), probs / probs.sum()


def _start_pose_for_mode(mode: str, home_p, home_q, grasp_p, grasp_q, obj_p, rng):
    """(start_pos (3,), start_quat wxyz (4,)) for one env's sampled start mode.
    All non-home modes: gripper open, roughly grasp orientation."""
    if mode == "home":
        return home_p.copy(), home_q.copy()
    gq = grasp_q.copy()
    if mode == "near_object":
        off = rng.normal(0, 1, 3).astype(np.float32)
        off /= (np.linalg.norm(off) + 1e-8)
        off *= rng.uniform(0.02, 0.06)
        off[2] = abs(off[2]) + rng.uniform(0.0, 0.04)   # bias to ABOVE the grasp point
        return _clamp_ws(grasp_p + off), gq
    if mode == "near_ground":
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.05, 0.14)
        off = np.array([r * np.cos(ang), r * np.sin(ang),
                        rng.uniform(-0.04, 0.02)], np.float32)
        p = grasp_p + off
        p[2] = min(p[2], grasp_p[2] + 0.03)             # keep it LOW
        return _clamp_ws(p), gq
    # mid_air: random point in a box around home
    off = np.array([rng.uniform(-0.10, 0.10), rng.uniform(-0.12, 0.12),
                    rng.uniform(-0.08, 0.06)], np.float32)
    # small orientation perturbation from home
    dq = Rot.from_rotvec(rng.normal(0, 0.15, 3))
    q = _rot_to_wxyz(dq * _wxyz_to_rot(home_q))
    return _clamp_ws(home_p + off), q


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
    width_cls  = np.array([p[2] - 0.0025 for p in poses], np.float32)

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
    for i in range(num_envs):
        sp, sq = _start_pose_for_mode(modes[i], home_pos[i], home_quat[i],
                                      pos_b[i], quat_b[i], obj_center[i], rng)
        start_pos[i], start_quat[i] = sp, sq
        n_appr[i] = RECORDED_APPROACH_STEPS[modes[i]]

    any_preroll = any(m != "home" for m in modes)
    n_preroll_steps = N_PREROLL_STEPS if any_preroll else 0

    def _lerp(a, b, alpha):
        return a + alpha[:, None] * (b - a)

    def _slerp_pair(qa, qb, t):
        return _rot_to_wxyz(Slerp([0., 1.], Rot.concatenate(
            [_wxyz_to_rot(qa), _wxyz_to_rot(qb)]))(float(np.clip(t, 0, 1))))

    # ── Unrecorded pre-roll: home -> start_pose ──
    move = np.array([m != "home" for m in modes])
    for j in range(n_preroll_steps):
        a = np.where(move, np.clip((j + 1) / N_PREROLL_STEPS, 0.0, 1.0), 0.0)
        qb = np.stack([_slerp_pair(home_quat[i], start_quat[i], a[i]) for i in range(num_envs)])
        worker.step(_lerp(home_pos, start_pos, a), qb.astype(np.float32), width_open)

    # ── Recorded buffers ──
    obs_bufs:   List[List[dict]]       = [[] for _ in range(num_envs)]
    act_bufs:   List[List[np.ndarray]] = [[] for _ in range(num_envs)]
    rew_bufs:   List[List[float]]      = [[] for _ in range(num_envs)]
    frame_bufs: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
    height_bufs: List[List[float]]     = [[] for _ in range(num_envs)]

    cur_obs_list = [{k: init_obs_batch[k][i] for k in init_obs_batch} for i in range(num_envs)]
    prev_pos  = start_pos.copy()
    prev_quat = start_quat.copy()
    prev_grip = width_open.copy()

    rec_slerps = [Slerp([0., 1.], Rot.concatenate(
        [_wxyz_to_rot(start_quat[i]), _wxyz_to_rot(quat_b[i])])) for i in range(num_envs)]

    # PHASES: (name, per-env duration array)
    dur_settle = np.full(num_envs, v1.N_SETTLE, np.int64)
    dur_grasp  = np.full(num_envs, v1.N_GRASP,  np.int64)
    dur_lift   = np.full(num_envs, v1.N_LIFT,   np.int64)
    dur_hold   = np.full(num_envs, v1.N_HOLD,   np.int64)
    PHASE_DUR  = [n_appr, dur_settle, dur_grasp, dur_lift, dur_hold]
    N_PHASES   = len(PHASE_DUR)

    def _env_target(i, ph, st):
        dur = int(PHASE_DUR[ph][i])
        if ph == 0:      # approach
            a = min((st + 1) / max(dur, 1), 1.0)
            pos = start_pos[i] + a * (pos_b[i] - start_pos[i])
            quat = _rot_to_wxyz(rec_slerps[i](float(a)))
            grip = width_open[i]
        elif ph == 1:    # settle
            pos, quat, grip = pos_b[i], quat_b[i], width_open[i]
        elif ph == 2:    # grasp
            a = (st + 1) / max(dur, 1)
            pos, quat = pos_b[i], quat_b[i]
            grip = width_open[i] + a * (width_cls[i] - width_open[i])
        elif ph == 3:    # lift
            a = (st + 1) / max(dur, 1)
            pos = pos_b[i] + a * (lift_b[i] - pos_b[i])
            quat, grip = quat_b[i], width_cls[i]
        else:            # hold
            pos, quat, grip = lift_b[i], quat_b[i], width_cls[i]
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
    success = obj_z_final > (grasp_pos[:, 2] + v1.LIFT_HEIGHT * 0.5)

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
    p.add_argument("--start-modes", type=str,
                   default="home:0.25,near_object:0.30,near_ground:0.25,mid_air:0.20",
                   help="comma list mode:weight -- per-episode EE start-pose distribution")
    p.add_argument("--top-down", dest="top_down", action="store_true", default=True,
                   help="clamp CMA-ES roll/pitch so grasps are top-down (default on)")
    p.add_argument("--no-top-down", dest="top_down", action="store_false")
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
    import tempfile
    deform_dir = tempfile.mkdtemp(prefix="gm_synth_deform_dsv2_") if do_scene_dr else None

    def _make_worker():
        if do_scene_dr:
            spec_dr, sdr = v1._apply_scene_dr(nominal_spec, dr_cfg, rng, deform_dir)
        else:
            spec_dr, sdr = nominal_spec, {"scale": float(nominal_spec.objects[0].scale or 1.0), "bend_deg": 0.0}
        w = GenesisWorker(spec_dr, num_envs=args.n_envs, show_viewer=False,
                          settle_steps=settle_steps, settle_max_steps=settle_max_steps,
                          settle_vel_thresh=settle_vel_thresh, render_obs_cameras=True)
        return w, sdr, (w.handle.spec.objects[0].mesh_path or v1.MUSHROOM_MESH)

    worker, scene_dr, actual_mesh = _make_worker()
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

    while total_saved < args.n_episodes:
        batch_idx += 1
        n = args.n_envs
        if do_scene_dr and batch_idx > 1 and (batch_idx - 1) % args.scene_dr_every == 0:
            worker.close()
            worker, scene_dr, actual_mesh = _make_worker()

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

        modes = list(np.asarray(modes_arr)[rng.choice(len(modes_arr), size=n, p=modes_p)])
        print(f"  start modes: {modes}")

        want_video = bool(args.record_video) and total_saved < args.record_video
        obs_bufs, act_bufs, rew_bufs, success, frame_bufs, _npr = execute_and_collect_diverse_v2(
            worker, all_best_x, init_obs_batch, perception, action_config,
            modes, rng, priv_cfg=priv_cfg, dr_vec=dr_vec, record_video=want_video,
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
