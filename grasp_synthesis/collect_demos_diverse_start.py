"""OmniReset-inspired diverse-start demo collection (2026-08-27).

Instead of explicit multi-attempt "reattempt" demonstrations (TIDE/ReTVL/RL-
finetune all failed to produce a genuine successful second-attempt regrasp via
that route -- see docs/cross_category_specialist_log.md), this collector follows
OmniReset's (arXiv:2603.15789) core insight translated from RL to BC: densely
cover the states a robot might find itself in AFTER a failed attempt with SINGLE-
ATTEMPT clean successful demonstrations, so a policy trained on this distribution
already knows what to do from any such state -- no explicit retry logic needed.

Each collected episode is EXACTLY ONE clean grasp+lift+hold success (no retry
phases, no FSM), but the STARTING configuration is diversified two ways:
  1. Object pose: WIDE randomization (configs/dr/food_shape_banana_soft_easy_wide.yaml)
     -- as if the object was lifted a bit, dropped, and bumped/rolled away.
  2. End-effector start (--near-object-start-prob fraction of episodes): instead
     of starting at the fixed home pose, the EE starts partway along a path toward
     a "prior attempt" target (the current CMA-ES grasp target + a random offset,
     representing where a previous attempt was aimed before the object moved) --
     an UNRECORDED pre-roll -- and the RECORDED approach phase then interpolates
     from THAT off-target point toward the actual (current) grasp target. This
     kink -- "was heading toward the old spot, then corrected toward the new
     grasp" -- is exactly the redirect behavior a genuine regrasp needs, taught
     here as a normal single continuous successful approach, not a multi-attempt
     demo.

Also implements early-success termination (trims the recorded trajectory once the
object has held above a height threshold for enough consecutive steps, instead of
padding out to the fixed lift+hold length) -- shorter, less redundant demos.

Reuses collect_demos_synth.py's (v1) helpers unchanged: CMA-ES grasp synthesis
(_synth_worker/_synth_bounds), action inversion, RawObs building, output sharding.
The scripted trajectory logic (execute_and_collect) is reimplemented here to add
the pre-roll + early-termination behavior -- v1's version stays untouched.

Usage:
    uv run --project envs/sim python grasp_synthesis/collect_demos_diverse_start.py \\
        --experiment single_lift_banana_soft_easy_diverse \\
        --n-episodes 450 --n-envs 5 --near-object-start-prob 0.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

ROOT = Path(__file__).resolve().parent.parent
GRASP_DIR = ROOT / "grasp_synthesis"
for _p in (str(ROOT), str(GRASP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_demos_synth as v1  # noqa: E402  reuse constants + helpers unchanged
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from gentle_manip.actions.action_config import ActionConfig  # noqa: E402
from gentle_manip.envs.genesis_worker import GenesisWorker  # noqa: E402
from gentle_manip.experiment import Experiment  # noqa: E402
from gentle_manip.perception.pipeline import PerceptionPipeline  # noqa: E402
from gentle_manip.domain_randomization.dr_config import DRConfig  # noqa: E402
from gentle_manip.tasks.single_lift import SingleLiftTask  # noqa: E402

# ── Early-success-termination constants (mirrors v3's FAST_SUCCESS_* convention) ──
SUCCESS_HEIGHT     = 0.10   # metres above grasp_z -- lower than v1's full LIFT_HEIGHT
                            # (0.2) target since we don't need the object at the very
                            # top of the lift motion to call the demo "done", just
                            # confirmed off the table and rising.
SUCCESS_HOLD_STEPS = 10     # consecutive steps object must stay >= SUCCESS_HEIGHT
                            # above grasp_z before the recorded trajectory is trimmed.
TRIM_MARGIN_STEPS  = 5      # extra steps kept AFTER the hold confirms, for a clean
                            # "settled" ending rather than cutting the instant it fires.

# ── Near-object-start constants ──
NEAR_START_SKIP_FRAC_RANGE = (0.5, 0.85)   # fraction of the home->grasp path run
                                           # UNRECORDED before recording begins
PRIOR_TARGET_OFFSET_RANGE  = (0.02, 0.06)  # metres -- how far the unrecorded pre-roll's
                                           # target is from the ACTUAL (current) grasp
                                           # target, representing "aimed at the object's
                                           # old (pre-fall) position"


def _wxyz_to_rot(q): return Rot.from_quat([q[1], q[2], q[3], q[0]])


def _rot_to_wxyz(r: Rot) -> np.ndarray:
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z], np.float32)


def execute_and_collect_diverse(
    worker: GenesisWorker,
    all_best_x: List[np.ndarray],
    init_obs_batch: dict,
    perception: PerceptionPipeline,
    action_config: ActionConfig,
    near_object_start: np.ndarray,   # (N,) bool -- per-env: use the diverse EE start?
    rng: np.random.Generator,
    priv_cfg=None,
    dr_vec=None,
):
    """Like v1's execute_and_collect, but with an unrecorded near-object pre-roll
    for envs flagged in `near_object_start`, and early-success trimming of the
    recorded trajectory. Returns the same tuple shape as v1 (obs_bufs, act_bufs,
    rew_bufs, success, frame_bufs=[] always -- video recording not supported here,
    this collector is data-volume-oriented, not video-oriented)."""
    scales = (np.asarray(action_config.scales, dtype=np.float64)
             if action_config.mode != "absolute" else None)
    num_envs = worker.num_envs
    poses    = [v1._x_to_targets(x, 1) for x in all_best_x]
    pos_b    = np.concatenate([p[0] for p in poses], axis=0).astype(np.float32)
    quat_b   = np.concatenate([p[1] for p in poses], axis=0).astype(np.float32)
    grasp_pos = pos_b.copy()

    width_open = np.full(num_envs, 0.08, np.float32)
    width_cls  = np.array([p[2] - 0.0025 for p in poses], np.float32)

    home_pos  = np.tile(worker.robot.home_pos[None].astype(np.float32),  (num_envs, 1))
    home_quat = np.tile(worker.robot.home_quat[None].astype(np.float32), (num_envs, 1))

    # ── Per-env "prior attempt" target + skip fraction (near-object-start envs only) ──
    skip_frac = np.zeros(num_envs, np.float32)
    prior_pos = pos_b.copy()   # default: no offset (standard-start envs never use this)
    for i in range(num_envs):
        if near_object_start[i]:
            skip_frac[i] = rng.uniform(*NEAR_START_SKIP_FRAC_RANGE)
            offset_mag = rng.uniform(*PRIOR_TARGET_OFFSET_RANGE)
            direction = rng.normal(size=3).astype(np.float32)
            direction /= (np.linalg.norm(direction) + 1e-8)
            prior_pos[i] = pos_b[i] + offset_mag * direction

    def _lerp(a, b, alpha_arr):
        return a + alpha_arr[:, None] * (b - a)

    def _slerp_batch(quat_a, quat_b_, alpha_arr):
        rows = []
        for i in range(num_envs):
            s = Slerp([0., 1.], Rot.concatenate([_wxyz_to_rot(quat_a[i]), _wxyz_to_rot(quat_b_[i])]))
            rows.append(_rot_to_wxyz(s(float(alpha_arr[i]))))
        return np.stack(rows).astype(np.float32)

    # ── Unrecorded pre-roll (home -> prior_pos), per-env variable step count ──
    n_preroll_steps = int(np.ceil((skip_frac * v1.N_HOME_TO_PRE).max())) if near_object_start.any() else 0
    for j in range(n_preroll_steps):
        # alpha_j: for env i, progress toward its own skip_frac*N_HOME_TO_PRE step count.
        # envs with fewer pre-roll steps (or none) just hold at their final pre-roll pose
        # once j exceeds their own count (no-op deltas -- harmless, unrecorded).
        env_n = np.maximum(np.round(skip_frac * v1.N_HOME_TO_PRE), 1).astype(np.int32)
        alpha = np.clip((j + 1) / np.maximum(env_n, 1), 0.0, 1.0)
        alpha = np.where(near_object_start, alpha, 0.0)   # standard-start envs: stay at home
        cur_pos  = _lerp(home_pos, prior_pos, alpha)
        cur_quat = _slerp_batch(home_quat, quat_b, alpha)   # orient toward the eventual grasp
                                                            # throughout (position-only offset)
        worker.step(cur_pos, cur_quat, width_open)   # NOT recorded

    # Recorded-phase start pose = wherever the pre-roll left off (home for standard-start envs)
    rec_start_pos  = _lerp(home_pos, prior_pos, skip_frac)
    rec_start_quat = _slerp_batch(home_quat, quat_b, skip_frac)
    n_recorded_approach = np.maximum(
        v1.N_HOME_TO_PRE - np.round(skip_frac * v1.N_HOME_TO_PRE).astype(np.int32), 1)
    n_recorded_approach_max = int(n_recorded_approach.max())

    def _wxyz_to_rot_(q): return _wxyz_to_rot(q)
    recorded_slerps = [Slerp([0., 1.], Rot.concatenate(
        [_wxyz_to_rot_(rec_start_quat[i]), _wxyz_to_rot_(quat_b[i])])) for i in range(num_envs)]

    def _interp_recorded_quat(alpha_arr) -> np.ndarray:
        rows = []
        for i, s in enumerate(recorded_slerps):
            rows.append(_rot_to_wxyz(s(float(alpha_arr[i]))))
        return np.stack(rows).astype(np.float32)

    obs_bufs:   List[List[dict]]       = [[] for _ in range(num_envs)]
    act_bufs:   List[List[np.ndarray]] = [[] for _ in range(num_envs)]
    rew_bufs:   List[List[float]]      = [[] for _ in range(num_envs)]
    height_bufs: List[List[float]]     = [[] for _ in range(num_envs)]   # obj_z - grasp_z, per step

    cur_obs_list = [{k: init_obs_batch[k][i] for k in init_obs_batch} for i in range(num_envs)]
    # NOTE: cur_obs_list here is the obs from BEFORE the pre-roll (reset-time obs) -- it is
    # only used as obs_0's *content placeholder* for the very first recorded step below, same
    # convention as v1 (obs_t is recorded alongside the ACTION that was taken FROM that obs).
    # Since the pre-roll already moved the arm, this is slightly stale for near-object-start
    # envs' first recorded frame (it shows the pre-pre-roll scene) -- acceptable: it's exactly
    # one frame, and PerceptionPipeline.process() is called fresh on every subsequent step.

    prev_pos  = rec_start_pos.copy()
    prev_quat = rec_start_quat.copy()
    prev_grip = width_open.copy()

    def _step(cur_pos, cur_quat, cur_grip):
        nonlocal prev_pos, prev_quat, prev_grip
        if action_config.mode == "absolute":
            actions = v1._invert_actions_absolute(cur_pos, cur_quat, cur_grip, action_config)
        else:
            actions = v1._invert_actions(prev_pos, cur_pos, prev_quat, cur_quat,
                                         prev_grip, cur_grip, scales)
        state = worker.step(cur_pos, cur_quat, cur_grip)
        raw_next = v1._state_to_raw_obs(state)
        next_obs_batch = perception.process(raw_next)
        if priv_cfg is not None:
            next_obs_batch.update(v1._privileged_obs_batch(
                state["object_center"], state["object_quat"], dr_vec, priv_cfg,
                contact_force=state.get("contact_force")))
        next_obs_list = [{k: next_obs_batch[k][i] for k in next_obs_batch} for i in range(num_envs)]
        obj_z = state["object_center"][:, 2]
        for i in range(num_envs):
            obs_bufs[i].append(cur_obs_list[i])
            act_bufs[i].append(actions[i])
            rew_bufs[i].append(0.0)
            height_bufs[i].append(float(obj_z[i] - grasp_pos[i, 2]))
        prev_pos[:]  = cur_pos
        prev_quat[:] = cur_quat
        prev_grip[:] = cur_grip
        return next_obs_list

    # ── Recorded phase 1: (pre-roll end or home) -> grasp pose ──
    for j in range(n_recorded_approach_max):
        alpha = np.clip((j + 1) / n_recorded_approach, 0.0, 1.0)
        cur_obs_list = _step(_lerp(rec_start_pos, pos_b, alpha), _interp_recorded_quat(alpha), width_open)

    # ── Phase 1b: settle ──
    for _ in range(v1.N_SETTLE):
        cur_obs_list = _step(pos_b, quat_b, width_open)

    # ── Phase 2: close gripper ──
    for j in range(v1.N_GRASP):
        alpha = (j + 1) / v1.N_GRASP
        cur_obs_list = _step(pos_b, quat_b, width_open + alpha * (width_cls - width_open))

    # ── Phase 3: lift ──
    lift_b = grasp_pos.copy(); lift_b[:, 2] += v1.LIFT_HEIGHT
    for j in range(v1.N_LIFT):
        alpha = (j + 1) / v1.N_LIFT
        cur_obs_list = _step(pos_b + alpha * (lift_b - pos_b), quat_b, width_cls)

    # ── Phase 4: hold ──
    for _ in range(v1.N_HOLD):
        cur_obs_list = _step(lift_b, quat_b, width_cls)

    # Final object height: MPMEntity has no get_pos() (rigid-only method) -- use the
    # per-step object_center already tracked in height_bufs (= obj_z - grasp_z) instead.
    obj_z_final = np.array([grasp_pos[i, 2] + height_bufs[i][-1] for i in range(num_envs)])
    success = obj_z_final > (grasp_pos[:, 2] + v1.LIFT_HEIGHT * 0.5)

    # ── Early-success trim: cut each env's recorded trajectory shortly after it
    #    first held >= SUCCESS_HEIGHT for SUCCESS_HOLD_STEPS consecutive steps ──
    for i in range(num_envs):
        h = np.asarray(height_bufs[i])
        held = h >= SUCCESS_HEIGHT
        run = 0
        cut_at = None
        for t in range(len(held)):
            run = run + 1 if held[t] else 0
            if run >= SUCCESS_HOLD_STEPS:
                cut_at = min(t + TRIM_MARGIN_STEPS + 1, len(held))
                break
        if cut_at is not None and cut_at < len(obs_bufs[i]):
            obs_bufs[i] = obs_bufs[i][:cut_at]
            act_bufs[i] = act_bufs[i][:cut_at]
            rew_bufs[i] = rew_bufs[i][:cut_at]

    # n_preroll_steps: worker.step() calls BEFORE recording starts (shared across the
    # whole batch -- standard-start envs just no-op through it). Exposed so a caller
    # capturing frames via an outer worker.step wrap (e.g. for video) can slice off the
    # unrecorded pre-roll and show exactly what the RECORDED/training episode starts
    # from -- for near_object_start envs that's a diverse near-object pose, not home.
    return obs_bufs, act_bufs, rew_bufs, success, [[] for _ in range(num_envs)], n_preroll_steps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True)
    p.add_argument("--task-name", default=None)
    p.add_argument("--out-dir", type=Path, default=Path("dataset") / "demos")
    p.add_argument("--shard-size", type=int, default=5)
    p.add_argument("--description", type=str, default="")
    p.add_argument("--n-episodes", type=int, default=450)
    p.add_argument("--n-envs", type=int, default=5)
    p.add_argument("--maxfevals", type=int, default=901)
    p.add_argument("--scene-dr-every", type=int, default=1)
    p.add_argument("--settle", type=int, default=None)
    p.add_argument("--settle-max", type=int, default=None)
    p.add_argument("--settle-vel-thresh", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument("--near-object-start-prob", type=float, default=0.5,
                   help="fraction of episodes that start the EE partway toward an "
                        "offset 'prior attempt' target instead of always home")
    args = p.parse_args()

    exp        = Experiment.load(args.experiment)
    task       = SingleLiftTask(exp.task_cfg)
    spec       = task.scene_spec
    obs_config = exp.collection_obs()
    priv_cfg   = obs_config.privileged
    action_config = exp.action_config
    dr_cfg     = DRConfig.from_dict(exp.dr)
    task_name  = args.task_name or exp._raw.get("task", args.experiment)
    rate_hz    = 1.0 / spec.sim_dt

    settle_steps      = args.settle           or int(exp.task_cfg.get("settle_steps",     30))
    settle_max_steps  = args.settle_max       or int(exp.task_cfg.get("settle_max_steps", 200))
    settle_vel_thresh = args.settle_vel_thresh or float(exp.task_cfg.get("settle_vel_thresh", 0.002))

    perception = PerceptionPipeline(obs_config)

    collection_config = {
        "task_name": task_name, "description": args.description,
        "source": "cmaes_synth_diverse_start", "git_commit": v1._git_commit(),
        "experiment": args.experiment,
        "control": {"n_envs": args.n_envs, "maxfevals": args.maxfevals,
                    "n_episodes": args.n_episodes, "scene_dr_every": args.scene_dr_every,
                    "seed": args.seed, "near_object_start_prob": args.near_object_start_prob,
                    "success_height": SUCCESS_HEIGHT, "success_hold_steps": SUCCESS_HOLD_STEPS},
        "dr": exp.dr,
    }

    print(f"\n=== collect_demos_diverse_start  experiment={args.experiment}"
         f" — target {args.n_episodes} episodes, {args.n_envs} envs/batch,"
         f" near_object_start_prob={args.near_object_start_prob}")

    rng = np.random.default_rng(args.seed)

    nominal_spec = spec
    do_scene_dr  = args.scene_dr_every > 0 and dr_cfg.has_scene_dr()
    import tempfile
    deform_dir = tempfile.mkdtemp(prefix="gm_synth_deform_diverse_") if do_scene_dr else None

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
        print(f"  scene DR ON (every {args.scene_dr_every} batch(es)) — deformed meshes → {deform_dir}")

    left_pts  = v1.sample_finger_surface(v1.LEFT_FINGER,  n=300)
    right_pts = v1.sample_finger_surface(v1.RIGHT_FINGER, n=300)
    executor = ProcessPoolExecutor(max_workers=args.n_envs)
    print(f"  Mesh: {Path(actual_mesh).name}")

    run_dir  = v1._make_run_dir(args.out_dir, task_name)
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(collection_config, f, sort_keys=False)
    print(f"  Config → {cfg_path.resolve()}")
    print(f"  Data   → {run_dir.resolve()}/data.pkl  (shards flushed every {args.shard_size} ep)")

    total_saved = 0
    total_failed = 0
    batch_idx = 0
    shard_buf: List[dict] = []
    shard_idx = 0
    t0 = time.time()

    while total_saved < args.n_episodes:
        batch_idx += 1
        n = args.n_envs

        if do_scene_dr and batch_idx > 1 and (batch_idx - 1) % args.scene_dr_every == 0:
            worker.close()
            worker, scene_dr, actual_mesh = _make_worker()

        print(f"\n── Batch {batch_idx}  [{total_saved}/{args.n_episodes} saved]"
             + (f"  scale={scene_dr['scale']:.3f} bend={scene_dr['bend_deg']:+.1f}°" if do_scene_dr else "") + " ──")

        object_dxy   = dr_cfg.sample_object_dxy(rng, n)
        object_euler = dr_cfg.sample_object_euler(rng, n)
        home_offset  = dr_cfg.sample_home_offset(rng, n)
        worker.reset(object_dxy=object_dxy, object_euler=object_euler, home_offset=home_offset)

        obj = worker.handle.objects[0]
        # Extra rigid-body settling only -- MPMEntity has no get_vel()/get_ang()
        # (only get_particles_vel(), a per-particle field), and worker.reset()'s
        # own settle_steps/settle_max_steps/settle_vel_thresh already handle
        # settling for soft/MPM objects during reset itself.
        if task.object_type != "soft":
            for _ in range(600):
                worker.handle.scene.step()
                lin = np.abs(v1._np(obj.get_vel())).max()
                ang = np.abs(v1._np(obj.get_ang())).max()
                if lin < 0.003 and ang < 0.01:
                    break

        init_state = worker.read_state()
        raw_init = v1._state_to_raw_obs(init_state)
        init_obs_batch = perception.process(raw_init)
        dr_vec = np.array([float(scene_dr.get("scale", 1.0)), float(scene_dr.get("bend_deg", 0.0))], dtype=np.float32)
        if priv_cfg is not None:
            init_obs_batch.update(v1._privileged_obs_batch(
                init_state["object_center"], init_state["object_quat"], dr_vec, priv_cfg,
                contact_force=init_state.get("contact_force")))

        obj_pos_all  = init_state["object_center"].astype(np.float64)
        # object_quat: use worker.read_state()'s already-computed value (correct for
        # BOTH rigid (obj.get_quat()) and soft/MPM (Kabsch-fit to particles) objects)
        # rather than calling obj.get_quat() directly, which does not exist on MPMEntity.
        obj_quat_all = init_state["object_quat"].astype(np.float64)

        payloads = []
        for i in range(n):
            lb, ub = v1._synth_bounds(obj_pos_all[i])
            payloads.append((actual_mesh, obj_pos_all[i], obj_quat_all[i],
                             left_pts, right_pts, args.maxfevals, lb, ub, str(run_dir / "cmaes_logs")))
        futures = [executor.submit(v1._synth_worker, p) for p in payloads]
        all_best_x = []
        for i, fut in enumerate(futures):
            best_x, score = fut.result()
            all_best_x.append(best_x)
            print(f"  Env {i}: cost={score:.4f}  tcp={best_x[:3].round(4)}  w={best_x[6]*1e3:.1f} mm")

        near_object_start = rng.random(n) < args.near_object_start_prob
        print(f"  near_object_start: {near_object_start.tolist()}")

        print(f"  Executing …")
        obs_bufs, act_bufs, rew_bufs, success, _, _n_preroll = execute_and_collect_diverse(
            worker, all_best_x, init_obs_batch, perception, action_config,
            near_object_start, rng, priv_cfg=priv_cfg, dr_vec=dr_vec,
        )
        print(f"  Success: {success.tolist()}")

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
                "near_object_start": bool(near_object_start[i]),
            }
            shard_buf.append(episode)
            total_saved += 1
            print(f"    ep {total_saved}: env {i}  {'✓' if success[i] else '✗'}  "
                 f"T={episode['actions'].shape[0]}  near_start={near_object_start[i]}")

            if len(shard_buf) >= args.shard_size:
                sp = v1._write_shard(run_dir, shard_buf, task_name, shard_idx, rate_hz)
                print(f"  Shard {shard_idx} → {sp.name}")
                shard_idx += 1
                shard_buf = []

            if total_saved >= args.n_episodes:
                break

    if shard_buf:
        v1._write_shard(run_dir, shard_buf, task_name, shard_idx, rate_hz)

    data_path = v1._merge_shards(run_dir)
    elapsed = time.time() - t0
    total_attempts = total_saved + total_failed
    success_rate = total_saved / total_attempts if total_attempts > 0 else 0.0

    print(f"\n=== Done ===")
    print(f"  Episodes saved   : {total_saved}")
    print(f"  Episodes failed  : {total_failed}")
    print(f"  Total attempts   : {total_attempts}")
    print(f"  Success rate     : {success_rate*100:.1f}%")
    print(f"  Elapsed          : {elapsed/60:.1f} min")
    print(f"  Data             : {data_path}")

    stats = {"episodes_saved": total_saved, "episodes_failed": total_failed,
            "total_attempts": total_attempts, "success_rate": round(success_rate, 4),
            "elapsed_min": round(elapsed / 60, 2)}
    stats_path = run_dir / "stats.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False)
    print(f"  Stats            : {stats_path}")

    executor.shutdown(wait=False)
    worker.close()


if __name__ == "__main__":
    main()
