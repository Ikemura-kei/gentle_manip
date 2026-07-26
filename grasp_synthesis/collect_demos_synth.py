"""Autonomous sim demo collection via CMA-ES grasp synthesis.

Collects demonstrations by: reset with pose DR → per-env CMA-ES synthesis →
scripted execution, recording (obs, action, reward) tuples in exactly the
same format as demos/record.py.  Drop-in data source for DP3 training.

Key differences from demos/record.py:
  - Runs in envs/sim (Python 3.12 + Genesis), not envs/deploy.
  - No teleop device — grasps are synthesized automatically.
  - Actions are inverted from the scripted trajectory (delta commands that
    reproduce each planned motion step).
  - --setup absent (sim, no hardware).  --obs-config / --action-config /
    --task-name / --out-dir / --shard-size / --description match record.py.
  - Rewards are 0 (matching teleop task=None).
  - Sim runs at 30 Hz (dt=1/30); rate_hz=30 in the dataset meta.

Constraint: the scene always builds cam_ext (pos=(0.9,0.35,0.55), fov=50°,
640×480).  The obs config must reference camera names present in the scene
(point_cloud_1cam.yaml with cam_ext works out of the box).

Usage:
    uv run --project envs/sim python grasp_synthesis/collect_demos_synth.py \\
        --obs-config gentle_manip/configs/obs/point_cloud_1cam.yaml \\
        --task-name single_lift_mushroom_rigid \\
        --n-episodes 50 --n-envs 5
"""
from __future__ import annotations

import argparse
import datetime
import os
import pickle
import random
import string
import subprocess
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

# headless by default; MUJOCO_GL=glfw in the shell to get the viewer
os.environ.setdefault("MUJOCO_GL", "egl")

from synth_utils import (  # noqa: E402
    build_object_sdf, sample_finger_surface, grasp_cost, run_cmaes,
    FINGER_TO_TCP_Z,
)
from gentle_manip.scenes.scene_spec import CameraEntry, FixtureEntry, ObjectEntry, SceneSpec
from gentle_manip.envs.genesis_worker import GenesisWorker
from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.perception.pipeline import PerceptionPipeline
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.actions.action_config import ActionConfig


# ── Constants (keep in sync with run_grasp_synth.py) ─────────────────────────

RATE_HZ       = 30           # sim dt = 1/RATE_HZ → control rate in meta
N_HOME_TO_PRE = 105          # home → grasp pose interpolation steps
N_SETTLE      = 4           # hold at grasp pose before closing
N_GRASP       = 95           # gripper close steps
N_LIFT        = 90          # lift steps
N_HOLD        = 60           # hold at lift height (success eval window)
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


# ── Scene ─────────────────────────────────────────────────────────────────────

def _build_spec() -> SceneSpec:
    return SceneSpec(
        objects=[ObjectEntry(name="mushroom", object_type="rigid")],
        fixtures=[FixtureEntry(fixture_type="table")],
        cameras=[CameraEntry(
            name="cam_ext",
            pos=(0.9, 0.35, 0.55),
            lookat=(0.4, 0.0, 0.1),
            fov=50.0,
            resolution=(640, 480),
        )],
        sim_dt=1 / RATE_HZ,
        sim_substeps=10,
    )


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synthesize_one(
    obj_pos: np.ndarray,
    obj_quat_wxyz: np.ndarray,
    sdf_fn,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    maxfevals: int,
) -> np.ndarray:
    """CMA-ES grasp synthesis for one env. Returns best_x (7,)."""
    t_lb_xy = (obj_pos[:2] - 1.5 * OBJ_SIZE[:2]).tolist()
    t_ub_xy = (obj_pos[:2] + 1.5 * OBJ_SIZE[:2]).tolist()
    tcp_z_min = float(obj_pos[2]) + FINGER_TO_TCP_Z - 0.04
    tcp_z_max = float(obj_pos[2]) + 0.25
    lb = t_lb_xy + [tcp_z_min, 0.8 * np.pi, -0.25 * np.pi, -0.25 * np.pi, 0.01]
    ub = t_ub_xy + [tcp_z_max, 1.0 * np.pi,  0.25 * np.pi,  0.25 * np.pi, 0.08]
    x0 = [(lo + hi) / 2 for lo, hi in zip(lb, ub)]

    def objective(x):
        return grasp_cost(x, left_pts, right_pts, sdf_fn, obj_pos, obj_quat_wxyz)

    best_x, score = run_cmaes(objective, x0, 1.0, lb, ub, maxfevals)
    print(f"    cost={score:.4f}  tcp={best_x[:3].round(4)}  w={best_x[6]*1e3:.1f} mm")
    return best_x


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


# ── Episode execution + collection ───────────────────────────────────────────

def execute_and_collect(
    worker:       GenesisWorker,
    all_best_x:   List[np.ndarray],
    init_obs_batch: dict,          # batched obs from reset, keyed by obs name → (N, ...)
    perception:   PerceptionPipeline,
    scales:       np.ndarray,      # (7,) action scales
    record_video: bool = False,
) -> Tuple[List[List[dict]], List[List[np.ndarray]], List[List[float]], np.ndarray, List[List]]:
    """Execute scripted grasp trajectory; record (obs, action, reward) per env.

    Returns:
        obs_bufs:    list[N] of obs-dict lists (T steps, unbatched per-env)
        act_bufs:    list[N] of (7,) float32 action arrays (T steps each)
        rew_bufs:    list[N] of float rewards (0.0 throughout)
        success:     (N,) bool — True if final object z > grasp_z + 0.5*LIFT_HEIGHT
        frame_bufs:  list[N] of (H,W,3) uint8 frame lists; empty lists if record_video=False
    """
    num_envs = worker.num_envs
    poses    = [_x_to_targets(x, 1) for x in all_best_x]
    pos_b    = np.concatenate([p[0] for p in poses], axis=0).astype(np.float32)  # (N, 3)
    quat_b   = np.concatenate([p[1] for p in poses], axis=0).astype(np.float32)  # (N, 4)
    grasp_pos = pos_b.copy()

    width_open = np.full(num_envs, 0.08, np.float32)
    width_cls  = np.array([p[2] - 0.001 for p in poses], np.float32)

    home_pos  = np.tile(worker.robot.home_pos[None].astype(np.float32),  (num_envs, 1))
    home_quat = np.tile(worker.robot.home_quat[None].astype(np.float32), (num_envs, 1))

    def _wxyz_to_rot(q): return Rot.from_quat([q[1], q[2], q[3], q[0]])
    home_r = _wxyz_to_rot(home_quat[0])
    slerps = [Slerp([0., 1.], Rot.concatenate([home_r, _wxyz_to_rot(quat_b[i])]))
              for i in range(num_envs)]

    def _interp_quat(alpha: float) -> np.ndarray:
        rows = []
        for s in slerps:
            xyzw = s(alpha).as_quat()
            rows.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        return np.array(rows, np.float32)

    # Per-env buffers
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

    def _step(cur_pos, cur_quat, cur_grip):
        nonlocal prev_pos, prev_quat, prev_grip

        # Invert scripted delta → normalized action
        actions = _invert_actions(prev_pos, cur_pos, prev_quat, cur_quat,
                                  prev_grip, cur_grip, scales)  # (N, 7)

        # Advance sim (depth rendered because render_obs_cameras=True)
        state = worker.step(cur_pos, cur_quat, cur_grip)

        # RGB frames (optional)
        if record_video:
            frames = worker.render_rgb(all_envs=True)   # (N, H, W, 3) uint8
            if frames is not None:
                for i in range(num_envs):
                    frame_bufs[i].append(frames[i])

        # Build obs from next state
        raw_next = _state_to_raw_obs(state)
        next_obs_batch = perception.process(raw_next)
        next_obs_list = [{k: next_obs_batch[k][i] for k in next_obs_batch}
                         for i in range(num_envs)]

        # Record (obs_t, action_t, reward_t=0)
        for i in range(num_envs):
            obs_bufs[i].append(cur_obs_list[i])
            act_bufs[i].append(actions[i])
            rew_bufs[i].append(0.0)

        prev_pos[:]  = cur_pos
        prev_quat[:] = cur_quat
        prev_grip[:] = cur_grip
        return next_obs_list

    # ── Phase 1: home → grasp ──
    for j in range(N_HOME_TO_PRE):
        alpha = (j + 1) / N_HOME_TO_PRE
        cur_obs_list = _step(home_pos + alpha * (pos_b - home_pos),
                             _interp_quat(alpha), width_open)

    # ── Phase 1b: settle at grasp pose (action ≈ 0) ──
    for _ in range(N_SETTLE):
        cur_obs_list = _step(pos_b, quat_b, width_open)

    # ── Phase 2: close gripper ──
    for _ in range(N_GRASP):
        cur_obs_list = _step(pos_b, quat_b, width_cls)

    # ── Phase 3: lift ──
    lift_b = grasp_pos.copy(); lift_b[:, 2] += LIFT_HEIGHT
    for j in range(N_LIFT):
        alpha = (j + 1) / N_LIFT
        cur_obs_list = _step(pos_b + alpha * (lift_b - pos_b), quat_b, width_cls)

    # ── Phase 4: hold ──
    for _ in range(N_HOLD):
        cur_obs_list = _step(lift_b, quat_b, width_cls)

    # Success check from final object position
    obj_z   = _np(worker.handle.objects[0].get_pos())[:, 2]
    success = obj_z > (grasp_pos[:, 2] + LIFT_HEIGHT * 0.5)
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
                 task: str, idx: int) -> Path:
    first = episodes[0]
    payload = {
        "meta": {
            "task": task,
            "obs_keys": sorted(first["observations"].keys()),
            "action_dim": int(first["actions"].shape[1]),
            "rate_hz": float(RATE_HZ),
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

    # record.py-compatible args
    p.add_argument("--obs-config",    type=Path,
                   default=ROOT / "gentle_manip/configs/obs/point_cloud_1cam.yaml",
                   help="obs config (must reference only cam_ext)")
    p.add_argument("--action-config", type=Path,
                   default=ROOT / "gentle_manip/configs/action/delta_pose_delta_gripper.yaml")
    p.add_argument("--task-name",   required=True,
                   help="e.g. single_lift_mushroom_rigid")
    p.add_argument("--out-dir",     type=Path, default=Path("dataset") / "demos")
    p.add_argument("--shard-size",  type=int, default=5,
                   help="episodes per shard file (merged into data.pkl at end)")
    p.add_argument("--description", type=str, default="",
                   help="free-text annotation saved in config.yaml")

    # Synth-specific args
    p.add_argument("--n-episodes",  type=int, default=50,
                   help="total successful episodes to collect")
    p.add_argument("--n-envs",      type=int, default=5,
                   help="parallel envs per batch (CMA-ES runs sequentially per env)")
    p.add_argument("--maxfevals",   type=int, default=1100,
                   help="CMA-ES function evaluations per env per batch")
    p.add_argument("--settle",      type=int, default=30,
                   help="genesis worker base settle steps after reset")
    p.add_argument("--seed",        type=int, default=0,
                   help="RNG seed for pose DR")
    p.add_argument("--xy-range",    type=float, default=0.04,
                   help="per-env xy jitter range (m)")
    p.add_argument("--pitch-roll-range", type=float, default=45.0,
                   help="per-env pitch/roll DR range (degrees); yaw is always full 360°")
    p.add_argument("--keep-failures", action="store_true",
                   help="also save episodes where the grasp failed (default: success only)")
    p.add_argument("--record-video", action="store_true",
                   help="write per-episode mp4 videos to <out-dir>/videos/ (slower)")
    args = p.parse_args()

    def _load_yaml(path: Path) -> dict:
        rp = path if path.is_file() else ROOT / path
        with open(rp) as f:
            return yaml.safe_load(f)

    obs_d    = _load_yaml(args.obs_config)
    action_d = _load_yaml(args.action_config)
    obs_config    = ObsConfig.from_dict(obs_d)
    action_config = ActionConfig.from_dict(action_d)
    scales = np.asarray(action_config.scales, dtype=np.float64)

    perception = PerceptionPipeline(obs_config)

    collection_config = {
        "task_name":  args.task_name,
        "description": args.description,
        "source":     "cmaes_synth",
        "git_commit": _git_commit(),
        "sources":    {"obs": str(args.obs_config), "action": str(args.action_config)},
        "control":    {
            "rate_hz": RATE_HZ, "n_envs": args.n_envs, "maxfevals": args.maxfevals,
            "n_episodes": args.n_episodes, "xy_range": args.xy_range,
            "pitch_roll_range_deg": args.pitch_roll_range,
        },
        "obs": obs_d, "action": action_d,
    }

    print(f"\n=== collect_demos_synth"
          f" — target {args.n_episodes} episodes, {args.n_envs} envs/batch")

    # ── Build scene + worker ──
    spec   = _build_spec()
    worker = GenesisWorker(spec, num_envs=args.n_envs,
                           show_viewer=False,
                           settle_steps=args.settle,
                           render_obs_cameras=True)   # depth rendered each step

    # SDF + finger geometry (built once; reads the actual post-DR mesh if any)
    actual_mesh = worker.handle.spec.objects[0].mesh_path or MUSHROOM_MESH
    print(f"  Mesh: {Path(actual_mesh).name}")
    sdf_fn    = build_object_sdf(actual_mesh)
    left_pts  = sample_finger_surface(LEFT_FINGER,  n=300)
    right_pts = sample_finger_surface(RIGHT_FINGER, n=300)

    rng = np.random.default_rng(args.seed)
    pr  = np.radians(args.pitch_roll_range)

    # ── Output dir + config snapshot ──
    run_dir  = _make_run_dir(args.out_dir, args.task_name)
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(collection_config, f, sort_keys=False)
    print(f"  Config → {cfg_path.resolve()}")
    print(f"  Data   → {run_dir.resolve()}/data.pkl  (shards flushed every {args.shard_size} ep)")

    total_saved = 0
    batch_idx   = 0
    shard_buf:  List[dict] = []
    shard_idx   = 0
    t0 = time.time()

    while total_saved < args.n_episodes:
        batch_idx += 1
        n = args.n_envs
        print(f"\n── Batch {batch_idx}  [{total_saved}/{args.n_episodes} saved] ──")

        # ── Reset with per-env pose DR ──
        object_dxy = rng.uniform(-args.xy_range, args.xy_range, (n, 2)).astype(np.float32)
        object_euler = np.stack([
            rng.uniform(-pr,      pr,      n),   # roll
            rng.uniform(-pr,      pr,      n),   # pitch
            rng.uniform(-np.pi,   np.pi,   n),   # yaw — full 360°
        ], axis=1).astype(np.float32)
        worker.reset(object_dxy=object_dxy, object_euler=object_euler)

        # Extra settling until object velocity is small
        obj = worker.handle.objects[0]
        for _ in range(600):
            worker.handle.scene.step()
            lin = np.abs(_np(obj.get_vel())).max()
            ang = np.abs(_np(obj.get_ang())).max()
            if lin < 0.003 and ang < 0.01:
                break

        # Read initial state (depth rendered; this is obs_0 for every env)
        init_state     = worker.read_state()
        raw_init       = _state_to_raw_obs(init_state)
        init_obs_batch = perception.process(raw_init)

        obj_pos_all  = init_state["object_center"].astype(np.float64)  # (N, 3)
        obj_quat_all = _np(obj.get_quat()).astype(np.float64)           # (N, 4) wxyz

        # ── Per-env CMA-ES grasp synthesis (sequential — trimesh BVH not thread-safe) ──
        all_best_x = []
        for i in range(n):
            print(f"  Env {i}: obj_pos={obj_pos_all[i].round(4)}")
            bx = _synthesize_one(obj_pos_all[i], obj_quat_all[i],
                                 sdf_fn, left_pts, right_pts, args.maxfevals)
            all_best_x.append(bx)

        # ── Execute scripted trajectory + collect data ──
        print(f"  Executing …")
        obs_bufs, act_bufs, rew_bufs, success, frame_bufs = execute_and_collect(
            worker, all_best_x, init_obs_batch, perception, scales,
            record_video=args.record_video,
        )
        print(f"  Success: {success.tolist()}")

        # ── Package and shard successful (or all) episodes ──
        for i in range(n):
            if not success[i] and not args.keep_failures:
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
            tag = "success" if success[i] else "fail"
            print(f"    ep {total_saved}: env {i}  {'✓' if success[i] else '✗'}  "
                  f"T={episode['actions'].shape[0]}")

            # Write video for this episode
            if args.record_video and frame_bufs[i]:
                import imageio
                vid_dir = run_dir / "videos"
                vid_dir.mkdir(exist_ok=True)
                vid_path = vid_dir / f"ep{total_saved:04d}_env{i}_{tag}.mp4"
                imageio.mimwrite(str(vid_path), frame_bufs[i], fps=RATE_HZ, quality=8)
                print(f"    video → {vid_path.name}")

            if len(shard_buf) >= args.shard_size:
                sp = _write_shard(run_dir, shard_buf, args.task_name, shard_idx)
                print(f"  Shard {shard_idx} → {sp.name}")
                shard_idx += 1
                shard_buf = []

            if total_saved >= args.n_episodes:
                break

    # ── Flush + merge ──
    if shard_buf:
        _write_shard(run_dir, shard_buf, args.task_name, shard_idx)

    data_path = _merge_shards(run_dir)
    elapsed   = time.time() - t0
    print(f"\n=== Done — {total_saved} episodes  {elapsed/60:.1f} min")
    print(f"    {data_path}")
    worker.close()


if __name__ == "__main__":
    main()
