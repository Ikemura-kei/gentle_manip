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

    payload: (mesh_path, obj_pos, obj_quat_wxyz, left_pts, right_pts, maxfevals, lb, ub, log_dir)
    returns: (best_x (7,), score float)
    """
    mesh_path, obj_pos, obj_quat_wxyz, left_pts, right_pts, maxfevals, lb, ub, log_dir = payload
    from synth_utils import build_object_sdf, grasp_cost, run_cmaes  # local import: safe in subprocess
    sdf_fn = build_object_sdf(mesh_path)
    x0 = [(lo + hi) / 2 for lo, hi in zip(lb, ub)]

    def objective(x):
        return grasp_cost(x, left_pts, right_pts, sdf_fn, obj_pos, obj_quat_wxyz)

    best_x, score = run_cmaes(objective, x0, 1.0, lb, ub, maxfevals, log_dir=log_dir)
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

def _privileged_obs_batch(object_center, object_quat, dr_vec, priv_cfg) -> dict:
    """Sim-only privileged fields from raw worker state — mirrors
    PolicyEnv._privileged_obs exactly, but sourced from GenesisWorker state
    (object_center + object_quat) since this collector bypasses PolicyEnv.

    object_center: (N, 3); object_quat: (N, 4) wxyz; dr_vec: (2,) [scale, bend_deg].
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


# ── Episode execution + collection ───────────────────────────────────────────

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
    """Execute scripted grasp trajectory; record (obs, action, reward) per env.

    The scripted trajectory itself (positions/quats/gripper widths, all physical
    units) is IDENTICAL regardless of action_config.mode — only how each step's
    *recorded action* is derived (delta-inverted vs absolute-inverted) differs.

    Returns:
        obs_bufs:    list[N] of obs-dict lists (T steps, unbatched per-env)
        act_bufs:    list[N] of float32 action arrays (T steps each; 7-dim delta or
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

    width_open = np.full(num_envs, 0.08, np.float32)
    width_cls  = np.array([p[2] - 0.0025 for p in poses], np.float32)

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

        # Invert scripted target → normalized action (delta needs prev_*; absolute
        # is a stateless per-step transform of the current target alone).
        if action_config.mode == "absolute":
            actions = _invert_actions_absolute(cur_pos, cur_quat, cur_grip, action_config)  # (N, 10)
        else:
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

        # Build obs from next state (+ sim-only privileged fields from the object state)
        raw_next = _state_to_raw_obs(state)
        next_obs_batch = perception.process(raw_next)
        if priv_cfg is not None:
            next_obs_batch.update(_privileged_obs_batch(
                state["object_center"], state["object_quat"], dr_vec, priv_cfg))
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

    # ── Phase 2: close gripper (gradual — speed = Δwidth / N_GRASP per step) ──
    for j in range(N_GRASP):
        alpha = (j + 1) / N_GRASP
        cur_obs_list = _step(pos_b, quat_b,
                             width_open + alpha * (width_cls - width_open))

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
    p.add_argument("--maxfevals",  type=int, default=901,
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
    executor = ProcessPoolExecutor(max_workers=args.n_envs)
    print(f"  Mesh: {Path(actual_mesh).name}")

    # ── Output dir + config snapshot ──
    run_dir  = _make_run_dir(args.out_dir, task_name)
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(collection_config, f, sort_keys=False)
    print(f"  Config → {cfg_path.resolve()}")
    print(f"  Data   → {run_dir.resolve()}/data.pkl  (shards flushed every {args.shard_size} ep)")

    total_saved  = 0
    total_failed = 0
    batch_idx   = 0
    shard_buf:  List[dict] = []
    shard_idx   = 0
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
        # Episode scene-DR vector [scale, bend_deg] for priv_object_dr_params (mirrors
        # SimBackend._episode_dr_vec). scene_dr may re-randomize this per batch (below).
        dr_vec = np.array([float(scene_dr.get("scale", 1.0)),
                           float(scene_dr.get("bend_deg", 0.0))], dtype=np.float32)
        if priv_cfg is not None:
            init_obs_batch.update(_privileged_obs_batch(
                init_state["object_center"], init_state["object_quat"], dr_vec, priv_cfg))

        obj_pos_all  = init_state["object_center"].astype(np.float64)  # (N, 3)
        obj_quat_all = _np(obj.get_quat()).astype(np.float64)           # (N, 4) wxyz

        # ── Per-env CMA-ES grasp synthesis (parallel — one subprocess per env) ──
        payloads = []
        for i in range(n):
            lb, ub = _synth_bounds(obj_pos_all[i])
            payloads.append((actual_mesh, obj_pos_all[i], obj_quat_all[i],
                             left_pts, right_pts, args.maxfevals, lb, ub,
                             str(run_dir / "cmaes_logs")))
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
                vid_path = vid_dir / f"ep{total_saved:04d}_env{i}_success.mp4"
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
