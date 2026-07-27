"""Convert recorded gentle_manip demos → DPPO pretrain dataset + normalization.

Genesis-free (pure numpy). Reads the canonical superset demo pickle(s)
(``{"meta", "episodes":[{"observations":{k:(T,..)}, "actions":(T,A), "rewards":(T,)}]}``,
see ``gentle_manip.demos.record``), selects a VIEW (an ordered obs-key list — the SAME
order the sim server serves, so the pretrain data and the live env agree channel-for-channel),
and writes the DPPO contract:

  <out>/train.npz / val.npz   states,actions,rewards,terminals,traj_lengths  (normalized [-1,1])
  <out>/normalization.npz     obs_min,obs_max,action_min,action_max           (raw units)

Normalization matches DPPO exactly: ``2*(x-min)/(max-min+1e-6) - 1`` (see
``third_party/dppo/script/dataset/process_robomimic_dataset.py``). The same
``normalization.npz`` is loaded by ``GenesisMultiStepVecEnv`` at finetune time, so the
BC-pretrained policy and the online env share one normalization.

State views only (concatenable flat obs). Point-cloud views are handled separately
(the DP3/PointNet path stores clouds, not a flat state vector).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Sequence

import numpy as np

# Default STATE view — MUST match the sim server's obs order (serl_sim_server teacher view).
STATE_VIEW = ["ee_pos", "ee_quat", "gripper_width", "priv_object_pos", "priv_object_vel"]
# Full privileged state: adds object orientation + DR params (scale, bend_deg) to STATE_VIEW.
# Use with superset_rigid_full_state.yaml (object_quat + object_dr_params enabled).
STATE_VIEW_FULL = ["ee_pos", "ee_quat", "gripper_width",
                   "priv_object_pos", "priv_object_rot6d",
                   "priv_object_dr_params"]
# PROPRIO view — deployable state for the point-cloud (student) pipeline: no privileged obs,
# the PointNet consumes the cloud instead. MUST match the sim server's student-view obs order.
PROPRIO_VIEW = ["ee_pos", "ee_quat", "gripper_width"]


def _load_episodes(paths: Sequence[Path]) -> list:
    episodes = []
    for p in paths:
        d = pickle.load(open(p, "rb"))
        episodes.extend(d["episodes"])
    if not episodes:
        raise ValueError(f"no episodes found in {list(map(str, paths))}")
    return episodes


def _episode_state(ep: dict, obs_keys: Sequence[str]) -> np.ndarray:
    """(T, obs_dim) flat state = the obs_keys concatenated per timestep, in order."""
    obs = ep["observations"]
    cols = [np.asarray(obs[k], np.float32).reshape(len(ep["actions"]), -1) for k in obs_keys]
    return np.concatenate(cols, axis=1)


def convert(demo_paths: Sequence[Path], out_dir: Path, obs_keys: Sequence[str] = STATE_VIEW,
            pointcloud_key: str = None, val_split: float = 0.1, seed: int = 0) -> dict:
    episodes = _load_episodes(demo_paths)
    states = [_episode_state(ep, obs_keys) for ep in episodes]
    actions = [np.asarray(ep["actions"], np.float32) for ep in episodes]
    rewards = [np.asarray(ep["rewards"], np.float32).reshape(-1) for ep in episodes]
    obs_dim, act_dim = states[0].shape[1], actions[0].shape[1]
    # point-cloud modality (raw xyz, NOT normalized — the PointNet consumes metric coords)
    clouds = ([np.asarray(ep["observations"][pointcloud_key], np.float32) for ep in episodes]
              if pointcloud_key else None)

    # normalization stats over ALL data (raw units)
    all_s = np.concatenate(states, axis=0)
    all_a = np.concatenate(actions, axis=0)
    obs_min, obs_max = all_s.min(0), all_s.max(0)
    action_min, action_max = all_a.min(0), all_a.max(0)

    def norm_s(s):
        return 2 * (s - obs_min) / (obs_max - obs_min + 1e-6) - 1

    def norm_a(a):
        return 2 * (a - action_min) / (action_max - action_min + 1e-6) - 1

    # train/val split BY TRAJECTORY (DPPO stores concatenated trajs + traj_lengths)
    rng = np.random.default_rng(seed)
    n = len(episodes)
    n_train = max(1, int(round(n * (1 - val_split))))
    perm = rng.permutation(n)
    train_idx, val_idx = set(perm[:n_train].tolist()), set(perm[n_train:].tolist())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(split_name: str, idxs) -> int:
        idxs = [i for i in range(n) if i in idxs]
        if not idxs:
            return 0
        s = np.concatenate([norm_s(states[i]) for i in idxs], axis=0)
        a = np.concatenate([norm_a(actions[i]) for i in idxs], axis=0)
        r = np.concatenate([rewards[i] for i in idxs], axis=0)
        tl = np.asarray([len(actions[i]) for i in idxs], np.int64)
        arrays = dict(states=s, actions=a, rewards=r,
                      terminals=np.zeros(len(s), bool), traj_lengths=tl)
        if clouds is not None:                       # (T_total, N, 3) raw xyz
            arrays["point_cloud"] = np.concatenate([clouds[i] for i in idxs], axis=0)
        np.savez_compressed(out_dir / f"{split_name}.npz", **arrays)
        return len(idxs)

    n_tr = _write("train", train_idx)
    n_va = _write("val", val_idx)
    np.savez_compressed(out_dir / "normalization.npz", obs_min=obs_min, obs_max=obs_max,
                        action_min=action_min, action_max=action_max)

    meta = dict(obs_keys=list(obs_keys), obs_dim=int(obs_dim), action_dim=int(act_dim),
                n_episodes=n, n_train_traj=n_tr, n_val_traj=n_va,
                total_steps=int(all_s.shape[0]), out_dir=str(out_dir),
                point_cloud=(None if clouds is None else list(clouds[0].shape[1:])))
    return meta


def _find_demo_pkls(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    # canonical superset lives in <run>/data.pkl; fall back to any *.pkl
    pkls = sorted(root.rglob("data.pkl")) or sorted(root.rglob("*.pkl"))
    return [p for p in pkls if p.name != "normalization.npz"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("demos", type=Path, help="demo pkl or dir (recurses for data.pkl)")
    ap.add_argument("--out", type=Path, required=True, help="output dir for train/val/normalization npz")
    ap.add_argument("--obs-keys", nargs="+", default=None,
                    help="ordered state view (default: STATE_VIEW, or PROPRIO_VIEW with --point-cloud)")
    ap.add_argument("--point-cloud", dest="pc_key", nargs="?", const="point_cloud", default=None,
                    help="also store a raw point-cloud modality (student view); default key 'point_cloud'")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--experiment", default=None,
                    help="derive obs-key order from Experiment.load(name).view_obs(--view). "
                         "Guarantees BC pretraining and online env use the identical key order. "
                         "Takes precedence over --obs-keys.")
    ap.add_argument("--view", default="teacher",
                    help="which experiment view to use with --experiment (default: teacher)")
    args = ap.parse_args()

    if args.experiment:
        from gentle_manip.experiment import Experiment
        exp = Experiment.load(args.experiment)
        view_cfg = exp.view_obs(args.view)
        # obs_keys() returns only flat (non-image, non-cloud) keys for teacher/state views.
        obs_keys = [k for k in view_cfg.obs_keys()
                    if k not in ("point_cloud", "voxel_grid")
                    and not k.startswith("image_")
                    and not k.startswith("tactile_")]
        print(f"  obs-keys from experiment '{args.experiment}' view '{args.view}':")
        print(f"    {obs_keys}")
    else:
        obs_keys = args.obs_keys or (PROPRIO_VIEW if args.pc_key else STATE_VIEW)

    paths = _find_demo_pkls(args.demos)
    print(f"converting {len(paths)} demo file(s): {[str(p) for p in paths]}")
    meta = convert(paths, args.out, obs_keys=obs_keys, pointcloud_key=args.pc_key,
                   val_split=args.val_split)
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
