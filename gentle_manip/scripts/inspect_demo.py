"""Print the obs/action ranges in a recorded demo .pkl (DemoRecorder output).

Mainly a sim/real parity check before deploying a real-trained policy on the sim:
it surfaces the gripper-open width, EE workspace, quaternion sign convention, point
cloud crop, and action ranges the policy was trained on — compare these against
what the sim backend produces. Loads with plain pickle + numpy, so any env works.

    uv run --project envs/sim python -m gentle_manip.scripts.inspect_demo \
        --demo dataset/demos/red_cube/26-06-18-jcd.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def _rng(a: np.ndarray) -> str:
    a = np.asarray(a)
    return " ".join(f"[{a[..., i].min():+.3f},{a[..., i].max():+.3f}]" for i in range(a.shape[-1]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a recorded demo .pkl")
    ap.add_argument("--demo", type=Path, required=True)
    args = ap.parse_args()

    d = pickle.load(open(args.demo, "rb"))
    eps = d["episodes"]
    print(f"demo: {args.demo}")
    print(f"meta: {d['meta']}")
    print(f"episodes: {len(eps)}  total frames: {sum(len(e['actions']) for e in eps)}")

    # Stack per-key obs + actions across all episodes.
    keys = list(eps[0]["observations"].keys())
    obs = {k: np.concatenate([e["observations"][k] for e in eps], axis=0) for k in keys}
    act = np.concatenate([e["actions"] for e in eps], axis=0)

    if "gripper_width" in obs:
        gw = obs["gripper_width"].reshape(-1)
        first = np.array([e["observations"]["gripper_width"][0, 0] for e in eps])   # per-episode reset value
        print("\n=== gripper_width (compare sim reset = 0.080) ===")
        print(f"  global:        [{gw.min():.4f}, {gw.max():.4f}]")
        print(f"  episode start: [{first.min():.4f}, {first.max():.4f}]  mean={first.mean():.4f}  "
              f"(this is the 'open' value the policy expects at reset)")

    if "ee_pos" in obs:
        print("\n=== ee_pos (workspace, m) ===")
        print(f"  x/y/z: {_rng(obs['ee_pos'])}")
    if "ee_quat" in obs:
        w = obs["ee_quat"][:, 0]
        print("\n=== ee_quat (wxyz) ===")
        print(f"  w range [{w.min():+.3f},{w.max():+.3f}]  all w>=0? {bool((w >= 0).all())}  "
              f"(sim enforces w>=0)")
    if "point_cloud" in obs:
        pc = obs["point_cloud"].reshape(-1, 3)
        print("\n=== point_cloud (crop region, m) ===")
        print(f"  x/y/z: {_rng(pc)}   pts/frame: {obs['point_cloud'].shape[1]}")

    print("\n=== action (7-dim, normalized [-1,1] teleop space) ===")
    print(f"  per-dim: {_rng(act)}")


if __name__ == "__main__":
    main()
