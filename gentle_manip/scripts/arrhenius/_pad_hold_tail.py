"""Append a settled 'hold aloft' tail to every SUCCESSFUL demo episode.

Failure mode the campaign hit: a hold-untrained BC policy, after a good lift, has no
stable fixed point -- the scripted demos end while still commanding dz~+0.476 (a
saturating up-command that just clips against EE_BOUNDS_MAX), so there is never a
"object at rest at height, EE still, action ~= 0" state in the data. At eval the
policy saturates the up-command, goes OOD, and drifts into the dominant approach /
recovery modes -> carries the object back down, releases, presses the floor.

This post-processing pass adds `--tail` frames to the end of each episode whose last
recorded frame is a committed lift (obj lifted clear + gripper stalled on the object):
  action  : zeros, except a small +dz (gravity-sag correction) and the last gripper cmd
  obs     : the last frame, repeated (ee/obj static; point cloud frozen -- a mild lie,
            standard for terminal-state BC padding, and cheaper than re-simming)
  rewards : the last reward, repeated

Operates on a staged data.pkl (post `_gen8_stage.py`, pre `convert_demos`). Idempotent
via a meta flag. Does NOT re-collect anything.

Usage:
  python _pad_hold_tail.py IN.pkl OUT.pkl [--tail 32] [--hold-dz 0.04] [--lift-frac 0.6]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def _is_committed_lift(ep: dict, lift_frac: float) -> bool:
    """Last frame looks like a real lift+hold: gripper stalled well under open width and
    the recorded up-command dominated the tail."""
    w = np.asarray(ep["observations"]["gripper_width"]).ravel()
    a = np.asarray(ep["actions"])
    if len(a) < 5:
        return False
    grip_stalled = w[-1] < 0.055                       # 0.08 == fully open; <0.055 == on the object
    tail = a[max(1, int(len(a) * lift_frac)):]
    lifted = float(np.mean(tail[:, 2] > 0.25)) > 0.5   # dz strongly positive over the tail
    return bool(grip_stalled and lifted)


def pad_episode(ep: dict, tail: int, hold_dz: float) -> dict:
    obs = ep["observations"]
    a = np.asarray(ep["actions"], np.float32)
    keys = list(obs.keys())
    last_obs = {k: np.asarray(obs[k])[-1] for k in keys}

    hold_a = np.zeros((tail, a.shape[1]), np.float32)
    hold_a[:, 2] = hold_dz                              # small, non-saturating up-hold vs gravity sag
    hold_a[:, 6] = a[-1, 6]                             # keep the last gripper command (stay closed)

    out = dict(ep)
    out["observations"] = {
        k: np.concatenate([np.asarray(obs[k]),
                           np.repeat(last_obs[k][None], tail, axis=0)], axis=0)
        for k in keys
    }
    out["actions"] = np.concatenate([a, hold_a], axis=0)
    if "rewards" in ep:
        r = np.asarray(ep["rewards"], np.float32)
        out["rewards"] = np.concatenate([r, np.repeat(r[-1:], tail)])
    out["hold_tail_added"] = int(tail)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--tail", type=int, default=20, help="frames to append (~0.7s at 30 Hz)")
    ap.add_argument("--hold-dz", type=float, default=0.04,
                    help="normalized +dz held during the tail (non-saturating gravity-sag correction)")
    ap.add_argument("--lift-frac", type=float, default=0.6)
    args = ap.parse_args()

    d = pickle.load(open(args.inp, "rb"))
    if d.get("meta", {}).get("hold_tail_padded"):
        print("already padded -> copy through")
        Path(args.out).write_bytes(Path(args.inp).read_bytes())
        return

    eps = d["episodes"]
    n_pad = n_skip = 0
    new_eps = []
    for ep in eps:
        if _is_committed_lift(ep, args.lift_frac):
            new_eps.append(pad_episode(ep, args.tail, args.hold_dz))
            n_pad += 1
        else:
            new_eps.append(ep)
            n_skip += 1

    d["episodes"] = new_eps
    meta = dict(d.get("meta", {}))
    meta["hold_tail_padded"] = {"tail": args.tail, "hold_dz": args.hold_dz,
                                "n_padded": n_pad, "n_unpadded": n_skip}
    d["meta"] = meta
    tmp = Path(args.out).with_suffix(".tmp")
    pickle.dump(d, open(tmp, "wb"))
    tmp.replace(args.out)
    total_frames = sum(len(np.asarray(e["actions"])) for e in new_eps)
    print(f"padded {n_pad} committed-lift eps (+{args.tail} frames each), left {n_skip} as-is")
    print(f"-> {args.out}  ({len(new_eps)} eps, {total_frames} frames)")


if __name__ == "__main__":
    main()
