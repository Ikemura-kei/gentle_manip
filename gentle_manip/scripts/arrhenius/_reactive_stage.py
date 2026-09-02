"""Stage the REACTIVE-policy (v3) training set: the gen8 v2 regrasp data + the new
reactive-recovery demos, mixed so reactive_recover episodes are ~`--reactive-frac` of
the total. Writes ONE data.pkl the DPPO converter consumes.

Sources:
  dataset/demos/_gen8_regrasp/data.pkl            (v2 regrasp: rebalanced + hold-tail padded)
  dataset/demos/single_lift_xcat_reactive/*/      (Phase C: object-dragged -> re-target demos)

The reactive demos are all kept; the v2 regrasp pool is subsampled (seeded) to hit the
target reactive fraction. A hold-tail is NOT re-added (the v2 data already has it; the
reactive demos end on a lift+brief hold from the FSM).

Usage:
  python _reactive_stage.py OUT_DIR [--reactive-frac 0.30] [--seed 0]
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np

REPO = Path("/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip")
V2 = REPO / "dataset/demos/_gen8_regrasp/data.pkl"
RX_GLOB = str(REPO / "dataset/demos/single_lift_xcat_reactive/*/")


def _load_eps(path):
    try:
        return pickle.load(open(path, "rb")).get("episodes", [])
    except Exception as e:
        print(f"  skip {path}: {e}")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--reactive-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    v2 = _load_eps(V2)
    print(f"v2 regrasp: {len(v2)} episodes")

    rx = []
    for d in sorted(glob.glob(RX_GLOB)):
        dd = Path(d)
        p = dd / "data.pkl"
        eps = _load_eps(p) if p.exists() else []
        if not eps:                                   # unmerged shards
            for s in sorted(dd.glob("shard_*.pkl")):
                eps += _load_eps(s)
        rx += eps
    rx_recover = [e for e in rx if e.get("start_mode") == "reactive_recover"]
    print(f"reactive: {len(rx)} episodes ({len(rx_recover)} tagged reactive_recover)")
    rx = rx_recover or rx                              # prefer the tagged ones

    if not rx:
        raise SystemExit("no reactive demos found -- collect first")

    # how many v2 episodes to keep so rx is `reactive_frac` of the total:
    #   frac = len(rx) / (len(rx) + n_v2)  ->  n_v2 = len(rx) * (1/frac - 1)
    n_v2 = int(round(len(rx) * (1.0 / max(args.reactive_frac, 1e-3) - 1.0)))
    n_v2 = min(n_v2, len(v2))
    keep = v2 if n_v2 >= len(v2) else [v2[i] for i in rng.choice(len(v2), n_v2, replace=False)]
    print(f"keeping {len(keep)} v2 + {len(rx)} reactive -> reactive frac "
          f"{len(rx) / (len(keep) + len(rx)):.2f}")

    eps = list(keep) + list(rx)
    rng.shuffle(eps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"task": "single_lift_xcat_reactive", "n_episodes": len(eps),
            "reactive_recover": len(rx), "v2_regrasp": len(keep),
            "obs_keys": ["ee_pos", "ee_quat", "gripper_width", "point_cloud", "priv_object_pos"]}
    tmp = out / "data.tmp"
    pickle.dump({"meta": meta, "episodes": eps}, open(tmp, "wb"))
    tmp.replace(out / "data.pkl")
    print(f"-> {out}/data.pkl  ({len(eps)} episodes)")


if __name__ == "__main__":
    main()
