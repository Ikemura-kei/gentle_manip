"""Stage the REACTIVE-policy v4 training set: gen8 v2 regrasp data + ALL reactive-recovery
demos (v3's 3-category set + v4's 8-category set, collected with the FIRM grasp phase and
the eval-matched 0.30-0.90 m/s drag), mixed so reactive_recover episodes are
~`--reactive-frac` of the total. Writes ONE data.pkl the DPPO converter consumes.

Usage:  python _reactive_stage_v4.py OUT_DIR [--reactive-frac 0.30] [--seed 0]
"""
from __future__ import annotations
import argparse, glob, pickle
from pathlib import Path
import numpy as np

REPO = Path("/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip")
V2 = REPO / "dataset/demos/_gen8_regrasp/data.pkl"
RX_GLOBS = [
    str(REPO / "dataset/demos/single_lift_xcat_reactive_v4/*/"),
    str(REPO / "dataset/demos/single_lift_xcat_reactive/*/"),
]

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
    for gl in RX_GLOBS:
        for d in sorted(glob.glob(gl)):
            dd = Path(d); p = dd / "data.pkl"
            eps = _load_eps(p) if p.exists() else []
            if not eps:
                for s in sorted(dd.glob("shard_*.pkl")):
                    eps += _load_eps(s)
            rx += eps
    rx_recover = [e for e in rx if e.get("start_mode") == "reactive_recover"]
    print(f"reactive: {len(rx)} episodes ({len(rx_recover)} tagged reactive_recover)")
    rx = rx_recover or rx
    if not rx:
        raise SystemExit("no reactive demos found -- collect first")

    n_v2 = int(round(len(rx) * (1.0 / max(args.reactive_frac, 1e-3) - 1.0)))
    n_v2 = min(n_v2, len(v2))
    keep = v2 if n_v2 >= len(v2) else [v2[i] for i in rng.choice(len(v2), n_v2, replace=False)]
    print(f"keeping {len(keep)} v2 + {len(rx)} reactive -> reactive frac "
          f"{len(rx) / (len(keep) + len(rx)):.2f}")

    eps = list(keep) + list(rx)
    rng.shuffle(eps)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = {"task": "single_lift_xcat_reactive_v4", "n_episodes": len(eps),
            "reactive_recover": len(rx), "v2_regrasp": len(keep),
            "obs_keys": ["ee_pos", "ee_quat", "gripper_width", "point_cloud", "priv_object_pos"]}
    tmp = out / "data.tmp"
    pickle.dump({"meta": meta, "episodes": eps}, open(tmp, "wb"))
    tmp.replace(out / "data.pkl")
    print(f"-> {out}/data.pkl  ({len(eps)} episodes)")

if __name__ == "__main__":
    main()
