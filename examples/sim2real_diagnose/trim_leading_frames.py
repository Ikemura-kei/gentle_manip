"""Drop the first N frames (default 1) from every episode in a paired hybrid dataset
(build_hybrid_arm_real_mushroom_sim.py output) — both Condition R and Condition S streams,
plus actions, stay aligned (obs[t] still pairs with actions[t] after trimming).

Post-processes an already-built pkl in place; no sim rerun.

Usage:
    python examples/sim2real_diagnose/trim_leading_frames.py \\
        dataset/real_deploy/ahaxs800_printed_mushrooms/sim2real_data_analysis/hybrid_arm_real_mushroom_sim.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl", type=Path)
    ap.add_argument("--n", type=int, default=1, help="number of leading frames to drop")
    ap.add_argument("--out", type=Path, default=None, help="default: overwrite in place")
    args = ap.parse_args()

    data = pickle.load(open(args.pkl, "rb"))
    n = args.n
    out_episodes = []
    too_short = 0
    for ep in data["episodes"]:
        T = ep["actions"].shape[0]
        if T <= n:
            too_short += 1
            continue
        new_obs = {k: v[n:] for k, v in ep["observations"].items()}
        out_episodes.append({"observations": new_obs, "actions": ep["actions"][n:]})

    data["episodes"] = out_episodes
    data["meta"]["n_leading_frames_dropped"] = n
    data["meta"]["n_episodes"] = len(out_episodes)
    data["meta"]["n_episodes_too_short_dropped"] = too_short

    out_path = args.out or args.pkl
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Trimmed {n} leading frame(s) from {len(out_episodes)} episodes "
          f"({too_short} episode(s) too short, dropped entirely) -> {out_path}")


if __name__ == "__main__":
    main()
