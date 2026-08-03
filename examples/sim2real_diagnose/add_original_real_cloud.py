"""Add the ORIGINAL (unedited) real point cloud as a third condition to a paired hybrid
dataset (build_hybrid_arm_real_mushroom_sim.py output, optionally already frame-trimmed by
trim_leading_frames.py) — no sim rerun needed. Uses the exact `t_cutoff` and source episode
index already recorded in the pkl's `meta["per_episode_stats"]` (written by the build
script) to re-slice the matching frames straight out of the original source deploy dataset.

After this, each episode has THREE point-cloud conditions, all frame-aligned to the same
actions:
  - point_cloud       Condition R: real arm + sim mushroom (hybrid)
  - point_cloud_sim   Condition S: pure sim (arm + mushroom both sim)
  - point_cloud_orig  Condition O: original real, unedited (arm + REAL mushroom)
ee_pos/ee_quat/gripper_width (no suffix) are real proprioception, shared by conditions R
and O (both are real underneath — only the point cloud differs between them).

Usage:
    python examples/sim2real_diagnose/add_original_real_cloud.py \\
        dataset/real_deploy/ahaxs800_printed_mushrooms/sim2real_data_analysis/hybrid_arm_real_mushroom_sim.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import replay_deploy_in_sim as rds   # noqa: E402  (reuse load_shards)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl", type=Path, help="hybrid_arm_real_mushroom_sim.pkl")
    ap.add_argument("--source-deploy-dir", type=Path, default=None,
                    help="default: meta['source_deploy_dir'] recorded in the pkl")
    ap.add_argument("--out", type=Path, default=None, help="default: overwrite in place")
    args = ap.parse_args()

    data = pickle.load(open(args.pkl, "rb"))
    meta = data["meta"]
    stats = meta.get("per_episode_stats")
    if stats is None or len(stats) != len(data["episodes"]):
        raise RuntimeError(
            "meta['per_episode_stats'] is missing or doesn't match episode count "
            f"({len(stats) if stats else 0} stats vs {len(data['episodes'])} episodes) -- "
            "was this pkl built by build_hybrid_arm_real_mushroom_sim.py?")

    n_dropped_leading = int(meta.get("n_leading_frames_dropped", 0))
    source_dir = args.source_deploy_dir or Path(meta["source_deploy_dir"])
    source_episodes = rds.load_shards(source_dir)
    print(f"Loaded {len(source_episodes)} source episodes from {source_dir}")
    print(f"Re-slicing with n_leading_frames_dropped={n_dropped_leading} "
          f"(from this pkl's meta, so the added condition stays frame-aligned)")

    for i, (ep, st) in enumerate(zip(data["episodes"], stats)):
        src_ep = source_episodes[st["episode"]]
        src_pc = np.asarray(src_ep["observations"]["point_cloud"], np.float32)
        t_cut = st["t_cutoff"]
        orig_pc = src_pc[n_dropped_leading:t_cut + 1]     # same [0,t_cutoff] slice, then trim
        expected_T = ep["actions"].shape[0]
        if orig_pc.shape[0] != expected_T:
            raise RuntimeError(
                f"episode {i} (source ep {st['episode']}): re-sliced original cloud has "
                f"{orig_pc.shape[0]} frames, expected {expected_T} -- t_cutoff/frame-drop "
                f"bookkeeping is out of sync with this pkl.")
        ep["observations"]["point_cloud_orig"] = orig_pc.copy()

    meta["obs_keys"] = sorted(set(meta.get("obs_keys", [])) | {"point_cloud_orig"})
    meta["condition_o"] = "original real, unedited (arm + REAL mushroom); proprioception == condition R's"

    out_path = args.out or args.pkl
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Added point_cloud_orig to {len(data['episodes'])} episodes -> {out_path}")


if __name__ == "__main__":
    main()
