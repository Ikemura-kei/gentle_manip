"""Top up all 9 held-in categories' raw v3 (FEM gentleness synthesis) demo sets
from ~50 to ~150 episodes each. This is the data-scale side of the direct-vs-
RLDG comparison the user asked for (2026-08-19): does training ONE generalist
directly on more raw synthesized demos (no specialist, no RLDG distillation)
match or beat the specialist->RLDG->merge pipeline used for the existing
generalist? ~150 episodes/category was chosen to roughly match the RLDG
generalist's effective per-category budget (raw demos + up to 150 distilled
rollouts).

Same recipe as the rest of this campaign's collection
(collect_rigid_cross_category.collect_one(), v3 FEM synthesis, --grasp-gpu,
scene_dr_every=1) -- purely a --n-episodes bump, --resume-dir already tops up
from each category's existing ~50-episode run dir rather than recollecting
from scratch. Sequential (one category at a time) -- same single-GPU-job
discipline used throughout this campaign.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.augment_heldin_to_150
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gentle_manip.scripts import collect_rigid_cross_category as collector  # noqa: E402

HELD_IN = ["banana", "cherry", "grape", "kiwi", "mushroom", "pasta_bundle",
          "raspberry", "shrimp", "tomato"]
N_EPISODES_TARGET = 150


def main() -> None:
    log_dir = REPO / "logs" / "collect_rigid_cross_category"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = REPO / "dataset" / "demos"
    for cat in HELD_IN:
        r = collector.collect_one(
            cat, N_EPISODES_TARGET, n_envs=5, maxfevals=800,
            out_dir=out_dir, timeout_s=2700, record_video=True, log_dir=log_dir,
            shard_size=5, experiment_template="single_lift_{category}_soft_easy",
            scene_dr_every=1)
        print(f"[augment150] {cat}: {r}", flush=True)
    print("[augment150] DONE", flush=True)


if __name__ == "__main__":
    main()
