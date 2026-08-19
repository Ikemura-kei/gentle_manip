"""Top up dumpling's zero-shot specialist to the full 50-episode target and
retrain. Dumpling's specialist trained on only 30 episodes -- a recurring
_merge_shards data-loss bug (now fixed with a monotonicity guard, commit
2a1cdec) truncated its collection twice this campaign (see
docs/cross_category_specialist_log.md's 2026-08-19 entries). This collects
the remaining ~20 episodes (resuming into the existing 30-episode run dir)
and forces a fresh convert->train->eval cycle by clearing the stale
30-episode-based caches, so the final specialist result reflects the full
50-episode dataset like every other category in the campaign.

Usage:
    uv run --project envs/dppo python -m gentle_manip.scripts.topup_dumpling
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gentle_manip.scripts import collect_rigid_cross_category as collector  # noqa: E402
from gentle_manip.scripts.run_zeroshot_specialists import (  # noqa: E402
    train_and_eval_one, CAMPAIGN_SPEC_OUT,
)

CATEGORY = "dumpling"
N_EPISODES_TARGET = 50


def collect_topup() -> None:
    log_dir = REPO / "logs" / "collect_rigid_cross_category"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = REPO / "dataset" / "demos"
    r = collector.collect_one(
        CATEGORY, N_EPISODES_TARGET, n_envs=5, maxfevals=800,
        out_dir=out_dir, timeout_s=2700, record_video=True, log_dir=log_dir,
        shard_size=5, experiment_template="single_lift_{category}_soft_easy",
        scene_dr_every=1)
    print(f"[topup] {CATEGORY}: collection {r}", flush=True)


def force_fresh_pipeline() -> None:
    """Clear every stale cache from the 30-episode run so convert/train/eval
    re-derive everything from the topped-up data.pkl -- mirrors the exact
    stale-cache fix already applied to mushroom/raspberry/grape earlier this
    campaign (project_generalist_12plus4_campaign.md, 2026-08-15 ~23:23)."""
    spec_json = REPO / "logs" / "fragile25_specialist" / f"{CATEGORY}.json"
    if spec_json.exists():
        spec_json.unlink()
        print(f"[topup] cleared stale {spec_json}", flush=True)

    dppo_data_dir = (REPO.parent / "robosuite_mog_private" / "dppo" / "data"
                     / f"single_lift_{CATEGORY}_soft_easy_pcd")
    if dppo_data_dir.exists():
        shutil.rmtree(dppo_data_dir)
        print(f"[topup] cleared stale {dppo_data_dir}", flush=True)

    out_path = CAMPAIGN_SPEC_OUT / f"{CATEGORY}.json"
    if out_path.exists():
        out_path.unlink()
        print(f"[topup] cleared stale {out_path}", flush=True)


def main() -> None:
    collect_topup()
    force_fresh_pipeline()
    r = train_and_eval_one(CATEGORY)
    print(f"[topup] DONE: {CATEGORY} success_rate={r.get('success_rate')}", flush=True)


if __name__ == "__main__":
    main()
