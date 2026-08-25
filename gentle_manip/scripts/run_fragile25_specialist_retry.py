"""Retrain the 9 held-in specialists on the retry-augmented 100-episode datasets
(50 original raw-synthesis + 50 genuine retry/regrasp episodes, see
merge_retry_datasets.py), then re-run the canonical eval + RLDG rollout gate --
the same convert->train->eval->rollout pipeline as run_fragile25_specialist.py,
reused via its `run_one()`, but isolated from the ORIGINAL specialist campaign
so nothing here silently reuses stale cached results or converted data:

- RESULTS_DIR is monkeypatched to a SEPARATE logs/fragile25_specialist_retry/
  tree (the original logs/fragile25_specialist/<cat>.json files are untouched --
  reusing them would make run_one() see every step already "done" and skip
  straight past retraining).
- DPPO_DATA_DIR is overridden to a separate dppo_data_retry/ root -- the
  original run_fragile25_specialist.convert() skips conversion if
  <DPPO_DATA_DIR>/<env>/train.npz already exists, which (with the DEFAULT
  DPPO_DATA_DIR) would silently reuse the OLD 150-episode conversion instead
  of converting the new merged 100-episode dataset.
- Each category's result JSON is pre-seeded with `demo_dir` pointing at its
  merge_retry_datasets.py output run dir -- run_one()'s own find_latest_demo_dir()
  picks whichever run dir has the MOST episodes, which would wrongly select the
  original 150-episode dataset (150 > the merged set's 100) if left to discover
  on its own.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.run_fragile25_specialist_retry
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Must happen BEFORE importing run_fragile25_specialist / gentle_manip.dppo.train,
# since the DPPO hydra configs read DPPO_DATA_DIR via ${oc.env:DPPO_DATA_DIR} at
# subprocess launch time -- os.environ is inherited by every subprocess this
# driver shells out to (train_with_resume, eval_specialist, collect_rollouts all
# copy os.environ for their subprocess env).
os.environ["DPPO_DATA_DIR"] = str(REPO / "dppo_data_retry")

from gentle_manip.scripts import run_fragile25_specialist as spec  # noqa: E402

spec.RESULTS_DIR = REPO / "logs" / "fragile25_specialist_retry"
spec.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HELD_IN = ["banana", "cherry", "grape", "kiwi", "mushroom", "pasta_bundle",
          "raspberry", "shrimp", "tomato"]


def _latest_merged_run_dir(cat: str) -> Path:
    """The merge_retry_datasets.py output run dir for `cat` -- always the
    latest-mtime run under dataset/demos/single_lift_<cat>_soft/ whose
    config.yaml says source=merge_retry_datasets, so this stays correct even
    if merge_retry_datasets.py is re-run later."""
    import yaml
    task_dir = REPO / "dataset" / "demos" / f"single_lift_{cat}_soft"
    candidates = []
    for d in task_dir.iterdir():
        cfg_path = d / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg.get("source") == "merge_retry_datasets":
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"{cat}: no merge_retry_datasets output found under {task_dir} "
                                f"-- run merge_retry_datasets.py first")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def main() -> None:
    port = int(os.environ.get("RETRY_SPECIALIST_PORT", "5571"))  # distinct from the
    # original campaign's default 5570, in case anything is still lingering
    results = []
    for cat in HELD_IN:
        result_path = spec.RESULTS_DIR / f"{cat}.json"
        if not result_path.exists():
            demo_dir = _latest_merged_run_dir(cat)
            result_path.write_text(json.dumps({"category": cat, "demo_dir": str(demo_dir)}, indent=2))
            print(f"[run_specialist_retry] {cat}: seeded demo_dir={demo_dir}", flush=True)
        print(f"[run_specialist_retry] {cat}: starting run_one()...", flush=True)
        r = spec.run_one(cat, port=port)
        print(f"[run_specialist_retry] {cat}: {r}", flush=True)
        results.append(r)
    print(f"[run_specialist_retry] DONE. {len(results)}/{len(HELD_IN)} categories processed.", flush=True)


if __name__ == "__main__":
    main()
