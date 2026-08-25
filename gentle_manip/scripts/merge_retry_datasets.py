"""Merge each held-in category's original 50 raw-synthesis episodes (the first
50 episodes of the existing 150-episode dataset/demos/ dataset) with the new
~50 genuine retry/regrasp episodes (dataset/demos_retry/, natural first-attempt
failures only, collected 2026-08-23) into a single ~100-episode dataset per
category -- the training set for the retry-aware specialist retrain.

Writes into a NEW run dir under the SAME task path (dataset/demos/single_lift_
<cat>_soft/) per the CLAUDE.md task-naming hard rule -- never invents a new
task name; distinguishes the merged set from the original 150-episode run only
via its own run-dir name + config.yaml description. Non-destructive: neither
source dataset is modified.

"Original 50" = episodes[0:50] of the existing 150-episode dataset, since that
150-episode set was itself built by augmenting an original ~50-episode
collection up to 150 (augment_heldin_to_150.py, --resume-dir continuation) --
episodes 0-49 are therefore exactly the original run, in collection order.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.merge_retry_datasets
"""
from __future__ import annotations

import datetime
import pickle
import random
import string
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HELD_IN = ["banana", "cherry", "grape", "kiwi", "mushroom", "pasta_bundle",
          "raspberry", "shrimp", "tomato"]
ORIGINAL_DIR = REPO / "dataset" / "demos"
RETRY_DIR    = REPO / "dataset" / "demos_retry"
N_ORIGINAL   = 50


def _latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [d for d in root.iterdir() if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def _make_run_dir(out_dir: Path, task_name: str) -> Path:
    """Same dated-run-dir convention as collect_demos_synth_v3.py's _make_run_dir."""
    base = out_dir / task_name
    base.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now().strftime("%y-%m-%d")
    for _ in range(10000):
        sfx = "".join(random.choices(string.ascii_lowercase, k=3))
        cand = base / f"{date}-{sfx}"
        if not cand.exists():
            cand.mkdir()
            return cand
    raise RuntimeError(f"could not create run dir under {base}")


def merge_one(cat: str) -> dict:
    task_name = f"single_lift_{cat}_soft"
    orig_run  = _latest_run_dir(ORIGINAL_DIR / task_name)
    retry_run = _latest_run_dir(RETRY_DIR / task_name)
    if orig_run is None:
        raise FileNotFoundError(f"{cat}: no original dataset under {ORIGINAL_DIR / task_name}")
    if retry_run is None:
        raise FileNotFoundError(f"{cat}: no retry dataset under {RETRY_DIR / task_name}")

    with open(orig_run / "data.pkl", "rb") as f:
        orig = pickle.load(f)
    with open(retry_run / "data.pkl", "rb") as f:
        retry = pickle.load(f)

    if len(orig["episodes"]) < N_ORIGINAL:
        raise ValueError(f"{cat}: original dataset {orig_run} only has "
                         f"{len(orig['episodes'])} episodes, need >= {N_ORIGINAL}")

    orig_keys  = set(orig["meta"].get("obs_keys", []))
    retry_keys = set(retry["meta"].get("obs_keys", []))
    if orig_keys and retry_keys and orig_keys != retry_keys:
        raise ValueError(f"{cat}: obs_keys mismatch between original ({sorted(orig_keys)}) "
                         f"and retry ({sorted(retry_keys)}) datasets -- refusing to merge "
                         f"incompatible observation schemas")

    original_50 = orig["episodes"][:N_ORIGINAL]
    retry_all   = retry["episodes"]
    merged_episodes = original_50 + retry_all

    n_recovered_in_original = sum(1 for e in original_50 if e.get("recovered_from_slip"))
    n_recovered_in_retry    = sum(1 for e in retry_all if e.get("recovered_from_slip"))

    out_run = _make_run_dir(ORIGINAL_DIR, task_name)
    meta = dict(orig["meta"])
    meta["n_episodes"] = len(merged_episodes)
    meta["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(out_run / "data.pkl", "wb") as f:
        pickle.dump({"meta": meta, "episodes": merged_episodes}, f)

    cfg = {
        "task_name": task_name,
        "description": (f"MERGED: original 50 raw-synthesis episodes (episodes[0:{N_ORIGINAL}] "
                        f"of {orig_run.name}) + {len(retry_all)} genuine retry/regrasp episodes "
                        f"({retry_run.name}, natural first-attempt failures only, loosened-first-"
                        f"attempt collection method) = {len(merged_episodes)} total"),
        "source": "merge_retry_datasets",
        "original_run": str(orig_run.relative_to(REPO)),
        "original_run_episodes_used": N_ORIGINAL,
        "retry_run": str(retry_run.relative_to(REPO)),
        "retry_run_episodes_used": len(retry_all),
        "n_recovered_in_original_50": n_recovered_in_original,
        "n_recovered_in_retry": n_recovered_in_retry,
        "created": meta["created"],
    }
    with open(out_run / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return {"category": cat, "out_run": str(out_run.relative_to(REPO)),
           "n_total": len(merged_episodes), "n_original": len(original_50),
           "n_retry": len(retry_all), "n_recovered_total": n_recovered_in_original + n_recovered_in_retry}


def main() -> None:
    results = []
    for cat in HELD_IN:
        r = merge_one(cat)
        print(f"[merge_retry_datasets] {r}", flush=True)
        results.append(r)
    print(f"[merge_retry_datasets] DONE. {len(results)}/{len(HELD_IN)} categories merged.", flush=True)


if __name__ == "__main__":
    main()
