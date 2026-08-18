"""Train + eval a solo specialist for each of the 4 zero-shot categories
(blackberry, scallop, dumpling, gelatin) -- previously these had NO specialist
at all (that's what "zero-shot" meant), but the full eval campaign
(run_full_eval_campaign.py) now wants a 13-category generalist-vs-specialist
comparison, not just the 9 held-in categories. This closes that gap: collect
50 FEM (v3) synthesized demos per category (same recipe as the 9 held-in
categories -- see collect_rigid_cross_category.py), train a solo BC-pretrain
policy, then run the same canonical 100-episode eval used everywhere else in
this campaign. Deliberately skips RLDG rollout collection (no plan to merge
these into the generalist -- they stay zero-shot by definition; this is only
about having a specialist BASELINE to compare against for the report).

Writes logs/full_eval_campaign/specialist/<cat>.json in the SAME schema
run_full_eval_campaign.py's specialist phase produces, so the report's
existing campaign-loading code picks these up with no changes needed.

Idempotent per stage -- safe to re-run after an interruption. Every heavy step
(collection, train, eval) shells out to its own uv --project env explicitly
(same pattern as collect_rigid_cross_category.py / run_fragile25_specialist.py),
so this orchestrator itself can run under any env with the gentle_manip
package importable (stdlib-only imports at module level).

Usage:
    uv run --project envs/dppo python -m gentle_manip.scripts.run_zeroshot_specialists
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gentle_manip.scripts.run_fragile25_specialist import (  # noqa: E402
    RESULTS_DIR, convert, write_configs, train, best_checkpoint, eval_specialist,
)
from gentle_manip.scripts import collect_rigid_cross_category as collector  # noqa: E402

CATEGORIES = ["blackberry", "scallop", "dumpling", "gelatin"]
N_EPISODES_COLLECT = 50
PORT = 5580
CAMPAIGN_SPEC_OUT = REPO / "logs" / "full_eval_campaign" / "specialist"
CAMPAIGN_SPEC_OUT.mkdir(parents=True, exist_ok=True)


def collect_one(category: str) -> None:
    """Same recipe used for the 9 held-in categories' collection this campaign."""
    log_dir = REPO / "logs" / "collect_rigid_cross_category"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = REPO / "dataset" / "demos"
    r = collector.collect_one(
        category, N_EPISODES_COLLECT, n_envs=5, maxfevals=800,
        out_dir=out_dir, timeout_s=2700, record_video=True, log_dir=log_dir,
        shard_size=5, experiment_template="single_lift_{category}_soft_easy",
        scene_dr_every=1)
    print(f"[zeroshot-specialist] {category}: collection {r}", flush=True)


def train_and_eval_one(category: str) -> dict:
    out_path = CAMPAIGN_SPEC_OUT / f"{category}.json"
    if out_path.exists():
        r = json.loads(out_path.read_text())
        print(f"[zeroshot-specialist] {category}: already done, "
             f"success_rate={r.get('success_rate')}", flush=True)
        return r

    spec_json = REPO / "logs" / "fragile25_specialist" / f"{category}.json"
    result = json.loads(spec_json.read_text()) if spec_json.exists() else {"category": category}

    if "demo_dir" not in result:
        from gentle_manip.scripts.run_fragile25_specialist import find_latest_demo_dir
        demo_dir = find_latest_demo_dir(category)
        if demo_dir is None:
            raise RuntimeError(f"{category}: no demos found -- run collection first "
                               f"(envs/sim, --skip-collect off)")
        result["demo_dir"] = str(demo_dir)
        spec_json.write_text(json.dumps(result, indent=2))

    if "dppo_data_dir" not in result:
        t0 = time.time()
        dppo_dir = convert(category, Path(result["demo_dir"]))
        result["dppo_data_dir"] = str(dppo_dir)
        result["convert_elapsed_s"] = time.time() - t0
        spec_json.write_text(json.dumps(result, indent=2))

    cfg_dir = write_configs(category, port=PORT)

    if "run_dir" not in result or not result.get("train_ok"):
        t0 = time.time()
        train_result = train(category, cfg_dir)
        result["run_dir"] = train_result["run_dir"]
        result["train_ok"] = train_result["ok"]
        result["train_elapsed_s"] = time.time() - t0
        spec_json.write_text(json.dumps(result, indent=2))

    if result.get("run_dir") and "checkpoint" not in result:
        ckpt = best_checkpoint(Path(result["run_dir"]), category)
        result["checkpoint"] = str(ckpt) if ckpt else None
        spec_json.write_text(json.dumps(result, indent=2))

    if not result.get("checkpoint"):
        raise RuntimeError(f"{category}: no checkpoint after training -- see {spec_json}")

    t0 = time.time()
    r = eval_specialist(category, cfg_dir, Path(result["checkpoint"]), port=PORT)
    r["category"] = category
    r["checkpoint"] = result["checkpoint"]
    r["elapsed_s"] = time.time() - t0
    out_path.write_text(json.dumps(r, indent=2))
    sm = r.get("summary") or {}
    print(f"[zeroshot-specialist] {category}: success_rate={r['success_rate']} "
         f"combined={sm.get('combined_sr_gentleness')} ({r['elapsed_s']:.0f}s)", flush=True)
    return r


def main() -> None:
    for cat in CATEGORIES:
        collect_one(cat)
    for cat in CATEGORIES:
        train_and_eval_one(cat)
    print("[zeroshot-specialist] DONE", flush=True)


if __name__ == "__main__":
    main()
