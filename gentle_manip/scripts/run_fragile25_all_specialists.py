"""Phase 4 top-level driver: run_fragile25_specialist.run_one() for every
train category in sequence (one Genesis process at a time, per AGENTS.md).
Idempotent/resumable -- re-invoking skips categories already marked "done"
in logs/fragile25_specialist/<category>.json.

Usage:
    python -m gentle_manip.scripts.run_fragile25_all_specialists
    python -m gentle_manip.scripts.run_fragile25_all_specialists --categories tofu mushroom
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gentle_manip.scripts.run_fragile25_specialist import run_one, RESULTS_DIR  # noqa: E402

TRAIN = ["mushroom", "raspberry", "grape", "kiwi", "egg_boiled", "strawberry",
        "banana", "tomato", "shrimp", "pasta_bundle", "cherry"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--port", type=int, default=5570,
                    help="sim server port for this driver's eval/rollout stages -- give each "
                         "concurrently-running driver instance a distinct port")
    args = ap.parse_args()
    categories = args.categories or TRAIN

    for cat in categories:
        result_path = RESULTS_DIR / f"{cat}.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior.get("status") == "done" and prior.get("eval_success_rate") is not None:
                print(f"[all_specialists] {cat}: already done "
                     f"(eval_success_rate={prior['eval_success_rate']}), skipping", flush=True)
                continue
        print(f"\n{'='*70}\n[all_specialists] Starting {cat}\n{'='*70}", flush=True)
        try:
            result = run_one(cat, port=args.port)
            print(f"[all_specialists] {cat}: status={result.get('status')} "
                 f"eval_success_rate={result.get('eval_success_rate')}", flush=True)
        except Exception as e:
            print(f"[all_specialists] {cat}: FAILED with exception: {e}", flush=True)

    print("\n[all_specialists] DONE", flush=True)


if __name__ == "__main__":
    main()
