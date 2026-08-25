"""Parallel-BC-pretrain variant of run_fragile25_specialist_retry.py: convert+
train run CONCURRENTLY across up to MAX_PARALLEL_TRAIN categories (bounded by
GPU memory -- BC-pretrain itself is lightweight, no Genesis), while eval and
RLDG-rollout collection (each spins up a full Genesis sim server -- the
memory-heavy, OOM-prone stage seen throughout this campaign's data collection)
stay STRICTLY SEQUENTIAL, processed one at a time in the main thread as each
category's training finishes.

Reuses run_fragile25_specialist.py's proven convert/write_configs/train/
best_checkpoint/eval_specialist/collect_rollouts functions directly (not
run_one(), which bundles eval+rollout into the same call -- this script needs
to split train from eval+rollout to parallelize only the former). Same
RESULTS_DIR / DPPO_DATA_DIR isolation as run_fragile25_specialist_retry.py.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.run_fragile25_specialist_retry_parallel [--max-parallel-train N] [--categories cat1 cat2 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

os.environ["DPPO_DATA_DIR"] = str(REPO / "dppo_data_retry")

from gentle_manip.scripts import run_fragile25_specialist as spec  # noqa: E402

spec.RESULTS_DIR = REPO / "logs" / "fragile25_specialist_retry"
spec.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_HELD_IN = ["banana", "cherry", "grape", "kiwi", "mushroom", "pasta_bundle",
              "raspberry", "shrimp", "tomato"]

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _latest_merged_run_dir(cat: str) -> Path:
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
        raise FileNotFoundError(f"{cat}: no merge_retry_datasets output found under {task_dir}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _load_result(cat: str) -> dict:
    p = spec.RESULTS_DIR / f"{cat}.json"
    return json.loads(p.read_text()) if p.exists() else {"category": cat}


def _save_result(cat: str, result: dict) -> None:
    (spec.RESULTS_DIR / f"{cat}.json").write_text(json.dumps(result, indent=2))


def convert_and_train(cat: str, port: int) -> tuple[str, dict]:
    """Runs in a worker thread: convert (cheap, CPU) + train (the GPU-heavy,
    parallelizable part). Mirrors run_one()'s per-step skip-if-already-done
    caching so this stays resumable/idempotent exactly like the sequential
    driver."""
    result = _load_result(cat)

    if "demo_dir" not in result or not Path(result.get("demo_dir", "")).exists():
        demo_dir = _latest_merged_run_dir(cat)
        result["demo_dir"] = str(demo_dir)
        _save_result(cat, result)
        _log(f"[parallel] {cat}: seeded demo_dir={demo_dir}")

    if "dppo_data_dir" not in result:
        _log(f"[parallel] {cat}: converting...")
        dppo_dir = spec.convert(cat, Path(result["demo_dir"]))
        result["dppo_data_dir"] = str(dppo_dir)
        _save_result(cat, result)

    cfg_dir = spec.write_configs(cat, port=port)

    if "run_dir" not in result or not result.get("train_ok"):
        _log(f"[parallel] {cat}: training started...")
        train_result = spec.train(cat, cfg_dir)
        result["run_dir"] = train_result["run_dir"]
        result["train_ok"] = train_result["ok"]
        _save_result(cat, result)
        _log(f"[parallel] {cat}: training done ok={train_result['ok']} run_dir={train_result['run_dir']}")
    else:
        _log(f"[parallel] {cat}: training already done, skipping")

    return cat, result


def eval_and_rollout(cat: str, result: dict, port: int) -> dict:
    """Runs SEQUENTIALLY in the main thread -- one Genesis sim server at a time."""
    if result.get("run_dir") and "checkpoint" not in result:
        ckpt = spec.best_checkpoint(Path(result["run_dir"]), cat)
        result["checkpoint"] = str(ckpt) if ckpt else None
        _save_result(cat, result)

    cfg_dir = spec.write_configs(cat, port=port)

    if result.get("checkpoint") and "eval_success_rate" not in result:
        _log(f"[parallel] {cat}: evaluating (sequential)...")
        eval_result = spec.eval_specialist(cat, cfg_dir, Path(result["checkpoint"]), port=port)
        result["eval_success_rate"] = eval_result["success_rate"]
        result["eval_ok"] = eval_result["ok"]
        result["eval_log"] = eval_result["eval_log"]
        _save_result(cat, result)
        _log(f"[parallel] {cat}: eval_success_rate={result['eval_success_rate']}")

    sr = result.get("eval_success_rate")
    if sr is not None and sr < spec.QUALITY_GATE:
        result["rollout_status"] = f"skipped: eval_success_rate {sr} < QUALITY_GATE {spec.QUALITY_GATE}"
        _save_result(cat, result)
        _log(f"[parallel] {cat}: rollout SKIPPED ({result['rollout_status']})")
    elif result.get("checkpoint") and sr is not None and "rollout_data_path" not in result:
        _log(f"[parallel] {cat}: collecting RLDG rollouts (sequential)...")
        rollout_result = spec.collect_rollouts(cat, Path(result["checkpoint"]), port=port)
        result["rollout_ok"] = rollout_result["ok"]
        result["rollout_n_episodes"] = rollout_result.get("n_episodes")
        result["rollout_data_path"] = rollout_result.get("data_path")
        _save_result(cat, result)
        _log(f"[parallel] {cat}: rollout done, n_episodes={result.get('rollout_n_episodes')}")

    result["status"] = "done"
    _save_result(cat, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parallel-train", type=int, default=3)
    ap.add_argument("--categories", nargs="+", default=None,
                    help="default: all held-in categories not yet status=done")
    ap.add_argument("--eval-port", type=int, default=5572,
                    help="distinct from the sequential driver's 5571, in case it's still running")
    args = ap.parse_args()

    if args.categories:
        categories = args.categories
    else:
        categories = [c for c in ALL_HELD_IN if _load_result(c).get("status") != "done"]

    print(f"[parallel] categories to process: {categories}  "
         f"max_parallel_train={args.max_parallel_train}  eval_port={args.eval_port}", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_parallel_train) as ex:
        futures = {ex.submit(convert_and_train, cat, args.eval_port): cat for cat in categories}
        for future in as_completed(futures):
            cat, result = future.result()
            # eval + rollout run HERE, in the main thread -- sequential across
            # categories, overlapping with the remaining categories' training
            # continuing in the background worker threads.
            eval_and_rollout(cat, result, args.eval_port)

    print("[parallel] DONE", flush=True)


if __name__ == "__main__":
    main()
