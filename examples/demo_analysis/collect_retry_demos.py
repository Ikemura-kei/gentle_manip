"""Produce GENUINE slip -> regrasp -> SUCCESS recovery demo video(s) per fragile25
category, using collect_demos_synth_v3.py's --retry-on-slip (see that file's
"Robustness idea #2" section).

IMPORTANT (2026-08-21, per user correction): this script NEVER uses
--induce-slip-envs. An earlier version deliberately forced a mid-lift release on
a known-good grasp to guarantee a demo; the user rightly rejected that as not a
genuine recovery (it manufactures the failure it then "recovers" from). Every
video this script keeps comes from an UNPROMPTED slip during the batch's own
scripted attempt -- the FSM only enters the regrasp phase because the first
attempt actually failed the real height check, never because we told it to.

Per category: run natural (--retry-on-slip only) batches across successive
seeds, collecting every genuinely-recovered episode found. Stops once at least
one natural recovery is found; if none turns up after --max-seed-tries batches,
the category is honestly reported as "no natural regrasp observed" -- no video
is fabricated.

Usage:
    uv run --project envs/sim python examples/demo_analysis/collect_retry_demos.py \\
        --categories tomato dumpling gelatin --out-dir .report_scratch/retry_demos
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ALL_CATEGORIES = ["banana", "cherry", "grape", "kiwi", "mushroom", "pasta_bundle",
                  "raspberry", "shrimp", "tomato", "blackberry", "scallop",
                  "dumpling", "gelatin"]


def _run_collect(category: str, out_dir: Path, seed: int, n_envs: int,
                 log_path: Path, maxfevals: int = 400) -> int:
    cmd = ["uv", "run", "--project", "envs/sim", "python",
          "grasp_synthesis/collect_demos_synth_v3.py",
          "--experiment", f"single_lift_{category}_soft_easy",
          "--n-episodes", str(n_envs), "--n-envs", str(n_envs),
          "--maxfevals", str(maxfevals), "--scene-dr-every", "0", "--seed", str(seed),
          "--out-dir", str(out_dir), "--record-video", "--keep-failures", "--retry-on-slip"]
    sub_env = os.environ.copy()
    sub_env.pop("PYTHONPATH", None)
    sub_env.setdefault("MUJOCO_GL", "egl")
    print(f"[{category}] $ {' '.join(cmd)}", flush=True)
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, env=sub_env)
    return r.returncode


def _parse_bool_list(text: str, prefix: str) -> list[bool] | None:
    m = re.search(prefix + r": \[(.*?)\]", text)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return []
    return [tok.strip() == "True" for tok in inner.split(",")]


def _parse_int_list(text: str, prefix: str) -> list[int]:
    m = re.search(prefix + r": \[(.*?)\]", text)
    if not m:
        return []
    inner = m.group(1).strip()
    return [int(tok) for tok in inner.split(",")] if inner else []


def _find_video(videos: list[Path], env_i: int, tag: str) -> str | None:
    v = next((v for v in videos if f"env{env_i}_{tag}" in v.name), None)
    return str(v) if v else None


def process_category(category: str, out_root: Path, seed0: int, n_envs: int,
                     max_seed_tries: int) -> dict:
    cat_dir = out_root / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    result = {"category": category, "recovered_video": None, "recovered_videos": [],
             "success_video": None, "seeds_tried": [], "note": ""}

    for attempt in range(max_seed_tries):
        seed = seed0 + attempt
        result["seeds_tried"].append(seed)
        run_dir = cat_dir / f"seed{seed}"
        log = cat_dir / f"seed{seed}.log"
        _run_collect(category, run_dir, seed, n_envs, log)
        text = log.read_text(errors="ignore")
        success = _parse_bool_list(text, "Success")
        recovered = _parse_int_list(text, "Recovered from slip")
        if success is None:
            print(f"[{category}] seed {seed}: run crashed, see {log}", flush=True)
            continue

        good_envs = [i for i, s in enumerate(success) if s]
        videos = sorted(run_dir.glob("**/videos/*.mp4"))
        recovered_success = [i for i in recovered if i < len(success) and success[i]]
        new_recovered = [_find_video(videos, i, "regrasp_recovered") for i in recovered_success]
        new_recovered = [v for v in new_recovered if v]
        result["recovered_videos"].extend(new_recovered)

        if not result["success_video"]:
            plain_good = [i for i in good_envs if i not in recovered]
            if plain_good:
                result["success_video"] = _find_video(videos, plain_good[0], "success")

        if new_recovered:
            print(f"[{category}] seed {seed}: {len(new_recovered)} NATURAL recovery "
                 f"instance(s) found", flush=True)
            break
        print(f"[{category}] seed {seed}: no natural slip+recovery this batch "
             f"({len(good_envs)}/{n_envs} clean successes, {len(recovered)} slipped "
             f"but none recovered)", flush=True)

    if result["recovered_videos"]:
        result["recovered_video"] = result["recovered_videos"][0]
        result["note"] = (f"{len(result['recovered_videos'])} natural recovery instance(s) "
                          f"across seeds {result['seeds_tried']}")
    else:
        result["note"] = (f"no natural regrasp observed across {len(result['seeds_tried'])} "
                          f"batches (seeds {result['seeds_tried']}) -- not fabricated")
    print(f"[{category}] {result['note']}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=ALL_CATEGORIES)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--max-seed-tries", type=int, default=5)
    ap.add_argument("--force", action="store_true",
                    help="reprocess even if summary.json already exists")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cat in args.categories:
        summary_path = args.out_dir / cat / "summary.json"
        if summary_path.exists() and not args.force:
            print(f"[{cat}] already processed -- skipping", flush=True)
            results.append(json.loads(summary_path.read_text()))
            continue
        r = process_category(cat, args.out_dir, args.seed, args.n_envs, args.max_seed_tries)
        (args.out_dir / cat).mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(r, indent=2))
        results.append(r)

    print("\n=== Retry-demo collection summary ===")
    for r in results:
        print(f"  {r['category']:15s} recovered={bool(r.get('recovered_video'))}  "
             f"n_recovered={len(r.get('recovered_videos') or [])}  "
             f"success_clip={bool(r.get('success_video'))}  note={r.get('note')}")


if __name__ == "__main__":
    main()
