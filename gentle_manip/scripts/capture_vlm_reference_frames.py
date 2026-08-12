"""Automate VLM reference-frame capture (gentle_manip 25-category speed/scale
pass, 2026-08-13): previously `category_reference_frames/*.png` was a MANUAL
step ("frame 0 of a canonical eval render clip") -- only apple/avocado/kiwi
existed. At 25-category scale this needs to be scripted: build a fresh 1-env
scene for each experiment, settle the object, grab one RGB frame, save it.

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.capture_vlm_reference_frames \
        --experiments single_lift_tofu_soft_easy single_lift_shrimp_soft_easy ... \
        [--categories tofu shrimp ...]   # optional; default: derived from experiment name
        [--precompute]                   # also run precompute_vlm_embeddings.py afterward

Run in envs/sim (needs genesis). Writes
gentle_manip/assets/category_reference_frames/<category>.png, one per
experiment, skipping any that already exist (use --overwrite to redo them).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "grasp_synthesis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import imageio  # noqa: E402

from gentle_manip.envs.genesis_worker import GenesisWorker  # noqa: E402
from gentle_manip.experiment import Experiment  # noqa: E402
from gentle_manip.tasks.single_lift import SingleLiftTask  # noqa: E402

REF_DIR = REPO / "gentle_manip" / "assets" / "category_reference_frames"

# Mirrors the naming convention used throughout this session's "_easy" configs
# and RolloutCollectorAgent._default_out_dir's category-dir derivation.
_SUFFIX_RE = re.compile(r"_(rigid|soft)(_easy)?$")


def _category_from_experiment(exp_name: str) -> str:
    stem = exp_name
    if stem.startswith("single_lift_"):
        stem = stem[len("single_lift_"):]
    return _SUFFIX_RE.sub("", stem)


def capture_one(experiment: str, category: str, settle_steps: int = 300) -> Path:
    exp = Experiment.load(experiment)
    task = SingleLiftTask(exp.task_cfg)
    spec = task.scene_spec
    worker = GenesisWorker(spec, num_envs=1, show_viewer=False, render_obs_cameras=True)
    try:
        worker.reset()
        for _ in range(settle_steps):
            worker.handle.scene.step()
        frame = worker.render_rgb(all_envs=False)   # (H, W, 3) uint8, env 0
        if frame is None:
            raise RuntimeError(f"{experiment}: no camera built, cannot capture a reference frame")
        REF_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REF_DIR / f"{category}.png"
        imageio.imwrite(str(out_path), frame)
        return out_path
    finally:
        worker.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--categories", nargs="+", default=None,
                    help="one per --experiments entry, same order (default: derived "
                         "from each experiment name, e.g. single_lift_tofu_soft_easy -> tofu)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--settle-steps", type=int, default=300)
    ap.add_argument("--precompute", action="store_true",
                    help="also invoke precompute_vlm_embeddings.py for every captured category")
    args = ap.parse_args()

    categories = args.categories or [_category_from_experiment(e) for e in args.experiments]
    if len(categories) != len(args.experiments):
        raise SystemExit("--categories must have the same length as --experiments")

    captured = []
    for exp_name, category in zip(args.experiments, categories):
        out_path = REF_DIR / f"{category}.png"
        if out_path.exists() and not args.overwrite:
            print(f"[capture_vlm] {category}: already exists ({out_path}), skipping "
                 f"(--overwrite to redo)", flush=True)
            captured.append(category)
            continue
        print(f"[capture_vlm] {category}: building {exp_name} scene...", flush=True)
        try:
            p = capture_one(exp_name, category, settle_steps=args.settle_steps)
            print(f"[capture_vlm] {category}: -> {p}", flush=True)
            captured.append(category)
        except Exception as e:
            print(f"[capture_vlm] {category}: FAILED ({e})", flush=True)

    print(f"\n[capture_vlm] DONE — {len(captured)}/{len(args.experiments)} reference "
         f"frames captured: {captured}", flush=True)

    if args.precompute and captured:
        import subprocess
        cmd = [sys.executable, "-m", "gentle_manip.scripts.precompute_vlm_embeddings",
              "--categories", *captured]
        print(f"[capture_vlm] running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()
