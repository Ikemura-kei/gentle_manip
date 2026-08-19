"""Direct-from-synth generalist: train ONE BC policy directly on merged RAW v3
(FEM gentleness synthesis) demonstrations for the 9 held-in categories --
skipping BOTH per-category specialist training AND RLDG rollout distillation
entirely. This is the comparison arm requested 2026-08-19: does the extra
specialist->eval-gate->RLDG-rollout pipeline actually earn its cost over just
throwing more raw synthesized demos (~150/category, see
augment_heldin_to_150.py) at one generalist?

Mirrors run_fragile25_merge_and_train.py's merge+train recipe exactly (same
architecture, same category_embed=vlm conditioning, same training config) and
run_fragile25_final_eval.py's eval_one() recipe for the canonical harness --
only the INPUT to the merge differs: raw synthesized data.pkl paths instead of
RLDG rollout_data_path values. Own MERGE_NAME so its configs/checkpoints never
collide with the existing RLDG-distilled generalist
(single_lift_fragile25_generalist_pcd).

Writes results to logs/full_eval_campaign/direct_generalist/<cat>.json --
same schema as run_full_eval_campaign.py's generalist/specialist dirs, so the
report's existing loading code picks this up as a third role with no changes.

Usage:
    uv run --project envs/dppo python -m gentle_manip.scripts.run_direct_generalist
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gentle_manip.scripts.run_fragile25_specialist import (  # noqa: E402
    RESULTS_DIR, DPPO_CFG_DIR,
)
from gentle_manip.scripts.run_fragile25_merge_and_train import TRAIN_TEMPLATE  # noqa: E402
from gentle_manip.scripts.run_fragile25_final_eval import EVAL_TEMPLATE  # noqa: E402

MERGE_NAME = "single_lift_fragile25_direct_generalist_pcd"
HELD_IN = ["banana", "cherry", "grape", "kiwi", "mushroom", "pasta_bundle",
          "raspberry", "shrimp", "tomato"]
ZERO_SHOT = ["blackberry", "scallop", "dumpling", "gelatin"]
PORT = 5580
N_EPISODES = 100

OUT_DIR = REPO / "logs" / "full_eval_campaign" / "direct_generalist"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = RESULTS_DIR / "direct_generalist.json"


def _find_raw_demo_dir(category: str) -> Path | None:
    """Best RAW (v3 CMA-ES/FEM-synthesized) demo run dir for this category --
    deliberately excludes "-rollout-" dirs (RLDG self-distilled rollout output,
    up to 150 episodes/category), which run_fragile25_specialist.find_latest_
    demo_dir()'s "most episodes wins" heuristic would otherwise pick over the
    much smaller raw collection dir. That's the whole point of this comparison
    arm -- training on RLDG rollouts here would just reproduce the existing
    generalist's data, not test the "no RLDG" hypothesis. Auto-merges leftover
    shard_*.pkl into data.pkl if a run was killed before its own final merge
    (mirrors find_latest_demo_dir's same fix)."""
    task_dir = REPO / "dataset" / "demos" / f"single_lift_{category}_soft"
    if not task_dir.exists():
        return None
    import pickle
    best, best_n = None, -1
    for d in task_dir.iterdir():
        if not d.is_dir() or "rollout" in d.name:
            continue
        dp = d / "data.pkl"
        if not dp.exists() and list(d.glob("shard_*.pkl")):
            sys.path.insert(0, str(REPO / "grasp_synthesis"))
            from collect_demos_synth_v3 import _merge_shards
            dp_merged = _merge_shards(d)
            if dp_merged is not None:
                dp = dp_merged
        if not dp.exists():
            continue
        try:
            with open(dp, "rb") as f:
                n = len(pickle.load(f)["episodes"])
        except Exception:
            continue
        if n > best_n:
            best, best_n = d, n
    return best


def build_merge(out_dir: Path) -> dict:
    """Raw v3 data.pkl path per held-in category -- NOT a rollout path, and NOT
    gated by the RLDG quality-gate (that gate exists to filter self-distilled
    rollouts; raw synthesized demos are already gentleness-gated at collection
    time, see grasp_synthesis/collect_demos_synth_v3.py)."""
    cats = {}
    for cat in HELD_IN:
        demo_dir = _find_raw_demo_dir(cat)
        if demo_dir is None:
            raise RuntimeError(f"{cat}: no raw v3 demos found -- run "
                               f"augment_heldin_to_150.py first")
        cats[cat] = str(demo_dir / "data.pkl")

    link_dir = REPO / "dataset" / "demos_merged_direct_generalist_TEMP"
    if link_dir.exists():
        for f in link_dir.iterdir():
            f.unlink()
        link_dir.rmdir()
    link_dir.mkdir(parents=True)
    for cat, src in cats.items():
        (link_dir / f"{cat}.pkl").symlink_to(src)

    cmd = ["uv", "run", "--project", "envs/dppo", "python", "-m", "gentle_manip.dppo.convert_demos",
          str(link_dir), "--out", str(out_dir), "--point-cloud",
          "--experiment", "single_lift_mushroom_soft_easy", "--view", "student",
          "--val-split", "0.1", "--category-embed", "--embed-source", "vlm"]
    print(f"[direct_generalist] merging {len(cats)} RAW categories: {sorted(cats)}", flush=True)
    print(f"[direct_generalist] $ {' '.join(cmd)}", flush=True)
    sub_env = os.environ.copy()
    sub_env.pop("PYTHONPATH", None)
    r = subprocess.run(cmd, cwd=str(REPO), env=sub_env)
    if r.returncode != 0:
        raise RuntimeError("convert_demos.py (direct-generalist merge) failed")
    return cats


def train() -> dict:
    from gentle_manip.scripts.train_with_resume import train_with_resume

    out_dir = Path(os.environ.get("DPPO_DATA_DIR", str(REPO / "dppo_data"))) / MERGE_NAME
    cats = build_merge(out_dir)

    cfg_dir = DPPO_CFG_DIR / MERGE_NAME
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "pre_diffusion_pointnet.yaml").write_text(
        TRAIN_TEMPLATE.format(merge_name=MERGE_NAME))

    result = train_with_resume(
        config_path=str(cfg_dir), config_name="pre_diffusion_pointnet",
        task="single_lift_mushroom_soft_easy", max_retries=5, timeout_s=21600,
        log_path=RESULTS_DIR / "direct_generalist_train.log")

    manifest = {"categories": sorted(cats), "n_categories": len(cats),
               "dppo_data_dir": str(out_dir), "cfg_dir": str(cfg_dir), **result}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def eval_one(category: str, checkpoint: str) -> dict:
    cfg_dir = DPPO_CFG_DIR / MERGE_NAME / f"eval_{category}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "eval_diffusion_pointnet.yaml").write_text(
        EVAL_TEMPLATE.format(obj=category, role="direct-generalist", merge_name=MERGE_NAME,
                             port=PORT, n_episodes=N_EPISODES))

    log_dir = RESULTS_DIR / "direct_generalist_eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / f"{category}_server.log"
    sub_env = os.environ.copy()
    sub_env.pop("PYTHONPATH", None)
    server_cmd = ["uv", "run", "--project", "envs/sim", "python", "-m",
                 "gentle_manip.scripts.serl_sim_server",
                 "--experiment", f"single_lift_{category}_soft_easy", "--view", "student",
                 "--num-envs", "5", "--render-rgb", "--subprocess", "--port", str(PORT)]
    print(f"[direct_generalist] {category}: starting sim server...", flush=True)
    with open(server_log, "w") as logf:
        proc = subprocess.Popen(server_cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
                                env=sub_env, start_new_session=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 120:
            if server_log.exists() and "SIM_SERVER_READY" in server_log.read_text(errors="ignore"):
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"sim server for {category} not ready in 120s")

        eval_log = log_dir / f"{category}.log"
        cmd = ["uv", "run", "--project", "envs/dppo", "python", "-m", "gentle_manip.dppo.train",
              "--config-name", "eval_diffusion_pointnet", "--config-path", str(cfg_dir),
              f"base_policy_path={checkpoint}"]
        with open(eval_log, "w") as logf:
            r = subprocess.run(cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, env=sub_env)
        text = eval_log.read_text(errors="ignore")
        m = re.search(r"DONE — success ([\d.]+)", text)
        sr = float(m.group(1)) if m else None
        summary = None
        eval_base = Path(checkpoint).parent.parent / f"eval_{category}"
        if eval_base.exists():
            run_dirs = sorted(eval_base.iterdir(), key=lambda p: p.stat().st_mtime)
            for d in reversed(run_dirs):
                sp = d / "summary.json"
                if sp.exists():
                    summary = json.loads(sp.read_text())
                    summary["render_dir"] = str(d / "render")
                    break
        return {"category": category, "role": "direct_generalist", "success_rate": sr,
               "ok": r.returncode == 0 and sr is not None, "eval_log": str(eval_log),
               "summary": summary}
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except ProcessLookupError:
            pass


def main() -> None:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        print(f"[direct_generalist] already trained: {manifest.get('checkpoint')}", flush=True)
    else:
        manifest = train()

    checkpoint = manifest.get("checkpoint")
    if not checkpoint:
        from gentle_manip.scripts.train_with_resume import find_best_checkpoint
        run_dir = Path(manifest["run_dir"])
        log_path = RESULTS_DIR / "direct_generalist_train.log"
        ckpt = find_best_checkpoint(run_dir, log_path if log_path.exists() else None)
        checkpoint = str(ckpt)
        manifest["checkpoint"] = checkpoint
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    for cat in HELD_IN + ZERO_SHOT:
        out_path = OUT_DIR / f"{cat}.json"
        if out_path.exists():
            print(f"[direct_generalist] {cat}: already evaluated, skipping", flush=True)
            continue
        r = eval_one(cat, checkpoint)
        out_path.write_text(json.dumps(r, indent=2))
        print(f"[direct_generalist] {cat}: success_rate={r['success_rate']} "
             f"combined={((r.get('summary') or {}).get('combined_sr_gentleness'))}", flush=True)

    print("[direct_generalist] DONE", flush=True)


if __name__ == "__main__":
    main()
