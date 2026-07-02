"""Per-run output directories — one place for every algorithm's artifacts.

A run lives at  logs/<algo>/<task>/<run_name>/  with:
    config/        snapshot of the experiment + referenced configs (reproducibility)
    videos/        behaviour clips (mp4/episode)
    checkpoints/   policy checkpoints
    run_meta.json  run name, timestamp, git commit, + free-form extras

run_name defaults to  <exp_name>_<YYYYmmdd_HHMMSS>  and is meant to MATCH the wandb
run name (train_serl sets wandb's unique_identifier from the same timestamp), so the
local dir and the wandb run line up. Shared across algos (SERL, DP3, ...) — pass algo.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[2]


def make_run_name(exp_name: str, ts: Optional[str] = None) -> str:
    return f"{exp_name}_{ts or datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_dir(algo: str, task: str, run_name: str, base: str = "logs") -> Path:
    d = _REPO / base / algo / task / run_name
    for sub in ("config", "videos", "checkpoints"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO, text=True).strip()
    except Exception:
        return "unknown"


def snapshot_experiment(exp, dest: Path) -> None:
    """Copy the experiment yaml + every config it references into dest/config/."""
    cfg_dir = dest / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    src = _REPO / "gentle_manip" / "configs"
    d = exp._raw
    # (subdir, name) pairs the experiment composes; skip missing/None.
    refs = [("experiments", exp.name), ("tasks", d.get("task")), ("action", d.get("action")),
            ("obs", d.get("obs")), ("dr", d.get("dr")), ("augmentation", d.get("augmentation"))]
    for sub, name in refs:
        if not name:
            continue
        f = src / sub / f"{name}.yaml"
        if f.exists():
            shutil.copy2(f, cfg_dir / f"{sub}__{name}.yaml")


def write_run_meta(dest: Path, **extras) -> None:
    meta = {"run_name": dest.name, "timestamp": datetime.now().isoformat(),
            "git_commit": _git_commit(), **extras}
    (dest / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
