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


def write_experiment_md(dest: Path, *, algo: str, motivation: str = "", hypothesis: str = "",
                        config: Optional[dict] = None, wandb: str = "", **misc) -> Path:
    """Write a human-readable EXPERIMENT.md into the run dir at launch: commit, motivation,
    hypothesis, key config, plus empty Observations / Final-summary sections for the agent OR
    the user to fill during/after the run. Idempotent-ish: won't clobber an existing file (so
    hand-edited notes survive a relaunch) — pass a fresh run dir for a fresh experiment."""
    md = dest / "EXPERIMENT.md"
    if md.exists():
        return md
    lines = [
        f"# Experiment: {dest.name}", "",
        f"- **Algorithm:** {algo}",
        f"- **Started:** {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- **Git commit:** `{_git_commit()}`",
        f"- **Run dir:** `{dest}`",
    ]
    if wandb:
        lines.append(f"- **wandb:** {wandb}")
    for k, v in misc.items():
        lines.append(f"- **{k}:** {v}")
    lines += ["", "## Motivation", motivation or "_(why this run)_", "",
              "## Hypothesis", hypothesis or "_(what we expect / are testing)_", ""]
    if config:
        lines += ["## Config (key knobs)", "```yaml"]
        lines += [f"{k}: {v}" for k, v in config.items()]
        lines += ["```", ""]
    lines += ["## Observations (append during/after — agent + user)", "- ", "",
              "## Final summary (fill when the run ends)",
              "- duration:", "- learner steps:", "- replay buffer size at end:",
              "- return / succeed:", "- verdict:", ""]
    md.write_text("\n".join(lines))
    return md


def append_experiment_note(dest: Path, note: str, section: str = "Observations") -> None:
    """Append a timestamped bullet under a section heading of EXPERIMENT.md (best-effort)."""
    md = dest / "EXPERIMENT.md"
    if not md.exists():
        return
    text = md.read_text()
    bullet = f"- [{datetime.now():%H:%M}] {note}"
    marker = f"## {section}"
    if marker in text:                       # insert right after the section heading line
        head, _, tail = text.partition(marker)
        after_heading = tail.split("\n", 1)
        rest = after_heading[1] if len(after_heading) > 1 else ""
        text = head + marker + after_heading[0] + "\n" + bullet + "\n" + rest
    else:
        text += f"\n{bullet}\n"
    md.write_text(text)
