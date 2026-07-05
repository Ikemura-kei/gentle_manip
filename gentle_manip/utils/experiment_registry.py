"""Global experiment-ID registry — one short unique handle per TRAINING run.

Problem: training log dirs were long (`<algo>/<task>/<name>_<YYYY-MM-DD_HH-MM-SS>_<seed>`), and
the only unique part (the datetime) was buried at the end. This module mints a short **5-letter
ID** (e.g. `ajchd`) that uniquely identifies a run regardless of algorithm (DPPO, SERL, …) or
task, records it in a project-root table (`experiments.csv`), and lets a reconcile pass keep the
table in sync when runs are manually deleted.

Scope: **training runs only.** Eval runs keep their datetime naming (they nest under the trained
policy's own run dir) and are NOT registered.

Genesis-free, dependency-free (csv + random + pathlib) so it imports in every env (serl 3.10,
dppo 3.10, sim 3.12).

Two integration modes:
- **SERL** (builds its own dir): `eid = new_id(); rdir = logs/serl/<task>/<eid>`; then
  `add_entry(eid, "serl", task, name, rdir)`.
- **DPPO** (hydra builds the dir from a template containing the ID): the `exp_id` OmegaConf
  resolver (registered in `dppo/train.py`) returns a fresh cached ID so the logdir leaf IS the
  ID; a hydra callback then calls `add_entry(...)` with the resolved dir.

Table columns: id, algo, task, name, run_dir, created, git_commit, status.
"""
from __future__ import annotations

import csv
import random
import string
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[2]
TABLE_PATH = _REPO / "experiments.csv"
_COLS = ["id", "algo", "task", "name", "run_dir", "created", "git_commit", "status"]
_ID_LEN = 5


# ── table I/O ───────────────────────────────────────────────────────────────────
def load_table(path: Path = TABLE_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_table(rows: list[dict], path: Path = TABLE_PATH) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _COLS})


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_REPO, text=True).strip()
    except Exception:
        return "unknown"


# ── ID allocation ─────────────────────────────────────────────────────────────
def _looks_like_id(name: str) -> bool:
    return len(name) == _ID_LEN and name.isalpha() and name.islower()


def _dir_ids(base: Path) -> set[str]:
    """Every ID-shaped leaf dir already under logs/<algo>/<task>/<leaf> (so a fresh ID never
    collides with an existing run dir even if the table was lost)."""
    ids: set[str] = set()
    if not base.exists():
        return ids
    for algo in base.iterdir():
        if not algo.is_dir():
            continue
        for task in algo.iterdir():
            if not task.is_dir():
                continue
            for leaf in task.iterdir():
                if leaf.is_dir() and _looks_like_id(leaf.name):
                    ids.add(leaf.name)
    return ids


def existing_ids(table_path: Path = TABLE_PATH, base: str = "logs") -> set[str]:
    tbl = {r["id"] for r in load_table(table_path) if r.get("id")}
    return tbl | _dir_ids(_REPO / base)


def new_id(table_path: Path = TABLE_PATH, base: str = "logs",
           rng: Optional[random.Random] = None) -> str:
    """Mint a 5-lowercase-letter ID not present in the table or any existing run dir."""
    rng = rng or random.Random()
    taken = existing_ids(table_path, base)
    for _ in range(10000):
        cand = "".join(rng.choices(string.ascii_lowercase, k=_ID_LEN))
        if cand not in taken:
            return cand
    raise RuntimeError("could not allocate a unique experiment id (table exhausted?)")


# ── registration ────────────────────────────────────────────────────────────────
def add_entry(exp_id: str, algo: str, task: str, name: str, run_dir, *,
              status: str = "running", table_path: Path = TABLE_PATH) -> None:
    """Append a run row (idempotent: updates in place if the ID already exists)."""
    rows = load_table(table_path)
    row = {"id": exp_id, "algo": algo, "task": task, "name": name,
           "run_dir": str(run_dir), "created": datetime.now().isoformat(timespec="seconds"),
           "git_commit": _git_commit(), "status": status}
    for i, r in enumerate(rows):
        if r.get("id") == exp_id:
            row["created"] = r.get("created") or row["created"]      # keep original create time
            rows[i] = row
            _write_table(rows, table_path)
            return
    rows.append(row)
    _write_table(rows, table_path)


def set_status(exp_id: str, status: str, table_path: Path = TABLE_PATH) -> None:
    rows = load_table(table_path)
    for r in rows:
        if r.get("id") == exp_id:
            r["status"] = status
            _write_table(rows, table_path)
            return


# ── reconcile ─────────────────────────────────────────────────────────────────
def _infer(run_dir: Path, base: Path) -> tuple[str, str]:
    """(algo, task) from a logs/<algo>/<task>/<id> path; ('?','?') if it doesn't fit."""
    try:
        rel = run_dir.relative_to(base).parts
        if len(rel) >= 3:
            return rel[0], rel[1]
    except ValueError:
        pass
    return "?", "?"


def reconcile(table_path: Path = TABLE_PATH, base: str = "logs") -> dict:
    """Sync the table with the log tree:
    - drop rows whose run_dir no longer exists (user deleted a deprecated run);
    - add rows for ID-shaped run dirs present on disk but missing from the table (a launch that
      didn't register, or a table that was lost).
    Returns {'dropped': [...], 'added': [...], 'kept': N}.
    """
    base_p = _REPO / base
    rows = load_table(table_path)
    kept, dropped = [], []
    for r in rows:
        rd = Path(r.get("run_dir", ""))
        (kept if rd.exists() else dropped).append(r)

    known = {r["id"] for r in kept}
    added = []
    for eid in _dir_ids(base_p):
        if eid in known:
            continue
        # find the dir for this id
        matches = [p for p in base_p.glob(f"*/*/{eid}") if p.is_dir()]
        if not matches:
            continue
        rd = matches[0]
        algo, task = _infer(rd, base_p)
        row = {"id": eid, "algo": algo, "task": task, "name": task,
               "run_dir": str(rd),
               "created": datetime.fromtimestamp(rd.stat().st_mtime).isoformat(timespec="seconds"),
               "git_commit": "", "status": "found"}
        kept.append(row)
        added.append(row)

    _write_table(kept, table_path)
    return {"dropped": [r["id"] for r in dropped],
            "added": [r["id"] for r in added],
            "kept": len(kept) - len(added)}


def format_table(table_path: Path = TABLE_PATH) -> str:
    rows = load_table(table_path)
    if not rows:
        return "(no experiments registered)"
    cols = ["id", "algo", "task", "status", "created", "run_dir"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    lines = [head, "  ".join("-" * widths[c] for c in cols)]
    for r in sorted(rows, key=lambda x: x.get("created", "")):
        lines.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)
