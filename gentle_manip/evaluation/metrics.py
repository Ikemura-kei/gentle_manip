"""Metrics aggregation + CSV/JSON writers for the shared eval harness.

Per-episode records (one per batch×env) -> episodes.csv (audit trail, one row each) and a
summary.json aggregate. Stress columns are NaN for rigid tasks (no von-Mises). Stdlib + numpy
only (no pandas) so it stays importable in every env.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# episodes.csv columns (stable order)
CSV_FIELDS = ["episode", "batch", "env", "scenario_seed", "success", "ever_success",
              "first_success_step", "steps", "episode_reward", "stress_peak", "stress_mean"]


def _nan(x) -> float:
    return float("nan") if x is None else float(x)


def write_episodes_csv(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def _mean_std(vals: List[float]):
    a = np.asarray([v for v in vals if v is not None and not math.isnan(v)], dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), float(a.std())


def aggregate(records: List[Dict[str, Any]], **meta) -> Dict[str, Any]:
    """Aggregate per-episode records into the summary dict. **meta = checkpoint, experiment,
    spec fields, etc. — passed straight through for provenance."""
    n = len(records)
    succ = [bool(r["success"]) for r in records]
    ever = [bool(r["ever_success"]) for r in records]
    peak_mean, peak_std = _mean_std([r.get("stress_peak") for r in records])
    smean_mean, _ = _mean_std([r.get("stress_mean") for r in records])
    has_stress = peak_mean is not None
    return {
        **meta,
        "n_episodes": n,
        "success_rate": float(np.mean(succ)) if n else 0.0,
        "ever_success_rate": float(np.mean(ever)) if n else 0.0,
        "mean_episode_reward": float(np.mean([r["episode_reward"] for r in records])) if n else 0.0,
        "stress_peak_mean": _nan(peak_mean) if has_stress else None,
        "stress_peak_std": _nan(peak_std) if has_stress else None,
        "stress_mean_mean": _nan(smean_mean) if has_stress else None,
        "is_soft_task": has_stress,
    }


def write_summary(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
