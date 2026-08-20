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

# episodes.csv columns (stable order). The obj_*/home_*/mat_* columns are the RANDOMIZATION
# parameters actually applied to that episode (object initial offset + orientation, arm-home
# jitter, object material) — the audit trail that makes runs comparable. Blank when a given DR
# is disabled; mat_* are constant during eval (material not per-episode randomized).
CSV_FIELDS = ["episode", "batch", "env", "scenario_seed", "success", "ever_success",
              # ever_in_band: object EVER within the task's [success_z_min,success_z_max] at
              # any single step, WITHOUT the hold_steps-consecutive requirement — the loose
              # counterpart to ever_success. The gap (ever_in_band - ever_success) measures
              # how often the policy reaches the target but fails to hold it stably there.
              # Blank for tasks with no absolute z-band.
              "ever_in_band",
              "first_success_step", "steps", "episode_reward",
              # 9 stress metrics: 4 spatial (mean/max/top10/top20 over particles) x {tmax, ttop20
              # over time} + mean_tmean backup. tmax=worst instant; ttop20=mean of hottest 20%% of
              # timesteps (idle-immune). max_tmax==old stress_peak; mean_tmean==old stress_mean.
              "stress_mean_tmax", "stress_mean_ttop20", "stress_max_tmax", "stress_max_ttop20",
              "stress_top10_tmax", "stress_top10_ttop20", "stress_top20_tmax", "stress_top20_ttop20",
              "stress_mean_tmean",
              # object height diagnostic (any task with SimFeedback.object_center) — peak reached
              # and value at the final step; cross-reference against the task's success z-band.
              "obj_z_max", "obj_z_final",
              "obj_dx", "obj_dy", "obj_roll", "obj_pitch", "obj_yaw",
              "home_dx", "home_dy", "home_dz", "mat_E", "mat_nu", "mat_rho", "mat_yield",
              "obj_scale", "obj_bend_deg", "obj_twist_deg", "obj_taper", "obj_rbf",
              "obj_axis_scale", "obj_axis",
              # ── APPEND-ONLY below. New columns go at the END so existing parsers that key on
              # position keep working, and any record missing them is written blank (see
              # write_episodes_csv) rather than erroring.
              #
              # Trajectory smoothness / "human-likeness" of the EE path (gentle_manip.evaluation
              # .smoothness). BLANK unless the venv exposes `policy_dt` — jerk is a third
              # derivative, so a chunked venv's aliased trace would give wrong, not just noisy,
              # numbers. sparc: spectral arc length (less negative = smoother, human reaches
              # ~ -1.4..-1.6); njerk: dimensionless jerk (lower = smoother); vpeaks: submovement
              # count (a human point-to-point reach has exactly 1).
              "ee_sparc", "ee_njerk", "ee_vpeaks", "ee_path_len", "grip_sparc", "grip_njerk",
              # Grasp-quality audit, contributed by the POLICY via the optional episode_metrics()
              # hook. Blank for every learned policy; populated by the scripted grasp synthesizer,
              # which is the only thing that knows which grasp it chose and why.
              # stem_grasp / pinch_grasp are 0/1 indicators of the two defects v4 targets.
              "grasp_tilt_deg", "grasp_min_pad_mm2", "grasp_com_lever_mm", "grasp_width_mm",
              "grasp_occ_pred", "grasp_stress_pred_pa", "grasp_score", "grasp_status",
              "stem_grasp", "pinch_grasp"]


def _nan(x) -> float:
    return float("nan") if x is None else float(x)


def write_episodes_csv(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def _clean(vals: List[float]) -> np.ndarray:
    return np.asarray([v for v in vals if v is not None and not math.isnan(v)], dtype=float)


def _mean_std(vals: List[float]):
    a = _clean(vals)
    if a.size == 0:
        return None, None
    return float(a.mean()), float(a.std())


def _pct(a: np.ndarray, q: float):
    return float(np.percentile(a, q)) if a.size else None


# The 9 per-episode stress metrics (see harness): 4 spatial reductions x {tmax, ttop20} + a
# time-mean backup. True flag -> also report P90/P95 over episodes (worst-case bruising tail).
STRESS_COLS = [
    ("stress_mean_tmax", False), ("stress_mean_ttop20", False),
    ("stress_max_tmax", True), ("stress_max_ttop20", False),     # max_tmax == the classic peak
    ("stress_top10_tmax", False), ("stress_top10_ttop20", False),
    ("stress_top20_tmax", False), ("stress_top20_ttop20", True),  # headline interaction tail
    ("stress_mean_tmean", False),                                 # backup == old stress_mean
]


# Auxiliary per-episode columns to aggregate into summary.json: (column, success_gated).
# Smoothness is NOT gated — a failed episode still has a trajectory worth scoring. The grasp audit
# IS gated, matching the stress convention (a grasp that never happened shouldn't set the average).
AUX_COLS = [
    ("ee_sparc", False), ("ee_njerk", False), ("ee_vpeaks", False), ("ee_path_len", False),
    ("grip_sparc", False), ("grip_njerk", False),
    ("grasp_tilt_deg", True), ("grasp_min_pad_mm2", True), ("grasp_com_lever_mm", True),
    ("grasp_width_mm", True), ("grasp_occ_pred", True), ("grasp_stress_pred_pa", True),
]
# 0/1 defect indicators -> reported as RATES over ALL episodes (counting failures is the point).
AUX_RATE_COLS = ["stem_grasp", "pinch_grasp"]


def aggregate(records: List[Dict[str, Any]], **meta) -> Dict[str, Any]:
    """Aggregate per-episode records into the summary dict. **meta = checkpoint, experiment,
    spec fields, etc. — passed straight through for provenance.

    IMPORTANT (gentleness metric): stress aggregates are SUCCESS-GATED — computed only over
    episodes with task success. A failed episode (agent never touched the object) has near-zero
    stress and would otherwise fool the mean into looking gentle while not doing the task.
    success_rate itself is over ALL episodes. See STRESS_COLS / harness for the 9 stress metrics.
    """
    n = len(records)
    succ = [bool(r["success"]) for r in records]
    ever = [bool(r["ever_success"]) for r in records]
    has_stress = _clean([r.get("stress_max_tmax") for r in records]).size > 0
    succ_recs = [r for r in records if bool(r["success"])]
    n_succ_stress = int(_clean([r.get("stress_max_tmax") for r in succ_recs]).size)

    # ever_in_band: blank/None for tasks with no absolute success z-band (see harness.py).
    in_band_vals = [r.get("ever_in_band") for r in records if r.get("ever_in_band") is not None]

    out = {
        **meta,
        "n_episodes": n,
        "success_rate": float(np.mean(succ)) if n else 0.0,
        "ever_success_rate": float(np.mean(ever)) if n else 0.0,
        "mean_episode_reward": float(np.mean([r["episode_reward"] for r in records])) if n else 0.0,
        "is_soft_task": has_stress,
    }
    if in_band_vals:
        out["ever_in_band_rate"] = float(np.mean([bool(v) for v in in_band_vals]))
        # the diagnostic gap: reaches the target vs. reaches AND holds it stably
        out["hold_failure_gap"] = out["ever_in_band_rate"] - out["ever_success_rate"]
    if has_stress:
        out["stress_n_success"] = n_succ_stress
        for col, want_pct in STRESS_COLS:
            vals = _clean([r.get(col) for r in succ_recs])            # success-gated
            out[col + "_mean"] = _nan(float(vals.mean())) if vals.size else None
            if want_pct:                                              # headline metrics get spread + tail
                out[col + "_std"] = _nan(float(vals.std())) if vals.size else None
                out[col + "_p90"] = _pct(vals, 90) if vals.size else None
                out[col + "_p95"] = _pct(vals, 95) if vals.size else None
        # all-episode backup mean (exposes the "gentle but failed" trap; compare to old evals)
        allm, _ = _mean_std([r.get("stress_mean_tmean") for r in records])
        out["stress_mean_tmean_mean_all"] = _nan(allm)

    # ── auxiliary aggregates (smoothness + grasp-quality audit) ──────────────
    # Only emitted when the columns are actually populated, so summaries from runs without them
    # are unchanged. Smoothness is over ALL episodes (a failed episode still has a trajectory, and
    # its smoothness is meaningful); the grasp audit is SUCCESS-GATED for the same reason stress is
    # — except the defect RATES, which are the whole point and must count failures too.
    for col, gated in AUX_COLS:
        src = succ_recs if gated else records
        vals = _clean([r.get(col) for r in src])
        if vals.size:
            out[col + "_mean"] = _nan(float(vals.mean()))
    for col in AUX_RATE_COLS:                       # defect rates over ALL episodes
        vals = _clean([r.get(col) for r in records])
        if vals.size:
            out[col + "_rate"] = float(vals.mean())
    return out


def write_summary(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
