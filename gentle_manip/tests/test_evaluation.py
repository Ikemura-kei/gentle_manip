"""Shared eval harness: EvalSpec determinism + metrics aggregation/CSV (genesis-free)."""
import csv
import json

import numpy as np
import pytest

from gentle_manip.evaluation import EvalSpec
from gentle_manip.evaluation.harness import eval_out_dir
from gentle_manip.evaluation.metrics import CSV_FIELDS, aggregate, write_episodes_csv, write_summary


# ── EvalSpec ──────────────────────────────────────────────────────────────────
def test_spec_canonical_defaults_and_batches():
    s = EvalSpec()
    assert (s.n_episodes, s.num_envs, s.seed) == (100, 5, 0)
    assert s.n_batches == 20

def test_spec_rejects_non_divisible():
    with pytest.raises(ValueError):
        EvalSpec(n_episodes=100, num_envs=3)

def test_seed_for_batch_is_deterministic_and_distinct():
    s = EvalSpec(seed=7)
    seeds = [s.seed_for_batch(i) for i in range(20)]
    assert seeds == [EvalSpec(seed=7).seed_for_batch(i) for i in range(20)]   # reproducible
    assert len(set(seeds)) == 20                                             # distinct per batch
    assert EvalSpec(seed=0).seed_for_batch(3) != EvalSpec(seed=1).seed_for_batch(3)


# ── output dir (option b) ─────────────────────────────────────────────────────
def test_eval_out_dir_uses_run_dir_when_under_checkpoint(tmp_path):
    ckpt = tmp_path / "myrun" / "checkpoint" / "state_3000.pt"
    ckpt.parent.mkdir(parents=True)
    out = eval_out_dir(ckpt)
    assert out.parent == tmp_path / "myrun" / "eval"          # <run>/eval/<datetime>

def test_eval_out_dir_falls_back_off_run_dir(tmp_path):
    out = eval_out_dir(tmp_path / "loose.pt")
    assert "logs/eval" in str(out)


# ── metrics ───────────────────────────────────────────────────────────────────
def _recs(soft=True):
    out = []
    for k in range(4):
        out.append({"episode": k, "batch": k // 2, "env": k % 2, "scenario_seed": k // 2,
                    "success": int(k % 2 == 0), "ever_success": 1, "first_success_step": 10,
                    "steps": 75, "episode_reward": 10.0 * k,
                    "stress_peak": (1000.0 * (k + 1)) if soft else None,
                    "stress_mean": (300.0 * (k + 1)) if soft else None})
    return out

def test_aggregate_success_and_stress_soft():
    agg = aggregate(_recs(soft=True), checkpoint="c", experiment="e")
    assert agg["n_episodes"] == 4
    assert agg["success_rate"] == 0.5 and agg["ever_success_rate"] == 1.0
    assert agg["is_soft_task"] and agg["stress_peak_mean"] == pytest.approx(2500.0)
    assert agg["checkpoint"] == "c" and agg["experiment"] == "e"

def test_aggregate_rigid_has_no_stress():
    agg = aggregate(_recs(soft=False))
    assert not agg["is_soft_task"]
    assert agg["stress_peak_mean"] is None and agg["stress_mean_mean"] is None

def test_write_episodes_csv_roundtrip(tmp_path):
    write_episodes_csv(_recs(soft=True), tmp_path / "episodes.csv")
    rows = list(csv.DictReader(open(tmp_path / "episodes.csv")))
    assert len(rows) == 4 and list(rows[0].keys()) == CSV_FIELDS
    assert rows[0]["success"] == "1" and float(rows[3]["stress_peak"]) == 4000.0

def test_write_summary_json(tmp_path):
    write_summary(aggregate(_recs(), checkpoint="c"), tmp_path / "summary.json")
    d = json.loads((tmp_path / "summary.json").read_text())
    assert d["success_rate"] == 0.5 and d["is_soft_task"]
