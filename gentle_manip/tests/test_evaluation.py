"""Shared eval harness: EvalSpec determinism + metrics aggregation/CSV (genesis-free)."""
import csv
import json

import numpy as np
import pytest

from gentle_manip.evaluation import EvalSpec
from gentle_manip.evaluation.harness import eval_out_dir, _scenario_columns
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

def test_scene_seed_for_group_deterministic_and_disjoint_from_batch():
    s = EvalSpec(seed=0)
    groups = [s.scene_seed_for_group(g) for g in range(4)]
    assert groups == [EvalSpec(seed=0).scene_seed_for_group(g) for g in range(4)]  # reproducible
    assert len(set(groups)) == 4                                                   # distinct per group
    batch = {s.seed_for_batch(i) for i in range(20)}
    assert not batch & set(groups)                          # scene seeds don't collide with pose seeds
    assert EvalSpec().scene_group_size == 0                 # default = fixed nominal geometry


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
_SCOLS = ["stress_mean_tmax", "stress_mean_ttop20", "stress_max_tmax", "stress_max_ttop20",
          "stress_top10_tmax", "stress_top10_ttop20", "stress_top20_tmax", "stress_top20_ttop20",
          "stress_mean_tmean"]
# per-record base value for each stress col (scaled by (k+1) in _recs)
_BASE = {"stress_mean_tmax": 400, "stress_mean_ttop20": 350, "stress_max_tmax": 1000,
         "stress_max_ttop20": 900, "stress_top10_tmax": 1100, "stress_top10_ttop20": 550,
         "stress_top20_tmax": 1050, "stress_top20_ttop20": 500, "stress_mean_tmean": 300}


def _recs(soft=True):
    out = []
    for k in range(4):
        r = {"episode": k, "batch": k // 2, "env": k % 2, "scenario_seed": k // 2,
             "success": int(k % 2 == 0), "ever_success": 1, "first_success_step": 10,
             "steps": 75, "episode_reward": 10.0 * k}
        for c in _SCOLS:
            r[c] = (_BASE[c] * (k + 1)) if soft else None
        out.append(r)
    return out

def test_aggregate_success_and_stress_soft():
    # success on k=0,2. stress cols scale by (k+1) -> success values are base*1 and base*3.
    agg = aggregate(_recs(soft=True), checkpoint="c", experiment="e")
    assert agg["n_episodes"] == 4
    assert agg["success_rate"] == 0.5 and agg["ever_success_rate"] == 1.0
    assert agg["is_soft_task"] and agg["stress_n_success"] == 2
    # headline peak = max_tmax: gated [1000, 3000] -> mean 2000, p90 2800, p95 2900
    assert agg["stress_max_tmax_mean"] == pytest.approx(2000.0)
    assert agg["stress_max_tmax_p90"] == pytest.approx(2800.0)
    assert agg["stress_max_tmax_p95"] == pytest.approx(2900.0)
    # interaction tail = top20_ttop20: gated [500,1500] -> 1000, p95 1450 (flagged for pctls)
    assert agg["stress_top20_ttop20_mean"] == pytest.approx(1000.0)
    assert agg["stress_top20_ttop20_p95"] == pytest.approx(1450.0)
    # backup mean_tmean: gated [300,900] -> 600; all-episode [300,600,900,1200] -> 750
    assert agg["stress_mean_tmean_mean"] == pytest.approx(600.0)
    assert agg["stress_mean_tmean_mean_all"] == pytest.approx(750.0)
    # a non-flagged col gets a gated mean but NO percentiles
    assert agg["stress_mean_tmax_mean"] == pytest.approx(800.0)      # [400,1200]
    assert "stress_mean_tmax_p95" not in agg
    assert agg["checkpoint"] == "c" and agg["experiment"] == "e"

def test_aggregate_stress_gate_ignores_gentle_but_failed():
    # the trap: a failed episode with tiny stress must NOT lower the gated mean.
    def _mk(succ, peak, tmean):
        r = {"episode": 0, "batch": 0, "env": 0, "scenario_seed": 0, "success": succ,
             "ever_success": succ, "first_success_step": 5 if succ else -1, "steps": 75,
             "episode_reward": 1.0}
        for c in _SCOLS:
            r[c] = 0.0
        r["stress_max_tmax"], r["stress_mean_tmean"] = peak, tmean
        return r
    recs = [_mk(1, 5000.0, 1000.0), _mk(0, 50.0, 10.0)]
    recs[1]["env"] = 1
    agg = aggregate(recs)
    assert agg["success_rate"] == 0.5 and agg["stress_n_success"] == 1
    assert agg["stress_max_tmax_mean"] == pytest.approx(5000.0)          # gated: only the success
    assert agg["stress_mean_tmean_mean_all"] == pytest.approx(505.0)     # all: (1000+10)/2 — the trap

def test_aggregate_no_success_stress_is_none():
    r = {"episode": 0, "batch": 0, "env": 0, "scenario_seed": 0, "success": 0, "ever_success": 0,
         "first_success_step": -1, "steps": 75, "episode_reward": 0.0}
    for c in _SCOLS:
        r[c] = 100.0
    agg = aggregate([r])
    assert agg["is_soft_task"] and agg["stress_n_success"] == 0
    assert agg["stress_max_tmax_mean"] is None and agg["stress_max_tmax_p95"] is None
    assert agg["stress_mean_tmean_mean_all"] == pytest.approx(100.0)

def test_aggregate_rigid_has_no_stress():
    agg = aggregate(_recs(soft=False))
    assert not agg["is_soft_task"]
    assert "stress_max_tmax_mean" not in agg

def test_write_episodes_csv_roundtrip(tmp_path):
    write_episodes_csv(_recs(soft=True), tmp_path / "episodes.csv")
    rows = list(csv.DictReader(open(tmp_path / "episodes.csv")))
    assert len(rows) == 4 and list(rows[0].keys()) == CSV_FIELDS
    assert rows[0]["success"] == "1" and float(rows[3]["stress_max_tmax"]) == 4000.0

def test_write_summary_json(tmp_path):
    write_summary(aggregate(_recs(), checkpoint="c"), tmp_path / "summary.json")
    d = json.loads((tmp_path / "summary.json").read_text())
    assert d["success_rate"] == 0.5 and d["is_soft_task"]


# ── per-episode randomization columns ─────────────────────────────────────────
def test_scenario_columns_parses_dr_and_material():
    sc = {"dr": {"object_dxy": [[0.01, -0.02], [0.03, 0.04]],
                 "object_euler": [[0.1, 0.2, 3.0], [0.0, 0.0, -1.0]],
                 "home_offset": None},                       # home DR disabled
          "material": {"E": 3e5, "nu": 0.35, "rho": 1050.0, "yield": 4e4}}
    cols = _scenario_columns(sc, 2)
    assert cols[0]["obj_dx"] == 0.01 and cols[0]["obj_yaw"] == 3.0
    assert "home_dx" not in cols[0]                          # disabled -> column blank
    assert cols[1]["mat_E"] == 3e5 and cols[0]["mat_yield"] == 4e4

def test_scenario_columns_none_is_empty():
    assert _scenario_columns(None, 3) == [{}, {}, {}]


# ── SimEvalVenv (state-policy venv over rpc) + ScriptedPolicy ──────────────────
class _FakeStateClient:
    """SimEnvClient stand-in: teacher-ish obs, per-env success/stress, scenario passthrough."""
    def __init__(self, n=5):
        self.n = n
        self.last_scenario = {"dr": {"object_dxy": [[0.0, 0.0]] * n, "object_euler": None,
                                     "home_offset": None}, "material": {"E": 3e5}}
        self.steps = 0

    def _obs(self):
        return {"ee_pos": np.zeros((self.n, 3), np.float32),
                "priv_object_pos": np.tile([0.4, 0.0, 0.02], (self.n, 1)).astype(np.float32),
                "gripper_width": np.full((self.n, 1), 0.08, np.float32)}

    def reseed(self, s): self.seeded = s
    def reset(self): return self._obs()
    def step(self, a):
        self.steps += 1
        info = [{"success": self.steps >= 3, "stress_max": 100.0 + i, "stress_mean": 10.0 + i}
                for i in range(self.n)]
        return self._obs(), np.ones(self.n, np.float32), np.zeros(self.n, bool), info
    def render(self): return None
    def close(self): pass


def test_sim_eval_venv_step_surfaces_success_stress_and_truncates():
    from gentle_manip.evaluation.sim_eval_venv import SimEvalVenv
    c = _FakeStateClient(5)
    v = SimEvalVenv(c, num_envs=5, max_episode_steps=3)
    v.seed([123, 123, 123, 123, 123]); assert c.seeded == 123
    v.reset_arg()
    obs, r, term, trunc, info = v.step(np.zeros((5, 7), np.float32))
    assert r.shape == (5,) and not trunc.any()
    assert "stress_max" in info and info["stress_max"].shape == (5,)
    v.step(np.zeros((5, 7), np.float32))
    _, _, _, trunc, info = v.step(np.zeros((5, 7), np.float32))   # cnt hits 3 -> truncate
    assert trunc.all() and info["success"].all() and "final_obs" in info
    assert v.scenario_params()["material"]["E"] == 3e5


def test_scripted_policy_produces_per_env_action_chunkless():
    from gentle_manip.scripts.eval_scripted import ScriptedPolicy
    p = ScriptedPolicy(num_envs=4, action_scales=[0.0052] * 6 + [0.005], rate_hz=30,
                       params=dict(lift_height=0.2, approach_height=0.12, grasp_z=0.006,
                                   grasp_gw=0.03, gripper_close=0.5, speed_cap=0.5))
    p.reset()
    obs = {"ee_pos": np.tile([0.4, 0.0, 0.3], (4, 1)),
           "priv_object_pos": np.tile([0.45, 0.02, 0.02], (4, 1)),
           "gripper_width": np.full((4, 1), 0.08)}
    a = p.act(obs)
    assert a.shape == (4, 7) and np.all(np.abs(a) <= 1.0)     # normalized deltas, per env

def test_dr_columns_survive_csv_roundtrip(tmp_path):
    recs = _recs(soft=True)
    for r, c in zip(recs, _scenario_columns({"dr": {"object_dxy": [[0.01, 0.02]] * 4,
                                                    "object_euler": None, "home_offset": None},
                                             "material": {"E": 3e5, "nu": 0.35, "rho": 1050.0,
                                                          "yield": 4e4}}, 4)):
        r.update(c)
    write_episodes_csv(recs, tmp_path / "e.csv")
    row = next(csv.DictReader(open(tmp_path / "e.csv")))
    assert float(row["obj_dx"]) == 0.01 and float(row["mat_E"]) == 3e5 and row["obj_yaw"] == ""
