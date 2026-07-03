"""Shared, algorithm-agnostic evaluation harness (SERL, DPPO, and future methods).

The CANONICAL evaluation protocol lives here so every algorithm's eval is apples-to-apples:
a fixed set of scenarios (EvalSpec: 100 episodes / 5 sub-envs / deterministic per-batch DR
seed), success + (soft-body) stress reported both in aggregate (summary.json) and per
episode×env (episodes.csv), and outputs written into the evaluated policy's own training run
dir (<run>/eval/<datetime>/). Genesis-free — imports in every env (dppo/serl/dp3).

Each algorithm supplies two thin adapters (see eval_venv.py): an EvalVenv (drives the sim,
returns per-env success/stress) and a Policy (obs -> action). run_eval() owns everything else.
"""
from gentle_manip.evaluation.eval_spec import EvalSpec
from gentle_manip.evaluation.harness import run_eval

__all__ = ["EvalSpec", "run_eval"]
