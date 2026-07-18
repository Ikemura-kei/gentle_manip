"""Evaluate the SCRIPTED lift policy (the demo-collection expert) under the shared canonical
eval harness — a baseline reported identically to the learned policies (same EvalSpec, fixed
scenario sequence, summary.json + episodes.csv with per-episode DR params + stress, env-cfg
snapshot, env-0 videos). Algorithm name: "scripted_policy".

The scripted ScriptedLiftDemonstrator is single-env and reads privileged state via a backend;
here we run one instance PER sub-env behind a tiny shim that feeds it that env's slice of the
teacher obs (ee_pos, priv_object_pos, gripper_width). So the exact expert that generated the
demos is what we score — no reimplementation.

Needs a teacher-view server (has priv_object_pos + priv_stress):
    uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server \\
        --experiment single_lift_mushroom_soft --view teacher --num-envs 5 --render-rgb --port 5572
    uv run --project envs/sim python -m gentle_manip.scripts.eval_scripted --port 5572
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from gentle_manip.demos.scripted_policy import ScriptedLiftDemonstrator
from gentle_manip.envs.rpc import SimEnvClient
from gentle_manip.evaluation import EvalSpec, run_eval
from gentle_manip.evaluation.sim_eval_venv import SimEvalVenv
from gentle_manip.experiment import Experiment

_REPO = Path(__file__).resolve().parents[2]


class _ShimFeedback:
    def __init__(self, ee, obj, gw):
        self.ee_pos = ee[None]                # (1,3) — the demonstrator reads [0]
        self.object_center = obj[None]
        self.gripper_width = np.array([gw])


class _ShimBackend:
    def __init__(self, scale=1.0):
        self._scale = float(scale)

    def set(self, ee, obj, gw):
        self._fb = _ShimFeedback(ee, obj, gw)

    def get_sim_feedback(self):
        return self._fb

    def scene_params(self):                       # so the demonstrator can size-adapt its grasp
        return {"scale": self._scale}


class ScriptedPolicy:
    """Batched scripted expert: one ScriptedLiftDemonstrator per sub-env, fed that env's state."""

    def __init__(self, num_envs, action_scales, rate_hz, params, venv=None):
        self.num_envs = int(num_envs)
        self.scales = np.asarray(action_scales, np.float64)
        self.rate_hz = float(rate_hz)
        self.params = dict(params)
        self.venv = venv                          # to read the current scene's object scale
        self.demos = None

    def reset(self):
        scale = 1.0                               # full-DR: adapt the grasp to this scene's size
        if self.venv is not None:
            sc = (self.venv.scenario_params() or {}).get("scene") or {}
            scale = float(sc.get("scale", 1.0)) or 1.0
        self.demos = [ScriptedLiftDemonstrator(_ShimBackend(scale), self.scales, n_episodes=1,
                                               rate_hz=self.rate_hz, **self.params)
                      for _ in range(self.num_envs)]

    def act(self, obs):
        ee = np.asarray(obs["ee_pos"], np.float64)
        obj = np.asarray(obs["priv_object_pos"], np.float64)
        gw = np.asarray(obs["gripper_width"], np.float64).reshape(-1)
        acts = []
        for j, demo in enumerate(self.demos):
            demo.backend.set(ee[j], obj[j], gw[j])
            acts.append(demo.get_action())
        return np.stack(acts).astype(np.float32)      # (num_envs, 7)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect-config", type=Path,
                    default=_REPO / "gentle_manip/configs/collect/single_lift_mushroom_soft_scripted.yaml",
                    help="scripted-expert params + rate (must match demo collection)")
    ap.add_argument("--experiment", default=None, help="override; default = collect cfg's experiment")
    ap.add_argument("--port", type=int, default=5572)
    ap.add_argument("--num-envs", type=int, default=5, help="sub-envs (MUST match the server)")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=300, help="policy steps per episode (sim steps)")
    ap.add_argument("--record-batches", type=int, default=None,
                    help="batches to record (each = 1 clip PER ENV). Default None = ALL batches "
                         "(num clips = num episodes); 0 disables video.")
    ap.add_argument("--scene-group-size", type=int, default=0,
                    help="rebuild the object geometry every K batches (0=fixed nominal). Needs a "
                         "subprocess server with full DR ranges (serl_sim_server --subprocess --dr food_shape).")
    args = ap.parse_args()

    cc = yaml.safe_load(args.collect_config.read_text())
    exp_name = args.experiment or cc["experiment"]
    exp = Experiment.load(exp_name)
    sc = cc.get("scripted", {})
    params = dict(lift_height=sc.get("lift_height", 0.2), hold_seconds=sc.get("hold_seconds", 2.0),
                  approach_height=sc.get("approach_height", 0.12), grasp_z=sc.get("grasp_z", 0.006),
                  grasp_gw=sc.get("grasp_gw", 0.030), grasp_firm_steps=sc.get("grasp_firm_steps", 1),
                  gripper_close=cc.get("gripper_value", 0.5), speed_cap=cc.get("speed", 0.5))

    client = SimEnvClient(port=args.port)
    venv = SimEvalVenv(client, args.num_envs, args.max_steps)
    policy = ScriptedPolicy(args.num_envs, exp.action_config.scales, cc.get("rate", 30), params, venv=venv)
    spec = EvalSpec(n_episodes=args.n_episodes, num_envs=args.num_envs, seed=args.seed,
                    max_policy_steps=args.max_steps, scene_group_size=args.scene_group_size)

    # No training run to nest under (untrained expert) -> its own top-level dir named after the
    # "algorithm", mirroring logs/dppo, logs/serl (trained policies nest eval under <run>/eval/).
    out_dir = _REPO / "logs" / "scripted_policy" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_eval(venv, policy, spec, out_dir, experiment_name=exp_name, checkpoint=None,
             record_batches=args.record_batches, extra_meta={"algorithm": "scripted_policy"})
    venv.close()


if __name__ == "__main__":
    main()
