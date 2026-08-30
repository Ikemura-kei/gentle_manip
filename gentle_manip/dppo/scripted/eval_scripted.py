"""Canonical-harness eval agent for the VISION-ONLY SCRIPTED baseline (no learning).

Reuses DPPO's EvalAgent purely for venv construction, then drives it with ScriptedTopDownPolicy.
Same EvalSpec / venv / metrics / dumps as every learned arm, so the comparison is apples-to-apples.
"""
import numpy as np

from agent.eval.eval_agent import EvalAgent
from gentle_manip.evaluation import EvalSpec, run_eval
from gentle_manip.experiment import Experiment
from gentle_manip.dppo.scripted.scripted_topdown import ScriptedTopDownPolicy


class NoModel:
    """cfg.model placeholder — EvalAgent instantiates cfg.model, but a scripted policy has none."""
    def __init__(self, **kw):
        pass


class ScriptedEvalAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        # The ActionPipeline semantics come from the EXPERIMENT, exactly as the sim server's do —
        # so the encoding here cannot silently disagree with how the backend decodes.
        exp = Experiment.load(cfg.experiment)
        self.action_config = exp.action_config
        nz = np.load(cfg.normalization_path)
        self.action_min, self.action_max = nz["action_min"], nz["action_max"]
        self.obs_min, self.obs_max = nz["obs_min"], nz["obs_max"]
        print(f"[scripted] action mode={self.action_config.mode} "
              f"rot={self.action_config.rot_repr} "
              f"gripper_delta={getattr(self.action_config,'gripper_delta',False)} "
              f"| action_dim={len(self.action_min)}", flush=True)

    def run(self):
        spec = EvalSpec(
            n_episodes=int(self.cfg.get("n_episodes", 100)),
            num_envs=self.n_envs,
            seed=int(self.cfg.get("seed", 0)),
            max_policy_steps=int(self.cfg.env.max_episode_steps) // self.act_steps,
            scene_group_size=int(self.cfg.get("scene_group_size", 0)),
        )
        policy = ScriptedTopDownPolicy(
            self.action_config, self.action_min, self.action_max,
            act_steps=self.act_steps, n_envs=self.n_envs,
            obs_min=self.obs_min, obs_max=self.obs_max)
        run_eval(self.venv, policy, spec, self.logdir,
                 experiment_name=self.cfg.get("experiment"),
                 checkpoint="SCRIPTED-vision-only-topdown",
                 record_batches=self.cfg.get("record_batches", None))
