"""Resume a DPPO PPO-finetune run from a saved checkpoint's (itr, model) state,
without discarding progress just because a config field (e.g. render settings)
needs to change.

Why this exists: DPPO's PPO finetune agent (third_party/dppo/agent/finetune/
train_agent.py) DEFINES a `load(itr)` method that restores `self.itr` and the model
state dict from `<checkpoint_dir>/state_<itr>.pt`, but nothing in DPPO's own CLI
(third_party/dppo/script/run.py) ever CALLS it -- there is no built-in resume flow.
`agent.finetune.train_agent.TrainAgent.save_model()` also only persists `itr` +
`model` (no optimizer/lr-scheduler state), so a resume is necessarily partial: the
POLICY WEIGHTS pick up exactly where they left off, but Adam momentum and the LR
schedule's internal step counters restart from their initial state. Acceptable
tradeoff for continuing training past a mid-run config change (e.g. video-recording
settings) rather than re-running from the original base BC checkpoint.

Usage (envs/dppo, mirrors gentle_manip.dppo.train's CLI exactly, plus one var):
    DPPO_RESUME_CKPT=/path/to/<old_run>/checkpoint/state_75.pt \\
    uv run --project envs/dppo python -m gentle_manip.dppo.resume_ppo \\
        --config-path <cfg_dir> --config-name ft_ppo_diffusion_pointnet_retry \\
        base_policy_path=<same base checkpoint as the original launch> wandb=null

This mints a NEW run dir/id (like any other launch) -- checkpoints/renders continue
under the new run, starting at the resumed itr, so the old run's files are untouched.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_RUN_PY = _REPO / "third_party" / "dppo" / "script" / "run.py"

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_RUN_PY.parent))

os.environ.setdefault("DPPO_LOG_DIR", str(_REPO / "logs" / "dppo"))
os.environ.setdefault("DPPO_DATA_DIR", str(_REPO / "dataset" / "dppo"))


def _eval_base(ckpt: str) -> str:
    p = Path(ckpt)
    return str(p.parent.parent if p.parent.name == "checkpoint" else p.parent)


_EXP_ID: str | None = None


def _exp_id() -> str:
    global _EXP_ID
    if _EXP_ID is None:
        from gentle_manip.utils.experiment_registry import new_id
        _EXP_ID = new_id()
    return _EXP_ID


def _patch_resume() -> None:
    """Monkeypatch TrainPPOImgDiffusionAgent.run to load a checkpoint first, iff
    DPPO_RESUME_CKPT is set. Patching `.run` (not `__init__`) means it fires AFTER
    the agent is fully constructed (checkpoint_dir etc. already set up), and we
    bypass self.checkpoint_dir entirely by loading from an explicit path -- the
    resumed run has its OWN fresh checkpoint_dir for everything saved from here on."""
    ckpt_path = os.environ.get("DPPO_RESUME_CKPT")
    if not ckpt_path:
        return
    import torch
    from agent.finetune.train_ppo_diffusion_img_agent import TrainPPOImgDiffusionAgent

    original_run = TrainPPOImgDiffusionAgent.run

    def run_with_resume(self):
        data = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.itr = data["itr"]
        self.model.load_state_dict(data["model"])
        print(f"[resume_ppo] loaded itr={self.itr} model weights from {ckpt_path}", flush=True)
        return original_run(self)

    TrainPPOImgDiffusionAgent.run = run_with_resume


def main() -> None:
    if not _RUN_PY.exists():
        raise FileNotFoundError(f"DPPO run.py not found at {_RUN_PY} — is the submodule initialised?")
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval_base", _eval_base, replace=True)
    OmegaConf.register_new_resolver("exp_id", _exp_id, replace=True)
    _patch_resume()
    runpy.run_path(str(_RUN_PY), run_name="__main__")


if __name__ == "__main__":
    main()
