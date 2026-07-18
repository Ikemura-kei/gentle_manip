"""Launcher for DPPO pretrain / PPO-finetune on the genesis bridge.

Why a wrapper: DPPO's ``script/run.py`` is a hydra entry point, and ``@hydra.main``
``chdir``s into the run's ``logdir`` before the agent (and thus ``make_async``'s
``env_type="genesis"`` branch) imports ``gentle_manip``. gentle_manip is imported via
``sys.path`` — deliberately NOT a dependency of envs/dppo (that would pull ``gymnasium``
alongside DPPO's pinned ``gym==0.22``; see envs/dppo/pyproject.toml). A cwd-based import
therefore breaks after the chdir. Pinning the repo root onto ``sys.path`` here (before
hydra runs) survives the chdir, since hydra changes the cwd but not ``sys.path``.

Also sets DPPO_LOG_DIR / DPPO_DATA_DIR defaults (under the repo) so configs resolve
without the caller exporting them, then hands off to DPPO's run.py unchanged.

Usage (from repo root, envs/dppo):
    uv run --project envs/dppo --no-sync python -m gentle_manip.dppo.train \\
        --config-path $(pwd)/gentle_manip/dppo/cfg/single_lift_mushroom_rigid_state \\
        --config-name ft_ppo_diffusion_mlp base_policy_path=<ckpt.pt>
A serl_sim_server (same experiment+view, matching --num-envs) must be running for finetune.
"""
import os
import runpy
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_RUN_PY = _REPO / "third_party" / "dppo" / "script" / "run.py"

# gentle_manip importable after hydra's chdir; run.py's `import download_url` (script/ dir).
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_RUN_PY.parent))

os.environ.setdefault("DPPO_LOG_DIR", str(_REPO / "logs" / "dppo"))
os.environ.setdefault("DPPO_DATA_DIR", str(_REPO / "dataset" / "dppo"))
# wandb: the training configs carry a `wandb:` block (entity = ${oc.env:DPPO_WANDB_ENTITY,null}
# -> the logged-in default account when unset). Enabled by default; disable per-run with the
# hydra override `wandb=null`, or force local-only with `WANDB_MODE=offline` in the environment.


def _eval_base(ckpt: str) -> str:
    """OmegaConf resolver: a checkpoint's own training run dir, so an eval config can point
    hydra.run.dir at <run>/eval/<datetime> (ONE folder — no separate dppo-eval/ tree).
    <run>/checkpoint/state_X.pt -> <run>; otherwise the checkpoint's parent."""
    p = Path(ckpt)
    return str(p.parent.parent if p.parent.name == "checkpoint" else p.parent)


_EXP_ID: str | None = None


def _exp_id() -> str:
    """OmegaConf resolver `${exp_id:}` — a fresh 5-letter run ID, minted ONCE per process and
    cached, so a training config's logdir leaf IS the global experiment ID (short + unique).
    The ExperimentSnapshot hydra callback records it in experiments.csv. Training only —
    eval configs use eval_base instead, so they never mint an ID."""
    global _EXP_ID
    if _EXP_ID is None:
        from gentle_manip.utils.experiment_registry import new_id
        _EXP_ID = new_id()
    return _EXP_ID


def main() -> None:
    if not _RUN_PY.exists():
        raise FileNotFoundError(f"DPPO run.py not found at {_RUN_PY} — is the submodule initialised?")
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval_base", _eval_base, replace=True)
    OmegaConf.register_new_resolver("exp_id", _exp_id, replace=True)
    runpy.run_path(str(_RUN_PY), run_name="__main__")


if __name__ == "__main__":
    main()
