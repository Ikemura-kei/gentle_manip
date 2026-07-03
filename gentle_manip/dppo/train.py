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
os.environ.setdefault("DPPO_WANDB_ENTITY", "")


def main() -> None:
    if not _RUN_PY.exists():
        raise FileNotFoundError(f"DPPO run.py not found at {_RUN_PY} — is the submodule initialised?")
    runpy.run_path(str(_RUN_PY), run_name="__main__")


if __name__ == "__main__":
    main()
