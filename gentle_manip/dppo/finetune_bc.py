"""BC finetune entry: pretrain-agent training WARM-STARTED from an existing checkpoint.

The DPPO pretrain agent has no init-from-checkpoint hook (its ``load`` reads only its own
run dir), so this thin entry mirrors ``script/run.py``'s hydra flow and injects the base
weights (model + EMA) after instantiation, before ``agent.run()``.

Usage (sim-pretrained -> real finetune):
    uv run --project envs/dppo python -m gentle_manip.dppo.finetune_bc \
        --config-path <cfg dir> --config-name pre_diffusion_pointnet \
        env=<finetune dataset env> +base_ckpt=<path/to/state_N.pt> \
        train.learning_rate=1e-5 ...

Notes:
  - the finetune DATASET must be normalized with the BASE run's normalization stats
    (actions/obs renormalized), or the warm-started policy sees a scaled world;
  - base arch overrides (mlp_dims etc.) must match the base checkpoint.
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_RUN_PY = _REPO / "third_party" / "dppo" / "script" / "run.py"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_RUN_PY.parent))
os.environ.setdefault("DPPO_LOG_DIR", str(_REPO / "logs" / "dppo"))
os.environ.setdefault("DPPO_DATA_DIR", str(_REPO / "dataset" / "dppo"))

import hydra                      # noqa: E402
from omegaconf import OmegaConf   # noqa: E402

from gentle_manip.dppo.train import _eval_base, _exp_id  # noqa: E402


@hydra.main(version_base=None, config_path=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    import torch
    base = cfg.get("base_ckpt")
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    if base:
        sd = torch.load(base, map_location=agent.device if hasattr(agent, "device") else "cpu",
                        weights_only=True)
        agent.model.load_state_dict(sd["model"])
        agent.ema_model.load_state_dict(sd["ema"])
        print(f"[finetune_bc] warm-started model+EMA from {base} (epoch {sd.get('epoch')})",
              flush=True)
    else:
        print("[finetune_bc] WARNING: no +base_ckpt given — training from scratch", flush=True)
    agent.run()


if __name__ == "__main__":
    import math
    OmegaConf.register_new_resolver("eval", eval, replace=True)          # as in dppo run.py
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    OmegaConf.register_new_resolver("eval_base", _eval_base, replace=True)
    OmegaConf.register_new_resolver("exp_id", _exp_id, replace=True)
    main()
