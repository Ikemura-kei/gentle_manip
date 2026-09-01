"""LoRA fine-tune of π0.5 on our LeRobot dataset, WITHOUT modifying openpi.

WHY THIS FILE EXISTS. LoRA cannot be requested through openpi's CLI. `scripts/train.py` resolves
a registered TrainConfig by name and applies `--flag` overrides, but LoRA needs BOTH
  model.paligemma_variant / action_expert_variant = "*_lora"      (a string -- CLI could set it)
  freeze_filter = Pi0Config(...).get_freeze_filter()              (an nnx Filter OBJECT -- CLI cannot)
Setting only the variants would add LoRA adapters and freeze NOTHING: full-parameter training with
extra adapters, i.e. the opposite of the point. So we build the TrainConfig HERE, from THEIR
classes, and hand it to THEIR `main()`. openpi's tree stays untouched -- the same pattern as
gentle_manip/pi05/compute_norm_stats.py.

WHY LoRA FOR THIS EXPERIMENT. 50 demos (~11k frames) is a low-data regime where a full fine-tune
of a 3B-parameter VLA overfits: openpi's own small-dataset examples run 20k steps at batch 64,
which is ~116 epochs over a set this size. LoRA is the standard mitigation and is what openpi ships
for exactly this case (`pi0_libero_low_mem_finetune`). Run it as the COMPARISON to the full
fine-tune, not as a replacement -- the pair is the informative result.

Usage (from third_party/openpi, PYTHONPATH=<repo>):
    uv run python <repo>/gentle_manip/pi05/train_lora.py \
        --repo-id gm/lowdata50_ext_wrist --exp-name pi05_lora_ext_wrist [--steps 20000]
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--base-config", default="pi05_libero")
    ap.add_argument("--steps", type=int, default=20_000)   # openpi's custom-dataset recipe
    ap.add_argument("--batch-size", type=int, default=64)  # ditto
    ap.add_argument("--save-interval", type=int, default=2_000)
    ap.add_argument("--overwrite", action="store_true", default=True)
    a = ap.parse_args()

    _REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_REPO / "third_party" / "openpi" / "scripts"))
    import train as openpi_train                     # THEIR training loop, unmodified
    from openpi.models import pi0_config
    from openpi.training import config as _config

    cfg = _config.get_config(a.base_config)
    lora_model = dataclasses.replace(
        cfg.model, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    cfg = dataclasses.replace(
        cfg,
        # NAME IS DELIBERATELY UNCHANGED. `assets_dirs` is derived from config.name, so renaming
        # to "<base>_lora" sends the norm-stats lookup to assets/<base>_lora/<repo_id>/ -- which
        # does not exist, because compute_norm_stats wrote them under assets/<base>/. That fails
        # as "Normalization stats not found", which reads like a missing prerequisite rather than
        # a renamed directory. The run is still distinguishable: exp_name (and hence the
        # checkpoint path) carries the `lora` label.
        model=lora_model,
        # The freeze filter MUST match the model config above -- openpi's own comment. Derive it
        # from the same object rather than hand-rolling a regex.
        freeze_filter=lora_model.get_freeze_filter(),
        ema_decay=None,                    # openpi turns EMA OFF for LoRA finetuning
        data=dataclasses.replace(cfg.data, repo_id=a.repo_id),
        exp_name=a.exp_name,
        batch_size=a.batch_size,
        num_train_steps=a.steps,
        save_interval=a.save_interval,
        overwrite=a.overwrite,
    )
    print(f"[lora] base={a.base_config} repo_id={a.repo_id} exp={a.exp_name}")
    print(f"[lora] paligemma={lora_model.paligemma_variant} "
          f"action_expert={lora_model.action_expert_variant} ema={cfg.ema_decay}")
    print(f"[lora] steps={cfg.num_train_steps} batch={cfg.batch_size} save_every={cfg.save_interval}")
    openpi_train.main(cfg)


if __name__ == "__main__":
    main()
