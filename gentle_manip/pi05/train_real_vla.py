"""Full fine-tune of pi05 on the REAL 7-object teleop data, wrist slot MASKED OFF.

    uv run python <repo>/gentle_manip/pi05/train_real_vla.py \
        --repo-id gm/real7_ext --exp-name pi05_real7_ext [--steps 30000]

WHY A WRAPPER (same reason as train_lora.py). Two things cannot be expressed through openpi's CLI:

1. **The left-wrist mask.** The real rig has ONE external camera. `LiberoInputs` hardcodes
   `image_mask["left_wrist_0_rgb"] = True`, so a zero-filled wrist slot is fed to the model as a
   VALID black frame. `masked_wrist.patch()` swaps in a subclass that zeros it AND masks it off --
   exactly openpi's own treatment of the absent `right_wrist_0_rgb`. Patched BEFORE the train
   config is built, because `LeRobotLiberoDataConfig.create()` resolves `LiberoInputs` as a module
   attribute at call time. **The same patch must run at inference** (eval + deploy) or the model
   sees a different mask than it trained with.
2. Nothing else -- the model itself is stock pi05, FULL fine-tuning. LoRA is deliberately NOT used:
   in sim at 50 demos LoRA gave 0.000 success against full FT's 0.225, with train loss converging
   to 0.0008 either way (a converged LoRA loss is not evidence the adapter learned the task).

NO OPENPI FILE IS EDITED.
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
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-interval", type=int, default=2_000)
    ap.add_argument("--no-mask-wrist", action="store_true",
                    help="DEBUG only: keep openpi's hardcoded left-wrist mask=True (a black frame "
                         "fed as a valid view). Reproduces the sim ext-only setup; not deployable.")
    ap.add_argument("--overwrite", action="store_true", default=True)
    a = ap.parse_args()

    _REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_REPO / "third_party" / "openpi" / "scripts"))

    if not a.no_mask_wrist:
        from gentle_manip.pi05 import masked_wrist
        masked_wrist.patch()                  # MUST precede config construction

    import train as openpi_train              # THEIR training loop, unmodified
    from openpi.training import config as _config

    cfg = _config.get_config(a.base_config)
    # NAME DELIBERATELY UNCHANGED: assets_dirs = assets_base_dir / config.name, so renaming the
    # config sends norm-stats lookup to a directory that does not exist ("Normalization stats not
    # found"). The run is distinguished by exp_name, not by config.name.
    cfg = dataclasses.replace(
        cfg,
        exp_name=a.exp_name,
        batch_size=a.batch_size,
        num_train_steps=a.steps,
        save_interval=a.save_interval,
        overwrite=a.overwrite,
        data=dataclasses.replace(cfg.data, repo_id=a.repo_id),
    )
    print(f"[real-vla] base={a.base_config} repo_id={a.repo_id} exp={a.exp_name}")
    print(f"[real-vla] steps={cfg.num_train_steps} batch={cfg.batch_size} "
          f"save_every={cfg.save_interval} mask_wrist={not a.no_mask_wrist}")
    openpi_train.main(cfg)


if __name__ == "__main__":
    main()
