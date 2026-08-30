"""Compute openpi normalization stats for OUR LeRobot dataset, without editing openpi.

WHY THIS EXISTS. `openpi/scripts/compute_norm_stats.py` takes only a registered config NAME
(`main(config_name, max_frames)`) and resolves `repo_id` from it -- unlike `scripts/train.py`,
which uses `tyro.extras.overridable_config_cli` and so DOES accept `--data.repo-id`. That single
asymmetry is the one place the "config + CLI only" path breaks. Rather than edit their script, we
override `repo_id` on the dataclass here and reuse THEIR dataloader construction and THEIR stats
code verbatim (imported, not reimplemented -- the numbers must be the ones openpi would compute).

Usage (from third_party/openpi, with PYTHONPATH=<repo>):
    uv run python <repo>/gentle_manip/pi05/compute_norm_stats.py \
        --config-name pi05_libero --repo-id gm/mushroom_pi05_ext_wrist
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path


def _load_their_script():
    """Import openpi's compute_norm_stats.py, reusing its helpers as-is.

    Imported by NAME off sys.path rather than via importlib.spec_from_file_location: their
    module defines `RemoveStrings`, which is part of the dataset transform and therefore gets
    PICKLED to torch DataLoader workers. A module loaded under a synthetic name is not
    importable in a spawned worker, so pickling fails with
    `Can't pickle <class 'openpi_compute_norm_stats.RemoveStrings'>`. Importing it under its
    real module name from its real directory keeps it re-importable. (We also set
    num_workers=0 below, which avoids the spawn entirely -- belt and braces.)
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "third_party" / "openpi" / "scripts"
        if (cand / "compute_norm_stats.py").exists():
            sys.path.insert(0, str(cand))
            import compute_norm_stats as mod      # noqa: PLC0415
            return mod
    raise FileNotFoundError("could not locate third_party/openpi/scripts/compute_norm_stats.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    import numpy as np
    import tqdm
    import openpi.shared.normalize as normalize
    import openpi.training.config as _config

    theirs = _load_their_script()

    # The ONLY deviation from their script: point the data config at our repo_id.
    config = _config.get_config(args.config_name)
    # num_workers=0: this is a single pass over a small dataset, and it removes the DataLoader
    # worker spawn (and with it every cross-process pickling hazard) from a one-off stats job.
    config = dataclasses.replace(config, num_workers=0,
                                 data=dataclasses.replace(config.data, repo_id=args.repo_id))
    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.repo_id == args.repo_id, "repo_id override did not take"

    # Everything below mirrors their main() and calls THEIR helpers.
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = theirs.create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, args.max_frames)
    else:
        data_loader, num_batches = theirs.create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model,
            config.num_workers, args.max_frames)

    keys = ["state", "actions"]
    stats = {k: normalize.RunningStats() for k in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for k in keys:
            stats[k].update(np.asarray(batch[k]))

    norm_stats = {k: s.get_statistics() for k, s in stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    for k, v in norm_stats.items():
        print(f"  {k}: mean[:4]={np.asarray(v.mean)[:4]}  std[:4]={np.asarray(v.std)[:4]}")
    print("NORM STATS OK")


if __name__ == "__main__":
    main()
