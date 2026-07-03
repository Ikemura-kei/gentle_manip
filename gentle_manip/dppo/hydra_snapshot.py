"""Hydra callback: snapshot the env (experiment) config into every DPPO training run dir.

Comment (1): DPPO's hydra `.hydra/config.yaml` records the DPPO hyperparameters, but NOT the
env definition (task + obs + action + dr + reward) — that lives in the experiment config the
sim server runs. This callback copies the referenced experiment config into `<run>/config/`
(separate from `.hydra/`), the same snapshot SERL writes, so every training run is
self-describing. Enabled by adding `experiment: <name>` to a DPPO config + registering this
callback under `hydra.callbacks`. Genesis-free (Experiment + snapshot_experiment are).
"""
from __future__ import annotations

from pathlib import Path

from hydra.experimental.callback import Callback


class ExperimentSnapshot(Callback):
    def on_job_start(self, config, **kwargs) -> None:
        exp_name = config.get("experiment", None)
        if not exp_name:
            return
        try:
            from hydra.core.hydra_config import HydraConfig
            from gentle_manip.experiment import Experiment
            from gentle_manip.utils.run_paths import snapshot_experiment

            run_dir = Path(HydraConfig.get().runtime.output_dir)
            snapshot_experiment(Experiment.load(exp_name), run_dir)      # -> <run>/config/
            print(f"[snapshot] env config for '{exp_name}' -> {run_dir}/config/", flush=True)
        except Exception as e:                                          # never fail a run over this
            print(f"[snapshot] env cfg snapshot skipped: {e}", flush=True)
