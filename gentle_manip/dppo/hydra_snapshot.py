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
            from gentle_manip.utils.run_paths import save_launch_command, snapshot_experiment

            run_dir = Path(HydraConfig.get().runtime.output_dir)
            snapshot_experiment(Experiment.load(exp_name), run_dir)      # -> <run>/config/
            save_launch_command(run_dir)                                 # -> <run>/launch_command.sh
            print(f"[snapshot] env config for '{exp_name}' -> {run_dir}/config/ (+ launch_command.sh)", flush=True)
        except Exception as e:                                          # never fail a run over this
            print(f"[snapshot] env cfg snapshot skipped: {e}", flush=True)

        # Register the run in the global experiment table (TRAINING only — the logdir leaf is the
        # ID minted by the exp_id resolver). Only fires for configs carrying this callback
        # (pretrain/finetune), never eval. Best-effort: never fail a run over bookkeeping.
        try:
            from hydra.core.hydra_config import HydraConfig
            from gentle_manip.utils.experiment_registry import _looks_like_id, add_entry

            run_dir = Path(HydraConfig.get().runtime.output_dir)
            exp_id = run_dir.name
            if _looks_like_id(exp_id):
                # algo from the log path (…/dppo-finetune/<task>/<id> -> "dppo-finetune")
                parts = run_dir.parts
                algo = parts[-3] if len(parts) >= 3 else "dppo"
                task = str(config.get("env_name", exp_name))
                add_entry(exp_id, algo, task, exp_name, run_dir)
                print(f"[registry] {exp_id}  {algo}  {task} -> experiments.csv", flush=True)
        except Exception as e:
            print(f"[registry] entry skipped: {e}", flush=True)
