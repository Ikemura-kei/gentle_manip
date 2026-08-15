"""Phase 7 driver: merge quality-gated rollout data (Phase 5) with VLM
category embeddings (Phase 6) and train the combined RLDG+VLM generalist --
the first time these two levers (validated separately earlier this session)
are combined, at 25-category scale.

Reads logs/fragile25_specialist/<category>.json for each TRAIN category and
includes it in the merge iff rollout_data_path is set (i.e. it passed
QUALITY_GATE and rollout collection succeeded) -- mirrors
merge_cross_category_demos.py's symlink-then-convert_demos.py pattern, but
sources paths from the specialist driver's own tracked results instead of a
glob over dataset/demos/single_lift_<cat>_rigid/ (which assumes the wrong
suffix for this soft/MPM campaign, and wouldn't find ROLLOUT-collector output
specifically vs. any raw demo dir for the category).

Usage:
    python -m gentle_manip.scripts.run_fragile25_merge_and_train
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gentle_manip.scripts.run_fragile25_specialist import RESULTS_DIR, DPPO_CFG_DIR  # noqa: E402
from gentle_manip.scripts.run_fragile25_all_specialists import TRAIN  # noqa: E402

MERGE_NAME = "single_lift_fragile25_generalist_pcd"

TRAIN_TEMPLATE = '''# [dppo-pretrain] Combined RLDG + VLM generalist -- fragile-food 25-category campaign.
# First time these two levers (validated SEPARATELY this session: RLDG rollout-
# distillation fixed held-in performance 39->62% etc; VLM conditioning enabled
# non-zero zero-shot generalization) are combined together, at 25-category scale.
defaults:
  - _self_
hydra:
  run:
    dir: ${{logdir}}
  callbacks:
    experiment_snapshot:
      _target_: gentle_manip.dppo.hydra_snapshot.ExperimentSnapshot
_target_: agent.pretrain.train_diffusion_agent.TrainDiffusionAgent

name: ${{env}}_pre_diffusion_pointnet_ta${{horizon_steps}}_td${{denoising_steps}}
logdir: ${{oc.env:DPPO_LOG_DIR}}/dppo-pretrain/${{env}}/${{exp_id:}}
train_dataset_path: ${{oc.env:DPPO_DATA_DIR}}/{merge_name}/train.npz
val_dataset_path: ${{oc.env:DPPO_DATA_DIR}}/{merge_name}/val.npz
experiment: single_lift_mushroom_soft_easy   # any train-category experiment -- obs schema only

seed: 42
device: cuda:0
env: {merge_name}
obs_dim: 8
action_dim: 7
category_embed_dim: 24
denoising_steps: 20
horizon_steps: 4
cond_steps: 2
pc_cond_steps: 1
n_points: 1024
visual_feature_dim: 256

wandb:
  entity: ${{oc.env:DPPO_WANDB_ENTITY,null}}
  project: gentle-manip-fragile25-generalist
  run: ${{exp_id:}}
  group: dppo-pretrain

train:
  n_epochs: 3000
  batch_size: 128
  learning_rate: 1e-4
  weight_decay: 1e-6
  lr_scheduler:
    first_cycle_steps: ${{train.n_epochs}}
    warmup_steps: 100
    min_lr: 1e-5
  epoch_start_ema: 10
  update_ema_freq: 5
  save_model_freq: 100
  val_freq: 10

model:
  _target_: model.diffusion.diffusion.DiffusionModel
  predict_epsilon: True
  denoised_clip_value: 1.0
  network:
    _target_: gentle_manip.dppo.pointnet_diffusion.PointNetDiffusionMLP
    action_dim: ${{action_dim}}
    horizon_steps: ${{horizon_steps}}
    cond_dim: ${{eval:'${{obs_dim}} * ${{cond_steps}}'}}
    pc_cond_steps: ${{pc_cond_steps}}
    visual_feature_dim: ${{visual_feature_dim}}
    category_embed_dim: ${{category_embed_dim}}
    time_dim: 16
    mlp_dims: [512, 512, 512]
    activation_type: ReLU
    residual_style: True
    pointnet:
      in_channels: 3
      use_layernorm: True
      final_norm: layernorm
  horizon_steps: ${{horizon_steps}}
  obs_dim: ${{obs_dim}}
  action_dim: ${{action_dim}}
  denoising_steps: ${{denoising_steps}}
  device: ${{device}}

ema:
  decay: 0.995

train_dataset:
  # gentle_manip 25-category speed pass (2026-08-14): MUST be the
  # ...Category subclass, not the plain point-cloud dataset -- the plain
  # class never sets conditions["category_embed"], which the model
  # (category_embed_dim=24, PointNetDiffusionMLP) unconditionally reads in
  # its forward pass. Caught via a real crash: 5 identical KeyError retries
  # (train_with_resume kept relaunching a deterministic failure, since this
  # isn't a transient crash) before tracing it to this line.
  _target_: gentle_manip.dppo.pointcloud_dataset.StitchedSequencePointCloudCategoryDataset
  dataset_path: ${{train_dataset_path}}
  horizon_steps: ${{horizon_steps}}
  cond_steps: ${{cond_steps}}
  pc_cond_steps: ${{pc_cond_steps}}
  device: ${{device}}

val_dataset:
  _target_: gentle_manip.dppo.pointcloud_dataset.StitchedSequencePointCloudCategoryDataset
  dataset_path: ${{val_dataset_path}}
  horizon_steps: ${{horizon_steps}}
  cond_steps: ${{cond_steps}}
  pc_cond_steps: ${{pc_cond_steps}}
  device: ${{device}}
'''


def qualified_categories() -> dict:
    """{category: rollout_data_path} for every TRAIN category that passed the
    quality gate and has a rollout dataset -- the merge input."""
    out = {}
    for cat in TRAIN:
        p = RESULTS_DIR / f"{cat}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if r.get("rollout_data_path"):
            out[cat] = r["rollout_data_path"]
    return out


def build_merge(out_dir: Path) -> dict:
    cats = qualified_categories()
    if len(cats) < 2:
        raise RuntimeError(f"only {len(cats)} qualified categories so far ({list(cats)}) -- "
                           f"need at least 2 for a meaningful merge")
    link_dir = REPO / "dataset" / "demos_merged_fragile25_TEMP"
    if link_dir.exists():
        for f in link_dir.iterdir():
            f.unlink()
        link_dir.rmdir()
    link_dir.mkdir(parents=True)
    for cat, src in cats.items():
        (link_dir / f"{cat}.pkl").symlink_to(src)

    cmd = ["uv", "run", "--project", "envs/dppo", "python", "-m", "gentle_manip.dppo.convert_demos",
          str(link_dir), "--out", str(out_dir), "--point-cloud",
          "--experiment", "single_lift_mushroom_soft_easy", "--view", "student",
          "--val-split", "0.1", "--category-embed", "--embed-source", "vlm"]
    print(f"[merge_and_train] merging {len(cats)} categories: {sorted(cats)}", flush=True)
    print(f"[merge_and_train] $ {' '.join(cmd)}", flush=True)
    sub_env = __import__("os").environ.copy()
    sub_env.pop("PYTHONPATH", None)
    r = subprocess.run(cmd, cwd=str(REPO), env=sub_env)
    if r.returncode != 0:
        raise RuntimeError("convert_demos.py (merge) failed")
    return cats


def main() -> None:
    import os
    from gentle_manip.scripts.train_with_resume import train_with_resume

    out_dir = Path(os.environ.get("DPPO_DATA_DIR", str(REPO / "dppo_data"))) / MERGE_NAME
    cats = build_merge(out_dir)

    cfg_dir = DPPO_CFG_DIR / MERGE_NAME
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "pre_diffusion_pointnet.yaml").write_text(
        TRAIN_TEMPLATE.format(merge_name=MERGE_NAME))

    # CORRECTED (2026-08-14): an earlier fix this session assumed
    # hydra_snapshot.py registers `config.get("env_name", exp_name)` and
    # that TRAIN_TEMPLATE's `env:` field (MERGE_NAME) would be picked up --
    # WRONG: the registration code looks up a key literally named
    # "env_name", which no config here ever sets (they all use `env:`), so
    # the lookup ALWAYS misses and falls back to `exp_name` (this file's
    # TRAIN_TEMPLATE `experiment:` field, hardcoded to
    # "single_lift_mushroom_soft_easy" -- just the obs-schema reference,
    # unrelated to MERGE_NAME). Confirmed by re-deriving from raspberry's
    # actual experiments.csv row this session: task registered = the
    # `experiment:` field's literal value, not `env:`. Passing MERGE_NAME
    # here would have made find_run_dir_for_task() never match, same
    # silent-crash-recovery-defeat bug, for the single most expensive run
    # in the campaign.
    result = train_with_resume(
        config_path=str(cfg_dir), config_name="pre_diffusion_pointnet",
        task="single_lift_mushroom_soft_easy", max_retries=5, timeout_s=21600,
        log_path=RESULTS_DIR / "generalist_train.log")

    manifest = {"categories": sorted(cats), "n_categories": len(cats),
               "dppo_data_dir": str(out_dir), "cfg_dir": str(cfg_dir), **result}
    (RESULTS_DIR / "generalist.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
