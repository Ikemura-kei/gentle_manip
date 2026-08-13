"""Phase 4 driver for the fragile-food 25-category campaign (2026-08-13):
given a category with a completed demo collection, runs the full
convert -> DPPO config generation -> crash-recoverable train -> canonical
eval pipeline that was done by hand for mushroom-soft-easy earlier this
session, automated for the remaining ~19 train categories.

Usage (run in envs/dppo for convert/train, but this script shells out to the
right env per step so it can be invoked from either):
    python -m gentle_manip.scripts.run_fragile25_specialist --category tofu

Writes results to logs/fragile25_specialist/<category>.json:
    {"category", "demo_dir", "dppo_data_dir", "run_dir", "checkpoint",
     "eval_success_rate", "eval_dir"}

Each step is skip-if-already-done (checks for the expected output before
re-running), so re-invoking this script for a category that already trained
just re-runs whatever's missing (e.g. only eval, if training finished but
eval didn't) -- the same idempotent-resume spirit as the rest of this
campaign's infra.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULTS_DIR = REPO / "logs" / "fragile25_specialist"
DPPO_CFG_DIR = REPO / "gentle_manip" / "dppo" / "cfg"

PRE_TEMPLATE = '''# [dppo-pretrain] BC-pretrain a PointNet diffusion policy on SOFT {obj} lift demos.
# Used by: third_party/dppo/script/run.py (TrainDiffusionAgent) via gentle_manip.dppo.train
# Status: experimental (fragile-food 25-category campaign, 2026-08-13)
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
train_dataset_path: ${{oc.env:DPPO_DATA_DIR}}/${{env}}/train.npz
val_dataset_path: ${{oc.env:DPPO_DATA_DIR}}/${{env}}/val.npz
experiment: single_lift_{obj}_soft_easy

seed: 42
device: cuda:0
env: single_lift_{obj}_soft_easy_pcd
obs_dim: 8
action_dim: 7
denoising_steps: 20
horizon_steps: 4
cond_steps: 2
pc_cond_steps: 1
n_points: 1024
visual_feature_dim: 256

wandb:
  entity: ${{oc.env:DPPO_WANDB_ENTITY,null}}
  project: gentle-manip-${{env}}
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
  _target_: gentle_manip.dppo.pointcloud_dataset.StitchedSequencePointCloudDataset
  dataset_path: ${{train_dataset_path}}
  horizon_steps: ${{horizon_steps}}
  cond_steps: ${{cond_steps}}
  pc_cond_steps: ${{pc_cond_steps}}
  device: ${{device}}

val_dataset:
  _target_: gentle_manip.dppo.pointcloud_dataset.StitchedSequencePointCloudDataset
  dataset_path: ${{val_dataset_path}}
  horizon_steps: ${{horizon_steps}}
  cond_steps: ${{cond_steps}}
  pc_cond_steps: ${{pc_cond_steps}}
  device: ${{device}}
'''

EVAL_TEMPLATE = '''# [dppo-eval] Canonical eval of the {obj} solo specialist (fragile-food 25-category campaign).
defaults:
  - _self_
hydra:
  run:
    dir: ${{logdir}}
_target_: gentle_manip.dppo.eval_agent.EvalHarnessAgent

name: ${{env_name}}_eval_diffusion_pointnet_ta${{horizon_steps}}_td${{denoising_steps}}
logdir: ${{eval_base:${{base_policy_path}}}}/eval/${{now:%Y-%m-%d}}_${{now:%H-%M-%S}}
base_policy_path: ???
normalization_path: ${{oc.env:DPPO_DATA_DIR}}/single_lift_{obj}_soft_easy_pcd/normalization.npz
experiment: single_lift_{obj}_soft_easy

seed: 42
device: cuda:0
env_name: single_lift_{obj}_soft_easy_pcd
obs_dim: 8
action_dim: 7
denoising_steps: 20
ft_denoising_steps: 0
cond_steps: 2
pc_cond_steps: 1
n_points: 1024
visual_feature_dim: 256
horizon_steps: 4
act_steps: 4
use_ddim: False
ddim_steps: ${{ft_denoising_steps}}

n_episodes: 100
scene_group_size: 4
record_batches: null
n_steps: 75
render_num: 5

env:
  n_envs: 5
  name: ${{env_name}}
  env_type: genesis
  max_episode_steps: 300
  reset_at_iteration: False
  save_video: True
  use_image_obs: True
  best_reward_threshold_for_success: 0.2
  specific:
    obs_steps: ${{cond_steps}}
    act_steps: ${{act_steps}}
    normalization_path: ${{normalization_path}}
    port: 5570
    obs_keys: [ee_pos, ee_quat, gripper_width]
    pointcloud_key: point_cloud

shape_meta:
  obs:
    state:
      shape: [8]
    point_cloud:
      shape: [1024, 3]
  action:
    shape: [7]

wandb: null

model:
  _target_: model.diffusion.diffusion_eval.DiffusionEval
  ft_denoising_steps: ${{ft_denoising_steps}}
  predict_epsilon: True
  denoised_clip_value: 1.0
  randn_clip_value: 3
  use_ddim: ${{use_ddim}}
  ddim_steps: ${{ddim_steps}}
  network_path: ${{base_policy_path}}
  network:
    _target_: gentle_manip.dppo.pointnet_diffusion.PointNetDiffusionMLP
    action_dim: ${{action_dim}}
    horizon_steps: ${{horizon_steps}}
    cond_dim: ${{eval:'${{obs_dim}} * ${{cond_steps}}'}}
    pc_cond_steps: ${{pc_cond_steps}}
    visual_feature_dim: ${{visual_feature_dim}}
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
'''


def _run(cmd, cwd=REPO, env=None):
    print(f"[run_specialist] $ {' '.join(str(c) for c in cmd)}", flush=True)
    sub_env = os.environ.copy()
    sub_env.pop("PYTHONPATH", None)
    if env:
        sub_env.update(env)
    return subprocess.run(cmd, cwd=str(cwd), env=sub_env)


def find_latest_demo_dir(category: str) -> Optional[Path]:
    task_dir = REPO / "dataset" / "demos" / f"single_lift_{category}_soft"
    if not task_dir.exists():
        return None
    best, best_n = None, -1
    for d in task_dir.iterdir():
        if not d.is_dir():
            continue
        dp = d / "data.pkl"
        if not dp.exists():
            continue
        import pickle
        try:
            with open(dp, "rb") as f:
                n = len(pickle.load(f)["episodes"])
        except Exception:
            continue
        if n > best_n:
            best, best_n = d, n
    return best


def convert(category: str, demo_dir: Path) -> Path:
    out_dir = Path(os.environ.get("DPPO_DATA_DIR", str(REPO / "dppo_data"))) / f"single_lift_{category}_soft_easy_pcd"
    if (out_dir / "train.npz").exists():
        print(f"[run_specialist] {category}: DPPO dataset exists, skipping convert", flush=True)
        return out_dir
    cmd = ["uv", "run", "--project", "envs/dppo", "python", "-m", "gentle_manip.dppo.convert_demos",
          str(demo_dir / "data.pkl"), "--out", str(out_dir), "--point-cloud",
          "--experiment", f"single_lift_{category}_soft_easy", "--view", "student", "--val-split", "0.1"]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"convert_demos failed for {category}")
    return out_dir


def write_configs(category: str) -> Path:
    cfg_dir = DPPO_CFG_DIR / f"single_lift_{category}_soft_easy_pcd"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "pre_diffusion_pointnet.yaml").write_text(PRE_TEMPLATE.format(obj=category))
    (cfg_dir / "eval_diffusion_pointnet.yaml").write_text(EVAL_TEMPLATE.format(obj=category))
    return cfg_dir


def train(category: str, cfg_dir: Path) -> dict:
    from gentle_manip.scripts.train_with_resume import train_with_resume
    result = train_with_resume(
        config_path=str(cfg_dir), config_name="pre_diffusion_pointnet",
        task=f"single_lift_{category}_soft_easy", max_retries=5, timeout_s=14400,
        log_path=RESULTS_DIR / "train_logs" / f"{category}.log")
    return result


def best_checkpoint(run_dir: Path, category: str) -> Optional[Path]:
    from gentle_manip.scripts.train_with_resume import find_best_checkpoint
    # train_with_resume's own log (not run_dir/run.log -- our launcher appends
    # every attempt, including pre-resume ones, to ONE shared per-category log).
    log_path = RESULTS_DIR / "train_logs" / f"{category}.log"
    return find_best_checkpoint(run_dir, log_path if log_path.exists() else None)


def eval_specialist(category: str, cfg_dir: Path, checkpoint: Path, port: int = 5570) -> dict:
    # Launch sim server, run eval, tear down -- mirrors this session's manual pattern.
    server_log = RESULTS_DIR / "eval_logs" / f"{category}_server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    sub_env = os.environ.copy()
    sub_env.pop("PYTHONPATH", None)
    server_cmd = ["uv", "run", "--project", "envs/sim", "python", "-m",
                 "gentle_manip.scripts.serl_sim_server",
                 "--experiment", f"single_lift_{category}_soft_easy", "--view", "student",
                 "--num-envs", "5", "--render-rgb", "--subprocess", "--port", str(port)]
    print(f"[run_specialist] {category}: starting sim server...", flush=True)
    with open(server_log, "w") as logf:
        server_proc = subprocess.Popen(server_cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
                                       env=sub_env, start_new_session=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 120:
            if server_log.exists() and "SIM_SERVER_READY" in server_log.read_text(errors="ignore"):
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"sim server for {category} did not become ready in 120s")

        eval_log = RESULTS_DIR / "eval_logs" / f"{category}.log"
        eval_cmd = ["uv", "run", "--project", "envs/dppo", "python", "-m", "gentle_manip.dppo.train",
                   "--config-name", "eval_diffusion_pointnet", "--config-path", str(cfg_dir),
                   f"base_policy_path={checkpoint}"]
        with open(eval_log, "w") as logf:
            r = subprocess.run(eval_cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, env=sub_env)
        text = eval_log.read_text(errors="ignore")
        m = re.search(r"DONE — success ([\d.]+)", text)
        success_rate = float(m.group(1)) if m else None
        return {"success_rate": success_rate, "ok": r.returncode == 0 and success_rate is not None,
               "eval_log": str(eval_log)}
    finally:
        try:
            pgid = os.getpgid(server_proc.pid)
            os.killpg(pgid, 9)
        except ProcessLookupError:
            pass


def run_one(category: str) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{category}.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"category": category}

    if "demo_dir" not in result or not Path(result.get("demo_dir", "")).exists():
        demo_dir = find_latest_demo_dir(category)
        if demo_dir is None:
            result["status"] = "no_demos"
            result_path.write_text(json.dumps(result, indent=2))
            return result
        result["demo_dir"] = str(demo_dir)
        result_path.write_text(json.dumps(result, indent=2))

    if "dppo_data_dir" not in result:
        dppo_dir = convert(category, Path(result["demo_dir"]))
        result["dppo_data_dir"] = str(dppo_dir)
        result_path.write_text(json.dumps(result, indent=2))

    cfg_dir = write_configs(category)

    if "run_dir" not in result or not result.get("train_ok"):
        train_result = train(category, cfg_dir)
        result["run_dir"] = train_result["run_dir"]
        result["train_ok"] = train_result["ok"]
        result_path.write_text(json.dumps(result, indent=2))

    if result.get("run_dir") and "checkpoint" not in result:
        ckpt = best_checkpoint(Path(result["run_dir"]), category)
        result["checkpoint"] = str(ckpt) if ckpt else None
        result_path.write_text(json.dumps(result, indent=2))

    if result.get("checkpoint") and "eval_success_rate" not in result:
        eval_result = eval_specialist(category, cfg_dir, Path(result["checkpoint"]))
        result["eval_success_rate"] = eval_result["success_rate"]
        result["eval_ok"] = eval_result["ok"]
        result["eval_log"] = eval_result["eval_log"]
        result_path.write_text(json.dumps(result, indent=2))

    result["status"] = "done"
    result_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", required=True)
    args = ap.parse_args()
    result = run_one(args.category)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
