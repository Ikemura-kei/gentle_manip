"""Phase 8 driver: canonical eval of the combined RLDG+VLM generalist
(Phase 7) across ALL 25 categories -- 20 held-in + 5 zero-shot (blackberry,
scallop, watermelon, dumpling, gelatin). The actual deliverable: compare
against the 70%+ held-in / 50%+ zero-shot targets.

Usage:
    python -m gentle_manip.scripts.run_fragile25_final_eval

Writes logs/fragile25_specialist/final_eval_<category>.json per category and
a combined logs/fragile25_specialist/final_eval_summary.json at the end.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gentle_manip.scripts.run_fragile25_specialist import RESULTS_DIR, DPPO_CFG_DIR  # noqa: E402
from gentle_manip.scripts.run_fragile25_all_specialists import TRAIN  # noqa: E402
from gentle_manip.scripts.run_fragile25_merge_and_train import MERGE_NAME  # noqa: E402

TEST = ["blackberry", "scallop", "watermelon", "dumpling", "gelatin"]

EVAL_TEMPLATE = '''# [dppo-eval] Fragile-25 combined RLDG+VLM generalist -- {role} eval on {obj}.
defaults:
  - _self_
hydra:
  run:
    dir: ${{logdir}}
_target_: gentle_manip.dppo.eval_agent.EvalHarnessAgent

name: ${{env_name}}_eval_diffusion_pointnet_ta${{horizon_steps}}_td${{denoising_steps}}
logdir: ${{eval_base:${{base_policy_path}}}}/eval_{obj}/${{now:%Y-%m-%d}}_${{now:%H-%M-%S}}
base_policy_path: ???
normalization_path: ${{oc.env:DPPO_DATA_DIR}}/{merge_name}/normalization.npz
experiment: single_lift_{obj}_soft_easy

seed: 42
device: cuda:0
env_name: single_lift_{obj}_soft_easy_pcd
obs_dim: 8
action_dim: 7
category_embed_dim: 24
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
    port: {port}
    obs_keys: [ee_pos, ee_quat, gripper_width]
    pointcloud_key: point_cloud
    category: {obj}
    category_embed_source: vlm

shape_meta:
  obs:
    state:
      shape: [8]
    point_cloud:
      shape: [1024, 3]
    category_embed:
      shape: [24]
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
'''


def eval_one(category: str, role: str, checkpoint: str, port: int = 5570) -> dict:
    cfg_dir = DPPO_CFG_DIR / MERGE_NAME / f"eval_{category}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "eval_diffusion_pointnet.yaml").write_text(
        EVAL_TEMPLATE.format(obj=category, role=role, merge_name=MERGE_NAME, port=port))

    server_log = RESULTS_DIR / "final_eval_logs" / f"{category}_server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    sub_env = os.environ.copy()
    sub_env.pop("PYTHONPATH", None)
    server_cmd = ["uv", "run", "--project", "envs/sim", "python", "-m",
                 "gentle_manip.scripts.serl_sim_server",
                 "--experiment", f"single_lift_{category}_soft_easy", "--view", "student",
                 "--num-envs", "5", "--render-rgb", "--subprocess", "--port", str(port)]
    print(f"[final_eval] {category} ({role}): starting sim server...", flush=True)
    with open(server_log, "w") as logf:
        proc = subprocess.Popen(server_cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
                                env=sub_env, start_new_session=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 120:
            if server_log.exists() and "SIM_SERVER_READY" in server_log.read_text(errors="ignore"):
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"sim server for {category} not ready in 120s")

        eval_log = RESULTS_DIR / "final_eval_logs" / f"{category}.log"
        cmd = ["uv", "run", "--project", "envs/dppo", "python", "-m", "gentle_manip.dppo.train",
              "--config-name", "eval_diffusion_pointnet", "--config-path", str(cfg_dir),
              f"base_policy_path={checkpoint}"]
        with open(eval_log, "w") as logf:
            r = subprocess.run(cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, env=sub_env)
        text = eval_log.read_text(errors="ignore")
        m = re.search(r"DONE — success ([\d.]+)", text)
        sr = float(m.group(1)) if m else None
        return {"category": category, "role": role, "success_rate": sr,
               "ok": r.returncode == 0 and sr is not None, "eval_log": str(eval_log)}
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except ProcessLookupError:
            pass


def main() -> None:
    generalist = json.loads((RESULTS_DIR / "generalist.json").read_text())
    checkpoint = generalist.get("checkpoint")
    if not checkpoint:
        # find_best_checkpoint on the generalist's run_dir
        from gentle_manip.scripts.train_with_resume import find_best_checkpoint
        run_dir = Path(generalist["run_dir"])
        log_path = RESULTS_DIR / "generalist_train.log"
        ckpt = find_best_checkpoint(run_dir, log_path if log_path.exists() else None)
        checkpoint = str(ckpt)
        generalist["checkpoint"] = checkpoint
        (RESULTS_DIR / "generalist.json").write_text(json.dumps(generalist, indent=2))

    results = []
    trained_categories = set(generalist["categories"])
    for cat in TRAIN:
        if cat not in trained_categories:
            continue   # only eval categories actually IN the merge as "held-in"
        result_path = RESULTS_DIR / f"final_eval_{cat}.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text()))
            continue
        r = eval_one(cat, "held-in", checkpoint, port=5580)
        result_path.write_text(json.dumps(r, indent=2))
        results.append(r)
        print(f"[final_eval] {cat} (held-in): success_rate={r['success_rate']}", flush=True)

    for cat in TEST:
        result_path = RESULTS_DIR / f"final_eval_{cat}.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text()))
            continue
        r = eval_one(cat, "zero-shot", checkpoint, port=5580)
        result_path.write_text(json.dumps(r, indent=2))
        results.append(r)
        print(f"[final_eval] {cat} (zero-shot): success_rate={r['success_rate']}", flush=True)

    held_in = [r["success_rate"] for r in results if r["role"] == "held-in" and r["success_rate"] is not None]
    zero_shot = [r["success_rate"] for r in results if r["role"] == "zero-shot" and r["success_rate"] is not None]
    summary = {
        "checkpoint": checkpoint,
        "held_in_mean": sum(held_in) / len(held_in) if held_in else None,
        "zero_shot_mean": sum(zero_shot) / len(zero_shot) if zero_shot else None,
        "held_in_target_met": (sum(held_in) / len(held_in) >= 0.70) if held_in else None,
        "zero_shot_target_met": (sum(zero_shot) / len(zero_shot) >= 0.50) if zero_shot else None,
        "results": results,
    }
    (RESULTS_DIR / "final_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
