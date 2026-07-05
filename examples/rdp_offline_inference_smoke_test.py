"""Offline inference smoke test for the RDP baseline (third_party/reactive_diffusion_policy).

Loads a trained checkpoint in a fresh process (the same mechanism
eval_real_robot_flexiv.py uses) and runs the policy's diffusion-sampling
predict_action on a real batch from the mini lift dataset, without any
ROS2/live-robot dependency. This is the "inference" half of the RDP baseline
integration smoke test described in envs/rdp/pyproject.toml; training is
train_dp.sh / train.py directly.

Run with:
    cd third_party/reactive_diffusion_policy && \
    ../../envs/rdp/.venv/bin/python ../../examples/rdp_offline_inference_smoke_test.py \
        --ckpt data/outputs/<run_dir>/checkpoints/latest.ckpt
"""
import argparse
import sys
import pathlib

RDP_ROOT = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "reactive_diffusion_policy"
sys.path.insert(0, str(RDP_ROOT))

import torch
import hydra
from omegaconf import OmegaConf
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to a .ckpt saved by train.py")
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.ckpt).resolve()
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=__import__("dill"))
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload)

    policy = workspace.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    # Real batch from the same mini dataset the checkpoint was trained on.
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    batch = dataset[0]
    obs_dict = {k: v.unsqueeze(0).to(device) for k, v in batch["obs"].items()}
    gt_action = batch["action"].unsqueeze(0).to(device)

    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    pred_action = result["action_pred"]

    mse = torch.nn.functional.mse_loss(pred_action, gt_action[:, : pred_action.shape[1]])
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"obs keys: {list(obs_dict.keys())}")
    print(f"gt_action shape: {tuple(gt_action.shape)}, pred_action shape: {tuple(pred_action.shape)}")
    print(f"action MSE (normalized space): {mse.item():.6f}")


if __name__ == "__main__":
    main()
