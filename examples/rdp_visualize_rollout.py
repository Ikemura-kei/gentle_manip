"""Visualize an RDP inference rollout: camera frames + GelSight tactile marker
deformation + predicted-vs-ground-truth action trajectory, rendered to an MP4.

Works for both checkpoint types trained in this repo:
  - plain DP  (reactive_diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy)
  - full RDP  (...policy.latent_diffusion_unet_image_policy.LatentDiffusionUnetImagePolicy,
               i.e. the AT + LDP two-stage model), detected via `'latent' in cfg.name`.

RDP ships no physics simulator (no MuJoCo/PyBullet/Isaac in the submodule — it's a
real-robot-only framework, see docs/franka_setup_instructions.md and the Flexiv/Franka
hardware requirements in its README). So "better visualization" here means a proper
rollout video reconstructed directly from the real recorded data + a trained
checkpoint, rather than a simulator replay.

Rollout semantics: every `stride` steps (default = n_action_steps, i.e. how RDP would
actually replan during deployment) the policy is conditioned on the last n_obs_steps
of REAL recorded observations and predicts the next n_action_steps of action; that
predicted open-loop chunk is what's plotted against the teleop ground truth. This is
the same mechanism as the `train_action_mse_error` metric logged during training,
just over a full episode and rendered rather than reduced to one MSE number.

Obs windows are read via `dataset[i]` (the real RealImageTactileDataset /
RealImageTactileLatentDiffusionDataset __getitem__) rather than hand-reconstructing the
T_slice/temporal-downsample logic — for a fixed episode with every timestep in the
train mask, sampler list-position i corresponds exactly to raw zarr row i (verified from
common/sampler.py's create_indices: pad_before == n_obs_steps-1 cancels the loop-index
offset), so dataset[ep_start + t] gives the correctly-preprocessed obs/extended_obs/action
for global timestep t with no manual slicing.

Run with:
    cd third_party/reactive_diffusion_policy && \
    ../../envs/rdp/.venv/bin/python ../../examples/rdp_visualize_rollout.py \
        --ckpt data/outputs/<run_dir>/checkpoints/latest.ckpt \
        --episode 0 --out /tmp/rdp_rollout_ep0.mp4
"""
import argparse
import dill
import os
import sys
import pathlib

RDP_ROOT = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "reactive_diffusion_policy"
sys.path.insert(0, str(RDP_ROOT))

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer

OmegaConf.register_new_resolver("eval", eval, replace=True)

ACTION_LABELS = ["x_l", "y_l", "z_l", "x_r", "y_r", "z_r", "grip_l", "grip_r"]


def load_policy(ckpt_path):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload)
    policy = workspace.model
    if hasattr(policy, "at"):
        # The AT submodule's own `normalizer` doesn't survive the state-dict
        # round-trip (its ParameterDict is empty at construction, so load_state_dict
        # has nothing to match new keys against) — upstream's own
        # eval_real_robot_flexiv.py works around this the same way after loading.
        policy.at.set_normalizer(policy.normalizer)
    policy.eval()
    return policy, cfg


def predict_chunk(policy, cfg, is_latent, obs, extended_obs, device):
    obs = {k: v.to(device) for k, v in obs.items()}
    with torch.no_grad():
        if is_latent:
            extended_obs = {k: v.to(device) for k, v in extended_obs.items()}
            result = policy.predict_action(
                obs,
                dataset_obs_temporal_downsample_ratio=cfg.task.dataset.obs_temporal_downsample_ratio,
                extended_obs_dict=extended_obs,
            )
        else:
            result = policy.predict_action(obs)
    return result["action"][0].cpu().numpy()  # (n_action_steps, action_dim)


def render_frame(ext_img, wrist_img, gel_img, gel_init, gel_offset, gt_action, pred_traj, t, action_labels, title):
    fig, axes = plt.subplot_mosaic(
        [["ext", "wrist", "gel"], ["action", "action", "action"]],
        figsize=(12, 7), gridspec_kw={"height_ratios": [1.2, 1]},
    )
    fig.suptitle(title)
    axes["ext"].imshow(ext_img)
    axes["ext"].set_title("external_img")
    axes["ext"].axis("off")

    axes["wrist"].imshow(wrist_img)
    axes["wrist"].set_title("left_wrist_img")
    axes["wrist"].axis("off")

    axes["gel"].imshow(gel_img)
    h, w = gel_img.shape[:2]
    origins = gel_init * np.array([w, h])
    scale = 8.0  # exaggerate small marker offsets for visibility
    dirs = gel_offset * np.array([w, h]) * scale
    axes["gel"].quiver(
        origins[:, 0], origins[:, 1], dirs[:, 0], dirs[:, 1],
        color="lime", angles="xy", scale_units="xy", scale=1, width=0.005,
    )
    axes["gel"].set_title("left_gripper2_img + GelSight marker offsets (8x)")
    axes["gel"].axis("off")

    T = gt_action.shape[0]
    ts = np.arange(T)
    for i, label in enumerate(action_labels):
        axes["action"].plot(ts, gt_action[:, i], "--", color=f"C{i}", alpha=0.5)
        valid = ~np.isnan(pred_traj[:, i])
        axes["action"].plot(ts[valid], pred_traj[valid, i], "-", color=f"C{i}", label=label)
    axes["action"].axvline(t, color="black", linewidth=1)
    axes["action"].set_xlabel("timestep (dashed = ground truth, solid = predicted)")
    axes["action"].legend(ncol=4, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--stride", type=int, default=None, help="replanning interval; default = n_action_steps")
    parser.add_argument("--out", default="/tmp/rdp_rollout.mp4")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    policy, cfg = load_policy(pathlib.Path(args.ckpt).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    is_latent = "latent" in cfg.name

    dataset = hydra.utils.instantiate(cfg.task.dataset)

    dataset_path = str((RDP_ROOT / cfg.task.dataset_path).resolve())
    load_keys = set(cfg.task.shape_meta.obs.keys()) | {
        "action", "external_img", "left_wrist_img",
        "left_gripper2_img", "left_gripper2_initial_marker", "left_gripper2_marker_offset",
    }
    rb = ReplayBuffer.copy_from_path(os.path.join(dataset_path, "replay_buffer.zarr"), keys=list(load_keys))
    episode_ends = rb.episode_ends[:]
    ep_start = 0 if args.episode == 0 else int(episode_ends[args.episode - 1])
    ep_end = int(episode_ends[args.episode])
    ep_len = ep_end - ep_start

    n_action_steps = cfg.n_action_steps
    stride = args.stride or n_action_steps

    gt_action = rb["action"][ep_start:ep_end]
    pred_traj = np.full_like(gt_action, np.nan)

    t = 0
    while t < ep_len:
        sample = dataset[ep_start + t]
        obs = {k: v.unsqueeze(0) for k, v in sample["obs"].items()}
        extended_obs = {k: v.unsqueeze(0) for k, v in sample.get("extended_obs", {}).items()}
        chunk = predict_chunk(policy, cfg, is_latent, obs, extended_obs, device)
        n_fill = min(len(chunk), ep_len - t)
        pred_traj[t : t + n_fill] = chunk[:n_fill]
        t += stride
    print(f"Rolled out episode {args.episode} ({ep_len} steps), replanning every {stride} steps, "
          f"{'LDP (full RDP)' if is_latent else 'plain DP'}")

    ext_imgs = rb["external_img"][ep_start:ep_end]
    wrist_imgs = rb["left_wrist_img"][ep_start:ep_end]
    gel_imgs = rb["left_gripper2_img"][ep_start:ep_end]
    gel_inits = rb["left_gripper2_initial_marker"][ep_start:ep_end]
    gel_offsets = rb["left_gripper2_marker_offset"][ep_start:ep_end]

    title = args.title or ("Full RDP (AT + LDP)" if is_latent else "Plain DP baseline")
    frames = []
    for i in range(ep_len):
        frames.append(render_frame(
            ext_imgs[i], wrist_imgs[i], gel_imgs[i], gel_inits[i], gel_offsets[i],
            gt_action, pred_traj, i, ACTION_LABELS, title,
        ))
        if (i + 1) % 50 == 0:
            print(f"  rendered {i+1}/{ep_len} frames")

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved rollout video to {args.out}")


if __name__ == "__main__":
    main()
