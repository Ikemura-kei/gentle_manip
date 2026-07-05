"""Compare the plain-DP baseline against the full two-stage RDP (AT + LDP) on the
same lift-mini smoke-test checkpoints, using two metrics:

  1. Action-prediction error (RMSE, denormalized to real units: meters for position,
     meters for gripper width) from the open-loop chunk rollout — the offline analogue
     of `train_action_mse_error` logged during training, evaluated over a full episode.
  2. predict_action() wall-clock latency per replanning call — loosely analogous to
     the paper's Table I (inference time per module on their RTX 4090; ours is on
     whatever GPU this machine has).

IMPORTANT SCOPE CAVEAT (read before citing these numbers):
The paper's actual headline metric is real-robot rollout success, scored per-task by a
human rubric (e.g. for Bimanual Lifting: 1.0 = lifted cleanly, 0.5 = lifted but
compressed, 0.0 = dropped/not lifted; Sec. V "Evaluation Protocols" of arXiv:2503.02881),
averaged over 10 trials per test-time condition. That requires a real robot and is not
reproducible offline. Nothing here is that metric. These are the best offline proxies
available from checkpoints trained on the public mini dataset for a couple of epochs
each (a smoke test of the integration, not a converged model) — treat the numbers as
"does the pipeline work and which is in the right ballpark", not as a paper-comparable
result.

Also note: our LDP rollout calls the batched `predict_action()` once per replanning
window (one-shot decode of the whole latent chunk) — not the paper's fully closed-loop
per-control-tick autoregressive AT decoding fed by the *latest* tactile reading
(Fig. 6b), which only runs in the live real-robot control loop (env_runner/real_runner.py)
that we don't have hardware to exercise. So the AT's <1ms "fast policy" number in the
paper's Table I is not something this script can measure at all; what we measure for
LDP is the combined LDP-diffusion-sample + AT-decode cost of a single replanning call.

Run with:
    cd third_party/reactive_diffusion_policy && \
    ../../envs/rdp/.venv/bin/python ../../examples/rdp_compare_dp_vs_rdp.py \
        --dp-ckpt data/outputs/<dp_run>/checkpoints/latest.ckpt \
        --rdp-ckpt data/outputs/<ldp_run>/checkpoints/latest.ckpt
"""
import argparse
import os
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rdp_visualize_rollout import load_policy, predict_chunk, RDP_ROOT  # noqa: E402

import numpy as np
import torch
import hydra
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

ACTION_LABELS = ["x_l", "y_l", "z_l", "x_r", "y_r", "z_r", "grip_l", "grip_r"]


def rollout_and_score(ckpt_path, device, n_timing_calls=5):
    policy, cfg = load_policy(pathlib.Path(ckpt_path).resolve())
    policy.to(device)
    is_latent = "latent" in cfg.name

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataset_path = str((RDP_ROOT / cfg.task.dataset_path).resolve())
    rb = ReplayBuffer.copy_from_path(
        os.path.join(dataset_path, "replay_buffer.zarr"), keys=["action"]
    )
    episode_ends = rb.episode_ends[:]

    n_action_steps = cfg.n_action_steps
    stride = n_action_steps

    per_episode_rmse = []
    latencies = []
    n_replans = 0
    all_pred, all_gt = [], []
    for ep in range(len(episode_ends)):
        ep_start = 0 if ep == 0 else int(episode_ends[ep - 1])
        ep_end = int(episode_ends[ep])
        ep_len = ep_end - ep_start
        gt_action = rb["action"][ep_start:ep_end]
        pred_traj = np.full_like(gt_action, np.nan)

        t = 0
        while t < ep_len:
            sample = dataset[ep_start + t]
            obs = {k: v.unsqueeze(0) for k, v in sample["obs"].items()}
            extended_obs = {k: v.unsqueeze(0) for k, v in sample.get("extended_obs", {}).items()}

            if len(latencies) < n_timing_calls:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                chunk = predict_chunk(policy, cfg, is_latent, obs, extended_obs, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t0)
            else:
                chunk = predict_chunk(policy, cfg, is_latent, obs, extended_obs, device)

            n_fill = min(len(chunk), ep_len - t)
            pred_traj[t : t + n_fill] = chunk[:n_fill]
            t += stride
            n_replans += 1

        valid = ~np.isnan(pred_traj).any(axis=1)
        se = (pred_traj[valid] - gt_action[valid]) ** 2
        per_episode_rmse.append(np.sqrt(se.mean()))
        all_pred.append(pred_traj)
        all_gt.append(gt_action)

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    valid = ~np.isnan(all_pred).any(axis=1)
    per_dim_rmse = np.sqrt(((all_pred[valid] - all_gt[valid]) ** 2).mean(axis=0))

    return {
        "is_latent": is_latent,
        "overall_rmse": float(np.mean(per_episode_rmse)),
        "per_episode_rmse": per_episode_rmse,
        "per_dim_rmse": per_dim_rmse,
        "latency_mean_ms": float(np.mean(latencies) * 1000),
        "latency_std_ms": float(np.std(latencies) * 1000),
        "n_replans": n_replans,
        "n_timing_calls": len(latencies),
    }


def print_report(name, stats):
    kind = "Full RDP (AT + LDP)" if stats["is_latent"] else "Plain DP baseline"
    print(f"\n=== {name} — {kind} ===")
    print(f"  Action RMSE (real units, all dims pooled): {stats['overall_rmse']:.4f}")
    print(f"  Per-episode RMSE: {[f'{x:.4f}' for x in stats['per_episode_rmse']]}")
    print("  Per-dimension RMSE:")
    for label, val in zip(ACTION_LABELS, stats["per_dim_rmse"]):
        unit = "m" if "grip" not in label else "m (width)"
        print(f"    {label:8s}: {val:.4f} {unit}")
    print(f"  predict_action() latency: {stats['latency_mean_ms']:.1f} ms "
          f"+/- {stats['latency_std_ms']:.1f} ms (n={stats['n_timing_calls']} calls)")
    print(f"  total replanning calls over full rollout: {stats['n_replans']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-ckpt", required=True)
    parser.add_argument("--rdp-ckpt", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dp_stats = rollout_and_score(args.dp_ckpt, device)
    print_report("DP", dp_stats)

    rdp_stats = rollout_and_score(args.rdp_ckpt, device)
    print_report("RDP", rdp_stats)

    print("\n=== Summary ===")
    print(f"{'Model':<12} {'RMSE':>10} {'Latency (ms)':>14}")
    print(f"{'DP':<12} {dp_stats['overall_rmse']:>10.4f} {dp_stats['latency_mean_ms']:>14.1f}")
    print(f"{'RDP':<12} {rdp_stats['overall_rmse']:>10.4f} {rdp_stats['latency_mean_ms']:>14.1f}")
    print("\n(See module docstring for scope caveats — these are offline proxies for a "
          "2-epoch smoke test, not the paper's real-robot success-rate metric.)")


if __name__ == "__main__":
    main()
