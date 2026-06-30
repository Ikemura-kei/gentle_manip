"""Offline evaluation of a trained DP3 checkpoint in the Genesis sim.

Runs in envs/dp3 (py3.8). Loads a trained checkpoint with its embedded training
config, then drives the SAME in-loop sim eval used during training
(SimXArm7Runner): it spawns gentle_manip.scripts.sim_server (genesis, 3.12) over
RPC, derives the eval scene/obs/augmentation/DR from the dataset the policy was
trained on (single source of truth, via the zarr's source_files attr), runs N
episodes with a fixed seed (reproducible), reports the success rate, and writes a
cam_ext mp4 per episode for the first --video-episodes.

    uv run --project envs/dp3 python gentle_manip/scripts/eval_sim.py \
        --ckpt third_party/DP3/3D-Diffusion-Policy/data/outputs/<run-dir> \
        --n-episodes 100 --seed 0 --video-episodes 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_DP3 = _REPO / "third_party" / "DP3" / "3D-Diffusion-Policy"
for p in (str(_REPO), str(_DP3)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _resolve_ckpt(ckpt: Path) -> Path:
    """Accept a run dir (→ checkpoints/latest.ckpt) or a .ckpt file directly."""
    ckpt = ckpt if ckpt.is_absolute() else (_REPO / ckpt)
    if ckpt.is_dir():
        cand = ckpt / "checkpoints" / "latest.ckpt"
        if not cand.is_file():
            ckpts = sorted((ckpt / "checkpoints").glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"no .ckpt under {ckpt / 'checkpoints'}")
            cand = ckpts[-1]
        return cand
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline sim eval of a trained DP3 checkpoint")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="run dir (uses checkpoints/latest.ckpt) or a .ckpt file")
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=1,
                    help="parallel Genesis envs (throughput); scenario set depends on this, "
                         "so keep it fixed when comparing runs")
    ap.add_argument("--seed", type=int, default=0, help="fixed eval seed (reproducible scenarios)")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--pose-scale", type=float, default=1.0,
                    help="multiply the 6 delta-pose dims of every command (e.g. 0.5 = "
                         "slower/gentler; gripper unaffected). Use the SAME value you'll "
                         "deploy with on the real robot. Raise --max-steps to compensate "
                         "(0.5 ~ doubles steps-to-lift).")
    ap.add_argument("--port", type=int, default=5571)
    ap.add_argument("--video-episodes", type=int, default=2,
                    help="record a cam_ext mp4 for the first N batches; each tiles ALL "
                         "num_envs sub-envs into a grid (0 = no video). Recorded batches "
                         "render every sub-env per step, so keep this small.")
    ap.add_argument("--video-dir", type=Path, default=None,
                    help="where to write videos (default: <run>/sim_eval/)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch  # noqa: F401  (ensure CUDA torch loads before genesis spawn)
    from train import TrainDP3Workspace
    from diffusion_policy_3d.env_runner.sim_xarm7_runner import SimXArm7Runner

    ckpt = _resolve_ckpt(args.ckpt)
    run_dir = ckpt.parent.parent          # <run>/checkpoints/x.ckpt -> <run>
    video_dir = args.video_dir if args.video_dir is not None else (run_dir / "sim_eval")
    video_dir = video_dir if video_dir.is_absolute() else (_REPO / video_dir)
    print(f"[eval_sim] checkpoint: {ckpt}")

    ws = TrainDP3Workspace.create_from_checkpoint(str(ckpt))   # builds with embedded cfg
    cfg = ws.cfg
    use_ema = bool(getattr(cfg.training, "use_ema", True))
    policy = ws.ema_model if (use_ema and ws.ema_model is not None) else ws.model
    policy.to(args.device)
    zarr_path = cfg.task.dataset.zarr_path
    print(f"[eval_sim] policy={'ema' if use_ema else 'model'} "
          f"n_obs_steps={cfg.n_obs_steps} n_action_steps={cfg.n_action_steps} "
          f"zarr={zarr_path}")
    print(f"[eval_sim] {args.n_episodes} episodes, num_envs={args.num_envs}, seed={args.seed}, "
          f"videos→{video_dir} (first {args.video_episodes})")

    runner = SimXArm7Runner(
        output_dir=str(run_dir),
        zarr_path=zarr_path,
        n_eval_episodes=args.n_episodes,
        max_steps=args.max_steps,
        port=args.port,
        eval_seed=args.seed,
        num_envs=args.num_envs,
        pose_scale=args.pose_scale,
        video_dir=str(video_dir),
        video_episodes=args.video_episodes,
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),
    )
    try:
        result = runner.run(policy)
    finally:
        per_success = [int(x) for x in getattr(runner, "last_successes", [])]
        per_steps = [int(x) for x in getattr(runner, "last_lengths", [])]
        runner.close()

    sr = result.get("sim_success_rate", result.get("test_mean_score", 0.0))

    # Persist a small record (the success number is otherwise stdout-only).
    import json
    video_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "checkpoint": str(ckpt),
        "zarr_path": str(zarr_path),
        "n_episodes": args.n_episodes,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "pose_scale": args.pose_scale,
        "success_rate": sr,
        "n_success": int(sum(per_success)),
        "per_episode_success": per_success,            # 0/1 per episode, in order
        "per_episode_steps": per_steps,                # steps-to-success (or horizon)
    }
    record_path = video_dir / "result.json"
    record_path.write_text(json.dumps(record, indent=2))

    print("\n" + "=" * 60)
    print(f"[eval_sim] DONE — sim success rate: {sr:.3f} "
          f"over {args.n_episodes} episodes (seed {args.seed})")
    print(f"[eval_sim] record: {record_path}")
    print(f"[eval_sim] videos in: {video_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
