"""Canonical-harness evaluation of a trained DP3 checkpoint (apples-to-apples with DPPO).

Routes a DP3 policy through the SAME shared harness (gentle_manip.evaluation.run_eval +
GenesisMultiStepVecEnv over a running serl_sim_server) that every DPPO eval uses — fixed
EvalSpec scenarios, per-episode DR audit, summary.json + episodes.csv + per-episode video —
instead of the bespoke SimXArm7Runner loop in eval_sim.py (hard requirement #1 in CLAUDE.md).

DP3 normalizes obs/actions INTERNALLY (its normalizer is embedded in the checkpoint), so the
venv is built with an IDENTITY normalization (obs/action min=-1, max=+1 → the venv's
2*(x-min)/(max-min+1e-6)-1 and its inverse become no-ops to ~5e-7): the policy sees RAW
proprio + cloud and emits RAW [-1,1]-space actions, exactly as in training on the zarr.

Run inside envs/dp3_arrhenius with cwd = third_party/DP3/3D-Diffusion-Policy (the
diffusion_policy_3d namespace package resolves via cwd). A serl_sim_server with the
MATCHING experiment (action pipeline!) must be listening on --port.

    cd third_party/DP3/3D-Diffusion-Policy && \
    uv run --project ../../../envs/dp3_arrhenius --no-sync python \
        ../../../gentle_manip/scripts/eval_dp3_harness.py \
        --ckpt <run>/checkpoints/epoch=0500-....ckpt --experiment <exp> --port 5570
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DP3 = _REPO / "third_party" / "DP3" / "3D-Diffusion-Policy"
for p in (str(_REPO), str(_DP3)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np


class _DP3Policy:
    """Harness Policy adapter: venv obs dict -> raw action chunk via DP3 predict_action."""

    def __init__(self, policy, device, n_action_steps):
        self.policy = policy
        self.device = device
        self.n_action_steps = int(n_action_steps)

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def act(self, obs):
        import torch
        with torch.no_grad():
            obs_dict = {
                "agent_pos": torch.from_numpy(np.asarray(obs["state"], np.float32)).to(self.device),
                "point_cloud": torch.from_numpy(np.asarray(obs["point_cloud"], np.float32)).to(self.device),
            }
            out = self.policy.predict_action(obs_dict)
            act = out["action"].detach().cpu().numpy()          # (n_env, n_action_steps, act_dim)
        return act[:, : self.n_action_steps]


def _identity_norm_npz(obs_dim: int, act_dim: int) -> str:
    """Temp normalization.npz that makes the venv's obs-norm and action-denorm identity."""
    f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(f, obs_min=-np.ones(obs_dim, np.float32), obs_max=np.ones(obs_dim, np.float32),
             action_min=-np.ones(act_dim, np.float32), action_max=np.ones(act_dim, np.float32))
    f.close()
    return f.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True, help=".ckpt file")
    ap.add_argument("--experiment", required=True,
                    help="experiment the sim server runs (sets the ActionPipeline — must match "
                         "the action space the checkpoint was trained on)")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--num-envs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-episode-steps", type=int, default=300)
    ap.add_argument("--scene-group-size", type=int, default=4)
    ap.add_argument("--port", type=int, default=5570)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default <run>/sim_eval_harness/<ckpt-stem>)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch  # noqa: F401
    from train import TrainDP3Workspace
    from gentle_manip.dppo.genesis_venv import build_genesis_venv
    from gentle_manip.evaluation import EvalSpec, run_eval

    ckpt = args.ckpt if args.ckpt.is_absolute() else (_REPO / args.ckpt)
    ws = TrainDP3Workspace.create_from_checkpoint(str(ckpt))
    cfg = ws.cfg
    use_ema = bool(getattr(cfg.training, "use_ema", True))
    policy = ws.ema_model if (use_ema and ws.ema_model is not None) else ws.model
    policy.to(args.device)
    policy.eval()
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    obs_dim = 8                                   # ee_pos + ee_quat + gripper_width
    act_dim = int(cfg.shape_meta.action.shape[0])
    print(f"[dp3-harness] ckpt={ckpt.name} ema={use_ema} n_obs_steps={n_obs_steps} "
          f"n_action_steps={n_action_steps} act_dim={act_dim}")

    out = args.out
    if out is None:
        run_dir = ckpt.parent.parent
        out = run_dir / "sim_eval_harness" / ckpt.stem.replace("=", "_")
    out = out if out.is_absolute() else (_REPO / out)
    out.mkdir(parents=True, exist_ok=True)

    # venv horizon must be EXACTLY policy_steps*act_steps: the truncation auto-reset then
    # fires once per batch like in DPPO evals (75*4=300), keeping the server-side DR RNG
    # stream aligned so both stacks see IDENTICAL per-batch scenarios (measured: with a
    # 296<300 horizon the auto-reset is skipped and pose draws diverge from batch 1 on).
    max_policy_steps = args.max_episode_steps // n_action_steps
    venv = build_genesis_venv(
        num_envs=args.num_envs, obs_steps=n_obs_steps, act_steps=n_action_steps,
        max_episode_steps=max_policy_steps * n_action_steps,
        normalization_path=_identity_norm_npz(obs_dim, act_dim),
        obs_keys=["ee_pos", "ee_quat", "gripper_width"], pointcloud_key="point_cloud",
        port=args.port)

    spec = EvalSpec(n_episodes=args.n_episodes, num_envs=args.num_envs, seed=args.seed,
                    max_policy_steps=max_policy_steps,
                    scene_group_size=args.scene_group_size)
    run_eval(venv, _DP3Policy(policy, args.device, n_action_steps), spec, str(out),
             experiment_name=args.experiment, checkpoint=str(ckpt))
    print(f"[dp3-harness] DONE -> {out}")


if __name__ == "__main__":
    main()
