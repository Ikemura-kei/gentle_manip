"""Offline pre-deploy probe: what does a policy command on a REAL observation?

Feeds a trained checkpoint (a) a real observation, (b) a simulated one, and (c) two HYBRIDS
that swap proprioception and point cloud between them, then decodes the predicted action chunk
into physical targets. No robot, no simulator — seconds to run.

Why: a co-trained policy can be simultaneously correct on simulated clouds (its 90%+ majority
data) and non-functional on real ones, and neither the training loss nor the simulated
evaluation can see it — sim eval never presents a real cloud. This probe is the cheap gate
that catches it at the desk. It found the v33 real-slice bug (2026-08-25): every v33 checkpoint
commanded gripper 44 mm and z 0.25 m on any REAL cloud, including real demos from its own
training set, while afucm handled all four combinations correctly. The hybrid rows are what
localize the cause — real cloud triggered it with either proprioception, so the visual branch
was implicated and the proprioceptive one exonerated.

Expected for a healthy policy at episode start: gripper stays OPEN (~80 mm) and commanded z
DESCENDS below the current end-effector height, on every row.

TWO THINGS THE PROBE GETS WRONG IF YOU ARE NOT CAREFUL:
  1. **Feed the real variant the policy TRAINED on.** A v33b_shift9 policy must be probed with
     `single_lift_mushroom_real_merged_shift9mm`, an uncorrected-slice policy with
     `single_lift_mushroom_real_merged`. Probing the poisoned orkam with SHIFT-CORRECTED clouds
     made it PASS; with its own uncorrected clouds it fails outright (grip 56 mm, z spread
     +-30 mm). Wrong input = wrong verdict, in both directions.
  2. **Diffusion sampling starts from random noise**, so a single draw is a noisy verdict — a
     marginal policy flips PASS/FAIL between identical runs. `--n-samples` (default 8) averages
     and reports the spread; the spread is itself diagnostic (a healthy policy sits at +-1-2 mm
     on z and +-0.0 mm on the gripper).

    uv run --project envs/dppo python examples/sim2real_diagnose/probe_policy_real_obs.py \
        --ckpt downloaded_runs/afucm/checkpoint/state_400.pt \
        --normalization downloaded_runs/afucm/normalization.npz \
        --real dataset/demos/single_lift_mushroom_real_merged \
        --sim  dataset/demos/single_lift_mushroom_soft/26-08-25-zrg
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]        # run by path from any cwd (envs/dppo has no
if str(_REPO) not in sys.path:                     # editable gentle_manip install)
    sys.path.insert(0, str(_REPO))

PROPRIO = ("ee_pos", "ee_quat", "gripper_width")
MODALITIES = PROPRIO + ("point_cloud",)


def load_policy(ckpt: Path, normalization: Path, *, obs_dim, action_dim, horizon, cond_steps,
                denoising, visual_dim, mlp_dims, device="cuda:0"):
    """Build the same network+wrapper the deploy adapter builds (kept in sync with
    gentle_manip/scripts/deploy_real_dppo.py::DPPOPolicyAdapter)."""
    from model.diffusion.diffusion_eval import DiffusionEval
    from gentle_manip.dppo.pointnet_diffusion import PointNetDiffusionMLP
    net = PointNetDiffusionMLP(
        action_dim=action_dim, horizon_steps=horizon, cond_dim=obs_dim * cond_steps,
        pc_cond_steps=1, visual_feature_dim=visual_dim, time_dim=16, mlp_dims=list(mlp_dims),
        activation_type="ReLU", residual_style=True,
        pointnet={"in_channels": 3, "use_layernorm": True, "final_norm": "layernorm"})
    model = DiffusionEval(
        network_path=str(ckpt), ft_denoising_steps=0, use_ddim=False, network=net,
        predict_epsilon=True, denoised_clip_value=1.0, randn_clip_value=3, ddim_steps=0,
        horizon_steps=horizon, obs_dim=obs_dim, action_dim=action_dim,
        denoising_steps=denoising, device=device).eval()
    stats = np.load(normalization)
    return model, stats


def first_obs(run: Path, idx: int = 0) -> dict:
    eps = pickle.load(open(run / "data.pkl", "rb"))["episodes"]
    o = eps[idx]["observations"]
    out = {}
    for k in MODALITIES:
        v = np.asarray(o[k])[0]
        out[k] = v.reshape(1, -1) if k == "gripper_width" else v[None]
    return out


def predict(model, stats, obs: dict, cond_steps: int, act_steps: int, device="cuda:0"):
    """Format exactly as DPPOPolicyAdapter._modalities/_stacked do: a normalized flat `state`
    (the proprio view concatenated) plus the raw-xyz cloud, each repeated over cond_steps."""
    import torch
    raw = np.concatenate([np.asarray(obs[k], np.float32).reshape(1, -1) for k in PROPRIO], axis=1)
    rng = (stats["obs_max"] - stats["obs_min"]) + 1e-6
    state = (2.0 * (raw - stats["obs_min"]) / rng - 1.0).astype(np.float32)
    pc = np.asarray(obs["point_cloud"], np.float32).reshape(1, -1, 3)
    cond = {k: torch.as_tensor(np.repeat(v[:, None], cond_steps, axis=1), device=device)
            for k, v in {"state": state, "point_cloud": pc}.items()}
    with torch.no_grad():
        traj = model(cond=cond, deterministic=True).trajectories.cpu().numpy()
    chunk = traj[0, :act_steps]
    arng = (stats["action_max"] - stats["action_min"]) + 1e-6
    return ((chunk + 1.0) / 2.0 * arng + stats["action_min"]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--normalization", type=Path, required=True)
    ap.add_argument("--real", type=Path, required=True, help="real demo run dir (data.pkl)")
    ap.add_argument("--sim", type=Path, required=True, help="sim demo run dir (data.pkl)")
    ap.add_argument("--real-episode", type=int, default=0)
    ap.add_argument("--sim-episode", type=int, default=0)
    ap.add_argument("--action-config", type=Path,
                    default=Path("gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml"))
    ap.add_argument("--obs-dim", type=int, default=8)
    ap.add_argument("--action-dim", type=int, default=7)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--act-steps", type=int, default=4)
    ap.add_argument("--cond-steps", type=int, default=2)
    ap.add_argument("--denoising", type=int, default=20)
    ap.add_argument("--visual-dim", type=int, default=512)
    ap.add_argument("--mlp-dims", type=int, nargs="+", default=[1024, 1024, 1024])
    ap.add_argument("--n-samples", type=int, default=8,
                    help="diffusion sampling starts from random noise, so a single draw is a "
                         "noisy verdict — average this many (default 8) and threshold the mean")
    args = ap.parse_args()

    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.actions.pipeline import ActionPipeline
    pipe = ActionPipeline(ActionConfig.from_dict(yaml.safe_load(args.action_config.read_text())))

    model, stats = load_policy(args.ckpt, args.normalization, obs_dim=args.obs_dim,
                               action_dim=args.action_dim, horizon=args.horizon,
                               cond_steps=args.cond_steps, denoising=args.denoising,
                               visual_dim=args.visual_dim, mlp_dims=args.mlp_dims)
    real = first_obs(args.real, args.real_episode)
    sim = first_obs(args.sim, args.sim_episode)
    rows = (("sim proprio  + sim cloud", sim),
            ("REAL proprio + sim cloud", {**sim, **{k: real[k] for k in PROPRIO}}),
            ("sim proprio  + REAL cloud", {**sim, "point_cloud": real["point_cloud"]}),
            ("REAL proprio + REAL cloud", real))

    print(f"ckpt: {args.ckpt}")
    print(f"healthy policy: gripper ~80 mm and commanded z DESCENDS on every row "
          f"(mean of {args.n_samples} diffusion samples)\n")
    bad = False
    for name, obs in rows:
        z0, g0 = [], []
        for _ in range(args.n_samples):
            raw = predict(model, stats, obs, args.cond_steps, args.act_steps)
            dec = np.stack([pipe.process(a.reshape(1, -1))[0] for a in raw])
            z0.append(dec[0, 2]); g0.append(dec[0, -1] * 1000)
        z0, g0 = np.array(z0), np.array(g0)
        z_here = float(obs["ee_pos"][0][2])
        climbing = z0.mean() > z_here + 0.002
        closing = g0.mean() < 70.0
        flag = " <-- SUSPECT" if (climbing or closing) else ""
        print(f"  {name:26s} ee_z {z_here:.4f} -> cmd z {z0.mean():.4f} +-{z0.std()*1000:.1f}mm "
              f"({'CLIMB' if climbing else 'descend'})  grip {g0.mean():5.1f} +-{g0.std():.1f} mm{flag}")
        bad |= climbing or closing
    print("\nRESULT:", "FAIL — do not deploy" if bad else "PASS")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
