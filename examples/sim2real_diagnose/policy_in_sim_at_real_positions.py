"""Run a DPPO policy CLOSED-LOOP in sim with the mushroom placed at the SAME positions as a
real rollout — to answer: does the policy squeeze in sim too, under the same initial conditions?

Sim gives von Mises STRESS (a direct crush measure real can't). One point-cloud video per
trajectory + a sim-vs-real gripper-width table. Same policy config as train/deploy:
DiffusionEval(PointNet) ft_denoising_steps=10, cond_steps 2, act_steps 4, denoising 20, obs
= point_cloud_1cam_outlier (crop/1024/outlier + quat jitter), action = delta_pose_delta_gripper.

    MUJOCO_GL=egl uv run --project envs/sim python \
      examples/sim2real_diagnose/policy_in_sim_at_real_positions.py \
      --ckpt logs/.../lqitl/checkpoint/state_249.pt --ft-denoising-steps 10

Runs the DPPO policy IN-PROCESS with genesis (envs/sim), so it needs the dppo policy code on
the path (this file adds third_party/dppo) + `einops` in envs/sim:
    uv pip install --python envs/sim/.venv/bin/python einops
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import deque
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import imageio.v2 as imageio     # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "third_party" / "dppo"))
sys.path.insert(0, str(_REPO))
_CFG = _REPO / "gentle_manip" / "configs"

from gentle_manip.actions.action_config import ActionConfig          # noqa: E402
from gentle_manip.assets.registry import get_object_def              # noqa: E402
from gentle_manip.envs.policy_env import PolicyEnv                   # noqa: E402
from gentle_manip.envs.sim_backend import SimBackend                 # noqa: E402
from gentle_manip.experiment import Experiment                       # noqa: E402
from gentle_manip.perception.obs_config import ObsConfig             # noqa: E402
from gentle_manip.tasks.single_lift import SingleLiftTask            # noqa: E402

_PROPRIO_VIEW = ["ee_pos", "ee_quat", "gripper_width"]


class _Policy:
    """Same obs/action handling as the sim bridge + deploy adapter."""
    def __init__(self, ckpt, norm_path, ft_denoising_steps, cond_steps=2, act_steps=4,
                 device="cuda:0"):
        import torch  # noqa: F401
        from model.diffusion.diffusion_eval import DiffusionEval
        from gentle_manip.dppo.pointnet_diffusion import PointNetDiffusionMLP
        self.device, self.act_steps, self.cond_steps = device, act_steps, cond_steps
        net = PointNetDiffusionMLP(action_dim=7, horizon_steps=4, cond_dim=8 * cond_steps,
            pc_cond_steps=1, visual_feature_dim=256, time_dim=16, mlp_dims=[512, 512, 512],
            activation_type="ReLU", residual_style=True,
            pointnet={"in_channels": 3, "use_layernorm": True, "final_norm": "layernorm"})
        self.model = DiffusionEval(network_path=str(ckpt), ft_denoising_steps=int(ft_denoising_steps),
            use_ddim=False, network=net, predict_epsilon=True, denoised_clip_value=1.0,
            randn_clip_value=3, ddim_steps=int(ft_denoising_steps), horizon_steps=4, obs_dim=8,
            action_dim=7, denoising_steps=20, device=device).eval()
        s = np.load(norm_path)
        self.omin, self.omax = s["obs_min"].astype(np.float32), s["obs_max"].astype(np.float32)
        self.amin, self.amax = s["action_min"].astype(np.float32), s["action_max"].astype(np.float32)
        self.orange = (self.omax - self.omin) + 1e-6
        self.arange = (self.amax - self.amin) + 1e-6
        self.hist = None

    def _mod(self, obs):
        raw = np.concatenate([np.asarray(obs[k], np.float32).reshape(1, -1) for k in _PROPRIO_VIEW], 1)
        return {"state": (2 * (raw - self.omin) / self.orange - 1).astype(np.float32),
                "point_cloud": np.asarray(obs["point_cloud"], np.float32).reshape(1, -1, 3)}

    def reset(self, obs): self.hist = deque([self._mod(obs)], maxlen=self.cond_steps + 1)
    def push(self, obs): self.hist.append(self._mod(obs))

    def _stack(self):
        h = list(self.hist)
        while len(h) < self.cond_steps: h.insert(0, h[0])
        h = h[-self.cond_steps:]
        return {k: np.stack([s[k] for s in h], 1) for k in h[0]}

    def predict(self):
        import torch
        cond = {k: torch.as_tensor(v, device=self.device) for k, v in self._stack().items()}
        with torch.no_grad():
            traj = self.model(cond=cond, deterministic=True).trajectories.cpu().numpy()
        chunk = traj[0, :self.act_steps]
        return ((chunk + 1) / 2 * self.arange + self.amin).astype(np.float32)


def _frame(fig, ax, rgb, gw, stress, tag):
    """RGB scene render (cam_ext view) with a gripper/stress overlay."""
    ax.clear()
    ax.imshow(rgb)
    ax.set_axis_off()
    ax.set_title(f"{tag}   grip={gw*1000:.0f} mm   stress={stress:.0f}", fontsize=11)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ft-denoising-steps", type=int, default=10)
    ap.add_argument("--real", type=Path, default=_REPO / "dataset/real_deploy/dppo/test.pkl")
    ap.add_argument("--norm", type=Path, default=_REPO / "dataset/dppo/single_lift_mushroom_soft_pcd/normalization.npz")
    ap.add_argument("--out", type=Path, default=Path("examples/sim2real_diagnose/figures/policy_in_sim"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    real = pickle.load(open(args.real, "rb"))["episodes"]
    obs_cfg = ObsConfig.from_dict(yaml.safe_load((_CFG / "obs/point_cloud_1cam_outlier.yaml").read_text()))
    act_cfg = ActionConfig.from_dict(yaml.safe_load((_CFG / "action/delta_pose_delta_gripper_fast_rot.yaml").read_text()))
    task = SingleLiftTask(Experiment.load("single_lift_mushroom_soft").task_cfg)  # full cfg (210/250)
    nominal = np.array(get_object_def("mushroom").default_pos[:2], np.float32)
    backend = SimBackend(task.scene_spec, 1, config={"sim": {"settle_steps": 30}}, use_subprocess=False)
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=None, max_episode_steps=10 ** 9)
    policy = _Policy(args.ckpt, args.norm, args.ft_denoising_steps, device=args.device)

    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111)
    print(f"{'ep':>3} {'obj_xy(=real grasp)':>22} {'real gw_min':>11} {'SIM gw_min':>10} {'SIM stress_pk':>13} {'SIM lifted':>10}")
    print("-" * 74)
    rows = []
    for i, ep in enumerate(real):
        ree = np.asarray(ep["observations"]["ee_pos"]); rgw = np.asarray(ep["observations"]["gripper_width"])[:, 0]
        T = len(ree)
        gxy = ree[int(np.argmin(ree[:, 2])), :2]                       # real grasp xy = object proxy
        obs = env.reset(object_dxy=(gxy - nominal)[None, :])
        policy.reset(obs)
        z0 = float(backend.get_sim_feedback().object_center[0, 2])
        frames, sim_gw, sim_stress = [], [], []
        steps = 0
        while steps < T:
            for a in policy.predict():
                obs = env.step(a[None, :].astype(np.float32))[0]
                policy.push(obs)
                fb = backend.get_sim_feedback()
                st = float(np.asarray(fb.extra["von_mises_stress"])[0].max())
                gw = float(obs["gripper_width"][0, 0])
                sim_gw.append(gw); sim_stress.append(st)
                rgb = backend.render_rgb()                             # (H, W, 3) cam_ext scene view
                frames.append(_frame(fig, ax, rgb, gw, st, f"ep{i} sim t={steps}"))
                steps += 1
                if steps >= T: break
        z_end = float(backend.get_sim_feedback().object_center[0, 2])
        lifted = z_end - z0
        imageio.mimsave(str(args.out / f"sim_ep{i}.mp4"), frames, fps=30, macro_block_size=1)
        rows.append((i, gxy, rgw.min(), min(sim_gw), max(sim_stress), lifted))
        print(f"{i:>3} ({gxy[0]:.3f},{gxy[1]:+.3f})        {rgw.min():>11.3f} {min(sim_gw):>10.3f} {max(sim_stress):>13.0f} {lifted*100:>8.1f}cm")
    plt.close(fig)
    env.close()
    print(f"\nvideos -> {args.out}/sim_ep*.mp4")


if __name__ == "__main__":
    main()
