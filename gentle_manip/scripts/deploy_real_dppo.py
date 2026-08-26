from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Deploy a trained DPPO diffusion policy on the real XArm7 — runs in the 3.10 env
# (envs/dppo_deploy), which has the DPPO policy code (dppo: model.diffusion.*,
# gentle_manip.dppo.pointnet_diffusion) AND the hardware SDKs (pyrealsense2 + xArm),
# so the policy and RealBackend share one process (no IPC), like deploy_real.py for DP3:
#   uv run --project envs/dppo_deploy python gentle_manip/scripts/deploy_real_dppo.py \
#     --ckpt <run>/checkpoint/state_249.pt --ft-denoising-steps 10   # 0 for a BC checkpoint
#
# Handles BOTH orientation encodings: the proprio view + obs_dim are DERIVED from --obs-config
# (ee_quat -> 8-dim state, ee_rot6d -> 10-dim), so a rot6d student (e.g. state_800.pt from
# single_lift_mushroom_soft_abs_pcd_rot6d) works with the rot6d defaults; pass the matching
# --normalization (same converted dataset) and --obs-config. For a quat student, pass
# --obs-config .../point_cloud_1cam_outlier.yaml + its normalization.npz.
#
# Reuses run_deploy_loop from deploy_real.py (the shared receding-horizon loop + safety keys).
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_PKG = _REPO / "gentle_manip"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gentle_manip.actions.action_config import ActionConfig          # noqa: E402
from gentle_manip.envs.policy_env import PolicyEnv                    # noqa: E402
from gentle_manip.envs.real_backend import RealBackend               # noqa: E402
from gentle_manip.perception.obs_config import ObsConfig             # noqa: E402
from gentle_manip.scripts.deploy_real import (                       # noqa: E402
    _load_yaml, _resolve_config, run_deploy_loop)

# Ordered proprio state view the DPPO point-cloud student was trained on (== PROPRIO_VIEW in
# convert_demos): concat -> state. Orientation is ee_quat (8-dim total) OR ee_rot6d (10-dim),
# depending on the obs config the student trained with — derived from ObsConfig, never hardcoded.
def _proprio_view(obs_config) -> list:
    ori = "ee_rot6d" if getattr(obs_config, "ee_rot6d", False) else "ee_quat"
    return ["ee_pos", ori, "gripper_width"]


def _proprio_dim(obs_config) -> int:
    # ee_pos(3) + orientation(6 rot6d | 4 quat) + gripper(1)
    return 3 + (6 if getattr(obs_config, "ee_rot6d", False) else 4) + 1


def _load_net_arch(ckpt: Path) -> dict:
    """Network architecture + dims read from the checkpoint's OWN training config
    (`<run>/.hydra/config.yaml`, where `<run>` = ckpt.parents[1]). This makes the adapter load
    ANY trained net size (e.g. a wider vpstw with visual_feature_dim=512, mlp_dims=[1024]*3)
    without hardcoded arch flags. All fields read are literals (no OmegaConf interpolation).
    Returns {} (adapter falls back to the small-net defaults) if the config is absent."""
    import yaml
    cfg_path = Path(ckpt).resolve().parents[1] / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        print(f"  [warn] no hydra config at {cfg_path}; using default network arch", flush=True)
        return {}
    cfg = yaml.safe_load(open(cfg_path))
    # network block lives under `model` (top-level `network` kept as a fallback for older configs).
    net = (cfg.get("model", {}) or {}).get("network") or cfg.get("network", {}) or {}
    arch = {
        "obs_dim":            cfg.get("obs_dim"),
        "action_dim":         cfg.get("action_dim"),
        "cond_steps":         cfg.get("cond_steps"),
        "horizon_steps":      cfg.get("horizon_steps"),
        "denoising_steps":    cfg.get("denoising_steps"),
        "visual_feature_dim": cfg.get("visual_feature_dim", net.get("visual_feature_dim")),
        "pc_cond_steps":      cfg.get("pc_cond_steps", net.get("pc_cond_steps")),
        "mlp_dims":           net.get("mlp_dims"),
        "time_dim":           net.get("time_dim"),
        "pointnet":           net.get("pointnet"),
    }
    return {k: v for k, v in arch.items() if v is not None}


class DPPOPolicyAdapter:
    """obs dict -> raw [-1,1] action chunk for a DPPO PointNet diffusion policy.

    Matches run_deploy_loop's policy interface (reset / push / predict / n_action_steps) and
    replicates the sim bridge's obs handling (GenesisMultiStepVecEnv): concat the proprio view
    -> normalized "state" + raw "point_cloud", with n_obs_steps history. The model outputs a
    NORMALIZED action chunk which is un-normalized (demo stats) to the [-1,1] raw action that
    PolicyEnv.step then scales through the ActionPipeline — identical to how the policy was
    driven in sim, so real matches training.
    """

    def __init__(self, ckpt: str, normalization_path: str, *, obs_dim: int = 8,
                 action_dim: int = 7, cond_steps: int = 2, act_steps: int = 4,
                 horizon_steps: int = 4, denoising_steps: int = 20, ft_denoising_steps: int = 10,
                 proprio_view: "list | None" = None,
                 visual_feature_dim: int = 256, mlp_dims: "list | None" = None,
                 time_dim: int = 16, pc_cond_steps: int = 1, pointnet: "dict | None" = None,
                 device: str = "cuda:0") -> None:
        import torch  # noqa: F401
        from model.diffusion.diffusion_eval import DiffusionEval
        from gentle_manip.dppo.pointnet_diffusion import PointNetDiffusionMLP

        self.device = device
        self.n_action_steps = int(act_steps)
        self._cond_steps = int(cond_steps)
        # proprio keys, in order, whose concat forms the normalized "state" (quat OR rot6d student).
        self._view = list(proprio_view) if proprio_view else ["ee_pos", "ee_quat", "gripper_width"]
        # Network arch (visual_feature_dim / mlp_dims / time_dim / pointnet) MUST match the trained
        # checkpoint — read from its hydra config by main() so any net size loads; these defaults are
        # the original small net.
        net = PointNetDiffusionMLP(
            action_dim=action_dim, horizon_steps=horizon_steps, cond_dim=obs_dim * cond_steps,
            pc_cond_steps=pc_cond_steps, visual_feature_dim=visual_feature_dim, time_dim=time_dim,
            mlp_dims=list(mlp_dims) if mlp_dims else [512, 512, 512],
            activation_type="ReLU", residual_style=True,
            pointnet=pointnet or {"in_channels": 3, "use_layernorm": True, "final_norm": "layernorm"})
        # Same construction as cfg/.../eval_diffusion_pointnet.yaml (network_path loads weights).
        self.model = DiffusionEval(
            network_path=str(ckpt), ft_denoising_steps=int(ft_denoising_steps), use_ddim=False,
            network=net, predict_epsilon=True, denoised_clip_value=1.0, randn_clip_value=3,
            ddim_steps=int(ft_denoising_steps), horizon_steps=horizon_steps, obs_dim=obs_dim,
            action_dim=action_dim, denoising_steps=denoising_steps, device=device).eval()

        stats = np.load(normalization_path)
        self.obs_min = stats["obs_min"].astype(np.float32)
        self.obs_max = stats["obs_max"].astype(np.float32)
        self.action_min = stats["action_min"].astype(np.float32)
        self.action_max = stats["action_max"].astype(np.float32)
        # Fail loudly on a normalization file from a DIFFERENT dataset than the ckpt trained
        # on. Without this the width mismatch surfaces as an opaque broadcast error deep in
        # predict() (or, if the widths happen to agree, does not surface at all — it silently
        # de-normalizes to the wrong physical targets and the robot moves somewhere plausible
        # but wrong). Widths are the only automatic check available; the VALUES still have to
        # come from the right dataset (see --normalization help).
        if self.action_min.shape[0] != action_dim:
            raise ValueError(
                f"--normalization action stats are {self.action_min.shape[0]}-dim but the "
                f"checkpoint's action_dim is {action_dim} — this normalization.npz is from a "
                f"different dataset (e.g. a 10-dim rot6d one paired with a 7-dim euler ckpt). "
                f"Use the normalization.npz written by the SAME converted dataset the ckpt "
                f"trained on.\n  file: {normalization_path}")
        if self.obs_min.shape[0] != obs_dim:
            raise ValueError(
                f"--normalization obs stats are {self.obs_min.shape[0]}-dim but the proprio "
                f"obs_dim is {obs_dim} — check --obs-config (quat student -> 8, rot6d -> 10) "
                f"and that this normalization.npz matches the ckpt's dataset.\n"
                f"  file: {normalization_path}")
        # +1e-6 denom: IDENTICAL to the demo converter + the sim bridge, so a channel maps the
        # same way live as in the pretrain data.
        self._obs_range = (self.obs_max - self.obs_min) + 1e-6
        self._act_range = (self.action_max - self.action_min) + 1e-6
        self._hist: "deque | None" = None
        print(f"loaded DPPO policy (ft_denoising_steps={ft_denoising_steps}) from {ckpt}")

    # ── obs handling (mirrors GenesisMultiStepVecEnv, n_envs=1) ────────────────────
    def _modalities(self, obs: dict) -> dict:
        raw = np.concatenate(
            [np.asarray(obs[k], np.float32).reshape(1, -1) for k in self._view], axis=1)
        state = (2.0 * (raw - self.obs_min) / self._obs_range - 1.0).astype(np.float32)
        pc = np.asarray(obs["point_cloud"], np.float32).reshape(1, -1, 3)   # raw xyz (meters)
        return {"state": state, "point_cloud": pc}

    def _stacked(self) -> dict:
        h = list(self._hist)
        while len(h) < self._cond_steps:              # left-pad with the earliest obs
            h.insert(0, h[0])
        h = h[-self._cond_steps:]
        return {k: np.stack([s[k] for s in h], axis=1) for k in h[0]}

    # ── run_deploy_loop interface ──────────────────────────────────────────────────
    def reset(self, obs: dict) -> None:
        self._hist = deque([self._modalities(obs)], maxlen=self._cond_steps + 1)

    def push(self, obs: dict) -> None:
        self._hist.append(self._modalities(obs))

    def predict(self) -> np.ndarray:
        import torch
        cond = {k: torch.as_tensor(v, device=self.device)
                for k, v in self._stacked().items()}
        with torch.no_grad():
            traj = self.model(cond=cond, deterministic=True).trajectories.cpu().numpy()
        chunk = traj[0, : self.n_action_steps]        # (act_steps, action_dim) normalized [-1, 1]
        # un-normalize to the raw [-1, 1] action PolicyEnv.step expects (it applies ActionPipeline).
        raw = ((chunk + 1.0) / 2.0 * self._act_range + self.action_min).astype(np.float32)
        # RAW policy actions this chunk: net output (normalized [-1,1]) and the un-normalized raw
        # action fed to the ActionPipeline. First action of the chunk (a0 = the one executed next).
        with np.printoptions(precision=4, suppress=True):
            print(f"[policy] a0 net(norm)={chunk[0]}  raw={raw[0]}", flush=True)
        return raw


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy a DPPO diffusion policy on the real XArm7")
    p.add_argument("--ckpt", type=Path, required=True, help="a ft_ppo_diffusion_pointnet or BC checkpoint")
    p.add_argument("--ft-denoising-steps", type=int, default=10,
                   help="10 for a finetuned checkpoint, 0 for a BC (pretrained) checkpoint")
    p.add_argument("--normalization", type=Path,
                   default=_REPO / "dataset/dppo/single_lift_mushroom_soft_abs_pcd_rot6d/normalization.npz",
                   help="normalization.npz from the SAME converted dataset the ckpt trained on "
                        "(its obs_min/max width must match the proprio dim: 10 rot6d / 8 quat)")
    p.add_argument("--setup", type=Path, default=_PKG / "configs/setup/real_lab.yaml")
    p.add_argument("--obs-config", type=Path,
                   default=_PKG / "configs/obs/point_cloud_1cam_outlier_rot6d.yaml",
                   help="must match the student's training obs (orientation encoding + point-cloud "
                        "crop/1024/outlier). rot6d default; use point_cloud_1cam_outlier.yaml for a quat student")
    p.add_argument("--action-config", type=Path,
                   default=_PKG / "configs/action/abs_pose_abs_gripper.yaml",
                   help="must match the ckpt's training action config — its action_dim has to "
                        "equal the ckpt's (checked at startup). Default is the 10-dim rot6d "
                        "absolute config; pass abs_pose_euler_abs_gripper.yaml for a 7-dim "
                        "euler-absolute policy (its euler_frame_offset_deg is part of the "
                        "encoding, so the deploy config must be the one it trained with)")
    p.add_argument("--cond-steps", type=int, default=2)
    p.add_argument("--act-steps", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--rate", type=float, default=30.0, help="control rate (Hz)")
    p.add_argument("--pose-scale", type=float, default=1.0,
                   help="(delta mode only) <1 shrinks the 6 delta-pose dims for slower/"
                        "gentler motion. No-op in absolute mode — see --smooth-alpha instead.")
    p.add_argument("--smooth-alpha", type=float, default=None,
                   help="(absolute mode only) EMA low-pass filter alpha on the commanded "
                        "pos+rotation (gripper dim excluded), persisted across chunk re-plans "
                        "and reset on re-home. Lower = smoother/slower to track a new target "
                        "(e.g. start at 0.3). None (default) = off. This is the fix for "
                        "shaky/jittery absolute-pose commands in place of pose_scale, which "
                        "does not apply to absolute targets.")
    p.add_argument("--max-pos-step-m", type=float, default=None,
                   help="(absolute mode only) hard per-tick cap, meters PER AXIS, on how far "
                        "the commanded position may move from the previous command — a slew-"
                        "rate limiter, independent of/in addition to --smooth-alpha. None = off.")
    p.add_argument("--record", type=Path, default=None,
                   help="save the run in the demo pickle schema (for sim2real obs comparison). "
                        "With --shard-size>0 this is a DIRECTORY of shard_XXXX.pkl instead of one pkl")
    p.add_argument("--shard-size", type=int, default=10,
                   help="episodes per shard pkl (0 = single pkl); keeps each read/write small")
    p.add_argument("--gripper-offset-m", type=float, default=0.0,
                   help="add a fixed OPEN-UP offset (metres) to every commanded gripper width. "
                        "For the over-squeeze seen with v3.3-family policies: their data closes to "
                        "31.3 mm vs afucm/alzey's 34.1 mm, and the policy commands a near-constant "
                        "width, so +0.002-0.003 restores the gentler operating point with no "
                        "retraining. 0 (default) = unchanged.")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    setup = _load_yaml(_resolve_config(args.setup))
    obs_config = ObsConfig.from_dict(_load_yaml(_resolve_config(args.obs_config)))
    action_config = ActionConfig.from_dict(_load_yaml(_resolve_config(args.action_config)))

    backend = RealBackend(setup)
    env = PolicyEnv(backend, obs_config, action_config, task=None, max_episode_steps=10 ** 9)
    # Proprio view + obs_dim are DERIVED from the obs config (quat -> 8, rot6d -> 10), so the adapter
    # matches whatever the student trained on without hardcoding either representation.
    proprio_view = _proprio_view(obs_config)
    obs_dim = _proprio_dim(obs_config)
    print(f"proprio view = {proprio_view}  (obs_dim={obs_dim})")
    # Network arch (+ dims) come from the checkpoint's own training config so any net size loads.
    arch = _load_net_arch(args.ckpt)
    if arch.get("obs_dim") and arch["obs_dim"] != obs_dim:
        print(f"  [warn] ckpt obs_dim={arch['obs_dim']} != obs-config-derived {obs_dim} "
              f"(check --obs-config); using the ckpt's {arch['obs_dim']}", flush=True)
    if arch:
        print(f"  net arch from ckpt config: visual_feature_dim={arch.get('visual_feature_dim')} "
              f"mlp_dims={arch.get('mlp_dims')} obs_dim={arch.get('obs_dim')} "
              f"action_dim={arch.get('action_dim')}", flush=True)
    # The ActionPipeline decodes the policy's raw action, so its width MUST equal the ckpt's.
    # A mismatch is unrecoverable (10-dim rot6d vs 7-dim euler are different encodings, not just
    # different widths), so fail here rather than let the backend act on a mis-decoded pose.
    if arch.get("action_dim") and arch["action_dim"] != action_config.action_dim:
        raise SystemExit(
            f"action_dim mismatch: ckpt={arch['action_dim']} but --action-config "
            f"{args.action_config.name} is {action_config.action_dim}-dim "
            f"(mode={action_config.mode}, rot_repr={getattr(action_config, 'rot_repr', 'n/a')}). "
            f"Pass the action config the ckpt trained with — 10-dim rot6d = "
            f"abs_pose_abs_gripper.yaml, 7-dim euler = abs_pose_euler_abs_gripper.yaml.")
    policy = DPPOPolicyAdapter(
        args.ckpt, args.normalization,
        obs_dim=arch.get("obs_dim", obs_dim),
        action_dim=arch.get("action_dim", action_config.action_dim),
        proprio_view=proprio_view,
        cond_steps=arch.get("cond_steps", args.cond_steps), act_steps=args.act_steps,
        horizon_steps=arch.get("horizon_steps", 4),
        denoising_steps=arch.get("denoising_steps", 20),
        ft_denoising_steps=args.ft_denoising_steps,
        visual_feature_dim=arch.get("visual_feature_dim", 256),
        mlp_dims=arch.get("mlp_dims"), time_dim=arch.get("time_dim", 16),
        pc_cond_steps=arch.get("pc_cond_steps", 1), pointnet=arch.get("pointnet"),
        device=args.device)
    run_deploy_loop(env, policy, args.max_steps, args.rate,
                    pose_scale=args.pose_scale, record_path=args.record,
                    shard_size=args.shard_size, action_config=action_config,
                    smooth_alpha=args.smooth_alpha, max_pos_step_m=args.max_pos_step_m,
                    gripper_offset_m=args.gripper_offset_m)


if __name__ == "__main__":
    main()
