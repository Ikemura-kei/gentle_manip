"""Deploy a fine-tuned pi0.5 VLA on the REAL XArm7 (single external camera).

    uv run --project envs/dp3 python -m gentle_manip.scripts.deploy_real_pi05 \
        --checkpoint third_party/openpi/checkpoints/pi05_libero/pi05_real7_ext/<step> \
        --prompt "pick up the object from table gently" \
        --record dataset/demos/single_lift_real7_pi05_deploy

WHY A SEPARATE SCRIPT. `deploy_real_dppo.py` builds a PointNet diffusion policy from a DPPO
checkpoint; `pi05/eval_harness.py` drives the SIM harness. Neither runs a pi0.5 policy against
`RealBackend`. This is the missing third path -- it reuses `deploy_real.run_deploy_loop` (the same
receding-horizon loop, keyboard controls, recording and safety caps the DPPO deploy uses) and only
swaps in a pi0.5 policy adapter.

⚠ **THE WRIST MASK MUST MATCH TRAINING.** The model was trained with
`gentle_manip.pi05.masked_wrist` active (left-wrist slot zeroed AND `image_mask=False`, because
the real rig has no wrist camera). `masked_wrist.patch()` is applied HERE before the policy is
constructed. Without it openpi's hardcoded `image_mask=True` would feed a black frame as a VALID
view -- a silent train/serve mismatch that degrades the policy without erroring.

⚠ **PROMPT.** The model was trained on 6 phrasings, two of which say the literal word "object".
Passing one of those tests generic-prompt behaviour (no object name); passing an object-named one
tests the specific case. There is no default -- state it explicitly.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_REPO = _PKG.parent
sys.path.insert(0, str(_REPO))

from gentle_manip.scripts.deploy_real import (_load_yaml, _resolve_config,  # noqa: E402
                                              run_deploy_loop)


class Pi05RealPolicy:
    """pi0.5 -> the (n_action_steps, 7) chunk `run_deploy_loop` expects.

    Mirrors `pi05.eval_policy.Pi05EvalPolicy._one` exactly (same letterbox, same 8-dim proprio
    order, same prompt handling) so the real observation is built the way training saw it.
    """

    def __init__(self, checkpoint_dir: Path, prompt: str, config_name: str = "pi05_libero",
                 repo_id: str | None = None, image_size: int = 224, n_action_steps: int = 10):
        # IMPORT ORDER IS LOAD-BEARING — this must be the FIRST openpi import in the process.
        # openpi.training.checkpoints (orbax) MUST precede openpi.policies.policy (torch);
        # the reverse SEGFAULTS this box's torch + jax-CUDA combination (measured 2026-09-03:
        # `import openpi.policies.policy` then `import openpi.training.checkpoints` -> SIGSEGV,
        # swapped -> fine). Both masked_wrist (via openpi.policies.libero_policy) and
        # policy_config pull policy.py, so this import comes before BOTH.
        import openpi.training.checkpoints as _ckpts
        from gentle_manip.pi05 import masked_wrist
        masked_wrist.patch()                      # BEFORE the policy is built -- see module docstring
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        cfg = _config.get_config(config_name)
        # Norm stats live in the checkpoint's assets under the TRAINING repo id (e.g.
        # gm/real7_ext), not the config's default (physical-intelligence/libero). Newer openpi
        # takes repo_id=; the pinned 215abfb checkout takes norm_stats= — support both.
        try:
            self._policy = _policy_config.create_trained_policy(cfg, str(checkpoint_dir),
                                                                repo_id=repo_id)
        except TypeError:
            ns = None
            if repo_id:
                ns = _ckpts.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", repo_id)
            self._policy = _policy_config.create_trained_policy(cfg, str(checkpoint_dir),
                                                                norm_stats=ns)
        self.prompt = str(prompt)
        self.image_size = int(image_size)
        self.n_action_steps = int(n_action_steps)
        print(f"[pi05-deploy] checkpoint={checkpoint_dir}")
        print(f"[pi05-deploy] prompt={self.prompt!r}  n_action_steps={self.n_action_steps}")
        self._last_obs = None

    # run_deploy_loop drives policies as reset(obs) / push(obs) / predict(); pi0.5 itself is
    # stateless per inference (single frame + proprio, no history buffer), so these just track
    # the latest observation and delegate to act(). Added 2026-09-03: the class previously
    # exposed only act(obs), which the shared loop never calls -> TypeError on reset.
    def reset(self, obs: dict | None = None) -> None:
        self._last_obs = obs

    def push(self, obs: dict) -> None:
        self._last_obs = obs

    def predict(self) -> np.ndarray:
        if self._last_obs is None:
            raise RuntimeError("predict() before reset()/push() — no observation available")
        return self.act(self._last_obs)

    def act(self, obs: dict) -> np.ndarray:
        from openpi.shared import image_tools
        S = self.image_size
        ext = np.asarray(obs["image_cam_ext"], np.uint8)
        if ext.ndim == 4:                          # (n_env, H, W, 3) -> the single real env
            ext = ext[0]
        base = np.asarray(image_tools.resize_with_pad(ext, S, S), np.uint8)
        state = np.concatenate([
            np.asarray(obs["ee_pos"], np.float32).reshape(-1)[:3],
            np.asarray(obs["ee_quat"], np.float32).reshape(-1)[:4],
            np.asarray(obs["gripper_width"], np.float32).reshape(-1)[:1],
        ]).astype(np.float32)
        assert state.shape[0] == 8, f"expected 8-dim proprio, got {state.shape[0]}"
        out = self._policy.infer({
            "observation/image": base,
            # zeroed AND masked off by masked_wrist -- the rig has no wrist camera
            "observation/wrist_image": np.zeros((S, S, 3), np.uint8),
            "observation/state": state,
            "prompt": self.prompt,
        })
        act = np.asarray(out["actions"], np.float32)[: self.n_action_steps]
        assert act.shape[-1] >= 7, f"expected >=7 action dims, got {act.shape}"
        return act[:, :7]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--prompt", required=True,
                    help='one of the 6 trained phrasings, e.g. "pick up the mushroom gently" or '
                         'the generic "pick up the object from table gently"')
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--setup", type=Path, default=_PKG / "configs/setup/real_lab.yaml")
    ap.add_argument("--action-config", type=Path,
                    default=_PKG / "configs/action/abs_pose_euler_abs_gripper.yaml")
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--max-pos-step-m", type=float, default=None,
                    help="per-step position cap (safety); recommended for a first real run")
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--shard-size", type=int, default=10)
    ap.add_argument("--gripper-offset-m", type=float, default=0.0)
    args = ap.parse_args()

    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.real_backend import RealBackend
    from gentle_manip.perception.obs_config import ObsConfig

    setup = _load_yaml(_resolve_config(args.setup))
    act_cfg = ActionConfig.from_dict(_load_yaml(_resolve_config(args.action_config)))
    # RGB only: no point cloud is needed or used by a pi0.5 policy.
    obs_cfg = ObsConfig.from_dict({"images": {"cameras": ["cam_ext"]}})
    backend = RealBackend(setup)
    # PolicyEnv REQUIRES rgb_shape once the obs config carries `images:` — take it from the
    # setup's camera entry (same as demos/record.py and deploy_real_dppo.py).
    _cam = setup["cameras"][obs_cfg.images.cameras[0]]
    rgb_shape = (int(_cam.get("height", 480)), int(_cam.get("width", 640)))
    env = PolicyEnv(backend=backend, obs_config=obs_cfg, action_config=act_cfg, task=None,
                    rgb_shape=rgb_shape)
    policy = Pi05RealPolicy(args.checkpoint, prompt=args.prompt, config_name=args.config_name,
                            repo_id=args.repo_id, n_action_steps=args.n_action_steps)
    run_deploy_loop(env, policy, max_steps=args.max_steps, rate=args.rate,
                    record_path=args.record, shard_size=args.shard_size,
                    action_config=act_cfg, max_pos_step_m=args.max_pos_step_m,
                    gripper_offset_m=args.gripper_offset_m)


if __name__ == "__main__":
    main()
