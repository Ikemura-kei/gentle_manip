"""Canonical-harness evaluation of a fine-tuned π0.5 checkpoint (apples-to-apples with DPPO).

CLAUDE.md hard requirement #1: every sim eval goes through `gentle_manip.evaluation.run_eval`
with the fixed EvalSpec, over the SAME `GenesisMultiStepVecEnv` + `serl_sim_server` bridge every
DPPO eval uses. Modelled directly on `gentle_manip/scripts/eval_dp3_harness.py`, which solves the
identical problem for DP3.

TWO ADAPTATIONS, both on OUR side (openpi is unmodified):

1. IDENTITY NORMALIZATION, exactly as the DP3 harness does it. openpi normalizes obs/actions
   internally from its own norm_stats, so the venv must not normalize on top. With
   obs_min=action_min=-1 and obs_max=action_max=+1 the venv's `2*(x-min)/(max-min+1e-6)-1` and its
   inverse become no-ops to ~5e-7. The policy therefore sees RAW proprio and emits actions in the
   ActionPipeline's own [-1,1] space -- which is exactly the space `derive_action_set` produced for
   the LeRobot dataset, so train and eval agree.

2. IMAGES IN THE OBS. `GenesisMultiStepVecEnv._modalities` returns state (+ point cloud); π0.5
   needs the RGB streams. The subclass below adds them, passed through raw (uint8) -- resizing is
   the policy's job, using openpi's own `resize_with_pad`, so training and inference letterbox
   identically.

The sim server MUST serve an experiment whose obs config includes the images
(`single_lift_mushroom_soft_pi05`) and must run with `--render-rgb`, else the image keys are absent
and this fails loudly at the first reset rather than silently evaluating on a blank view.

Run inside third_party/openpi's env with PYTHONPATH=<repo>:
    uv run python <repo>/gentle_manip/pi05/eval_harness.py \
        --checkpoint <.../checkpoints/pi05_libero/<exp>/<step>> \
        --experiment single_lift_mushroom_soft_pi05 --port 5570 [--no-wrist]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _identity_normalization(obs_dim: int, act_dim: int) -> Path:
    """A normalization.npz whose transforms are no-ops (the DP3 harness's trick)."""
    p = Path(tempfile.mkdtemp()) / "normalization.npz"
    np.savez(p,
             obs_min=-np.ones(obs_dim, np.float32), obs_max=np.ones(obs_dim, np.float32),
             action_min=-np.ones(act_dim, np.float32), action_max=np.ones(act_dim, np.float32))
    return p


def _make_venv(num_envs, act_steps, max_policy_steps, port, image_keys):
    from gentle_manip.dppo.genesis_venv import GenesisMultiStepVecEnv, build_genesis_venv

    class _ImageVenv(GenesisMultiStepVecEnv):
        """Adds the RGB streams to the obs the Policy receives. Everything else is inherited,
        so scenario seeding, DR audit, video recording and success/stress reporting are the
        SAME code paths every other eval uses."""

        def _modalities(self, obs: dict) -> dict:
            m = super()._modalities(obs)
            for k in image_keys:
                if k not in obs:
                    raise KeyError(
                        f"{k} missing from the sim obs -- the server must serve an experiment "
                        f"whose obs config includes `images:` AND run with --render-rgb. "
                        f"Got: {sorted(obs)}")
                m[k] = np.asarray(obs[k], np.uint8)
            return m

    # PROPRIO_VIEW = [ee_pos, ee_quat, gripper_width] -> 8-dim, matching the LeRobot `state`.
    from gentle_manip.dppo.convert_demos import PROPRIO_VIEW
    norm = _identity_normalization(obs_dim=8, act_dim=7)
    venv = build_genesis_venv(
        num_envs=num_envs, obs_steps=1, act_steps=act_steps,
        max_episode_steps=max_policy_steps * act_steps, normalization_path=norm,
        obs_keys=PROPRIO_VIEW, pointcloud_key=None, port=port)
    venv.__class__ = _ImageVenv          # same instance, image-aware _modalities
    return venv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--experiment", default="single_lift_mushroom_soft_pi05")
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--port", type=int, default=5570)
    ap.add_argument("--no-wrist", action="store_true",
                    help="ext-only variant: feed ZEROS as the wrist image, matching how its "
                         "dataset was converted (train/eval must agree)")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--batched-infer", action="store_true",
                    help="one model call per policy step for ALL envs instead of one per env. "
                         "openpi's Policy.infer is hardcoded to batch 1; the model it wraps is "
                         "not. Verify with .agent_tmp/check_batched_infer.py before trusting.")
    ap.add_argument("--norm-stats-from", default=None,
                    help="ZERO-SHOT: assets dir (…/assets/<asset_id>) whose norm_stats.json to "
                         "use, for a base checkpoint that ships only `params`. See the caveat in "
                         "eval_policy.Pi05EvalPolicy — the resulting number is about ACTION-SPACE "
                         "TRANSFER, not about whether pi0.5 can do the task.")
    ap.add_argument("--repo-id", default=None,
                    help="LeRobot repo id the checkpoint was trained on; inferred from the "
                         "checkpoint's assets/ tree when omitted")
    ap.add_argument("--n-episodes", type=int, default=None, help="override for a smoke run")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--record-batches", type=int, default=None)
    ap.add_argument("--scene-group-size", type=int, default=1,
                    help="rebuild object GEOMETRY every K batches. 1 = a distinct geometry per "
                         "batch (what every width/adaptation claim requires: >=40 distinct "
                         "geometries). EvalSpec's own default is 0 = ONE fixed geometry for the "
                         "whole eval, which silently makes an eval a single-object measurement.")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    from gentle_manip.evaluation.eval_spec import EvalSpec
    from gentle_manip.evaluation.harness import run_eval
    from gentle_manip.pi05.eval_policy import DEFAULT_PROMPT, Pi05EvalPolicy

    policy = Pi05EvalPolicy(args.checkpoint, config_name=args.config_name,
                            prompt=args.prompt or DEFAULT_PROMPT,
                            use_wrist=not args.no_wrist, repo_id=args.repo_id,
                            norm_stats_from=args.norm_stats_from,
                            batched=args.batched_infer)

    # PROBE the real action horizon instead of hardcoding it: the venv executes exactly
    # act_steps actions per policy step, so a mismatch would silently drop or repeat commands.
    from openpi.policies.libero_policy import make_libero_example
    _probe = np.asarray(policy._policy.infer(make_libero_example())["actions"], np.float32)
    horizon, adim = (_probe.shape[0], _probe.shape[1]) if _probe.ndim == 2 else (1, _probe.shape[0])
    print(f"[eval] probed policy: action_horizon={horizon}, action_dim={adim}")
    assert adim == 7, f"policy emits {adim}-dim actions; the fixed setup is 7-dim (CHECKLISTS §0)"

    spec_kwargs = {}
    if args.n_episodes is not None:
        spec_kwargs["n_episodes"] = args.n_episodes
    if args.num_envs is not None:
        spec_kwargs["num_envs"] = args.num_envs
    spec_kwargs["scene_group_size"] = args.scene_group_size
    spec = EvalSpec(**spec_kwargs)
    print(f"[eval] n_episodes={spec.n_episodes} num_envs={spec.num_envs} "
          f"scene_group_size={spec.scene_group_size} -> "
          f"{spec.n_batches // max(spec.scene_group_size,1) if spec.scene_group_size else 1} "
          f"distinct geometries")

    image_keys = ["image_cam_ext", "image_cam_wrist"]
    venv = _make_venv(spec.num_envs, act_steps=horizon, max_policy_steps=spec.max_policy_steps,
                      port=args.port, image_keys=image_keys)

    out_dir = args.out_dir or (args.checkpoint.parent / "eval" /
                               __import__("datetime").datetime.now().strftime("%y-%m-%d-%H%M%S"))
    # NOTE: the width dump flushes on reset(), i.e. at the START of each batch, so the LAST
    # batch is never flushed by the loop -- and that is the largest-scale bin, the highest-leverage
    # point in the size regression. Flush explicitly after the run. (Observed: 4 dumps for 5
    # batches on the first probe pass.)
    res = run_eval(venv, policy, spec, out_dir, experiment_name=args.experiment,
                   checkpoint=str(args.checkpoint), record_batches=args.record_batches,
                   extra_meta={"policy": "pi0.5", "config_name": args.config_name,
                               "wrist": not args.no_wrist,
                               "prompt": args.prompt or DEFAULT_PROMPT})
    policy._flush_width_dump()          # final batch (see note above)
    print("summary:", {k: res.get(k) for k in
                       ("success_rate", "ever_success_rate", "stress_top20_ttop20_mean")})
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
