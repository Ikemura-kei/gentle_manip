"""Batched scripted demo collection for the mushroom lift (soft) — N parallel envs, success-only.

Collects successful pick-lift demonstrations with WIDE state coverage: the eval object pose/
orientation/material DR (food_shape) PLUS an enlarged robot-pose box applied HERE (collection-
only, NOT the shared DR). Uses the eval sim fidelity (sim_substeps / mpm_grid_density from the
single_lift_mushroom_soft task cfg = 210 / 250) and the batched scripted expert (ScriptedPolicy,
one ScriptedLiftDemonstrator per sub-env). Records the SUPERSET obs (state + point cloud +
privileged + per-step reward) so the demos convert to DP3 or DPPO later.

Synchronous batches: run N envs to a fixed horizon, keep only the DONE (lifted+held) ones,
re-randomize, repeat until --n-demos successes. Saved incrementally (interrupt-safe).

    MUJOCO_GL=egl uv run --project envs/sim python examples/collect_mushroom_demos_batched.py \
        --n-demos 300 --n-envs 5 --pose-box 0.15 0.15 0.10 --scene-dr-every 20
"""
from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
_CFG = _REPO / "gentle_manip" / "configs"

from gentle_manip.envs.policy_env import PolicyEnv                    # noqa: E402
from gentle_manip.envs.sim_backend import SimBackend                 # noqa: E402
from gentle_manip.experiment import Experiment                       # noqa: E402
from gentle_manip.perception.obs_config import ObsConfig             # noqa: E402
from gentle_manip.tasks.single_lift import SingleLiftTask            # noqa: E402
from gentle_manip.demos.scripted_policy import ScriptedLiftDemonstrator  # noqa: E402
from gentle_manip.scripts.eval_scripted import ScriptedPolicy        # noqa: E402


class _SceneView:
    """Lets ScriptedPolicy read the current scene's object scale (grasp size-adaptation)."""
    def __init__(self, backend):
        self.backend = backend

    def scenario_params(self):
        return {"scene": self.backend.scene_params()}


def _write_shard(path: Path, episodes: list, meta: dict) -> None:
    """Atomically write one shard pkl ({meta, episodes}); each shard is self-contained."""
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"meta": {**meta, "shard": path.stem, "n_episodes": len(episodes)},
                     "episodes": episodes}, f)
    tmp.replace(path)


def _flush_shards(out_dir: Path, demos: list, shard_size: int, shards_done: int, meta: dict) -> int:
    """(Re)write only the not-yet-frozen shards (the trailing partial + any newly-crossed),
    so each write touches at most ~shard_size trajectories — no growing-pkl rewrite. Returns
    the new count of FULLY-written (frozen) shards."""
    n = len(demos)
    n_shards = (n + shard_size - 1) // shard_size
    for k in range(shards_done, n_shards):
        lo, hi = k * shard_size, min((k + 1) * shard_size, n)
        _write_shard(out_dir / f"shard_{k:04d}.pkl", demos[lo:hi], meta)
    return n // shard_size                                          # freeze completed shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-demos", type=int, default=300)
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--pose-box", type=float, nargs=3, default=[0.15, 0.15, 0.10],
                    help="per-axis HALF-range (m) for the initial robot EE pose box (x y z); "
                         "0.15 0.15 0.10 = a 30x30x20 cm box around the home pose")
    ap.add_argument("--experiment", default="single_lift_mushroom_soft")
    ap.add_argument("--obs", default="superset_soft")
    ap.add_argument("--collect-config", type=Path,
                    default=_CFG / "collect/single_lift_mushroom_soft_scripted.yaml")
    ap.add_argument("--scene-dr-every", type=int, default=20,
                    help="rebuild object geometry/material every N batches (subprocess backend); 0=fixed nominal")
    ap.add_argument("--max-steps", type=int, default=320, help="policy steps per attempt (episode horizon)")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None, help="run dir for the shard pkls")
    ap.add_argument("--shard-size", type=int, default=20,
                    help="trajectories per shard pkl (keeps each incremental write small)")
    args = ap.parse_args()

    exp = Experiment.load(args.experiment)
    task = SingleLiftTask(exp.task_cfg)                              # 210/250, reward + success
    # Effective sim fidelity actually used (GM_SIM_SUBSTEPS / GM_MPM_SAMPLER override the task cfg)
    # so the recorded config/meta is HONEST when we run at the eval fidelity (235 + regular).
    eff_substeps = int(os.environ.get("GM_SIM_SUBSTEPS") or task.sim_substeps)
    mpm_sampler = os.environ.get("GM_MPM_SAMPLER") or "genesis-default"
    obs_cfg = ObsConfig.from_dict(yaml.safe_load((_CFG / "obs" / f"{args.obs}.yaml").read_text()))
    act_cfg = exp.action_config
    cc = yaml.safe_load(args.collect_config.read_text())
    sc = cc.get("scripted", {})
    params = dict(lift_height=sc.get("lift_height", 0.2), hold_seconds=sc.get("hold_seconds", 2.0),
                  approach_height=sc.get("approach_height", 0.12), grasp_z=sc.get("grasp_z", 0.006),
                  grasp_gw=sc.get("grasp_gw", 0.030), grasp_firm_steps=sc.get("grasp_firm_steps", 1),
                  gripper_close=cc.get("gripper_value", 0.5), speed_cap=cc.get("speed", 0.5))

    use_sub = args.scene_dr_every > 0                               # geometry DR needs relaunch
    backend = SimBackend(task.scene_spec, num_envs=args.n_envs, use_subprocess=use_sub,
                         config={"seed": args.seed,        # seeds the OBJECT pose/orientation/scene-DR
                                                           # RNG too (not just home offsets) -> a new
                                                           # --seed gives genuinely new initial conditions
                                 "sim": {"settle_steps": int(cc.get("settle_steps", 30)),
                                         "scene_dr_every": args.scene_dr_every},
                                 "dr": exp.dr})
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=task, max_episode_steps=10 ** 9)
    scripted = ScriptedPolicy(args.n_envs, act_cfg.scales, args.rate, params, venv=_SceneView(backend))

    out_dir = args.out_dir or (_REPO / "dataset" / "demos" / "mushroom_soft_batched" /
                               datetime.now().strftime("%y-%m-%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Self-document the run so it's reproducible without gentle_manip/dppo/demo_convert.sh:
    # the exact launch command + the resolved config (args + scripted params + sim fidelity).
    from gentle_manip.utils.run_paths import save_launch_command
    save_launch_command(out_dir)                                    # -> <run>/launch_command.sh
    (out_dir / "config.yaml").write_text(yaml.safe_dump({
        "source": "collect_mushroom_demos_batched", "experiment": args.experiment,
        "obs": args.obs, "collect_config": str(args.collect_config), "n_demos": args.n_demos,
        "n_envs": args.n_envs, "pose_box_halfrange_m": list(args.pose_box),
        "scene_dr_every": args.scene_dr_every, "max_steps": args.max_steps, "rate": args.rate,
        "seed": args.seed, "shard_size": args.shard_size,
        "sim_substeps": eff_substeps, "mpm_grid_density": task.mpm_grid_density, "mpm_sampler": mpm_sampler,
        "scripted_params": {k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in params.items()}}, sort_keys=False))

    box = np.array(args.pose_box, np.float32)
    rng = np.random.default_rng(args.seed)
    DONE = ScriptedLiftDemonstrator.DONE

    def _meta():
        return {"task": args.experiment,
                "obs_keys": sorted(demos[0]["observations"].keys()) if demos else [],
                "action_dim": int(demos[0]["actions"].shape[1]) if demos else 0,
                "rate_hz": args.rate, "n_envs": args.n_envs, "shard_size": args.shard_size,
                "pose_box_halfrange_m": list(args.pose_box), "scene_dr_every": args.scene_dr_every,
                "sim_substeps": eff_substeps, "mpm_grid_density": task.mpm_grid_density, "mpm_sampler": mpm_sampler,
                "source": "collect_mushroom_demos_batched",
                "created": datetime.now(timezone.utc).isoformat()}

    demos, attempts, batch, shards_done = [], 0, 0, 0
    print(f"collecting {args.n_demos} demos | {args.n_envs} envs | pose_box +/-{box.tolist()} m | "
          f"scene_dr_every={args.scene_dr_every} | obs={args.obs} sub={use_sub} | "
          f"shard_size={args.shard_size} -> {out_dir}", flush=True)
    while len(demos) < args.n_demos:
        home = rng.uniform(-box, box, (args.n_envs, 3)).astype(np.float32)
        obs = env.reset(home_offset=home)                           # object pose/orientation from DR
        scripted.reset()
        ob = [[] for _ in range(args.n_envs)]
        ab = [[] for _ in range(args.n_envs)]
        rb = [[] for _ in range(args.n_envs)]
        done_step = [None] * args.n_envs
        for t in range(args.max_steps):
            acts = scripted.act(obs)                                # (n_envs, 7) normalized [-1,1]
            for j in range(args.n_envs):
                ob[j].append({k: np.asarray(v)[j].copy() for k, v in obs.items()})
                ab[j].append(acts[j].astype(np.float32).copy())
            obs, rewards, dones, infos = env.step(acts)
            for j in range(args.n_envs):
                rb[j].append(float(np.asarray(rewards)[j]))
                if done_step[j] is None and scripted.demos[j].phase == DONE:
                    done_step[j] = t
        attempts += args.n_envs
        batch += 1
        n_new = 0
        for j in range(args.n_envs):
            if scripted.demos[j].phase == DONE:                     # success = lifted + held
                T = (done_step[j] + 1) if done_step[j] is not None else len(ab[j])
                demos.append({
                    "observations": {k: np.stack([o[k] for o in ob[j][:T]]) for k in ob[j][0]},
                    "actions": np.stack(ab[j][:T]),
                    "rewards": np.array(rb[j][:T], np.float32)})
                n_new += 1
        shards_done = _flush_shards(out_dir, demos, args.shard_size, shards_done, _meta())
        print(f"  batch {batch}: +{n_new}/{args.n_envs} | {len(demos)}/{args.n_demos} demos | "
              f"success {len(demos)}/{attempts} = {len(demos)/attempts:.0%}", flush=True)
    env.close()
    n_shards = (len(demos) + args.shard_size - 1) // args.shard_size
    print(f"DONE -> {out_dir} ({len(demos)} demos in {n_shards} shards)")


if __name__ == "__main__":
    main()
