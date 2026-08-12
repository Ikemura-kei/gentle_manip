"""RLDG-style specialist-rollout collection: run a TRAINED solo specialist checkpoint
in its own sim, keep only the SUCCESSFUL episodes, write them out in the canonical
demo pickle schema so they can feed straight into convert_demos.py /
merge_cross_category_demos.py exactly like a raw CMA-ES demo collection would.

Rationale (docs/cross_category_specialist_log.md, 2026-08-11): neither an unconditioned
merge nor two different category-conditioning embeddings fixed cross-category zero-shot
generalization when merging RAW CMA-ES demos. Per RLDG (arXiv 2412.09858), the fix that
worked there was changing the TRAINING DATA -- distill from ROLLOUTS of working
specialists, not from the original demo pickles. This module is the collection half of
that: `EvalHarnessAgent`'s sibling for "run this checkpoint and harvest its own good
behavior" instead of "score this checkpoint."

Runs in envs/dppo via gentle_manip.dppo.train (hydra _target_), same as
EvalHarnessAgent -- reuses EvalAgent's venv+model construction, just swaps run_eval's
scoring loop for a save-good-episodes loop. Genesis-free at import time (only touches
the sim through the already-built self.venv).
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from agent.eval.eval_agent import EvalAgent

from gentle_manip.evaluation.eval_spec import EvalSpec

SHARD_SIZE = 10   # episodes per incremental flush -- see _write_shard/_merge_shards


def _write_shard(out_dir: Path, episodes: list, meta: dict, idx: int) -> Path:
    meta = dict(meta)
    meta["action_dim"] = int(episodes[0]["actions"].shape[1])
    path = out_dir / f"shard_{idx:04d}.pkl"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"meta": meta, "episodes": episodes}, f)
    os.replace(tmp, path)
    return path


def _count_existing_episodes(out_dir: Path) -> Tuple[int, int]:
    """(episodes already saved, next free shard index) -- gentle_manip 25-category
    speed/scale pass (2026-08-13), same crash-recovery pattern as
    collect_demos_synth_v2.py's _count_existing_episodes."""
    total = 0
    data_path = out_dir / "data.pkl"
    if data_path.exists():
        with open(data_path, "rb") as f:
            total += len(pickle.load(f)["episodes"])
    max_idx = -1
    for p in sorted(out_dir.glob("shard_*.pkl")):
        with open(p, "rb") as f:
            total += len(pickle.load(f)["episodes"])
        max_idx = max(max_idx, int(p.stem.split("_")[1]))
    return total, max_idx + 1


def _merge_shards(out_dir: Path, meta: dict) -> Optional[Path]:
    """Merge shard_*.pkl (+ a pre-existing data.pkl, if resuming a previously-
    complete run to top it up further) into data.pkl."""
    shards = sorted(out_dir.glob("shard_*.pkl"))
    prior_path = out_dir / "data.pkl"
    if not shards and not prior_path.exists():
        return None
    all_eps: list = []
    if prior_path.exists():
        with open(prior_path, "rb") as f:
            all_eps.extend(pickle.load(f)["episodes"])
    for p in shards:
        with open(p, "rb") as f:
            all_eps.extend(pickle.load(f)["episodes"])
    meta = dict(meta)
    meta["n_episodes"] = len(all_eps)
    if all_eps:
        meta["action_dim"] = int(all_eps[0]["actions"].shape[1])
    out = out_dir / "data.pkl"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"meta": meta, "episodes": all_eps}, f)
    os.replace(tmp, out)
    for p in shards:
        p.unlink()
    return out


class _DiffusionPolicy:
    """Identical to eval_agent.py's adapter -- kept local to avoid a cross-module import
    into an eval-specific file for what's conceptually a sibling, not a dependent."""

    def __init__(self, model, obs_keys, device, act_steps):
        self.model = model.eval()
        self.obs_keys = list(obs_keys)
        self.device = device
        self.act_steps = int(act_steps)

    def reset(self):
        pass

    def act(self, obs):
        with torch.no_grad():
            cond = {k: torch.from_numpy(np.asarray(obs[k])).float().to(self.device)
                    for k in self.obs_keys}
            traj = self.model(cond=cond, deterministic=True).trajectories.cpu().numpy()
        return traj[:, : self.act_steps]


def _process_raw_episode(steps: list) -> dict | None:
    """One env's raw per-physical-step dicts (from GenesisMultiStepVecEnv.drain_raw_episodes)
    -> a demo-pickle-schema episode, truncated right after the FIRST success (the object has
    already been successfully held by then -- the task's own is_success already requires the
    hold duration, so trimming here just drops the idle post-success tail, keeping episode
    length comparable to the original CMA-ES demos instead of running to max_episode_steps).
    Returns None for an episode that never succeeded (discarded, not a good demonstration)."""
    succ_idx = next((i for i, s in enumerate(steps) if s["success"]), None)
    if succ_idx is None:
        return None
    kept = steps[: succ_idx + 1]
    keys = [k for k in kept[0].keys() if k not in ("action", "success")]
    observations = {k: np.stack([s[k] for s in kept]).astype(np.float32) for k in keys}
    actions = np.stack([s["action"] for s in kept]).astype(np.float32)
    rewards = np.zeros(len(kept), np.float32)  # BC doesn't need reward; kept for schema parity
    return {"observations": observations, "actions": actions, "rewards": rewards}


class RolloutCollectorAgent(EvalAgent):
    """Drives self.venv (built record_raw=True by the eval-style cfg.env.specific) through
    batches of rollouts, keeping successful episodes, until --target-episodes are collected
    or --max-batches is exhausted. Writes dataset/demos/single_lift_<category>_rigid/
    <date>-rollout-<id>/data.pkl in the exact schema convert_demos.py expects."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.obs_keys = list(cfg.shape_meta.obs.keys())   # ["state", "point_cloud"] -- the
                                                            # policy's cond dict keys (same
                                                            # pattern as EvalHarnessAgent)

    def run(self):
        spec = EvalSpec(
            n_episodes=self.n_envs * int(self.cfg.get("max_batches", 60)),
            num_envs=self.n_envs,
            seed=int(self.cfg.get("seed", 1000)),   # DIFFERENT base seed than canonical eval's
                                                       # 0 -- non-overlapping scenario coverage,
                                                       # so rollout data isn't just a replay of
                                                       # the 100 canonical eval scenarios.
            max_policy_steps=int(self.cfg.env.max_episode_steps) // self.act_steps,
            scene_group_size=int(self.cfg.get("scene_group_size", 0)),
        )
        target_n = int(self.cfg.get("target_episodes", 150))
        policy = _DiffusionPolicy(self.model, self.obs_keys, self.device, self.act_steps)

        # Per-trajectory video, same convention as the canonical eval harness (run_eval):
        # ONE clip per episode attempt (success AND failure -- failures are valuable too,
        # for inspecting collection quality), render/batchNN_envM.mp4 under this run's logdir.
        render_dir = Path(self.logdir) / "render"
        save_video = bool(self.cfg.env.get("save_video", True))
        if save_video:
            render_dir.mkdir(parents=True, exist_ok=True)

        # gentle_manip 25-category speed/scale pass (2026-08-13): resolve out_dir
        # and flush shards incrementally instead of writing ONE atomic data.pkl
        # only at the very end -- previously a kill mid-run persisted NOTHING.
        # cfg.resume_dir (optional) continues a partially-collected prior run.
        resume_dir = self.cfg.get("resume_dir", None)
        out_dir = Path(resume_dir) if resume_dir else (
            Path(self.cfg.get("out_dir")) if self.cfg.get("out_dir") else self._default_out_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        already_saved, shard_idx = _count_existing_episodes(out_dir) if resume_dir else (0, 0)
        if resume_dir:
            print(f"[rollout] RESUME → {out_dir} ({already_saved} episode(s) already saved)",
                  flush=True)
        meta = dict(task=self.cfg.env_name, obs_keys=list(self.venv.obs_keys) + (
                    [self.venv.pointcloud_key] if self.venv.pointcloud_key else []),
                    created=datetime.now().isoformat(), source="rldg_rollout",
                    checkpoint=str(self.cfg.base_policy_path))

        saved: list = []
        attempts = 0
        for i in range(spec.n_batches):
            if spec.scene_group_size > 0 and i % spec.scene_group_size == 0 and hasattr(self.venv, "randomize_scene"):
                self.venv.randomize_scene(spec.scene_seed_for_group(i // spec.scene_group_size))
            self.venv.seed([spec.seed_for_batch(i)] * self.n_envs)
            options = ([{"video_path": str(render_dir / f"batch{i:02d}_env{j}.mp4")}
                       for j in range(self.n_envs)] if save_video else None)
            obs = self.venv.reset_arg(options)
            policy.reset()
            info = {}
            for _t in range(spec.max_policy_steps):
                action = policy.act(obs)
                obs, _reward, _terminated, truncated, info = self.venv.step(action)
                if bool(np.asarray(truncated).all()):
                    break
            attempts += self.n_envs
            raw_eps = info.get("raw_episodes", [[] for _ in range(self.n_envs)])
            for steps in raw_eps:
                ep = _process_raw_episode(steps)
                if ep is not None:
                    saved.append(ep)
            total_saved = already_saved + len(saved)
            print(f"[rollout] batch {i + 1}/{spec.n_batches}: {total_saved}/{target_n} kept "
                  f"({100 * (already_saved + len(saved)) / max(attempts, 1):.1f}% this run)",
                  flush=True)
            if len(saved) >= SHARD_SIZE:
                _write_shard(out_dir, saved, meta, shard_idx)
                shard_idx += 1
                saved = []
            if total_saved >= target_n:
                break

        if saved:
            _write_shard(out_dir, saved, meta, shard_idx)

        merged = _merge_shards(out_dir, meta)
        if merged is None:
            raise RuntimeError(f"collected 0 successful rollouts after {attempts} attempts -- "
                               f"checkpoint may be broken or target_episodes/max_batches too low")
        with open(merged, "rb") as f:
            final_n = len(pickle.load(f)["episodes"])
        print(f"[rollout] DONE — {final_n} successful episodes total / {attempts} attempts "
              f"this run ({100 * (final_n - already_saved) / max(attempts, 1):.1f}%) -> {merged}",
              flush=True)

    def _default_out_dir(self) -> Path:
        # env_name like "single_lift_apple_rigid_easy_pcd" -> "single_lift_apple_rigid"
        # (find_latest_data_pkl's expected dataset/demos/single_lift_<category>_rigid/ layout).
        category_dir = self.env_name.replace("_easy_pcd", "").replace("_pcd", "")
        ts = datetime.now().strftime("%y-%m-%d")
        import random, string
        suffix = "".join(random.choices(string.ascii_lowercase, k=3))
        return Path("dataset/demos") / category_dir / f"{ts}-rollout-{suffix}"
