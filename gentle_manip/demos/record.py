from __future__ import annotations

import argparse
import os
import pickle
import random
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

import gentle_manip
from gentle_manip.demos.keyboard_pygame import DISCARD, QUIT, SAVE, PygameKeyboard
from gentle_manip.demos.teleop_spacemouse import SpaceMouseTeleop
from gentle_manip.actions.action_config import ActionConfig
from gentle_manip.actions.pipeline import invert_absolute_action

# Teleop demonstration collection on the real robot. The SpaceMouse command is a
# normalized [-1,1] raw_action — the same thing a policy outputs — so it flows
# through the env's ActionPipeline, and recorded obs are exactly what a policy
# will see. Runs in the 3.11 deploy env:
#   uv run --project envs/deploy python -m gentle_manip.demos.record ...

_PKG = Path(gentle_manip.__file__).parent


class DemoRecorder:
    """Drives a PolicyEnv with a teleop device and records (obs, action) episodes.

    env / teleop / keyboard are injected so the loop is testable without hardware
    or a display. Episode boundaries come from keyboard edge events:
        SAVE → stack + keep the episode, re-home;  DISCARD → drop, re-home;
        QUIT → stop (in-progress buffer is dropped, like the old ESC).
    """

    def __init__(self, env, teleop, keyboard, task_name: str,
                 out_dir: Path, rate_hz: float = 20.0,
                 idle_threshold: float = 1e-3, keep_trailing_idle: int = 5,
                 max_interior_idle: int = 3, action_noise_std: float = 0.0,
                 dataset_path: Optional[Path] = None, frame_fn=None,
                 video_dir: Optional[Path] = None, video_fps: int = 20,
                 video_episodes: int = 0, video_failed_episodes: int = 0,
                 cloud_viewer=None, collection_config: Optional[dict] = None,
                 shard_size: int = 5,
                 record_action_config: Optional[ActionConfig] = None) -> None:
        # If set, the RECORDED action for each step is NOT the raw teleop action (which
        # drives the env — still delta, unchanged) but the equivalent ABSOLUTE action
        # (see invert_absolute_action) computed from the robot's ACTUAL resulting pose
        # after the step. Lets you teleop with delta control (smoother/easier) while
        # collecting a dataset directly usable to train an absolute-action policy — no
        # separate conversion pass needed. Must be an ActionConfig with mode="absolute";
        # only used for its pos_min/pos_max/gripper_min/gripper_max/clip fields (never
        # passed to the env, so it never affects how the robot is actually driven).
        self._record_action_config = record_action_config
        self.env = env
        self.teleop = teleop
        self.keyboard = keyboard
        self.task_name = task_name
        # Verbatim + resolved config snapshot (setup/obs/action + CLI knobs) written once
        # next to the dataset as <stem>_config.yaml, for reproducibility (mirrors
        # collect_demos_sim.py). None -> no snapshot (e.g. tests).
        self._collection_config = collection_config
        self._config_written = False
        # Optional live Open3D view of the PROCESSED cloud (LiveCloudViewer); fed
        # obs["point_cloud"] each step so you watch what the policy will see.
        self.cloud_viewer = cloud_viewer
        self.out_dir = Path(out_dir)
        # Optional RGB video: frame_fn() returns an (H, W, 3) uint8 frame; one mp4 is
        # written per episode for the first `video_episodes` saved (a quality check).
        self.frame_fn = frame_fn
        self.video_dir = Path(video_dir) if video_dir else None
        self.video_fps = int(video_fps)
        self.video_episodes = int(video_episodes)
        # Also record the first N DISCARDED (failed) episodes as ep_FAILED_NNN.mp4 for
        # diagnosis — otherwise a failed grasp leaves no trace to look at.
        self.video_failed_episodes = int(video_failed_episodes)
        self._failed_videos = 0
        self._frames: list = []
        # Gaussian noise (std, in normalized action units) added to the demonstrator's
        # POSE deltas (not gripper) each step — recorded and executed alike, so the demos
        # carry action variation for more robust training. 0.0 disables it.
        self.action_noise_std = float(action_noise_std)
        self._rng = np.random.default_rng()
        # Explicit output file (e.g. <run_dir>/data.pkl) overrides the default
        # <out_dir>/<task>/YY-MM-DD-xyz.pkl naming when the caller owns the run dir.
        self._dataset_path_override = Path(dataset_path) if dataset_path else None
        self.rate_hz = float(rate_hz)
        # Idle trim (online, at save time), per consecutive idle run:
        #   leading run  → drop entirely (don't teach "wait before starting")
        #   interior run → keep <= max_interior_idle (a long mid-task pause is
        #                  usually operator hesitation, not signal)
        #   trailing run → keep <= keep_trailing_idle (so the policy still learns
        #                  to stop once the target is reached)
        # A frame is idle if its full action norm (incl. gripper) <= idle_threshold.
        # idle_threshold <= 0 disables trimming entirely.
        self.idle_threshold = float(idle_threshold)
        self.keep_trailing_idle = int(keep_trailing_idle)
        self.max_interior_idle = int(max_interior_idle)

        self._obs_buf: List[Dict[str, np.ndarray]] = []
        self._act_buf: List[np.ndarray] = []
        # Always the RAW DELTA teleop action, even when record_action_config makes
        # _act_buf hold absolute actions instead -- idle detection (near-zero norm =
        # no movement) only makes sense for deltas; an absolute command stays a large
        # nonzero vector even while the operator is idling.
        self._idle_act_buf: List[np.ndarray] = []
        self._rew_buf: List[float] = []
        self.episodes: List[dict] = []
        # Shard-based writing: episodes are flushed in groups of shard_size to small
        # numbered files, then merged into data.pkl at end. Avoids the O(N²) rewrite
        # cost of atomically rewriting a growing single file after every episode.
        # shard_size=0 falls back to the old single-file atomic-rewrite behaviour.
        self._shard_size = int(shard_size)
        self._shard_buf: List[dict] = []    # episodes in the current (not yet flushed) shard
        self._shard_count = 0               # number of shard files written so far
        self._total_saved = 0              # total episodes saved across all shards
        self._run_dir_path: Optional[Path] = None  # lazily set on first save

    # ── Loop ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.teleop.open()
        self.keyboard.open()
        period = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.0
        print(f"recording '{self.task_name}' — SPACE=save  BACKSPACE=discard  ESC=quit")

        obs = self.env.reset()
        try:
            while True:
                t0 = time.perf_counter()
                events = self.keyboard.poll()

                if QUIT in events:
                    break
                if SAVE in events:
                    n = self._save_episode()
                    print(f"  saved episode {self._total_saved} ({n} steps) → {self._run_dir_path}")
                    self._flush_video()
                    obs = self.env.reset()
                    continue
                if "toggle_filter" in events:               # key M: live A/B of the ground_residual filter
                    pp = self.env.perception
                    if pp.cfg.point_cloud is not None and pp.cfg.point_cloud.gr_voxel_size is not None:
                        pp.ground_residual_enabled = not pp.ground_residual_enabled
                        print(f"  [filter] ground_residual {'ON' if pp.ground_residual_enabled else 'OFF'}", flush=True)
                    else:
                        print("  [filter] this obs config has no ground_residual block — nothing to toggle", flush=True)
                if DISCARD in events:
                    self._discard_episode()
                    print("  discarded episode")
                    if self._failed_videos < self.video_failed_episodes:
                        self._flush_video(label="FAILED", index=self._failed_videos)
                        self._failed_videos += 1
                    self._frames = []
                    obs = self.env.reset()
                    continue

                # Capture frames if EITHER budget (saved or failed) still has room.
                if self.frame_fn is not None and (len(self.episodes) < self.video_episodes
                                                  or self._failed_videos < self.video_failed_episodes):
                    self._frames.append(np.asarray(self.frame_fn(), dtype=np.uint8))

                action = np.asarray(self.teleop.get_action(), dtype=np.float32)
                if self.action_noise_std > 0:
                    # Noise on the pose deltas only (all but the last/gripper dim),
                    # clipped back to the [-1, 1] action range.
                    action[:-1] = np.clip(
                        action[:-1] + self._rng.normal(0.0, self.action_noise_std, action.shape[0] - 1),
                        -1.0, 1.0,
                    ).astype(np.float32)
                obs = self._step_and_record(obs, action)

                if self.cloud_viewer is not None and "point_cloud" in obs:
                    self.cloud_viewer.update(obs["point_cloud"][0])

                self._rate_limit(t0, period)
        finally:
            self.teleop.close()
            self.keyboard.close()
            if self.cloud_viewer is not None:
                self.cloud_viewer.close()
            self.env.close()

    def _flush_video(self, label: str = "", index: Optional[int] = None) -> None:
        """Write the captured frames of the just-ended episode to an mp4 (label = "" for a
        saved episode, "FAILED" for a discarded one)."""
        if not self._frames:
            return
        import imageio.v2 as imageio
        vdir = (self.video_dir or self._run_dir()) / "videos"   # default: beside the run's data.pkl
        vdir.mkdir(parents=True, exist_ok=True)
        idx = index if index is not None else len(self.episodes)
        name = f"ep_{label}_{idx:03d}.mp4" if label else f"ep_{idx:03d}.mp4"
        imageio.mimsave(str(vdir / name), self._frames, fps=self.video_fps, macro_block_size=1)
        print(f"  saved video → {vdir / name}")
        self._frames = []

    def _step_and_record(self, obs, action: np.ndarray):
        """Record (obs_t, action_t, reward_t), advance the env, return the next obs.

        The env's per-step reward is logged too (0 when the env has task=None). This
        makes a demo directly usable for reward-based RL (RLPD); reward-free consumers
        (DP3) simply ignore the "rewards" array.

        `action` is what actually drives the env (still delta, unchanged). If
        `record_action_config` is set, the SAVED action is instead the equivalent
        absolute command inverted from the robot's ACTUAL resulting pose (obs_next) —
        see invert_absolute_action / DemoRecorder.__init__."""
        obs_next, reward, _done, _info = self.env.step(action[None, :])
        self._obs_buf.append({k: np.asarray(v)[0] for k, v in obs.items()})  # drop num_envs
        self._idle_act_buf.append(action.copy())            # always the raw delta
        if self._record_action_config is not None:
            recorded = invert_absolute_action(
                obs_next["ee_pos"], obs_next["ee_quat"], obs_next["gripper_width"],
                self._record_action_config)[0]
        else:
            recorded = action.copy()
        self._act_buf.append(recorded)
        self._rew_buf.append(float(np.asarray(reward).ravel()[0]))
        return obs_next

    def _rate_limit(self, t0: float, period: float) -> None:
        if period <= 0:
            return
        dt = time.perf_counter() - t0
        if dt < period:
            time.sleep(period - dt)
        elif dt > period * 1.5:
            print(f"  [warn] tick {dt * 1000:.0f} ms > budget {period * 1000:.0f} ms "
                  f"(can't hold {self.rate_hz:.0f} Hz)")

    # ── Episode buffer ────────────────────────────────────────────────────────

    def _save_episode(self) -> int:
        obs_buf, act_buf, rew_buf = self._trim_idle(
            self._obs_buf, self._act_buf, self._idle_act_buf, self._rew_buf)
        n = len(act_buf)
        if n == 0:
            self._clear()
            return 0
        keys = obs_buf[0].keys()
        episode = {
            "observations": {k: np.stack([o[k] for o in obs_buf]) for k in keys},
            "actions": np.stack(act_buf),
            "rewards": np.asarray(rew_buf, dtype=np.float32),
        }
        self.episodes.append(episode)
        self._total_saved += 1
        self._run_dir()                        # ensure the run directory exists
        self._clear()
        if self._shard_size > 0:
            self._shard_buf.append(episode)
            if len(self._shard_buf) >= self._shard_size:
                self._flush_shard()
        else:
            self._flush()                      # single-file mode: rewrite whole file
        return n

    def _trim_idle(self, obs_buf, act_buf, idle_act_buf, rew_buf):
        """Cap each consecutive idle run by position (leading/interior/trailing).

        Idle = full RAW DELTA action norm <= idle_threshold (idle_act_buf — always
        delta, even when act_buf holds inverted absolute actions instead; see
        DemoRecorder.__init__). Keeps the first `cap` frames of each idle run (0
        leading, max_interior_idle interior, keep_trailing_idle trailing).
        idle_threshold <= 0 disables trimming.
        """
        if self.idle_threshold <= 0 or not act_buf:
            return obs_buf, act_buf, rew_buf
        idle = np.linalg.norm(np.stack(idle_act_buf), axis=1) <= self.idle_threshold
        T = len(act_buf)
        if idle.all():
            return [], [], []                  # whole episode idle → nothing to keep

        keep = np.ones(T, dtype=bool)
        i = 0
        while i < T:
            if not idle[i]:
                i += 1
                continue
            j = i
            while j < T and idle[j]:            # idle run [i, j)
                j += 1
            if i == 0:
                cap = 0                         # leading
            elif j == T:
                cap = self.keep_trailing_idle   # trailing
            else:
                cap = self.max_interior_idle    # interior
            keep[i + cap:j] = False
            i = j

        obs_out = [o for o, k in zip(obs_buf, keep) if k]
        act_out = [a for a, k in zip(act_buf, keep) if k]
        rew_out = [r for r, k in zip(rew_buf, keep) if k]
        # RGB frames (--record-rgb) are captured once per tick, exactly like the buffers —
        # apply the SAME mask or the saved video runs ahead of the saved steps wherever the
        # operator paused (leading idle is dropped entirely, so the video started seconds early).
        if self._frames:
            if len(self._frames) == T:
                self._frames = [f for f, k in zip(self._frames, keep) if k]
            else:  # defensive: never let a count mismatch silently desync — trim to min, warn
                print(f"  [warn] frame/step count mismatch ({len(self._frames)} vs {T}); "
                      f"masking best-effort")
                m = min(len(self._frames), T)
                self._frames = [f for f, k in zip(self._frames[:m], keep[:m]) if k]
        return obs_out, act_out, rew_out

    def _discard_episode(self) -> None:
        self._clear()
        # Drop captured RGB frames too (the SAVE path flushes them AFTER _clear, so the wipe
        # must live here, not in _clear): without this, a DISCARDED episode's frames stayed
        # buffered and PREPENDED the next episode's video (ghost prefix -> desynced videos,
        # found on the cube3 probe recordings).
        self._frames = []

    def _clear(self) -> None:
        self._obs_buf = []
        self._act_buf = []
        self._idle_act_buf = []
        self._rew_buf = []

    # ── Output ────────────────────────────────────────────────────────────────

    def _run_dir(self) -> Path:
        """The session run directory. Created lazily on first use.

        Either the parent of the caller-supplied dataset_path override, or a fresh
        <out>/<task>/YY-MM-DD-xyz/ directory (matching sim demo collection layout).
        Data files (data.pkl, shard_*.pkl, config.yaml) all live inside this dir.
        """
        if self._run_dir_path is None:
            if self._dataset_path_override is not None:
                d = self._dataset_path_override.parent
                d.mkdir(parents=True, exist_ok=True)
                self._run_dir_path = d
            else:
                out = self.out_dir / self.task_name
                out.mkdir(parents=True, exist_ok=True)
                date = datetime.now().strftime("%y-%m-%d")
                for _ in range(10000):
                    sfx = "".join(random.choices(string.ascii_lowercase, k=3))
                    cand = out / f"{date}-{sfx}"
                    if not cand.exists():
                        cand.mkdir()
                        self._run_dir_path = cand
                        break
                else:
                    raise RuntimeError(f"no free run dir in {out}")
        return self._run_dir_path

    def _flush(self) -> None:
        """Single-file mode: atomically rewrite the full dataset after every save.

        Used only when shard_size=0. Writes to run_dir/data.pkl (or the override path
        when dataset_path was supplied by the caller).
        """
        if not self.episodes:
            return
        run_dir = self._run_dir()
        data_path = self._dataset_path_override if self._dataset_path_override else run_dir / "data.pkl"
        dataset = {
            "meta": {
                "task": self.task_name,
                "obs_keys": sorted(self.episodes[0]["observations"].keys()),
                "action_dim": int(self.episodes[0]["actions"].shape[1]),
                "rate_hz": self.rate_hz,
                "created": datetime.now(timezone.utc).isoformat(),
            },
            "episodes": self.episodes,
        }
        tmp = data_path.with_name(data_path.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(dataset, f)
        os.replace(tmp, data_path)
        self._write_config_snapshot()

    def _flush_shard(self) -> None:
        """Shard mode: write the current shard buffer to shard_NNNN.pkl and clear it."""
        if not self._shard_buf:
            return
        run_dir = self._run_dir()
        shard_path = run_dir / f"shard_{self._shard_count:04d}.pkl"
        tmp = shard_path.with_suffix(".tmp")
        first_ep = self._shard_buf[0]
        data = {
            "meta": {
                "task": self.task_name,
                "obs_keys": sorted(first_ep["observations"].keys()),
                "action_dim": int(first_ep["actions"].shape[1]),
                "rate_hz": self.rate_hz,
            },
            "episodes": self._shard_buf,
        }
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp, shard_path)
        self._shard_count += 1
        self._shard_buf = []
        self._write_config_snapshot()

    def _merge_shards(self) -> Optional[Path]:
        """Merge all shard_*.pkl into data.pkl and remove the shards."""
        run_dir = self._run_dir()
        shards = sorted(run_dir.glob("shard_*.pkl"))
        if not shards:
            return None
        all_episodes = []
        meta = None
        for p in shards:
            with open(p, "rb") as f:
                d = pickle.load(f)
            if meta is None:
                meta = dict(d["meta"])
            all_episodes.extend(d["episodes"])
        meta["n_episodes"] = len(all_episodes)
        meta["created"] = datetime.now(timezone.utc).isoformat()
        data_path = run_dir / "data.pkl"
        tmp = data_path.with_name("data.pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"meta": meta, "episodes": all_episodes}, f)
        os.replace(tmp, data_path)
        for p in shards:
            p.unlink()
        return data_path

    def _write_config_snapshot(self) -> None:
        """Write the collection config to run_dir/config.yaml (once)."""
        if self._collection_config is None or self._config_written:
            return
        cfg_path = self._run_dir() / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(self._collection_config, f, sort_keys=False)
        self._config_written = True
        print(f"  wrote config snapshot → {cfg_path}")

    def write(self) -> Optional[Path]:
        """Flush remaining data, merge shards (if any), and report the final path."""
        if self._shard_size > 0:
            if self._shard_buf:
                self._flush_shard()
            if self._total_saved == 0:
                print("no episodes recorded — nothing written")
                return None
            data_path = self._merge_shards()
            print(f"saved {self._total_saved} episodes → {data_path}")
            return data_path
        else:
            if not self.episodes:
                print("no episodes recorded — nothing written")
                return None
            self._flush()
            run_dir = self._run_dir()
            data_path = self._dataset_path_override if self._dataset_path_override else run_dir / "data.pkl"
            print(f"saved {len(self.episodes)} episodes → {data_path}")
            return data_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _git_commit() -> str:
    """Short current commit for the config snapshot; '' if not a repo / git missing."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_PKG),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _resolve_config(path: Path) -> Path:
    """Find a config path regardless of cwd.

    Tries the path as given, then relative to the repo root, so repo-root-relative
    args like 'gentle_manip/configs/obs/state_ee_only.yaml' work from anywhere.
    """
    if path.is_file():
        return path
    alt = _PKG.parent / path
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"config not found: {path} (also tried {alt})")


def main() -> None:
    p = argparse.ArgumentParser(description="Real-robot SpaceMouse demo collection")
    p.add_argument("--setup", type=Path,
                   default=_PKG / "configs" / "setup" / "real_lab.yaml")
    p.add_argument("--obs-config", type=Path,
                   default=_PKG / "configs" / "obs" / "superset_real.yaml")
    p.add_argument("--action-config", type=Path,
                   default=_PKG / "configs" / "action" / "abs_pose_abs_gripper.yaml")
    p.add_argument("--record-action-config", type=Path, default=None,
                   help="an ABSOLUTE-mode action config (e.g. abs_pose_abs_gripper.yaml) — "
                        "if set, teleop still drives the robot via --action-config (delta, "
                        "smoother to operate) but the SAVED action is the equivalent absolute "
                        "command inverted from the robot's actual resulting pose each step, so "
                        "the dataset is directly usable to train an absolute-action policy with "
                        "no separate conversion pass. Not used to drive the robot.")
    p.add_argument("--task-name", required=True)
    p.add_argument("--input", choices=["spacemouse", "spacemouse-kb", "keyboard"],
                   default="spacemouse",
                   help="teleop device: spacemouse (buttons for gripper), "
                        "spacemouse-kb (SpaceMouse pose + Z/X keys for gripper), "
                        "keyboard (all keys)")
    p.add_argument("--out-dir", type=Path, default=Path("dataset") / "demos")
    p.add_argument("--rate", type=float, default=30.0, help="control rate (Hz)")
    p.add_argument("--speed", type=float, default=0.75,
                   help="teleop motion magnitude in [0,1] (lower = slower); "
                        "per-step delta = speed * action-scale")
    p.add_argument("--gripper-value", type=float, default=0.45,
                   help="per-step gripper delta magnitude in [0,1]")
    p.add_argument("--idle-threshold", type=float, default=1e-3,
                   help="action-norm at/below which a frame is idle; 0 disables idle trim")
    p.add_argument("--keep-trailing-idle", type=int, default=5,
                   help="max trailing idle frames to keep (so the policy learns to stop)")
    p.add_argument("--max-interior-idle", type=int, default=3,
                   help="max frames to keep in a mid-episode idle run (collapses long pauses)")
    p.add_argument("--record-rgb", nargs="?", const="cam_ext", default=None, metavar="CAM",
                   help="save one RGB mp4 per SAVED episode from this camera (default cam_ext when "
                        "the flag is given bare) into <run>/videos/. Presentation-only: the frame "
                        "is already captured every step for the depth pipeline (depth is aligned "
                        "TO the color stream), so this adds no sensor work and does not touch the "
                        "recorded obs/action data.")
    p.add_argument("--show-pointcloud", action="store_true",
                   help="open a live Open3D window of the PROCESSED cloud (crop+filters) "
                        "as you teleop — visualize online instead of collecting to inspect")
    p.add_argument("--shard-size", type=int, default=5,
                   help="episodes per shard file written during collection (merged at end); "
                        "0 disables sharding (old atomic-rewrite mode)")
    p.add_argument("--description", type=str, default="",
                   help="short free-text annotation saved in config.yaml (e.g. 'gentle grasps, left side only')")
    args = p.parse_args()

    print("note: the robot moves under teleop — keep the e-stop in reach.", file=sys.stderr)

    # Imports deferred so --help doesn't require the env to be built.
    from gentle_manip.envs.real_backend import RealBackend
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.perception.obs_config import ObsConfig

    setup = _load_yaml(_resolve_config(args.setup))
    obs_d = _load_yaml(_resolve_config(args.obs_config))
    action_d = _load_yaml(_resolve_config(args.action_config))
    obs_config = ObsConfig.from_dict(obs_d)
    action_config = ActionConfig.from_dict(action_d)

    record_action_config = None
    record_action_d = None
    if args.record_action_config is not None:
        record_action_d = _load_yaml(_resolve_config(args.record_action_config))
        record_action_config = ActionConfig.from_dict(record_action_d)
        if record_action_config.mode != "absolute":
            raise ValueError(
                f"--record-action-config must be mode: absolute, got "
                f"{record_action_config.mode!r} ({args.record_action_config})")
        print(f"recording ABSOLUTE actions (inverted from actual pose) from "
              f"{args.record_action_config} while teleop drives via {args.action_config}")

    # Reproducibility snapshot: the resolved configs + control knobs that shaped the data,
    # written next to the dataset as <stem>_config.yaml (mirrors collect_demos_sim.py).
    collection_config = {
        "task_name": args.task_name,
        "description": args.description,
        "input": args.input,
        "git_commit": _git_commit(),
        "sources": {"setup": str(args.setup), "obs": str(args.obs_config),
                    "action": str(args.action_config),
                    "record_action": str(args.record_action_config) if record_action_config else None},
        "control": {"rate_hz": args.rate, "speed": args.speed,
                    "gripper_value": args.gripper_value, "idle_threshold": args.idle_threshold,
                    "keep_trailing_idle": args.keep_trailing_idle,
                    "max_interior_idle": args.max_interior_idle},
        "setup": setup, "obs": obs_d, "action": action_d, "record_action": record_action_d,
    }

    backend = RealBackend(setup)

    # build_obs_space needs the tactile image (H, W) when tactile is configured.
    # The setup's per-sensor output_size is (W, H) (cv2 resize order); the frame is (H, W, 3).
    tactile_shape = None
    if obs_config.tactile is not None:
        w, h = setup["tactile"][obs_config.tactile.sensors[0]]["output_size"]
        tactile_shape = (int(h), int(w))

    rgb_shape = None
    if obs_config.images is not None:
        cam0 = setup["cameras"][obs_config.images.cameras[0]]
        rgb_shape = (int(cam0.get("height", 480)), int(cam0.get("width", 640)))

    env = PolicyEnv(backend, obs_config, action_config, task=None,
                    tactile_shape=tactile_shape, rgb_shape=rgb_shape,
                    max_episode_steps=10 ** 9)  # huge → no auto-reset; keys bound episodes

    if args.input == "keyboard":
        # One pygame object serves both motion (held keys) and episode events.
        from gentle_manip.demos.teleop_keyboard import KeyboardTeleop
        kb = KeyboardTeleop(move_speed=args.speed, rot_speed=args.speed,
                            gripper_value=args.gripper_value)
        teleop, keyboard = kb, kb
    elif args.input == "spacemouse-kb":
        # SpaceMouse for pose, Z/X keys for gripper; one object serves both roles.
        from gentle_manip.demos.teleop_spacemouse_kb import SpaceMousePygameTeleop
        sm_kb = SpaceMousePygameTeleop(scale=args.speed,
                                       gripper_value=args.gripper_value)
        teleop, keyboard = sm_kb, sm_kb
    else:
        teleop = SpaceMouseTeleop(scale=args.speed, gripper_value=args.gripper_value)
        keyboard = PygameKeyboard()

    cloud_viewer = None
    if args.show_pointcloud:
        if obs_config.point_cloud is None:
            print("--show-pointcloud ignored: obs config has no point_cloud", file=sys.stderr)
        else:
            from gentle_manip.visualization.live_cloud_viewer import LiveCloudViewer
            cloud_viewer = LiveCloudViewer(
                crop_min=obs_config.point_cloud.crop_min,
                crop_max=obs_config.point_cloud.crop_max,
            )

    frame_fn = None
    if args.record_rgb:
        _cam = args.record_rgb
        def frame_fn(_cam=_cam):
            # RawObs.rgb_images[cam] is (1, H, W, 3) uint8 — captured every step regardless
            # (the RealSense aligns depth TO color), we just stop dropping it.
            return env.last_raw_obs.rgb_images[_cam][0]
    recorder = DemoRecorder(
        env=env,
        teleop=teleop,
        keyboard=keyboard,
        task_name=args.task_name,
        out_dir=args.out_dir,
        rate_hz=args.rate,
        idle_threshold=args.idle_threshold,
        keep_trailing_idle=args.keep_trailing_idle,
        max_interior_idle=args.max_interior_idle,
        cloud_viewer=cloud_viewer,
        collection_config=collection_config,
        shard_size=args.shard_size,
        record_action_config=record_action_config,
        frame_fn=frame_fn,
        video_episodes=10**9 if args.record_rgb else 0,   # every saved episode
    )
    recorder.run()
    recorder.write()


if __name__ == "__main__":
    main()
