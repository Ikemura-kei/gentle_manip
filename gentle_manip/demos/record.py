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

# Teleop demonstration collection on the real robot. The SpaceMouse command is a
# normalized [-1,1] raw_action — the same thing a policy outputs — so it flows
# through the env's ActionPipeline, and recorded obs are exactly what a policy
# will see. Runs in the 3.11 deploy env:
#   uv run --directory deploy python -m gentle_manip.demos.record ...

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
                 max_interior_idle: int = 3) -> None:
        self.env = env
        self.teleop = teleop
        self.keyboard = keyboard
        self.task_name = task_name
        self.out_dir = Path(out_dir)
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
        self.episodes: List[dict] = []
        self._path: Optional[Path] = None   # chosen lazily on the first saved episode

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
                    print(f"  saved episode {len(self.episodes)} ({n} steps) → {self._path}")
                    obs = self.env.reset()
                    continue
                if DISCARD in events:
                    self._discard_episode()
                    print("  discarded episode")
                    obs = self.env.reset()
                    continue

                action = np.asarray(self.teleop.get_action(), dtype=np.float32)
                obs = self._step_and_record(obs, action)

                self._rate_limit(t0, period)
        finally:
            self.teleop.close()
            self.keyboard.close()
            self.env.close()

    def _step_and_record(self, obs, action: np.ndarray):
        """Record (obs_t, action_t), advance the env, return the next obs."""
        obs_next, _reward, _done, _info = self.env.step(action[None, :])
        self._obs_buf.append({k: np.asarray(v)[0] for k, v in obs.items()})  # drop num_envs
        self._act_buf.append(action.copy())
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
        obs_buf, act_buf = self._trim_idle(self._obs_buf, self._act_buf)
        n = len(act_buf)
        if n == 0:
            self._clear()
            return 0
        keys = obs_buf[0].keys()
        observations = {k: np.stack([o[k] for o in obs_buf]) for k in keys}
        actions = np.stack(act_buf)
        self.episodes.append({"observations": observations, "actions": actions})
        self._clear()
        self._flush()                          # persist immediately — crash-safe
        return n

    def _trim_idle(self, obs_buf, act_buf):
        """Cap each consecutive idle run by position (leading/interior/trailing).

        Idle = full action norm <= idle_threshold. Keeps the first `cap` frames of
        each idle run (0 leading, max_interior_idle interior, keep_trailing_idle
        trailing). idle_threshold <= 0 disables trimming.
        """
        if self.idle_threshold <= 0 or not act_buf:
            return obs_buf, act_buf
        idle = np.linalg.norm(np.stack(act_buf), axis=1) <= self.idle_threshold
        T = len(act_buf)
        if idle.all():
            return [], []                      # whole episode idle → nothing to keep

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
        return obs_out, act_out

    def _discard_episode(self) -> None:
        self._clear()

    def _clear(self) -> None:
        self._obs_buf = []
        self._act_buf = []

    # ── Output ────────────────────────────────────────────────────────────────

    def _dataset_path(self) -> Path:
        """Pick a short unique session filename once: <out>/<task>/YY-MM-DD-xyz.pkl."""
        if self._path is None:
            out = self.out_dir / self.task_name
            out.mkdir(parents=True, exist_ok=True)
            date = datetime.now().strftime("%y-%m-%d")
            for _ in range(10000):
                sfx = "".join(random.choices(string.ascii_lowercase, k=3))
                cand = out / f"{date}-{sfx}.pkl"
                if not cand.exists():
                    self._path = cand
                    break
            else:
                raise RuntimeError(f"no free filename in {out}")
        return self._path

    def _flush(self) -> None:
        """Atomically (re)write the full dataset so an interrupt can't lose saved
        episodes. Called after every saved episode; rewrites the one session file."""
        if not self.episodes:
            return
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
        path = self._dataset_path()
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(dataset, f)
        os.replace(tmp, path)                  # atomic swap (POSIX)

    def write(self) -> Optional[Path]:
        """Final report — episodes are already flushed incrementally as saved."""
        if not self.episodes:
            print("no episodes recorded — nothing written")
            return None
        self._flush()
        print(f"saved {len(self.episodes)} episodes → {self._path}")
        return self._path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_config(path: Path) -> Path:
    """Find a config path regardless of cwd (the deploy env runs from deploy/).

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
                   default=_PKG / "configs" / "obs" / "state_ee_only.yaml")
    p.add_argument("--action-config", type=Path,
                   default=_PKG / "configs" / "action" / "delta_pose_delta_gripper.yaml")
    p.add_argument("--task-name", required=True)
    p.add_argument("--input", choices=["spacemouse", "keyboard"], default="spacemouse",
                   help="teleop device")
    p.add_argument("--out-dir", type=Path, default=Path("dataset") / "demos")
    p.add_argument("--rate", type=float, default=20.0, help="control rate (Hz)")
    p.add_argument("--speed", type=float, default=0.55,
                   help="teleop motion magnitude in [0,1] (lower = slower); "
                        "per-step delta = speed * action-scale")
    p.add_argument("--gripper-value", type=float, default=0.05,
                   help="per-step gripper delta magnitude in [0,1]")
    p.add_argument("--idle-threshold", type=float, default=1e-3,
                   help="action-norm at/below which a frame is idle; 0 disables idle trim")
    p.add_argument("--keep-trailing-idle", type=int, default=5,
                   help="max trailing idle frames to keep (so the policy learns to stop)")
    p.add_argument("--max-interior-idle", type=int, default=3,
                   help="max frames to keep in a mid-episode idle run (collapses long pauses)")
    args = p.parse_args()

    print("note: the robot moves under teleop — keep the e-stop in reach.", file=sys.stderr)

    # Imports deferred so --help doesn't require the env to be built.
    from gentle_manip.envs.real_backend import RealBackend
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.actions.action_config import ActionConfig

    setup = _load_yaml(_resolve_config(args.setup))
    obs_config = ObsConfig.from_dict(_load_yaml(_resolve_config(args.obs_config)))
    action_config = ActionConfig.from_dict(_load_yaml(_resolve_config(args.action_config)))

    backend = RealBackend(setup)
    env = PolicyEnv(backend, obs_config, action_config, task=None,
                    max_episode_steps=10 ** 9)  # huge → no auto-reset; keys bound episodes

    if args.input == "keyboard":
        # One pygame object serves both motion (held keys) and episode events.
        from gentle_manip.demos.teleop_keyboard import KeyboardTeleop
        kb = KeyboardTeleop(move_speed=args.speed, rot_speed=args.speed,
                            gripper_value=args.gripper_value)
        teleop, keyboard = kb, kb
    else:
        teleop = SpaceMouseTeleop(scale=args.speed, gripper_value=args.gripper_value)
        keyboard = PygameKeyboard()

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
    )
    recorder.run()
    recorder.write()


if __name__ == "__main__":
    main()
