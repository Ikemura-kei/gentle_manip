from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from collections import deque
from pathlib import Path

import numpy as np
import yaml

# Deploy a trained DP3 policy on the real XArm7 — runs in the unified 3.8 env
# (envs/dp3), which has DP3 + pytorch3d AND the hardware SDKs, so the policy and
# RealBackend share one process (no IPC):
#   uv run --project envs/dp3 python gentle_manip/scripts/deploy_real.py --ckpt <latest.ckpt>
#
# gentle_manip and DP3 are imported from source via sys.path (neither is installed
# as a package in envs/dp3): repo root for gentle_manip, the DP3 package dir for
# `train` / `diffusion_policy_3d`.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]                                   # <repo>/gentle_manip/scripts/deploy_real.py
_PKG = _REPO / "gentle_manip"
_DP3 = _REPO / "third_party" / "DP3" / "3D-Diffusion-Policy"
for p in (str(_REPO), str(_DP3)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gentle_manip.actions.action_config import ActionConfig          # noqa: E402
from gentle_manip.envs.policy_env import PolicyEnv                    # noqa: E402
from gentle_manip.envs.real_backend import RealBackend               # noqa: E402
from gentle_manip.perception.obs_config import ObsConfig             # noqa: E402


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_config(path: Path) -> Path:
    """Find a config path regardless of cwd (tries as-given, then repo-root-relative)."""
    if path.is_file():
        return path
    alt = _REPO / path
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"config not found: {path} (also tried {alt})")


class DP3PolicyAdapter:
    """Wraps a trained DP3 policy as `obs dict -> action chunk`.

    Loads the policy with the *exact training config embedded in the checkpoint*
    (`create_from_checkpoint` → `TrainDP3Workspace(payload['cfg'])`), so the model
    is built identically to training. Maintains the last `n_obs_steps` observations
    and returns the `n_action_steps` action chunk in the raw [-1,1] teleop space.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda:0") -> None:
        import torch
        from train import TrainDP3Workspace

        self._torch = torch
        ws = TrainDP3Workspace.create_from_checkpoint(ckpt_path)     # uses payload['cfg']
        cfg = ws.cfg
        use_ema = bool(getattr(cfg.training, "use_ema", True))
        self.policy = ws.ema_model if (use_ema and ws.ema_model is not None) else ws.model
        self.policy.eval()
        self.policy.to(device)
        self.device = next(self.policy.parameters()).device
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.n_action_steps = int(cfg.n_action_steps)
        self._hist: deque = deque(maxlen=self.n_obs_steps)
        print(f"loaded DP3 policy ({'ema' if use_ema else 'model'}) — "
              f"n_obs_steps={self.n_obs_steps} n_action_steps={self.n_action_steps} device={self.device}")

    # obs dict from PolicyEnv has a leading num_envs=1 dim; squeeze it here.
    @staticmethod
    def _agent_pos(obs: dict) -> np.ndarray:
        return np.concatenate(
            [obs["ee_pos"][0], obs["ee_quat"][0], obs["gripper_width"][0]]
        ).astype(np.float32)                                         # (8,)

    @staticmethod
    def _point_cloud(obs: dict) -> np.ndarray:
        return obs["point_cloud"][0].astype(np.float32)             # (1024, 3)

    def reset(self, obs: dict) -> None:
        self._hist.clear()
        for _ in range(self.n_obs_steps):                           # seed history with the first obs
            self._hist.append((self._point_cloud(obs), self._agent_pos(obs)))

    def push(self, obs: dict) -> None:
        self._hist.append((self._point_cloud(obs), self._agent_pos(obs)))

    def predict(self) -> np.ndarray:
        """Action chunk (n_action_steps, 7) in [-1,1] from the last n_obs_steps obs."""
        torch = self._torch
        pcs = np.stack([h[0] for h in self._hist])                  # (To, 1024, 3)
        aps = np.stack([h[1] for h in self._hist])                  # (To, 8)
        obs_dict = {
            "point_cloud": torch.from_numpy(pcs).unsqueeze(0).to(self.device),  # (1, To, 1024, 3)
            "agent_pos": torch.from_numpy(aps).unsqueeze(0).to(self.device),    # (1, To, 8)
        }
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
        return result["action"][0].detach().cpu().numpy()          # (n_action_steps, 7)


class KeyPoller:
    """Non-blocking single-key reader for the control loop (stdlib, no display).

    Puts the terminal in cbreak mode so `poll()` returns one pending keypress
    (or None) without blocking or needing Enter. Degrades to a no-op when stdin
    is not a tty (e.g. piped), so deployment still runs unattended.
    """

    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self.enabled else None
        self._old = None

    def __enter__(self) -> "KeyPoller":
        if self.enabled:
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self.enabled and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self):
        if self.enabled and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def _wait_for_start(keys: "KeyPoller") -> None:
    """Block (homed, not moving) until the start key 'k'. 'q'/ESC aborts.

    No-op when stdin is not a tty so unattended runs proceed automatically.
    """
    if not keys.enabled:
        return
    print("  homed — press 'k' to start (q to quit) ...")
    while True:
        key = keys.poll()
        if key == "k":
            return
        if key in ("q", "\x1b"):
            raise KeyboardInterrupt
        time.sleep(0.02)


def _pc_health(obs: dict) -> str:
    """One-line health check of the policy's point-cloud input — the cube-position
    signal. An empty/odd-range cloud is the usual cause of cube-agnostic motion."""
    if "point_cloud" not in obs:
        return "point_cloud: NOT in obs"
    pc = np.asarray(obs["point_cloud"])[0]                     # (P, 3), squeeze num_envs
    nz = pc[np.any(pc != 0.0, axis=1)]
    if len(nz) == 0:
        return "point_cloud: EMPTY (0 nonzero pts) — crop/extrinsic/scale mismatch?"
    c, lo, hi = nz.mean(0), nz.min(0), nz.max(0)
    return (f"point_cloud: {len(nz)}/{len(pc)} nonzero  "
            f"centroid=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})  "
            f"x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]")


def run_deploy_loop(env, policy: "DP3PolicyAdapter", max_steps: int, rate: float,
                    pose_scale: float = 1.0, record_path: "Path | None" = None) -> None:
    """Receding-horizon deploy loop shared by real and sim deployment.

    env: PolicyEnv-like — reset()->obs dict, step(action)->(obs, ...). policy:
    DP3PolicyAdapter. Re-plans every policy.n_action_steps; k starts, SPACE re-homes,
    q quits. Owns env.close() on exit.

    pose_scale (<1) shrinks the 6 delta-pose dims of every command for slower, gentler
    motion (gripper dim is left at full range so grasps still close). The policy
    re-plans from the actual state each chunk, so it still converges — just slower.

    record_path: if set, save each (obs seen, action taken) step into the SAME pickle
    schema as recorded demos ({"episodes": [{"observations": {k: (T,...)}, "actions":
    (T,7)}]}), so visualize_demo / episode_player render real runs identically to sim
    demos — for sim2real obs comparison (esp. the point cloud).
    """
    period = 1.0 / rate if rate > 0 else 0.0
    steps = 0
    record = record_path is not None
    episodes: list = []
    obs_buf: list = []
    act_buf: list = []

    def _flush_episode() -> None:
        if not act_buf:
            return
        ep_obs = {k: np.stack([o[k] for o in obs_buf]) for k in obs_buf[0]}
        episodes.append({"observations": ep_obs, "actions": np.stack(act_buf)})
        obs_buf.clear()
        act_buf.clear()

    def _save() -> None:
        if not record:
            return
        import pickle
        from datetime import datetime, timezone
        _flush_episode()
        if not episodes:
            return
        record_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = {
            "meta": {                                  # same schema as recorded demos
                "task": "real_deploy",
                "source": "deploy_real",
                "obs_keys": sorted(episodes[0]["observations"].keys()),
                "action_dim": int(episodes[0]["actions"].shape[1]),
                "rate_hz": rate,
                "created": datetime.now(timezone.utc).isoformat(),
            },
            "episodes": episodes,
        }
        with open(record_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"  saved {len(episodes)} real episode(s) → {record_path}")

    try:
        with KeyPoller() as keys:
            controls = ("k = start   SPACE = reset episode (re-home)   q = quit"
                        if keys.enabled else "(stdin not a tty — manual keys disabled)")
            print(f"deploying up to {max_steps} steps at {rate:.0f} Hz "
                  f"(re-plan every {policy.n_action_steps}).  {controls}")
            obs = env.reset()                                       # homes the robot
            policy.reset(obs)
            print("  " + _pc_health(obs))                           # cube-signal sanity at home
            _wait_for_start(keys)                                   # hold until 'k'
            while steps < max_steps:
                chunk = policy.predict()                            # (n_action_steps, 7)
                if pose_scale != 1.0:
                    chunk = chunk.copy()
                    chunk[:, :6] *= pose_scale                      # slow pose; keep gripper full-range
                reset_now = False
                for action in chunk:
                    key = keys.poll()
                    if key in (" ", "r"):
                        print("  manual reset — re-homing")
                        reset_now = True
                        break
                    if key in ("q", "\x1b"):                        # q or ESC
                        raise KeyboardInterrupt
                    if steps >= max_steps:
                        break
                    if record:                                     # obs the policy acted on + action taken
                        obs_buf.append({k: np.asarray(v)[0].copy() for k, v in obs.items()})
                        act_buf.append(np.asarray(action, dtype=np.float32).copy())
                    t0 = time.perf_counter()
                    obs = env.step(action[None, :].astype(np.float32))[0]
                    policy.push(obs)
                    steps += 1
                    if period > 0:
                        dt = time.perf_counter() - t0
                        if dt < period:
                            time.sleep(period - dt)
                if reset_now:                                      # abandon chunk, re-home, fresh budget
                    _flush_episode()                               # close the recorded episode
                    obs = env.reset()
                    policy.reset(obs)
                    print("  " + _pc_health(obs))
                    steps = 0
                    _wait_for_start(keys)                          # hold until 'k' again
        print(f"done — executed {steps} steps")
    except KeyboardInterrupt:
        print("\ninterrupted — stopping (arm holds position)", file=sys.stderr)
    finally:
        _save()
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy a trained DP3 policy on the real XArm7")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--setup", type=Path, default=_PKG / "configs" / "setup" / "real_lab.yaml")
    p.add_argument("--obs-config", type=Path, default=_PKG / "configs" / "obs" / "point_cloud_1cam.yaml")
    p.add_argument("--action-config", type=Path,
                   default=_PKG / "configs" / "action" / "delta_pose_delta_gripper.yaml")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--rate", type=float, default=30.0, help="control rate (Hz)")
    p.add_argument("--pose-scale", type=float, default=1.0,
                   help="multiply the 6 delta-pose dims of every command (e.g. 0.5 = "
                        "half-speed, gentler motion; gripper unaffected). Match this to "
                        "the value you eval with in sim.")
    p.add_argument("--record", type=Path, default=None,
                   help="save (obs, action) per step to this pickle in the demo schema, so "
                        "visualize_demo / episode_player can compare the real run against "
                        "sim demos (esp. the point cloud). e.g. dataset/real_deploy/run1.pkl")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    print("note: the robot moves under policy control — keep the e-stop in reach.", file=sys.stderr)

    setup = _load_yaml(_resolve_config(args.setup))
    obs_config = ObsConfig.from_dict(_load_yaml(_resolve_config(args.obs_config)))
    action_config = ActionConfig.from_dict(_load_yaml(_resolve_config(args.action_config)))

    backend = RealBackend(setup)
    env = PolicyEnv(backend, obs_config, action_config, task=None, max_episode_steps=10 ** 9)
    policy = DP3PolicyAdapter(str(args.ckpt), device=args.device)

    run_deploy_loop(env, policy, args.max_steps, args.rate, pose_scale=args.pose_scale,
                    record_path=args.record)


if __name__ == "__main__":
    main()
