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
from scipy.spatial.transform import Rotation

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
                    pose_scale: float = 1.0, record_path: "Path | None" = None,
                    shard_size: int = 0, action_config=None,
                    smooth_alpha: "float | None" = None,
                    max_pos_step_m: "float | None" = None) -> None:
    """Receding-horizon deploy loop shared by real and sim deployment.

    env: PolicyEnv-like — reset()->obs dict, step(action)->(obs, ...). policy:
    DP3PolicyAdapter. Re-plans every policy.n_action_steps; k starts, SPACE re-homes,
    q quits. Owns env.close() on exit.

    action_config: the same ActionConfig `main()` already loaded — used to read
    `.mode` (delta/absolute) and, for the position cap below, `.pos_min/.pos_max/.clip`.
    None (default) behaves like delta mode with no position cap, for callers that don't
    pass one.

    pose_scale (<1, DELTA MODE ONLY) shrinks the 6 delta-pose dims of every command for
    slower, gentler motion (gripper dim is left at full range so grasps still close).
    The policy re-plans from the actual state each chunk, so it still converges — just
    slower. Meaningless for mode="absolute" (there scaling the raw [-1,1] target toward 0
    pulls the commanded pose toward the workspace-normalization CENTER, not toward the
    current pose — NOT gentler, just a different, wrong target) — it is a no-op there
    regardless of the value passed in.

    smooth_alpha (ABSOLUTE MODE ONLY, None = off): EMA / first-order low-pass filter on
    the raw action chunk — smoothed = alpha*raw + (1-alpha)*prev_smoothed, applied per
    step (not per chunk) and PERSISTED across re-plans (only reset at env reset/re-home),
    so it actually attenuates step-to-step jitter in the commanded absolute pose instead
    of just resampling it. Only pos(3)+rot6d(6) [dims 0:9] are smoothed — the gripper dim
    is passed through raw so grasp/release stays decisive. Lower alpha = smoother/slower
    to track a new target; start around 0.3 and tune from there. This is the "shakiness"
    knob for absolute-pose policies — the delta-mode equivalent of pose_scale.

    max_pos_step_m (ABSOLUTE MODE ONLY, None = off): HARD per-tick slew-rate cap on the
    commanded position, in meters PER AXIS (not Euclidean norm) — unlike smooth_alpha
    (a proportional low-pass that still lets a single huge outlier move the blended
    target partway there immediately), this clamps the position delta from the
    PREVIOUSLY SENT command to at most this many meters, so no single tick can ever
    move the target further than that, no matter what the network outputs. Applied in
    raw [-1,1] units (position maps affinely into [pos_min,pos_max], so clamping the raw
    delta is exactly equivalent to clamping the physical delta) AFTER smooth_alpha, as a
    final safety bound. Position only (rotation is already covered by smooth_alpha;
    gripper is intentionally uncapped so grasp/release stays decisive).

    record_path: if set, save each (obs seen, action taken) step into the SAME pickle
    schema as recorded demos ({"episodes": [{"observations": {k: (T,...)}, "actions":
    (T,7)}]}), so visualize_demo / episode_player render real runs identically to sim
    demos — for sim2real obs comparison (esp. the point cloud).

    shard_size: 0 (default) = write ALL episodes into the single pkl `record_path`
    (legacy). >0 = treat `record_path` as a DIRECTORY and write `shard_XXXX.pkl` of at
    most `shard_size` episodes each; full shards are flushed incrementally as episodes
    complete (interrupt-safe — a crash keeps the finished shards, only the trailing
    partial waits for exit), keeping each read/write small.
    """
    period = 1.0 / rate if rate > 0 else 0.0
    steps = 0
    record = record_path is not None
    shard = record and shard_size and shard_size > 0
    episodes: list = []
    obs_buf: list = []
    act_buf: list = []
    shards_done = 0                                    # count of fully-written shards (sharding mode)

    def _flush_episode() -> None:
        if not act_buf:
            return
        ep_obs = {k: np.stack([o[k] for o in obs_buf]) for k in obs_buf[0]}
        episodes.append({"observations": ep_obs, "actions": np.stack(act_buf)})
        obs_buf.clear()
        act_buf.clear()

    def _write_pkl(path: "Path", eps: list, shard_idx: "int | None" = None) -> None:
        import pickle
        from datetime import datetime, timezone
        meta = {                                       # same schema as recorded demos
            "task": "real_deploy", "source": "deploy_real",
            "obs_keys": sorted(eps[0]["observations"].keys()),
            "action_dim": int(eps[0]["actions"].shape[1]),
            "rate_hz": rate, "created": datetime.now(timezone.utc).isoformat(),
            "n_episodes": len(eps),
        }
        if shard_idx is not None:
            meta["shard"] = f"shard_{shard_idx:04d}"
        path.parent.mkdir(parents=True, exist_ok=True)  # safe for mid-run shard flush + final save
        tmp = path.with_suffix(path.suffix + ".tmp")   # atomic write
        with open(tmp, "wb") as f:
            pickle.dump({"meta": meta, "episodes": eps}, f)
        tmp.replace(path)

    def _flush_full_shards() -> None:
        """Write any newly-completed FULL shards (shard_size episodes each), so an interrupt
        keeps the finished shards; only the trailing partial waits for _save()."""
        nonlocal shards_done
        while len(episodes) >= (shards_done + 1) * shard_size:
            lo, hi = shards_done * shard_size, (shards_done + 1) * shard_size
            _write_pkl(record_path / f"shard_{shards_done:04d}.pkl", episodes[lo:hi], shards_done)
            print(f"  saved shard_{shards_done:04d} ({shard_size} episodes) → {record_path}/")
            shards_done += 1

    def _save() -> None:
        if not record:
            return
        _flush_episode()
        if not episodes:
            return
        if shard:
            record_path.mkdir(parents=True, exist_ok=True)
            _flush_full_shards()                       # any remaining full shards
            if len(episodes) > shards_done * shard_size:          # trailing partial shard
                n = len(episodes) - shards_done * shard_size
                _write_pkl(record_path / f"shard_{shards_done:04d}.pkl",
                           episodes[shards_done * shard_size:], shards_done)
                print(f"  saved shard_{shards_done:04d} ({n} episodes) → {record_path}/")
            n_sh = (len(episodes) + shard_size - 1) // shard_size
            print(f"  recorded {len(episodes)} real episode(s) in {n_sh} shard(s) → {record_path}/")
        else:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            _write_pkl(record_path, episodes)
            print(f"  saved {len(episodes)} real episode(s) → {record_path}")

    action_mode = action_config.mode if action_config is not None else "delta"
    _abs_filters_on = action_mode == "absolute" and action_config is not None

    # Raw<->physical position conversion constants (absolute mode only) — shared by both
    # filters below so each can be SEEDED from the robot's ACTUAL current pose right after a
    # reset/re-home. Without seeding, the very first predicted action has no "previous
    # command" to blend/clamp against and would be sent raw/uncapped — exactly the "first
    # action is abrupt compared to the initial position" gap.
    _pos_min = _pos_max = _clip_lo = _clip_hi = _raw_pos_cap = None
    if _abs_filters_on:
        _pos_min = np.asarray(action_config.pos_min, np.float32)
        _pos_max = np.asarray(action_config.pos_max, np.float32)
        _clip_lo, _clip_hi = action_config.clip
        if max_pos_step_m is not None:
            _raw_pos_cap = float(max_pos_step_m) / (_pos_max - _pos_min) * (_clip_hi - _clip_lo)  # (3,)

    def _current_raw_pose(obs: dict) -> np.ndarray:
        """9-dim raw-space pose built from the robot's ACTUAL current state: dims 0:3 =
        position, inverse-mapped through the same affine pos_min/pos_max transform
        ActionPipeline uses; dims 3:9 = a valid 6D rotation rep from the current ee_quat
        (first two columns of its rotation matrix — the priv_object_rot6d convention;
        already orthonormal, so a Gram-Schmidt pass reproduces this exact rotation)."""
        phys_pos = np.asarray(obs["ee_pos"], np.float32)[0]         # (3,), num_envs=1 squeeze
        t = (phys_pos - _pos_min) / (_pos_max - _pos_min)
        pos_raw = _clip_lo + t * (_clip_hi - _clip_lo)
        quat_wxyz = np.asarray(obs["ee_quat"], np.float32)[0]
        R = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()  # wxyz -> scipy's xyzw
        rot6d = R[:, :2].reshape(-1, order="F")                     # [col0(3), col1(3)]
        return np.concatenate([pos_raw, rot6d]).astype(np.float32)

    # EMA low-pass filter state for absolute-mode smoothing (see run_deploy_loop docstring).
    # Only pos(3)+rot6d(6) [dims 0:9] are smoothed — persists across chunk re-plans, reset
    # (re-seeded from the actual pose) on env reset / re-home.
    _SMOOTH_DIMS = slice(0, 9)
    prev_smoothed = [None]                                          # boxed for closure mutation

    def _smooth(action: np.ndarray) -> np.ndarray:
        if action_mode != "absolute" or smooth_alpha is None:
            return action
        action = action.copy()
        prev_smoothed[0] = (smooth_alpha * action[_SMOOTH_DIMS]
                            + (1.0 - smooth_alpha) * prev_smoothed[0])
        action[_SMOOTH_DIMS] = prev_smoothed[0]
        return action

    prev_pos_raw = [None]                                           # boxed for closure mutation
    _last_clip_print = [0.0]                                        # wall-clock throttle (boxed)
    _CLIP_PRINT_PERIOD_S = 0.5                                      # at most 2 prints/sec

    def _cap_pos(action: np.ndarray) -> np.ndarray:
        if _raw_pos_cap is None:
            return action
        action = action.copy()
        raw_delta = action[0:3] - prev_pos_raw[0]
        hit = np.abs(raw_delta) > _raw_pos_cap                      # per-axis: was this axis clamped?
        if hit.any():
            now = time.perf_counter()
            if now - _last_clip_print[0] > _CLIP_PRINT_PERIOD_S:
                axes = "".join(a for a, h in zip("xyz", hit) if h)
                over_mm = (np.abs(raw_delta) - _raw_pos_cap)[hit] * 1000.0
                print(f"  [pos-cap] clipped axis={axes}  over by {np.round(over_mm, 1)} mm "
                      f"(cap={max_pos_step_m * 1000:.0f} mm/tick)", flush=True)
                _last_clip_print[0] = now
        delta = np.clip(raw_delta, -_raw_pos_cap, _raw_pos_cap)
        action[0:3] = prev_pos_raw[0] + delta
        prev_pos_raw[0] = action[0:3].copy()
        return action

    def _seed_abs_filters(obs: dict) -> None:
        """Anchor both filters to the robot's ACTUAL current pose right after a reset/
        re-home (see _current_raw_pose docstring for why this matters)."""
        if not _abs_filters_on:
            return
        pose = _current_raw_pose(obs)
        if smooth_alpha is not None:
            prev_smoothed[0] = pose[_SMOOTH_DIMS].copy()
        if _raw_pos_cap is not None:
            prev_pos_raw[0] = pose[0:3].copy()

    try:
        with KeyPoller() as keys:
            controls = ("k = start   SPACE = reset episode (re-home)   q = quit"
                        if keys.enabled else "(stdin not a tty — manual keys disabled)")
            print(f"deploying up to {max_steps} steps at {rate:.0f} Hz "
                  f"(re-plan every {policy.n_action_steps}).  {controls}")
            obs = env.reset()                                       # homes the robot
            policy.reset(obs)
            _seed_abs_filters(obs)                                  # anchor filters to the actual home pose
            print("  " + _pc_health(obs))                           # cube-signal sanity at home
            _wait_for_start(keys)                                   # hold until 'k'
            while steps < max_steps:
                chunk = policy.predict()                            # (n_action_steps, act_dim)
                if pose_scale != 1.0 and action_mode == "delta":
                    chunk = chunk.copy()
                    chunk[:, :6] *= pose_scale                      # slow pose; keep gripper full-range
                reset_now = False
                for action in chunk:
                    if steps < 2:
                        # First two EXECUTED steps of every (re-)start are forced to a NULL action so
                        # the arm holds still while perception/filters warm up before the policy drives
                        # it. absolute -> command the robot's CURRENT pose (no motion), gripper held
                        # OPEN so we never begin already grasping; delta -> all-zeros (no pose delta,
                        # no gripper change). Keyed on `steps`, so it re-applies after each SPACE re-home.
                        if action_mode == "absolute":
                            action = np.concatenate(
                                [_current_raw_pose(obs), np.array([1.0], np.float32)]).astype(np.float32)
                        else:
                            action = np.zeros_like(action)
                    action = _cap_pos(_smooth(action))
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
                    if shard:
                        _flush_full_shards()                       # persist finished shards mid-run
                    obs = env.reset()
                    policy.reset(obs)
                    _seed_abs_filters(obs)                         # anchor filters to the actual re-homed pose
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
                   default=_PKG / "configs" / "action" / "abs_pose_abs_gripper.yaml")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--rate", type=float, default=30.0, help="control rate (Hz)")
    p.add_argument("--pose-scale", type=float, default=1.0,
                   help="(delta mode only) multiply the 6 delta-pose dims of every command "
                        "(e.g. 0.5 = half-speed, gentler motion; gripper unaffected). Match "
                        "this to the value you eval with in sim. No-op in absolute mode.")
    p.add_argument("--smooth-alpha", type=float, default=None,
                   help="(absolute mode only) EMA low-pass filter alpha on the commanded "
                        "pos+rotation (gripper dim excluded); lower = smoother/slower to "
                        "track a new target. None = off.")
    p.add_argument("--max-pos-step-m", type=float, default=None,
                   help="(absolute mode only) hard per-tick cap, meters PER AXIS, on how far "
                        "the commanded position may move from the previous command — a slew-"
                        "rate limiter, independent of/in addition to --smooth-alpha. None = off.")
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
                    record_path=args.record, action_config=action_config,
                    smooth_alpha=args.smooth_alpha, max_pos_step_m=args.max_pos_step_m)


if __name__ == "__main__":
    main()
