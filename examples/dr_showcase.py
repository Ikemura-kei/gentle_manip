"""Showcase the full object domain randomization — pose + orientation + SIZE + SHAPE.

Runs N episodes of a RANDOM-motion policy on the soft mushroom, each in a FRESH process so it
gets a different randomized object (a fresh scale + procedural shape deformation are sampled at
scene build; pose/orientation vary per reset). Each episode's RGB rollout (external camera) is
recorded with the applied DR params overlaid, and the clips are stitched into one video —
so you can watch size/curvature/twist/taper/pose vary across episodes.

Fresh process per episode = clean geometry with no in-process rebuild leak (the subprocess
backend can't return RGB, and the in-process one leaks on rebuild — so we isolate at the OS).

    MUJOCO_GL=egl uv run --project envs/sim python examples/dr_showcase.py --episodes 12
    # -> logs/dr_showcase/<datetime>/{ep00..N.mp4, dr_showcase.mp4, params.csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DR_CFG = _REPO / "gentle_manip" / "configs" / "dr" / "food_shape.yaml"


# ── one episode (child process): build fresh -> random rollout -> labelled clip ───
def _worker(seed: int, out_mp4: Path, steps: int, params_out: Path) -> None:
    import yaml
    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.experiment import Experiment
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.robot.xarm7_sim import _np
    from gentle_manip.tasks.single_lift import SingleLiftTask

    exp = Experiment.load("single_lift_mushroom_soft")
    task = SingleLiftTask(exp.task_cfg)
    dr = yaml.safe_load(_DR_CFG.read_text())
    backend = SimBackend(task.scene_spec, num_envs=1, use_subprocess=False, show_viewer=False,
                         render_cameras=False, record_camera=True,
                         config={"sim": {"settle_steps": 40}, "dr": dr, "seed": seed})
    scene = backend.scene_params()                      # scale / bend_deg / twist_deg / taper
    cam = next(iter(backend.process.handle.cameras.values()))[0]   # in-process clip cam (env 0)
    env = PolicyEnv(backend, ObsConfig(), ActionConfig.from_dict(
        {"scales": exp.action_config.scales, "clip": [-1.0, 1.0]}), task=None,
        max_episode_steps=10 ** 9)

    label = "  ".join([f"scale={scene.get('scale', 1.0):.2f}"] +
                      [f"{k}={scene[k]:+.0f}" for k in ("bend_deg", "twist_deg") if k in scene] +
                      ([f"taper={scene['taper']:+.2f}"] if "taper" in scene else []))
    frames = []
    env.reset()
    rng = np.random.default_rng(1000 + seed)
    a = np.zeros((1, 7), np.float32)
    for _ in range(steps):
        a = np.clip(0.85 * a + rng.normal(0, 0.35, (1, 7)), -1, 1).astype(np.float32)  # smooth walk
        env.step(a)
        frames.append(_overlay(_np(cam.render(rgb=True, depth=False)[0]), f"ep{seed}  DR: {label}"))
    _write_video(out_mp4, frames)
    backend.close()
    params_out.write_text(json.dumps({"seed": seed, **scene}))


def _overlay(frame: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(np.ascontiguousarray(frame[..., :3]).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 20], fill=(0, 0, 0))
        d.text((5, 4), text, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:
        return frame[..., :3].astype(np.uint8)


def _write_video(path: Path, frames, fps: int = 30) -> None:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    fr = [f[: f.shape[0] - f.shape[0] % 2, : f.shape[1] - f.shape[1] % 2] for f in frames]
    try:
        imageio.mimsave(str(path), fr, fps=fps, macro_block_size=1)
    except TypeError:
        imageio.mimsave(str(path), fr, fps=fps, codec="libx264")


# ── driver: fan out episodes as fresh processes, then stitch ──────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=120, help="random-policy steps per episode")
    ap.add_argument("--out", type=Path, default=None)
    # internal:
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--mp4", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--params", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        _worker(args.seed, args.mp4, args.steps, args.params)
        return

    out = args.out or (_REPO / "logs" / "dr_showcase" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    out.mkdir(parents=True, exist_ok=True)
    clips, rows = [], []
    for i in range(args.episodes):
        mp4 = out / f"ep{i:02d}.mp4"
        pj = out / f"ep{i:02d}.json"
        print(f"[dr_showcase] episode {i + 1}/{args.episodes} (fresh build, seed {i}) ...", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                        "--seed", str(i), "--mp4", str(mp4), "--params", str(pj),
                        "--steps", str(args.steps)], check=True, env={**os.environ, "MUJOCO_GL": "egl"})
        if mp4.exists():
            clips.append(mp4)
        if pj.exists():
            rows.append(json.loads(pj.read_text()))

    # per-episode DR param table
    if rows:
        keys = ["seed", "scale", "bend_deg", "twist_deg", "taper", "rbf"]
        with open(out / "params.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
    # stitch (same resolution) into one showcase video via ffmpeg concat
    if clips:
        lst = out / "clips.txt"
        lst.write_text("".join(f"file '{c}'\n" for c in clips))
        show = out / "dr_showcase.mp4"
        r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c", "copy", str(show)], capture_output=True, text=True)
        if r.returncode != 0:      # fallback: re-encode if stream-copy concat fails
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(show)], check=False)
        print(f"[dr_showcase] stitched -> {show}")
    print(f"[dr_showcase] {len(clips)} clips + params.csv in {out}")


if __name__ == "__main__":
    main()
