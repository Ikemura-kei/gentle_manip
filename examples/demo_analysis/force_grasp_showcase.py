"""Showcase video: paired grasp RGB + rolling contact-force signal, side by side.

Left panel: the recorded RGB grasp video (rendered during collection, --record-video).
Right panel: contact force vs time, x-axis FIXED to the episode horizon (so the plot
frame stays still — only the line grows as the episode plays), y-axis fixed to a shared
scale across all shown episodes (so grip strength is visually comparable episode to
episode). A moving marker + live value readout tracks the current step.

Usage:
    uv run --project envs/sim python examples/demo_analysis/force_grasp_showcase.py \\
        dataset/demos/single_lift_mushroom_rigid/26-07-28-cjk \\
        --n-episodes 5
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

PANEL_H = 480
PANEL_W = 480
HEADER_H = 40
FPS = 30


def _fig_to_rgb(fig, w, h) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)
    return np.array(Image.fromarray(buf).resize((w, h), Image.BILINEAR))


def _render_force_panel(t: np.ndarray, force: np.ndarray, step: int, t_max: float,
                        f_max: float) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(t[: step + 1], force[: step + 1], color="#d65f5f", lw=2)
    if step > 0:
        ax.scatter([t[step]], [force[step]], color="#d65f5f", s=40, zorder=5)

    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.05 * f_max, 1.08 * f_max)
    ax.set_xlabel("time (s)", fontsize=9)
    ax.set_ylabel("contact force (N)", fontsize=9)
    ax.set_title(f"{force[step]:.1f} N", fontsize=12, color="#d65f5f")
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    fig.tight_layout(pad=0.5)
    return _fig_to_rgb(fig, PANEL_W, PANEL_H)


def _make_header(step: int, total: int, gripper_w: float) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(PANEL_W * 2 / 100, HEADER_H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    label = f"step {step:>4d} / {total}    gripper {gripper_w * 100:.1f} cm"
    ax.text(0.5, 0.5, label, transform=ax.transAxes, fontsize=11, color="#333333",
            ha="center", va="center", family="monospace")
    fig.tight_layout(pad=0)
    return _fig_to_rgb(fig, PANEL_W * 2, HEADER_H)


def _resize_rgb(img: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((PANEL_W, PANEL_H), Image.BILINEAR))


def render_episode(video_path: Path, force: np.ndarray, gripper: np.ndarray,
                   dt: float, t_max: float, f_max: float, out_path: Path) -> None:
    reader = imageio.get_reader(str(video_path))
    frames_in = [f for f in reader]
    reader.close()
    T = min(len(frames_in), len(force))
    t = np.arange(T) * dt

    writer = imageio.get_writer(str(out_path), fps=FPS, codec="libx264",
                                output_params=["-crf", "20"])
    for step in range(T):
        rgb_img = _resize_rgb(frames_in[step])
        force_img = _render_force_panel(t, force, step, t_max, f_max)
        row = np.concatenate([rgb_img, force_img], axis=1)
        header = _make_header(step, T, float(gripper[step]))
        frame = np.concatenate([header, row], axis=0)
        writer.append_data(frame)
    writer.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="collect_demos_synth.py output dir (has data.pkl + videos/)")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--random", action="store_true",
                    help="randomly select --n-episodes episodes instead of the first N")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --random selection")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--rate-hz", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.run_dir / "showcase")
    out_dir.mkdir(parents=True, exist_ok=True)

    d = pickle.load(open(args.run_dir / "data.pkl", "rb"))
    episodes = d["episodes"]
    # videos/ is named ep0001_env.._success.mp4, ep0002_..., in SAVE order — sorted()
    # on the zero-padded names matches episodes[] index order 1:1 by construction
    # (collect_demos_synth(_v2).py appends the episode dict and writes its video in
    # the same loop iteration).
    videos = sorted((args.run_dir / "videos").glob("ep*_success.mp4"))
    n_avail = min(len(episodes), len(videos))
    n = min(args.n_episodes, n_avail)

    if args.random:
        rng = np.random.default_rng(args.seed)
        picks = sorted(rng.choice(n_avail, size=n, replace=False).tolist())
    else:
        picks = list(range(n))
    print(f"Pairing {n} episode(s) with videos from {args.run_dir} "
          f"({'random, seed='+str(args.seed) if args.random else 'first N'}): {picks}")

    dt = 1.0 / args.rate_hz
    # Shared scales across all shown episodes, so grip strength is visually comparable.
    all_forces = {idx: np.asarray(episodes[idx]["observations"]["priv_contact_force"],
                                  np.float32).reshape(-1) for idx in picks}
    t_max = max(len(f) for f in all_forces.values()) * dt
    f_max = max(f.max() for f in all_forces.values())
    print(f"  shared x-axis: [0, {t_max:.2f}s]   shared y-axis: [0, {f_max:.1f}N]")

    pad = max(3, len(str(max(picks))))
    for idx in picks:
        force = all_forces[idx]
        gripper = np.asarray(episodes[idx]["observations"]["gripper_width"],
                             np.float32).reshape(-1)
        out_path = out_dir / f"showcase_ep{idx:0{pad}d}.mp4"
        print(f"  ep {idx}: {videos[idx].name} -> {out_path.name}")
        render_episode(videos[idx], force, gripper, dt, t_max, f_max, out_path)

    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
