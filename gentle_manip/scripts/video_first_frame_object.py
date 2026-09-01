"""Rotating 3D video of the first-frame object crop — for judging the crop BY EYE.

The 2D projections in plot_first_frame_object.py cannot show whether a rejected cluster is part
of the object or a gripper finger; depth is exactly the axis they collapse. This orbits the
camera 360 deg around the FULL first-frame cloud so the spatial relationship is unambiguous.

Colour code (per frame, all drawn together):
  grey   = points ABOVE the crop ceiling  -> what the z-crop discarded (arm, object top, scene)
  blue   = KEPT as the object             -> the label
  red    = rejected by the component filter ("outliers")
  green  = the END-EFFECTOR origin at t=0  -> is the gripper inside the crop?
  plane  = the crop ceiling at z_max

Reading it: if red sits ON the blue object surface, the filter is severing one object and should
be disabled. If red sits under the green EE marker, it is a finger and the filter is right.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gentle_manip.scripts.label_first_frame_object import largest_component


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slice", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="explicit episode indices; default = worst-outlier ones + clean ones")
    ap.add_argument("--n-outlier", type=int, default=2)
    ap.add_argument("--n-clean", type=int, default=1)
    ap.add_argument("--frames", type=int, default=72, help="rotation frames (72 = 5 deg steps)")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--z-max", type=float, default=None)
    ap.add_argument("--voxel", type=float, default=0.01)
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    lab = np.load(args.slice / "first_frame_object.npz")
    zmax = float(args.z_max if args.z_max is not None else lab["z_max"])
    d = np.load(args.slice / "train.npz")
    n = np.load(args.slice / "normalization.npz")
    tl = d["traj_lengths"]; off = np.concatenate([[0], np.cumsum(tl)[:-1]]).astype(int)
    pc = d["point_cloud"]; st = d["states"]
    lo, hi = n["obs_min"][:3], n["obs_max"][:3]
    ou = lab["n_outlier"]

    if args.episodes:
        eps = [(e, "chosen") for e in args.episodes]
    else:
        bad = [int(i) for i in np.argsort(-ou)[:args.n_outlier] if ou[i] > 0]
        clean = [int(i) for i in np.where((ou == 0) & (lab["obj_n"] > 0))[0][:args.n_clean]]
        eps = [(e, "outliers") for e in bad] + [(e, "clean") for e in clean]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ep, kind in eps:
        p = pc[off[ep]]; p = p[np.any(p != 0, axis=1)]
        below = p[p[:, 2] < zmax]; above = p[p[:, 2] >= zmax]
        m = largest_component(below, args.voxel) if len(below) else np.zeros(0, bool)
        obj, out = below[m], below[~m]
        ee = (st[off[ep], :3] + 1) / 2 * (hi - lo) + lo

        ctr = obj.mean(0) if len(obj) else below.mean(0)
        rad = 0.13
        frames = []
        for k in range(args.frames):
            fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
            if len(above):
                ax.scatter(*above.T, s=3, c="#bbbbbb", alpha=.45, label=f"above crop {len(above)}")
            ax.scatter(*obj.T, s=14, c="#1f77b4", label=f"KEPT object {len(obj)}")
            if len(out):
                ax.scatter(*out.T, s=55, c="red", marker="x", depthshade=False,
                           label=f"rejected {len(out)}")
            ax.scatter(*ee, s=160, c="lime", marker="*", edgecolors="k",
                       depthshade=False, label=f"EE t=0 (z={ee[2]*100:.1f}cm)")
            gx, gy = np.meshgrid(np.linspace(ctr[0]-rad, ctr[0]+rad, 2),
                                 np.linspace(ctr[1]-rad, ctr[1]+rad, 2))
            ax.plot_surface(gx, gy, np.full_like(gx, zmax), alpha=.12, color="orange")
            ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
            ax.set_zlim(0, max(zmax*2.2, ee[2]*1.15))
            ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8); ax.set_zlabel("z", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.view_init(elev=18, azim=k * 360.0 / args.frames)
            ax.set_title(f"{args.slice.name}  ep{ep} ({kind})   crop z<{zmax*100:.0f}cm "
                         f"(orange plane)", fontsize=10)
            ax.legend(fontsize=7, loc="upper left")
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
        o = args.out_dir / f"{args.slice.name}_ep{ep}_{kind}.mp4"
        imageio.mimsave(o, frames, fps=args.fps, macro_block_size=1)
        print(f"  wrote {o}  ({len(frames)} frames, kept {len(obj)}, rejected {len(out)}, "
              f"above-crop {len(above)}, EE z={ee[2]*100:.1f}cm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
