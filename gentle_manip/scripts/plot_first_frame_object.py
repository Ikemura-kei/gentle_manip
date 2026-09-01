"""Visualize the first-frame object crop: episodes WITH outliers (before/after) and clean ones.

Companion to label_first_frame_object.py. Recomputes the crop + connected-component split from
the source slice so the BEFORE panel shows exactly what was rejected -- the label npz stores only
the kept points, so a picture drawn from it alone could not show what filtering removed.

Each row = one episode: LEFT = cropped cloud with outliers in red (BEFORE), RIGHT = kept object
(AFTER). Top-down (xy) and side (xz) views, shared axes per row so the two panels are comparable.
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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-outlier", type=int, default=4, help="episodes WITH outliers to draw")
    ap.add_argument("--n-clean", type=int, default=3, help="clean episodes to draw")
    ap.add_argument("--z-max", type=float, default=None)
    ap.add_argument("--voxel", type=float, default=0.01)
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = np.load(args.slice / "first_frame_object.npz")
    zmax = float(args.z_max if args.z_max is not None else lab["z_max"])
    d = np.load(args.slice / "train.npz")
    tl = d["traj_lengths"]; off = np.concatenate([[0], np.cumsum(tl)[:-1]]).astype(int)
    pc = d["point_cloud"]
    ou = lab["n_outlier"]

    bad = np.argsort(-ou)[:args.n_outlier]
    bad = [i for i in bad if ou[i] > 0]
    clean = [i for i in np.where((ou == 0) & (lab["obj_n"] > 0))[0][:args.n_clean]]
    rows = [(i, "OUTLIERS") for i in bad] + [(i, "clean") for i in clean]
    if not rows:
        print("nothing to draw"); return 1

    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.1 * len(rows)), squeeze=False)
    for r, (ep, kind) in enumerate(rows):
        p = pc[off[ep]]; p = p[np.any(p != 0, axis=1)]
        c = p[p[:, 2] < zmax]
        m = largest_component(c, args.voxel) if len(c) else np.zeros(0, bool)
        obj, out = c[m], c[~m]
        for col, (proj, lbl) in enumerate([((0, 1), "xy (top)"), ((0, 2), "xz (side)")]):
            a, b = proj
            axL, axR = axes[r][col * 2], axes[r][col * 2 + 1]
            axL.scatter(obj[:, a], obj[:, b], s=5, c="#1f77b4", label=f"kept {len(obj)}")
            if len(out):
                axL.scatter(out[:, a], out[:, b], s=22, c="red", marker="x",
                            label=f"outlier {len(out)}")
            axR.scatter(obj[:, a], obj[:, b], s=5, c="#1f77b4")
            for ax in (axL, axR):
                ax.set_xlim(c[:, a].min() - .01, c[:, a].max() + .01)
                ax.set_ylim(c[:, b].min() - .01, c[:, b].max() + .01)
                ax.set_aspect("equal"); ax.tick_params(labelsize=6)
            axL.set_title(f"ep{ep} {kind} — BEFORE  {lbl}", fontsize=8)
            axR.set_title(f"ep{ep} — AFTER  {lbl}", fontsize=8)
            axL.legend(fontsize=6, loc="upper right")
    fig.suptitle(f"{args.slice.name}: first-frame object crop (z < {zmax*100:.0f} cm), "
                 f"outliers = points outside the largest {args.voxel*100:.0f} cm voxel component",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(bad)} outlier rows, {len(clean)} clean rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
