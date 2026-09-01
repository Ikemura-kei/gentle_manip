"""Per-episode OBJECT point-cloud label from the FIRST frame (fixed-height crop).

Motivation (user, 2026-09-01): once the object sits between the fingers its points merge with
the gripper's and become hard to perceive -- measured as an occlusion dip in the width-prediction
head (corr 0.850 at phase 0.0 -> 0.565 at 0.6). The policy sees only ONE cloud frame
(`pc_cond_steps: 1`), so that pre-occlusion view is absent at grasp time. This labels the object
region from frame 0, where it is unoccluded, so it can later be used as conditioning.

CROP: z < `--z-max` (default 0.06 m).
  * The TCP sits just below the finger ends, so finger points fall below the TCP only under
    significant tilt -- a fixed-height crop is therefore sound (user, 2026-09-01).
  * 6 cm is below EVERY observed t=0 EE origin across the 12-object set (minimum 6.61 cm;
    20.8% of episodes start low, at 6.6-14.2 cm, matching `regrasp_prob 0.2`).
  * It is above every object's max-scale height (tallest: prim_lamp 5.98 cm) -- but by only
    ~2 mm, and the +-45 deg pitch/roll DR can stand an elongated object's long axis vertical.
    `truncated_frac` in the report flags episodes where the object likely extends past the crop.

OUTLIER FILTER: OFF by default (user decision, 2026-09-01) -- the height crop alone is enough.

The first version filtered by voxel connected components (1 cm), keeping only the largest. That
was WRONG and is disabled. Evidence:
  * the EE at t=0 sits at a MEDIAN 19.8 cm across all 79 flagged episodes -- 14 cm ABOVE the 6 cm
    ceiling -- so no gripper point can be inside the crop at all (only 2/79 had EE below 8 cm);
  * the rejected clusters sat 3.0-3.5 cm apart on a 5.2 cm prim_lamp, i.e. inside one object;
  * user confirmed by inspecting the rotating 3D renders: "what you rejected are part of the lamp".
So the components were ONE object severed at a thin or occluded waist, and filtering DELETED REAL
OBJECT POINTS. `--outlier-filter` still exists to reproduce the old behaviour; do not enable it
without new evidence.

Writes `<out>/first_frame_object.npz` per slice:
    obj_points   (n_ep, max_points, 3) float32, zero-padded
    obj_n        (n_ep,) int32     -- points kept after crop (+filter)
    n_outlier    (n_ep,) int32     -- points dropped as outliers
    n_cropped    (n_ep,) int32     -- points in the crop BEFORE outlier filtering
    truncated    (n_ep,) bool      -- object plausibly extends above the crop (see report)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np


def largest_component(pts: np.ndarray, voxel: float = 0.01):
    """Index mask of the largest 26-connected voxel component. Empty/1-point -> all True."""
    from scipy import ndimage
    if len(pts) <= 1:
        return np.ones(len(pts), bool)
    ijk = np.floor((pts - pts.min(0)) / voxel).astype(np.int64)
    dims = ijk.max(0) + 1
    if np.prod(dims) > 4_000_000:                      # pathological spread -> fall back
        return np.ones(len(pts), bool)
    grid = np.zeros(dims, bool)
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    if n <= 1:
        return np.ones(len(pts), bool)
    sizes = ndimage.sum(grid, lab, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]] == keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slices", nargs="+", type=Path, help="converted dppo slice dirs (g12_*)")
    ap.add_argument("--z-max", type=float, default=0.06)
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--max-points", type=int, default=256)
    ap.add_argument("--outlier-filter", action="store_true",
                    help="re-enable the largest-connected-component filter (OFF by default: it "
                         "severs objects rather than removing gripper points -- see the docstring)")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    print(f"crop z < {args.z_max:.3f} m | voxel {args.voxel:.3f} m | cap {args.max_points} pts\n")
    print(f"{'slice':<18}{'eps':>5}{'kept':>8}{'crop':>8}{'outl':>7}{'ep w/outl':>11}{'trunc':>7}")
    grand = {}
    for s in args.slices:
        f = s / f"{args.split}.npz"
        if not f.exists():
            print(f"{s.name:<18} MISSING {f}"); continue
        d = np.load(f)
        tl = d["traj_lengths"]
        off = np.concatenate([[0], np.cumsum(tl)[:-1]]).astype(int)
        pc = d["point_cloud"]
        n_ep = len(tl)
        OP = np.zeros((n_ep, args.max_points, 3), np.float32)
        ON = np.zeros(n_ep, np.int32); OU = np.zeros(n_ep, np.int32)
        NC = np.zeros(n_ep, np.int32); TR = np.zeros(n_ep, bool)
        for i, e in enumerate(off):
            p = pc[e]
            p = p[np.any(p != 0, axis=1)]                 # drop zero padding
            c = p[p[:, 2] < args.z_max]
            NC[i] = len(c)
            if len(c) == 0:
                continue
            if args.outlier_filter:
                m = largest_component(c, args.voxel)
                OU[i] = int((~m).sum())
                obj = c[m]
            else:
                obj = c                      # height crop only: every cropped point is object
                OU[i] = 0
            # truncation flag: object touching the crop ceiling => it likely continues above
            TR[i] = bool((obj[:, 2] > args.z_max - 0.005).mean() > 0.05)
            if len(obj) > args.max_points:
                obj = obj[np.random.default_rng(0).choice(len(obj), args.max_points, replace=False)]
            ON[i] = len(obj)
            OP[i, :len(obj)] = obj
        np.savez_compressed(s / "first_frame_object.npz", obj_points=OP, obj_n=ON,
                            n_outlier=OU, n_cropped=NC, truncated=TR, z_max=args.z_max)
        nw = int((OU > 0).sum())
        print(f"{s.name:<18}{n_ep:>5}{ON.mean():>8.1f}{NC.mean():>8.1f}{OU.mean():>7.2f}"
              f"{nw:>6} ({100*nw/n_ep:4.1f}%){int(TR.sum()):>7}")
        grand[s.name] = dict(n_ep=n_ep, with_outlier=nw, trunc=int(TR.sum()),
                             empty=int((ON == 0).sum()))
    tot = sum(v["n_ep"] for v in grand.values())
    print(f"\ntotal episodes {tot} | with outliers {sum(v['with_outlier'] for v in grand.values())}"
          f" | truncated {sum(v['trunc'] for v in grand.values())}"
          f" | EMPTY crop {sum(v['empty'] for v in grand.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
