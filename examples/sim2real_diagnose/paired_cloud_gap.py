#!/usr/bin/env python3
"""Quantify the sim<->real point-cloud gap from replay_demo_in_sim.py --save-clouds output,
and measure whether an augmentation config CLOSES it.

Metrics (per frame, then aggregated):
  centroid offset  — a RIGID mismatch (extrinsic / object placement), fixable by calibration
  extent ratio     — scale mismatch (fov / focal), fixable by intrinsics
  chamfer          — overall shape agreement AFTER removing the rigid part, i.e. what
                     augmentation can actually influence
  z-profile        — where in height the clouds disagree (board vs object vs gripper)

Reporting chamfer alone is misleading: a pure translation inflates it while telling you nothing
about noise realism. So the rigid part is reported separately and also removed.

    uv run --project envs/sim python examples/sim2real_diagnose/paired_cloud_gap.py \
        --npz <dir>/paired_clouds.npz --augmentation gentle_manip/configs/augmentation/d435i_noise.yaml
"""
import argparse, sys
from pathlib import Path
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gentle_manip.perception.augmentation import AugmentationConfig, ObsAugmentor


def _valid(c):
    return c[np.any(c != 0.0, axis=-1)]


def chamfer(a, b, cap=4000, rng=None):
    """Symmetric mean nearest-neighbour distance (m). Subsampled for tractability."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    rng = rng or np.random.default_rng(0)
    if len(a) > cap:
        a = a[rng.choice(len(a), cap, replace=False)]
    if len(b) > cap:
        b = b[rng.choice(len(b), cap, replace=False)]
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def stats(real, sim, n_frames, aug=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.linspace(0, len(real) - 1, min(n_frames, len(real))).astype(int)
    cen, ext, ch, ch_al = [], [], [], []
    for i in idx:
        r, s = _valid(real[i]), _valid(sim[i])
        if aug is not None:
            s = _valid(aug({"point_cloud": sim[i][None].copy()})["point_cloud"][0])
        if len(r) < 10 or len(s) < 10:
            continue
        dc = s.mean(0) - r.mean(0)
        cen.append(dc)
        ext.append((s.max(0) - s.min(0)) / np.maximum(r.max(0) - r.min(0), 1e-6))
        ch.append(chamfer(r, s, rng=rng))
        ch_al.append(chamfer(r, s - dc, rng=rng))      # rigid part removed
    cen, ext = np.array(cen), np.array(ext)
    return dict(n=len(ch),
                centroid_mm=cen.mean(0) * 1000, centroid_norm_mm=np.linalg.norm(cen, axis=1).mean() * 1000,
                extent_ratio=ext.mean(0),
                chamfer_mm=np.nanmean(ch) * 1000, chamfer_aligned_mm=np.nanmean(ch_al) * 1000)


def show(tag, s):
    print(f"{tag:26s} centroid |d|={s['centroid_norm_mm']:6.2f} mm "
          f"(dx {s['centroid_mm'][0]:+.2f} dy {s['centroid_mm'][1]:+.2f} dz {s['centroid_mm'][2]:+.2f})  "
          f"extent {np.round(s['extent_ratio'],3)}  "
          f"chamfer {s['chamfer_mm']:.2f} mm  aligned {s['chamfer_aligned_mm']:.2f} mm  (n={s['n']})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--augmentation", type=Path, default=None)
    p.add_argument("--sweep", action="store_true",
                   help="sweep pc_axial_coeff to show which value best matches real noise")
    a = p.parse_args()
    d = np.load(a.npz)
    real, sim = d["real"], d["sim"]
    print(f"{a.npz}\n  real {real.shape}  sim {sim.shape}\n")
    base = stats(real, sim, a.frames)
    show("SIM RAW (no aug)", base)
    if a.augmentation:
        cfg = AugmentationConfig.from_dict(yaml.safe_load(a.augmentation.read_text()))
        show(f"SIM + {a.augmentation.name}", stats(real, sim, a.frames, ObsAugmentor(cfg)))
    if a.sweep:
        print()
        cam = (0.69895466, -0.06643963, 0.31680030)
        for c in (0.0, 1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3):
            cfg = AugmentationConfig(pc_axial_coeff=c, pc_axial_cam_pos=cam, seed=0)
            show(f"axial_coeff={c:.4f}", stats(real, sim, a.frames, ObsAugmentor(cfg)))


if __name__ == "__main__":
    main()
