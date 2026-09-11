"""Point-cloud video, from a SIM eval dump or a REAL deploy shard, rendered identically so the two
can be compared side by side.

  python render_cloud.py sim  <obs_b0.npz> <out_prefix>
  python render_cloud.py real <shard.pkl>  <out_prefix>

Top-down (x-y) + side (x-z). The object band (z 15-120 mm, away from the arm) is highlighted and its
principal axis drawn, since that axis is what the grasp yaw has to align across.
"""
import sys, subprocess, tempfile, pathlib
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

kind, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
if kind == "sim":
    pc = np.load(src)["point_cloud"]                       # (T, n_env, N, 3)
    clouds = [pc[:, e] for e in range(pc.shape[1])]
    labels = [f"env{e}" for e in range(pc.shape[1])]
else:
    import pickle
    d = pickle.load(open(src, "rb"))
    eps = d["episodes"] if isinstance(d, dict) and "episodes" in d else d
    clouds, labels = [], []
    for i, e in enumerate(eps):
        o = e.get("observations") or e.get("obs")
        clouds.append(np.asarray(o["point_cloud"], float)); labels.append(f"ep{i}")

def axis_angle(fr):
    m = (fr[:, 2] > 0.015) & (fr[:, 2] < 0.12) & (fr[:, 0] > 0.30) & (fr[:, 0] < 0.55)
    p = fr[m]
    if len(p) < 20: return None, p
    xy = p[:, :2] - p[:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov(xy.T))
    return (np.degrees(np.arctan2(V[1, -1], V[0, -1])), np.sqrt(w[-1] / max(w[0], 1e-12))), p

for c, lab in zip(clouds, labels):
    tmp = pathlib.Path(tempfile.mkdtemp())
    for t in range(len(c)):
        fr = c[t]
        res, obj = axis_angle(fr)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6)); fig.patch.set_facecolor("white")
        for k, (i, j, xl, yl) in enumerate([(0, 1, "x (m)", "y (m)"), (0, 2, "x (m)", "z (m)")]):
            ax[k].scatter(fr[:, i], fr[:, j], s=1.2, c="#b8c2cc", linewidths=0)
            if len(obj): ax[k].scatter(obj[:, i], obj[:, j], s=2.4, c="#2a78d6", linewidths=0)
            ax[k].set_xlabel(xl, fontsize=9); ax[k].set_ylabel(yl, fontsize=9)
            ax[k].set_xlim(0.20, 0.62)
            ax[k].set_ylim(-0.26, 0.26) if k == 0 else ax[k].set_ylim(0.0, 0.40)
            ax[k].set_aspect("equal"); ax[k].grid(color="#eceff1", lw=0.6); ax[k].set_axisbelow(True)
            ax[k].tick_params(labelsize=8)
        if res and len(obj):
            ang, elong = res
            ctr = obj[:, :2].mean(0); d = 0.06 * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
            ax[0].plot([ctr[0]-d[0], ctr[0]+d[0]], [ctr[1]-d[1], ctr[1]+d[1]], c="#eb6834", lw=2.2)
            ax[0].set_title(f"{lab}  t={t}   object axis {ang:.0f}°  elong {elong:.1f}x  n={len(obj)}",
                            fontsize=10, loc="left")
        else:
            ax[0].set_title(f"{lab}  t={t}   (no object cluster)", fontsize=10, loc="left")
        fig.tight_layout(); fig.savefig(tmp / f"f{t:04d}.png", dpi=90); plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10",
                    "-i", str(tmp / "f%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt", "yuv420p", f"{out}_{lab}.mp4"])   # h264 needs even dimensions
    print(f"  wrote {out}_{lab}.mp4  ({len(c)} frames)")
