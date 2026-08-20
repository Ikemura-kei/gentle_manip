"""Merge the new real collection + smoke into one 55-ep dataset, then analyze the grasp pose per
episode (pose at the descent bottom = grasp moment) + aggregate distribution + a summary figure."""
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRCS = ["dataset/demos/single_lift_mushroom_real/26-08-20-cmh/data.pkl",
        "dataset/demos/single_lift_mushroom_real_smoke/26-08-20-tdn/data.pkl"]
OUT_PKL = Path("dataset/demos/single_lift_mushroom_real_merged/data.pkl")
OUT_FIG = Path("dataset/demos/single_lift_mushroom_real_merged/grasp_analysis.png")

eps = []
for s in SRCS:
    b = pickle.load(open(s, "rb")); eps += b["episodes"]
OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
pickle.dump({"meta": {"task": "single_lift_mushroom_real", "source": "cmh+smoke merged",
                      "n_episodes": len(eps)}, "episodes": eps}, open(OUT_PKL, "wb"))
print(f"merged {len(eps)} episodes -> {OUT_PKL}")

rows = []
for i, ep in enumerate(eps):
    o = ep["observations"]; T = len(ep["actions"])
    ee = np.asarray(o["ee_pos"]).reshape(T, 3)
    q = np.asarray(o["ee_quat"]).reshape(T, 4)                 # wxyz
    gw = np.asarray(o["gripper_width"]).reshape(T, -1)[:, 0]
    g = int(np.argmin(ee[:, 2]))                               # grasp moment = descent bottom
    eul = R.from_quat(q[g][[1, 2, 3, 0]]).as_euler("xyz", degrees=True)
    lift = float(ee[g:, 2].max() - ee[g, 2])                   # lift after the grasp
    grip = float(gw.min())                                     # CLOSED grasp width (jaws on object)
    rows.append(dict(ep=i, gx=ee[g, 0], gy=ee[g, 1], gz=ee[g, 2], grip=grip,
                     roll=eul[0], pitch=eul[1], yaw=eul[2], lift=lift, T=T))

import numpy as np
def col(k): return np.array([r[k] for r in rows])
print("\n=== grasp-pose summary over %d episodes (pose at descent bottom) ===" % len(rows))
for k, lab, u in [("gx", "grasp x", "m"), ("gy", "grasp y", "m"), ("gz", "grasp z", "m"),
                  ("grip", "grip width", "m"), ("roll", "roll", "deg"), ("pitch", "pitch", "deg"),
                  ("yaw", "yaw", "deg"), ("lift", "lift height", "m")]:
    v = col(k)
    print(f"  {lab:12s} mean {v.mean():7.3f}  std {v.std():6.3f}  [{v.min():7.3f}, {v.max():7.3f}] {u}")

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
sc = ax[0, 0].scatter(col("gx"), col("gy"), c=col("gz"), cmap="viridis", s=40)
ax[0, 0].set_title("grasp position (top view)"); ax[0, 0].set_xlabel("x (m)"); ax[0, 0].set_ylabel("y (m)")
ax[0, 0].axis("equal"); fig.colorbar(sc, ax=ax[0, 0], label="z (m)")
ax[0, 1].hist(col("gz"), bins=20, color="steelblue"); ax[0, 1].set_title("grasp height z (m)")
ax[0, 2].hist(col("grip"), bins=20, color="darkorange"); ax[0, 2].set_title("grasp gripper width (m)")
for k, c, lab in [("roll", "r", "roll"), ("pitch", "g", "pitch"), ("yaw", "b", "yaw")]:
    ax[1, 0].hist(col(k), bins=20, alpha=0.5, color=c, label=lab)
ax[1, 0].legend(); ax[1, 0].set_title("grasp orientation (deg)")
ax[1, 1].hist(col("lift"), bins=20, color="seagreen"); ax[1, 1].set_title("lift height after grasp (m)")
ax[1, 1].axvline(0.05, color="k", ls="--", label="5cm"); ax[1, 1].legend()
ax[1, 2].axis("off")
txt = (f"episodes: {len(rows)}\n"
       f"grasp z:   {col('gz').mean():.3f} +/- {col('gz').std():.3f} m\n"
       f"grip:      {col('grip').mean():.3f} +/- {col('grip').std():.3f} m\n"
       f"yaw range: [{col('yaw').min():.0f}, {col('yaw').max():.0f}] deg\n"
       f"lift:      {col('lift').mean():.3f} +/- {col('lift').std():.3f} m\n"
       f"lifted >5cm: {(col('lift') > 0.05).sum()}/{len(rows)}")
ax[1, 2].text(0.05, 0.5, txt, fontsize=13, family="monospace", va="center")
fig.suptitle("Real single_lift_mushroom grasp-pose analysis (cmh + smoke, 55 eps)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=110)
print(f"\nsaved figure -> {OUT_FIG}")
print(f"lifted >5cm: {(col('lift') > 0.05).sum()}/{len(rows)}")
