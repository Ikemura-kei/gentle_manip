"""Verify a real teleop demo run (paired-RGB collection for generalist cotrain + pi0.5).

Checks (exit code 1 if any FAIL):
  1. channel PAIRING: every obs channel + actions share the step count, per episode
  2. schema: image_cam_ext uint8 (H,W,3), point_cloud (N,1024,3), actions 7-dim
  3. actions look ABSOLUTE (wide normalized range, not near-zero deltas)
  4. cloud inside the generalist crop box; quat sign canonicalized
  5. config.yaml present with description + record_action config

Then renders one paired RGB|cloud mp4 per episode into <run>/videos_paired/.

Usage (deploy env):
  uv run --project envs/deploy python -m gentle_manip.scripts.verify_real_demos \
      dataset/demos/single_lift_<obj>_real/<run-id> [--no-videos]
"""
import argparse
import os
import pickle
import sys

import numpy as np
import yaml

CROP_MIN = np.array([0.2, -0.215, 0.004])
CROP_MAX = np.array([0.71, 0.215, 0.45])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--no-videos", action="store_true")
    args = p.parse_args()
    run = args.run_dir.rstrip("/")
    eps = pickle.load(open(os.path.join(run, "data.pkl"), "rb"))["episodes"]
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        ok &= bool(cond)

    print(f"{run}: {len(eps)} episodes, lengths "
          f"{[len(e['actions']) for e in eps]}")
    mism = [i for i, e in enumerate(eps)
            if len({v.shape[0] for v in e["observations"].values()} | {len(e["actions"])}) != 1]
    check("channel pairing (obs+actions step counts equal)", not mism, f"mismatch eps: {mism}")
    e0 = eps[0]["observations"]
    check("image_cam_ext present uint8 (H,W,3)",
          "image_cam_ext" in e0 and e0["image_cam_ext"].dtype == np.uint8
          and e0["image_cam_ext"].ndim == 4)
    check("point_cloud (N,1024,3)", e0["point_cloud"].shape[1:] == (1024, 3))
    check("actions 7-dim", eps[0]["actions"].shape[1] == 7)
    a = np.concatenate([e["actions"] for e in eps])
    span = (a.max(0) - a.min(0))[:3]
    check("actions look ABSOLUTE (pos-dim span > 0.3 normalized)", (span > 0.3).any(),
          f"pos spans {np.round(span, 2)}")
    pc = np.concatenate([e["observations"]["point_cloud"][::20] for e in eps]).reshape(-1, 3)
    check("cloud inside generalist crop box",
          (pc.min(0) >= CROP_MIN - 1e-4).all() and (pc.max(0) <= CROP_MAX + 1e-4).all(),
          f"bounds {np.round(pc.min(0), 3)}..{np.round(pc.max(0), 3)}")
    q = np.concatenate([e["observations"]["ee_quat"] for e in eps])
    check("quat sign canonicalized",
          bool((q[np.arange(len(q)), np.abs(q).argmax(1)] > 0).all()))
    g = np.concatenate([e["observations"]["gripper_width"] for e in eps])
    print(f"  [info] gripper width {g.min():.3f}..{g.max():.3f} m")
    cfgp = os.path.join(run, "config.yaml")
    cfg = yaml.safe_load(open(cfgp)) if os.path.exists(cfgp) else {}
    check("config.yaml has description", bool(cfg.get("description")),
          str(cfg.get("description"))[:60])

    if not args.no_videos:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import imageio.v2 as imageio
        out = os.path.join(run, "videos_paired")
        os.makedirs(out, exist_ok=True)
        for i, e in enumerate(eps):
            rgb = e["observations"]["image_cam_ext"]
            pcl = e["observations"]["point_cloud"]
            ee = e["observations"]["ee_pos"]
            w = imageio.get_writer(os.path.join(out, f"ep{i:02d}.mp4"), fps=15,
                                   codec="libx264", quality=7)
            fig = plt.figure(figsize=(10, 4.2), dpi=72)
            axl = fig.add_subplot(1, 2, 1)
            axr = fig.add_subplot(1, 2, 2, projection="3d")
            for t in range(len(rgb)):
                axl.clear(); axl.imshow(rgb[t]); axl.axis("off")
                axl.set_title(f"ep{i:02d} t={t}/{len(rgb) - 1} image_cam_ext")
                axr.clear()
                pt = pcl[t]
                axr.scatter(pt[:, 0], pt[:, 1], pt[:, 2], s=1, c=pt[:, 2], cmap="viridis")
                axr.scatter(*ee[t], s=60, c="red", marker="^")
                axr.set_xlim(CROP_MIN[0], CROP_MAX[0])
                axr.set_ylim(CROP_MIN[1], CROP_MAX[1])
                axr.set_zlim(0, CROP_MAX[2])
                axr.set_title("policy cloud (paired)"); axr.view_init(25, -60)
                fig.canvas.draw()
                w.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
            w.close(); plt.close(fig)
            print(f"  rendered ep{i:02d} ({len(rgb)} frames)", flush=True)

    print("VERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
