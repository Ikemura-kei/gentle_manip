"""Post-filter a collected demo run: drop PINCH episodes (object dangling from fingertips).

A proper enveloping grasp holds the mushroom cap between the finger pads: at hold, the TCP
sits ~17 mm BELOW the object centre with settle width ~33-40 mm. A pinch grabs the cap rim:
the object dangles BELOW/BESIDE the fingertips (TCP at/above the object centre, large
horizontal offset) with a nearly-closed gripper. Measured on a v3.2 collection (95 eps),
the two populations separate cleanly; the user-flagged pinch video was the top outlier.

Flags an episode when, averaged over the last HOLD_WIN recorded steps:
    vert  = TCP_z - obj_z  >  0 mm            (object centre below the fingertips)
 OR width < 25 mm                             (rim pinch: fingers nearly closed)
 OR (vert > -5 mm AND horiz > 15 mm)          (high + off-axis: top/edge pinch)

Requires `priv_object_pos` in the recorded obs (any superset collection has it).
Writes a NEW run dir `<run>-filt/` (source untouched): filtered data.pkl + config.yaml
provenance + pinch_report.yaml (per-episode metrics + kept/dropped).

    uv run --project envs/sim python -m gentle_manip.scripts.filter_pinch_episodes \
        dataset/demos/single_lift_mushroom_soft/<run>
"""
from __future__ import annotations

import argparse
import glob
import pickle
import shutil
from pathlib import Path

import numpy as np
import yaml

HOLD_WIN = 12
VERT_PINCH = 0.0        # m: TCP above object centre => dangling
WIDTH_PINCH = 0.025     # m: near-closed fingers => rim pinch
VERT_SOFT = -0.005      # m
HORIZ_SOFT = 0.015      # m


def load_episodes(run: Path):
    pkl = run / "data.pkl"
    if pkl.exists():
        d = pickle.load(open(pkl, "rb"))
        return d.get("meta", {}), d["episodes"]
    eps = []
    for f in sorted(glob.glob(str(run / "shard_*.pkl"))):
        s = pickle.load(open(f, "rb"))
        eps.extend(s["episodes"] if isinstance(s, dict) else s)
    return {}, eps


def episode_metrics(ep):
    o = ep["observations"]
    p = np.asarray(o["ee_pos"])
    obj = np.asarray(o["priv_object_pos"])
    w = np.asarray(o["gripper_width"]).reshape(len(p), -1)[:, 0]
    vert = float((p[-HOLD_WIN:, 2] - obj[-HOLD_WIN:, 2]).mean())
    horiz = float(np.linalg.norm(p[-HOLD_WIN:, :2] - obj[-HOLD_WIN:, :2], axis=1).mean())
    width = float(w[-HOLD_WIN:].mean())
    return vert, horiz, width


def is_pinch(vert, horiz, width):
    return (vert > VERT_PINCH) or (width < WIDTH_PINCH) or (vert > VERT_SOFT and horiz > HORIZ_SOFT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>-filt")
    args = ap.parse_args()

    meta, eps = load_episodes(args.run)
    report, keep = [], []
    for i, ep in enumerate(eps):
        vert, horiz, width = episode_metrics(ep)
        pinch = is_pinch(vert, horiz, width)
        report.append({"episode": i, "vert_mm": round(vert * 1000, 1),
                       "horiz_mm": round(horiz * 1000, 1), "width_mm": round(width * 1000, 1),
                       "pinch": bool(pinch)})
        if not pinch:
            keep.append(ep)
    n_drop = len(eps) - len(keep)
    print(f"{args.run}: {len(eps)} episodes, dropping {n_drop} pinches "
          f"({[r['episode'] for r in report if r['pinch']]})")

    out = args.out or args.run.parent / (args.run.name + "-filt")
    out.mkdir(exist_ok=True)
    meta = dict(meta, n_episodes=len(keep), pinch_filtered=n_drop, filter_source=str(args.run))
    with open(out / "data.pkl", "wb") as f:
        pickle.dump({"meta": meta, "episodes": keep}, f)
    src_cfg = args.run / "config.yaml"
    if src_cfg.exists():
        cfg = yaml.safe_load(src_cfg.read_text())
        cfg["description"] = (str(cfg.get("description", "")) +
                              f" | PINCH-FILTERED: {n_drop}/{len(eps)} episodes dropped "
                              f"(filter_pinch_episodes.py; report in pinch_report.yaml)")
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "pinch_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    for aux in ("dr_params.csv", "stats.yaml"):
        if (args.run / aux).exists():
            shutil.copy(args.run / aux, out / aux)
    print(f"saved {out} ({len(keep)} episodes)")


if __name__ == "__main__":
    main()
