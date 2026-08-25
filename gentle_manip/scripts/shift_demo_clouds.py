"""Write a bias-corrected copy of a REAL demo run: translate the stored point clouds.

Real recordings carry the rig's measured perception bias baked into their clouds (the clouds
are stored post-processing, so the bias cannot be corrected later at training time). This tool
produces a corrected copy so that recordings made before the calibration can be used alongside
simulated data, whose clouds are unbiased by construction.

What it does and does NOT touch:
  * point clouds  translated by --shift, with zero-padded rows left exactly zero
  * proprioception, actions, rewards  UNCHANGED — the bias is perception-side; the arm was
    always where its encoders said it was
  * provenance     the shift, the source run and the git commit are written into the copy's
    config.yaml and meta, so a dataset can never be silently confused with its source

**Deployment pairing rule (important).** A policy trained on corrected clouds must be deployed
with the matching `point_cloud_shift` ACTIVE in the setup config, and one trained on
uncorrected clouds must be deployed with it at zero. Either consistent pairing is fine; a
mismatch reintroduces the full bias. Record which variant a run used in its EXPERIMENT.md.

    uv run --project envs/sim python -m gentle_manip.scripts.shift_demo_clouds \
        dataset/demos/single_lift_mushroom_real_merged --shift 0.009 0 0
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="source demo run dir (data.pkl) — never modified")
    ap.add_argument("--shift", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                    help="translation applied to every valid point, metres, world frame")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir; default <run>_shift<N>mm using the x component")
    args = ap.parse_args()

    shift = np.asarray(args.shift, np.float32)
    out = args.out or args.run.parent / f"{args.run.name}_shift{round(float(shift[0]) * 1000)}mm"
    if out.resolve() == args.run.resolve():
        raise SystemExit("refusing to overwrite the source run")
    out.mkdir(parents=True, exist_ok=True)

    d = pickle.load(open(args.run / "data.pkl", "rb"))
    n_pts = 0
    for ep in d["episodes"]:
        pc = np.asarray(ep["observations"]["point_cloud"])
        valid = ~np.all(pc == 0, axis=2)              # keep zero-padding exactly zero
        pc = pc.copy()
        pc[valid] += shift
        ep["observations"]["point_cloud"] = pc
        n_pts += int(valid.sum())
    d["meta"] = dict(d.get("meta", {}), cloud_shift_applied=shift.tolist(),
                     cloud_shift_source=str(args.run))
    with open(out / "data.pkl", "wb") as f:
        pickle.dump(d, f)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    src_cfg = args.run / "config.yaml"
    cfg = yaml.safe_load(src_cfg.read_text()) if src_cfg.exists() else {}
    cfg["description"] = (str(cfg.get("description", "")) +
                          f" | CLOUD-SHIFTED by {shift.tolist()} m (perception-bias correction; "
                          f"proprio unchanged). Deploy a policy trained on this ONLY with the "
                          f"matching point_cloud_shift active.")
    cfg["cloud_shift_applied"] = shift.tolist()
    cfg["cloud_shift_source"] = str(args.run)
    cfg["cloud_shift_commit"] = commit
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for aux in ("stats.yaml", "dr_params.csv"):
        if (args.run / aux).exists():
            shutil.copy(args.run / aux, out / aux)

    print(f"{args.run} -> {out}")
    print(f"  {len(d['episodes'])} episodes, {n_pts} points shifted by {shift.tolist()} m")
    print("  proprio/actions unchanged; zero-padding preserved")


if __name__ == "__main__":
    main()
