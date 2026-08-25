"""Pre-flight gates for a CONVERTED (dppo) absolute-action dataset.

Run this on every `convert_demos` output BEFORE training. Each gate corresponds to a failure
mode that is silent in training loss AND in simulated evaluation, and that has cost us a full
train+deploy cycle at least once:

  derivation  the commanded actions must actually be DERIVED absolute targets, not the raw
              recorded DELTAS written through as if absolute. A delta of ~0 decodes to the
              MIDDLE of each absolute range, so the tell-tale signature is a commanded
              gripper/z pinned near the range midpoint while the demos' ACHIEVED values sit
              somewhere else entirely (v33 bug, 2026-08-25: commanded grip 44 mm = midpoint of
              [0, 88] while the demos hold 80 mm; commanded z 0.252 m = midpoint of the z
              range while achieved was 0.096 m). Needs --demos to compare against.
  seam        euler dim 3 must be free of +/-pi wraps WITHIN an episode. Diff within
              trajectories only: concatenated diffs cross episode boundaries and false-trip on
              diverse end poses (v32: within-episode 0.016 = clean, boundary 1.131).
  lead        commanded targets must LEAD the achieved pose, else BC finds a closed-loop fixed
              point (perfect loss, 0% success).
  dwell       consecutive actions must rarely be near-identical, for the same reason. A small
              number of held frames at the END of an episode is intended (stop supervision) and
              is excluded from the statistic.

    uv run --project envs/dppo python -m gentle_manip.scripts.verify_derived_dataset \
        dataset/dppo/<name> --demos dataset/demos/<task>/<run>
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

MID_TOL = 0.05          # |median - range midpoint| below this (fraction of range) = suspicious
LEAD_MIN_MM = 5.0       # p75 commanded-vs-achieved lead
DWELL_EPS = 0.01        # normalized action difference counted as "no change"
DWELL_MAX = 0.20        # allowed fraction (stop frames push this up by design)
SEAM_MAX = 1.0          # max within-episode jump on the euler dim


def _episodes(npz_dir: Path):
    z = np.load(npz_dir / "train.npz")
    n = np.load(npz_dir / "normalization.npz")
    a = z["actions"]
    raw = (a + 1) / 2 * (n["action_max"] - n["action_min"] + 1e-6) + n["action_min"]
    starts = np.concatenate([[0], np.cumsum(z["traj_lengths"])[:-1]])
    return raw, z["traj_lengths"], starts, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="dppo dataset dir (train.npz + normalization.npz)")
    ap.add_argument("--demos", type=Path, default=None,
                    help="source demo run dir (data.pkl) — enables the DERIVATION gate")
    ap.add_argument("--action-config", type=Path,
                    default=Path("gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml"))
    args = ap.parse_args()

    raw, tl, starts, norm = _episodes(args.dataset)
    import yaml
    from gentle_manip.actions.action_config import ActionConfig
    cfg = ActionConfig.from_dict(yaml.safe_load(args.action_config.read_text()))
    lo = np.asarray(list(cfg.pos_min) + [0.0], float)
    hi = np.asarray(list(cfg.pos_max) + [cfg.gripper_max], float)

    def phys(dim, col):                       # raw [-1,1] -> physical, for pos dims and gripper
        j = dim if dim < 3 else 3
        return (col + 1) / 2 * (hi[j] - lo[j]) + lo[j]

    fails = []
    print(f"dataset: {args.dataset}  ({len(tl)} trajs, {len(raw)} steps)")

    # ── derivation ────────────────────────────────────────────────────────────────────
    if args.demos:
        eps = pickle.load(open(args.demos / "data.pkl", "rb"))["episodes"]
        ach_z = np.concatenate([np.asarray(e["observations"]["ee_pos"])[:, 2] for e in eps])
        ach_g = np.concatenate([np.asarray(e["observations"]["gripper_width"]).reshape(-1)
                                for e in eps])
        cmd_z, cmd_g = phys(2, raw[:, 2]), phys(6, raw[:, 6])
        for name, cmd, ach, rng in (("z", cmd_z, ach_z, (lo[2], hi[2])),
                                    ("gripper", cmd_g, ach_g, (lo[3], hi[3]))):
            mid = 0.5 * (rng[0] + rng[1])
            med = float(np.median(cmd))
            at_mid = abs(med - mid) < MID_TOL * (rng[1] - rng[0])
            offset = abs(med - float(np.median(ach)))
            bad = at_mid and offset > 0.02
            print(f"  derivation[{name}]: commanded median {med:.3f} (range midpoint {mid:.3f}), "
                  f"achieved median {float(np.median(ach)):.3f}, |offset| {offset:.3f}"
                  f"  -> {'FAIL (looks like raw deltas passed through)' if bad else 'ok'}")
            if bad:
                fails.append(f"derivation[{name}]")

        # lead: commanded should lead achieved, but not by centimetres
        n = min(len(cmd_z), len(ach_z))
        lead = np.abs(cmd_z[:n] - ach_z[:n])
        p75 = float(np.percentile(lead, 75)) * 1000
        print(f"  lead(z): p75 |commanded-achieved| {p75:.1f} mm "
              f"-> {'FAIL (no lead)' if p75 < LEAD_MIN_MM else 'FAIL (implausible, >5 cm)' if p75 > 50 else 'ok'}")
        if p75 < LEAD_MIN_MM or p75 > 50:
            fails.append("lead")

    # ── seam (within episodes only) ───────────────────────────────────────────────────
    jumps = [np.abs(np.diff(raw[s:s + l, 3])).max() for s, l in zip(starts, tl) if l > 1]
    j = float(max(jumps))
    print(f"  seam: max WITHIN-episode euler-dim3 jump {j:.3f} -> {'FAIL' if j > SEAM_MAX else 'ok'}")
    if j > SEAM_MAX:
        fails.append("seam")

    # ── dwell (excluding the intended trailing stop frames) ───────────────────────────
    fr = []
    for s, l in zip(starts, tl):
        seg = raw[s:s + l]
        if l > 14:
            seg = seg[:-12]                    # drop the deliberate end-of-episode hold
        if len(seg) > 1:
            fr.append(np.abs(np.diff(seg, axis=0)).max(1) < DWELL_EPS)
    d = float(np.concatenate(fr).mean())
    print(f"  dwell: frac(|dA|<{DWELL_EPS}) interior {d:.3f} -> {'FAIL' if d > DWELL_MAX else 'ok'}")
    if d > DWELL_MAX:
        fails.append("dwell")

    print("\nRESULT:", "PASS" if not fails else f"FAIL ({', '.join(fails)})")
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
