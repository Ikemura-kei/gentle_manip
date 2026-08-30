"""WIDTH-ADAPTATION PROBE, binned over object size.

METRIC (settled 2026-08-27; see docs/CHECKLISTS.md §3): mm of gripper opening per mm of object
size. Regress the AT-GRASP commanded width on object size, ONE POINT PER DISTINCT GEOMETRY, and
report slope + 95% CI + intercept. Demonstrator = 1.08. 0 = constant width.

Retired metrics, do not resurrect: correlation (direction, not magnitude), half-split "% of
demonstrator range" (inflated by a uniform mean shift), per-episode regression (episodes in a
batch SHARE the object, so it triple-counts correlated samples).

BINNED DESIGN (this file): instead of drawing object_scale at random and hoping for spread, we
pin it to 5 levels across the DR range (1.0 .. 1.5) via the `wprobe_*` DR configs, and keep
scene_group_size=1 so shape/material still vary per batch. So each bin contributes SEVERAL
distinct geometries at a FIXED size. Two reasons this beats random sampling here:
  * leverage — slope SE scales as 1/sd(x); pinned extremes maximise sd(x), so fewer geometries
    reach the same precision as ~40 random draws clustered mid-range;
  * de-confounding — under random DR, size and shape co-vary; pinning size isolates the size term.

Inputs: the per-batch dumps `.agent_tmp/<tag>_widthcmd_b*.npz` (written by Pi05EvalPolicy under
GM_WIDTH_DUMP) joined to each run's episodes.csv by (batch, env) for `obj_scale`.
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
TMP = _REPO / ".agent_tmp"
NOMINAL_MM = 33.0        # mushroom nominal extent; 1 unit of scale = this many mm of object
DEMO_SLOPE = 35.7 / NOMINAL_MM      # 1.08 mm/mm


def at_grasp(w: np.ndarray, z: np.ndarray) -> float:
    """Commanded width at the grasp: the minimum width between the lowest EE point and the
    moment the EE has risen 2 cm. Phase-detection-free and measured to be indistinguishable
    from the per-episode minimum (corr 0.474 vs 0.471)."""
    j = int(np.argmin(z))
    r = np.where(z[j:] > z[j] + 0.02)[0]
    return float("nan") if len(r) == 0 else float(w[j:j + r[0] + 1].min())


def collect(tag: str, eval_dir: Path):
    rows = list(csv.DictReader(open(eval_dir / "episodes.csv")))
    W, S = [], []
    files = sorted(glob.glob(str(TMP / f"{tag}_widthcmd_b*.npz")),
                   key=lambda p: int(p.split("_b")[-1][:-4]))
    for b, f in enumerate(files):
        d = np.load(f)
        for e in range(d["width_cmd_mm"].shape[1]):
            m = [x for x in rows if int(x["batch"]) == b and int(x["env"]) == e]
            if not m or not m[0].get("obj_scale"):
                continue
            a = at_grasp(d["width_cmd_mm"][:, e], d["ee_z_m"][:, e])
            if np.isfinite(a):
                W.append(a); S.append(float(m[0]["obj_scale"]))
    return np.array(W), np.array(S)


def regress(W, S):
    """Per-geometry aggregation, then OLS. Returns (k, intercept, slope_mm_per_mm, lo, hi, r2)."""
    uniq = np.unique(np.round(S, 6))
    Sg = uniq
    Wg = np.array([W[np.round(S, 6) == u].mean() for u in uniq])
    k = len(Sg)
    if k < 3:
        return k, *(float("nan"),) * 5
    b1 = np.cov(Sg, Wg)[0, 1] / np.var(Sg)
    b0 = Wg.mean() - b1 * Sg.mean()
    pred = b0 + b1 * Sg
    ss_r = float(((Wg - pred) ** 2).sum()); ss_t = float(((Wg - Wg.mean()) ** 2).sum())
    r2 = 1 - ss_r / ss_t if ss_t > 0 else float("nan")
    se = np.sqrt(ss_r / max(k - 2, 1) / (((Sg - Sg.mean()) ** 2).sum() + 1e-9))
    return (k, b0, b1 / NOMINAL_MM, (b1 - 1.96 * se) / NOMINAL_MM,
            (b1 + 1.96 * se) / NOMINAL_MM, r2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="TAG=EVAL_DIR",
                    help="repeatable; TAG matches GM_WIDTH_DUMP, EVAL_DIR holds episodes.csv. "
                         "Give one per BIN and they are pooled into a single regression.")
    ap.add_argument("--label", default="pi0.5")
    a = ap.parse_args()

    allW, allS = [], []
    print(f"{'bin (obj_scale)':>16}{'#geo':>6}{'#eps':>6}{'mean at-grasp width':>22}{'sd':>8}")
    for spec in a.arm:
        tag, d = spec.split("=", 1)
        W, S = collect(tag, Path(d))
        if len(W) == 0:
            print(f"{tag:>16}{0:>6}{0:>6}{'no data':>22}"); continue
        print(f"{np.unique(S).mean():>16.3f}{len(np.unique(np.round(S,6))):>6}{len(W):>6}"
              f"{W.mean():>22.1f}{W.std():>8.1f}")
        allW.append(W); allS.append(S)
    if not allW:
        sys.exit("no data collected")
    W = np.concatenate(allW); S = np.concatenate(allS)
    k, b0, m, lo, hi, r2 = regress(W, S)
    print(f"\nWIDTH ADAPTATION = mm gripper opening per mm object size.  demonstrator = {DEMO_SLOPE:.2f}")
    print(f"{'arm':<14}{'#geo':>5}{'interc':>9}{'slope':>8}{'  95% CI':>16}{'%demo':>8}{'R2':>7}  verdict")
    if k < 3:
        print(f"{a.label:<14}{k:>5}   too few distinct geometries to regress"); return
    sig = "ADAPTS" if lo > 0.15 else ("uncertain" if hi > 0.15 else "NO ADAPTATION")
    print(f"{a.label:<14}{k:>5}{b0:>9.1f}{m:>8.2f}  [{lo:>5.2f},{hi:>5.2f}]{100*m/DEMO_SLOPE:>8.0f}%{r2:>7.2f}  {sig}")
    if k < 40:
        print(f"\n⚠ {k} distinct geometries. The 40-geometry rule exists because three mechanisms "
              "CHANGED VERDICT between 12 and 40 (DEVLOG 2026-08-27). The binned design buys "
              "leverage, not immunity: treat a borderline CI as unresolved.")
    print("1.00 = opens exactly in proportion to the object; 0 = constant width.")


if __name__ == "__main__":
    main()
