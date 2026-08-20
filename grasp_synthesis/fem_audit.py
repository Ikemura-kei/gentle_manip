"""FEM audit: is the number we rank grasps by mesh-converged, and what does it cost?

Every gentleness comparison, the whole benchmark, and every objective weight is judged by
`width_grasp_stress`'s output. `grasp_synthesis/CLAUDE.md` §11.6 warns that absolute grip/stress
are mesh-sensitive ("coarse = fast RANKING, fine = trustworthy Pa/N") but no convergence gate
exists for the width-controlled path — the only such test covers the retired Q_SM metric.

What actually matters for synthesis is NOT that absolute stress converges, but that the RANKING of
candidate grasps is stable: CMA-ES only ever compares candidates. So this reports both, and the
selection rule is "the coarsest resolution whose ranking still matches the fine reference".

Usage:
    uv run --project envs/sim python grasp_synthesis/fem_audit.py \
        --mesh gentle_manip/assets/objects/mushroom.obj --n-grasps 40

Outputs a table + JSON to logs/fem_audit/<mesh-stem>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "grasp_synthesis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from smgrasp import finger_grasp as fg  # noqa: E402


def sample_grasp_set(mesh: str, obj_com, obj_quat, n: int, *, seed: int = 0,
                     voxel_div: int = 9, target_tets: int = 600) -> np.ndarray:
    """A DIVERSE, REALISTIC set of candidate grasps to rank.

    Taken from the candidates a real CMA-ES run explored (`record_history=True`), stratified across
    the score range, rather than sampled uniformly from the bounds — the ranking only has to be
    trustworthy among plausible grasps, and a uniform sample is almost entirely infeasible misses.
    """
    obj, pad, _ = fg.build_grasp_fem(mesh, voxel_div=voxel_div, target_tets=target_tets,
                                     use_gpu=False)
    out = fg.synthesize_grasp(obj, pad, obj_com, obj_quat, maxfevals=600, n_starts=4, seed=seed,
                              table_z=0.0, record_history=True)
    hist = [h for h in out.get("history", []) if h.get("holdable") and h.get("stress") is not None]
    if len(hist) < n:                                    # fall back to every scored candidate
        hist = [h for h in out.get("history", []) if h.get("score") is not None]
    if not hist:
        raise RuntimeError("no candidates recorded; cannot build a grasp set")
    hist.sort(key=lambda h: h["score"])
    idx = np.linspace(0, len(hist) - 1, min(n, len(hist))).astype(int)   # stratified over score
    return np.stack([np.asarray(hist[i]["x"], float) for i in idx])


def score_set(mesh: str, grasps, obj_com, obj_quat, *, voxel_div: int, target_tets: int,
              prepare: bool = True, use_gpu: bool = False, E=3e5, density=1000.0, mu=0.7):
    """Score every grasp at one resolution. Returns (scores, stresses, meta, build_s, per_eval_ms)."""
    t0 = time.time()
    obj, pad, meta = fg.build_grasp_fem(mesh, voxel_div=voxel_div, target_tets=target_tets,
                                        prepare=prepare, use_gpu=use_gpu)
    build_s = time.time() - t0
    sdf = fg.build_object_sdf(obj)
    scores, stresses = [], []
    t0 = time.time()
    for x in grasps:
        r = fg.score_finger_grasp(obj, x, obj_com=obj_com, obj_quat_wxyz=obj_quat, pad_geo=pad,
                                  E=E, density=density, mu=mu, table_z=0.0, obj_sdf=sdf)
        scores.append(r["score"])
        st = r.get("stress_top10")
        stresses.append(float(st) if st is not None and np.isfinite(st) else np.nan)
    per_ms = (time.time() - t0) / max(len(grasps), 1) * 1e3
    return np.array(scores), np.array(stresses), meta, build_s, per_ms


def _spearman(a, b) -> float:
    """Rank correlation without scipy.stats (finite entries only)."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _topk_overlap(a, b, k) -> float:
    """Fraction of the reference top-k that this resolution also ranks top-k — the property CMA
    actually depends on (it only needs the good candidates to stay good)."""
    ta = set(np.argsort(a)[::-1][:k].tolist())
    tb = set(np.argsort(b)[::-1][:k].tolist())
    return len(ta & tb) / float(k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default=str(ROOT / "gentle_manip/assets/objects/mushroom.obj"))
    ap.add_argument("--n-grasps", type=int, default=40)
    ap.add_argument("--obj-com", type=float, nargs=3, default=[0.47, 0.0, 0.016])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-prepare", action="store_true",
                    help="skip the watertight voxel remesh, which ROUNDS SHARP EDGES — required for "
                         "an analytic mesh like a cube, wrong for a scanned organic mesh")
    ap.add_argument("--grid", type=str,
                    default="7:400,9:600,11:1000,12:1200,14:1500,16:2500,18:4000",
                    help="comma-separated voxel_div:target_tets settings, coarse -> fine")
    ap.add_argument("--reference", type=str, default=None,
                    help="voxel_div:target_tets to treat as ground truth (default: the finest)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    settings = [tuple(int(v) for v in s.split(":")) for s in args.grid.split(",")]
    com = np.asarray(args.obj_com, float)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    mesh_name = Path(args.mesh).stem

    print(f"[fem_audit] {args.mesh}  prepare={not args.no_prepare}")
    print(f"[fem_audit] building a stratified grasp set (n={args.n_grasps}) from a real CMA run...")
    grasps = sample_grasp_set(args.mesh, com, quat, args.n_grasps, seed=args.seed)
    print(f"[fem_audit] {len(grasps)} candidate grasps\n")

    rows = []
    for vd, tt in settings:
        s, st, meta, build_s, per_ms = score_set(args.mesh, grasps, com, quat, voxel_div=vd,
                                                 target_tets=tt, prepare=not args.no_prepare)
        rows.append(dict(voxel_div=vd, target_tets=tt, tets=meta["tets"], ndof=meta["ndof"],
                         build_s=build_s, per_eval_ms=per_ms, scores=s, stress=st))
        print(f"  voxel_div={vd:3d} target_tets={tt:5d} -> tets={meta['tets']:6d} "
              f"ndof={meta['ndof']:6d}  build={build_s:6.1f}s  {per_ms:8.1f} ms/eval")

    ref = rows[-1]
    if args.reference:
        rvd, rtt = (int(v) for v in args.reference.split(":"))
        ref = next(r for r in rows if r["voxel_div"] == rvd and r["target_tets"] == rtt)
    print(f"\n[fem_audit] reference = voxel_div {ref['voxel_div']} ({ref['tets']} tets)")
    print(f"{'voxel_div':>10} {'tets':>7} {'ms/eval':>9} {'spearman':>9} {'top5':>6} {'top10':>6} "
          f"{'stress_ratio':>13}")
    table = []
    for r in rows:
        rho = _spearman(r["scores"], ref["scores"])
        t5 = _topk_overlap(r["scores"], ref["scores"], min(5, len(grasps)))
        t10 = _topk_overlap(r["scores"], ref["scores"], min(10, len(grasps)))
        m = np.isfinite(r["stress"]) & np.isfinite(ref["stress"])
        ratio = float(np.median(r["stress"][m] / ref["stress"][m])) if m.sum() else float("nan")
        print(f"{r['voxel_div']:>10} {r['tets']:>7} {r['per_eval_ms']:>9.1f} {rho:>9.4f} "
              f"{t5:>6.2f} {t10:>6.2f} {ratio:>13.3f}")
        table.append(dict(voxel_div=r["voxel_div"], target_tets=r["target_tets"], tets=r["tets"],
                          ndof=r["ndof"], per_eval_ms=r["per_eval_ms"], build_s=r["build_s"],
                          spearman_vs_ref=rho, top5_overlap=t5, top10_overlap=t10,
                          median_stress_ratio=ratio))

    print("\n  spearman/top-k = does this resolution RANK grasps like the reference (what CMA needs)")
    print("  stress_ratio   = median absolute stress vs reference (what a reported Pa figure needs)")

    out = args.out or (ROOT / "logs" / "fem_audit" / f"{mesh_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mesh": str(args.mesh), "prepare": not args.no_prepare,
                               "n_grasps": int(len(grasps)), "reference_voxel_div": ref["voxel_div"],
                               "rows": table}, indent=2))
    print(f"\n[fem_audit] wrote {out}")


if __name__ == "__main__":
    main()
