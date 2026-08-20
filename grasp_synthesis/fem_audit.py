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


def measure_regret(mesh: str, obj_com, obj_quat, settings, ref, *, n_poses: int = 5,
                   maxfevals: int = 600, n_starts: int = 4, seed: int = 0) -> list:
    """REGRET of planning on a coarse mesh — the decision-relevant measurement.

    Rank correlation over a stratified candidate set is a harsher test than the decision needs:
    CMA-ES never reports a ranking, it reports one WINNER. What matters is whether the grasp chosen
    on a coarse mesh is still good when judged properly. So for each object pose:

        x_C = argmax score_C   (plan at the coarse resolution)
        x_F = argmax score_F   (plan at the fine reference)
        regret = score_F(x_F) - score_F(x_C)     >= 0, in fine-mesh score units

    A coarse resolution with poor rank correlation but near-zero regret is perfectly usable — it
    means the near-optimal set is broad and it does not matter which member you land on. Only
    non-trivial regret actually costs anything.
    """
    from smgrasp import width_grasp as wg  # noqa: F401  (ensures the module is importable)

    fine_obj, fine_pad, fine_meta = fg.build_grasp_fem(mesh, voxel_div=ref[0], target_tets=ref[1],
                                                       use_gpu=False)
    fine_sdf = fg.build_object_sdf(fine_obj)
    rng = np.random.default_rng(seed)
    # a few plausible resting poses: jitter xy like the pose DR does
    poses = [(np.asarray(obj_com, float) + np.r_[rng.uniform(-0.03, 0.03, 2), 0.0], obj_quat)
             for _ in range(n_poses)]

    def plan(o, p, com, quat, s):
        return fg.synthesize_grasp(o, p, com, quat, maxfevals=maxfevals, n_starts=n_starts,
                                   seed=s, table_z=0.0)

    def fine_score(x, com, quat):
        r = fg.score_finger_grasp(fine_obj, x, obj_com=com, obj_quat_wxyz=quat, pad_geo=fine_pad,
                                  E=3e5, density=1000.0, mu=0.7, table_z=0.0, obj_sdf=fine_sdf)
        return r["score"], r.get("stress_top10")

    # Plan the fine reference ONCE per pose and reuse it for every resolution row — it does not
    # depend on the coarse setting, and re-planning it per row would triple the CMA work.
    fine_ref = {}
    for k, (com, quat) in enumerate(poses):
        xf = plan(fine_obj, fine_pad, com, quat, 1000 + k)["x"]
        fine_ref[k] = (xf, *fine_score(xf, com, quat)) if xf is not None else None
    ref_scale = float(np.mean([abs(v[1]) for v in fine_ref.values() if v and np.isfinite(v[1])]))

    rows = []
    for vd, tt in settings:
        obj, pad, meta = fg.build_grasp_fem(mesh, voxel_div=vd, target_tets=tt, use_gpu=False)
        regrets, dstress = [], []
        for k, (com, quat) in enumerate(poses):
            if fine_ref[k] is None:
                continue
            _, sf, stf = fine_ref[k]
            xc = plan(obj, pad, com, quat, 1000 + k)["x"]   # SAME seed as the reference plan
            if xc is None:
                continue
            sc, stc = fine_score(xc, com, quat)
            if not (np.isfinite(sc) and np.isfinite(sf)):
                continue
            regrets.append(sf - sc)                       # >= 0 up to CMA noise
            if stc is not None and stf is not None:
                dstress.append(float(stc - stf))          # Pa the coarse choice costs
        rows.append(dict(voxel_div=vd, tets=meta["tets"], n=len(regrets),
                         regret_mean=float(np.mean(regrets)) if regrets else float("nan"),
                         regret_max=float(np.max(regrets)) if regrets else float("nan"),
                         ref_scale=ref_scale,
                         dstress_mean=float(np.mean(dstress)) if dstress else float("nan")))
        r = rows[-1]
        print(f"  voxel_div={vd:3d} tets={meta['tets']:6d}  n={r['n']}  "
              f"regret mean {r['regret_mean']:+10.1f} max {r['regret_max']:+10.1f}  "
              f"(stress cost {r['dstress_mean']:+8.1f} Pa)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regret", action="store_true",
                    help="measure the REGRET of planning coarse (re-score the coarsely-chosen grasp "
                         "on the fine mesh) instead of rank correlation over a fixed candidate set. "
                         "This is the decision-relevant quantity: CMA reports a winner, not a ranking.")
    ap.add_argument("--n-poses", type=int, default=5, help="object poses to plan on (--regret)")
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

    if args.regret:
        ref = settings[-1]
        if args.reference:
            ref = tuple(int(v) for v in args.reference.split(":"))
        print(f"[fem_audit] REGRET of planning coarse, vs reference voxel_div {ref[0]} "
              f"({args.n_poses} poses)")
        rows = measure_regret(args.mesh, com, quat, [s for s in settings if s != ref], ref,
                              n_poses=args.n_poses, seed=args.seed)
        out = args.out or (ROOT / "logs" / "fem_audit" / f"{mesh_name}_regret.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"mesh": str(args.mesh), "reference": list(ref),
                                   "rows": rows}, indent=2))
        print(f"\n  regret is in fine-mesh SCORE units; compare against |score| ~ "
              f"{rows[0]['ref_scale']:.0f} if any row is non-zero.")
        print(f"[fem_audit] wrote {out}")
        return

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
