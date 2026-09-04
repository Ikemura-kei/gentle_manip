#!/usr/bin/env python3
"""DEV TOOL — inspect what the CMA-ES is SEEDED with, and how it then evolves.

Answers two questions the collection logs cannot:
  1. WHERE do the seeds sit?  -> a figure of every antipodal pair sampled (or the yaw fan),
     drawn on the object, with the axis and width each one implies.
  2. WHAT does CMA do with them? -> the per-start score trace, so a start that begins infeasible
     (shaped penalty) and never recovers is visible as a flat line.

No Genesis, no MPM — pure FEM/geometry, so it runs in seconds and is safe to iterate on.

    uv run --project envs/sim python grasp_synthesis/inspect_seeds.py mushroom --antipodal
    uv run --project envs/sim python grasp_synthesis/inspect_seeds.py mushroom            # yaw fan
    uv run --project envs/sim python grasp_synthesis/inspect_seeds.py cube3_soft --antipodal \
        --n-starts 6 --maxfevals 800 --out .agent_tmp/seeds_cube.png
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "grasp_synthesis"))
from gentle_manip.assets.registry import OBJECT_MAP           # noqa: E402
from smgrasp import finger_grasp as fg                        # noqa: E402
from smgrasp.viz import boundary_faces                        # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("object")
    p.add_argument("--antipodal", action="store_true", help="antipodal seeds instead of the yaw fan")
    p.add_argument("--n-starts", type=int, default=3)
    p.add_argument("--maxfevals", type=int, default=800)
    p.add_argument("--mu", type=float, default=0.7)
    p.add_argument("--table-z", type=float, default=0.0)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--n-show", type=int, default=40, help="antipodal pairs to DRAW (sampling is richer)")
    a = p.parse_args()

    od = OBJECT_MAP[a.object]; mat = od.material
    obj, pad_geo, meta = fg.build_grasp_fem(od.mesh_path, voxel_div=14, target_tets=1500,
                                            use_gpu=a.gpu, nu=mat.poisson_ratio)
    com = np.array([0.30, 0.0, 0.0298]); quat = np.array([1.0, 0, 0, 0])
    print(f"{a.object}: tets={meta['tets']} ndof={meta['ndof']} gpu={meta['gpu']}")

    # ── 1. the seeds ────────────────────────────────────────────────────────────────
    pairs = fg.antipodal_seed_pairs(obj, a.n_show, mu=a.mu, rng_seed=0) if a.antipodal else []
    if a.antipodal:
        ax3 = np.array([q[1] for q in pairs]); w3 = np.array([q[2] for q in pairs]) * 1000
        vert = np.abs(ax3[:, 2]); horiz = np.linalg.norm(ax3[:, :2], axis=1)
        print(f"antipodal pairs drawn: {len(pairs)}")
        print(f"  |axis_z| med {np.median(vert):.2f}  >0.5: {100*np.mean(vert>0.5):.0f}%")
        print(f"  3D width med {np.median(w3):.1f} mm   horizontal span med "
              f"{np.median(w3*horiz):.1f} mm  <- the top-down gripper can only use THIS")

    # ── 2. the optimisation trace ───────────────────────────────────────────────────
    trace = {"n": 0, "best": [], "cur": []}
    _orig = fg._score_finger_grasp_impl
    best = [-np.inf]
    def _traced(*args, **kw):
        r = _orig(*args, **kw)
        s = float(r.get("score", -np.inf))
        best[0] = max(best[0], s)
        trace["n"] += 1; trace["cur"].append(s); trace["best"].append(best[0])
        return r
    fg._score_finger_grasp_impl = _traced
    t0 = time.perf_counter()
    out = fg.synthesize_grasp(obj, pad_geo, com, quat, E=mat.youngs_modulus,
                              density=mat.density, mu=a.mu, table_z=a.table_z,
                              maxfevals=a.maxfevals, n_starts=a.n_starts, seed=0,
                              antipodal_seeds=a.antipodal)
    fg._score_finger_grasp_impl = _orig
    dt = time.perf_counter() - t0
    print(f"synthesis {dt:.1f}s  FEM calls {trace['n']}  final score {out.get('score', float('nan')):.4f}")
    print(f"  chosen x = {np.round(out['x'], 4)}   width {1000*out['x'][6]:.1f} mm")

    # ── figure ──────────────────────────────────────────────────────────────────────
    tri, _ = boundary_faces(obj.tets)
    V = obj.verts
    fig = plt.figure(figsize=(16, 5.5)); fig.patch.set_facecolor("white")
    for k, (e, az, ttl) in enumerate([(90, -90, "TOP (xy)"), (0, -90, "FRONT (xz)")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.plot_trisurf(V[:, 0]*1000, V[:, 1]*1000, V[:, 2]*1000, triangles=tri,
                        color=(.75, .75, .8), alpha=.25, linewidth=0, shade=True)
        for mid, axis, w in pairs:
            h = 0.5 * w * np.asarray(axis)
            seg = np.array([mid - h, mid + h]) * 1000
            vz = abs(axis[2])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], lw=1.4,
                    color=("tab:red" if vz > 0.5 else "tab:green"), alpha=.85)
        ax.view_init(elev=e, azim=az); ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("x mm"); ax.set_ylabel("y mm")
    # Penalty scores (~-1e8) are orders below real grasps (~-1e4), so a linear axis hides the
    # actual optimisation. Split: mark infeasible calls on a rug, plot only FEASIBLE scores.
    ax = fig.add_subplot(1, 3, 3)
    cur = np.asarray(trace["cur"], float)
    feas = cur > -1e6                                  # everything above the shaped-penalty band
    xs = np.arange(len(cur))
    if feas.any():
        ax.plot(xs[feas], cur[feas], ".", ms=2.0, color="tab:blue", alpha=.55,
                label=f"feasible candidate ({100*feas.mean():.0f}% of calls)")
        bf = np.maximum.accumulate(np.where(feas, cur, -np.inf))
        ax.plot(xs, bf, lw=2.0, color="tab:red", label="best so far")
        lo, hi = np.percentile(cur[feas], 2), cur[feas].max()
        ax.set_ylim(lo - .05 * abs(lo), hi + .05 * abs(hi) + 1e-9)
    ax.plot(xs[~feas], np.full((~feas).sum(), ax.get_ylim()[0]), "|", ms=6,
            color="tab:grey", alpha=.35, label="INFEASIBLE (shaped penalty)")
    per = max(a.maxfevals // a.n_starts, 20)
    for s_ in range(1, a.n_starts):
        ax.axvline(s_ * per, color="k", ls=":", lw=.9)
    ax.set_xlabel(f"FEM scorer call  (dotted = restart, {per}/start)")
    ax.set_ylabel("score (higher = gentler)"); ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"CMA-ES trace — {'ANTIPODAL' if a.antipodal else 'yaw fan'} seeds", fontsize=10)
    print(f"  feasible calls: {100*feas.mean():.0f}%   per start: " +
          " ".join(f"s{i}={100*feas[i*per:(i+1)*per].mean():.0f}%" for i in range(a.n_starts)))
    seedlbl = "antipodal" if a.antipodal else "yawfan"
    fig.suptitle(f"{a.object} — {seedlbl} seeds, n_starts={a.n_starts}, maxfevals={a.maxfevals}  "
                 f"(GREEN = usable by a top-down gripper, RED = mostly-vertical pair)", fontsize=11)
    out_p = a.out or f".agent_tmp/seeds_{a.object}_{seedlbl}.png"
    Path(out_p).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_p, dpi=110)
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
