#!/usr/bin/env python3
"""DEV TOOL — see what the CMA-ES grasp search is SEEDED with, and how it then evolves.

Standalone: pure FEM/geometry, NO Genesis and no MPM, so it runs in seconds and is safe to
iterate on. Answers the three things a collection log cannot:

  1. WHERE do the seeds sit?     -> every multi-start seed drawn on the object as the jaw
                                    segment it actually represents (antipodal or medial).
  2. HOW does the pose EVOLVE?   -> the running-best grasp pose at each improvement, early->late,
                                    with the finally-selected grasp on top.
  3. WHY did it end there?       -> the score trace, separating feasible candidates from the
                                    shaped-penalty band.

Seeds and candidates are taken from the planner's OWN records (`record_history=True` returns
`seeds` + `history`) rather than reconstructed here, so what is drawn is what was searched.

    uv run --project envs/sim python grasp_synthesis/inspect_seeds.py tofu
    uv run --project envs/sim python grasp_synthesis/inspect_seeds.py cube3_soft --gpu --anim
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "grasp_synthesis"))
from gentle_manip.assets.registry import OBJECT_MAP           # noqa: E402
from smgrasp import finger_grasp_final as fg                  # noqa: E402
from smgrasp.viz import boundary_faces                        # noqa: E402

VIEWS = [(90, -90, "TOP (xy)"), (2, -90, "FRONT (xz)")]       # (elev, azim, title)


def grasp_segment(x, pad_geo):
    """The JAW GAP segment for a 7-DOF TCP grasp: the commanded width `w`, centred on the pad
    midplane, along the closing axis. Both come from the planner's own transform
    (`fg.tcp_to_local_grasp` with an identity object pose = world), NOT re-derived here — an
    earlier hand-rolled version drew the finger BODY origins instead of the pads and was 31 mm
    off in z and 35 mm too wide. Returns (end_a, end_b, tcp, approach_dir)."""
    x = np.asarray(x, float)
    tcp, w = x[:3], float(x[6])
    centre, axis, _u1, _u2, _wf = fg.tcp_to_local_grasp(x, obj_com=np.zeros(3),
                                                        obj_quat_wxyz=[1.0, 0, 0, 0], pad_geo=pad_geo)
    half = 0.5 * w * np.asarray(axis, float)
    return centre - half, centre + half, tcp, fg.approach_dir(x)


def draw_grasp(ax, x, pad_geo, color, *, lw=1.6, alpha=0.9, pads=True, approach=False,
               fingers=False, extent=False):
    """Draw one grasp: the jaw line at the pad CENTRE. `extent` adds each pad's real reach along
    the approach (+-half_u2 about the centre, 22.7 mm on this finger) so the picture shows where the
    face actually contacts — the centre alone reads ~23 mm higher than the fingertip. `fingers`
    scatters the full finger surface points (honest, but heavy)."""
    end_a, end_b, tcp, adir = grasp_segment(x, pad_geo)
    seg = np.array([end_a, end_b]) * 1000.0
    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], lw=lw, color=color, alpha=alpha, solid_capstyle="round")
    if extent:
        h = float(pad_geo["half_u2"]) * np.asarray(adir, float)
        for e in (end_a, end_b):
            p2 = np.array([e - h, e + h]) * 1000.0
            ax.plot(p2[:, 0], p2[:, 1], p2[:, 2], lw=lw * 0.8, color=color, alpha=alpha * 0.8)
    if pads:
        ax.scatter(seg[:, 0], seg[:, 1], seg[:, 2], s=12, color=color, alpha=alpha, depthshade=False)
    if fingers:
        Lw, Rw = fg.finger_world_pts(x, pad_geo)
        pts = np.concatenate([Lw, Rw]) * 1000.0
        ax.scatter(pts[::4, 0], pts[::4, 1], pts[::4, 2], s=1.0, color=color,
                   alpha=alpha * 0.22, depthshade=False)
    if approach:
        a = np.array([tcp - adir * 0.03, tcp]) * 1000.0
        ax.plot(a[:, 0], a[:, 1], a[:, 2], lw=lw * 0.6, color=color, alpha=alpha * 0.6, ls=":")


def draw_fork(ax, x, pad_geo, color, *, lw=1.4, alpha=0.9, shaft=0.02):
    """AnyGrasp-style 'tuning fork' gripper glyph for a 7-DOF TCP grasp: two finger bars at +-w/2
    along the closing axis, each spanning the pad's reach along the approach; a palm crossbar
    joining their back ends; a shaft behind the palm pointing away from the object. Same pose
    convention as `grasp_segment` (verified against the planner's transform)."""
    end_a, end_b, _tcp, adir = grasp_segment(x, pad_geo)      # jaw ends at the pad CENTRE
    a = np.asarray(adir, float)
    h = float(pad_geo["half_u2"]) * a                          # half the pad's reach along the approach
    tip_a, tip_b, back_a, back_b = end_a + h, end_b + h, end_a - h, end_b - h
    palm = 0.5 * (back_a + back_b)
    segs = [(back_a, tip_a), (back_b, tip_b), (back_a, back_b), (palm, palm - a * float(shaft))]
    for p0, p1 in segs:
        P = np.array([p0, p1]) * 1000.0
        ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=lw, color=color, alpha=alpha, solid_capstyle="round")


def draw_object(ax, V, tri, elev, azim, title):
    ax.plot_trisurf(V[:, 0] * 1000, V[:, 1] * 1000, V[:, 2] * 1000, triangles=tri,
                    color=(.75, .75, .8), alpha=.22, linewidth=0, shade=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x mm", fontsize=7); ax.set_ylabel("y mm", fontsize=7)
    ax.tick_params(labelsize=6)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("object", help="registry name, e.g. tofu / mushroom / cube3_soft")
    p.add_argument("--mu", type=float, default=0.7)
    p.add_argument("--table-z", type=float, default=0.0)
    p.add_argument("--obj-z", type=float, default=0.0298, help="object COM height (board = 0.0298)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--anim", action="store_true", help="also write a GIF of the search evolving")
    p.add_argument("--anim-stride", type=int, default=6, help="candidates per GIF frame")
    p.add_argument("--step", action="store_true",
                   help="INTERACTIVE: draw each synthesis stage in a window and wait for q (stages so far: seeds)")
    a = p.parse_args()

    od = OBJECT_MAP[a.object]; mat = od.material
    obj, pad_geo, meta = fg.build_grasp_fem(od.mesh_path, voxel_div=14, target_tets=1500,
                                            use_gpu=a.gpu)          # nu = fg.FEM_NU, same as the collector
    com = np.array([0.30, 0.0, float(a.obj_z)]); quat = np.array([1.0, 0, 0, 0])
    print(f"{a.object}: tets={meta['tets']} ndof={meta['ndof']} gpu={meta['gpu']}")

    viewer = None
    if a.step:
        from live_seed_viz import StageViewer
        viewer = StageViewer(obj, com, quat, pad_geo, label=a.object)
    t0 = time.perf_counter()
    out = fg.synthesize_grasp(obj, pad_geo, com, quat, E=mat.youngs_modulus, density=mat.density,
                              mu=a.mu, table_z=a.table_z, seed=a.seed, record_history=True,
                              yield_stress=float(mat.von_mises_yield_stress),
                              stage_cb=(viewer.on_stage if viewer else None))
    dt = time.perf_counter() - t0
    seeds, hist = out.get("seeds") or [], out.get("history") or []
    final_x = out.get("x")
    print(f"synthesis {dt:.1f}s  FEM calls {len(hist)}  final score "
          f"{out.get('score', float('nan')):.1f}")
    if final_x is None:
        print("  NO FEASIBLE GRASP — drawing seeds + trace only")
    else:
        print(f"  chosen x = {np.round(final_x, 4)}   width {1000*final_x[6]:.1f} mm")

    if seeds:
        kinds = {}
        for s in seeds:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
        print(f"seeds: {len(seeds)} ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})")

    # ── figure ────────────────────────────────────────────────────────────────
    tri, _ = boundary_faces(obj.tets)
    V = com + Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).apply(obj.verts)   # local -> world

    scores = np.array([h["score"] for h in hist], float)
    best_curve = np.maximum.accumulate(scores) if len(scores) else np.array([])
    improve_idx = [i for i in range(len(scores)) if i == 0 or scores[i] > best_curve[i - 1]]

    fig = plt.figure(figsize=(16.5, 9.2)); fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("viridis")

    # Row 1 — the SEEDS
    for k, (e, az, ttl) in enumerate(VIEWS):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        draw_object(ax, V, tri, e, az, f"SEEDS — {ttl}")
        for j, s in enumerate(seeds):
            c = cmap(j / max(len(seeds) - 1, 1))
            draw_grasp(ax, s["x"], pad_geo, c, lw=2.6, approach=True)

    # Row 2 — the POSE EVOLUTION (running best only: the search's actual progress)
    for k, (e, az, ttl) in enumerate(VIEWS):
        ax = fig.add_subplot(2, 3, k + 4, projection="3d")
        draw_object(ax, V, tri, e, az, f"EVOLUTION — {ttl}")
        for n, i in enumerate(improve_idx):
            frac = n / max(len(improve_idx) - 1, 1)
            draw_grasp(ax, hist[i]["x"], pad_geo, cmap(frac),
                       lw=1.0 + 1.6 * frac, alpha=0.25 + 0.55 * frac, pads=False)
        if final_x is not None:
            draw_grasp(ax, final_x, pad_geo, "tab:green", lw=3.0, approach=True, fingers=True)

    # Score trace. Penalty scores (~-1e8) sit orders below real grasps (~-1e4), so a single linear
    # axis hides the optimisation entirely: plot only FEASIBLE scores and rug the rest.
    ax = fig.add_subplot(2, 3, 3)
    if len(scores):
        feas = scores > -1e6
        xs = np.arange(len(scores))
        if feas.any():
            ax.plot(xs[feas], scores[feas], ".", ms=2.2, color="tab:blue", alpha=.55,
                    label=f"feasible ({100*feas.mean():.0f}% of calls)")
            bf = np.maximum.accumulate(np.where(feas, scores, -np.inf))
            ax.plot(xs, bf, lw=2.0, color="tab:red", label="best so far")
            lo, hi = np.percentile(scores[feas], 2), scores[feas].max()
            ax.set_ylim(lo - .05 * abs(lo), hi + .05 * abs(hi) + 1e-9)
        ax.plot(xs[~feas], np.full((~feas).sum(), ax.get_ylim()[0]), "|", ms=6,
                color="tab:grey", alpha=.35, label="infeasible (shaped penalty)")
        r2 = [i for i, h in enumerate(hist) if h.get("round", 1) == 2]
        if r2:
            ax.axvspan(min(r2), len(scores) - 1, color="tab:purple", alpha=.08, label="width refine")
        print(f"  feasible calls: {100*feas.mean():.0f}%   improvements: {len(improve_idx)}")
    ax.set_xlabel("FEM scorer call", fontsize=8)
    ax.set_ylabel("score (higher = gentler)", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower right"); ax.tick_params(labelsize=7)
    ax.set_title("CMA-ES trace", fontsize=9)

    # Summary panel
    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    rows = [f"object        {a.object}  ({meta['tets']} tets)",
            f"seeds         {fg.N_ANTIPODAL} antipodal + {fg.N_MEDIAL_AXIS} medial, top-{fg.TOP_K} -> CMA {fg.CMA_BUDGET_PER_SEED}/seed",
            f"FEM calls     {len(hist)}      wall {dt:.1f}s",
            f"improvements  {len(improve_idx)}"]
    if final_x is not None:
        rows += [f"final score   {out['score']:.1f}",
                 f"final width   {1000*final_x[6]:.1f} mm",
                 f"stress_top10  {out.get('stress_top10', float('nan')):.0f} Pa",
                 f"grip          {out.get('grip', float('nan')):.2f} N",
                 f"align         {out.get('align', float('nan')):.3f}"]
    else:
        rows += ["final         NO FEASIBLE GRASP"]
    ax.text(0.0, 0.98, "\n".join(rows), va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle(f"{a.object} — antipodal+medial seeds, top-{fg.TOP_K} CMA   "
                 f"(seeds & evolution: dark=early -> bright=late; GREEN = selected grasp)",
                 fontsize=11)
    out_p = Path(a.out or f".agent_tmp/seeds_{a.object}.png")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_p, dpi=110); plt.close(fig)
    print(f"wrote {out_p}")

    if a.anim and hist:
        _write_anim(out_p.with_suffix(".gif"), hist, final_x, V, tri, pad_geo, a.anim_stride)


def _write_anim(path, hist, final_x, V, tri, pad_geo, stride):
    """GIF of the search: each frame is the current candidate over the running-best trail."""
    import imageio.v2 as imageio
    cmap = plt.get_cmap("viridis")
    frames, best, trail = [], -np.inf, []
    for k, h in enumerate(hist):
        improved = h["score"] > best
        if improved:
            best = h["score"]; trail.append(h["x"])
        if not improved and (k % max(1, stride)):
            continue
        fig = plt.figure(figsize=(5.2, 4.6)); fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection="3d")
        draw_object(ax, V, tri, 24, -70, f"eval {h['eval']}   best {best:.0f}")
        for n, x in enumerate(trail):
            draw_grasp(ax, x, pad_geo, cmap(n / max(len(trail) - 1, 1)),
                       lw=1.2, alpha=.5, pads=False)
        draw_grasp(ax, h["x"], pad_geo,
                   "tab:blue" if h.get("holdable") else "tab:red", lw=2.0, pads=False)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    if final_x is not None and frames:
        frames += [frames[-1]] * 8                      # hold on the final pose
    imageio.mimsave(path, frames, duration=0.08, loop=0)
    print(f"wrote {path}  ({len(frames)} frames)")


if __name__ == "__main__":
    # Only force the headless backend when run as a SCRIPT. Setting it at import time would
    # clobber the interactive backend of anything importing the drawing helpers (live_seed_viz).
    # --step needs an interactive backend; everything else renders headless
    plt.switch_backend("TkAgg" if "--step" in sys.argv else "Agg")
    main()
