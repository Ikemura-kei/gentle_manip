"""Sweep MPM grid_density x substeps for the mushroom soft body — speed vs accuracy.

Extends mushroom_soft_dev.py to a SWEEP: each (grid_density, substeps) combo runs in its OWN
subprocess (Genesis can't re-init in one process), settles the mushroom, and reports:
  SPEED    : MPM particles, physics steps/s
  STABILITY: did it blow up? (center drift; NaN / out-of-bounds particles)
  SHAPE    : settled particle bounding box (W x D x H) + center z (penetration if it sinks)
  ACCURACY : deviation of shape/center vs the FINEST config in the sweep (treated as reference)

    MUJOCO_GL=egl uv run --project envs/sim python examples/mushroom_grid_substep_sweep.py \
        --grids 120,150,185,250 --substeps 40,80,150,210 [--pairs] [--video] [--steps 200]

--pairs zips grids with substeps (one config each) instead of the full cartesian product.
--video writes examples/mushroom_sweep/gd<g>_ss<s>.mp4 per config (visual accuracy check).
"""
import argparse
import csv
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

import numpy as np

_OBJ = Path(__file__).resolve().parents[1] / "gentle_manip" / "assets" / "objects" / "mushroom.obj"
_OUT = Path(__file__).resolve().parent / "mushroom_sweep"


def _run_one(grid, substeps, steps, E, nu, rho, yld, pos, scale, n_envs, sampler, video_path, q):
    """Worker (fresh process): build the mushroom at (grid, substeps), settle, return metrics.
    Batched (n_envs): steps/s is scene.step()/s at n_envs (training-relevant throughput);
    shape/stability metrics use env 0, and the NaN blow-up check spans all envs."""
    try:
        import time
        os.environ.setdefault("MUJOCO_GL", "egl")
        import genesis as gs
        gs.init(backend=gs.gpu)
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1.0 / 30.0, substeps=substeps),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(0.35, -0.13, -0.012), upper_bound=(0.63, 0.13, 0.23),
                grid_density=grid),
            rigid_options=gs.options.RigidOptions(gravity=(0.0, 0.0, -9.81)),
            vis_options=gs.options.VisOptions(visualize_mpm_boundary=True),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        mush = scene.add_entity(
            morph=gs.morphs.Mesh(file=str(_OBJ), pos=tuple(pos), scale=scale),
            material=gs.materials.MPM.ElastoPlastic(E=E, nu=nu, von_mises_yield_stress=yld, rho=rho,
                                                    sampler=sampler),
            surface=gs.surfaces.Default(vis_mode="particle"))
        cam = None
        if video_path is not None:
            cam = scene.add_camera(res=(640, 480), pos=(pos[0] + 0.16, pos[1] - 0.16, 0.14),
                                   lookat=(pos[0], pos[1], 0.02), fov=30, GUI=False)
        scene.build(n_envs=n_envs)

        def _parts():
            p = mush.get_state().pos                                 # torch CUDA tensor
            if hasattr(p, "detach"):
                p = p.detach().cpu().numpy()
            p = np.asarray(p)
            return p[None] if p.ndim == 2 else p                     # -> (n_envs, n_p, 3)

        def _stress():
            v = mush.get_state().von_mises                           # per-particle von Mises (Pa)
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            v = np.asarray(v)
            return v[None] if v.ndim == 1 else v                     # -> (n_envs, n_p)

        P0 = _parts()
        n_part = int(P0.shape[1])                                    # particles PER env
        c0 = P0[0].mean(0)                                           # env-0 center (representative)
        frames, phys_t, drift_max, vm_samples = [], 0.0, 0.0, []
        for t in range(steps):
            t0 = time.perf_counter(); scene.step(); phys_t += time.perf_counter() - t0
            if cam is not None and t % 3 == 0:
                fr = np.asarray(cam.render(rgb=True)[0], np.uint8)
                frames.append(fr[0] if fr.ndim == 4 else fr)         # env 0 if batched render
            if t % 20 == 0:                                          # sampled OUTSIDE the timed step
                drift_max = max(drift_max, float(np.linalg.norm(_parts()[0].mean(0) - c0)))
                vm_samples.append(float(_stress()[0].mean()))        # env-0 mean von Mises
        mean_vm = float(np.mean(vm_samples)) if vm_samples else 0.0  # time-averaged mean stress
        Pf = _parts()
        nan = bool(~np.isfinite(Pf).all())                          # ANY env blew up -> NaN
        cf = Pf[0].mean(0)
        lo, hi = Pf[0].min(0), Pf[0].max(0)
        bbox = (hi - lo)
        # blow-up detector: NaN (any env), env-0 center outside the MPM box, or a degenerate/
        # exploded env-0 shape (intact mushroom ~3 cm; collapsed -> ~0, exploded -> large).
        in_box = bool((cf > [0.35, -0.13, -0.012]).all() and (cf < [0.63, 0.13, 0.23]).all())
        shape_ok = bool(np.all(bbox > 0.015) and np.all(bbox < 0.08))
        stable = (not nan) and in_box and shape_ok
        if cam is not None and frames:
            import imageio.v2 as imageio
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=15, macro_block_size=1)
        q.put(dict(grid=grid, substeps=substeps, n_part=n_part, steps_per_s=steps / phys_t if phys_t else 0,
                   center=cf.tolist(), bbox=bbox.tolist(), drift_mm=drift_max * 1000,
                   mean_vonmises=mean_vm, stable=stable, err=None))
    except Exception as e:
        q.put(dict(grid=grid, substeps=substeps, n_part=0, steps_per_s=0, center=[0, 0, 0],
                   bbox=[0, 0, 0], drift_mm=0, mean_vonmises=0, stable=False, err=f"{type(e).__name__}: {e}"))


def main():
    from gentle_manip.assets.materials import MATERIALS
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="105")
    ap.add_argument("--substeps", default="95,100,105")
    ap.add_argument("--pairs", action="store_true", help="zip grids+substeps instead of cartesian product")
    ap.add_argument("--material", default="mushroom", choices=sorted(MATERIALS))
    ap.add_argument("--n-envs", type=int, default=100,
                    help="parallel MPM envs (batched) — steps/s then reflects training throughput")
    ap.add_argument("--sampler", default="regular", choices=("regular", "pbs", "random"),
                    help="MPM particle sampler for the soft body (regular = deterministic grid)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--pos", type=float, nargs=3, default=[0.5, 0.0, 0.055],
                    help="spawn (m); raised to clear the coarse-grid MPM boundary padding for all "
                         "grids in the sweep (a resting z fails to BUILD on coarse grids). The small "
                         "settle drop is uniform across configs, so still apples-to-apple.")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    m = MATERIALS[args.material]
    E, nu, rho, yld = m.youngs_modulus, m.poisson_ratio, m.density, m.von_mises_yield_stress
    grids = [int(x) for x in args.grids.split(",")]
    subs = [int(x) for x in args.substeps.split(",")]
    combos = list(zip(grids, subs)) if args.pairs else [(g, s) for g in grids for s in subs]
    print(f"material={args.material} E={E:.0f} nu={nu} rho={rho} yield={yld:.0f} sampler={args.sampler} | "
          f"{len(combos)} configs, n_envs={args.n_envs}, {args.steps} steps each "
          f"(steps/s = scene.step()/s at {args.n_envs} envs)\n")

    # Per-run artifact folder (videos + CSV) under mushroom_sweep/<datetime>/.
    run_dir = _OUT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"artifacts -> {run_dir}\n")

    ctx = mp.get_context("spawn")
    rows = []
    for g, s in combos:
        vp = str(run_dir / f"gd{g}_ss{s}.mp4") if args.video else None
        q = ctx.Queue()
        p = ctx.Process(target=_run_one, args=(g, s, args.steps, E, nu, rho, yld, args.pos, args.scale, args.n_envs, args.sampler, vp, q))
        p.start(); r = q.get(); p.join()
        rows.append(r)
        tag = "ERR " + r["err"] if r["err"] else ("OK  " if r["stable"] else "UNSTABLE")
        print(f"  gd={g:>4} ss={s:>4} | {r['n_part']:>5}p | {r['steps_per_s']:>6.1f} st/s | "
              f"stress={r['mean_vonmises']:>7.0f} | drift={r['drift_mm']:>5.1f}mm | "
              f"bbox=({r['bbox'][0]*100:.1f},{r['bbox'][1]*100:.1f},{r['bbox'][2]*100:.1f})cm | {tag}", flush=True)

    # CSV table of the full sweep -> <run_dir>/results.csv
    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grid_density", "substeps", "n_envs", "sampler", "particles_per_env",
                    "steps_per_s", "mean_vonmises_pa", "stable", "drift_mm", "center_z_mm",
                    "bbox_x_cm", "bbox_y_cm", "bbox_z_cm", "error"])
        for r in rows:
            w.writerow([r["grid"], r["substeps"], args.n_envs, args.sampler, r["n_part"],
                        f"{r['steps_per_s']:.2f}", f"{r['mean_vonmises']:.1f}", int(r["stable"]),
                        f"{r['drift_mm']:.1f}", f"{r['center'][2]*1000:.1f}",
                        f"{r['bbox'][0]*100:.2f}", f"{r['bbox'][1]*100:.2f}", f"{r['bbox'][2]*100:.2f}",
                        r["err"] or ""])
    print(f"\nCSV -> {csv_path}")

    # ── SUMMARY: fps + stability for every combination ──
    def _status(r):
        return "ERR" if r["err"] else ("stable" if r["stable"] else "UNSTABLE")
    print("\n=== SUMMARY: fps + stability + stress per combination ===")
    print(f"{'grid':>5} {'substeps':>9} {'particles':>10} {'steps/s':>9} {'vonMises':>9} {'status':>9}")
    for r in rows:
        print(f"{r['grid']:>5} {r['substeps']:>9} {r['n_part']:>10} {r['steps_per_s']:>9.1f} "
              f"{r['mean_vonmises']:>9.0f} {_status(r):>9}")
    if not args.pairs:                       # matrix view for a product sweep
        gset = sorted({g for g, _ in combos}); sset = sorted({s for _, s in combos})
        bk = {(r["grid"], r["substeps"]): r for r in rows}
        print("\nsteps/s matrix (rows=grid_density, cols=substeps; x=unstable, !=err):")
        print("  gd\\ss " + "".join(f"{s:>8}" for s in sset))
        for g in gset:
            cells = [f"{'-':>8}" if (r := bk.get((g, s))) is None else
                     f"{'!':>8}" if r["err"] else
                     f"{'x':>8}" if not r["stable"] else f"{r['steps_per_s']:>8.1f}" for s in sset]
            print(f"  {g:>4}  " + "".join(cells))
    _fast = [r for r in rows if r["stable"]]
    if _fast:
        b = max(_fast, key=lambda r: r["steps_per_s"])
        print(f"\nfastest STABLE: gd={b['grid']} ss={b['substeps']} -> {b['steps_per_s']:.1f} steps/s "
              f"({b['n_part']} particles/env)")

    # accuracy vs the finest STABLE config (max grid*substeps) as reference
    stable = [r for r in rows if r["stable"]]
    if stable:
        ref = max(stable, key=lambda r: r["grid"] * r["substeps"])
        rc, rb = np.array(ref["center"]), np.array(ref["bbox"])
        print(f"\nReference (finest stable): gd={ref['grid']} ss={ref['substeps']}  "
              f"center_z={rc[2]*1000:.1f}mm bbox=({rb[0]*100:.1f},{rb[1]*100:.1f},{rb[2]*100:.1f})cm")
        print(f"{'config':>16} | {'steps/s':>8} | {'Δcenter':>8} | {'Δbbox':>8}  (deviation from reference)")
        for r in stable:
            dc = np.linalg.norm(np.array(r["center"]) - rc) * 1000
            db = np.linalg.norm(np.array(r["bbox"]) - rb) * 1000
            print(f"  gd={r['grid']:>4} ss={r['substeps']:>4} | {r['steps_per_s']:>8.1f} | {dc:>6.1f}mm | {db:>6.1f}mm")
    if args.video:
        print(f"videos -> {run_dir}/gd<g>_ss<s>.mp4")


if __name__ == "__main__":
    main()
