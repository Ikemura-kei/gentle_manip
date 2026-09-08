"""Stage 5: orchestrate mesh_prep -> fem_gate_check -> register_object -> pilot collection for a
batch of accepted candidates from candidates.csv, and append EXPANSION_LOG.csv (read by the web
atlas, tools/object_viewer/serve.py).

For each selected candidate:
  1. mesh_prep: Y-up->Z-up, repair, uniform rescale to the filter's suggested_scale, centre.
  2. fem_gate_check: direct-tet + extent-ratio + tet-count gate (adding_new_objects.md ss2).
     A gate FAIL is a hard reject (mesh needs manual fixing) -- object is NOT registered.
  3. register_object: registry entry + task/dr/experiment yaml trio.
  4. Pilot collection: collect_demos_synth_v4.py, n-episodes (default 5), n-envs 5, headless
     (MUJOCO_GL=egl), scene-dr-every 1, videos on -- this doubles as the smoke test (per
     docs/final/adding_new_objects.md ss5-6) AND the first few real demo trajectories (they land
     in the normal dataset/demos/<task>/ location, so if the object is accepted they simply ARE
     part of the collected set already).
  5. Parse stats.yaml; ACCEPT (keep registry+data) iff success_rate >= --min-success, else the
     object is left registered but flagged 'low_success' in EXPANSION_LOG for a manual look
     (per-object levers in adding_new_objects.md ss6: mesh thickness, material) -- data is not
     deleted automatically, only reported (deleting a run is a destructive action left to the user).

Usage:
    uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.batch_expand \
        --n-objects 20 --n-episodes 5 --n-envs 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATES_CSV = REPO_ROOT / "dataset" / "object_expansion" / "candidates.csv"
EXPANSION_LOG = Path(__file__).resolve().parent / "EXPANSION_LOG.csv"
ASSET_DIR = REPO_ROOT / "gentle_manip" / "assets" / "objects"

LOG_FIELDS = ["name", "category", "bucket", "source_uid", "geom_verdict", "suggested_scale",
              "prep_ok", "fem_gate_pass", "fem_tets", "collection_status", "success_rate",
              "ever_success_rate", "sub_yield_frac", "episodes_saved", "stage", "note", "timestamp"]


def load_candidates(path: Path = CANDIDATES_CSV) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def pick_batch(rows: list[dict], n: int, per_category: int = 1) -> list[dict]:
    """Prefer 'accept' over 'borderline'; spread across categories/buckets for diversity;
    within a category prefer the candidate closest to a 'nice' round min-width."""
    from collections import defaultdict
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["verdict"] in ("accept", "borderline") and r.get("download_ok") == "True":
            by_cat[r["category"]].append(r)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda r: (r["verdict"] != "accept", r["name"]))
    cats = sorted(by_cat.keys())
    out = []
    round_idx = 0
    while len(out) < n and any(by_cat.values()):
        progressed = False
        for cat in cats:
            if len(out) >= n:
                break
            if round_idx < min(per_category, len(by_cat[cat])):
                out.append(by_cat[cat][round_idx])
                progressed = True
        round_idx += 1
        if not progressed:
            break
    return out[:n]


def log_row(**kw) -> None:
    write_header = not EXPANSION_LOG.exists()
    with open(EXPANSION_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            w.writeheader()
        row = {k: kw.get(k, "") for k in LOG_FIELDS}
        row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        w.writerow(row)


_NAMES_CLAIMED_THIS_RUN: set[str] = set()


def next_free_name(base: str) -> str:
    """base if free, else base2, base3, ... (following the tomatoN / bs_cubeN convention).

    Reads names from registry.py's FILE TEXT (via register_object.already_registered), not the
    imported OBJECT_MAP dict -- a batch registering several objects in one process would otherwise
    see a STALE in-memory dict (Python caches the module on first import) and hand out the same
    name to two candidates of the same category, silently losing the second to the
    already-registered guard. `_NAMES_CLAIMED_THIS_RUN` additionally blocks names claimed earlier
    in this same run before their registry.py write, same staleness concern belt-and-suspenders.
    """
    from gentle_manip.scripts.object_expansion.register_object import already_registered
    cand = base
    i = 2
    while already_registered(cand) or cand in _NAMES_CLAIMED_THIS_RUN:
        cand = f"{base}{i}"
        i += 1
    _NAMES_CLAIMED_THIS_RUN.add(cand)
    return cand


def slugify(category: str) -> str:
    import re
    s = re.sub(r"\(.*?\)", "", category)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "obj"


def process_one(cand: dict, *, n_episodes: int, n_envs: int, min_success: float,
                material: str, skip_collect: bool) -> None:
    from gentle_manip.scripts.object_expansion.mesh_prep import prep_mesh
    from gentle_manip.scripts.object_expansion.fem_gate_check import check as fem_check
    from gentle_manip.scripts.object_expansion.register_object import register, already_registered

    category, bucket, uid = cand["category"], cand["bucket"], cand["uid"]
    name = next_free_name(slugify(category))
    scale = float(cand["suggested_scale"])
    src = cand["local_path"]
    dst = ASSET_DIR / f"{name}.obj"

    print(f"\n=== {name}  (category={category} bucket={bucket} uid={uid[:8]} scale={scale:.4f}) ===")
    log_kw = dict(name=name, category=category, bucket=bucket, source_uid=uid,
                  geom_verdict=cand["verdict"], suggested_scale=scale)

    try:
        prep = prep_mesh(src, str(dst), scale)
    except Exception as e:  # noqa: BLE001
        print(f"  [prep] FAILED: {e}")
        log_row(**log_kw, prep_ok=False, stage="mesh_prep", note=str(e)[:200])
        return
    print(f"  [prep] ok: extents {prep['final_extents_mm']} mm, watertight={prep['watertight']}, "
          f"verts={prep['n_verts']} faces={prep['n_faces']}")
    if not prep["watertight"]:
        log_row(**log_kw, prep_ok=False, stage="mesh_prep", note="not watertight after repair")
        return

    try:
        gate = fem_check(str(dst))
    except Exception as e:  # noqa: BLE001
        print(f"  [fem_gate] FAILED (exception): {e}")
        log_row(**log_kw, prep_ok=True, fem_gate_pass=False, stage="fem_gate", note=str(e)[:200])
        return
    print(f"  [fem_gate] direct_tet={gate['direct_tet']} tets={gate['tets']} "
          f"ratio_dev={gate['extent_ratio_max_dev']} -> {'PASS' if gate['passed'] else 'FAIL'}")
    if not gate["passed"]:
        log_row(**log_kw, prep_ok=True, fem_gate_pass=False, fem_tets=gate["tets"],
                stage="fem_gate", note="direct_tet/ratio/tetcount gate failed")
        return

    extents_m = tuple(x / 1000 for x in prep["final_extents_mm"])
    default_pos_z = prep["suggested_default_pos_z"]
    try:
        cfg = register(name, category, material, extents_m, default_pos_z,
                       source_note=f"# {name}: Objaverse-LVIS {category} ({bucket} bucket, uid {uid}), "
                                   f"object-expansion batch")
    except Exception as e:  # noqa: BLE001
        print(f"  [register] FAILED: {e}")
        log_row(**log_kw, prep_ok=True, fem_gate_pass=True, fem_tets=gate["tets"],
                stage="register", note=str(e)[:200])
        return
    print(f"  [register] ok: spawn_z={cfg['spawn_z']:.4f} grasp_gate={cfg['grasp_gate_dist']:.3f} "
          f"substeps={cfg['substeps']} grid={cfg['grid_density']}")

    if skip_collect:
        log_row(**log_kw, prep_ok=True, fem_gate_pass=True, fem_tets=gate["tets"],
                collection_status="skipped", stage="registered_no_collect")
        return

    exp_name = f"single_lift_{name}_soft_abs_action_armfocus_7d_realws"
    task_name = f"single_lift_{name}_soft"
    env = dict(os.environ)
    env["MUJOCO_GL"] = "egl"
    env["OMP_NUM_THREADS"] = "8"
    cmd = [sys.executable, "grasp_synthesis/collect_demos_synth_v4.py",
           "--experiment", exp_name, "--task-name", task_name,
           "--table-z", "0.0138", "--n-episodes", str(n_episodes), "--n-envs", str(n_envs),
           "--seed", "0", "--scene-dr-every", "1", "--record-video", "100000"]
    print(f"  [collect] launching pilot: {n_episodes} episodes x {n_envs} envs ...")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                          timeout=1800)
    dt = time.time() - t0
    log_path = REPO_ROOT / "dataset" / "object_expansion" / "logs" / f"{name}_pilot_collect.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + "\n\n=== STDERR ===\n" + proc.stderr)
    if proc.returncode != 0:
        print(f"  [collect] FAILED rc={proc.returncode} ({dt:.0f}s) -- see {log_path}")
        log_row(**log_kw, prep_ok=True, fem_gate_pass=True, fem_tets=gate["tets"],
                collection_status="crashed", stage="collect", note=f"rc={proc.returncode}, see {log_path.name}")
        return

    import glob
    import yaml
    demo_base = REPO_ROOT / "dataset" / "demos" / task_name
    run_dirs = sorted(demo_base.glob("*")) if demo_base.exists() else []
    stats = None
    if run_dirs:
        stats_path = run_dirs[-1] / "stats.yaml"
        if stats_path.exists():
            stats = yaml.safe_load(stats_path.read_text())
    if stats is None:
        print(f"  [collect] finished ({dt:.0f}s) but no stats.yaml found -- see {log_path}")
        log_row(**log_kw, prep_ok=True, fem_gate_pass=True, fem_tets=gate["tets"],
                collection_status="no_stats", stage="collect", note=f"see {log_path.name}")
        return

    sr = float(stats.get("success_rate", 0.0))
    status = "accepted" if sr >= min_success else "low_success"
    print(f"  [collect] done ({dt:.0f}s): success={sr*100:.0f}% ever_lifted="
          f"{100*float(stats.get('ever_success_rate', 0)):.0f}% saved={stats.get('episodes_saved')} "
          f"-> {status.upper()}")
    log_row(**log_kw, prep_ok=True, fem_gate_pass=True, fem_tets=gate["tets"],
            collection_status=status, success_rate=sr,
            ever_success_rate=stats.get("ever_success_rate"), sub_yield_frac=stats.get("sub_yield_frac"),
            episodes_saved=stats.get("episodes_saved"), stage="done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-objects", type=int, default=20)
    ap.add_argument("--per-category", type=int, default=1)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--min-success", type=float, default=0.5)
    ap.add_argument("--material", default="soft_shape")
    ap.add_argument("--skip-collect", action="store_true", help="register only, no pilot run (fast pass)")
    ap.add_argument("--only-categories", nargs="*", default=None)
    ap.add_argument("--candidates-csv", type=Path, default=CANDIDATES_CSV,
                    help="override the candidates.csv location -- e.g. sourcing ran in a "
                         "different checkout/filesystem than this batch is executing from "
                         "(this pipeline currently spans a login-node checkout for CPU-only "
                         "sourcing and a separate aarch64 cluster checkout for GPU stages)")
    args = ap.parse_args()

    rows = load_candidates(args.candidates_csv)
    if args.only_categories:
        rows = [r for r in rows if r["category"] in args.only_categories]
    batch = pick_batch(rows, args.n_objects, per_category=args.per_category)
    print(f"selected {len(batch)} candidates from {len(rows)} candidate rows")
    for cand in batch:
        process_one(cand, n_episodes=args.n_episodes, n_envs=args.n_envs,
                   min_success=args.min_success, material=args.material,
                   skip_collect=args.skip_collect)
    print("\nBatch done.")


if __name__ == "__main__":
    main()
