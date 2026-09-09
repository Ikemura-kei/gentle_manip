# Session handoff — object-set expansion (2026-09-09)

This document is the single entry point for resuming this work **locally**, after the
NAISS Arrhenius cluster's GPU queue became too congested to make further progress today
(see "Why we moved to local" below). Read this first; it links to the deeper docs.

## 1. Goal

Expand the gentle_manip soft-body object set from ~30 hand-built objects to **200 object
categories, each with 20 collected grasp-and-lift demonstration trajectories**, sourced
from Objaverse-LVIS (and briefly, MetaFood3D — parked, see DEVLOG), filtered for
graspability under the xArm7 gripper's true max opening width (**79 mm**, not the commonly
quoted 88.9 mm raw travel — see `grasp_synthesis/smgrasp/finger_grasp_final.py:515`).
Objects should be shape-diverse (rope-like, thin, tall, concave, "crazy" shapes), not
necessarily food, as long as they can be convexified/simulated correctly in Genesis MPM.

## 2. Where things stand right now

- **43 categories / 60 individual objects** have a **complete 20-episode collection**
  (`collection_status` = `accepted` or `low_success` in `EXPANSION_LOG.csv`), up from 6
  categories / ~18 objects at the start of today's session. **21.5% of the 200-category
  target.**
- **425 total mesh assets are registered** (valid mesh + FEM-gate-passed + task/DR/
  experiment configs written) in `gentle_manip/assets/registry.py`, but most have NOT yet
  been through the 20-episode MPM collection step — they are the immediate queue for
  local collection to pick up.
- All of this (registry, mesh assets, configs, the full pipeline audit log) is now
  **committed to git** on branch `expand-categories-30` (commit `3b15d6c` and earlier).
  `git pull` (or `git clone`) on your local machine gets everything except the actual
  collected trajectory data (`dataset/demos/`, which is `.gitignore`d — see Β§5).

## 3. The pipeline (how an object goes from "idea" to "20 demos")

```
source_and_filter.py  →  geom_filter.py  →  mesh_prep.py  →  fem_gate_check.py  →  register_object.py  →  collect_demos_synth_v4.py
  (download from            (graspability     (repair to        (FEM tet-mesh        (registry entry +      (the actual 20-episode
   Objaverse-LVIS,            filter: true      watertight,       gate: direct_tet,    task/dr/experiment     MPM collection run,
   category by category)      min-width via     Y-up→Z-up,        extent_ratio,        YAML trio)             scripted CMA-ES grasp
                               convex-hull       rescale,          tet_count ≤ 4500)                           synthesis)
                               direction scan)   recenter)
```

All of this is orchestrated end-to-end for a batch of candidates by
**`gentle_manip/scripts/object_expansion/batch_expand.py`** — this is the ONE script you
need to run repeatedly to keep making progress locally. See Β§6.

Full pipeline design: `docs/final/adding_new_objects.md`, `docs/final/grasp_synthesis_final.md`.
Everything that happened, in order, with root causes and fixes: `docs/final/DEVLOG.md`
(search for `2026-09-09` — several critical bugs were found and fixed today, see Β§4).

## 4. Bugs found and fixed today (read DEVLOG for full diagnosis on each)

These compound — each one unblocked the next, so if you're debugging low throughput
locally, check these are all still in place (they're in the commits pushed today):

1. **Mesh repair failing 92.7% of candidates** (`mesh_prep.py`) — heavy quadric decimation
   after a coarse voxel-remesh reliably broke watertightness. Fixed: search progressively
   coarser voxel resolutions instead, so the mesh is watertight by construction and rarely
   needs decimation at all.
2. **Face budget mismatched to the FEM gate** — `max_faces` lowered 6000→2000 (empirically
   tuned against the tet-count cap).
3. **Blocky voxel-remesh geometry crashing MPM** — added Taubin smoothing to the
   voxel-remeshed output (10/10 objects were crashing at collection before this fix).
4. **A collection subprocess timeout could kill an entire batch job** — one slow object
   under cluster contention took down every other queued candidate in the same job.
   Fixed: caught and logged as `collection_status=timeout`, loop continues.
5. **Category-exclusion bug** in the round-to-round candidate selection (compared a
   regex-stripped object *name* to the real category *string* — never matched). Fixed to
   use exact category strings from `EXPANSION_LOG.csv`.
6. Two Artifact/webpage bugs (three.js CDN loading failing silently, twice) — see Β§7.

## 5. Key paths

| What | Path | In git? | Size |
|---|---|---|---|
| Collected demo trajectories (the actual data) | `dataset/demos/single_lift_<name>_soft/<run>/data.pkl` (+ `videos/`, `stats.yaml`, `config.yaml`) | **No** (gitignored) | ~49 GB total (34 GB pkl, 3.7 GB video) |
| Registered mesh CAD assets | `gentle_manip/assets/objects/*.obj` (+ `.mtl`, textures) | **Yes** (committed today) | ~65 MB |
| Object registry (name → mesh/material/size) | `gentle_manip/assets/registry.py` | **Yes** | small |
| Per-object task/DR/experiment configs | `gentle_manip/configs/{tasks,dr,experiments}/single_lift_<name>_soft*.yaml` | **Yes** | small |
| Full pipeline audit log (every candidate, every outcome) | `gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv` | **Yes** | small |
| Sourced-candidate metadata (pre-registration) | `dataset/object_expansion/candidates*.csv` | **Yes** | ~40 MB |
| Raw downloaded Objaverse meshes (candidates, pre-repair) | real location: `/nobackup/proj/disk/softenable-codesign26/personal/yifeid/object_expansion_sources/objaverse_cache/` (reached via a symlink at `dataset/object_expansion/objaverse_cache` in the home checkout — NOT present as a real dir under the nobackup repo checkout, which is why an early version of this doc wrongly called it "not needed") | **No** | 9.3 GB, 1,566 files |
| Orchestrator script | `gentle_manip/scripts/object_expansion/batch_expand.py` | Yes | — |
| Web pages (see Β§7) | n/a — hosted, not local files | — | — |

**Two things need manual copying, not one:**
1. `dataset/demos/` — only if you want to inspect/reuse already-collected episodes; not
   read by registration, only written by collection.
2. **`objaverse_cache/` (9.3 GB) — REQUIRED to register any NEW (not-yet-registered)
   category from the existing `candidates_all.csv` pool.** `batch_expand.py` reads each
   candidate's source mesh from the `local_path` column in that CSV and does **not**
   auto-download — the file must already be on disk, or `mesh_prep` fails outright. Skip
   this only if you're going to re-run `source_and_filter.py` to re-download from
   Objaverse fresh (slower, and re-download of everything already vetted is wasted work).

```bash
# from your local machine, after cloning the repo:
rsync -avz --progress \
  yifeid@arrhenius1.hpc.arrhenius.naiss.se:/nobackup/proj/disk/softenable-codesign26/personal/yifeid/object_expansion_sources/objaverse_cache/ \
  dataset/object_expansion/objaverse_cache/
```

## 6. How to resume locally

### 6.1 Get the code
```bash
git clone git@github.com:Ikemura-kei/gentle_manip.git   # or: cd <existing checkout> && git pull
cd gentle_manip
git checkout expand-categories-30
git submodule update --init --recursive
```

### 6.2 Set up the sim environment
Use **`envs/sim`** (NOT `envs/sim_arrhenius` — that one is pinned to the cluster's aarch64
GH200 + cu126 Hopper build; `envs/sim` is the standard x86_64 environment already set up
for a local lab box). **You need an NVIDIA GPU with CUDA** — Genesis's MPM solver runs on
GPU; this will not run (or will be extremely slow) on CPU-only or non-NVIDIA hardware.
```bash
uv sync --project envs/sim
# torch is installed separately (see root CLAUDE.md's "torch is installed manually" note) —
# check envs/sim/pyproject.toml's header comment for the exact cu-version triple, and swap
# it for whatever matches your local GPU/driver if different from what's pinned.
uv pip install --python envs/sim/.venv/bin/python "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
```
Verify: `uv run --project envs/sim python -m pytest gentle_manip/tests/ -q`

### 6.3 Copy data from the cluster (two separate things)
Run these **from your local machine** (needs your cluster SSH access configured), from
your repo root:

**(a) Already-collected demo trajectories** — only needed if you want to inspect/reuse old
episodes; `batch_expand.py` never reads this at registration time, only writes fresh run
dirs there at collection time:
```bash
mkdir -p dataset/demos
rsync -avz --progress \
  yifeid@arrhenius1.hpc.arrhenius.naiss.se:/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/dataset/demos/ \
  dataset/demos/
```
~49 GB — expect it to take a while.

**(b) The sourced-candidate mesh cache — REQUIRED to register any new category** from the
existing `candidates_all.csv` pool (not optional like (a); `mesh_prep` fails outright
without the source file on disk, `batch_expand.py` has no auto-download fallback):
```bash
mkdir -p dataset/object_expansion/objaverse_cache
rsync -avz --progress \
  yifeid@arrhenius1.hpc.arrhenius.naiss.se:/nobackup/proj/disk/softenable-codesign26/personal/yifeid/object_expansion_sources/objaverse_cache/ \
  dataset/object_expansion/objaverse_cache/
```
~9.3 GB, 1,566 files — much faster than (a), do this one first.

### 6.4 Run the pipeline
This is the direct local equivalent of what `gentle_manip/scripts/arrhenius/
object_expansion_pilot.sbatch` did on the cluster (skip the SLURM wrapper, submodule-sync
step, and pymeshlab GH200 lib-swap — none of that applies locally):
```bash
cd gentle_manip   # repo root
uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.batch_expand \
  --n-objects 30 --per-category 1 --n-episodes 20 --n-envs 5 --min-success 0.5 \
  --candidates-csv dataset/object_expansion/candidates_all.csv \
  --only-categories "bowl coffeepot flute_glass sponge ..."   # or omit for the full pool
```
Useful flags (all in `batch_expand.py --help`):
- `--only-categories "<space-separated list>"` — restrict to specific categories (use this
  to pick up where the cluster left off; see Β§8 for the exact list of not-yet-collected
  categories).
- `--skip-collect` — register only (mesh_prep + fem_gate + registry write), no MPM run —
  useful for a fast first pass to see what geometry passes before spending GPU time.
- `--min-success 0.5` — success-rate threshold for `accepted` vs `low_success`.

Since there's no SLURM queue locally, just run this in a loop / re-invoke it with a fresh
`--only-categories` slice each time — there's no batching/priority concern, only your own
GPU's throughput.

### 6.5 Regenerate the web pages after local runs (optional)
Both artifact pages pull their data from `EXPANSION_LOG.csv` + `dataset/demos/*/stats.yaml`
+ the registry. To refresh them after a local session:
```bash
uv run --project envs/sim python gentle_manip/scripts/object_expansion/export_artifact_data.py \
  --out /tmp/artifact_data.json --max-tris 400
uv run --project envs/sim python gentle_manip/scripts/object_expansion/export_category_summary.py \
  --out /tmp/category_summary.json
uv run --project envs/sim python gentle_manip/scripts/object_expansion/export_video_gallery.py \
  --out /tmp/video_gallery.json --max-kb 260
```
Then splice each into its template (`tools/object_viewer/artifact_template.html`,
`tools/object_viewer/video_gallery_template.html` — replace the `__DATA_JSON__` /
`__CATEGORY_JSON__` markers) and publish via the Artifact tool (only Claude can publish —
ask in a Claude Code session with these files).

## 7. The web pages

Two published Claude Artifacts, both showing live data as of the last refresh this session:

- **Grasp Specimen Catalog** — <https://claude.ai/code/artifact/59ea2dff-f704-4334-99da-ab7f9f39235b>
  Per-object 3D viewer (vanilla WebGL, no external library — see DEVLOG for why), material/
  size stats, grasp-success gauge, and the full sourced-candidates table.
- **Grasp Reel** — <https://claude.ai/code/artifact/84076dfb-7344-4de9-80c3-283058894c7a>
  One representative video clip per *category* (deduped — `pea_(food)`'s 12 instances show
  once, not 12 times), a progress card (categories collected / 200 target / recent-rate /
  ETA), and a full per-category table including the original hand-built dev/food objects
  (which correctly show 0 episodes — they were never collected under this protocol).

Both are private artifacts owned by this account; re-publish from a Claude Code session
with `url` set to the links above to update them (see Β§6.5).

## 8. Categories NOT yet collected (the local queue)

The candidate pool (`dataset/object_expansion/candidates_all.csv`) has 420 accepted
categories; 43 have a complete collection. To see the exact remaining list at any time:
```bash
python3 -c "
import csv
already = set()
with open('gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv') as f:
    for r in csv.DictReader(f):
        if r.get('collection_status') in ('accepted','low_success'):
            already.add(r['category'])
with open('dataset/object_expansion/candidates_all.csv') as f:
    rows = list(csv.DictReader(f))
cats = sorted(set(r['category'] for r in rows if r['verdict']=='accept'))
fresh = [c for c in cats if c not in already]
print(len(fresh), 'categories remaining')
print(' '.join(fresh))
"
```
For shape diversity (recommended — the last cluster rounds were prioritizing this),
sort remaining categories by `bucket` in `candidates_all.csv`, preferring buckets least
represented among already-collected categories. See the round-19/20 chunk-building logic
in this session's DEVLOG entry (2026-09-09, "Category-level view: exclusion bug audit +
shape-diversity priority") for the exact algorithm if you want to replicate it.

## 9. Why we moved to local

The cluster's shared GPU partition (382 nodes × 4 GH200 GPUs = 1528 GPUs) became severely
congested late in the session — login-node load spiked to 50-120 (vs. a normal ~16), and
our submitted jobs sat pending for hours with SLURM's own scheduler estimating ~10-16 hour
wait times, ranking ~400-500th among pending jobs on the partition by fairshare priority.
This is an external, cluster-wide condition (900+ other users' jobs running/queued), not a
bug in our pipeline. Local execution avoids this entirely at the cost of using your own
GPU's throughput instead of the cluster's (potentially many GH200s in parallel).
