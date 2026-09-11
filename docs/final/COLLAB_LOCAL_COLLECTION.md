# Collaborator guide — local grasp-demo collection

For a second person to run `batch_expand.py` on their own GPU box, in parallel with the
cluster, without duplicating work. Branch: **`expand-categories-30`**. Read
`docs/final/SESSION_HANDOFF_2026-09-09.md` first for the full pipeline background — this doc
is the delta for a *collaborator* and assumes that one is read.

## 1. Goal & current state

- **Target: 200 object categories, each with a complete 20-episode grasp-and-lift
  demo collection.** Not "exactly 200 to run" — collect until 200 categories have a
  completed run, or the candidate pool is exhausted (whichever first).
- **Candidate pool:** 420 categories that passed the geometry filter
  (`dataset/object_expansion/candidates_all.csv`, `verdict == accept`).
- **Collected so far (as of this doc, cluster only): 78 categories / ~105 object
  instances.** Full list in Β§7 and in
  `gentle_manip/scripts/object_expansion/partner_chunks/collected_so_far.txt`.
- **Remaining: 342 candidate categories.** Split 50/50 for parallel work (Β§4).
- Expect a **~15% collection success rate** among candidates that reach the MPM step (the
  rest crash / fail the FEM gate / time out). So 342 remaining × ~15% first-attempt ≈ ~50
  more, plus retries on categories with multiple candidate mesh UIDs. Reaching 200 total
  is **marginal** with the current pool — if we plateau, the next step is re-sourcing more
  Objaverse categories with `source_and_filter.py`.

## 2. What you need to get onto your box

### (a) The code + all registered assets — from git
```bash
git clone git@github.com:Ikemura-kei/gentle_manip.git   # or: git pull in an existing checkout
cd gentle_manip
git checkout expand-categories-30
git submodule update --init --recursive
```
This gets you: all ~700 registered object meshes (`gentle_manip/assets/objects/*.obj` —
**committed, not gitignored**), the registry (`gentle_manip/assets/registry.py`), every
per-object config trio (`gentle_manip/configs/{tasks,dr,experiments}/single_lift_<name>_soft*.yaml`),
the pipeline audit log (`gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv`), and your
assigned category chunks (`gentle_manip/scripts/object_expansion/partner_chunks/`).

### (b) The raw sourced-mesh cache — REQUIRED, via rsync (9.3 GB, not in git)
`batch_expand.py` reads each candidate's source mesh from a path in `candidates_all.csv`
(`local_path` column) and does **not** auto-download. Without this, `mesh_prep` fails
outright for every not-yet-registered category — which is exactly the categories you'll be
working on.
```bash
# from your repo root, after cloning:
mkdir -p dataset/object_expansion/objaverse_cache
rsync -avz --progress \
  yifeid@arrhenius1.hpc.arrhenius.naiss.se:/nobackup/proj/disk/softenable-codesign26/personal/yifeid/object_expansion_sources/objaverse_cache/ \
  dataset/object_expansion/objaverse_cache/
```
(needs SSH access to Arrhenius. It's a symlink target on the cluster, hence not in git.)

### (c) Already-collected demo trajectories — OPTIONAL (49 GB)
Only if you want to inspect what's already done. `batch_expand.py` never reads
`dataset/demos/` at registration time — it only writes fresh run dirs there at collection
time. Skip unless needed:
```bash
rsync -avz --progress \
  yifeid@arrhenius1.hpc.arrhenius.naiss.se:/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/dataset/demos/ \
  dataset/demos/
```

## 3. Local environment

Use **`envs/sim`** (the standard x86_64 env — NOT `envs/sim_arrhenius`, which is pinned to
the cluster's aarch64 GH200 + cu126 build). **You need an NVIDIA GPU with CUDA** — Genesis's
MPM solver runs on GPU.
```bash
uv sync --project envs/sim
# torch is installed separately — see envs/sim/pyproject.toml header for the pinned cu-triple,
# swap it to match your GPU/driver if different:
uv pip install --python envs/sim/.venv/bin/python "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
uv run --project envs/sim python -m pytest gentle_manip/tests/ -q   # sanity check
```
You do NOT need `pymeshlab` unless a specific object hits the meshlab hole-close path — the
cluster sbatch self-heals it, locally just `uv pip install pymeshlab` if you see
`No module named 'pymeshlab'`.

## 4. Your assigned categories — avoid duplicating the cluster

The 342 remaining categories are split by the shape-diversity ordering:
`gentle_manip/scripts/object_expansion/partner_chunks/`

| File | Contents |
|---|---|
| `ALL_partner_categories.txt` | **Your 171 categories** (back half of the diversity-ordered remaining pool) |
| `chunk_0.txt` … `chunk_7.txt` | The same 171, round-robin split into 8 for local parallelism (run several at once if your GPU has headroom) |
| `cluster_working_front_half.txt` | The 171 the cluster is working — **don't run these**; if the cluster clearly stalls we'll re-split |
| `collected_so_far.txt` | The 78 already done — the pipeline skips these automatically, listed for reference |

The pipeline's own guard (`already_registered()` reading `registry.py`, plus the
`EXPANSION_LOG.csv` collected-category check if you use `--only-categories` derived from it)
prevents re-registering a name that exists. Worst case if both sides do the same category:
it just gets a second registered instance and a second collection — not catastrophic (the
Grasp Reel dedupes to one clip per category), just wasted GPU time. The front/back split
avoids that.

## 5. Run collection

Local equivalent of the cluster's sbatch (no SLURM, no submodule-sync, no pymeshlab swap):
```bash
cd gentle_manip   # repo root
# one chunk:
uv run --project envs/sim python -m gentle_manip.scripts.object_expansion.batch_expand \
  --n-objects 40 --per-category 1 --n-episodes 20 --n-envs 5 --min-success 0.5 \
  --candidates-csv dataset/object_expansion/candidates_all.csv \
  --only-categories "$(cat gentle_manip/scripts/object_expansion/partner_chunks/chunk_0.txt)"
```
Loop over `chunk_0 … chunk_7` (sequentially, or in parallel if your GPU has memory
headroom — each run owns one Genesis process). Re-invoke with the next chunk when one
finishes; there's no queue, only your GPU throughput. `--skip-collect` does a fast
register-only pass (mesh_prep + FEM gate, no MPM) if you want to see what geometry passes
before spending GPU time.

Every object writes:
- `gentle_manip/assets/objects/<name>.obj` + `configs/{tasks,dr,experiments}/single_lift_<name>_soft*.yaml`
- a row in `gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv`
- if it reaches collection: `dataset/demos/single_lift_<name>_soft/<run>/{data.pkl,stats.yaml,videos/,config.yaml}`

## 6. Feeding the shared Grasp Reel page (regular monitoring)

**Your collected categories WILL show up on the same Grasp Reel page** —
<https://claude.ai/code/artifact/84076dfb-7344-4de9-80c3-283058894c7a> — so we can watch
combined progress. Mechanism:

1. **After each batch, commit and push your results** to `expand-categories-30`:
   ```bash
   git add gentle_manip/assets/objects gentle_manip/assets/registry.py \
           gentle_manip/configs/{tasks,dr,experiments} \
           gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv
   git commit -m "local collection: <N> more categories"
   git pull --rebase origin expand-categories-30   # see conflict note below
   git push origin expand-categories-30
   ```
   `dataset/demos/` is gitignored (49 GB of it) — don't try to commit it. The Grasp Reel
   only needs `EXPANSION_LOG.csv` + `stats.yaml`, and the stats numbers are in the commit;
   the video *clips* on the Reel are regenerated cluster-side (Β§6.3) so a missing local
   `dataset/demos/` on the publishing box just means your categories show with stats but no
   clip until the next cluster-side clip export — acceptable for monitoring.
2. **The pages are refreshed from a Claude Code session** (currently yifeid's) that pulls
   the branch and re-runs:
   ```bash
   uv run --project envs/sim python gentle_manip/scripts/object_expansion/export_artifact_data.py --out /tmp/a.json --max-tris 400
   uv run --project envs/sim python gentle_manip/scripts/object_expansion/export_category_summary.py --out /tmp/c.json
   # (video clips: export_video_gallery.py — needs the actual dataset/demos/ + ffmpeg)
   ```
   then splices into `tools/object_viewer/{artifact_template,video_gallery_template}.html`
   and republishes via the Artifact tool. Cadence: **every ~30 min** while collection is
   active. If you want to drive a refresh yourself from your own Claude Code session:
   `Artifact` tool → `action: "read"` the Reel URL → splice fresh JSON into the template →
   publish with `url` set to that same link.

### EXPANSION_LOG.csv merge conflicts
Both sides append rows to this file, so `git pull --rebase` will conflict on it. It's
**append-only** — resolve by **keeping every row from both sides** (union; order doesn't
matter, each row is a distinct attempt). `git checkout --theirs` then re-append your new
rows, or just hand-merge — never drop rows. The category-summary/artifact exporters tolerate
duplicate rows fine.

## 7. Collected vs. not — priority

**Already collected (78) — skip these:**
```
Dixie_cup apple apricot armor asparagus atomizer avocado ball bandanna basketball beachball
beer_can bell_pepper belt_buckle blender bolt bottle broccoli bubble_gum bucket bullhorn
calendar candle candle_holder cantaloup cappuccino chocolate_bar chocolate_cake
cleansing_agent clementine clothes_hamper coconut coin cream_pitcher crouton cup cupcake
diaper die doorknob egg fan fleece funnel garbage gargoyle goggles grape hairnet hand_glass
handbag headlight honey ice_skate icecream keg lime log mallard mandarin_orange meatball
medicine melon milk milkshake money olive_oil orange_(fruit) orange_juice parasail_(sports)
pea_(food) piggy_bank pita_(bread) potato salad shot_glass soccer_ball softball
```

**Your priority list (171) — `partner_chunks/ALL_partner_categories.txt`**, already
ordered by shape-bucket diversity (concave / thin-shell / tall-narrow first, then
compact / unknown). Highlights of the underrepresented shape buckets to hit early:
`globe golf_club green_onion gun hair_dryer hammock harmonium hose hot-air_balloon iPod
inhaler iron_(for_clothing) joystick lamp lantern lawn_mower lightbulb mop pan_(for_cooking)
pistol projector propeller radar saxophone shovel ski strainer sword telephone thimble
toaster umbrella wind_chime ...`

The remaining front-half 171 is the cluster's; `partner_chunks/cluster_working_front_half.txt`.

## 8. Known failure modes (so you know what's "normal")

From this session's DEVLOG (`docs/final/DEVLOG.md`, 2026-09-09 entries):
- `mesh_prep` FAILED — repair couldn't make it watertight (some Objaverse scans are hopeless);
- `fem_gate` FAIL — tet count over cap / extent ratio off (blocky or degenerate geometry);
- `collect` crashed `rc=-6` — MPM "soft body blew up" (sharp geometry, stress blowup);
- `collect` crashed `rc=1` — rigid-solver NaN at scene settle, or particles-outside-boundary
  (object too big for the fixed MPM domain after DR scale-up);
- `collect` **timeout** — collection exceeded 30 min (fixed 2026-09-09 to not kill the batch).

All of these are logged per-object in `EXPANSION_LOG.csv` and the batch moves on. A ~85%
loss rate at the collection stage is expected.

## 9. Training data — the demos collected so far

As of 2026-09-11: **93 categories / 129 object instances / 2,580 episodes** have a
complete, accepted collection (`collection_status == accepted` in `EXPANSION_LOG.csv` —
this means the object hit `--min-success 0.5` and saved the full `--n-episodes 20`).
This is what's ready to train on right now (separate from the "98 categories" figure on
the Grasp Reel page, which also counts categories still mid-collection with partial data).

**Full path on the cluster (all demos, including in-progress/failed attempts, 67 GB):**
```
/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/dataset/demos/
```
Each object instance's demos live at `dataset/demos/single_lift_<name>_soft/<run_id>/`,
containing `data.pkl` (the (obs, action) episodes), `stats.yaml` (success rate etc.),
`videos/` (per-episode render), `config.yaml`/`config_resolved.yaml` (env snapshot —
hard requirement #7 in `CLAUDE.md`).

**You only need the 129 accepted-object directories for training (~11 GB, not the full
67 GB)** — the rest is failed/partial/low-success attempts kept only for pipeline
diagnostics. Regenerate the exact list from the (git-tracked) `EXPANSION_LOG.csv` and
pull just those:
```bash
cd gentle_manip   # repo root, after `git pull origin expand-categories-30`
python3 -c "
import csv
cats = {}
with open('gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv') as f:
    for row in csv.DictReader(f):
        if row['collection_status'] == 'accepted':
            cats.setdefault(row['category'], []).append(row['name'])
print(f\"{len(cats)} categories, {sum(len(v) for v in cats.values())} object instances\")
with open('/tmp/accepted_demo_dirs.txt', 'w') as o:
    for names in cats.values():
        for n in names:
            o.write(f'single_lift_{n}_soft\n')
"
rsync -avz --progress --files-from=/tmp/accepted_demo_dirs.txt \
  yifeid@arrhenius1.hpc.arrhenius.naiss.se:/nobackup/proj/disk/softenable-codesign26/personal/yifeid/gentle_manip/dataset/demos/ \
  dataset/demos/
```

**Accompanying files needed for training (all already in git, come from `git pull`):**
- `gentle_manip/assets/objects/*.obj` — the meshes (needed if any eval/env-rebuild step
  re-instantiates the Genesis scene, e.g. DP3's in-training sim eval bridge).
- `gentle_manip/assets/registry.py` — `OBJECT_MAP`, maps each object name to its mesh +
  material defaults.
- `gentle_manip/configs/{tasks,dr,experiments}/single_lift_<name>_soft*.yaml` — per-object
  task/DR/experiment config trio, one set per registered name.
- `gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv` — full pipeline audit log
  (per-object status, success rate, episode counts) — the source of truth used above.

You do **not** need `dataset/object_expansion/objaverse_cache` (9.3 GB) for training —
that's only the raw source-mesh cache used to *register new* categories (§2b). Skip it
unless you're also going to run collection yourself.

## 10. DR coverage (open item)

Current per-object DR: position XY, full yaw, ±45Β° pitch/roll, 25% upside-down flip,
scale ×[0.6,1.6] uniform + ×[0.6,1.6] on one random axis, mild bend/twist/taper, E/ν/ρ
bands, friction, robot start pose, 10% disturbance. **Not covered: distinct stable resting
orientations** (e.g. a can upright vs. lying on its side) — ±45Β° tilt isn't enough to lay
an object flat. Adding `trimesh.poses.compute_stable_poses()`-based orientation sampling to
the scene-DR path is a flagged TODO; if you want that before collecting, coordinate first —
it's a change to shared sim code (`domain_randomization/`, `SimBackend._apply_scene_dr`).
