# TripoSG photo → mesh pipeline — setup and use

Turn photographs of an object into clean, watertight, decimated triangle meshes for
`assets/objects/` / FEM. Written to be followed start-to-finish by someone who has not
touched this pipeline before.

- **Model:** [TripoSG](https://github.com/VAST-AI-Research/TripoSG) (VAST-AI) — **MIT for
  both code and weights**. Paper: <https://arxiv.org/abs/2502.06608>.
- **Why not Hunyuan3D:** its community licence excludes the EU from its "Territory" and
  §5.c forbids using the model *or its output* outside that Territory. Arrhenius is in
  Sweden. Details + the TRELLIS evaluation: `docs/mesh_from_photos.md` §1.
- **Findings, failure modes and gate rationale:** `docs/mesh_from_photos.md`.
- **What has been produced and what is fit to use:** `obj_meshes/USABLE.md`.

---

## 0. TL;DR

```bash
# once, INSIDE a GPU-node job (aarch64 wheels)
srun -A naiss2026-3-141-gpu -p gpu --gres=gpu:1 -n1 -c16 -t 1:00:00 \
     bash scripts/mesh_from_photos_setup.sh

# per object: 1) drop photos in   2) matte + REVIEW   3) generate/clean/render   4) select
mkdir -p obj_images/<object> && cp ~/photos/*.jpg obj_images/<object>/

uv run --project envs/triposg_arrhenius python scripts/mesh_from_photos/prep_images.py \
    --input-dir obj_images/<object> --output-dir obj_meshes/<object>
#   >>> now LOOK at obj_meshes/<object>/prepped/_contact_sheet.png <<<

sbatch --export=ALL,OBJ=<object> scripts/mesh_from_photos_object.sbatch

uv run --project envs/triposg_arrhenius python scripts/mesh_from_photos/select_per_image.py \
    --object <object>
uv run --project envs/triposg_arrhenius python scripts/mesh_from_photos/shape_consistency.py \
    --object <object>
```

---

## 1. Setup

TripoSG is a **git submodule** pinned to upstream `fc5c409`
(`https://github.com/VAST-AI-Research/TripoSG.git`, branch `main`).

```bash
git submodule update --init third_party/TripoSG
bash scripts/mesh_from_photos_setup.sh     # in a GPU job -- see below
```

### Two things that will bite you

**(a) Run setup on a GPU NODE, not the login node.** Arrhenius login nodes are **x86_64**
while GPU nodes are **aarch64 (GH200)**. `envs/triposg_arrhenius` pins aarch64 cu126 torch
wheels, so `uv sync` on the login node resolves the wrong build. Unlike Alvis, **Arrhenius
GPU nodes have outbound network**, so HF weights download inside the job — no login-node
pre-staging needed.

**(b) The submodule is deliberately CLEAN — do not patch it.** TripoSG's
`inference_utils.py` does `from diso import DiffDMC` at module level, so the import fails
without `diso` even though `diso` is only *used* by the flash decoder we never call. `diso`
is sdist-only with no aarch64 wheel and the only CUDA module here is 13.0. Rather than
carry a local patch, we supply a **stub** at `scripts/mesh_from_photos/shims/diso/`, which
`generate.py` **appends** to `sys.path` (so a real `diso`, if ever installed, wins). If you
write your own driver script, append that shims dir too.

There is also `third_party/patches/triposg-lazy-diso.patch` — the equivalent local patch,
kept only for reference. **You should not need it.**

---

## 2. Running it

### Two directory conventions — pick the right one

| Your input | Meaning | Output |
|---|---|---|
| `obj_images/<obj>/{front,left,back,right}.jpg` | **ONE object, several views** | one mesh at `obj_meshes/<obj>/clean.obj` |
| `obj_images/<obj>/{a,b,c}.jpg` | **SEVERAL DIFFERENT objects**, one per photo | one mesh per photo in `obj_meshes/<obj>/selected/` |

⚠️ **TripoSG is a SINGLE-IMAGE model.** It has no analogue of Hunyuan3D-2mv's
`run_multi_image`, so it **cannot fuse views**. In the multi-view case each view still
produces an *independent* mesh and you pick the best; the views are alternatives, not
inputs to one reconstruction.

### Step 1 — matte the photos (this is a REVIEW GATE, not a formality)

```bash
uv run --project envs/triposg_arrhenius python scripts/mesh_from_photos/prep_images.py \
    --input-dir obj_images/<object> --output-dir obj_meshes/<object>
```

Removes background (`rembg`), drops non-largest foreground blobs (stock-agency banner bars
would otherwise stretch the crop box), crops to the alpha bbox +5%, resizes longest side to
1024. Accepts jpg/jpeg/png/webp/avif/bmp/tif/tiff.

**Then actually open `obj_meshes/<object>/prepped/_contact_sheet.png`.** Segmentation
failure is the main cause of bad meshes and is obvious to a human in two seconds. The
automated flags in `_prep_report.json` are necessary but *not sufficient* — a hand gripping
the fruit is one connected component, does not touch the border, and barely moves the
soft-alpha number, so nothing fires. Look for:

- rig hardware kept as foreground (skewers, tape, a hand) → it becomes geometry
- the object clipped at the image border
- a retained shadow (a skirt of geometry on the mesh)

### Step 2 — generate, clean, render (one GPU job)

```bash
sbatch --export=ALL,OBJ=<object> scripts/mesh_from_photos_object.sbatch
# optional: SEEDS="0 1 2 3 4"   VIEWS="front back"
```

Per (view, seed): generate → floater removal → staged quadric decimation to ~12k faces →
watertight repair → §6 validation → 90-frame turntable with the reference photo pinned
beside it. 3 seeds by default because output varies and you want the spread.
~7–15 s generation per mesh; turntables are the slow part (~2 min each).

### Step 3 — select and audit

```bash
python scripts/mesh_from_photos/select_per_image.py --object <object>   # per-image dirs
python scripts/mesh_from_photos/shape_consistency.py --object <object>
python scripts/mesh_from_photos/write_readme.py --object <object> --select <tag>  # single-object dirs
```

Then look at `obj_meshes/<object>/_candidates.png`.

---

## 3. Reading the output — the gates and what they miss

`report.json` per candidate carries the spec's §6 numbers. **Every one of those gates is
TOPOLOGICAL** (watertight / euler==2 / face count / positive volume / one component).

**They cannot tell you the mesh is the wrong shape.** A real case:
`cherry_tomato1_seed1` passed all of them while being a **3.63:1 rod generated from a photo
of a round tomato**, with 13× too little volume. Hence `shape_consistency.py`, which runs
two independent checks:

1. **in-plane** — mesh X/Y vs photo width/height (TripoSG emits +X = image right,
   +Y = up, +Z = depth)
2. **depth** — `Z / max(X,Y)`, **upper bound only**: a single-view model inventing depth
   beyond the visible silhouette is a hallucination, whereas a flat or curled object
   legitimately measures 0.2–0.3

`select_per_image.py` ranks **shape consistency ABOVE the §6 gates**, because genus > 0 does
not block tetgen (it needs closed + manifold + self-intersection-free) whereas wrong
geometry is fatal.

Usability tiers and per-mesh verdicts: **`obj_meshes/USABLE.md`**.

### Which objects work

Yield tracks one property: **how much roughly-PARALLEL thin surface the object has**,
separated by less than a voxel of the ~512³ grid.

| | example | yield |
|---|---|---|
| chunky, convex | mushroom2 | **9/9** |
| bumpy but convex (V-shaped valleys) | raspberry druplets | 12/18 |
| flat-lying calyx | cherry_tomato, tomato4 | 3/3 each |
| **thin parallel sheets** | strawberry calyx, shrimp tail fan, mushroom3 torn stem, tomato2 dried sepals | **0/3 – 3/9** |

Bumpy convex detail is cheap; thin parallel geometry is not. No cleanup setting fixes it —
`genus_trace.py` shows the handles are already in the generated surface before any
decimation (strawberry: genus 36 at 2.1 M faces), and decimation actually *reduces* them.

---

## 4. Two properties of EVERY mesh from this pipeline

1. **Not metrically scaled.** Meshes are in TripoSG's normalised frame;
   `report.json` carries `longest_axis_normalised`. To make metric, drop
   `obj_images/<object>/measurements.json` containing `{"longest_axis_mm": <caliper>}` and
   re-run `postprocess.py` — you get a `scaled.obj` in metres. Without a measurement the
   pipeline refuses to guess.
2. **Unobserved surfaces are invented, not measured.** Single-image reconstruction: any
   face not visible in the source photo is a prior-driven guess. For a side-on photo that
   is the **underside** — precisely the region that sets the grasp contact patch. See
   `obj_meshes/mushroom1/_underside_check.png`: a smooth featureless dome, no gills.
   Treat the unobserved hemisphere as fiction.

## 5. Provenance

Meshes derived from **watermarked stock photos** (shrimps, strawberry1, cherry_tomato,
raspberry, tomato) are fine for internal simulation but are derivative works of licensed
imagery — check the position before any paper figure or released dataset. The model licence
is not the constraint; the photographs are. Lab-shot objects (mushroom1/2/3, banana1) are
unrestricted. Table in `obj_meshes/USABLE.md`.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: diso` | your driver script did not append `scripts/mesh_from_photos/shims` to `sys.path` (§1b) |
| `AttributeError: 'NoneType' object has no attribute 'astype'`, minutes in | flash decoder was selected. TripoSG defaults `use_flash_decoder=True` and swallows the failure in a bare `except`. Pass `use_flash_decoder=False` |
| `CUDA error: ...device(s) is/are busy or unavailable` | that GPU node is bad, not your code. Resubmit with `--exclude=<node>` |
| `uv sync` picks wrong torch / fails on wheels | you are on the x86_64 login node. Run inside a GPU job (§1a) |
| Postprocess is slow or dies on the login node | login nodes are too contended for 2 M-face meshes (a load taking 2.3 s on a GH200 did not finish in 110 s). Go through SLURM |
| A mesh looks wrong but passed all gates | run `shape_consistency.py`; the §6 gates are topological only (§3) |
| `pymeshlab` / `open3d` install failure | neither has an aarch64 Linux wheel at any version. Use `fast-simplification` + `manifold3d`, already in the env |
