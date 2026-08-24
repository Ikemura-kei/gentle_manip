# Photographs → FEM-ready meshes

Deep-dive subpage for the object-scanning pipeline. Linked from `docs/DEVLOG.md`.

Turns photographs of a real food object into a clean, watertight, decimated triangle
mesh for `assets/objects/`, ready for tetrahedralisation and soft-body FEM.

---

## 1. Model choice

Tencent's Hunyuan 3D generator is **banned in this project**: its community licence
excludes the EU from its territory and the restriction explicitly reaches the model's
OUTPUT, so on a Swedish cluster a generated mesh would contaminate `assets/objects/`
and every derived dataset or figure. Do not clone it or download its weights here.

### What we use instead

| | (banned generator) | TRELLIS | **TripoSG (chosen)** |
|---|---|---|---|
| Licence | EU-excluded (see above) | MIT | **MIT — code AND weights** |
| Views | multi-view (fuses 3–4) | multi-image | **single image** |
| aarch64 build | n/a | needs `spconv` + `flash-attn`, neither has ARM wheels; multi-hour source build against the only available toolchain (CUDA 13) | **pure PyTorch, installs in minutes** |

TRELLIS was the first MIT candidate but is impractical here: `spconv` has no aarch64
wheel at any version and neither does `xformers`; `flash-attn` is sdist-only. Its
sparse backend is not optional — `trellis/modules/sparse/basic.py` uses
`spconv.pytorch.SparseConvTensor` as the tensor container itself.

**The cost of this choice: TripoSG is single-image.** It cannot fuse views the way
the banned generator `run_multi_image` does. `generate.py` therefore produces one
INDEPENDENT mesh per view, to be compared, not one mesh conditioned on all views.

The clones for all three live in `third_party/`; only `TripoSG/` is wired up.

---

## 2. Environment (the commands that actually ran)

Target: NAISS Arrhenius. The login node is **x86_64** with a compute-mode-Prohibited
L40; all GPU nodes are **aarch64 GH200 120 GB, glibc 2.34**. Wheels differ between
the two, so the env must be synced **from inside a GPU job**.

Unlike Alvis, **Arrhenius GPU nodes have outbound network**, so HF weights download
inside the job — no login-node pre-staging needed.

```bash
git clone https://github.com/VAST-AI-Research/TripoSG.git third_party/TripoSG
sbatch scripts/mesh_from_photos_sync.sbatch      # uv sync on a GH200 node
```

`envs/triposg_arrhenius/pyproject.toml` holds the full spec. Python 3.12, torch
2.9.1+cu126 aarch64 (same triple as `envs/sim_arrhenius`).

### Divergences from the task spec's install snippets

| Spec says | Reality |
|---|---|
| `pip install pymeshlab` | **No aarch64 Linux wheel exists at any version.** <2025 is x86-only; 2025.x needs `manylinux_2_35` and Arrhenius glibc is 2.34. Replaced with `fast-simplification` for quadric decimation (+ `manifold3d` for watertight repair). |
| Use the repo's `FloaterRemover` / `DegenerateFaceRemover` / `FaceReducer` | Those are the banned generatorclasses and are `pymeshlab`-backed. `postprocess.py` implements the same sequence on `trimesh` + `manifold3d` + `fast-simplification`. |
| `HF_HOME` on login node before submitting | Not required here — GPU nodes have network. Still redirected to project storage (`$HOME` has only ~24 GB). |
| Pipeline call | n/a — TripoSG is `TripoSGPipeline(image=<PIL>)`. |

### Local patches to `third_party/TripoSG` (marked in-file)

1. `triposg/inference_utils.py` — `from diso import DiffDMC` made lazy. `diso` is an
   sdist-only CUDA extension with no aarch64 wheel; the only CUDA module on Arrhenius
   is 13.0, so we do not build it.

**Related trap:** `TripoSGPipeline.__call__` has `use_flash_decoder=True` by
**default**, and that is the path that needs `diso`. Its failure is swallowed by a
bare `except` inside `flash_extract_geometry`, which returns `(None, None)` and
surfaces ~4 minutes later as `AttributeError: 'NoneType' object has no attribute
'astype'`. `generate.py` therefore passes `use_flash_decoder=False` explicitly,
selecting `hierarchical_extract_geometry` + skimage marching cubes.

---

## 3. Running it

```bash
# 1. matte + crop + resize the photos, then LOOK at obj_meshes/<obj>/prepped/_contact_sheet.png
python scripts/mesh_from_photos/prep_images.py \
    --input-dir obj_images/mushroom1 --output-dir obj_meshes/mushroom1

# 2. generate (GPU job; 4 views x 3 seeds)
sbatch scripts/mesh_from_photos_generate.sbatch

# 3. clean + validate every run, then turntable the chosen one (CPU, login node)
bash scripts/mesh_from_photos_finalize.sh mushroom1 <best_tag>
```

Stage 1 is a **review gate**, not a formality: segmentation failure is the most common
cause of a bad mesh and is obvious to a human in two seconds. `_prep_report.json`
flags fragmented masks, border-touching masks, and high partial-alpha fractions
(retained shadow).

The turntable renderer is a self-contained numpy z-buffer rasteriser
(`turntable.py`). Deliberately no OpenGL/EGL: compute nodes are headless and neither
`pyrender` nor `open3d` has a working aarch64 path here. It pins the reference photo
beside the render, because these models apply a symmetry prior and invent the
unphotographed underside — the mesh has to be eyeballed against the actual object.

---

## 3b. Traps hit while building this (each cost a job)

1. **`use_flash_decoder=True` is the TripoSG default.** That path needs `diso`, and
   `flash_extract_geometry` swallows the ImportError in a bare `except`, returns
   `(None, None)`, and fails minutes later as `AttributeError: 'NoneType' object has
   no attribute 'astype'`. Always pass `use_flash_decoder=False` here.
2. **`pkill -f <pattern>` kills your own shell** if the pattern appears in that
   shell's command line — exit 144, no output, no traceback. This masqueraded as a
   "login node killed my job" for several iterations and silently discarded a file
   edit. Verify with `bash -c 'pkill -f "zzz_marker"; echo SURVIVED'`.
   Separately, the login node is too CONTENDED for mesh-scale work (a 2.1M-face GLB
   would not even load in 110 s there, vs 2.3 s load+split on a GH200 node), so run
   everything past `prep_images.py` through SLURM — for speed, not because it is killed.
   This account can only submit to the `gpu` partition.
3. **`manifold3d` VALIDATES, it does not repair.** Given non-manifold input its
   constructor reports an error status and yields an EMPTY manifold. Accepting that
   silently deletes the mesh (symptom: `RuntimeWarning: Mean of empty slice` from
   `centroid`, then `TypeError: 'NoneType' object is not iterable` from `extents`).
   `postprocess.py::_manifold_repair` checks `is_empty()`/`num_tri()` and falls back
   to the input mesh rather than shipping a destroyed one.
4. **Decimate BEFORE hole filling / manifold repair** — repair costs minutes at 2M
   faces and milliseconds at 12k. (An earlier version of this file also claimed
   `trimesh.split()` was minutes-slow and had been replaced with a scipy
   connected-components pass. That was wrong on both counts: the edit never applied,
   and `split()` measures 2.3 s at 1.87M faces on a GH200 node.)

---

## 4. Known limitations of this pipeline (method, not bugs)

- **The underside is invented, not measured.** Nothing in a set of equatorial photos
  constrains the bottom surface. For a mushroom's gills this is likely wrong in
  exactly the region that determines the grasp contact patch. A **bottom-view photo**
  is the single highest-value addition to an input set — and it is obtainable here,
  since the object hangs from a skewer rather than sitting on a table.
- **No metric information exists in the input.** Scale comes entirely from a caliper
  measurement in `obj_images/<obj>/measurements.json`. Without it `postprocess.py`
  skips scaling and says so in `report.json` rather than guessing.
- **Symmetry prior.** These models favour symmetric completions; an asymmetric object
  may come back suspiciously regular. Compare with the turntable.
- **Rig contamination.** The skewer and its tape flag are foreground to `rembg` and
  become geometry. Occlude or minimise them at capture time.

---

## 5. Results — `mushroom1` (2026-08-24)

Input: 4 photos in `obj_images/mushroom1/` (front/left/back/right, object hanging from
a skewer). **No `measurements.json`**, so the meshes are unscaled — see §4.

12 candidates (4 views x 3 seeds), 6-13 s each on one GH200. All decimated to ~12k faces.

| tag | gate | faces | euler | genus | floaters dropped | volume (normalised) |
|---|---|---|---|---|---|---|
| **back_seed0** | **PASS** | 11994 | 2 | 0 | 236 | 2.125 |
| back_seed1 | PASS | 12000 | 2 | 0 | 207 | 2.214 |
| back_seed2 | FAIL | 11990 | -8 | 5 | 240 | 2.079 |
| front_seed0 | FAIL | 11944 | -2 | 2 | 141 | 0.895 |
| front_seed1 | FAIL | 11980 | -6 | 4 | 196 | 0.949 |
| front_seed2 | PASS | 11998 | 2 | 0 | 144 | 0.923 |
| left_seed0/1/2 | PASS | ~12000 | 2 | 0 | 121-166 | 1.219-1.279 |
| right_seed0/1/2 | PASS | ~12000 | 2 | 0 | 103-172 | 0.929-0.970 |

### The two findings that matter

**1. Seed variance is topological, not just cosmetic.** 3 of 12 came back watertight but
with genuine handles (genus 2-5) and were rejected by the §6 euler gate. Running three
seeds and keeping all three is not ceremony — it is the difference between shipping a
genus-5 mesh into a FEM solver and catching it.

**2. The four views disagree about volume by 2.3x** (back 2.12 vs right 0.93), and the
cause is the segmentation, not the model. Each mesh reproduces its OWN input silhouette
faithfully:

| view | photo silhouette w/h | generated mesh X/Y | matte |
|---|---|---|---|
| back | 0.927 | 0.918 | clean |
| left | 0.807 | 0.745 | tape flag retained |
| front | 0.734 | 0.691 | tape flag retained |
| right | 0.699 | 0.648 | tape flag retained |

`_candidates.png` shows this directly: the front/left/right meshes have the tape
reconstructed as a **protruding flat fin off the stem top**, while all three `back`
meshes are clean. The white tape flag holding the mushroom to the skewer is foreground
to `rembg`, so in front/left/right it extends the silhouette upward. TripoSG pads the input to square, so
a taller silhouette makes the object relatively narrower — and the reconstruction
faithfully builds that narrower object. Only `back` has a clean matte, and only `back`
matches the real proportions (cap width / total height ~ 0.92 measured off the photos).

**Selected: `back_seed0`** — passes every §6 gate and is the only view whose input was
uncontaminated. Copied to `obj_meshes/mushroom1/{clean.obj,report.json,turntable.mp4}`.

This is a concrete instance of the doc's warning that segmentation failure is the main
cause of bad output: the §3 review gate caught it, and the volume table quantifies what
it would have cost.

**3. The invented underside is confirmed, and it is the wrong shape for grasping.**
`obj_meshes/mushroom1/_underside_check.png` renders the selected mesh from below: the
bottom is a **smooth featureless dome** — no gills, no annulus, no cap rim. The top view
in the same sheet has real structure because it was photographed. All four input views
are equatorial, so nothing constrains the bottom and the model filled it with the
plausible closed surface its prior prefers.

For FEM grasping this is the region that matters: a real *Agaricus* underside is
flat-to-concave with a gill face and a distinct rim, and that geometry sets the contact
patch and therefore the stress field. Treat the current mesh's lower hemisphere as
fiction. **Highest-value fix: one photo from below.** The object hangs from a skewer, so
unlike the table-top case in the spec's §7 this view is actually obtainable.

---

## 6. Five objects — the euler gate tracks THIN STRUCTURE (2026-08-24)

| object | passed | euler spread | limiting factor |
|---|---|---|---|
| **mushroom2** | **9/9** | all 2 | none — chunky, clean mattes |
| mushroom1 | 9/12 | -8 … 2 | tape flag on 3 of 4 views |
| mushroom3 | 3/9 | -30 … 2 | stem torn into a thin sheet/fin |
| banana1 | 1/6 | -2 … 2 | thin stalk |
| strawberry1 | 0/3 | -25 … -16 | thin curling calyx sepals |

**mushroom2 vs mushroom3 is the closest thing to a controlled experiment here**: same
species, same rig, photographed minutes apart, both with clean mattes. The only material
difference is that mushroom3's stem is torn into a thin fin — and the pass rate goes
9/9 -> 3/9. Within mushroom3 it falls further by view, with how sheet-like the stem
looks in that view: U-notched stem 2/3, thin spike + skewer wire 1/3, broad thin fin 0/3.

This corroborates `genus_trace.py`: the handles are in the GENERATED surface (strawberry
largest component is genus 36 at 2.1M faces, before any decimation), they arise where
two thin surfaces pass within a voxel of the ~512^3 grid, and no cleanup setting removes
them (decimation ratio and target have no measurable effect; decimation actually LOWERS
the strawberry's genus from 36 to 14).

### Practical consequences

- **Chunky objects are near-free.** For anything mushroom2-shaped, expect ~100% yield and
  just take the first passing seed.
- **Thin appendages are the cost driver.** For a calyx/stalk/torn stem, either accept
  genus > 0 or crop the appendage. Note `mushroom3`'s CAP is fine in every candidate —
  only the stem carries the handles.
- **genus > 0 does not block tetgen.** tetgen needs closed + manifold +
  self-intersection-free; a genus-11 watertight manifold tetrahedralises fine. The
  euler==2 rule is a plausibility check for spotting artifacts (working as intended),
  not a solver requirement.
- **An ODD euler is a different fault.** `mushroom3/IMG20260824150710_seed1` returned
  euler=1. A closed ORIENTABLE surface has even euler (2-2g), so an odd value means the
  surface is non-orientable or not truly manifold — `is_watertight` only checks that
  every edge has two faces, which a non-orientable surface can satisfy. The `genus`
  field in `report.json` assumes orientability and is meaningless there; check
  `is_winding_consistent` instead. (TODO: report genus as null for odd euler.)
- **Rig hardware keeps showing up as geometry.** mushroom1's tape became a fin;
  mushroom2 view 3 reconstructs the SKEWER as a rod off the stem in all three seeds;
  mushroom3 view 2's skewer wire coincides with its worst result. Occlude the rig.
- **`rembg` removes a hand well.** Both banana frames came back essentially hand-free.
  Where the hand OCCLUDES the object (banana view 1's stem) the geometry is invented.
