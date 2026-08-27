# Generated meshes — which are for use

Photographs -> meshes via **TripoSG** (VAST-AI, MIT licence for code and weights).
Method, failure analysis and gate definitions: `docs/mesh_from_photos.md`.
Generated 2026-08-24 .. 2026-08-27.

## Tiers

| Tier | Meaning | Use it? |
|---|---|---|
| **READY** | watertight, manifold, winding-consistent, euler 2 (genus 0), single component, and silhouette+depth consistent with its input photo | **Yes** — drop into tetgen/FEM |
| **USABLE** | all of the above EXCEPT genus > 0 — handles confined to a thin appendage (tail fan, stalk) | **Yes, with care** — tetgen accepts closed manifold surfaces of any genus; crop the appendage or accept it |
| **REJECT** | shape disagrees with the input photo, OR not watertight/multi-component, OR **odd euler** (non-orientable — tetgen WILL refuse) | **No** |

## Counts per category

| category | READY | USABLE | REJECT | total | usable (R+U) |
|---|---|---|---|---|---|
| `banana1` | 1 | 1 | 0 | 2 | **2/2** |
| `cherry_tomato` | 6 | 0 | 0 | 6 | **6/6** |
| `mushroom1` | 1 | 0 | 0 | 1 | **1/1** |
| `mushroom2` | 1 | 0 | 0 | 1 | **1/1** |
| `mushroom3` | 1 | 0 | 0 | 1 | **1/1** |
| `raspberry` | 6 | 0 | 0 | 6 | **6/6** |
| `shrimps` | 5 | 3 | 0 | 8 | **8/8** |
| `tomato` | 4 | 0 | 1 | 5 | **4/5** |
| `strawberry1` | 0 | 0 | 0 | 0 | **0/0** — no candidate selected; all 3 seeds fail on calyx genus |
| **TOTAL** | **25** | **4** | **1** | **30** | **29/30** |

## Per-mesh

| category | mesh | tier | euler | genus | faces | shape err | file |
|---|---|---|---|---|---|---|---|
| banana1 | IMG20260824145616 | **USABLE** | 0 | 1 | 12000 | 2.9% | `obj_meshes/banana1/selected/IMG20260824145616.obj` |
| banana1 | IMG20260824145624 | **READY** | 2 | 0 | 11982 | 0.3% | `obj_meshes/banana1/selected/IMG20260824145624.obj` |
| cherry_tomato | cherry_tomato1 | **READY** | 2 | 0 | 12000 | 2.8% | `obj_meshes/cherry_tomato/selected/cherry_tomato1.obj` |
| cherry_tomato | cherry_tomato2 | **READY** | 2 | 0 | 12000 | 1.5% | `obj_meshes/cherry_tomato/selected/cherry_tomato2.obj` |
| cherry_tomato | cherry_tomato3 | **READY** | 2 | 0 | 12000 | 14.8% | `obj_meshes/cherry_tomato/selected/cherry_tomato3.obj` |
| cherry_tomato | cherry_tomato4 | **READY** | 2 | 0 | 12000 | 1.2% | `obj_meshes/cherry_tomato/selected/cherry_tomato4.obj` |
| cherry_tomato | cherry_tomato5 | **READY** | 2 | 0 | 11704 | 3.5% | `obj_meshes/cherry_tomato/selected/cherry_tomato5.obj` |
| cherry_tomato | cherry_tomato6 | **READY** | 2 | 0 | 12000 | 1.7% | `obj_meshes/cherry_tomato/selected/cherry_tomato6.obj` |
| mushroom1 | mushroom1 | **READY** | 2 | 0 | 11994 | 0.0% | `obj_meshes/mushroom1/clean.obj` |
| mushroom2 | mushroom2 | **READY** | 2 | 0 | 11988 | 0.7% | `obj_meshes/mushroom2/clean.obj` |
| mushroom3 | mushroom3 | **READY** | 2 | 0 | 12000 | 1.4% | `obj_meshes/mushroom3/clean.obj` |
| raspberry | raspberry1 | **READY** | 2 | 0 | 11988 | 8.3% | `obj_meshes/raspberry/selected/raspberry1.obj` |
| raspberry | raspberry2 | **READY** | 2 | 0 | 11992 | 3.0% | `obj_meshes/raspberry/selected/raspberry2.obj` |
| raspberry | raspberry3 | **READY** | 2 | 0 | 11924 | 5.7% | `obj_meshes/raspberry/selected/raspberry3.obj` |
| raspberry | raspberry4 | **READY** | 2 | 0 | 11988 | 1.7% | `obj_meshes/raspberry/selected/raspberry4.obj` |
| raspberry | raspberry5 | **READY** | 2 | 0 | 12000 | 8.5% | `obj_meshes/raspberry/selected/raspberry5.obj` |
| raspberry | raspberry6 | **READY** | 2 | 0 | 11936 | 3.2% | `obj_meshes/raspberry/selected/raspberry6.obj` |
| shrimps | shrimp1 | **USABLE** | -32 | 17 | 11782 | 0.1% | `obj_meshes/shrimps/selected/shrimp1.obj` |
| shrimps | shrimp2 | **READY** | 2 | 0 | 12000 | 1.6% | `obj_meshes/shrimps/selected/shrimp2.obj` |
| shrimps | shrimp3 | **READY** | 2 | 0 | 12000 | 7.6% | `obj_meshes/shrimps/selected/shrimp3.obj` |
| shrimps | shrimp4 | **READY** | 2 | 0 | 11994 | 23.7% | `obj_meshes/shrimps/selected/shrimp4.obj` |
| shrimps | shrimp5 | **USABLE** | -64 | 33 | 10884 | 1.0% | `obj_meshes/shrimps/selected/shrimp5.obj` |
| shrimps | shrimp6 | **USABLE** | 0 | 1 | 12000 | 1.5% | `obj_meshes/shrimps/selected/shrimp6.obj` |
| shrimps | shrimp7 | **READY** | 2 | 0 | 12000 | 5.2% | `obj_meshes/shrimps/selected/shrimp7.obj` |
| shrimps | shrimp8 | **READY** | 2 | 0 | 12000 | 0.3% | `obj_meshes/shrimps/selected/shrimp8.obj` |
| tomato | tomato1 | **READY** | 2 | 0 | 12000 | 13.6% | `obj_meshes/tomato/selected/tomato1.obj` |
| tomato | tomato2 | **REJECT** | 1 | n/a | 11962 | 0.3% | `obj_meshes/tomato/selected/tomato2.obj` |
| tomato | tomato3 | **READY** | 2 | 0 | 12000 | 8.0% | `obj_meshes/tomato/selected/tomato3.obj` |
| tomato | tomato4 | **READY** | 2 | 0 | 12000 | 1.0% | `obj_meshes/tomato/selected/tomato4.obj` |
| tomato | tomato5 | **READY** | 2 | 0 | 12000 | 4.0% | `obj_meshes/tomato/selected/tomato5.obj` |

## Manual caveats the automated tiers do NOT capture

The tiers measure **validity**, not **fidelity**. A mesh can be watertight, genus 0 and
silhouette-consistent while still being a poor likeness.

- **`shrimps/shrimp5` — USABLE by the tiers, but visually poor.** Watertight and only 1.0%
  off its silhouette, yet the surface is lumpy and torn (its source photo is a partly
  broken shrimp with a near-closed curl). Inspect `obj_meshes/shrimps/runs/shrimp5_seed2/`
  before using it for anything where surface detail matters.
- **`shrimps/shrimp1` (genus 17) and `shrimp6` (genus 1)** look good; their handles sit in
  the tail fan / legs, away from any grasp contact patch.
- **`tomato/tomato2` is the only REJECT.** All 3 seeds fail; its calyx is dried papery
  sepals lying flat against the fruit. Odd euler (1) means non-orientable, so tetgen will
  refuse it — this is not a 'crop the appendage' case. Re-shoot without the calyx.
- **`strawberry1` has no mesh at all** — 0/3 seeds pass, curling calyx sepals.

## Two properties shared by EVERY mesh here

1. **Not metrically scaled.** No `measurements.json` for any object. Each mesh is in
   TripoSG's normalised frame; its `report.json` carries `longest_axis_normalised`.
   To make metric: `scale = longest_axis_metres / longest_axis_normalised`.
2. **Unobserved surfaces are invented, not measured.** These are single-image
   reconstructions, so any face not visible in the source photo is a prior-driven guess.
   For a side-on photo that means the underside — the region that sets the grasp contact
   patch. See `obj_meshes/mushroom1/_underside_check.png`.
## Provenance — READ BEFORE PUBLISHING

Source images differ in origin and this affects what you may do with the meshes:

| category | source | note |
|---|---|---|
| `mushroom1/2/3`, `banana1` | photographed in the lab | ours; unrestricted |
| `shrimps`, `strawberry1`, `cherry_tomato`, `raspberry`, `tomato` | **watermarked stock comps** (Alamy / Shutterstock / Dreamstime / depositphotos) | derivative works of licensed imagery |

The stock-derived meshes are fine for pipeline development and internal simulation. Before
any of them appears in a paper figure, a released dataset, or a public artefact, check the
licence position — a mesh generated from a watermarked comp is a derivative of it. In
`shrimp1` the Alamy watermark is tiled directly across the fruit body and is a genuine
input confound, not merely a legal footnote.

The MODEL is not the constraint here: TripoSG is MIT for both code and weights. The source
photographs are.
