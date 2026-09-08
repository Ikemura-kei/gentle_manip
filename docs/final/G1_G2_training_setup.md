# G1 / G2 training setup — full reference for review

Everything about the two cluster generalist runs, for someone auditing whether anything should be
tuned (batch size, epochs, network width, diffusion horizon, denoising steps, ...). All values below
are read from the runs' own resolved configs (`downloaded_runs/<id>/.hydra/config.yaml`) and from the
checkpoints themselves, not from the template. Local twin: `wiayg`, same dataset and settings as G1.

| run | id | what differs from recipe v5 |
|---|---|---|
| **G1** | `bmbrv` | nothing — recipe v5 exactly |
| **G2** | `ttukt` | encoder-consistency weight 0.3 → 1e-8 (log-only, i.e. ablated) |
| (G0) | `fdcjk` | paired real–sim weight 0.5 → 1e-8 (log-only), listed for context |

The three configs are otherwise byte-identical; `diff` reports exactly one changed line each.

## 1. Dataset

`dataset/dppo/single_lift_generalist_soft_v5` — one converted set, shared by all runs, so the
validation split is pinned and every run is comparable.

| | trajectories | steps | mean episode length |
|---|---|---|---|
| train | 5,643 | 1,074,275 | 190.4 |
| val | 627 | 119,716 | 190.9 |
| total | 6,270 | 1,193,991 | |

Per-sample arrays: `states (8,)`, `actions (7,)`, `point_cloud (1024, 3)`. Split is by trajectory
(10 %), so no episode appears in both. On-disk size 13 GB; the whole train tensor is resident on the
GPU during training (~14 GB), which is the current ceiling on batch size and dataset growth.

Sources: 36 collection runs, 33 objects, from the frozen collector v4.2 (scripted CMA-ES grasp
synthesis, not teleoperation). Composition:

| episodes | objects |
|---|---|
| 550 each | mushroom, tofu, prim_cylinder_mush, prim_ellipsoid_mush, strawberry (475 + 75 merged) |
| 420 | tomato |
| 300 | cherry_tomato |
| 250 each | banana_chunk, prim_cuboid_mush, prim_sphere_mush |
| 100 each | 18 `bs_*` basic shapes (cubes, cylinders, pyramids, spheres, star, ...) |
| 50 each | pasta_bundle, prim_capsule_mush, prim_frustum_mush, prim_hexprism_mush, prim_torus_mush |

Note the imbalance: the four largest objects are 35 % of the data, and the 18 basic shapes together
are 29 %. No resampling or class weighting is applied — plain concatenation.

## 2. Observation and action spaces

**Observation (`cond` dict), two parts:**
- `state`: `(B, 2, 8)` — proprioception history, 2 steps. Per step: ee_pos (3) + ee_quat (4, wxyz,
  sign-canonicalized) + gripper_width (1).
- `point_cloud`: `(B, 1, 1024, 3)` — 1 step of history, 1024 points, xyz only (no RGB, no normals).
  Cropped to the board workspace, outlier-filtered and object-focused, then farthest-point sampled.

**Action:** `(B, 4, 7)` — a chunk of 4 steps, each 7-D absolute: position (3) + euler rotation (3,
frame-offset encoded) + gripper width (1). Absolute targets, not deltas. At deployment `act_steps=4`,
so the whole predicted chunk is executed before re-planning (control at 30 Hz → replan every ~133 ms).
Everything is normalized to [-1, 1] by the dataset's `normalization.npz`.

## 3. Diffusion policy

| parameter | value | note |
|---|---|---|
| denoising steps | **20** | full 20-step denoising at eval too (`use_ddim: false`) |
| beta schedule | cosine | DPPO default, `cosine_beta_schedule(20)` |
| prediction target | epsilon (`predict_epsilon: true`) | loss is MSE on the noise |
| action horizon `Ta` | **4** | steps predicted per forward pass |
| observation history `To` | **2** | proprio steps |
| cloud history `Tpc` | **1** | if raised, features are mean-pooled across steps |
| `denoised_clip_value` | 1.0 | |
| `randn_clip_value` | 10 (train) / 3 (eval cfg) | |

## 4. Network — 2,890,828 parameters (11.6 MB fp32)

`gentle_manip.dppo.pointnet_diffusion.PointNetDiffusionMLP`.

**Point-cloud encoder** (DP3-style PointNet, permutation invariant):
`Linear(3→64) → LayerNorm → ReLU → Linear(64→128) → LayerNorm → ReLU → Linear(128→256) → LayerNorm →
ReLU → max-pool over the 1024 points → Linear(256→512) → LayerNorm`.

**Conditioning vector:** `[cloud feature (512) ⊕ flattened proprio (2 × 8 = 16)] = 528`.

**Denoiser:** `ResidualMLP` over `[noisy action chunk (4 × 7 = 28) ⊕ time embedding (16) ⊕ cond (528)]
= 572` input, hidden `[1024, 1024, 1024]`, ReLU, residual style, output `28`. Time embedding is a
sinusoidal position embedding of dimension 16.

| module | parameters | share |
|---|---|---|
| denoiser ResidualMLP [1024, 1024, 1024] | 2,714,652 | **93.9 %** |
| PointNet projection 256→512 | 132,608 | 4.6 % |
| PointNet per-point MLP 3→64→128→256 | 42,496 | 1.5 % |
| time embedding | 1,072 | < 0.1 % |

The split is worth attention: **the perception encoder is 6 % of the model.** A measured ablation
(`gentle_manip/scripts/final/probe_cloud_reliance.py`) shows the encoder is nonetheless load-bearing —
substituting another episode's cloud raises validation loss by 880 % — so it is doing real work with
few parameters.

## 5. Optimization

| parameter | value |
|---|---|
| epochs | **120** (= 8,393 batches/epoch → 1,007,160 gradient steps) |
| batch size | **128** |
| optimizer | AdamW, lr **1e-4**, weight decay 1e-6 |
| lr schedule | one cosine cycle over the whole run, `first_cycle_steps = n_epochs`, warmup **6 epochs**, `min_lr` 1e-5 |
| EMA | decay 0.995, starts epoch 1, updated every 5 batches; **eval and deploy use the EMA weights** |
| checkpoints | every 20 epochs (6 kept) |
| validation | every 5 epochs |
| seed | 42 |
| wall clock | ~5.6 h on a 4090 at ~21 ms/gradient step (local twin `wiayg`) |

Epoch count is derived from a target of 1,000,000 gradient steps, not chosen directly:
`epochs = ceil(target / ceil(train_steps / batch))`. The schedule knobs above are scaled to the run
length; the template's defaults (100 warmup epochs, checkpoints every 250) were written for
2,000-epoch runs on 100 demonstrations and would misbehave at 120 epochs.

## 6. Auxiliary loss terms

Total loss = behaviour-cloning denoising MSE + the two regularizers below. Aux heads (contact,
object position, grasp width) are all disabled (weight 0).

**Paired real–sim encoder term**, weight **0.5** (G1, G2) / 1e-8 (G0). Cosine distance between the
encoder features of 1,031 real/sim cloud pairs from red-cube play data, 64 pairs sampled per step.
The sim twins get the same cloud noise as the behaviour-cloning path.

**Clean-vs-perturbed encoder consistency**, weight **0.3** (G1) / **1e-8 (G2, ablated)**. On a random
30 % of each batch, cosine distance between the clean cloud's embedding (stop-gradient) and a
strongly perturbed copy's. Perturbation `d435i_noise_strong`: 1.5× sensor noise, 10 % dropout, one
occlusion patch of radius ≤ 1 cm above z = 10 cm, leaked table residue with probability 0.30. No
positional offset in this view.

A weight of 1e-8 rather than 0 keeps the term computed and logged, so the ablated run still reports
its raw value and the curves stay comparable across arms.

## 7. Train-time cloud augmentation (behaviour-cloning path)

The collector records clean clouds, so all augmentation happens in the loss, once per batch, in
training mode only. Validation and deployment see clean clouds.

`d435i_noise_train`: measured D435i stereo noise, axial coefficient 2e-3 m per m² along the camera
ray plus lateral 5e-4, 3 % random dropout, and leaked table-residue clusters with probability 0.15.
Plus a per-sample rigid offset of the whole cloud, uniform in ±5 mm (x), ±3.5 mm (y), ±1.5 mm (z),
with proprioception left untouched, so the policy tolerates cloud-versus-proprio disagreement of the
size measured between real and sim.

## 8. Known open questions for a reviewer

- **Capacity split.** 94 % of parameters are in the denoiser, 6 % in the perception encoder, while
  the measured failures are perception-side (approach precision, and a width bias on small objects).
- **Cherry tomato width bias.** The policy closes to ~27 mm on a 25 mm object where demonstrations
  close to 21 mm. It did not respond to 2.6× more gradient steps, so it is not a convergence issue.
- **Batch size and GPU memory.** The full training tensor is resident on the GPU (~14 GB of 24 GB),
  which constrains both batch size and future dataset growth.
- **Class imbalance.** Plain concatenation, no resampling, with a 11:1 ratio between the largest and
  smallest per-object counts.
- **Horizon and replanning.** `Ta = 4` predicted and all 4 executed; whether a shorter execution
  stride (predict 4, execute 1 or 2) would help has not been tested here.
- **Denoising steps.** 20 at both train and eval; no DDIM acceleration is used, so inference cost is
  20 network passes per chunk.
