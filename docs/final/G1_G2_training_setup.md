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
| wall clock | **G1 `bmbrv` 8 h 41 m, G2 `ttukt` 7 h 30 m** (GH200, ~260 / ~237 s per epoch = ~31 / ~28 ms per gradient step; the spread is node-to-node variation, not a setting). Local twin `wiayg`: 5 h 38 m on a 4090 at ~21 ms/step — the 4090 is ~1.4× faster per step than a GH200 node here |

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

## 8. What the runs produced

Final epoch 120, all three cluster arms:

| run | train | val | `loss_paired` | `loss_consistency` | `loss_diffusion` |
|---|---|---|---|---|---|
| G1 `bmbrv` | 0.00109 | 0.00120 | 2.49e-5 | 5.74e-5 | 1.056e-3 |
| G2 `ttukt` | **0.00102** | **0.00115** | 1.58e-5 | **3.10e-5** (ablated) | 1.011e-3 |
| G0 `fdcjk` | 0.00110 | 0.00120 | **1.74e-3** (ablated) | 1.03e-4 | 1.068e-3 |

Two results bear directly on tuning:

- **The paired real–sim term is doing heavy lifting.** Un-optimized (G0) the real–sim feature distance
  ends **70× higher** than optimized (1.74e-3 vs 2.49e-5; 80× on val), and the gap *widened* over
  training — from 2.6× at epoch 11. The encoder does not merely fail to close the sim/real gap on its
  own, it drifts further apart while fitting sim. Do not weaken this term.
- **The consistency term may be counter-productive, but it is NOT yet established.** G2, which only
  observes the term, ends ~1.9× BETTER on the very quantity the objective minimizes (3.10e-5 vs
  5.74e-5) and has the best train, val and diffusion loss of the three. **Caveat that blocks the
  conclusion:** G1 and G0 differ only in the *paired* weight yet their consistency losses differ by
  1.8× — cross-run variation is the same size as the claimed effect. The clean test is `ttukt` vs
  `bmbrv`, both cluster, same seeds, at 200 episodes.

**Val plateaus from ~epoch 110 in all three runs** (G2: 0.00114 → 0.00116 → 0.00115 over epochs
110/115/120) with no overfit knee. So **1M gradient steps is at the point of diminishing returns for
this dataset — more steps is NOT the lever.** The local checkpoint sweep agrees from the success side:
epoch 80 already matched epoch 120 (31/40 both), though that comparison sits inside teaser noise.


## 9. Measurements on this dataset (ours, not literature)

Three probes that constrain the tuning discussion below. Scripts in `.agent_tmp/`.

**The encoder is load-bearing** (`probe_cloud_reliance.py`, §4): substituting another episode's cloud
raises val loss **+880 %**, and reliance *grew* with 2.6× more steps (+696 % → +880 %). So the 6 %-of-
parameters encoder is used hard.

**Zero-padded cloud frames are rare, phase-specific, and NOT dropout** (`probe_padding.py`): 61 of
119,716 val frames (0.051 %), in only **7 of 627 episodes**, occurring *exclusively* at normalized
episode phase 0.0–0.1 or 0.9–1.0 and never in the middle 80 %. Dropout is excluded twice over —
augmentation is applied in the loss at train time and never written to stored data, and p = 0.03
dropout would be uniform rather than clustered at episode ends. The signature matches crop +
`object_focus` leaving fewer than `max_points` valid points when the arm is at home (start) or the
object is lifted (end), so the pipeline pads to 1024. Harmless for training as it stands; it becomes a
**masking requirement** the moment pooling changes away from max (§10.6).

**Small objects are NOT point-starved** (`probe_object_points.py`) — points within 3 cm of
`priv_object_pos` at t = 0, mean over 12 episodes:

| cherry_tomato | strawberry | banana_chunk | tomato | mushroom | tofu |
|---|---|---|---|---|---|
| **124** | 218 | 212 | 200 (452 at 5 cm) | 303 | 347 |

Cherry gets ~2.5× fewer points than mushroom/tofu, but **124 points on a 25 mm object is ample to
resolve a 2–5 mm difference in principle**. This kills "too few points on small objects" as the
explanation for the width bias and shifts the weight onto representation (§10.6, pooling) and data
balance (§10.7).

## 10. Literature comparison and what to try next

A single consolidated survey (2026-09-08). **Nothing here has been changed in the runs** — this exists
to rank what is worth trying.

### 10.1 Match the regime before borrowing a number

Ours is a **from-scratch, single-task-family train**: 6,270 episodes / 1.19 M timesteps, 2.89 M
parameters, one robot, one workspace, one camera. Numbers from large multi-embodiment **pre-training**
do not transfer to it, so they are quarantined in 9.3.

### 10.2 Comparable regime (from-scratch, per-task)

| | **ours (G1/G2)** | DP (Chi 2023) | DP3 (Ze 2024) | Data-Scaling-Laws (Lin 2024) |
|---|---|---|---|---|
| data | **6,270 eps / 1.19 M steps** | 50–200 demos/task | 10–100 demos/task | up to ~1,600 demos/task |
| params | **2.89 M** | ~250 M (CNN-UNet) | ~255 M | DINOv2 ViT-L + UNet |
| lr | 1e-4 cosine → 1e-5 | 1e-4 | 1e-4, 500-step warmup, cosine | 3e-4 denoiser / 3e-5 encoder (fine-tuning a *pretrained* DINOv2) |
| batch | **128** | 64–256 | 128–2048 | 256 |
| optimizer | AdamW, wd 1e-6 | AdamW | AdamW | AdamW β 0.95/0.999 |
| steps | **1.0 M** | — | — | 5×10⁵ (largest set, 75 epochs) |
| predict / execute | **4 / 4** | 10–16 / **8** | 4 / 4 | 16 / 8 (temporal ensemble) |
| denoise steps | **20 / 20, no DDIM** | 100 train / 10 DDIM | 100 train / **2 DDIM** | 16 |

And the point-cloud line our architecture actually descends from:

| | points | encoder | data | batch | steps |
|---|---|---|---|---|---|
| **ours** | **1024**, 1 view | MLP PointNet, from scratch, **LayerNorm**, global max-pool | 6,270 eps | **128** | 1.0 M |
| DP3 | 512–1024 | MLP PointNet, from scratch | 10–100 demos/task | 128–2048 | — |
| iDP3 (IROS 2025) | **4096** | **pyramid convolutional** (replaces the MLP) | real humanoid | — | — |
| [FP3](https://arxiv.org/abs/2503.08950) (2025) | **4000/view × 2** | **pretrained Uni3D ViT, fine-tuned** | **60 k trajectories** (DROID, 86 tasks), 1.3 B params | **128** | 3 M, 8× A800, ~48 h |
| [R3D](https://arxiv.org/html/2604.15281v1) | 1024 / 8192 | ViT-tiny / ViT-small, **LayerNorm**, structured N×C tokens | RoboTwin | 256 | 1000 epochs |

### 10.3 Scale context only — PRE-training runs, not comparable

| | Octo (pre-train) | Octo (fine-tune) | π0 (pre-train) |
|---|---|---|---|
| data | **800 k trajectories** (OXE) | ~100 trajectories/domain | **~10,000 h**, 68 tasks, 7 robot configs (+9.1 % OXE/Bridge v2/DROID) |
| params | 93 M Base / 27 M Small | same | **3.3 B** (3 B PaliGemma + 300 M action expert) |
| steps | 300 k | 50 k | 700 k |
| batch | 2048 | not stated | **not stated in the paper** |
| lr | 3e-4, 2000 warmup, reciprocal-sqrt, wd 0.1, clip 1.0 | cosine + linear warmup | **not stated in the paper** |

⚠ The π0 paper gives its step count and data scale but **not** batch size, learning rate or optimizer.
Figures circulating for those come from secondary sources, not the paper — do not cite them as π0's.

### 10.4 Settled — these are NOT levers

- **Batch size.** 128 is inside the comparable range (DP 64–256, DP3 128–2048), and FP3 pre-trains a
  **1.3 B** model on 60 k trajectories at **batch 128**. The 2048 figure is Octo pre-training and does
  not transfer. Closed.
- **lr / schedule / optimizer.** lr 1e-4, cosine decay, AdamW, EMA, warmup fraction — DP3 uses the
  identical lr and schedule shape. Conventional.
- **More gradient steps.** 1.0 M already exceeds Data-Scaling-Laws' largest run (5×10⁵), and §8 shows
  val plateaus from ~epoch 110 in all three arms. Closed.
- **Widening the encoder on its own.** R3D's pairing rule: encoder capacity must scale *with point
  density* — at **1024 points ViT-tiny is optimal (83.8 %)**, at 8192 points ViT-small, and **larger
  encoders UNDERPERFORM at fixed point count**. We are at 1024, so extra encoder width alone is
  predicted not to help.
- **Growing the denoiser.** Three independent sources agree: R3D uses a 4-block decoder vs ManiFlow's
  12; [HDP3](https://arxiv.org/html/2605.01581v4) shows trajectories are low-frequency-dominant (first
  two DCT modes = 98.5 % of energy) so heavy denoisers are wasted; [ScaleDP](https://arxiv.org/html/2409.14411v1)
  measured naive deepening *hurting* (80.1 % → 74.6 %). Our 94 % denoiser share should shrink or hold.
- **The aux grasp-width head.** Implemented (`aux_grasp_width_weight`, plus an `aux_width_blind`
  variant) and set to 0.0 — but **already tried, and it did not help** (user, 2026-09-08). It predates
  the camera move (horizontal → diagonal) so the null is not strictly binding, but a prior negative on
  this architecture outweighs inference from loss curves.
- **A separate encoder learning rate.** Lin et al.'s 10×-lower encoder lr applies to fine-tuning a
  *pretrained* DINOv2; ours is a small PointNet from scratch, where the argument runs the other way.
  Not a literature-backed setting for us.

### 10.5 Lever 1 — inference denoising (cheapest, no retraining)

We run **20 full denoising passes per chunk**; DP3 runs **2 DDIM steps** from a 100-step training
schedule, and HDP3 explains why: denoising error saturates almost immediately on low-frequency
trajectories (~0.25 % relative error at 2 steps), reporting 4.5 ms vs DP3's 51.4 ms. `use_ddim` and
`ddim_steps` are **already wired** in `eval_diffusion_pointnet.yaml` (currently `use_ddim: False`), so
this is testable on the existing checkpoints with zero training. Directly relevant to real-robot
control rate.

### 10.6 Lever 2 — pooling (the one encoder change available at 1024 points)

**"Lightweight PointNet is enough" is a BatchNorm artefact, and it is the belief our architecture
inherited.** R3D traced DP3's scaling paradox — stronger encoders performing *worse* — to BatchNorm
degradation, not an architectural ceiling; with LayerNorm, Uni3D encoders **beat PointNet 64.7 % vs
59.6 %**. We already use LayerNorm, so we never had the bug, but we did inherit its conclusion.
Combined with 9.4 (don't widen at 1024 points), the actionable part is **how we pool, not how wide**:
R3D keeps structured N×C tokens where we max-pool 1024 points into a single 512-d vector.

**Order: mean⊕max concat first, attention pooling second.** Concatenating mean to max is
representationally a **superset** of max — max survives in the vector and the net can ignore the rest —
so it risks only mild overfitting. Attention *replaces* max: it is a soft-argmax that approximates max
only in the low-temperature limit and must learn to, so it can genuinely regress. **Attention pooling
is not strictly better than max**: max is a parameter-free detector with a real inductive bias
(PointNet's universal-approximation result is proved with max), and most published wins for attentive
pooling are dense segmentation rather than single-vector regression. Any mean- or attention-based
variant **must mask the padded points** (§9) or it pools over zeros — a failure that would masquerade
as "attention didn't help".

Supporting evidence that pooling is the right axis: [Equivariant vs. Invariant Layers](https://arxiv.org/abs/2306.05553)
(ICML 2024) finds complex pooling most helps **simple** backbones (ours is a 3-layer MLP), that pooling
choice can matter **more than backbone width and depth**, and that **pairwise pooling combinations**
significantly improve a fixed backbone — a direct endorsement of mean⊕max.

### 10.7 Lever 3 — data composition and balance

Lin et al. find generalization follows a power law in **environment and object diversity** and
correlates only weakly with demonstration count (r −0.62…−0.79), saturating near **50 demos per
environment-object pair**. We have 550 each for the five largest objects (well past saturation), 100
for each of 18 basic shapes, and **one environment**. This predicts more objects at ~50–100 each, plus
scene/workspace variation, beats more episodes of the same objects. It also makes the untouched
**11:1 class imbalance** worth a balanced-sampling test — cherry is ~5 % of episodes, so a conditional
model can hedge toward the population width and pay almost nothing in loss, which §9 leaves as one of
the two live explanations for the width bias.

### 10.8 Lever 4 — action horizon and replanning

We predict 4 and execute all 4. DP's ablation puts the optimum at **executing 8 from a 10–16
prediction** — predict long, execute a fraction. A horizon > 1 buys temporal consistency, but executing
the whole prediction costs reaction time; we are short on prediction *and* execute everything, the one
combination that ablation argues against.

### 10.9 Blocked by the data, not the model

**Point count is capped at 1024 by what was stored.** The demos keep the cloud already
farthest-point-sampled to 1024 with **no depth images retained** (verified: episode obs are `ee_pos`,
`ee_quat`, `gripper_width`, `point_cloud (T,1024,3)`, `priv_*`). Going to 4096 points as iDP3 did is a
**re-collection decision, not a config change**. The same blocks R3D's FPS-randomization augmentation,
which needs a larger stored cloud to resample from — worth knowing before anyone plans an
encoder-scaling experiment, since 9.4 says extra encoder width only pays if point count rises with it.
