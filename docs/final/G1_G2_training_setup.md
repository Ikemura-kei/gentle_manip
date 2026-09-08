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

## 7b. What the runs actually produced (for calibrating any tuning decision)

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

## 9. How our setup compares to the literature (survey, 2026-09-08)

Our hyperparameters against published diffusion-policy work. **Nothing here has been changed** — this
is a survey to rank what is worth trying next.

**Match the regime before borrowing a number.** Ours is a **from-scratch, single-task-family train**
on 6,270 episodes / 1.19 M timesteps with a 2.89 M-parameter model, one robot, one workspace. Octo and
π0's headline numbers are **large-scale multi-embodiment PRE-training** — different regime, and their
batch/lr do not transfer. They are listed for scale context only; the comparable references are DP,
DP3 and Data-Scaling-Laws.

| | **ours (G1/G2)** | DP (Chi 2023) | DP3 (Ze 2024) | Data-Scaling-Laws (Lin 2024) |
|---|---|---|---|---|
| regime | from scratch, 1 task family | from scratch, per task | from scratch, per task | from scratch, per task |
| data | **6,270 eps / 1.19 M steps** | 50–200 demos/task | 10–100 demos/task | up to ~1,600 demos/task |
| params | **2.89 M** | ~250 M (CNN-UNet) | ~255 M | DINOv2 ViT-L + UNet |
| lr | 1e-4 cosine → 1e-5 | 1e-4 | 1e-4, 500-step warmup, cosine | 3e-4 denoiser / **3e-5 encoder** (fine-tuning a *pretrained* DINOv2) |
| batch | **128** | 64–256 | 128–2048 | 256 |
| optimizer | AdamW, wd 1e-6 | AdamW | AdamW | AdamW β 0.95/0.999 |
| steps | **1.0 M** | — | — | 5×10⁵ (largest set, 75 epochs) |
| predict / execute | **4 / 4** | 10–16 / **8** | 4 / 4 | 16 / 8 (temporal ensemble) |
| denoise steps | **20 / 20 (no DDIM)** | 100 train / 10 DDIM | 100 train / **2 DDIM** | 16 |

Scale context only — **pre-training** runs, not comparable to ours:

| | Octo (pre-train) | Octo (fine-tune) | π0 (pre-train) |
|---|---|---|---|
| data | **800 k trajectories** (Open X-Embodiment) | ~100 trajectories/domain | **~10,000 h**, 68 tasks, 7 robot configs (+9.1 % OXE/Bridge v2/DROID) |
| params | 93 M (Base) / 27 M (Small) | same | **3.3 B** (3 B PaliGemma + 300 M action expert) |
| steps | 300 k | 50 k | 700 k |
| batch | 2048 | not stated | **not stated in the paper** |
| lr | 3e-4, 2000 warmup, reciprocal-sqrt, wd 0.1, clip 1.0 | cosine decay + linear warmup | **not stated in the paper** |
| hardware | TPU v4-128, 14 h | 1× A5000, ~5 h | — |

⚠ The π0 paper states its step count and data scale but **not** batch size, learning rate or
optimizer; any such figures circulating for π0 come from secondary sources or reproductions, not the
paper. Do not cite them as π0's.

**Where we are conventional:** lr 1e-4 with cosine decay, AdamW, EMA and a warmup fraction are all
squarely standard; DP3 uses the identical lr and schedule shape, and batch 128 sits inside DP's and
DP3's ranges. Our 1 M gradient steps is *more* than Data-Scaling-Laws spends on its largest set
(5×10⁵). **Optimization is not where the headroom is** — and §7b's val plateau at epoch ~110 says the
same thing from our own data. Note our model is ~100× SMALLER than DP/DP3's (2.89 M vs ~250 M), which
makes capacity, not schedule, the interesting axis.

**Where we differ, in rough order of expected value:**

1. **Inference denoising: 20 full steps, no DDIM.** DP3 runs 2 DDIM steps at inference from a
   100-step training schedule; [HDP3](https://arxiv.org/html/2605.01581v4) proves why — robot
   trajectories are low-frequency-dominant (first two DCT modes carry 98.5 % of energy), so denoising
   error saturates almost immediately, ~0.25 % relative error at 2 steps, and reports 4.5 ms vs DP3's
   51.4 ms. We pay 20 network passes per chunk. **This is testable on the existing checkpoints with
   no retraining** — `use_ddim`/`ddim_steps` are already wired in `eval_diffusion_pointnet.yaml`
   (currently `use_ddim: False`). Directly relevant to real-robot control rate.
2. **Action horizon 4 predicted / 4 executed.** DP's ablation puts the optimum at **execute 8 from a
   10–16 prediction**, i.e. predict long, execute a fraction (receding horizon): a horizon > 1 gives
   temporally consistent actions, but executing the whole prediction costs reaction time. We are short
   on prediction AND execute the entire chunk — the one combination the DP ablation argues against.
3. **Capacity split, 94 % denoiser / 6 % encoder.** Two literature results cut in opposite directions
   and together are informative. [ScaleDP](https://arxiv.org/html/2409.14411v1) found naively
   deepening the denoiser *hurts* (80.1 % → 74.6 %, 8 → 14 layers, from gradient instability in
   observation fusion), and HDP3 argues the DP3-family denoiser is over-parameterized for
   low-frequency trajectories. So **do not widen the denoiser.** But our total is only 2.89 M — HDP3's
   own "right-sized" model is 2.52 M — so we are not in the bloated regime by absolute size; the
   question is only the *split*. Given the measured cloud-ablation result (+880 % val loss on a wrong
   cloud, and reliance *grew* with more steps), the encoder is load-bearing at ~175 k parameters, and
   widening it is the cheap direction.
4. **Global max-pool may be what caps size precision.** The encoder compresses 1024 points to one
   512-d vector by max-pooling; max-pool keeps the strongest activation per channel and is known to
   discard fine geometric detail. The cherry-tomato failure is exactly a fine-size read (closes to
   ~27 mm on a 25 mm object, unmoved by 2.6× more steps). Cheap probes, in increasing cost:
   (a) mean-pool ⊕ max-pool concat; (b) structured tokens / attention pool; (c) wider encoder —
   but see §10.3, which shows (c) is predicted NOT to help at 1024 points.
   ⚠ **The aux width head is NOT a candidate**: `aux_grasp_width_weight` is implemented and merely
   set to 0.0, but it was **already tried and did not help** (user, 2026-09-08). It predates the
   camera move (horizontal → diagonal), so the null is not strictly binding now, but a prior negative
   on the same architecture outweighs any inference from loss curves.
5. **Data composition beats data volume.** Lin et al. find generalization follows a power law in
   **environment and object diversity** and correlates only weakly with demonstration count
   (r −0.62…−0.79), saturating around **50 demos per environment-object pair**. We have 550 for each
   of the five largest objects — likely well past saturation — 100 for each of the 18 basic shapes,
   and **one environment**. This predicts that rebalancing toward more objects at ~50–100 each, and
   adding scene/workspace variation, buys more than more episodes of the same objects. It also makes
   the untouched 11:1 class imbalance (cherry is 8 % of data) worth a balanced-sampling test.
6. **Batch 128 and weight decay 1e-6.** Batch 128 is INSIDE the comparable range (DP 64–256, DP3 128–2048); the 2048 figure is Octo PRE-training on 800k trajectories and does not transfer. Weight decay 1e-6 is light next to
   Octo's 0.1. Both are plausible mild wins but neither addresses a plateaued val loss, and batch is
   constrained by the ~14 GB train tensor being GPU-resident. Low priority.

## 10. Large-scale POINT-CLOUD policy training specifically (survey, 2026-09-08)

The §9 references are mostly RGB. These are the point-cloud line our architecture descends from, and
they change two of §9's recommendations.

| | points | encoder | data | batch | steps |
|---|---|---|---|---|---|
| **ours** | **1024**, 1 view | MLP PointNet, from scratch, **LayerNorm**, global max-pool | 6,270 eps | **128** | 1.0 M |
| DP3 (2024) | 512–1024 | MLP PointNet, from scratch | 10–100 demos/task | 128–2048 | — |
| iDP3 (IROS 2025) | **4096** | **pyramid convolutional** encoder (replaces the MLP) | real humanoid, 2000+ eval trials | — | — |
| [FP3](https://arxiv.org/abs/2503.08950) (2025) | **4000 per view × 2** (3rd-person + wrist) | **pretrained Uni3D ViT, fine-tuned** (not frozen) | **60 k trajectories** (DROID, 86 tasks), 1.3 B params | **128** | 3 M, 8× A800, ~48 h |
| [R3D](https://arxiv.org/html/2604.15281v1) | 1024 / 8192 | ViT-tiny / ViT-small, **LayerNorm**, structured N×C tokens | RoboTwin | 256 | 1000 epochs |

**1. Batch 128 is settled — it is not a lever.** FP3 pre-trains a **1.3 B**-parameter policy on 60 k
trajectories at **batch 128**. Our 128 on a 2.89 M model is in good company; drop this from the list.

**2. "Lightweight PointNet is enough" is a BatchNorm ARTEFACT, and it is the belief our architecture
inherited.** DP3's "scaling paradox" — stronger encoders performing *worse* — was traced by R3D to
**BatchNorm degradation**, not an architectural ceiling. With LayerNorm, Uni3D encoders
**significantly outperform** PointNet (64.7 % vs 59.6 % on RoboTwin). The whole field trend agrees:
DP3 → iDP3 → FP3 moves from a from-scratch MLP to a pyramid conv encoder to a pretrained ViT.
**Good news for us: our encoder already uses LayerNorm, not BatchNorm**, so we never had the bug — but
we did inherit the conclusion it produced, and a stronger encoder is therefore a legitimate lever
rather than a refuted one.

**3. …but R3D gives a pairing rule that CONSTRAINS that lever, and it lands exactly on us.** Encoder
capacity must scale *with point density*: at **1024 points ViT-tiny is optimal (83.8 %)**, at 8192
points ViT-small is preferred, and **larger encoders UNDERPERFORM at a fixed point count**. We are at
1024 points. So **widening the encoder alone is predicted not to help** — this supersedes §9 lever 3's
"widening it is the cheap direction". More encoder capacity only pays if point count rises with it.

**4. ⚠ And point count is capped by the DATA, not the model.** Our demos store the cloud **already
farthest-point-sampled to 1024 points, and no depth images are retained** (verified: episode obs are
`ee_pos, ee_quat, gripper_width, point_cloud (T,1024,3), priv_*`). Going to 4096 points the way iDP3
did is therefore a **re-collection decision, not a config change** — the information is gone from the
stored dataset. Same for R3D's FPS-randomization augmentation, which needs a larger stored cloud to
resample from. Worth knowing before anyone plans an encoder-scaling experiment.

**4b. Measured: the padding is NOT dropout, and small objects are NOT point-starved.**
Two probes (`.agent_tmp/probe_padding.py`, `.agent_tmp/probe_object_points.py`):

- **Zero-padded frames are rare and phase-specific**: 61 of 119,716 val frames (0.051 %), in only
  **7 of 627 episodes**, and they occur *exclusively* at normalized phase 0.0–0.1 or 0.9–1.0 — never
  in the middle 80 %. **Dropout is ruled out twice over**: augmentation is applied in the loss at
  train time and never written to the stored dataset, and p = 0.03 dropout would in any case be
  uniform across frames rather than clustered at episode ends. The signature matches crop +
  `object_focus` leaving < `max_points` valid points when the arm is at home (start) or the object is
  lifted (end), so the pipeline pads to 1024. Negligible for training; it matters only as a masking
  requirement if pooling ever changes from max.
- **Points actually on the object**, within 3 cm of `priv_object_pos` at t = 0 (mean over 12 episodes):

  | cherry_tomato | strawberry | banana_chunk | tomato | mushroom | tofu |
  |---|---|---|---|---|---|
  | **124** | 218 | 212 | 200 (452 at 5 cm) | 303 | 347 |

  Cherry gets ~2.5× fewer points than mushroom/tofu — but **124 points on a 25 mm object is ample to
  resolve a 2–5 mm size difference in principle**. So "small objects are too sparsely sampled" is
  NOT the explanation for the width bias. That shifts the weight onto the representation (global
  max-pool bottleneck) and the objective/data balance (cherry is ~5 % of episodes and the loss is
  dominated by the rest, so a conditional model can hedge toward the population width).

**5. The one encoder change that IS available at 1024 points: stop collapsing to a global vector.**
R3D keeps **structured N×C tokens** rather than pooling to one global feature; we max-pool 1024 points
into a single 512-d vector. Max-pool keeps only the strongest activation per channel, which is exactly
the operation that would discard a few-millimetre size difference — and the cherry-tomato failure is a
few-millimetre size read that 2.6× more gradient steps did not fix. This needs **no extra points and
no re-collection**, so it is the highest-value encoder experiment available on the current dataset
**mean⊕max concat first, attention pooling second.** The asymmetry matters: concatenating mean to
max is representationally a SUPERSET of max (max survives in the vector, the network can ignore the
rest), so it risks only mild overfitting, whereas attention *replaces* max — it is a soft-argmax that
approximates max only in the low-temperature limit and must learn to, so it can genuinely regress.
**Attention pooling is not strictly better than max:** max is a parameter-free detector with a real
inductive bias (PointNet's universal-approximation result is proved with max), and most published
wins for attentive pooling are dense segmentation tasks rather than single-vector regression. Any
mean- or attention-based variant **must mask the padded points** (§10.4b) or it will pool over zeros,
which max is naturally immune to — skipping that would look like "attention didn't help".
Evidence that pooling is the right axis at all: [Equivariant vs. Invariant
Layers](https://arxiv.org/abs/2306.05553) (ICML 2024) finds complex pooling most helps SIMPLE
backbones (ours is a 3-layer MLP), that pooling choice can matter MORE than backbone width and depth
(converging with R3D's "don't widen at 1024 points"), and that pairwise pooling COMBINATIONS
significantly improve a fixed backbone — a direct endorsement of mean⊕max.

**6. Do not grow the denoiser — three independent sources now agree.** R3D uses a 4-block decoder
against ManiFlow's 12; HDP3 shows trajectories are low-frequency-dominant so heavy denoisers are
wasted; ScaleDP found naive deepening *hurts*. Our 94 % denoiser share should shrink or stay, not grow.

**7. Augmentation is load-bearing at scale.** R3D reports that without augmentation (FPS
randomization, dropout, Gaussian noise) learning curves fluctuate and success drops markedly — an
independent confirmation of what our own G2-vs-G1 result hints at, namely that the BC-path
augmentation is doing the robustness work. We have noise/dropout/offset; we lack FPS randomization
(blocked by item 4).

**One caution on the encoder-LR idea:** Lin et al. use a 10× *lower* lr for their vision encoder, but
theirs is a pretrained DINOv2 being fine-tuned. Ours is a small PointNet trained from scratch, where
the argument runs the other way — so their number is not transferable, and a separate encoder lr would
be a speculative experiment rather than a literature-backed setting.
