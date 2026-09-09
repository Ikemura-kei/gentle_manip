# Real-world baselines — training setups

One page per baseline trained on REAL teleop demonstrations only (no sim, no co-training). Each
entry records what was run, how to reproduce it, and what was deliberately left out, so that
baselines stay comparable to each other and to the point-cloud main line.

**Shared across every entry below.** Dataset `single_lift_real7_bc_v1` (point cloud) or
`single_lift_real7_bc_rgb_v1` (RGB) — the same 130 teleop episodes over 7 objects, 117 train /
13 val, 29,120 train steps. Actions are 7-dim absolute pose+euler+gripper
(`abs_pose_euler_abs_gripper_z15.yaml`, `euler_frame_offset_deg [180,0,0]` to avoid the ±π seam).
Proprio is the 8-dim `[ee_pos, ee_quat, gripper_width]`. 30 Hz.

**Real-only means three terms are OFF and must stay off.** `paired_consistency_weight`,
`consistency_weight` and `pc_aug` all exist to bridge sim→real; with no sim half they are
meaningless, so every real-only run sets them to 0 / "".

**The hold-tail decision, which applies to every horizon-16 real run.** Real teleop demos carry
essentially no trailing hold — the run of near-identical final commands is 1.1 frames on average,
3 at most — so there are ZERO all-hold chunks at horizon 16, and also zero at horizon 4. The sim
recipe solved this with a 22-frame tail (rule: `tail − horizon = 6`, holding the all-hold share at
3.7 %). We deliberately do NOT apply it here (user, 2026-09-09): matching `jiupy` keeps the
comparison clean. Consequence: both horizon-16 real policies train with no stop supervision. This
is a shared limitation, not a difference between them.

---

## 1. Point cloud, G5 recipe — PLANNED

The real twin of the sim generalist's best configuration: **mean(+)max cloud pooling + action
horizon 16** (executing 4 at deploy), trained real-only on the full 7-object mix.

**Why these two changes.** In sim they are the only pair that produced a replicated gain. On cherry
tomato, pooling + horizon 16 with temporal ensembling scored 10/20 and 9/20 against a baseline's
3/20, at the lowest sustained stress of any horizon-16 run, and it matched the best baseline on
mushroom (18/20) while being gentler. The two INTERACT rather than add — alone they give 6/20 and
4.5/20, together 9.5/20. Full table in `docs/final/results.md`.

**Architecture.** PointNet encoder with `pooling: meanmax`, `use_layernorm: True`,
`final_norm: layernorm`, visual feature dim 512; ResidualMLP denoiser `[1024, 1024, 1024]`,
`time_dim 16`. **3,194,016 parameters** — the meanmax shape, +131,072 over max pooling
(`Linear(256→512)` becomes `Linear(512→512)`). Check this number after loading any checkpoint: a
mismatch means the pooling override did not apply and the run is max-pooling.

**Diffusion.** DDPM, 20 denoising steps, `predict_epsilon: True`, 2 conditioning steps, 1
point-cloud conditioning step. Do NOT raise the step count at inference — the beta schedule is
built for the trained value, so 20 is a ceiling. DDIM may subsample below it, but measured in sim
that costs success (DDIM-10 halved it; DDIM-20 lost ~2.5 episodes).

**Optimization.** **2000 epochs (fixed)**, batch 128, AdamW lr 1e-4 with one cosine cycle to
min_lr 1e-5, warmup 100 epochs, EMA from epoch 10, checkpoint every 100, validate every 10, seed 42.
Epochs are held at 2000 rather than matching gradient steps, so the 130-episode set gets ~18 % more
updates than a 110-episode run at the same epoch count. That is the deliberate choice: the schedule
(warmup, cosine cycle, EMA start, checkpoint cadence) stays identical across runs.

**Launch.**
```bash
DATASET=single_lift_real7_bc_v1 EPOCHS=2000 \
  bash gentle_manip/scripts/final/train_dppo_dp3_real.sh \
    horizon_steps=16 +model.network.pointnet.pooling=meanmax
```
The `+` on pooling is required: the training config has no such key (unlike the eval config, which
now reads it from the checkpoint). Without `+` hydra errors; without the override entirely the run
silently trains max pooling, which the parameter count above will catch.

**Expected shape of the run.** ~2.1 s/epoch, so roughly 1 h 30 m. Expect the val minimum EARLY —
around a fifth of the way in — and then a long rise while train loss keeps falling: 117 demos ×
2000 epochs means each sample is seen ~2000 times.

⚠ **Do not pick the checkpoint by val loss alone.** It has already misled in this project: on the
sim run G4, the val-minimum checkpoint scored 15/20 while a checkpoint with 12 % worse val scored
18/20. With no sim eval available for a real-only policy, the options are robot trials across
several checkpoints or an informed guess — start at the val minimum and sweep later ones if it
underperforms.

**Deploy.** Add a block to `gentle_manip/scripts/final/deploy_dppo.sh`: exec 4,
`--temporal-ensemble --ensemble-m <see below>`, `--smooth-alpha 0.6`, `--max-pos-step-m 0.0065`,
and the matching `normalization.npz`. Horizon 16 is what makes ensembling possible at all — it
needs horizon > act-steps, so it is inert on any horizon-4 policy. On `m`: sim prefers 0.01
(9.5/20 vs 6/20 at 0.33), while 0.33 — the ACT-proportional value for a 4-deep window — was clearly
better in one robot session. Sim scores success and ignores smoothness, which is what heavier
weighting buys, so both can hold. State which value produced any number you report.

## 2. RGB baseline — TBD

To be filled when the matched RGB arm is trained. Open decisions recorded 2026-09-09:
horizon 16 to match §1 (the existing RGB runs `xkhrc`/`fuuoy` are horizon 4, so a modality
comparison against them confounds modality with horizon); colour jitter, which DPPO does not ship
(`RandomShiftsAug` is its only image augmentation) and which is the actual answer to lighting drift;
and an ImageNet-pretrained ResNet-18 with GroupNorm instead of DPPO's from-scratch depth-1 ViT,
which would also want a 224×224 re-convert — the raw frames are 480×640, so the data supports it.

## 3. pi0.5 — TBD
