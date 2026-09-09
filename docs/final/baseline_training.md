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

## 1. Point cloud, G5 recipe — `jfwqs` (2026-09-09)

The real twin of the sim generalist's best configuration. Against the previous real-only baseline
`jiupy`, **only two things change** — encoder pooling and action horizon — so any difference is
attributable to them.

| | `jiupy` (previous baseline) | **`jfwqs`** |
|---|---|---|
| cloud pooling | max | **concat(max, masked mean)** |
| action horizon | 4 | **16** (execute 4 at deploy) |
| everything else | — | identical |

**Why these two.** In sim they are the only changes that produced a replicated gain: on cherry
tomato, G5 (pooling + horizon 16) with temporal ensembling scored 10/20 and 9/20 against a
baseline's 3/20, at the lowest sustained stress of any horizon-16 run, and matched the best
baseline on mushroom (18/20) while being gentler. Pooling and ensembling INTERACT — alone they give
6/20 and 4.5/20, together 9.5/20. See `docs/final/results.md`.

**Architecture.** PointNet encoder with `pooling: meanmax`, `use_layernorm: True`,
`final_norm: layernorm`, visual feature dim 512; ResidualMLP denoiser `[1024, 1024, 1024]`,
`time_dim 16`. **3,194,016 parameters** — the meanmax shape, +131,072 over max-pooling
(`Linear(256→512)` becomes `Linear(512→512)`). Verify this number when loading a checkpoint: a
mismatch means the pooling override did not apply.

**Diffusion.** DDPM, 20 denoising steps, `predict_epsilon: True`, 2 conditioning steps, 1
point-cloud conditioning step. Do NOT raise the step count at inference — the beta schedule is
built for the trained value, so 20 is a ceiling; DDIM may subsample below it, but measured in sim
that costs success (DDIM-10 halved it, DDIM-20 lost ~2.5 episodes).

**Optimization.** 2000 epochs, batch 128, AdamW lr 1e-4 with one cosine cycle to min_lr 1e-5,
warmup 100 epochs, EMA from epoch 10, checkpoint every 100, validate every 10, seed 42.

**Reproduce.**
```bash
DATASET=single_lift_real7_bc_v1 EPOCHS=2000 \
  bash gentle_manip/scripts/final/train_dppo_dp3_real.sh \
    horizon_steps=16 +model.network.pointnet.pooling=meanmax
```
`+` is required for pooling: the training config has no such key (unlike the eval config, which now
reads it from the checkpoint). Without the `+` hydra errors; without the override entirely the run
silently trains max-pooling, which the parameter count above will catch.

**Result (on `single_lift_real6_bc_v1`, 110 episodes — the 130-episode rerun is the finalised one).**
2000/2000 epochs in ~1 h 30 m at 2.1 s/epoch. Train loss 0.00865, val 0.03759. **Val minimum was
epoch 390 at 0.02205**, after which val rose ~70 % while train kept falling — expected, since 99
demos × 2000 epochs means each sample is seen ~2000 times.

⚠ **Do not pick the checkpoint by val loss alone.** It has already misled in this project: on the
sim run G4, the val-minimum checkpoint scored 15/20 while one with 12 % worse val scored 18/20, and
`jiupy` itself deploys `state_1500`, not its val minimum. With no sim eval available for a real-only
policy, the options are robot trials across several checkpoints or an informed guess. The deploy
entry starts at `state_400` and the box comment says to sweep 800 / 1500 / 2000 if it underperforms.

**Deploy.** `gentle_manip/scripts/final/deploy_dppo.sh`, the `jfwqs` block — exec 4,
`--temporal-ensemble --ensemble-m 0.01`, `--smooth-alpha 0.6`, `--max-pos-step-m 0.0065`. This is
the first real-only policy that can ensemble at all: ensembling needs horizon > act-steps, so it is
inert on every horizon-4 entry. `m` was 0.33 on the robot in one session (the ACT-proportional value
for a 4-deep window) and clearly better qualitatively, while sim preferred 0.01 (9.5/20 vs 6/20) —
sim scores success and ignores smoothness, so both can hold. State which value produced any number
you report.

---

## 2. RGB baseline — TBD

To be filled when the matched RGB arm is trained. Open decisions recorded 2026-09-09:
horizon 16 to match §1 (the existing RGB runs `xkhrc`/`fuuoy` are horizon 4, so a modality
comparison against them confounds modality with horizon); colour jitter, which DPPO does not ship
(`RandomShiftsAug` is its only image augmentation) and which is the actual answer to lighting drift;
and an ImageNet-pretrained ResNet-18 with GroupNorm instead of DPPO's from-scratch depth-1 ViT,
which would also want a 224×224 re-convert — the raw frames are 480×640, so the data supports it.

## 3. pi0.5 — TBD
