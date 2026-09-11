# G7 / G8 — sim + real co-training (cluster)

**G7 = G5 sim + real.  G8 = G6 sim + real.**  Same recipe, same real half, only the sim half
differs — so G7-vs-G8 isolates the value of G6's extra sim data under co-training, and each
against its sim-only parent (`fsynt` for G5, `vnhnr` for G6) isolates the value of the real data.

Written 2026-09-11. The local box is busy training G6 to epoch 100, so G7/G8 run on the cluster.

---

## 1. What is transferred, and what is already there

| dataset | where | size | note |
|---|---|---|---|
| `single_lift_generalist_soft_v6_tail22` | **transferred** | 19.7 GiB | G6, 9,125 train / 1,014 val |
| `single_lift_real7_bc_v1_tail` | **transferred** | 0.33 GiB | real, 117 train / 13 val, **already hold-tailed** |
| `single_lift_generalist_soft_v5_tail22` | already on cluster | 13.0 GiB | G5, 5,643 train / 627 val |

Landing directory: `…/gentle_manip/dataset/transfer/`.
Move both into `$DPPO_DATA_DIR` (`<repo>/dataset/dppo/`) before use — the trainer resolves
`train_dataset_path = ${DPPO_DATA_DIR}/${env}/train.npz`, so **the dataset directory name IS the
`env=` value**. No env yaml is needed.

### The real set is already tailed — do not tail it again

The sim sets carry a trailing hold (24 identical final action frames; the convert's own ~11 plus
`augment_hold_tail 12`). The raw real teleop set carries **1**. Merging them untailed would give the
policy two different episode endings and the hold behaviour would be learned from sim only. It was
therefore tailed locally with `augment_hold_tail … 23`, which brings it to 24, matching both sim
sets exactly. Verified before transfer.

---

## 2. Data prep on the cluster

Two merges, nothing else. `merge_npz_datasets` works at the npz level — de-normalises each source
with its own stats, concatenates, re-normalises jointly — so it needs no sim, no GPU and no
conversion budget.

```bash
# G7 = G5 + real
uv run --project envs/dppo python -m gentle_manip.dppo.merge_npz_datasets \
    dataset/dppo/single_lift_generalist_soft_v5_tail22 \
    dataset/dppo/single_lift_real7_bc_v1_tail \
    --out dataset/dppo/single_lift_generalist_simreal_g7

# G8 = G6 + real
uv run --project envs/dppo python -m gentle_manip.dppo.merge_npz_datasets \
    dataset/dppo/single_lift_generalist_soft_v6_tail22 \
    dataset/dppo/single_lift_real7_bc_v1_tail \
    --out dataset/dppo/single_lift_generalist_simreal_g8
```

**Expect this warning, and ignore it:**
`WARNING train: dropping non-common arrays ['aux_contact', 'aux_object_pos']` — the sim sets carry
aux labels the real rows cannot. The recipe has `aux_contact_weight: 0.0`,
`aux_object_pos_weight: 0.0`, `aux_grasp_width_weight: 0.0`, so nothing reads them.

**Use the merged `normalization.npz`** for training, eval and deploy. It holds joint min/max and
differs from either parent's.

### Expected after merge

| | train eps | train steps | val eps |
|---|---|---|---|
| G7 | 5,760 | ~1,173,800 | 640 |
| G8 | 9,242 | ~1,890,300 | 1,027 |

Real is 2.0 % of G7 and 1.3 % of G8 by episode. Plain concatenation, no oversampling — the
validated "noos" recipe.

---

## 3. Training

Recipe is `fsynt`'s verbatim (and G6 `vnhnr`'s): horizon 16, mean(+)max PointNet pooling, paired
consistency 0.5, consistency 1e-8 @ frac 0.3, `d435i_noise_train` + pc_offset, seed 42, EMA from
epoch 1, warmup 5, save/20, val/5. **Only `env=` and `train.n_epochs` change between runs.**

```bash
uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path <repo>/gentle_manip/dppo/cfg/sim2real_v1 --config-name pre_diffusion_pointnet \
  env=single_lift_generalist_simreal_g8 \
  experiment=single_lift_mushroom_soft_abs_action_armfocus_7d_realws \
  train.n_epochs=98 \
  model.paired_consistency_weight=0.5 \
  model.pc_aug=d435i_noise_train \
  model.pc_offset=[0.005,0.0035,0.0015] \
  model.consistency_weight=1e-8 \
  model.consistency_frac=0.3 \
  model.consistency_aug=d435i_noise_strong \
  model.consistency_offset=0 \
  seed=42 \
  wandb.project=gentle_manip_generalist \
  horizon_steps=16 \
  +model.network.pointnet.pooling=meanmax \
  train.save_model_freq=20 \
  train.val_freq=5 \
  train.lr_scheduler.warmup_steps=5 \
  train.epoch_start_ema=1
```

For G7: `env=single_lift_generalist_simreal_g7 train.n_epochs=158`.

### Epochs — match the gradient-step budget, not the epoch count

An epoch is a pass over CHUNKS, not frames, so equal epochs across differently sized datasets are
NOT equal training. G6 (`vnhnr`) used **1,343,900 steps**; match it.

```
chunks  ≈ train_steps − (horizon − 1) × train_episodes        (horizon 16)
batches/epoch = chunks / 128
epochs = 1,343,900 / batches_per_epoch
```

| | chunks (est.) | batches/epoch | epochs for ~1.34 M steps |
|---|---|---|---|
| G7 | ~1,087,400 | ~8,495 | **158** |
| G8 | ~1,751,700 | ~13,685 | **98** |

This estimate was within 0.1 % on G6 (predicted 1,721,620 chunks, actual 1,720,157). **Confirm from
the loader's own reported batches/epoch at startup and adjust `n_epochs` if it differs by more than
a per cent.**

### Wall clock

`fsynt` ran at **32.0 steps/s** on the cluster, so ~1.34 M steps ≈ **11.7 h** per run. Two runs
sequentially ≈ 23 h; in parallel they need separate GPUs (the cloud bank alone is ~21 GiB for G8).

---

## 4. Verify before leaving it running

| check | expected |
|---|---|
| network parameters at startup | **3,194,016** (same arch as G5/G6; a mismatch means the pooling override was lost) |
| train / val episodes | G7 5,760 / 640 · G8 9,242 / 1,027 |
| batches/epoch | within ~1 % of the table above |
| merged normalization | used by training AND by any later eval/deploy |
| first epochs | train loss ~0.2 at epoch 1 falling to ~0.01 by epoch 10 (G6's trajectory) |

Each run needs its own `EXPERIMENT.md` (motivation, hypothesis, git commit) and a row in
`experiments.csv` — the DPPO hydra callback writes the row, the `EXPERIMENT.md` is manual.

---

## 5. Caveats worth knowing before reading the results

- **The real half is tiny** — 117 episodes against 5,643 (G7) or 9,125 (G8). With plain concat it is
  ~1–2 % of batches, so expect a small effect either way. That is the validated recipe, not an
  oversight.
- **The real set covers 7 objects** (cherry_tomato, cube, grape, mushroom, padron_pepper, tofu,
  tomato) — all small and compact. It adds nothing at large grasp yaw, which is G6's known gap
  (see `G6_training_setup.md`): 95.4 % of sim demos sit within ±60° yaw and the policy commands
  almost no rotation when ~85° is required. Co-training will not fix that.
- **G7 vs G8 is the clean comparison** (real held constant, sim varied). G8 vs `vnhnr` is the other
  clean one (sim held constant, real added). Comparing G7 to `vnhnr` varies both.
