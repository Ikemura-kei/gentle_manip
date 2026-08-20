# Cluster experiment: auxiliary-objective ablation (contact / object-pos) on the soft armfocus BC student

**Goal:** do privileged **auxiliary training objectives** sharpen the point-cloud diffusion policy's
representation and improve the lift? Four BC-pretrain runs, IDENTICAL except the aux objective, all on
ONE shared 300-demo soft dataset, all evaluated on the same checkpoint sweep:

| # | variant | contact head + BCE | object-pos head + MSE |
|---|---------|:---:|:---:|
| 1 | **baseline** (no aux) | — | — |
| 2 | **+contact** | ✓ | — |
| 3 | **+objpos** | — | ✓ |
| 4 | **+both** | ✓ | ✓ |

The aux heads read the noise-INDEPENDENT conditioning feature `[pointnet_feat ⊕ proprio]` and add a
loss (total = diffusion + λc·BCE(contact) + λp·MSE(object_pos)). **Training-only** — the deployed
policy samples via `forward()` only, so aux adds ZERO inference/deploy cost, and ONE eval config
evaluates all four variants (checkpoint load is `strict=False`; extra head weights are ignored).

> **UPDATE (2026-08-20):** the `strict=False` claim above was true for `DiffusionModel`'s own
> loader (training) but NOT for eval — `DiffusionEval` (`third_party/dppo/model/diffusion/
> diffusion_eval.py`) still used `strict=True` and crashed on every +contact/+objpos/+both
> checkpoint (`Unexpected key(s) in state_dict: "pos_head.0.weight", ...`). Fixed (submodule
> commit `f390f2f`, bumped in the main repo) with a relaxed loader that ignores extra keys but
> still raises on any genuinely MISSING key. Verified against a real crashed eval (ufwhv/state_200)
> before resubmitting the 3 affected checkpoints (contact/objpos/both @ state_200).

## Baseline & what's held constant
- Backbone = **bwvei** (`single_lift_mushroom_soft_abs_pcd_rot6d`, 0.81 @ ep400): soft MPM, abs action
  (action_dim 10), PointNet diffusion, horizon 4 / cond 2 / pc_cond 1, batch 128, lr 1e-4, cosine
  (warmup 100, min_lr 1e-5), 1024-pt cloud. **The only arch delta vs bwvei: quat proprio (obs_dim 8)**
  instead of ee_rot6d (obs_dim 10) — per request.
- **Aux dataset:** 300 soft demos, FEM (v3) gentleness synthesis + **2.5 mm** extra squeeze, **arm-focus**
  point cloud (`superset_soft_armfocus`), `scene_dr_every 1`. Records the two privileged AUX LABELS:
  `priv_contact` (proper binary gripper-object contact — geometric finger↔particle, NOT a stress proxy)
  and `priv_object_pos` (object COM). `convert_demos` auto-detects and stores both.
- Experiment config: **`single_lift_mushroom_soft_abs_action_armfocus`** (committed).
- **Epochs = 1000, save_model_freq = 100.** 300 demos is HALF of bwvei's 600, so an epoch is ~half the
  gradient updates → the peak shifts ~2× later in epoch-count (bwvei's ep400 peak ≈ our ep800). 1000
  epochs gives the peak room; the cosine cycle completes at 1000.
- Aux loss weights **λc = λp = 1.0** (set in the config / overrides below; tunable).

## Prerequisites
- `git pull` on master first (uses the committed aux code + configs).
- Do **NOT** export `DPPO_LOG_DIR`/`DPPO_DATA_DIR` (launcher defaults to `logs/dppo`, `dataset/dppo`);
  if you must, `DPPO_LOG_DIR` MUST end in `logs/dppo`.
- **wandb ONLINE**, all four under the SAME project `gentle-manip-single_lift_mushroom_soft_abs_pcd`
  (the env name has no `/`, so the project name is valid — no `wandb=null` needed). The wandb run name
  is the auto-minted 5-letter `exp_id`; **record which exp_id is which variant** (printed at launch and
  in `experiments.csv`) so the four runs are identifiable in wandb.
- Eval needs the sim server with **`--subprocess`** (`soft_orientation` DR carries shape/scale DR →
  the harness rebuilds geometry every `scene_group_size=4` batches).
- Prefix sim/collect commands with `env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl`.

---

> ⚠️ **This dataset is FLAWED (contact label) — a FRESH one is coming, but run the primary experiments NOW.**
> The `priv_contact` label here is a **geometric heuristic** (finger-link↔nearest-particle distance < 0.05 m),
> NOT the true physics, and it has a false-negative failure mode. The 22 episodes where it clearly failed
> (contact never/barely fired despite a successful lift) are **filtered out**, so what remains is clean
> (every kept episode holds contact ≥36% of its steps, single 0→1 transition) — good enough to run the
> **primary ablation now** on this 278-ep dataset. In parallel a proper **MPM→gripper coupling-force**
> contact is being implemented and the dataset **re-collected**; when that lands, **re-run the +contact /
> +both variants** on the fresh dataset (baseline & +objpos are unaffected — they don't use the contact
> label, so their results carry over). Do NOT block on the fresh dataset — do the primary runs with this one.

## STEP 0 — Dataset (already collected + FILTERED + CONVERTED on the dev box)
- Filtered demos (278 eps, the 22 contact-false-negatives removed):
  `dataset/demos/single_lift_mushroom_soft/26-08-19-isl-filt/data.pkl` (source: `…/26-08-19-isl`, 300 eps,
  92.9% success, FEM + 2.5 mm + arm-focus).
- **Already converted** to the training npz: `dataset/dppo/single_lift_mushroom_soft_abs_pcd/{train,val,normalization}.npz`
  (obs_dim 8 quat, 250 train / 28 val trajs, aux_contact + aux_object_pos). **Sync this `dataset/dppo/…`
  dir to the cluster and skip STEP 1** — all four runs read from it (shared env name).

## STEP 1 — (already done; only if re-converting from the filtered pkl)
```bash
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
  dataset/demos/single_lift_mushroom_soft/26-08-19-isl-filt/data.pkl \
  --out $DPPO_DATA_DIR/single_lift_mushroom_soft_abs_pcd \
  --experiment single_lift_mushroom_soft_abs_action_armfocus --view student --point-cloud
# verify meta: obs_dim 8, n_episodes 278, aux_contact True, aux_object_pos True.
```

## STEP 2 — Train the four variants (1000 epochs, save/100, wandb online, ONE config + overrides)
Same config for all four; the variant is chosen entirely by the model/network overrides (keys already
exist in the config — plain value overrides, no `+`). n_epochs/save_freq/wandb are already set in the
config, so **baseline needs NO overrides**.
```bash
CFG="--config-path gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd --config-name pre_diffusion_pointnet"
TRAIN="env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train"

# 1) baseline (no aux)
$TRAIN $CFG
# 2) +contact
$TRAIN $CFG model.network.aux_contact=true model.aux_contact_weight=1.0
# 3) +objpos
$TRAIN $CFG model.network.aux_object_pos=true model.aux_object_pos_weight=1.0
# 4) +both
$TRAIN $CFG model.network.aux_contact=true model.aux_contact_weight=1.0 \
            model.network.aux_object_pos=true model.aux_object_pos_weight=1.0
```
Each → `logs/dppo/dppo-pretrain/single_lift_mushroom_soft_abs_pcd/<exp_id>/checkpoint/state_{100,200,…,1000}.pt`
plus wandb (loss curves + `aux - loss_contact` / `aux - loss_object_pos` diagnostics). Note the four
`<exp_id>`s and their variant. Schedule the four however GPU allows (4 concurrent fits in ~24 GB; else
stagger). Each run is independent — separate `exp_id`, logdir, wandb run, but the same project + dataset.

## STEP 3 — Eval sweep, EACH CHECKPOINT AS SOON AS IT APPEARS
Evaluate **state_200, 300, 400, 500, 600, 800, 1000** (final) for EVERY variant — do NOT wait for a run
to finish; kick off each eval the moment its checkpoint is written (monitor `checkpoint/` per run). Note
`state_700 / state_900` are NOT saved (save/100 gives them, but the sweep skips them to bound cost — add
if a curve is still moving). All evals go through the shared canonical harness (200 episodes, 5 envs,
deterministic per-batch DR) → `summary.json` + `episodes.csv` + per-episode videos in
`<run>/eval/<datetime>/`.

Sim server — ONE per concurrently-evaluated run, reused across that run's checkpoints (the experiment
MUST be the arm-focus one so the eval cloud is sampled the SAME way as training):
```bash
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  -m gentle_manip.scripts.serl_sim_server --experiment single_lift_mushroom_soft_abs_action_armfocus \
  --view student --num-envs 5 --render-rgb --subprocess --port <P>          # wait for SIM_SERVER_READY
```
Eval agent, per checkpoint (ONE eval config works for every variant — no aux flags needed here):
```bash
env -u PYTHONPATH -u ROS_DISTRO uv run --project envs/dppo python -m gentle_manip.dppo.train \
  --config-path gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd --config-name eval_diffusion_pointnet \
  base_policy_path=logs/dppo/dppo-pretrain/single_lift_mushroom_soft_abs_pcd/<exp_id>/checkpoint/state_<N>.pt \
  env.specific.port=<P>
# experiment + normalization_path are already the armfocus dataset in the eval config.
```

## STEP 4 — Read out
Per variant, plot success_rate (+ ever_success, stress) vs epoch across the sweep; compare the four
curves at matched epochs. The question: does +contact / +objpos / +both beat baseline, and where does
each peak. Record peak success + epoch per variant; keep the per-episode videos for the best of each.

---

## Appendix — re-collect the dataset (if not syncing the pkl)
```bash
env -u PYTHONPATH -u ROS_DISTRO MUJOCO_GL=egl uv run --project envs/sim python \
  grasp_synthesis/collect_demos_synth_v3.py --experiment single_lift_mushroom_soft_abs_action_armfocus \
  --n-episodes 300 --n-envs 8 --maxfevals 1145 --grasp-gpu --seed 0 --scene-dr-every 1 \
  --grasp-extra-close 0.0025 --record-video 3
# soft FEM gentleness synthesis + 2.5 mm extra squeeze + arm-focus cloud; records priv_contact +
# priv_object_pos. -> dataset/demos/single_lift_mushroom_soft/<datetime>/
```

## Notes / rationale
- **Why quat (obs_dim 8):** per request — the only backbone change vs bwvei.
- **Contact label is PROPER, not a stress proxy:** geometric finger-link↔nearest-particle distance <
  0.05 m (both fingers) — a soft body stresses under gravity with nothing touching it, so stress ≠
  contact. Verified as a clean single 0→1 transition at the grasp moment.
- **Object-pos label** is the object COM, normalized to [-1,1] (norm stats in `normalization.npz`), so
  its MSE is balanced against the diffusion loss.
- **Aux weights 1.0** are a starting point; if an aux loss dominates or the diffusion loss degrades,
  drop the weight (e.g. 0.1–0.5) — it's a single override. Watch `aux - loss_*` in wandb.
- **Baseline ≡ bwvei arch (quat):** the baseline uses `AuxDiffusionModel` with both weights 0 and no
  heads → behaviorally identical to the plain `DiffusionModel` pipeline (verified).
