# Action-Space Ablation & Follow-ups — Final Report (2026-08-20 → 2026-08-23)

Consolidated results of the cluster campaign started by
`docs/cluster_experiment_action_space_ablation.md`, plus the two follow-up rounds it
triggered. Companion deep-dives: `docs/debug_partC_euler_action_anomaly.md` (the two
derivation bugs) and `docs/hwo_dataset_investigation.md` (collection-recipe forensics).
All sim numbers are the canonical eval harness (200 episodes, 5 envs, fixed DR sequence,
per-episode video); "peak" = best checkpoint of the 6-point sweep (state_100..600).

---

## 1. Headline conclusions

1. **7d euler-absolute ≈ 10d rot6d-absolute.** On identical demos and supervision,
   `eibno` (7d) peaks at 0.84 vs `jfhlu` (10d) 0.88 — the compact rotation encoding costs
   ~0.04 (single-seed noise range). The 7d action is the standard going forward.
2. **Absolute ≈ delta in sim** (0.62-0.785 vs 0.625-0.745 on matched data). Absolute is
   preferred for deployment: no accumulation drift, re-anchors every step.
3. **Getting absolute BC to work required fixing two derivation bugs** (§3): the euler ±π
   seam and a closed-loop fixed-point stall — and ultimately deriving actions from the
   **recorded commanded targets**, not the achieved pose trajectory.
4. **The hwo dataset's edge was its collection recipe, not its cloud and not luck** (§4):
   v3 collector with `grasp_extra_close=5mm` + soft-firm phase. Arm-focus cloud costs
   ≈ 0 in sim once the recipe is equalized.
5. **The real-workspace spawn box works** (§5): 0.71 peak on a 4.6×-larger region.
6. **Co-training with the 50 real demos is free in sim at ~8% real fraction** (§6);
   ×4 oversampling costs sim success and adds seed variance.

---

## 2. Part A / B / C ablation results

### Part C — rotation encoding (hwo demos, hwo cloud, commanded supervision)
| run | action | curve (100→600) | peak |
|---|---|---|---|
| `jfhlu` (reference) | 10d rot6d | 0.75 / 0.88 / 0.785 / 0.855 / 0.815 / 0.84 | **0.88 @200** |
| `eibno` | **7d euler** | 0.84 / 0.84 / 0.68 / 0.705 / 0.435 / 0.39 | **0.84 @100-200** |

### Part B — abs vs delta (xhk arm-focus collection; NOTE: v2 recipe, see §4)
| run | action | peak |
|---|---|---|
| `hrqdm` | delta s43 | 0.745 @100 (collapses → 0 by 400) |
| `wicfr` | 7d abs s43 (commanded) | 0.70 @100 |
| `uzgjm` | delta s42 | 0.625 @100 (collapses → 0 by 300) |
| `igjmd` | 7d abs s42 (commanded) | 0.62 @300 |

Delta arms degrade sharply with epochs; commanded-abs arms are stable (see §3.3).

### Part A — real 55-demo set (no sim eval; judge on the robot)
| run | algo / action | checkpoints |
|---|---|---|
| `qjzsf` | DPPO 7d abs (commanded+K4) | state_500..6000; **val-loss minimum ≈ state_500-1000** |
| DPPO delta | (first-round run, valid) | state_500..6000 |
| `...realablcmd_seed42` | DP3 7d abs (commanded+K4) | epoch 500..6000 (DP3 outputs dir) |
| `...realabl_seed42` | DP3 delta | epoch 500..6000 |

Invalidated runs (`oppsu`, `bpczv`, `ppomw`, `aurlv`, `zwiex`, `qvwdj`, `fvfnx`,
`ubyrh`, `bpkic`, `iwsbs`) are marked in `experiments.csv` with reason strings and
EXPERIMENT.md notes.

---

## 3. The three fixes that made absolute BC work

Full forensics: `docs/debug_partC_euler_action_anomaly.md`.

1. **Euler ±π wraparound seam** (`76f5efa`): a top-down grasp's roll sits at the
   `as_euler` branch cut; 18-27% of consecutive action labels sign-flipped in every abs
   dataset. Fix: `euler_frame_offset_deg: [180,0,0]` in `abs_pose_euler_abs_gripper.yaml`
   — angles encode relative to the flipped-home frame, applied identically in encode
   (`invert_absolute_action`) and decode (`ActionPipeline`). Post-fix flip rate: 0.00%.
2. **Closed-loop fixed-point stall** (`8eedf9e`): deriving the target from the ACHIEVED
   next pose gives ~zero mean lead (E[cmd_z−obs_z] = 0.0 mm at every height) — a
   deterministic diffusion rollout stalls ~3 cm above the object (policies reproduced
   training actions with ~0 bias offline, execution was mm-exact, yet eval never commanded
   below z=0.034). Fix: `--derive-lookahead K` (target = pose at t+K).
3. **Commanded-target derivation** (`c8997d0`): the definitive form —
   `--derive-source-action <collection action cfg>` reconstructs the commanded targets the
   collector actually sent (absolute: per-step decode; real teleop deltas: per-step-clamped
   accumulation) and re-encodes them. Commanded supervision is also what makes checkpoint
   curves STABLE (jfhlu, eibno@100-200, vdmtb, khxdo) where achieved-derived arms collapse.
   Real teleop's commanded lead is only ±2.6 mm, so Part A combines it with `K=4`.

Deploy-side counterpart: `deploy_real.py` warmup/smoothing now handles 7d euler
(`9938b40`); a `single_lift_mushroom_real_abs_7d` experiment config is still TODO before
real deployment (the existing real experiment carries the delta action config).

---

## 4. hwo investigation (R1/R2): recipe, not cloud, not luck

Full details: `docs/hwo_dataset_investigation.md`.

- hwo was collected with **v3** (`grasp_extra_close 5mm` + soft-firm +2mm, phases 77/30);
  xhk with **v2** (no extra squeeze; firm phase inert on soft bodies). Collection success:
  94.75% vs 84.86%.
- **R2** (`vdmtb`): fresh armfocus-cloud collection with the v3 recipe → 94.2% collect,
  policy 0.715 / **0.76 @200** / 0.745 / 0.755 / 0.725 / 0.63 — stable.
- **R1** (`khxdo`): fresh hwo-cloud reproduction → 94.2% collect, peak **0.76 @200**.
- R1 ≈ R2 ⇒ **cloud cost ≈ 0**. Both fresh collections sit ~0.08 below original-hwo
  policies (0.84-0.88) — residual run variance or the genesis submodule bump (87f0dc9).
- **Standing rule:** collect soft-body demos via `collect_demos_synth.sbatch` (v3) with
  `N_HOME_TO_PRE=77 N_GRASP=30 GRASP_EXTRA_CLOSE=0.005`.

---

## 5. Real-workspace spawn box (realws)

`DRConfig` gained absolute spawn ranges (`8339631`): `object_pos_x: [0.29, 0.48]`,
`object_pos_y: [-0.11, 0.11]` (robot-base frame; 4.6× the old ±4.5 cm box around
(0.47, 0)), one sampling path shared by collector, training-eval and offline eval.
Configs: `dr/soft_orientation_realws.yaml`, experiments `*_armfocus[_7d]_realws`.

- Collection `26-08-22-lov` (v3 recipe, armfocus obs): **92.99%** success.
- Pure-sim policy `nmbtz`: 0.47 / 0.655 / 0.69 / 0.665 / **0.71 @500** / 0.675 —
  evaluated ON the realws box (harder; not comparable to standard-box numbers).

---

## 6. Sim+real co-training (2×2×2)

Sim demos + the 50 real teleop demos (`real_abs_cmd`), union normalization, sim-only aux
labels dropped; {plain concat ≈8% real, ×4 oversample ≈25%} × seeds {42, 43}:

| run | sim data / eval box | real | curve (100→600) | peak |
|---|---|---|---|---|
| `wyigy` | armfocus_firm / std | 8% | 0.785 / 0.74 / 0.71 / 0.60 / 0.46 / 0.39 | **0.785 @100** |
| `zgwyi` | armfocus_firm / std | 8% | 0.745 / 0.76 / 0.605 / 0.63 / 0.625 / 0.59 | 0.76 @200 |
| `fbeoe` | armfocus_firm / std | 25% | 0.625 / 0.745 / 0.48 / 0.375 / 0.425 / 0.425 | 0.745 @200 |
| `gmxsx` | armfocus_firm / std | 25% | 0.29 / 0.525 / 0.65 / 0.515 / 0.46 / 0.435 | 0.65 @300 |
| `afucm` | realws / realws | 8% | 0.58 / 0.63 / 0.66 / 0.685 / 0.635 / 0.47 | **0.685 @400** |
| `jbtmt` | realws / realws | 8% | 0.535 / 0.49 / 0.585 / 0.585 / 0.33 / 0.43 | 0.585 @300-400 |
| `yrwdd` | realws / realws | 25% | 0.29 / 0.65 / 0.515 / 0.455 / 0.43 / 0.45 | 0.65 @200 |
| `eswpt` | realws / realws | 25% | 0.07 / 0.21 / 0.23 / – / 0.17 / 0.225 | 0.23 (outlier seed) |

Reads: plain concat matches the pure-sim baselines (0.76-0.785 vs 0.76; 0.685 vs 0.71) —
the real data is free in sim and strictly promising for transfer. ×4 oversampling costs
sim success and adds seed variance; its real-robot value is untested.

---

## 7. Deployment guide

**Shortlist (sim-ranked; real value to be measured on the robot):**

| policy | checkpoint | trained on | eval box | sim peak |
|---|---|---|---|---|
| `wyigy` | state_100 | armfocus_firm + real (8%) | std | 0.785 |
| `vdmtb` | state_200 | armfocus_firm (pure sim) | std | 0.76 |
| `zgwyi` | state_200 | armfocus_firm + real (8%) | std | 0.76 |
| `nmbtz` | state_500 | realws (pure sim) | realws | 0.71 |
| `afucm` | state_400 | realws + real (8%) | realws | 0.685 |
| `qjzsf` | state_500-1000 | real only (Part A) | — | n/a |

Checkpoints: `logs/dppo/dppo-pretrain/<dataset>/<run>/checkpoint/state_N.pt`.
**Each policy must be paired with ITS OWN dataset's `normalization.npz`**
(`dataset/dppo/<dataset>/normalization.npz`) — mixing normalizations silently mis-scales
obs and actions. Deployment requires a checkout with `76f5efa` (euler offset) and
`9938b40` (deploy warmup) or the decode is wrong; the deploy experiment must reference
`abs_pose_euler_abs_gripper` (for real: create `single_lift_mushroom_real_abs_7d` — TODO).
For a real-table setup matching the realws policies, place the mushroom inside
x [0.29, 0.48] × y [−0.11, 0.11] (robot-base frame).

---

## 8. Practices adopted (why the next campaign will be cheaper)

- Pre-flight verify every derived abs dataset before training: euler seam-free
  (0 dim3 jumps) AND commanded lead p75 ≥ 5 mm (chain scripts gate on this).
- One raw collection now serves delta / 7d-euler / 10d-rot6d via converter flags —
  no re-collection for action-space changes.
- Check a collection's `config.yaml control:` schema to identify the collector version
  before comparing datasets (v2 vs v3 squeeze confound).
- Early checkpoints win: best checkpoint was ≤300/600 in every stable run; always sweep,
  never ship the final epoch (Part A: val-loss minimum ≈ epoch 500-1000 of 6000).
- Eval jobs need >2h; the 4h limit still TIMEOUTs occasionally under node contention —
  the watch timer retries them (`state_*_eval_retry`).
