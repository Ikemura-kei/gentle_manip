# G6 — G5's recipe on the enlarged generalist dataset (planned, 2026-09-11)

**G6 = G5 exactly, with one change: the dataset.** No architecture, optimiser or augmentation
change. `mean⊕max` cloud pooling is inherited and is the key carried-over ingredient; the horizon-16
/ tail-22 pairing comes with it unchanged.

Baseline chain: G2 `ttukt` → G3 (horizon 4 → 16, tail 10 → 22) → **G5 `fsynt`** (+ pooling) →
**G6** (+ data). Setup references: `docs/final/G5_training_setup.md`,
`docs/final/G3_G4_training_setup.md`, `docs/final/G1_G2_training_setup.md`.

| | G5 `fsynt` | **G6 (planned)** |
|---|---|---|
| action horizon | 16 | 16 |
| executed steps | 4 | 4 |
| hold tail | 22 | 22 |
| cloud pooling | concat(max, masked mean) | concat(max, masked mean) |
| paired real–sim | 0.5 | 0.5 |
| encoder consistency | 1e-8 | 1e-8 |
| **train episodes** | **5,643** | **9,125** (of 10,139 filtered; 10 % val) |
| **gradient steps** | 933,493 (7.4 % short — see §4) | **1.34 M** (100 epochs × 13,439) |

Unchanged: lr 1e-4 cosine → 1e-5, batch 128, AdamW wd 1e-6, EMA 0.995 from epoch 1, seed 42,
`d435i_noise_train` with the ±[5, 3.5, 1.5] mm cloud offset, warmup 5 %, checkpoints every 20
epochs, val every 5.

## 1. Data components — exact folders

Merged **cleanly from raw `data.pkl`**, not from the existing npz, so one `convert_demos` call
governs the whole set and the action derivation cannot drift between sources.

| # | component | source folder | episodes |
|---|---|---|---|
| **A** | G5's original set | `dataset/demos_staging/generalist_v5/` (36 runs) | 6,270 |
| **B** | 91-object campaign | `dataset/demos/single_lift_<obj>_soft/` — 84 × `box_*_mush`, `can_lying_mush`, `beer_can_lying_mush`, `beer_can_standing_mush`, `cyl_d070h025_mush`, `paprika`, `pepper`, `banana_mush` | 1,015 |
| **C** | rod + donuts | `dataset/demos/single_lift_{rod_d015l100_mush,donut_mush_small,donut_mush}_soft/` | 75 |
| **D** | Objaverse expansion | 35 run dirs enumerated in `logs/campaign/objexp_manifest.json` → `integrate[]` | 324 |
| **E** | colleague's bundle — **sim**, the other half of the Objaverse split | `dataset/gentle_manip_demo_bundle_2026-09-11.zip` | TBD |

**Raw 10,264 episodes across 294 runs / 280 objects; 125 filtered out → 10,139 kept** (+62 % over
G5's 6,270). Per-run bookkeeping: **`logs/campaign/g6_sources_filtered.csv`**
(`episodes_raw / episodes_dropped / episodes_kept` per run).

| component | runs | raw | dropped | **kept** |
|---|---|---|---|---|
| A · G5 6k | 36 | 6,270 | 10 | **6,260** |
| B · campaign91 | 91 | 1,015 | 1 | **1,014** |
| C · rod+donuts | 3 | 75 | 0 | **75** |
| D · objaverse (local) | 35 | 324 | 17 | **307** |
| E · objaverse (cluster) | 129 | 2,580 | 97 | **2,483** |
| **total** | **294** | **10,264** | **125** | **10,139** |

⚠ **For D, read the manifest — do not glob.** Two objects (`hose`, `lemonade`) carry an extra run
from before the mass retarget, listed under `superseded[]`. Those used a different density for the
same object name; mixing them in would blend two different objects under one label.

### Bookkeeping artefact (required)

`convert_demos` writes `sources.yaml`; G6 additionally writes
**`dataset/dppo/single_lift_generalist_soft_v6/sources.csv`** — one row per source subfolder:

```
component,object,run_dir,episodes,experiment,task_name,git_commit,material,mass_g
```

so per-subfolder counts are readable without parsing YAML, and the D rows carry the per-object
material and mass that only exist in the objexp manifest.

## 2. Build steps

1. **Inspect E on arrival.** Confirmed **sim** (the cluster's half of the same Objaverse split), so
   it takes the same derivation as B/C/D and belongs in the single convert — no npz merge needed.
   Still verify its action config is `abs_pose_euler_abs_gripper_z15` and that it is raw `data.pkl`
   run dirs rather than an already-converted npz.
2. **Stage** B, C, D (and E if raw) into one tree mirroring `demos_staging/generalist_v5/` — one
   directory per run holding `data.pkl` + `config.yaml`. **Symlink, do not copy**: B+C+D is ~40 GB.
3. **Convert**, flags taken verbatim from
   `dataset/dppo/single_lift_generalist_soft_v5/launch_command.sh`:

   ```
   convert_demos <staging> --out dataset/dppo/single_lift_generalist_soft_v6 \
     --experiment single_lift_tofu_soft_abs_action_armfocus_7d_realws \
     --view student --point-cloud point_cloud \
     --derive-action        gentle_manip/configs/action/abs_pose_euler_abs_gripper_z15.yaml \
     --derive-source-action gentle_manip/configs/action/abs_pose_abs_gripper_z15.yaml \
     --val-split 0.1
   ```
4. **Hold tail**: `augment_hold_tail.py <v6> <v6_tail22> 12`. Not optional at horizon 16 — chunks
   never run off the end of an episode, so all-hold chunks number `K − horizon + 1`; at tail 10 the
   policy would see **zero** and never learn to commit to stillness. Rule from G3: **tail − horizon = 6**.
5. **Train** with `fsynt`'s overrides verbatim apart from `train.n_epochs` (§4).

## 3. Verify before launching

- **Every source's trailing hold is 10 frames.** The +12 augmentation assumes it. B/C/D come from
  the frozen v4 collector, which trims held runs (`HELD_RUN_KEEP`), but this has not been measured on
  the new data — if a component ends with a different tail, `tail − horizon = 6` breaks for that
  component only, silently.
- **Action space is `abs_pose_euler_abs_gripper_z15` for all sources.** A mismatch is not loud.
- **Point clouds are 1024 points everywhere.** D's objects were registered by `batch_expand`, which
  did not go through our obs-config path.
- **Parameter count on startup** must equal `fsynt`'s **3,194,016**. A different number means an
  override silently failed to apply.

## 4. Epochs — derive from CHUNKS, not frames

G5's own record (`G5_training_setup.md` §5) declares the trap: EPOCHS was derived from
`traj_lengths.sum()` (raw frames), but the loader yields chunks, and a horizon-16 chunk cannot start
in the last 15 frames of an episode. G5 therefore ran **933,493 steps against a 1 M target, 7.4 % short**.

```
chunks  = frames_after_tail − (H−1) × train_episodes
EPOCHS  = target_steps × batch ÷ chunks            [batch 128, H 16]
```

Computed on the **filtered** set with a **10 % val split** (settled), using measured episode
lengths (merged mean 191.5 frames — see §7b):

| | |
|---|---|
| episodes (filtered) | 10,139 → **train 9,125 / val 1,014** |
| frames after +12 tail | 1,857,032 |
| chunks | 1,720,157 |
| **batches/epoch** | **13,439** |

| epochs | gradient steps | wall clock (local 4090) |
|---|---|---|
| 69 | 0.93 M — matches G5's step count | 5.4 h |
| 82 | 1.10 M | 6.4 h |
| **100 ← settled** | **1.34 M** | **7.7 h** |
| 133 | 1.79 M | 10.5 h |

**Settled: `train.n_epochs=100`** → **1,343,900 gradient steps, ~7.7 h local** (user, 2026-09-11).
That is ~1.44× G5's 933,493 steps, chosen to hold the *step budget* the 133 originally implied
(~1.35 M) rather than the epoch count, now that the set is 10,139 filtered episodes rather than the
7,684 assumed when 133 was picked. `warmup_steps` stays proportional at 5 (5 % of the run).

## 5. Decisions still open

| decision | options | note |
|---|---|---|
| **val split** | *(settled: fresh 10 % — 9,125 train / 1,014 val)* | A fresh split makes val loss **not comparable** to `fsynt`'s 0.0024951. Accepted; compare on teasers/eval instead. |
| **include D?** | in / out | 324 Objaverse episodes use `objexp_*` materials at densities 150–2,500 and their own DR — a wider distribution than A's mushroom/cube/banana. If G6 underperforms G5 on cherry, an ablation without D is the first thing to try. |
| **cuboid balance** | plain concat / downweight | 84 near-identical boxes × 10 = **840 episodes, 11 % of the set** on one shape family. G5's validated recipe is plain concat, no oversampling ("noos"); deliberately skewing toward boxes is a choice, not a default. |
| **epochs** | *(settled: **100**)* | On the FILTERED set at 10 % val: 9,125 train eps → 1,720,157 chunks → **13,439 batches/epoch**. 100 epochs = **1.34 M steps, ~7.7 h local**. |

## 6. Wall-clock estimate

Measured from this project's own horizon-16 runs, not extrapolated:

| reference | batches/epoch | rate |
|---|---|---|
| `mmgyy` (G3, horizon 16, **local 4090**) | 8,261 | **47.3 steps/s** (175 s/epoch) |
| `bpfnl` (horizon 16, local) | 8,261 | 49.1 steps/s |
| `fsynt` (G5, **cluster** n200) | 8,261 | 32.0 steps/s |

G6 at **133 epochs × ~10,124 batches = ~1,346,000 steps**:

| where | estimate |
|---|---|
| **local (4090)** | **≈ 8 h** |
| cluster (G5's node rate) | ≈ 11.7 h |

At 109 epochs it would be ≈ 6.5 h local / 9.6 h cluster.

Two caveats. `mmgyy` used **max** pooling, not `meanmax` — the extra branch adds a
`Linear(512→512)` and a masked mean over 1024 points, so expect a few per cent slower, not more.
And the local figure assumes the GPU is otherwise idle; the estimate degrades directly with anything
else sharing it. Add the dataset build on top: convert + hold-tail on ~7,700 episodes is roughly an
hour, and the G5 record is explicit that a queued job must never race a background build.

### MEASURED (2026-09-11, run `vnhnr`) — the estimate above was wrong

The estimate assumed the cloud bank fits on the local GPU. It does not, and correcting that
cost most of the projected speed.

`StitchedSequencePointCloudDataset` puts the whole point-cloud bank on the GPU as float32:
**21.3 GiB train + 2.3 GiB val = 23.6 GiB on a 24.0 GiB 4090**, before the model. `fsynt` ran on
the cluster, where that fits. Locally it cannot. The fix is `GM_CLOUD_DEVICE=cpu` (added to
`gentle_manip/dppo/pointcloud_dataset.py`): the bank stays in host RAM as float32 and each sample
ships ~12 KB to the GPU. Numerically identical to GPU residency — deliberately not a
half-precision bank, which would have quantized cloud coordinates to ~0.5 mm.

| | predicted | measured |
|---|---|---|
| batches/epoch | 13,439 | **13,439** ✓ |
| train steps | 1,858,495 | **1,858,495** ✓ |
| network parameters | 3,194,016 | **3,194,016** ✓ |
| rate | 47.3 steps/s | **26.8 steps/s** |
| s/epoch | ~263 s | **502 s** (epochs 1 and 2, steady) |
| **100 epochs** | **≈ 7.7 h** | **≈ 13.9 h** |

Build stages measured: convert 11 min, hold-tail 13 min.

If the local wall clock matters more than staying byte-identical to `fsynt`, the options are the
cluster (where the bank fits on-device) or a half-precision bank (11.8 GiB, full local speed, ~0.5 mm
quantization — below the d435i augmentation noise the recipe already injects).

### `augment_hold_tail` could not handle a set this size (fixed)

The original loaded every key, built a per-episode list, then concatenated — three full copies of a
24 GB cloud array, ~70 GB peak, OOM-killed (rc=137) on the 62 GB box. G5 never hit it at half the
size. Rewritten to build one gather-index array and stream each key into the zip in 20k-row chunks;
peak is now one source key plus a chunk (6.2 GB observed). Verified bit-identical to the original
semantics on a synthetic case before running it on the real data.

## 7a. Sanity checks on every component that is not A

**Everything except A is unvalidated data.** A trained G5 and is known-good; B, C, D and E have never
been through a training run. A degenerate component cannot be spotted from a loss curve after the
fact — it just quietly shifts the policy. Run these per component *before* converting, and record the
output alongside `sources.csv`.

### Blocking — a failure here means fix or drop the component

| check | why | signal of trouble |
|---|---|---|
| **trailing hold length** | the `+12 → tail 22` rule assumes every episode ends with a 10-frame hold; `tail − horizon = 6` is what makes all-hold chunks exist at horizon 16 | any component whose modal tail ≠ 10 |
| **success ≥ 1 per episode** | `episodes_saved > 0` is already gated by the collector, but a *saved* episode that never lifts is possible | `ever_success` false on a saved episode |
| **NaN / Inf** in states, actions, clouds | one NaN poisons a whole batch | any |
| **action saturation** | actions are normalised to [−1, 1]; a component whose actions pile up at the bounds means the action config does not fit that data | > ~1 % of values at \|a\| > 0.99 |
| **point-cloud padding fraction** | **specific to G6**: `meanmax` pooling masks padded `(0,0,0)` points, and the G5 doc measured only 61 of 119,716 val frames padded. A component with substantially more padding changes what the mean branch sees | padded-point fraction far above A's |
| **duplicate episodes** | the top-up pass re-ran 10 objects; a seed collision would produce byte-identical episodes | hash of (actions, states) repeating across runs |

### Distributional — not pass/fail, but read them before training

Your list, all worth doing, plus what I would add:

- **grasp-pose euler distribution** (roll/pitch/yaw at the grasp frame) — D's objects got the full
  ±90° yaw box for large extents while A's did not, so expect a genuinely different shape; the thing
  to look for is a *spike* at a bound, which means the search was clipped rather than converged.
- **object size distribution** — B is 84 near-identical cuboids; this is where the 11 % shape-family
  skew becomes visible.
- **grasp width distribution** — the single most diagnostic plot here. A bimodal or
  pinned-at-`WIDTH_MAX` width is exactly the `_local_width` void bug found on the donut, where 63 %
  of seeds asked for an impossible 79 mm grasp. Anything pinned at 79 mm should be treated as suspect.
- **episode length distribution** — feeds the epoch calculation directly (§4 assumes A's 190.4
  frames/episode) and exposes truncated runs.
- **width closing speed at the grasp moment** — good idea and cheap: the executor closes at a fixed
  rate, so an inconsistent slope means either a different `GRIP_SPEED` or a truncated close. Worth
  extending to **commanded-vs-measured width**, which is what exposed the 0.8 mm `EXEC_EXTRA_CLOSE`
  and the FEM/MPM inset.
- **stress / sub-yield per component** — already in each `stats.yaml`. B's rod came in at
  sub-yield 0.914, the only object today that bruised; if gentleness is a headline metric, a
  component that routinely exceeds yield is training the policy toward ungentle grasps.
- **joint normalisation range** — ⚠ the one I would not skip. Convert re-derives min/max over the
  *merged* set. A single component with extreme states or actions widens the range and **compresses
  every other component's signal**. Compare the merged `normalization.npz` against A's: if any
  dimension's range grows materially, find which component caused it.

Cheapest useful form: one script over each component's `data.pkl` emitting a small per-component
row into `sources.csv` plus a one-page figure per component. This is worth doing *before* the
~8 h training run, not after.

## 7b. Raw-data analysis — results (2026-09-11)

Ran over all 294 runs; the table below is the **FILTERED** set (10,139 episodes) — the final train+val population. Per-run and per-episode outputs:
`logs/campaign/g6_analysis_{runs,episodes,grasp}.csv`, `g6_analysis_dups.json`.
Figures: `docs/final/figures/g6/`.

| component | episodes | grasp ev | len mean | w med | w p5 | w p95 | w≥78% | close | grasp@% |
|---|---|---|---|---|---|---|---|---|---|
| A · G5 6k | 6,260 | 5,740 | 190.4 | 33.1 | 19.1 | 50.9 | 0.00 | 2.16 | 64 |
| B · campaign91 | 1,014 | 809 | 199.4 | 35.4 | 19.9 | 63.3 | 0.00 | 2.16 | 67 |
| C · rod+donuts | 75 | 48 | 198.4 | 16.6 | 9.9 | 34.0 | 0.00 | 2.18 | 64 |
| D · objaverse (local) | 307 | 221 | 189.0 | 51.3 | 13.5 | 70.1 | 0.00 | 2.13 | 64 |
| E · objaverse (cluster) | 2,483 | 1,817 | 191.3 | 51.4 | 28.1 | 71.0 | 0.00 | 2.13 | 65 |
| **merged** | **10,139** | **8,635** | **191.5** | **35.2** | | | **0.00** | | |

*(w = grasp width mm; close = peak closing rate mm/frame; grasp@ = % through the episode.)*

### Blocking checks — all pass

| check | result |
|---|---|
| trailing hold | **11 frames on all 294 runs**, every component — uniform, so `+12 → tail 22` is valid everywhere |
| NaN / Inf | **0** episodes |
| action dimension | **10** everywhere |
| duplicate episodes | **0** action-prefix collisions across 10,264 episodes |
| grasp width at the 79 mm wall | **0.00 % in every component** |
| action saturation | ≤ 0.31 (A is the highest; the new data is better behaved than the set that already trained) |

⚠ The doc's "hold tail 10" is a counting convention that excludes the final frame. Measured, **A
itself reads 11** — so 11 is the reference, not a defect, and what matters is that it is *identical*
across components.

### Figures

| file | shows |
|---|---|
| `01_composition_length.png` | episodes per component; episode-length distributions |
| `02_grasp_width.png` | grasp width per component vs the 79 mm gripper maximum |
| `03_grasp_orientation.png` | \|roll\| from top-down, pitch, yaw at the grasp |
| `04_size_vs_width.png` | object thinnest extent vs median grasp width, per object |
| `05_speed_timing.png` | peak closing rate; when in the episode the grasp happens |
| `06_cloud_padding.png` | max padded cloud points per run |

**`07_combined_distributions.png` — the final set, all components pooled (added 2026-09-11).**
Figures 01-06 break the metrics down per component; this one answers what the policy actually sees
in aggregate. Rescanned directly from the staged tree that `convert_demos` read, so it is the final
10,139 episodes by construction rather than a raw table filtered after the fact —
`logs/campaign/g6_final_episodes.csv`, which carries `run_dir` + `ep` per row (the earlier
`g6_analysis_grasp.csv` carried no episode identity, which is why it could not be filtered directly).

| metric | n | median | p5 | p95 |
|---|---|---|---|---|
| grasp width | 9,954 | 36.2 mm | 19.1 | 65.7 |
| episode length | 10,139 | 195 frames | 141 | 232 |
| grasp moment | 9,954 | 60.0 % | 44.7 | 66.4 |
| peak closing rate | 9,954 | 2.16 mm/frame | 2.04 | 2.21 |
| grasp roll | 9,954 | 179.9° | 162.7 | 197.1 |
| grasp pitch | 9,954 | 0.1° | -13.3 | 13.6 |
| grasp yaw | 9,954 | -1.0° | -58.3 | 58.8 |
| point-cloud padding | 48 | 0.89 % | 0.07 | 3.71 |

Reading it: grasp width is broad and single-peaked at 36 mm with a long tail to the 70 mm end (the
Objaverse components) and no pile at the 79 mm gripper wall. Closing rate is a spike at
2.16 mm/frame — the scripted demonstrator is highly consistent, which is what makes any outlier
meaningful. Roll is unimodal on gripper-down; yaw is near-uniform across +/-60 deg, i.e. the full
commanded range, so approach direction is well covered. Padding is a non-issue: **48 of 10,139
episodes (0.47 %) contain any padded point at all**, and the worst of those is 4.5 %.

**`08_category_histogram.png` — episodes per object category.** 280 distinct object names collapse
to **153 categories**: every cuboid variant (84 `box_WxDxH` sizes + 5 `bs_cube` meshes +
`prim_cuboid`) is pooled into one `cuboid` category, and Objaverse variant suffixes (`pea2`…`pea12`)
merge into their base name. Log x-scale, because counts span 3 to 1,590.

| | |
|---|---|
| largest | **cuboid 1,590** (A 750 + B 840) |
| next | mushroom / prim_cylinder / prim_ellipsoid / tofu 550 each, strawberry 549, tomato 418 |
| Objaverse body | ~20 episodes per category — 10 demos x 2 runs for most |
| smallest | hose 3, shredder 3, pet / strainer / wallet 7, pan / pillow / postcard 8 |

The shape is strongly bimodal and that is the honest headline: the **A + B core objects carry
hundreds of episodes each, while the ~130 Objaverse categories carry ~20**. Roughly 62 % of all
episodes sit in the 12 largest categories. So G6 buys breadth of object identity, not depth per
novel object — a per-object generalization claim on the Objaverse tail rests on ~20 demos each.

Only 2 categories span more than one component (`cuboid` = A 750 + B 840, `squid` = D 10 + E 20), so
the single bar colour is accurate for the other 151.

#### The grasp detector had 14.7 % false negatives (fixed 2026-09-11)

The first pass reported 8,647 grasp events and implied 1,492 episodes had none. They all had one:
median 37 mm of closing and ~200 mm of lift travel, none static, none that never lifted.

The local-minimum detector assumes the width stops moving exactly at lift-off. It usually does — but
the width then sits on a dead-flat hold plateau to the end of the episode, and with sub-micron drift
on that plateau the only *strict* local minima fall late in it, after the lift. Its EE-at-table gate
then correctly rejected those, while the real grasp moment earlier in the close was never a
candidate. 1,208 of the 1,492 died on that gate, 284 on the 6 mm close gate.

**v2: the grasp is where the CLOSE COMPLETES** — the first frame at or below the settled hold width,
searched only AFTER the fully-open frame. Searching after the open frame is what stops a part-closed
start from masquerading as the grasp, which was the original `argmin(width)` bug.

| | v1 (local minimum) | v2 (close completes) |
|---|---|---|
| grasps found (6,589-episode A/B) | 6,005 (91.1 %) | **6,585 (99.94 %)** |
| width where both fire | — | agrees to **0.11 mm** median; >1 mm on 1 of 6,005 |
| EE height above episode floor at the grasp | later, into the lift | **0.0 mm, p95 1.4 mm** |

That last row is the independent check: v2's grasp moment is physically at the table, which is what
the grasp moment should be. Effect on the full set: 8,647 -> **9,954** grasp events (98.2 % of
episodes; the 185 that remain remain genuinely close by less than 6 mm). Grasp width moved
35.3 -> 36.2 mm and the grasp moment 64.7 -> 60.0 % (v2 fires ~10 frames earlier); orientation is
unchanged. **The per-component table in section 7b above still uses v1** and therefore understates
grasp counts — its width/orientation columns are within ~1 mm / ~1 deg of the corrected values.

Two further notes on how the numbers are computed. **Roll needs unwrapping** — it sits exactly on the
+/-180 deg branch cut, so a naive histogram splits one peak into two piles at opposite edges; mapped
to [0,360) it is cleanly unimodal at 179.9 deg. And "trailing constant-width frames" (median 34)
measures something different from the **trailing hold = 11 frames** blocking check above — that check
counts repeated trailing ACTION frames, this counts frames whose gripper width has stopped changing,
which begins earlier. They do not contradict.

### What the distributions say

**The Objaverse halves are a genuinely different size regime.** D and E both sit at a **51.3 mm**
median grasp width against A's 33.1 — their objects are larger, and `04_size_vs_width.png` shows
them clustered at 40–55 mm thinnest extent where A spans 12–45 mm. This is the intended broadening,
but it is the single largest distributional change in the merge. Merged median width lands at
~34 mm because A still dominates by count.

**Everything process-side is consistent across components.** Peak closing rate 2.13–2.18 mm/frame,
grasp at 64–67 % through the episode, episode length 188–200 frames (merged mean **191.5**, against
A's 190.4 — so the §4 epoch projection's frames-per-episode assumption holds).

**Cloud padding is the one G6-specific risk.** Max padded points per run: **40 % in E**, 20 % in A,
0 % in B/C/D. `mean⊕max` masks padded `(0,0,0)` points, so this does not corrupt the mean branch —
but E contains runs where a large share of a sampled cloud is padding, which is worth a look if the
encoder behaves oddly.

**Grasp detection covers 84.5 % of episodes** (8,672/10,264). The detector is deliberately
conservative — it requires a ≥6 mm close with the EE at the table beforehand — so the shortfall is
mostly multi-attempt or shallow-close episodes, not bad data. Coverage is lowest on C (64 %, n=75).

### Method note — a metric that was wrong, and how

The first pass keyed the grasp moment on `argmin(width)`. **That is wrong**: `start_modes`
randomises the initial gripper width, so an episode that begins part-closed has its minimum at
frame 0, never at a grasp (user, 2026-09-11). It produced two convincing but false findings:

- a **pile at the 79 mm gripper wall** (D 5.2 %, E 2.8 %) — actually start poses, not grasps;
- **erratic closing speed in D and E** (cv 0.41 / 1.30) — actually episodes with no close to measure.

Both vanish with the verified detector ported from `refresh_results_table.py::_at_grasp`: a local
width minimum ≥6 mm below the preceding open level, **with the EE at the table in the frames BEFORE
it** — the minimum itself coincides with lift-off, so testing the EE *at* the minimum rejects every
real grasp. Corrected widths ran +0.9 to +4.0 mm higher than the argmin figure.

### Excluded episodes — empty clouds and solver blow-ups (user, 2026-09-11)

Two independent defects, neither of which subsumes the other. Authoritative list:
**`logs/campaign/g6_dropped_episodes.{json,csv}`** — **8 episodes across 7 runs**, applied at
**staging time** (filtered copies of those `data.pkl` files); raw data left intact.

| object | comp | ep | reason | detail |
|---|---|---|---|---|
| `tomato` | **A** | 389 | explosion | EE jump 554 mm/frame |
| `cleansing_agent3` | E | 7 | explosion | EE jump 1168 mm/frame (p99 767) |
| `condiment4` | E | 5 | both | 63/176 empty frames (35.8 %) + 1029 mm/frame |
| `condiment4` | E | 6 | both | 52/184 (28.3 %) + 1145 mm/frame |
| `doorknob` | E | 0 | both | 33/206 (16.0 %) + 1385 mm/frame |
| `goggles3` | E | 3 | empty cloud | 55/189 (29.1 %) |
| `parasail` | E | 6 | both | 44/202 (21.8 %) + 1165 mm/frame |
| `shot_glass2` | E | 6 | both | 36/217 (16.6 %) + 1368 mm/frame |

**Criteria.** *Empty cloud*: `empty_frames >= 5 and empty_frac > 0.02` — a 100 %-padded frame means
perception returned nothing, so the policy is asked to act on a blank observation. *Explosion*:
**max per-frame EE displacement > 200 mm** — the arm teleporting, i.e. a solver blow-up.

Both cuts sit in a wide gap, so neither is a judgement call. The six empty-cloud episodes carry
33–63 empty frames (16–36 %); the next-worst episode anywhere has **2** (1.0 %). The seven
explosions move **554–1385 mm in a single frame**; the next-highest anywhere is 93.5 mm. Normal
motion is a **median 4.5 mm/frame, p99 8.6**.

Five episodes trip both signals — the solver explodes, particles scatter out of the crop box, and
the cloud empties — but `cleansing_agent3` ep7 explodes with **zero** empty frames, and `goggles3`
ep3 empties with normal motion. A single filter would have missed one or the other.

⚠ **One is in component A**, the data G5 already trained on: `tomato` ep389, a 554 mm/frame jump.
G5 trained on it; G6 will not.

**After exclusion: 10,256 episodes.**

### Deliberately NOT excluded

| signal | count | why it stays |
|---|---|---|
| `priv_stress` = NaN | 29 eps | The student view is `PROPRIO_VIEW` (`ee_pos, ee_quat, gripper_width`) — **`priv_stress` is dropped at convert** and never reaches training. Verified in `convert_demos.py`. |
| `dquat ≈ 2.000`, normal motion | 4 eps | Quaternion **sign flip** — q and −q are the same rotation (CLAUDE.md). `banana_mush`, `beer_can_lying_mush`, `telephoto_lens`. |
| `donut_mush_small` ep15 | 1 ep | 93 mm/frame over 13 **contiguous** frames, EE inside the workspace (z 22–278 mm), no NaN — a fast retarget, not a teleport. |
| partial cloud padding | 37 eps in A | `bs_cylinder2` 63–65 %, `cherry_tomato` 54–59 %, zero fully-empty frames. **G5 trained on these**; `mean⊕max` masks padded points. |

⚠ My earlier "0 NaN episodes" was true only of the fields I checked (`actions`, `gripper_width`).
`priv_stress` went NaN in 29 episodes and was not covered — the scan above closes that gap.

## 8. Evaluation and deployment

Compare **G6 against G5 `fsynt`**, not against G3 — G6 and G5 share the pooling and horizon, so the
difference isolates the data. Canonical eval is 100 episodes (`EvalSpec`); teasers at n = 20 are
diagnosis, not ranking, since the spread (5) sits inside 1 σ (2.8).

Loss is comparable to G5 only if the val split is held fixed, which it is not here — see §5.

⚠ A `pooling=meanmax` checkpoint carries a **different encoder shape**; evaluating or deploying it
requires the same `+model.network.pointnet.pooling=meanmax` override, otherwise the state-dict load
fails (loudly).

⚠ The merge **changes normalization**. The deploy entry must point at
`dataset/dppo/single_lift_generalist_soft_v6_tail22/normalization.npz`; a stale path fails after the
arm has already homed.

## 9. Disk

v6 npz ≈ **15.5 GB** at G5's ~2 MB/episode, plus E unzipped, plus staging symlinks (negligible).
Free at time of writing: **87 GB**, after removing seven pre-September log runs (0.67 GB) and
`single_lift_generalist_soft_v5/{train,val}.npz` (12 GB — the non-tail22 intermediate, superseded by
`_tail22`; its `normalization.npz`, `sources.yaml` and `launch_command.sh` were **kept** because a
deploy entry and the rebuild recipe reference them).
