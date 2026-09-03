# Development Log — Gentle Manipulation Sim2Real

**Mission.** Develop a sim2real policy-learning framework for gentle manipulation of
real-world food, where demonstrations come from a **stress-aware scripted demonstrator with
grasp synthesis** (CMA-ES over grasp poses + von-Mises-stress-aware execution on soft MPM
bodies). End goal: a policy that handles **multi-category objects** (tofu, mushroom,
fruits, …). Current stage: a **specialist** policy (mushroom only) used to nail down the
training/data foundation; the **generalist** data-collection/training recipe is a later
investigation.

Subpages (detail documents):
- [Paper TODO](PAPER_TODO.md) — what is needed to SUBMIT (blocking / strengthening / verification / cut), plus the settled story and related-work positioning
- [Action-space ablation & follow-ups — final report](action_space_ablation_final_report.md) — all runs, curves, numbers
- [hwo dataset investigation](hwo_dataset_investigation.md) — collection-recipe forensics (R1/R2)
- [Abs-action derivation debugging](debug_partC_euler_action_anomaly.md) — the euler-seam and fixed-point-stall root causes
- [Campaign instructions (original)](cluster_experiment_action_space_ablation.md)

---

## Current foundation (adopted 2026-08-23, pending real-robot validation)

The following stack is the recommended baseline for further development. **Every element
is backed by the sim experiments in the final report; none of it is yet validated on the
real robot — that is the next gate.**

| element | choice | why |
|---|---|---|
| point cloud | **arm-focus filter** (1024 pts) | matches the real rig's processed cloud; costs ≈0 in sim (R1 0.76 = R2 0.76) |
| rotation obs | **quaternion** (canonicalized sign) | obs_dim 8; rot6d obs adds nothing for the EE's bounded range |
| action space | **7d absolute** (pos3 + euler3 + grip1, frame-offset euler) | ≈ delta in sim (0.76 vs 0.75); absolute re-anchors every step, so it does NOT accumulate drift in real, where the sim2real gap would otherwise compound delta errors; ≈ 10d rot6d (0.84 vs 0.88) at 3 fewer dims |
| action derivation | **recorded COMMANDED targets** (`--derive-source-action`), +K=4 lookahead for slow real teleop | achieved-pose derivation stalls at a closed-loop fixed point (see debug doc); commanded supervision also gives stable checkpoint curves. NOTE: commanded targets carry the closed-loop lead INHERENTLY (controller tracking lag, 5-10 mm) — natively recording 7d-euler commands at collection time would be bit-equivalent (decode→re-encode is lossless, ~1e-7) and needs no lookahead; we record 10d + derive only to keep one collection convertible to ANY action space. Lookahead is a patch for lead-deficient sources only (achieved poses; slow real teleop at ±2.6 mm) |
| workspace DR | **real-workspace absolute spawn box** x [0.29, 0.48] × y [−0.11, 0.11] | matches the physical table; 0.71 sim success on the 4.6×-larger region |
| demo collection | **v3 collector, hwo recipe**: phases 77/30, `grasp_extra_close=5 mm` (+2 mm soft-firm) | the recipe (not the cloud, not luck) explains the 0.62→0.76+ policy gap and 85%→94% collection success |
| real demos (mushroom) | **`dataset/demos/single_lift_mushroom_real_merged`** — THE default/foundation real dataset; reuse whenever real mushroom data is appropriate (co-training, real-only training, sim2real diagnosis) | 55 teleop episodes (51-ep `26-08-20-cmh` + 4-ep top-up), uniformly recorded through `point_cloud_1cam_armfocus` (verified fingerprint) with delta fast_rot actions → convertible to any action space via `--derive-source-action <delta cfg> --derive-lookahead 4`; fed qjzsf (real-only) and every co-train run |
| trainer | **DPPO** PointNet diffusion (big net) | beats DP3 on both regimes: sim 0.76-stable vs 0.74-unstable/0.53-plateau; real 0.60 vs 0.52 (canonical harness, resolved 2026-08-23) |
| checkpoint selection | sweep every checkpoint with the canonical harness; peak is at 100-300/600 | every stable run peaks early; never ship the final epoch |

### Reproduction recipe (exact commands)

**1. Collect** (GH200 node, ~6 h / 650 episodes, ~93-94% demonstrator success):
```bash
EXPERIMENT=single_lift_mushroom_soft_abs_action_armfocus_realws N_EPISODES=650 N_ENVS=8 \
MAXFEVALS=1145 SEED=0 SCENE_DR_EVERY=1 N_HOME_TO_PRE=77 N_GRASP=30 GRASP_EXTRA_CLOSE=0.005 \
sbatch gentle_manip/scripts/arrhenius/collect_demos_synth.sbatch   # runs the v3 collector
```
(non-realws variant: experiment `single_lift_mushroom_soft_abs_action_armfocus`)

**2. Convert** (commanded-euler 7d; obs = ee_pos + ee_quat + gripper_width + cloud):
```bash
uv run --project envs/dppo_arrhenius --no-sync python -m gentle_manip.dppo.convert_demos \
  <collection>/data.pkl --out dataset/dppo/<name> \
  --obs-keys ee_pos ee_quat gripper_width --point-cloud \
  --derive-action gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
  --derive-source-action gentle_manip/configs/action/abs_pose_abs_gripper.yaml
# real teleop pkls (recorded delta actions): --derive-source-action <delta cfg> --derive-lookahead 4
```
Pre-flight gate before training: euler dim3 seam-free (0 jumps) AND commanded lead
p75 ≥ 5 mm (see the verify blocks in the scratchpad chain scripts / debug doc).

**3. Train** (DPPO, jfhlu config: horizon 4 / cond 2 / denoise 20, visual_feature_dim 512,
mlp [1024,1024,1024], 600 epochs, checkpoint every 100):
```bash
DATA_ENV=<name> N_EPOCHS=600 SAVE_FREQ=100 \
CFG_DIR=$PWD/gentle_manip/dppo/cfg/single_lift_mushroom_soft_abs_pcd_hwo CFG_NAME=pre_diffusion_pointnet \
GM_EXTRA_OVERRIDES="action_dim=7 experiment=single_lift_mushroom_soft_abs_action_armfocus_7d_realws" \
sbatch gentle_manip/scripts/arrhenius/dppo_pretrain.sbatch
```

**4. Evaluate** every checkpoint with the canonical harness (200 eps, 5 envs, seed 42,
scene_group_size 4, 300 sim steps, per-episode video + DR audit) via
`dppo_eval.sbatch` with the SAME experiment; pick the peak checkpoint.

**5. Deploy**: pair the checkpoint with ITS dataset's `normalization.npz`; checkout must
include commits `76f5efa` (euler frame offset) + `9938b40` (7d deploy warmup); the deploy
experiment must reference `abs_pose_euler_abs_gripper`. For real: create
`single_lift_mushroom_real_abs_7d` experiment config (TODO). Real-table object placement:
inside x [0.29, 0.48] × y [−0.11, 0.11] (robot-base frame).

---

## Conclusions to date (2026-08-23)

**Multi-object grasp synthesis (2026-08-27).** ONE auto recipe now covers four objects with no
per-object parameters — `--grasp-area-min-mm2 auto --grasp-width-max-mm auto`, with E / density /
yield resolved from the object's own material (previously ALL objects silently used the
mushroom's 3e5 / 1000, which corrupted both stress AND grip force per object):

| object | demonstrator success | stress vs yield | align |
|---|---|---|---|
| mushroom | 16/16 = 100 % | 29 % | 0.94 |
| raspberry | 16/16 = 100 % | 40 % | 0.89 |
| tofu | 23/24 = 96 % | 15 % | 0.98 |
| strawberry | 22/24 = 92 % | 60 % | 0.92 |

Tofu needed no special handling. Strawberry at 60 % of yield is the closest to bruising and is
worth watching. **The banana is PARKED** — see Open questions; it is a contact-model validity
limit, not a tuning gap.

1. **7d euler-abs ≈ 10d rot6d** in sim (0.84 vs 0.88) once the euler encoding carries a
   frame offset and commanded-target derivation → the compact action is (nearly) free.
2. **Abs ≈ delta in sim**; abs chosen for real deployment because it re-anchors to an
   absolute pose every step — no drift accumulation under sim2real execution error,
   unlike delta where within-chunk open-loop drift and target-vs-actual accumulation
   compound.
3. **Demonstrator grasp firmness is a first-order lever**: the v3 recipe's extra squeeze
   (5 mm + soft-firm) moved trained-policy success 0.62-0.70 → 0.76+ and demonstrator
   success 85% → 94%. Collection reproduces deterministically (3 collections within 0.6%).
4. **Arm-focus cloud is free in sim** and matches the real rig → strictly preferable.
5. **BC'd absolute actions must be supervised with targets that LEAD the pose**
   (commanded or ≥4-step lookahead); achieved next-pose targets create a closed-loop
   fixed-point stall (0% success with perfect training loss). Two encoding bugs (euler ±π
   seam; the stall) cost the first two rounds — both now have automated pre-flight gates.
6. **Co-training with the 50 real demos: no harm in sim, and plain concat (no
   oversampling) is the best variant** — confirmed: no-oversample peaks 0.785/0.76 (std
   box) and 0.685/0.585 (realws) vs ×4-oversample 0.745/0.65 and 0.65/0.23. Whether it
   HELPS in real is untested (the whole point of adding real data).
7. **Early checkpoints win**: every stable run peaked at 100-300 of 600 epochs (DPPO) and
   real-data val loss bottomed at ~epoch 600-1000 of 6000 — sweep-and-pick is mandatory,
   and DP3 must retain all periodic checkpoints (`checkpoint.topk.k=12`; its default k=1
   with no sim score silently keeps only the most-overfit checkpoint).
8. **Commanded lead is necessary but NOT sufficient — trajectory DWELL is an independent
   stall mechanism** (local agent, 2026-08-23). A v4-collector dataset with min-jerk time
   scaling passed BOTH pre-flight gates (seam 0 jumps; commanded lead p75 9.4 mm) and still
   evaluated 0/75 with perfect training loss, in every cell of a (rot6d vs euler) ×
   (measured vs commanded) 2×2. Cause: min-jerk's velocity→0 tails put 32% of consecutive
   actions within 0.01 (normalized) of each other (hwo/v3-linear: 2%) — a cond=2 policy
   fixed-points there exactly as in the achieved-pose stall, and `act_steps=1` does not
   rescue it. Reverting to LINEAR time scaling (same collection otherwise) dropped the
   dwell fraction to 7% and took the same recipe from 0 → **0.66** (sqcvc/state_600, 200
   eps; 7d-commanded twin 0.50). Proposed third pre-flight gate: frac(|ΔA| < 0.01) ≲ 10%.
   Full forensics: `grasp_synthesis_v41_review.md` + this repo's v5/v6/v7 datasets.
9. **Rate-bounded absolute actions are free** (local agent): per-step clamp of the decoded
   absolute target against the previous command (delta-`scales` convention, rotation ×1.5),
   enforced at BOTH the backends (real-arm safety: a policy-emitted pose jump executes as a
   bounded walk) and the scripted collector (datasets bounded by construction;
   `audit_demo_rate_bound.py` verifies the artifact — 0 violations in 115k steps). n=100
   paired benchmark: success 0.960 = 0.960, stress at noise, act-jerk −13%.
10. **Two demonstrator-side roadmap items are DONE** (local agent): occlusion-aware
   synthesis — a hard closing-axis-azimuth bound (45°) at every CMA ladder rung (the soft
   `w_occ` weight is provably inert: occluding-candidate scores sit on a flat infeasibility
   floor); ground-truth fully-hidden grasps 24% → 4% at unchanged success. And
   failure+recovery: `--retry-max` regrasp-on-slip with the trigger window measured
   (0.15–0.30 of the lift recovers 5/5; the original 0.45 guess recovers 0/3), failed
   attempts kept in the demos by design.
11. **Grasp-synth benchmark scale bug** (local agent): `eval_grasp_synth` planned on
   NOMINAL-size meshes for every scaled scene (server bakes scale onto ObjectEntry.scale;
   the deformed file is nominal) — executed widths silently over-squeezed up to ~10 mm.
   Fixed (43b388a); absolute numbers from that benchmark before the fix are tainted
   (paired comparisons survive). Related: `execute_offset` (scoring at the executed width)
   is retired from collection — the FEM's 2μ·grip ≥ mg hold margin does not survive MPM at
   honest widths (8/8 → 1/8 on true-size meshes; a 4× margin does not rescue it).
12. **REAL-ROBOT results (2026-08-23): co-training wins.** `afucm/state_400` ~**75%** real
   success (best so far); `qjzsf/state_1000` (real-only) second; `nmbtz/state_500` (pure
   sim; sim-best 0.71) worst. See the resolved open question — sim rankings invert across
   data regimes. Deploy wiring for all three: `deploy_real.sh` (armfocus obs, euler-7d,
   rate-limit clamp).
13. **Best current deployment candidates** (sim-ranked; real value untested):
   `nmbtz/state_500` (pure-sim realws 0.71), `afucm/state_400` (realws + real 0.685),
   `wyigy/state_100` (std box + real 0.785), `vdmtb/state_200` (pure-sim std 0.76),
   `qjzsf/state_500-1000` (real-only DPPO abs).

## Open questions (gates before further building)

- **~~Can the banana be added as a multi-object category?~~ PARKED (2026-08-27, user decision:
  "give banana up").** Not a tuning problem and not worth more effort: the FEM contact model
  scores **0/8** of the grasps that demonstrably lift it (`degenerate` 5/8, `no_contact` 3/8),
  because it evaluates contact against the NOMINAL undeformed mesh and the banana's working
  grasps need ~54 % compression — outside small-strain validity (`max_indent` 0.01 m). The model
  is correct within its assumptions; the banana violates them. **Do not re-attempt via seeding,
  CMA budget, width caps or area floors — all four were tried and cannot work, because the
  target grasps are unscoreable, not merely unfound.** Reopening this requires a contact model
  evaluated in the DEFORMED configuration plus a large-deformation stress model (the linear FEM's
  stress is untrustworthy at 54 % strain), which is a project in itself.
  Stopgap if ever needed: the geometric heuristic (centre, closing perpendicular to the longest
  axis) lifts it 80 % at 26 mm with 11 % compression — see the 2026-08-27 log entries.
  **The other four objects are unaffected and READY**: mushroom 100 %, raspberry 100 %, tofu 96 %,
  strawberry 92 % under the all-auto recipe.

- **Real-robot validation of the whole foundation** — nothing above is real-verified yet.
- **~~Does real co-training help in real?~~ ANSWERED (2026-08-23, real-robot runs): YES.**
  Real deployment of the shortlist: `afucm` (realws sim + 8% real co-train) **~75% real
  success — best**; `qjzsf` (real-only, 55 demos) second; `nmbtz` (pure sim, the SIM-best
  0.71) **worst**. The sim ranking INVERTS in real: a pure-sim policy still carries a
  sim2real gap that 8% real co-training largely bridges, and 55 real demos alone beat
  pure sim — the gap is in the data domain, not the recipe. Sim scores remain useful for
  in-family model selection but do NOT rank across data regimes.
  Deploy-prep note (local agent, corrected): the co-train real slice is
  `single_lift_mushroom_real_merged` (55 demos = the 51-ep `26-08-20-cmh` session + a 4-ep
  top-up; uniform cloud fingerprint across all 55), RECORDED through
  `point_cloud_1cam_armfocus` — so sim and real
  training clouds are consistent, and ALL THREE shortlist deploys use the armfocus obs
  config (qjzsf included: its pkl clouds are record-time armfocus even though the run's
  `superset_real` env snapshot has no focus block — the snapshot describes the experiment
  env, not the pkl's baked-in processing). An earlier caveat claiming the real slice was
  unfocused looked at the obsolete July recordings; retracted.
- ~~DP3 vs DPPO codebase~~ **RESOLVED: DPPO** (sim: stable 0.76 vs DP3's unstable curve
  plateauing ~0.53; real: 0.60 vs 0.52 — see final report §6b). DP3's real-delta arm
  (0.48 ever → 0.01 success, hold-drift) doubles as the empirical proof of abs-over-delta.
- Residual ~0.08 original-hwo vs fresh-collection gap: run variance vs the genesis
  submodule bump — only matters if it reappears.

## Roadmap / TODO

### PROPOSED (user, 2026-08-27) — hover-start demos to teach grasp RETRY

Add a fraction of demos that START from a post-failure-like state instead of from home: the
gripper **6-10 cm above the object with the WIDTH RANDOMIZED**, then execute the normal
synthesized grasp. Rationale: BC only sees states on the expert trajectory, so after a failed
grasp the policy is off-distribution and has never seen "hovering over the object, gripper
partly closed, holding nothing" — exactly where a failure leaves it.

**Why it should help, precisely:** the normal approach ALREADY passes through 6-10 cm above the
object, but always with the gripper **OPEN**. The novel, genuinely-unseen state is the
**partially-closed gripper at hover** — so the randomized WIDTH is where the value is, not the
height. In absolute-action mode the policy commands a target pose+width directly, so it can
recover in a single step once it has seen the state.

**Three design caveats:**
1. **Object pose mismatch.** After a real failed grasp the object is usually displaced/rotated by
   the failed contact. Hover-starts over a PRISTINE spawn pose under-cover the true post-failure
   distribution — consider perturbing the object slightly for these episodes.
2. **This teaches RESTART, not RETRY.** There is no failure *detection*: the policy learns "from
   this state, grasp", not "that attempt failed, so reopen and retry". Probably sufficient (the
   state is the trigger), but do not claim learned failure-recovery from it alone.
3. **Mixing fraction.** Too many mid-air starts under-represent the full approach and could
   degrade the primary behaviour. Start around **15 %**, not half.

**Verification must be at TRAINING time, not collection time** — the collector will happily
produce these demos and their success rate says nothing about whether retry emerges. The test is
a deploy/eval comparison against a matched no-hover-start baseline.

Related: this is a cheaper cousin of the long-standing "deliberate induced failure for
retry-coverage" idea in CLAUDE.md (v2 collector brainstorm item 3), which perturbs the grasp so a
genuine failure+recovery is recorded. That one covers failure DETECTION too, at the cost of
needing slip detection in the collector.


### Standing rule: rigorous monitoring on EVERYTHING launched (2026-08-24, user)

Anything launched (training, eval, collection, chain, probe) MUST carry a rigorous health
monitor, and problems MUST be solved as they appear — detection or after-the-fact
reporting alone is a failure. The full stack, at launch time (not retrofitted):
1. per-line SLURM failure alerts (FAILED/TIMEOUT/OOM) with immediate handling —
   root-cause from `logs/slumr_logs/<jid>.{out,err}`, then resubmit the same recipe;
2. startup-survival check (~45 s) after every sbatch — and vigilance for delayed failure
   modes (first validation pass, first checkpoint);
3. periodic progress sweep: trainings emit new epoch lines, chains pass each stage
   (ABORT/WARNING markers in their logs), eval sweeps produce summaries;
4. follow-up automations (eval-sweep monitors, timeout-retry watchers) armed together
   with the launch;
5. the acknowledgement of any failure already contains the fix (retry job id or
   root-cause commit).
Context: two escapes motivated this — a training that died at init while its detector's
alerts went to an unread file, and a val-pass crash minutes after a clean startup (see the
2026-08-24 monitoring post-mortem in the Log).

### Work allocation & sequencing (2026-08-23, user)

| # | item | owner | status / sequencing |
|---|---|---|---|
| 1 | Real data: 3 cm cube placed right below the arm; analyze sim-vs-real data difference | **local agent** | **DONE** — probe recorded, sim twin generated, gap decomposed (~9 mm perception x-bias + placement offset; [report](item1_cube3_simreal_gap.md)) |
| 2 | Real-vs-sim demo analysis (trajectory character, speed, grasp speed, grasp width, …); match the scripted demo to real properties → better co-training | **local agent** | FIRST (with 1). Slow/pausing trajectories are fine when the derivation carries lead — qjzsf (real-only, slow teleop, K=4 lookahead) works in real; the v6 stall was K=1 with near-zero lead. Just derive slowed sim demos with lookahead (or verify the lead/dwell gates) as done for teleop |
| 3 | afucm real-data-amount ablation {1, 5, 10, 20, 30 demos}, tested in real | cluster agent | SIM DONE — flat across N (peaks 0.635-0.735, seed-noise range; sfpom/wclac/luewz/ibkzr/ordtr in the canonical table). Real ranking = user, today |
| 4 | Sync colleague on the FIXED SETUP: native 7d euler action · arm-focus cloud · quat proprio · realws DR · DPPO codebase · ×3.5 big net (512 + [1024]³, 2.89 M EMA) | user | next working day. (Note: "native 7d" — recording native euler commands is bit-equivalent to the validated 10d-record + `--derive-source-action` path, ~1e-7; either satisfies the setup) |
| 5 | Reduce demo occlusion (penalty or hard angle constraint) | **local agent** | mechanism ALREADY BUILT + validated (hard azimuth bound 45°, `v5c`; fully-hidden 24%→4%, soft penalty provably inert — conclusion 10); remaining work = integrate into the post-item-2 synthesis version once 2 is confirmed |
| 6 | More mushroom variants closer to real shapes | user (mesh prep) | DONE — in production (v3.3 campaign collects on the 4-mesh pool). MESHES IN ASSETS (2026-08-24): 3 TripoSG real-mushroom scans (`obj_meshes/mushroom{1,2,3}/clean.obj`) normalized to the nominal mushroom's convention — rotated y-up→z-up (cap +z, stem −z), uniformly scaled to the nominal mean extent (~33 mm), origin at xy bbox center / 42.7% above bottom — written to `gentle_manip/assets/objects/mushroom{1,2,3}.obj` + registered in `assets/registry.py` (same mushroom material/spawn, sizes 32.3×32.2×35.1 / 31.8×31.9×35.9 / 35.7×32.6×31.3 mm). Side-by-side check: `docs/figures/mushroom_variants_2026-08-24.png`. Remaining = rerun the collection pipeline over the variant set (multi-mesh DR or per-mesh tasks) |
| 7 | OOD size+shape test scenario | cluster agent | SIM DONE — ASYMMETRIC: big-OOD (1.5-1.8×) easier than in-domain (0.75-0.92); small-OOD (0.7-0.95×) collapses (0.13-0.22) for every policy. ACTION: extend training scale DR downward if real mushrooms can be < nominal |
| 8 | Robustness to missed grasps | **local agent** | DEFERRED until everything else checks out. Partially exists: retry-on-slip (`--retry-max`, window 0.15–0.30, 5/5 recovery) is built + validated; open remainder = induced-failure coverage (idea 3) |
| 9 | Promote to generalist (end-to-end) | cluster agent | STARTED (2026-08-25/26): SECOND CATEGORY = 3cm tofu block — task/experiments live, smoke series v3→v10 drove demonstrator 63.5%→97.6% (spawn-z, anti-pinch, peak+tilt terms); v11 (grasp-quality polish) in queue; 650+trainings gated on user video OK |
| 10 | Gentler grasp test (small/no over-squeeze, + real co-train) | cluster agent | collection DONE (26-08-23-mfa, 2.5mm extra squeeze, 91.55% demonstrator — squeeze dose-response: 2.5mm→84.9%, 5mm→91.6%, 7.5mm+→94.2%); co-train prmaw DONE — NEGATIVE: 0.54 peak @200, sustained 25.1 kPa (0.15 success cost, NO stress benefit vs afucm 0.685/24.0). Gentler demos do not yield a gentler policy at this margin |
| 11 | Gentleness-aware model selection | (either) | from now on; harness already records the 9 stress metrics per episode. Recommend ranking on sustained (`top20_ttop20`) not peak — peak is pinned 49–53 kPa across 9 demonstrator configs (likely contact/metric artifact, conclusion 11) |
| 12 | Memory in the policy (first-frame context token variant) | cluster agent | SIM NEGATIVE (ptpii 0.38 peak vs 0.685 baseline — success halved). NOT taken to real. Follow-ups if revisited: bottlenecked context, FiLM, gating; RNN/transformer untested |
| 13 | Aux objectives on the real-data co-train | cluster agent | SIM: no reliable success gain — seed spread 0.55/0.60/0.70 (dfyqx/wffpe/uknld); uknld's gentle profile was partly seed luck. Not adopted; gentleness-vs-seed check via stress columns pending |
| 14 | Camera-pose DR (slight: ~0.5 cm/axis, 1–5°) | cluster agent | SIM DONE (jtzqc 0.57 @400 vs 0.685 — mild expected robustness cost). REAL-TEST CANDIDATE (its whole point is extrinsics drift) |
| 15 | DP horizon ablation: predict 8 / execute 4 | cluster agent | SIM DONE — h8/e4 alone FAILS (jjjjy 0.05; e8 diagnostic 0.20), but hold-tail data RESCUES it (ymbve 0.68). Verdict: keep 4/4 default; h8 viable only with stay-still tails |
| 17 | **Bias-corrected real dataset as the default going forward** (user, 2026-08-25): the real demos carry the rig's ~9 mm perception bias baked into their stored clouds, while sim clouds are unbiased — so co-training currently mixes two halves that disagree by ~9 mm. Build the corrected variant with `gentle_manip/scripts/shift_demo_clouds.py` (`--shift 0.009 0 0` → `single_lift_mushroom_real_merged_shift9mm`; proprio/actions/zero-padding untouched, provenance recorded), keep BOTH variants, and use the corrected one for later trainings. **⚠ Pairing rule:** trained-on-corrected must deploy with `point_cloud_shift` ACTIVE, trained-on-uncorrected with it at ZERO — a mismatch silently reintroduces the bias. **Every run must record which variant it used** in EXPERIMENT.md + the experiments.csv description, so the two families stay distinguishable. Fold into the v33 re-convert (same pass). Open: the true bias may be ~12–13 mm — one measure-shift iteration on the cube3 pair would pin it first | cluster agent | with the v33 re-convert ([v33_real_slice_bug.md §4.1](v33_real_slice_bug.md)) |
| 16 | **Paired-feature encoder regularization**: add a consistency term to the BC loss pulling the encoder features of PAIRED real/sim steps together (e.g. L2/cosine between the PointNet features of real step t and its sim-twin step t), using the paired cube3 datasets below (and any future real recording — the twin generator works on any run). Hypothesis: aligning the visual representation across domains improves sim2real beyond raw co-training | cluster agent | SIM POSITIVE — w=0.5 (alzey) 0.785 @200 (+0.10 over baseline), w=0.1 (vexvd) 0.715; heavier alignment better. TOP real-test candidate: the hypothesis IS real transfer |
| 17 | **Small-object failure investigation & fix**: failures are monotonically size-dependent (in-domain scale 1.0-1.125: 0.32-0.48 ever vs 0.80-0.90 at 1.25+; small-OOD 0.7-0.95 collapses to 0.13-0.22; thin/low-axis_scale worst) — determine whether the policy learned width adaptation (demos DO adapt: min-width spread 26-45 mm) via the width-probe correlation (commanded width vs obj_scale), then fix accordingly: weak correlation → data-side (extend scale DR below 1.0, oversample small); strong correlation but still failing → perception-side (small objects underrepresented in the 1024-pt cloud; consider object-region point budget) | cluster agent | VERDICT (refined after per-bin analysis): on SUCCESSFUL episodes the policy grips a NEAR-CONSTANT ~27-29 mm at every scale (= the data's small-object commanded width; data range 26-42 mm adaptive) — width adaptation essentially not learned. Small-object FAILURES show commanded width collapsing to 11-15 mm = closing on air after a POSITIONING miss (symptom, not cause). Root cause leans APPROACH/CENTERING PRECISION on small targets (less margin, fewer of the 1024 cloud points), width mis-adaptation secondary. Fixes reranked: (1) small-scale DR extension + oversampling, (2) object-region point budget / perception, (3) item-18 width head (still useful: forces size-awareness into the encoder, aiding centering too). Full per-bin numbers in the 2026-08-24 width-probe Log entry. METRIC REFINED (2026-08-24, user): episode-MIN commanded width is inflated by closing-on-air after misses (small scales get artificially tiny mins → spurious positive corr); the honest metric is width AT the grasp→lift transition (EE-z min, first frame risen >2 cm — scratchpad width_at_grasp.py). At-grasp corr: afucm −0.04 (vs 0.27 min-based), prmaw 0.24 (vs 0.44) — baseline width adaptation is ABSENT, not weak |
| 18 | **Aux grasp-width prediction head**: regression head (mirroring the aux_object_pos machinery) predicting the episode's GRASP WIDTH — defined as the min achieved gripper width of the episode (per-episode-constant label, computable from the dataset's own states at convert/merge time; no external join, and REAL demos carry it too, so no masking needed unlike aux_object_pos). Forces the encoder to extract object size from the cloud at every step — directly targeting item 17's finding that width adaptation is under-learned (0.27-0.44 vs 0.85). Alternative label if revisited: the demonstrator's CMA-ES-synthesized width (dr_params width_mm; cleaner intent signal but needs an attempt→episode join). Success metric: policy corr(cmd width, scale) recovers toward 0.85 AND small-object (1.0-1.125) ever-success rises from 0.32-0.48 | cluster agent | SIM DONE — NO success gain: w=0.5 (dgvmu) 0.610 @100, w=2.0 (eqrth) 0.555 @200 and degrading with epochs, vs afucm 0.685. The head itself CONVERGES (loss_grasp_width 0.0061) — encoder can extract size, but the aux loss alone doesn't alter the executor → 18b (feed the prediction into conditioning) launched as iteration 2. Width-probe rerun DONE: dgvmu at-grasp corr −0.02, eqrth +0.30 (+0.51 succ-only), pyzpl +0.04; dgvmu small-half ever 0.67 vs afucm 0.57 — full read in the 2026-08-24 width-at-grasp Log entry + `docs/figures/width_at_grasp_2026-08-24.png`. ATTEMPT 3 (data-side, user 2026-08-24): extend size support to [0.8, 1.5] — 120 complement demos collected at the below-nominal band [0.8, 1.0) (collection 1651703, experiment `..._realws_smallband`), appended to the afucm dataset → `single_lift_mushroom_simreal_realws_noos_s08_cmd`; 3 arms: plain afucm recipe, aux w=1.5, aux w=2.5, all evaluated in-domain on the WIDENED [0.8, 1.5] range (experiment `..._7d_realws_s08`) — note s08 eval numbers are NOT directly comparable to the [1.0, 1.5] table rows; the 3 arms compare against their own s08_plain baseline. DONE (2026-08-26): on the WIDENED [0.8,1.5] eval (own baseline, NOT comparable to the [1.0,1.5] rows): peikp plain best 0.585/0.630 @400 sust 22.9 (stable curve); rturn aux1.5 0.560 @100 then COLLAPSES 0.18-0.23; dxrxd aux2.5 0.525 @100, 0.45-0.52 band. VERDICTS: (a) DATA WORKS — peikp small-scale (0.8-0.95) ever 0.45 vs 0.13-0.22 OOD collapse pre-complement (>2x), though the size gradient persists (0.45/0.66/0.82 across bins — centering hypothesis stands); (b) width-aux again NO gain and destabilizing at w>=1.5 on this data. Plain + small data is the item-18 attempt-3 winner |
| 18b | **Planned-width feed-forward** (user proposal): the width head's own DETACHED prediction appended to the denoiser conditioning — a learned grasp-width planner feeding a conditioned executor (prediction fed in training too: no exposure bias; stop-grad: head calibrated only by its aux loss; object visible from step 0 → plan committed pre-occlusion, delivering what item 12 attempted). IMPLEMENTED flag-gated (`+model.network.feed_width_pred=true`, requires aux_grasp_width) | cluster agent | LAUNCHED 2026-08-24 (job 1650036, w0.5 recipe + feed_width_pred) — gate met: dgvmu's head converged (loss_grasp_width 0.0061) while success stayed flat/lower → the aux loss alone doesn't change the executor; 18b feeds the prediction in explicitly. Eval watcher armed |

Sequencing summary: **1+2 first (local)** · 3 ongoing · 4 immediately next working day (user) ·
5 after 2 confirms · 6 mesh-prep parallel (user) · 7 parallel (cluster) · 10 after 2+5 (or on
afucm now) · 8 deferred · 9 last · 11 from now on · 12/13/14/15 on afucm in parallel (cluster).
Local agent starts items only on explicit user go.

### Maybe-look-later items (flagged at plan review, 2026-08-23 — not scheduled)

- **Pre-flight dwell gate in a repo script**: fold the dwell-fraction check (frac(|ΔA|<0.01)
  ≲ 10%, conclusion 8) into `verify_derived_dataset.py` alongside the existing seam + lead
  gates. Cheap insurance for item-2 recollects (though K-lookahead derivation already
  handles slow sources — see qjzsf — so this is a verification, not a blocker).
- **FEM-vs-MPM hold-margin calibration**: why do honest (no-over-squeeze) widths slip in MPM
  when the FEM says holdable (conclusion 11)? Unowned; the enabler for item 10's "no
  over-squeeze" goal — without it, expect demonstrator success to crater.
- **Peak-stress investigation**: per-episode PEAK stress is pinned at 49–53 kPa across nine
  demonstrator configs while sustained/bulk move ±30% — needs a per-step stress trace to
  locate the worst instant; decides whether peak is a real signal or a contact/metric
  artifact (i.e. whether item 11 may ever rank on it).
- **Real-world gentleness validation**: every gentleness number is sim-measured; a real
  bruising/quality check (could fold into item 10's real test) closes the loop on the
  project's actual mission.
- Bookkeeping: `single_lift_mushroom_real_abs_7d` experiment config (deploy-side leftover,
  non-blocking — deploy composes from obs/action files directly).


**Evaluation & analysis**
- [x] **OOD generalization test** — DONE (sim): big-OOD easy (0.75-0.92), small-OOD
  collapses (0.13-0.22) → generalization is asymmetric in object size. NEW ACTION ITEM:
  extend training scale DR below 1.0 (e.g. [0.8, 1.5]) before real deployment on small
  specimens. Zero-shot other categories (tofu/fruit) still open (generalist preview).
- [ ] **Real-vs-sim demo trajectory analysis**: compare the scripted demonstrator's
  kinematics (speeds, pauses, approach angles, grasp-close timing) against the real human
  teleop demos; match the scripted trajectory properties to human execution where they
  differ (a likely sim2real lever on the DATA side).
- [ ] **Gentleness-aware model selection**: the harness already records per-episode stress
  — rank checkpoints by success AND stress percentiles, not success alone (the mission is
  gentle manipulation, not just lift success).

**Demonstrator improvements**
- [x] **Occlusion-aware grasp synthesis** — DONE (local agent): hard azimuth bound
  `cam_azimuth_max_deg=45` (grasp_profiles `v5c`); a soft weight cannot work (flat
  infeasibility floor). Ground truth: fully-hidden 24% → 4%. See conclusion 10.
- [x] Failure+recovery demonstrations — DONE (local agent): `--retry-max` regrasp-on-slip
  (v4 collector), trigger window measured (0.15–0.30 of the lift), failures kept in the
  demos. Induced-failure (idea 3) still open. See conclusion 10.

**Policy / architecture**
- [ ] (deprioritized: DPPO chosen) **DP horizon ablation**: prediction horizon 8 with execution steps 4 (current DPPO:
  4/4; DP3 default: 16 predict / 8 execute) — re-planning at half-horizon is the standard
  diffusion-policy sweet spot.
- [x] **Memory in the policy** — first-frame context token variant TESTED, NEGATIVE in
  sim (0.38 vs 0.685; not taken to real). RNN/temporal-transformer variants remain
  untested; if revisited, try a bottlenecked context or FiLM modulation first.
- [x] Aux objectives through co-training (masked to sim rows) — TESTED, no reliable
  success gain (seed spread 0.55-0.70); not adopted. Real object-pos labels feeding the
  same mask remain an option if gentleness analysis favors the aux head.

**Scaling toward the generalist**
- [ ] Multi-category collections (tofu/fruit meshes + per-category material presets from
  `assets/materials.py`); category-mixed training; measure specialist-vs-generalist gap.
- [ ] Data-scale study (650 → 2k episodes; collection is cheap at ~6 h/650 and
  deterministic).
- [x] Camera-pose DR — TESTED in sim (jtzqc, 0.57 vs 0.685 mild cost); real-test
  candidate — its value only shows under real extrinsics drift.
- [ ] **Real-recording cloud provenance**: record-time processing is BAKED into real pkls
  (only the final 1024-pt cloud is stored) — a run's experiment/env obs snapshot does NOT
  describe it, which nearly caused a wrong deploy obs config for qjzsf. Store the pre-
  subsample cropped cloud (or raw depth) in future recordings for filter-agnostic
  conversion, and always read the RECORDING's own config.yaml when choosing deploy obs.

**Bookkeeping**
- [x] Deploy-script entries for the shortlist: `afucm/state_400`, `nmbtz/state_500` and
  `qjzsf/state_1000` all in `deploy_real.sh` (local agent, 2026-08-23), each wiring-verified
  and load-smoked. Obs configs differ deliberately: armfocus for afucm/nmbtz (their training
  cloud), plain outlier for qjzsf (its superset_real obs has no object_focus — verified
  value-identical). afucm-vs-nmbtz on the same rig answers the "does real co-training help
  in real?" open question. Remaining: `single_lift_mushroom_real_abs_7d` experiment config
  (not blocking — deploy composes from obs/action files directly).
  (was: `afucm/state_400` added to `deploy_real.sh`
  (local agent, 2026-08-23) — wiring verified (armfocus obs config, euler offset,
  rate-limit clamp active at the RealBackend, big-net auto-load, load-smoked). Remaining:
  the other shortlist checkpoints + the `single_lift_mushroom_real_abs_7d` experiment
  config (deploy_real_dppo composes from obs/action files directly, so not blocking).
- [ ] Port the pre-flight dataset gates (seam + lead) from the scratchpad chain scripts
  into a repo script (`gentle_manip/scripts/verify_derived_dataset.py`).

---

## Canonical results table (all runs, standard presentation — updated 2026-08-24, complete)

Format (user-mandated): run · log location · best ckpt (by success) · success / ever /
in-band · sustained stress (`top20_ttop20`, kPa) · peak stress (kPa; pinned ~50-57 —
sustained is the discriminating gentleness metric) · remark. Canonical harness, 200 eps.
Mushroom yield reference: 40 kPa. **Backfill rule (2026-08-24, user): EVERY DEVLOG edit
includes a pass over placeholders — "(running)", "(pending)", "(curve filling)", "evals
running" — replacing any whose experiments have since finished.**

| run | log location (logs/dppo/dppo-pretrain/) | best ckpt | succ | ever | in-band | sustained kPa | peak kPa | remark |
|---|---|---|---|---|---|---|---|---|
| jfhlu | single_lift_mushroom_soft_abs_pcd_hwo/jfhlu | 200 | 0.880 | 0.890 | 0.895 | 28.8 | 52.8 | REFERENCE: 10d rot6d abs, recorded commanded, hwo demos/cloud |
| eibno | single_lift_mushroom_soft_hwo_7d_cmd/eibno | 100 | 0.840 | 0.860 | 0.865 | 29.1 | 53.6 | Part C: 7d euler vs jfhlu — encoding cost ~0.04 |
| khxdo | single_lift_mushroom_soft_hwo_repro_cmd/khxdo | 300 | 0.760 | 0.780 | 0.810 | 32.5 | 53.9 | R1: fresh hwo reproduction (recipe-not-luck control) |
| vdmtb | single_lift_mushroom_soft_armfocus_firm_cmd/vdmtb | 200 | 0.760 | 0.800 | 0.805 | 24.1 | 51.4 | R2: armfocus cloud + v3 firm recipe (cloud cost ~0) |
| wicfr | single_lift_mushroom_soft_hwo_armfocus_abs_cmd/wicfr | 100 | 0.700 | 0.780 | 0.795 | 29.7 | 53.8 | Part B abs s43 (xhk v2-recipe collection) |
| igjmd | single_lift_mushroom_soft_hwo_armfocus_abs_cmd/igjmd | 300 | 0.620 | 0.625 | 0.635 | 26.4 | 52.2 | Part B abs s42 (commanded-derivation validation arm) |
| hrqdm | single_lift_mushroom_soft_hwo_armfocus_delta/hrqdm | 100 | 0.745 | 0.770 | 0.775 | 31.2 | 54.2 | Part B delta s43 (epoch-collapse pathology) |
| uzgjm | single_lift_mushroom_soft_hwo_armfocus_delta/uzgjm | 100 | 0.625 | 0.650 | 0.660 | 27.9 | 53.8 | Part B delta s42 (epoch-collapse pathology) |
| wyigy | single_lift_mushroom_simreal_armfocus_noos_cmd/wyigy | 100 | 0.785 | 0.835 | 0.865 | 31.3 | 53.7 | co-train std box, 8% real, s42 |
| zgwyi | single_lift_mushroom_simreal_armfocus_noos_cmd/zgwyi | 200 | 0.760 | 0.810 | 0.820 | 27.3 | 52.4 | co-train std box, 8% real, s43 |
| fbeoe | single_lift_mushroom_simreal_armfocus_cmd/fbeoe | 200 | 0.745 | 0.805 | 0.825 | 27.8 | 52.2 | co-train std box, 25% real (x4), s42 |
| gmxsx | single_lift_mushroom_simreal_armfocus_cmd/gmxsx | 300 | 0.650 | 0.730 | 0.770 | 22.3 | 51.3 | co-train std box, 25% real (x4), s43 |
| nmbtz | single_lift_mushroom_soft_armfocus_realws_cmd/nmbtz | 500 | 0.710 | 0.720 | 0.730 | 30.8 | 53.4 | pure-sim REALWS box (N=0 endpoint) |
| afucm | single_lift_mushroom_simreal_realws_noos_cmd/afucm | 400 | 0.685 | 0.720 | 0.740 | 24.0 | 51.5 | BASELINE: realws co-train 8% real (N=50); real-tested ~75% |
| jbtmt | single_lift_mushroom_simreal_realws_noos_cmd/jbtmt | 400 | 0.585 | 0.720 | 0.765 | 21.2 | 50.4 | afucm seed twin (s43) |
| yrwdd | single_lift_mushroom_simreal_realws_cmd/yrwdd | 200 | 0.650 | 0.735 | 0.755 | 25.7 | 51.7 | realws co-train 25% real (x4), s42 |
| eswpt | single_lift_mushroom_simreal_realws_cmd/eswpt | 300 | 0.230 | 0.580 | 0.700 | 24.7 | 52.0 | realws co-train 25% real (x4), s43 — outlier seed |
| qjzsf | single_lift_mushroom_real_abs_cmd/qjzsf | 1000 | 0.600 | 0.610 | 0.610 | 37.4 | 56.3 | Part A: real-only DPPO 7d abs (commanded+K4) |
| sfpom | single_lift_mushroom_simreal_realws_n1_cmd/sfpom | 500 | 0.695 | 0.710 | 0.715 | 26.4 | 52.6 | realN ablation: 1 real demo |
| wclac | single_lift_mushroom_simreal_realws_n5_cmd/wclac | 300 | 0.735 | 0.740 | 0.755 | 28.4 | 53.3 | realN ablation: 5 real demos |
| luewz | single_lift_mushroom_simreal_realws_n10_cmd/luewz | 500 | 0.670 | 0.695 | 0.720 | 24.8 | 52.0 | realN ablation: 10 real demos |
| ibkzr | single_lift_mushroom_simreal_realws_n20_cmd/ibkzr | 400 | 0.645 | 0.725 | 0.725 | 21.1 | 51.5 | realN ablation: 20 real demos |
| ordtr | single_lift_mushroom_simreal_realws_n30_cmd/ordtr | 300 | 0.635 | 0.675 | 0.700 | 24.7 | 51.4 | realN ablation: 30 real demos |
| alzey | single_lift_mushroom_simreal_realws_noos_cmd/alzey | 200 | 0.785 | 0.800 | 0.805 | 33.9 | 54.4 | item 16: paired-feature reg w=0.5 (+0.10 sim) |
| vexvd | single_lift_mushroom_simreal_realws_noos_cmd/vexvd | 300 | 0.715 | 0.755 | 0.795 | 23.5 | 51.2 | item 16: paired-feature reg w=0.1 |
| uknld | single_lift_mushroom_simreal_realws_aux_cmd/uknld | 400 | 0.705 | 0.735 | 0.760 | 23.9 | 51.0 | item 13: masked aux s42 (best seed of 3) |
| wffpe | single_lift_mushroom_simreal_realws_aux_cmd/wffpe | 100 | 0.605 | 0.715 | 0.735 | 37.3 | 55.6 | item 13: masked aux s43 (seed check) |
| dfyqx | single_lift_mushroom_simreal_realws_aux_cmd/dfyqx | 100 | 0.545 | 0.675 | 0.700 | 26.7 | 53.0 | item 13: masked aux s44 (seed check) |
| jtzqc | single_lift_mushroom_simreal_realws_noos_cmd/jtzqc | 400 | 0.575 | 0.580 | 0.590 | 32.5 | 53.9 | item 14: camera-pose DR (real-test candidate) |
| ptpii | single_lift_mushroom_simreal_realws_noos_cmd/ptpii | 200 | 0.380 | 0.410 | 0.460 | 22.4 | 50.3 | item 12: first-frame context — NEGATIVE |
| jjjjy | single_lift_mushroom_simreal_realws_noos_cmd/jjjjy | 600 | 0.055 | 0.615 | 0.745 | 28.1 | 53.4 | item 15: h8/e4 — FAILED config (e8 diag 0.20) |
| prmaw | single_lift_mushroom_simreal_gentle_realws_cmd/prmaw | 200 | 0.540 | 0.600 | 0.625 | 25.1 | 51.3 | item 10: gentle demos co-train — NEGATIVE (no stress benefit) |
| kiouk | single_lift_mushroom_simreal_realws_n5_ht_cmd/kiouk | 100 | 0.640 | 0.675 | 0.690 | 24.6 | 52.5 | hold-tail on wclac setup, s42 |
| ynfhn | single_lift_mushroom_simreal_realws_n5_ht_cmd/ynfhn | 600 | 0.715 | 0.745 | 0.760 | 20.5 | 49.8 | hold-tail on wclac setup, s43 |
| wberw | single_lift_mushroom_simreal_realws_noos_ht_cmd/wberw | 100 | 0.530 | 0.685 | 0.735 | 25.8 | 53.4 | hold-tail on afucm setup, s42 (mildly negative) |
| cutkl | single_lift_mushroom_simreal_realws_noos_ht_cmd/cutkl | 100 | 0.490 | 0.715 | 0.730 | 25.6 | 52.5 | hold-tail on afucm setup, s43 (mildly negative) |
| ymbve | single_lift_mushroom_simreal_realws_noos_ht_cmd/ymbve | 600 | 0.675 | 0.745 | 0.785 | 23.9 | 51.4 | hold-tail + h8/e4, s42 — RESCUES h8 (0.04→0.68) |
| udvpq | single_lift_mushroom_simreal_realws_noos_ht_cmd/udvpq | 100 | 0.670 | 0.710 | 0.725 | 30.6 | 54.0 | hold-tail + h8/e4, s43 — rescue replicates (0.67) |
| njhbz | single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9/njhbz | 300 | 0.805 | 0.820 | 0.825 | 28.1 | 52.9 | v3.3 + shift9 clouds, PLAIN — curve PEAKS then decays (0.81@300 -> 0.66@600); real-obs PASS; width flat 0.083, small/big 0.80/0.97 |
| mqlxj | single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9/mqlxj | 400 | 0.770 | 0.800 | 0.815 | 25.4 | 52.1 | shift9 + paired-reg, seed 42 — real-obs PASS |
| avfnp | single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9/avfnp | 400 | 0.830 | 0.835 | 0.835 | 28.8 | 53.0 | shift9 + paired-reg, seed 27 — BEST sim; small-object gap ELIMINATED (0.90/0.90); width corr 0.295; real-obs fails only the OOD hybrid row |
| lulkx | single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9/lulkx | 600 | 0.820 | 0.865 | 0.865 | 28.1 | 53.1 | shift9 + paired-reg, seed 43 — PLATEAU 0.81-0.82 across ckpts 300-600 (no peak-hunting needed); real-obs PASS; small/big 0.83/0.93 |
| dgvmu | single_lift_mushroom_simreal_realws_noos_cmd/dgvmu | 100 | 0.610 | 0.725 | 0.750 | 30.7 | 54.1 | item 18 aux width head w=0.5 — head converges (loss 0.0061) but NO success gain vs afucm 0.685; NEGATIVE alone |
| eqrth | single_lift_mushroom_simreal_realws_noos_cmd/eqrth | 200 | 0.555 | 0.650 | 0.675 | 27.6 | 52.2 | item 18 aux width head w=2.0 — worse, degrades with epochs (0.225 @600); heavy aux weight hurts |
| pyzpl | single_lift_mushroom_simreal_realws_noos_cmd/pyzpl | 100 | 0.610 | 0.665 | 0.685 | 28.2 | 53.7 | fix 2 gripper-dim loss ×3 — no gain vs afucm; @300 near-tie 0.600/0.710/0.790 sust 22.7 (gentler pick) |
| tysvo | single_lift_mushroom_simreal_realws_noos_cmd/tysvo | 100 | 0.350 | 0.585 | 0.600 | 44.2 | 56.8 | fix 5 FiLM UNet head — NEGATIVE: collapses after 100 (0.00-0.18, @600 0.060); FiLM conditioning harmful here |
| bcvrt | single_lift_mushroom_simreal_realws_noos_cmd/bcvrt | 100 | 0.675 | 0.720 | 0.740 | 33.8 | 55.1 | 18b planned-width feed-forward (aux0.5+feed) — MATCHES afucm success (no aux-pressure cost) but width still flat (at-grasp corr 0.09); curve declines after 100 |

**OOD evals** (existing checkpoints on out-of-range geometry; separate distribution):

| policy/ckpt | eval | succ | ever | in-band | sustained kPa | peak kPa | remark |
|---|---|---|---|---|---|---|---|
| afucm/400 | ood | 0.920 | 0.925 | 0.925 | 41.2 | 56.8 | BIG (scale 1.5-1.8, shape OOD) — EASIER than in-domain |
| afucm/400 | ood_small | 0.130 | 0.165 | 0.200 | 17.5 | 47.7 | SMALL (scale 0.7-0.95, shape OOD) — COLLAPSE |
| nmbtz/500 | ood | 0.905 | 0.910 | 0.920 | 41.3 | 56.8 | BIG (scale 1.5-1.8, shape OOD) — EASIER than in-domain |
| nmbtz/500 | ood_small | 0.150 | 0.155 | 0.165 | 27.0 | 51.2 | SMALL (scale 0.7-0.95, shape OOD) — COLLAPSE |
| qjzsf/1000 | ood | 0.750 | 0.805 | 0.815 | 45.2 | 56.8 | BIG (scale 1.5-1.8, shape OOD) — EASIER than in-domain |
| qjzsf/1000 | ood_small | 0.220 | 0.240 | 0.245 | 31.8 | 53.0 | SMALL (scale 0.7-0.95, shape OOD) — COLLAPSE |
| vdmtb/200 | ood | 0.915 | 0.915 | 0.920 | 39.8 | 56.8 | BIG (scale 1.5-1.8, shape OOD) — EASIER than in-domain |
| vdmtb/200 | ood_small | 0.180 | 0.205 | 0.220 | 18.8 | 48.2 | SMALL (scale 0.7-0.95, shape OOD) — COLLAPSE |


---

## Log

**2026-09-03 (later) — ADOPTED: plane-corrected D435i extrinsic + z_min 18 mm crop. A hand-eye
fit that is self-consistent can still be TILTED against the world.** After the re-collect
(0.94 mm hand-eye residual, 4/5 solvers agreeing within 2.4 mm) the extrinsic still read the
board at **4.1-13.5 mm across the workspace** against a true 12.3-15.2 mm — a ~9 mm
POSITION-DEPENDENT error from ~1 deg of residual tilt, invisible to every internal metric.

- **Ground truth came from the robot, not the camera:** touching the board with the gripper at
  6 poses fits a plane to **0.31 mm rms** (tilt 0.438 deg, 12.3-15.2 mm). That is what proved the
  board flat and the camera tilted.
- **Correction** = the rigid transform mapping the camera plane onto the touch plane (1.436 deg
  + 5.45 mm at centre). Error vs touch truth: median **-4.20 -> +0.06 mm**, p5/p95 -8.87/-0.03
  -> **-1.43/+1.36 mm**. Cost: hand-eye self-consistency 0.94 -> 1.76 mm — accepted, because the
  touch plane is INDEPENDENT truth from the robot's kinematics while the hand-eye residual only
  measures agreement among the calibration poses. A plane fixes 3 DOF (2 tilt + height); in-plane
  x/y and yaw are unchanged and STILL UNVALIDATED (needs a 3D probe).
- **Crop z_min 18 mm** (real-deploy armfocus configs). Board survival over 10 frames: adopted
  extrinsic @16 mm 0.03%, corrected @16 mm 0.82%, corrected **@18 mm 0.00%**. The old 16 mm only
  worked because the uncorrected cloud read ~4 mm low.
- **`point_cloud_shift` ZEROED** — the +9 mm x was an L515-era correction; the D435i correction
  now lives in `WORLD_T_CAM_EXT`, so leaving it on would double-correct.

**LESSON (reusable): do not compensate a sensor/extrinsic error with geometric domain
randomization.** The proposal on the table was to randomize the sim board thickness 9-14 mm to
"cover" the mismatch. That is the wrong instrument: randomizing sim GEOMETRY moves the object's
true position and its observed position together, so the policy learns to FOLLOW cloud z — which
is correct in sim and wrong in real, where only the observation is displaced. With a
position-dependent error it would grasp correct at near-right and ~10 mm low at far-left. Fix a
measurable systematic error (calibration); use OBSERVATION-level augmentation for the residual
so the policy learns to IGNORE absolute cloud z. This is the DR-vs-augmentation split in
CLAUDE.md, and the board case is the clean worked example.

**2026-09-03 (cable) — HARD object_focus is now the default on every `*armfocus*` config.**
The gripper CABLE is real-only geometry with no sim counterpart. Two things measured on the rig:
- **Raising `min_neighbors` is the WRONG tool and actively backfires.** `remove_outliers_voxel` is
  a DENSITY filter; the cable is a solid connected object CLOSE to the camera, so it is DENSER than
  a small object seen obliquely at range. Cable survives more than the object at every setting —
  at min_nb=200 the test cube was 0% kept while 58% of cable survived. Smaller voxels don't flip it.
- **`r_ee` alone did nothing under SOFT focus.** With `arm_weight` set, the pipeline runs
  `focus_weights` (down-SAMPLES the arm), not `focus_object` (drops it). ~12% of the 1024-point
  budget went to arm/cable at BOTH r_ee 0.13 and 0.11 — the weight, not the radius, is what admits
  them. Removing `arm_weight` -> hard focus -> **0.0% cable, z max 150 mm**, and the object gains
  points (127 -> 144 on the test cube). r_ee 0.11 vs 0.13 then matters (0.13 still leaks 1.0%).

Chosen deliberately over a cable-specific hack: the arm exists in BOTH domains, so dropping the
whole arm is sim2real-clean and removes the cable as a side effect, with no real-only special case.
The fallback if the cable ever reappears is a TCP-FRAME AXIAL cut — measured separation is wide
(object -12..+17 mm along the tool axis, cable -186..-124 mm), but that needs validating across arm
poses, and everything here is ONE pose.

Also corrected: `TCP_API_TO_TCP_OURS_OFFSET` has been `[0,0,0]` for a long time, but its comment
(and CLAUDE.md) still described the abandoned 0.13 m separation as current. Verified on hardware —
`get_ee_pose() == get_position_aa()`, and a gripper-touch reads the board's own height — so the
reported EE frame already sits at the fingers and `r_ee` is measured from the API-native TCP.

**PENDING (must land together):** the sim board is not built yet. The crop is SHARED geometry,
so `superset_*_armfocus.yaml` still carry `crop_min z = 0.004` ON PURPOSE — they must move to
0.018 in the SAME change that adds the ~14 mm board to the sim scene, or sim and real will crop
different amounts. Existing checkpoints were trained at 0.004 and are NOT compatible with the
new deploy configs.

**New viewer:** `gentle_manip/visualization/live_obs_cloud.py` shows the FINAL policy cloud live
through the real deployment path (RealBackend -> RawObs -> PerceptionPipeline), so extrinsic +
crop + outlier filter + object_focus + FPS are all applied. `--no-home` reads the arm read-only
(XArm7Real.connect() otherwise HOMES the arm).

**2026-09-03 — Rig change to D435i + hand-eye calibration: ONE OUTLIER POSE MOVED THE ANSWER
126 mm. New robust selection tool.** Camera swapped L515 -> **D435i** (serial `335522071488`),
lifted and angled ~38 deg down at the table. Re-ran eye-to-hand ChAruco calibration
(`diagnostics/calibration.py`, 11 poses) — and the script's printed result was WRONG.

- The board is clamped in the gripper, so `inv(T_gripper2base_i) @ X @ T_board2cam_i` (the board
  pose in the GRIPPER frame) is a physical constant. Spread across poses = residual; one pose
  far from consensus = outlier. Pose #2 sat at 12.7 mm vs a 5.5 mm median.
- Dropping it: median residual **5.54 -> 1.69 mm**, and the solution MOVED **126 mm / 7.5 deg**.
- **Decisive external check — the table plane.** The table is at z=0 in base coordinates by
  definition, so fit the dominant plane of a live cloud and read its height. All-11 result put
  the table at **+97 mm**; drop-#2 put it at **-1.7 mm**. This is independent of the calibration
  data and is the check to trust. Caveat learned the hard way: the ChAruco board was still in the
  gripper filling the near field, so the FIRST plane fit locked onto the board (80 deg tilt) and
  looked like both candidates failed — restrict to points beyond ~0.33 m, or clear the scene.
- Root cause is conditioning, not bad luck: median pairwise relative rotation only **14.3 deg**
  (22/55 pairs under 10 deg), rotation axes clustered (anisotropy 4.8), EE height spread just
  **5 cm**. OpenCV itself logs "Not enough informative motions--include larger rotations" on
  subsets of this set. Tsai degrades sharply with small rotations, which is exactly why a single
  bad pose could dominate.

**New: `gentle_manip/diagnostics/calib_select.py`** — searches POSE SUBSETS x all five OpenCV
solvers (TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS) and picks by RANSAC consensus: fit X on a subset,
then count how many of ALL N poses agree with it (`--inlier-mm/--inlier-deg`), tie-break on
median residual. Scoring on held-out poses is what stops it collapsing onto the smallest subset.
Exhaustive when the subset count is small (11 poses = 1024 subsets x 5 methods = 266 distinct
fits in 1.8 s), RANSAC sampling above `--max-solves`. `--table-check` runs the external
table-plane validation on the winner. Validated: it independently rediscovers pose #2 as the sole
outlier without being told.

Re-collect ordered (larger rotations, 45-60 deg board tilt, wider EE height range). The
2026-09-03 numbers + best-estimate matrix are archived in
`dataset/camera_calibration/eye-to-hand/NOTES_2026-09-03_d435i.md` as a fallback. NOT written
into `xarm7_config.py` yet — that still holds the old L515 value, and `real_lab.yaml` still
declares `type: l515`, the old serial, and a +9 mm x bias correction that is INVALID for this
camera.

**2026-09-03 — Camera-only raw point-cloud viewer + L515 `short_range` preset (tooling).**
User wanted to inspect the raw cloud "without doing anything about the robot arm", with the
sensor's short-range mode on. Two pieces:
- `RealSenseCamera(visual_preset=...)` — applied in `start()` via `rs.option.visual_preset` /
  `rs.l500_visual_preset`. Devices without the option (D405) print a note and continue; an
  unknown name raises listing what the device offers. Confirmed available on our L515:
  `automatic / custom / default / low_ambient_light / max_range / no_ambient_light / short_range`.
  `short_range` is the right profile for our <1 m tabletop (fewer flying pixels on close edges).
- `gentle_manip/visualization/raw_cloud_viewer.py` (new) — camera-only live Open3D cloud: no arm
  connection, no `PolicyEnv`, no obs config, no cropping/filtering. Uses the SHARED
  `depth_to_pointcloud`, colours each point with its own pixel, and offers `--frame camera`
  (default, raw sensor frame) or `--frame world` (applies the setup extrinsic — a way to eyeball
  the extrinsic without touching the robot). `point_cloud_viewer.py` (the crop-tuning viewer) got
  the same `--short-range` / `--visual-preset` flags.

  ```bash
  uv run --project envs/deploy python -m gentle_manip.visualization.raw_cloud_viewer --short-range
  ```

  Hardware-verified: preset applies (`rs2_l500_visual_preset=5`), 10-frame grabs at both default
  and `short_range` give ~74 % valid pixels, median depth 0.546 m. Coverage is essentially
  unchanged by the preset on a static scene — the expected benefit is edge/flying-pixel noise,
  not point count, so judge it visually rather than by valid-pixel percentage.

**2026-09-03 — CHECKPOINT SELECTION BY VAL LOSS IS WRONG FOR THESE POLICIES (measured).**
User challenge: "deployment results show overfitting on val doesn't mean much." Tested on the
h4 seed-twin pair (xagzg IS xgwhc continued — identical seed/config, so this is one trajectory
at two epochs):

| checkpoint | val | mean min commanded width | lift-proxy | aborts |
|---|---|---|---|---|
| ckpt 1000 (near val low) | 0.0063 | 57.2 mm (barely closes) | 16/44 (36 %) | 2 |
| ckpt 2250 (2.2x the low, deep overfit) | 0.0138 | **44.8 mm** | **9/19 (47 %)** | 0 |

The MORE overfit checkpoint behaved BETTER. Mechanism: val is MSE on action prediction, and
under multimodal targets (stochastic human close timing) the MSE-optimal prediction is the mode
AVERAGE = hovering; longer training sharpens modes, raising val while producing the decisive
commitment we want. Same root as the initiation finding — the offline metric rewards the
failure mode. CONSEQUENCE: earlier "RECOMMENDED state_XXX (val low)" notes for tiatg/uirro are
WITHDRAWN (corrected in their EXPERIMENT.md); deploy entries now say sweep early/mid/late on the
robot. Caveats: 19 vs 44 episodes, lift-proxy is a heuristic (EE rise >40 mm after first close),
sessions differ. Worth a proper checkpoint-vs-success curve if the paper claims anything here.


**2026-09-03 — Per-episode reset bug: every deploy episode AFTER the first was ruined by a
stale closed-gripper observation. Two causes, both fixed.**
Symptom (user, tiatg500_purereal_pc): episode 1 normal, episodes 2-4 "weird in the first few
steps" then dead (40-47 steps each vs 188). Recorded traces show why:
`ep1 observed grip: 30 80 80 80 80 80 74 69 ... 30` with `commanded: 88 88 88 88 30 30 30 ...`.
1. **`RealBackend.reset()` opened the gripper with `set_gripper_width(..., wait=False)` and
   immediately read the obs**, so t0 still reported the PREVIOUS episode's ~30 mm closed width —
   a state no demo ever starts in. Episode 1 was fine only because the gripper was already open.
   FIX: reset now polls until the gripper reaches the target (3 s bound, warns on timeout).
2. **The `--warmup-steps` hold I added 2026-09-02 froze that stale width** (stay-put = current
   pose INCLUDING gripper), so after the deploy loop's own 4-step "open" override the hold
   commanded 30 mm for the rest of the 16-step chunk and re-closed the gripper; the policy then
   read a mid-grasp state and committed a closing chunk. FIX: the hold now commands pose-hold
   with the gripper WIDE OPEN (verified: pose err 0.0000 mm, gripper 88.0 mm).
   Lesson: the deploy loop ALREADY had start-up overrides (4-step gripper-open, 2-step pose-hold);
   the new adapter-level mask duplicated that mechanism and contradicted it. Check for existing
   warm-up handling before adding another layer.
Note the interaction with long chunks: with horizon/act 16 a single bad first inference is
committed for 16 steps, so reset-state errors are far more damaging than they were at act 4 —
worth stating in the paper's deployment notes.


**2026-09-02 (night) — Real-world main-table baselines trained: MODALITY barely matters,
SIM PRETRAINING matters a lot (offline metric).** Three runs, identical arch (PointNet or ViT
+ [3072]^3 MLP), proprio, horizon 16, batch/lr/schedule; the same 141 real teleop eps:

| policy | best val (16-step chunk) | @epoch | behaviour after low |
|---|---|---|---|
| point cloud, pure real (tiatg) | 0.0113 | 520 | overfits -> 0.0160 @1200 |
| RGB 96x96 ViT, pure real (uirro) | 0.0110 | 320 | overfits -> 0.0175 @1200 |
| **point cloud, SIM-PRETRAINED (xdxvc)** | **0.0073** | 600 | **flat through 800** |

RGB ~= point cloud on held-out loss (0.0110 vs 0.0113, indistinguishable), while sim
pretraining is worth ~35% AND acts as a regularizer (no overfit where both pure-real runs turn
by ep320-520). Figure: examples/sim2real_diagnose/figures/real_baselines_val.png.
⚠ This is an OFFLINE proxy — the real-table numbers are on-robot success/safe rate, and the
RGB run carries two flagged caveats (full-frame vs the cloud's crop+arm-focus prior;
from-scratch ViT rather than ImageNet-pretrained ResNet18). Recommended ckpts: tiatg
state_500, uirro state_300 (both keep 12 ckpts).
⚠ RGB is NOT yet deployable: deploy_real_dppo.py hardcodes the PointNet branch; an RGB deploy
adapter (VisionDiffusionMLP + image obs) is TODO before that row of the real table exists.
Dataset/config additions: convert_demos --images/--image-size (additive), cfg
pre_diffusion_rgb.yaml, dataset dataset/dppo/single_lift_real7_rgb.
Process note (repeat offender): a chain script's `pgrep -f "dppo.train.*<env>"` wait-condition
matched one of the agent's OWN shell wrappers, so the queued run never fired — always use
bracketed patterns (`[d]ppo.train`) in wait/kill conditions.


**2026-09-02 (evening) — LONGER ACTION CHUNKS FIX THE HESITATION (confirmed on the robot).**
xdxvc (cvzth sim-generalist -> real7 finetune, horizon 4 -> 16, deployed with --act-steps 16):
across 6 checkpoints / 10 episodes there were **0 abort waves**, and the commanded width lands
at 22-45 mm — inside the 13-63 mm object-contact band — versus the h4 policies which either
never closed (sim->real finetune h4: min cmd 68-75 mm) or aborted (cotrained h4), and versus
the sim-only policy which closes to 12.6-21 mm (crushing). Deploy comparison table + verdict
figure: examples/sim2real_diagnose/figures/deploy_h16_verdict.png. User confirms ckpt 800 OK.
This validates the close-INITIATION diagnosis: initiation was a minority diffusion mode under
stochastic human close timing, and committing to a sampled chunk (16 steps ~0.5 s) carries the
policy into the close, after which it tracks demos exactly.

| deploy run | eps | mean min commanded width | aborts |
|---|---|---|---|
| cvzth80 (sim-only, h4) | 7 | 21.3 mm (crushes) | 2/7 |
| zdwii91 (cotrained, h4) | 9 | 57.3 mm | 0/9 (but never grasps) |
| xgwhc1000 (real-only, h4) | 39 | 57.2 mm | 2/39 |
| zjdmn280/400 (sim->real ft, h4) | 14 | 67.8-74.5 mm | 0 (never closes) |
| **xdxvc (sim->real ft, h16)** | **10** | **22-45 mm** | **0/10** |

Also added: deploy `--warmup-steps` (default 16) — after each reset the policy still observes
and infers normally but its output is masked to a HOLD-POSE command (absolute mode: the raw
action that maps back to the CURRENT measured pose via invert_absolute_action, verified 0.0 mm
round-trip), riding out the unstable first frames the user reported.

**Real-world main-table baselines launched (user request):** (1) tiatg = PURE-REAL point-cloud
DP, h16, from scratch, generalist arch, 1200 ep — the key baseline; (2) RGB DP, h16, same arch
and schedule with DPPO's ViT image branch (96x96 cam_ext, img_cond_steps 2, from-scratch
encoder for parity with the from-scratch PointNet), queued to start when (1) finishes.
convert_demos gained `--images/--image-size` (additive) and the merged RGB dataset is
dataset/dppo/single_lift_real7_rgb (29,648 train / 3,189 val frames). Deliberate choices
flagged in the cfg header for review: full-frame RGB (the point-cloud branch has a crop +
arm-focus prior, so this is "each modality with its conventional pipeline", not matched
framing) and no ImageNet init.


**2026-09-02 — Real-deploy hesitation ROOT-CAUSED: close-INITIATION is a minority diffusion
mode under stochastic human close timing (not gripper speed, not actuation lag, not only the
sim/real style conflict).** Diagnostic chain (figures in examples/sim2real_diagnose/figures/):
(1) deploy gripper traces (3 policies): cotrained = command-level close/abort waves; sim-only
= decisive deep close (crushes); real-only = smooth but timid. Obs tracks commands -> lag
refuted. (2) cvzth->real finetune (zjdmn, lr 1e-5, 400ep, val 0.0026 flat, NO overfit; ckpt
state_280) still never closes (min cmd 71-88mm). (3) teacher-forced probes: policy matches
demo actions EXACTLY on mid-close states; reaches demo close-onset height at deploy; but at
PRE-onset states only ~1/3-1/5 samples initiate closing (demo: firm 1.0->0.81 chunk). Human
timing stochasticity dilutes per-state initiation probability; receding-horizon re-sampling
(act 4) keeps drawing hover -> hesitation. Sim's scripted deterministic timing masks this.
RECOMMENDED (standard, no data change): longer action chunks (act/horizon 4 -> ~16), the
canonical fix for exactly this; complements (does not replace) the close-style question the
v5 discussion covers. Encoder probe (paired red-cube): PairedReg encoders align sim/real
features well (cos .95-.99 vs .69 unreg) but cotrained alignment dips exactly in the grasp
window. Runs: zjdmn (finetune), paired replay + probes under sim2real_diagnose/figures/.


**2026-09-01 — REAL paired-RGB demo campaign COMPLETE: 7 objects x 20(-21) eps, 141 episodes,
all runs VERIFIED PASS.** Teleop (SpaceMouse pose + Z/X gripper), obs =
`point_cloud_1cam_armfocus_rgb.yaml` (generalist student cloud view + PAIRED in-obs
`image_cam_ext` 480x640 uint8 — idle-trimmed with all channels, so RGB step i == obs step i by
construction), saved actions = 7d euler ABSOLUTE via --record-action-config (matches the cvzth
generalist training config); point_cloud_shift [0.009,0,0] active BY USER INTENT (standing
extrinsic compensation). Dual-purpose: DPPO generalist cotrain (cloud+abs actions) AND pi0.5
VLA baseline (RGB; convert via gentle_manip/pi05/convert_to_lerobot.py). Per-object runs
(dataset/demos/single_lift_<o>_real/<id>, paired videos in <run>/videos_paired/):

| object | run | eps | gripper floor (mm) |
|---|---|---|---|
| mushroom | 26-09-01-xlb | 20 | 31 |
| grape | 26-09-01-ioa | 20 | 16 |
| tomato | 26-09-01-cfw | 20 | 63 |
| padron_pepper | 26-09-01-euq | 20 | 13 |
| cherry_tomato | 26-09-01-ezm | 20 | 24 |
| strawberry | 26-09-01-biu | 20 | 22 |
| tofu | 26-09-01-wbz | 21 | 26 |

Verification = `gentle_manip/scripts/verify_real_demos.py` (pairing, schema, absolute-action
sanity, crop bounds, quat canonicalization, config snapshot, paired-video render). Gripper
floors track object size cleanly (13-63 mm) — good size signal for the generalist. Fixes that
came out of this campaign (committed): record.py rgb_shape plumbing; checklist commands for
pcd preview (delta action config) + collection recipe in deploy_real.sh.


**2026-09-01 (morning) — Overnight COMPLETE: gn1b full 6-object row lands; CGN closed as
unusable-on-stack; visual index built.** gn1b: 26–64 % success, and the WORST gentleness in
the grid on compact fruits (raspberry 6 % sub-yield median 1.24×, cherry 36 %) — a modern
learned 7-DOF planner + close-until-contact execution crushes exactly the objects
gentleness targets; strongest external row for the paper table. CGN probe killed at batch 2
(2/21 synth successes) — reported as integration-blocked, no number claimed. One adapter
crash fixed mid-run (gn1b emitted a left-handed rotation on raspberry → guard + retry, clean
100-ep... 16-ep retry OK). Visual review map: `docs/e1_visual_index.md` + per-object method
contact sheets in `docs/figures/e1_sheets/`. Committed locally (NO push, per user).

**2026-09-01 (overnight) — Learned-planner baselines: GraspNet-1Billion WORKING, Contact-
GraspNet integrated-but-unstable; rigid_v41w_occ round near-complete (occ bound changes
NOTHING on 5/5 so far — the challenger's wins are not occlusion-freedom).**

- User-requested: adapt the two open AnyGrasp-class planners with original code+weights,
  minimal glue. Both built from source (`learned_baselines/SETUP.md` = full recipes + the
  measured pitfalls: conda toolchain hijack, stale shipped .so files, TF≥2.16 custom-op ABI
  break → TF 2.15 venv, OkStatus rename, sm_89 gencode, ptxas-10.1-in-PATH XLA miscompile
  → EMPTY predictions, LD_LIBRARY_PATH cudart mixing → illegal-address aborts).
- **gn1b (`--baseline gn1b`)**: deterministic, 6–9 s/call; 16-ep × 6 smoke queued overnight.
- **cgn (`--baseline cgn`)**: 2019-era pointnet2 TF ops nondeterministically broken on
  RTX 4090+CUDA 12 (identical seeded input → 0/N/abort). Retry ×3 wrapper; 8-ep probe only;
  proper fix = their pinned TF2.5/CUDA11 container (post-deadline).
- **Adapter findings worth the paper**: (a) the classic ~54° tabletop viewpoint yields ZERO
  executable proposals for our workspace (side approaches stab the table with 45 mm fingers
  on 3–4 cm objects) — steep views (77°/90°) required; (b) learned planners emit PRE-SHAPE
  openings — commanded literally they never touch the object; faithful conversion = local
  cross-section at the final slice − 2 mm (close-until-contact semantics). Both mirror the
  GPD width finding: rigid planners do not answer "how far to close".
- Third-party clones/venvs/weights NOT committed (.gitignore); local patches exported to
  `learned_baselines/patches/`.

**2026-08-31 — E1 baseline comparison: GPD integrated as the ESTABLISHED rigid-body planner;
18-run sweep (naive/antipodal/gpd × 6 objects) running. First results: naive < classical <
v4.1 on mushroom; naive = 0 % on strawberry.**

- **GPD (ten Pas et al., IJRR 2017) built + adapted** (`third_party/gpd`, local clone —
  NOT a registered submodule). Build fights (documented for reproduction): const-comparator
  patch in `cloud.h`; system toolchain (`CC=/usr/bin/gcc`, conda compilers hijack CMake);
  conda-libffi link poisoning (sed miniconda3 tokens out of link.txt + link system
  `libffi.so.7`). Local patches: `GRASP_POSE` stdout line per hand; `plot_* = 0` — GPD's PCL
  viewer windows were blocking runs on the user's desktop until manually closed (an earlier
  cfg edit was raced/reverted by a backgrounded task; re-applied with read-back verification).
  Rebuild gotcha: the make target is `gpd_detect_grasps`, not `detect_grasps` (the latter
  silently no-ops).
- **Adapter** (`baseline_synth.gpd_planner`, `--baseline gpd`): privileged dense surface cloud
  (15 k area-weighted pts) → GPD → hand frame [approach, binormal, axis] mapped to our TCP as
  columns [−axis, binormal, approach]; hand-base + depth/2 = pad mid-plane centre; first
  candidate passing the v4.1 geometric validity ladder wins. ~3 s/call. Verified in-collector:
  sensible cap-straddling mushroom grasps, width ≈ aperture − 2 mm.
- **Attempts cap 200 (wrapper-only, v4.1 untouched):** the collector runs until N SUCCESSES,
  so strawberry_naive (0/152) would never have terminated. The wrapper captures v4's live args
  namespace and trips `n_episodes` at 200 attempts → graceful exit, honest stats. Lesson:
  success-count collection loops + near-0 % policies = non-termination; cap attempts for any
  baseline/ablation sweep.
- **Sweep protocol:** two parallel chains (naive+antipodal chain, gpd chain) on matched
  experiment configs, 16-ep target, videos on, watchdog v2 (18 runs). Full table + analysis in
  `docs/paper/synthesis_experiments.md` §4 when done.
- **COMPLETE (same day): full 4×6 grid + mushroom width-swaps done.** Full table + five
  findings in `docs/paper/synthesis_experiments.md` §4 (results + run-dir/video map).
  Headlines: (1) only v4.1 holds success AND sub-yield simultaneously across the set —
  blind baselines either drop (GPD 1.5–25 % everywhere: rigid planners never answer "how
  far to close") or damage (cherry antipodal/rigid at-yield, 38–46 % sub-yield); (2) the
  surrogate CLOSURE transfers: antipodal poses + v4.1 closure = 94.1 % on mushroom
  (= v4.1) at sub-yield stress; (3) honest exceptions: lamp (rigid 100 % > v4.1 57 % —
  area-floor limitation, third confirmation), sphere (tie, gentler), cherry naive slightly
  gentler. One silent run death (raspberry_rigid, coincided with a WiFi/kernel hiccup, no
  OOM/traceback) — retried clean: 100 %. **gpd_v41w × 6 COMPLETE (all 31 runs done):**
  closure lifts GPD everywhere (16→46, 1.5→29, 18→28, 18→23, 25→67, 12→21 %) but raspberry
  shows the limit — closure on bad poses secures them PAST yield (19 % sub-yield, median
  1.10×). Refined finding 3: closure transfers to decent poses; joint pose+closure
  optimization is required to hold success AND gentleness. Full width-swap table in
  synthesis_experiments.md §4.
- **rigid_v41w × 6 COMPLETE (23:20): object-dependent split.** Strong rigid poses + v4.1
  closure WINS on mushroom (100|100), strawberry (100|100 — fixes v4.1's p98 degeneracy
  there: flush poses → healthy 2.5–4 mm commands), lamp (100|100); LOSES gentleness on
  cherry (19 % sub-yield, median 1.17×) and raspberry (62 %, 0.96×) — stress-blind poses
  force the closure to yield. Conclusion: stress-aware POSE selection earns its cost
  exactly on compact fragile fruits; strawberry indicts v4.1's own pose search (limitation
  to state). Full table + caveats in synthesis_experiments.md §4. Occ-bounded re-run
  (rigid_v41w_occ) auto-started. 100-ep confirmatory round designed (feasible ~2 days,
  2 lanes; paired-geometry fix needed — same-seed scene draws diverge once failure paths
  differ; wrapper will pin a dedicated batch-indexed scene-DR stream + `--baseline v41`
  passthrough) — awaiting user decision. Full design doc: `docs/e1_100ep_ablation_design.md`
  (protocol, 7 methods incl. naive−5mm, occ-bound-everywhere, paired-geometry fix,
  implementation deltas, cost, cluster suitability). EXECUTION ON HOLD per user
  (2026-08-31 night) — cluster may run it instead of local.
- **Follow-ups (same day):** `docs/paper/related_work_synthesis.md` written (parallel-jaw-only
  related-work map: rigid pose planners / grip-force control / deformable squeeze precedents,
  with the positioning claim); `method_v4.md` re-audited line-by-line against the frozen v4.1
  code (A.9/A.10 exact — note the 2.0/2.5 mm FIRM_* module constants are v3 leftovers
  overridden by `_firm_base=0.0` at runtime; B.3 marked complete, B.5 rewritten to cite the
  finished E1/B2 grid). **rigid_v41w × 6 chain launched** (strong rigid re-ranker poses +
  v4.1 FEM closure) — the sharpest "rigid planner + our width" test.

**2026-08-31 — PROBE ANSWER: yes, a hard min-contact-area floor helps the lamp. 57.1 % -> 72.7 %
at 100 % sub-yield, zero synthesis failures. OFF-RECIPE — the frozen v4.1 recipe is unchanged;
this is a post-deadline v4.x candidate and paper-analysis material.**

User question: "would raising the min contact area help the lamp?" Probe: lamp_mush, 16 eps,
frozen recipe EXCEPT `--grasp-area-min-mm2 50` (hard) instead of `auto`
(run `single_lift_prim_lamp_mush_soft/26-08-31-qpo`, clearly labelled ANALYSIS PROBE):

| lamp_mush | success | sub-yield | median stress | synth failures |
|---|---|---|---|---|
| frozen recipe (auto floor) | 57.1 % | 100 % | 0.61x | 0 |
| hard floor 50 mm2 | **72.7 %** | 100 % | 0.44x | **0** |

The floor pruned the thin-pad neck/edge poses (failure min_pad 64 vs success 91 mm2) at zero
feasibility cost — the on-record prediction ("modest at best, may even drop; pools may collapse
to fallbacks") was too pessimistic; the user's instinct was right. The residual failures remain
TILT-separated (0 vs 10 deg), which an area floor cannot bound — that is the 72.7 -> 100 gap and
would need the (weight-0) w_tilt term or a hard tilt bound in a future version.

Filed as v4.x candidate: object-conditional? No — the probe suggests a HIGHER GLOBAL hard floor
might help geometry-limited shapes at no cost to others (cuboid/ellipsoid pads are huge), but that
is untested cross-object and NOT for now. Freeze holds.


**2026-08-31 — MATERIAL A/B COMPLETE on all six primitives (same meshes, same frozen recipe,
tofu vs mushroom material). The force-budget theory holds with one honest miss.**

| shape | tofu succ | mush succ | mush sub-yield | verdict (mush) |
|---|---|---|---|---|
| cylinder | 53.3 % | **100 %** | 100 % | PASS |
| sphere | 53.3 % | **100 %** | 100 % | PASS |
| ellipsoid | 72.7 % | **100 %** | 100 % | PASS |
| cuboid | 88.9 % | 94.1 % | **81 %** (was 100) | PASS |
| lamp | 57.1 % | 57.1 % (IDENTICAL) | 100 % | REVIEW |
| torus | 19.0 % | **38.1 %** | 88 % (was 100) | REVIEW |

Readings:
1. **Where force was the binding constraint (curved convex bodies), the stiffer material fully
   fixes success** — cylinder/sphere/ellipsoid all reach 100 %, at equal-or-LOWER median stress.
2. **Where geometry binds, material does nothing or little**: lamp bit-identical at 57.1 %
   (bulb-neck poses; tilt signature); torus doubled (19 -> 38 %) — MORE force effect than
   predicted ("little change" was the prediction; wrong by half) but still pose-dominated.
3. **The flip side, first seen here**: on shapes that never needed the force (cuboid), stiffness
   costs gentleness margin — cuboid sub-yield 100 -> 81 %, torus 100 -> 88 %. Nothing is free;
   the per-episode `priv_stress` filter covers the residue at conversion.

Handoff annotations updated to the final numbers. The off-recipe lamp area-floor probe
(hard 50 mm2) now runs behind the chain; the frozen recipe remains untouched throughout.


**2026-08-31 — ADDITIVE `prim_*_mush` variants (mushroom material) replace the in-place material
override; material A/B first rows: cylinder & sphere 53 % -> 100 %, lamp UNCHANGED at 57 %.**

Per user instruction: never override existing registry entries for variants — the earlier in-place
swap (fdbc320) is REVERTED (plain `prim_*` back to tofu material, exactly as smoked) and six
ADDITIVE `prim_*_mush` entries created instead (same meshes, `MATERIALS["mushroom"]`, mushroom DR
ranges; 18 new config files). Reduces merge-conflict risk with the cluster and preserves both
halves of the material A/B. Saved as a standing memory.

_mush smoke (frozen recipe, full renderings), first three:
| variant | success | tofu-material baseline |
|---|---|---|
| prim_cylinder_mush | **100 %** | 53.3 % |
| prim_sphere_mush | **100 %** | 53.3 % |
| prim_lamp_mush | 57.1 % | 57.1 % (IDENTICAL) |

The dissociation is clean: cylinder/sphere were FORCE-BUDGET-limited (material fixes them); the
lamp is GEOMETRY-limited (bulb-neck poses; material does nothing). Cuboid/ellipsoid/torus pending;
torus predicted to behave like the lamp. An off-recipe ANALYSIS PROBE (lamp_mush, hard area floor
50 mm2) is queued behind the chain to answer "would a min-contact-area floor help the lamp?" —
prediction on record: modest at best, tilt-driven failures remain. The frozen v4.1 recipe is
untouched by all of this.

Cluster: collect the `_mush` prim experiments (handoff updated), not the plain tofu ones.


**2026-08-31 — PRIMITIVES SWAPPED TO MUSHROOM MATERIAL (user-approved object-definition change;
v4.1 recipe untouched). Cluster had NOT started prim collection, so no data forks. Smoke rerunning.**

Follow-up to the three-layer analysis: the force-budget layer predicts that the mushroom material
(E 3e5, yield 4e4 — 6x stiffer, 2x the yield of tofu) lifts the curved prims into the ~75-90 %
band via two compounding channels (higher sub-yield force ceiling; deeper p98 commanded closures).
The user approved the swap explicitly. Changes: the six `prim_*` registry entries now reference
`MATERIALS["mushroom"]` (`MATERIALS["tofu"]` itself untouched — it is shared with the food tofu);
prim DR ranges E [2.0e5, 3.0e5], nu [0.32, 0.38], rho [900, 1000]. Task substeps already exceed
the mushroom-stable minimums (235@250 / 190@200 vs 210@250 needed) — unchanged. **No v4.1
synthesis/executor parameter was modified.**

The tofu-material smoke runs stay on disk as the material half of a clean same-shape A/B
(tofu-material: cuboid 92 / ellipsoid 71 / lamp 59 / cylinder 56 / sphere 50 / torus 18 %, all
100 % sub-yield). Predictions to check against the rerun: curved prims ~75-90 %; torus improves
little (its failure is pose-driven: 62 % edge/pinch poses).


**2026-08-31 — WHY lamp/cylinder/sphere/torus succeed at only 19-57 % while mushroom hits ~100 %:
a three-layer analysis (observation only; v4.1 frozen, nothing modified).**

Per-attempt planner metrics (`dr_params.csv`) for ALL attempts including failures, prims vs the
food anchors, success-vs-failure medians:

| object | succ | align S/F | min_pad S/F (mm2) | tilt S/F (deg) | closure S/F (mm) |
|---|---|---|---|---|---|
| prim_cuboid | 92 % | 0.93/0.79 | 378/69 | 0/9 | 8.0/3.0 |
| prim_ellipsoid | 71 % | 0.95/0.81 | 129/45 | 0/8 | 8.0/2.5 |
| prim_lamp | 59 % | 0.94/0.85 | 90/40 | 0/16 | 8.0/8.0 |
| prim_cylinder | 56 % | 0.96/0.81 | 437/75 | 0/15 | 8.0/6.9 |
| prim_sphere | 50 % | 0.94/0.80 | 88/46 | 13/13 | 8.0/7.8 |
| prim_torus | 18 % | 0.72/0.72 | 44/31 | 7/12 | 3.4/3.1 |
| mushroom (p98) | 100 % | 0.94/– | 42/– | 0/– | – |
| tofu (p98) | 67 % | 0.96/0.78 | 177/76 | 0/14 | 8.0/3.4 |

**Layer 1 — the user's observation is confirmed: pinch/edge POSES exist and NEVER lift.**
Defining a poor pose as align < 0.75 OR min_pad < 10 mm2: they are 12-22 % of attempts on
lamp/cylinder/sphere and their success is **0 %** (0/16 pooled). On the TORUS they are **62 %** of
attempts (success 16 %) — the ring's geometry makes a flush two-pad pose rare, and this alone
explains most of its 19 %. (Selection can only pick the best of what CMA finds; on a torus the
feasible pool itself is mostly edge grasps. Documented, not fixed — freeze.)

**Layer 2 — but good poses still only lift 64-70 % on the curved soft prims** (vs 92 % cuboid), so
pose quality is only half the gap. Among GOOD poses, failures differ from successes by:
- **tilt**: successes are almost exactly top-down (0 deg median); failures 8-16 deg. A tilted
  approach on a curved soft surface slides.
- **commanded closure**: successes cluster AT the 8 mm cap; failures at 2.5-7 mm — the familiar
  under-command signature: tofu material has yield strain sigma_y/E = 40 %, so the p98 crossing is
  DEEP and pose-noisy; shallow-scan poses get gentle commands that slip.
- **min_pad**: successes carry ~2-6x the pad contact of failures — curvature shrinks the patch.

**Layer 3 — the material force budget explains mushroom vs everything.** Staying sub-yield caps
the contact pressure at sigma_y; friction capacity ~ 2*mu*sigma*A. The mushroom has 6x the E and
2x the yield of tofu material: at the same gentleness fraction it generates several times the grip
force, so even its modest 42 mm2 patches hold with margin (~100 %). Tofu-material prims at 0.4-0.6x
of a LOW yield (20 kPa) have thin margins that only survive with a LARGE patch: flat faces
(cuboid 378 mm2, tofu-cube 177 mm2) -> 67-92 %; curved patches (88-129 mm2) -> 50-71 %; the
66 g cylinder needs the most force of all (3.4x the mushroom's mass) yet has line contact.

**One sentence:** low success = (a) a minority of pinch/edge poses that never lift — dominant on
the torus; (b) on good poses, soft-material sub-yield force ceilings x curvature-shrunk contact
patches x tilt sensitivity — the price of erring gentle on soft curved objects, paid in unsaved
attempts, never in saved-demo quality (100 % sub-yield throughout).


**2026-08-31 — PRIMITIVES SMOKE COMPLETE (6 x 16 eps, frozen v4.1, full renderings). Sub-yield is
100 % ON EVERY OBJECT; success orders EXACTLY by local contact flatness. Recipe untouched.**

| primitive | success (of attempts) | sub-yield | median stress | verdict | run |
|---|---|---|---|---|---|
| prim_cuboid | **88.9 %** (16/18) | 100 % | 0.56x | PASS | `26-08-30-lue` |
| prim_ellipsoid | **72.7 %** (16/22) | 100 % | 0.61x | PASS | `26-08-30-adf` |
| prim_lamp | 57.1 % (16/28) | 100 % | 0.50x | REVIEW | `26-08-30-pnv` |
| prim_cylinder | 53.3 % (16/30) | 100 % | 0.38x | REVIEW | `26-08-30-qil` |
| prim_sphere | 53.3 % (16/30) | 100 % | 0.40x | REVIEW | `26-08-30-kjx` |
| prim_torus | **19.0 %** (16/84) | 100 % | 0.42x | REVIEW | `26-08-30-ofm` |

**Two findings, both observations (v4.1 is frozen; nothing was or will be tuned):**
1. **Saved-demo gentleness is INVARIANT: 100 % sub-yield on all six**, max stress 0.50-0.96x. The
   gentle-erring recipe never saved a damaged episode; every failure is an unsaved slip, i.e. pure
   wall-clock. Combined with the food A/B this is now 13 categories where the p98 recipe's saved
   demos are 88-100 % sub-yield.
2. **Success orders exactly by local contact flatness**: flat faces (cuboid 89 %) > gently curved
   long sides (ellipsoid 73 %) > strongly curved / bulb-and-neck (lamp/cylinder/sphere 53-57 %) >
   thin ring (torus 19 %). Rank-perfect with zero exceptions — flat pads on curved soft surfaces
   slip at gentle closures. A clean, presentable geometry result; also note the non-convex lamp
   behaves like the convex curved shapes (57 %), so non-convexity per se is not the driver.

The torus (16 saved from 84 attempts, ~5x wall-clock) met the "thin scope probe" expectation but
DID collect, all sub-yield — the small-strain scope limit did not bite the way the full banana's
did. Cluster guidance stands: collect it if the wall-clock is acceptable, skip otherwise.

Every episode (success AND failure) has a rendering under each run's `videos/` /
`videos_failed/` per the new standing rule; the failure clips are the material for reading slip
vs topple per shape. `docs/smoke_datasets.md` regenerated (122 runs).


**2026-08-30 — v4.1 IS FROZEN AS FINAL (user decision). Paper deadline 2026-09-15 (16 days);
large-scale cluster collection has started; there is NO room for recollection. NO further edits
to any v4.1 parameter — the scan metric (p98), the gain (4.92), the auto rules, the executor —
regardless of how anything performs, unless the submission itself is abandoned.**

This includes new objects: if a new object underperforms under the frozen recipe, the outcome is
DOCUMENTED, not fixed. (The `--scan-metric p98` default committed in c5ff4d4 makes an unflagged
v4 run exactly the frozen v4.1 behaviour.)

**Six primitive diversity objects added (all TOFU material, user request):**

| object | nominal | source |
|---|---|---|
| prim_cylinder | r 2 cm, h 5 cm | user spec |
| prim_sphere | r 2 cm | user spec |
| prim_lamp | bulb r 1.5 cm + tapered neck + cylindrical base, ~5.2 cm tall | user spec (proportions mine) |
| prim_cuboid | 4 x 3 x 2.5 cm | proposed: flat-face anisotropy (existing tofu is an equal cube) |
| prim_ellipsoid | 5 x 3 x 2.5 cm | proposed: smooth anisotropy |
| prim_torus | R 1.4 / tube 0.7 cm | proposed: topology + ring grasp; deliberately thin (1.4 cm) |

All procedurally generated, watertight, `prim_` namespace (bare `cylinder` was taken). Configs
templated from tofu (task grid 200/250 by bbox volume, DR = tofu template with scale narrowed to
[0.85, 1.15] since several prims run near the gripper span, experiment = tofu armfocus template →
`superset_soft_armfocus` obs, so per-episode `priv_stress` is recorded). Registry entries carry
the freeze note.

16-ep smoke chain running under the frozen recipe (verified at launch: `metric=p98 gain=4.92`),
watchdog armed. Expectations, honestly: sphere and cylinder may roll/topple under orientation DR;
the torus is banana-chunk-thin and MAY hit the small-strain scope limit — if so, that is a
documented scope observation, NOT a reason to touch the recipe.


**2026-08-30 — v4.1-vs-v4.2 A/B COMPLETE (7/7): the metrics BRACKET the object
set. Collection recipe decided: p98, by the asymmetric-bars argument. Cluster handoff written
(`docs/collection_v4_handoff.md`); `integrate-all-2026-08-29` merged.**

| object | v4.1 (p98) succ/sub-yield | v4.2 (masked) succ/sub-yield |
|---|---|---|
| mushroom | 88.9 / 100 ✓ | 94.1 / 100 ✓ |
| raspberry | 100 / 88 ✓ | 100 / **56** ✗ |
| cherry_tomato | 76.2 / 81 ✓ | 57.1 / 75 ✗ |
| tomato | 80.0 / 100 ✓ | 66.7 / 100 ✓ |
| tofu | 66.7 / 100 ✓ | 76.2 / 100 ✓ |
| strawberry | **45.7** / 94 ✗ | 88.9 / 94 ✓ |
| banana_chunk | **42.1** / 100 ✗ | 59.3 / 100 ✗ (hair) |

Neither metric dominates; each fails two objects, in OPPOSITE directions. **The bars are
asymmetric for a dataset**: p98's failures are SUCCESS shortfalls (failed lifts are never saved →
pure wall-clock cost), masked's include SUB-YIELD shortfalls (damaged episodes enter the data).
**Collection uses `--scan-metric p98`** (saved-demo sub-yield 88-100 % on every object; ~2x
collection time accepted on strawberry/banana_chunk), with per-episode `priv_stress` filtering as
the final guard. `--closure-gain` now defaults per metric (masked 1.31 / p98 4.92) so the two can
never be mispaired.

Merged `integrate-all-2026-08-29` (their side: category conditioning + VLM reference frames,
delta-gripper action path, GAP replication vendored under `third_party/GAP`, scripted top-down
vision-only baseline, arrhenius sbatch, dppo submodule advance; only textual overlap was this
file). Note for E1: their `dppo/scripted/` baseline may be adaptable as the gentleness-blind
comparison.


**2026-08-30 (v4.2) — v4.1's p98 crossing was an OVER-CORRECTION bundled with the real fix.
On soft/low-yield objects it degenerates (c_y = 0 at the plan width itself), collapsing commands
to the clip minimum. v4.2: masked-top10 crossing, pen_tol relaxation KEPT, gain 1.31.**

v4.1 7-object results (16 eps, both bars = sub-yield >= 80 % AND success >= 60 %):

| object | success | sub-yield | median | verdict |
|---|---|---|---|---|
| mushroom | 88.9 % | 100 % | 0.32x | PASS |
| raspberry | 100 % | 88 % | 0.72x | PASS |
| cherry_tomato | 76.2 % | 81 % | 0.74x | PASS |
| tomato | 80.0 % | 100 % | 0.39x | PASS |
| banana_chunk | **42.1 %** | 100 % | 0.47x | REVIEW |
| strawberry | **45.7 %** | 94 % | 0.32x | REVIEW |
| tofu | (running) | | | |

Both REVIEWs are the same monotone signature: success rises steadily with commanded closure
(banana_chunk 11 % at <2 mm -> 67 % at >4 mm; strawberry 0 % at <1.5 mm -> 100 % at >4 mm) with
huge measured stress headroom — pure under-command.

**Diagnosis (canonical-pose scans, all four key objects):**
1. **The holdability floor is dead.** Predicted grip satisfies the scalar Coulomb inequality at
   FIRST CONTACT on every object (c_hold = 0.0 everywhere) — the surrogate's holdability
   prediction carries no information about simulator slip. Ruled out as a fix.
2. **The p98 crossing degenerates on soft objects: raspberry and strawberry give c_y = 0.0 mm** —
   the unmasked contact concentration exceeds yield at the plan width itself, so lambda*c_y = 0
   and the command collapses to the 0.8 mm clip (strawberry's 0 %-success bin).
3. The MASKED top10 curve under the relaxed scan is smooth and non-degenerate, and its crossing at
   the identification pose is a genuine stress crossing (statuses `ok` throughout — not the
   pen_tol artifact that motivated v4.1's switch).

**Lesson (own it): v4.1 changed two things at once** — the pen_tol relaxation (real fix, kept) and
the crossing metric (harmful, reverted). A confounded double-change that cost one chain pass.

v4.2: `--scan-metric masked|p98` (default masked; p98 kept for comparison), gain re-identified
under the masked relaxed interpolated scan: mushroom c_y = 4.88 mm vs measured-good 6.4 mm ->
**default 1.31**. `scan_metric` + `closure_gain` now recorded in config.yaml. Full 7-object rerun
queued after tofu completes the v4.1 chain. (Tofu stall alert was a false alarm — soft-MPM batches
exceed the 20-min episode window; the process is alive at 21 FPS.)


**2026-08-30 (v4.1) — the v4 scan's c_y was often GEOMETRIC, not stress-based: the search's 3 mm
gross-clipping tolerance terminated it. Fixed (user's pen_tol question exposed it); crossing now
on UNMASKED p98 with interpolation; lambda re-identified = 4.92; chain restarted.**

v4.0 first pass: mushroom 94.1 % / 100 % sub-yield (PASS) but raspberry 100 % / 56 % — and the new
per-episode `closure_cmd_mm` column made the diagnosis immediate: episodes with commanded closure
<= 3 mm were 88 % sub-yield, > 3 mm only 25 %. The c_y values clustered at 3.0-3.5 mm = pad depth
(~1.5 mm) + dw/2 crossing `pen_tol` = 3 mm — the scan was being cut off by the SEARCH's
gross-clipping SDF filter, so c_y reflected geometry, not predicted damage. (The user asked
whether 3 mm was too strict / whether deeper indents are sometimes wanted — exactly the right
question: for the search it is a clipping filter that intended contact never trips; for the scan
it was a bug, and yes, deep indents are legitimate whenever predicted stress stays sub-yield.)

v4.1 scan: crossing on the **UNMASKED p98** (`stress_p98`, newly exposed from the same solve —
the mask is right for pose ranking, wrong for damage onset), `pen_tol` relaxed to 5 cm for scan
calls only (search unchanged; validity bounded by the 10 mm small-strain limit), statuses handled
by meaning (`no_contact` -> deepen; `degenerate`/`table` -> validity edge), and the crossing
LINEARLY INTERPOLATED (gain ~5 x a 0.5 mm step would otherwise quantize the command by ~2.5 mm).
Lambda re-identified on the mushroom under the new criterion: c_y = 1.30 mm vs measured-good
closure 6.4 mm -> **gain 4.92** (default). First batch: mushroom c_y 0.5-1.3 mm -> commands
2.5-6.2 mm varying per pose.

Also: a botched slice-based edit corrupted v4 mid-change (`FIRM_FORCE_THRESH_N` first occurs in a
COMMENT before the function, so the slice was reversed-empty and `str.replace('', ...)` exploded
the file) — restored from git, redone with exact-text replacement. Lesson: never build a
replacement slice from two independent `index()` calls without asserting start < end.

7-object v4.1 chain restarted from scratch; v4.0 partial runs deleted (n40/regrasp/probe runs
kept). Results table to follow.


**2026-08-30 — `docs/paper/method_v4.md`: the complete v4 method + validation reference in
paper-adaptable prose.** Every constant verified against `collect_demos_synth_v4.py`/`smgrasp/`
on this date. Part A: problem statement, preprocessing, E=1 FEM + inertia-relief + Schur
per-candidate solve, pad contact model, E-linearity, holdability, full objective with recipe
weights, feasibility ladder, search + auto bounds, the v4 surrogate-selected width (lambda = 1.28,
sole executor constant), execution FSM incl. re-grasp mode, provenance recording. Part B: the
n=40 ranking validation with the full statistics triplet (rho +0.669 CI [+0.38,+0.86], tau
+0.528, concordance 76%) and the decision-relevant decile table (predicted-gentlest 10: 0/10
past yield; predicted-harshest 10: 8/10); the 4-object closure-transfer table; per-object v4
outcomes (mushroom 94.1% / 100% sub-yield / 0.49x median — rest running); the saturation finding
as a standalone contribution; E1 as the gate for comparative claims. DO-NOT-OVERSTATE markers
inline (e.g. "one global gain identified once", never "calibration-free").

First v4 transfer result: mushroom 16-ep verification 94.1% success, 100% sub-yield, median
0.49x yield with adaptive closures 3.8-8.0mm (v3: fixed 6.4mm).


**2026-08-30 (v4) — `collect_demos_synth_v4.py`: surrogate-selected executed width (v3 untouched
as fallback). And the n=40 ranking validation is SIGNIFICANT: rho = +0.669, p = 2.4e-6.**

- **v4** (fork of v3 per the fallback convention): the executor's closure constants (2.5 mm
  `width_cls` baseline, `extra_close`, firm base) are replaced by `gain * c_y`, where `c_y` is the
  closure at which the SURROGATE predicts yield at the chosen pose (scanned with the refine-round
  primitive, using the DR-drawn E so the scan matches the simulated material). One global constant
  (`--closure-gain 1.28`, identified once on the mushroom). Weak-grasp firm check kept as fallback
  (0.5 x commanded closure, capped 2 mm). New `closure_cmd_mm` column in dr_params.csv.
  First batch confirms adaptive behaviour: mushroom c_y 4.0-7.0 mm -> commands 5.1-8.0 mm varying
  with pose and material draw (v3: fixed 6.4 mm for all).
- **n=40 controlled sub-yield correlation (fixed scene, width swept): Spearman rho = +0.669,
  p = 2.4e-6** (Pearson +0.563, p = 1.5e-4; MPM span 0.32-1.13x yield). The surrogate's stress
  ranking is now VALIDATED in the operating regime — quotable. Past yield it remains rho = 0
  (saturation); any paper statement must name the regime.
- 7-object v4 verification chain running (16 eps each, all-auto, no pasta).


**2026-08-30 (later) — CALIBRATION IS NOT NECESSARY: the surrogate's own stress-vs-width curve
predicts the per-object safe closure. Rank-perfect across 4 objects, stable ~0.5-0.6 conservative
bias. Raspberry probe confirms: baseline-only closure -> 100 % sub-yield (was 19 %).**

User pushback: "if calibration is needed, the significance of the method reduces." Correct — and
testable. The closure constants exist only to turn the planned width into a commanded width, but
the surrogate already computes sigma(width) (the refine-round primitive). Scanned it per object and
compared against the measured safe/unsafe closures from the 08-29/30 runs:

| object | surrogate closure@yield | measured yield-closure | bias |
|---|---|---|---|
| mushroom | 5.0 mm | ~10 mm | ~0.5 |
| raspberry | 2.0 mm | ~3-4 mm | ~0.57 |
| cherry_tomato | 2.0 mm | ~4 mm | ~0.5 |
| banana_chunk | 3.5 mm | > 6 mm | <~0.58 |

Rank-perfect ordering (raspberry ~ cherry < banana_chunk < mushroom) — the exact pattern the
analytic K*(yield/E)*L rule provably mispredicted in both directions — and a conservative bias
stable enough that ONE global factor transfers across all four, covering both the over-squeezed
objects and the under-gripped banana_chunk.

**Design consequence (supersedes the 08-29 "measure per object" recommendation):** replace the
baseline/squeeze/firm constants with a SURROGATE-SELECTED executed width — command the width where
predicted stress = target x one global bias factor (identified once, on one object). Zero
per-object constants; the executor's closure becomes part of the model; the cross-object closure
table turns from a bug list into evidence FOR the surrogate. `fem_surrogate_status.md` section 5
updated (old item 1 superseded by 1').

**Experimental confirmation of the executor diagnosis:** raspberry probe with
`--grasp-extra-close 0` (closure = the hardcoded 2.5 mm baseline only): median **0.65x yield,
100 % sub-yield** (was 1.07x / 19 %). The surrogate and the evaluation metric were never the
problem.

NOT yet implemented: the surrogate-selected width executor (moderate change: derive w_cmd from the
already-computed width scan at synthesis time, delete the three constants; keep the weak-grasp firm
check as a fallback). Next work item before the cluster collection, together with wiring
`execute_offset` so the yield guard sees the executed width.


**2026-08-30 — FEM surrogate scientific-status study (`docs/fem_surrogate_status.md`) + paper
experiment design (`docs/paper/synthesis_experiments.md`). Raspberry diagnosed: NOT a surrogate or
metric failure — a THIRD unscaled closure constant, plus `execute_offset` was never wired.**

User asked whether the FEM synthesis is scientifically OK for the paper given the raspberry's 19 %
sub-yield, whether the evaluation is skewed, and what a near-full-FEM-but-fast path looks like.

**Diagnosis (strain accounting on `26-08-29-zlb`, airtight):** executed closure beyond the planned
contact width is 4.4–5.5 mm on a 13.7 mm object = 32–40 % strain vs a 15 % yield strain — the
raspberry is COMMANDED to 2.1–2.7x its yield strain, and the measured top10 pinned at 1.0–1.2x
yield is honest MPM plastic saturation. The evaluation is NOT skewed (top10/mean stable ~2.3).
The dominant term is a **hardcoded 2.5 mm baseline in `width_cls`** — the THIRD instance of the
unscaled-constant class (18 % of a raspberry, 8 % of a mushroom). Also: **the collector never
passes `execute_offset`**, so every candidate is scored at a width 4–7 mm wider than executed —
the planner's own yield guard never sees the real operating point. Confirmation probe
(`--grasp-extra-close 0` → closure = baseline only) running.

**Study doc** (`fem_surrogate_status.md`): claim-by-claim writability verdict ("gentleness-aware
selector" SAFE; "stress predictor" FALSE; ranking PENDING n=40); full validation ledger; why
Genesis FEM+IPC in-loop is orders too slow (7k–35k candidates/grasp vs settling-length implicit
solves); ranked fidelity ladder — (1) measured per-object closure calibration, (2) wire
`execute_offset`, (3) plastic-excess objective from the SAME solve (fixes the provable flatness
past yield), (4) deformed-configuration Picard pass on top-K only, (5) nonlinear rescoring of the
argmax, (6) DefGraspSim offline spot-checks.

**Paper experiments** (`docs/paper/synthesis_experiments.md`): community context — DefGraspSim
(RA-L'22, open corotational-FEM evaluator for deformables; the closest work and a potential
external gold standard) and DefGraspNets (ICRA'23, learned FEM surrogate — validates the surrogate
idea from the learned side); AnyGrasp SDK is license-gated, Contact-GraspNet/GPD are the open
baselines. Ranked experiment list E1–E8; **the critical missing one is E1: a gentleness-blind
antipodal+rigid-metric baseline executed through our own pipeline** — everything so far compares
us only to ourselves. Iteration order before freezing the table: executor calibration → E1 →
n=40.


**2026-08-30 — RE-GRASP (hover-start) demos implemented: `--regrasp-prob`. Smoke on mushroom
15/15, all three pieces of user feedback applied.**

The idea (user, 2026-08-27): BC only sees states on the expert trajectory, so after a FAILED grasp
the policy is off-distribution — hovering over the object, gripper part-closed, holding nothing.
These episodes START in exactly that state.

**Behaviour** (`--regrasp-prob P`, per-episode; 0 = off, ~0.2 for a real collection, 1.0 to test):
1. Gripper is placed **6-12 cm above the grasp pose** with a **3 cm-radius xy scatter** (the object
   may have ROLLED after the failed attempt) and a **+-8 deg** orientation jitter.
2. Gripper starts at a **random part-closed width (10-80 mm)** — the genuinely unseen part of the
   state, since a normal approach passes through that height but always fully OPEN.
3. The first **12 recorded steps RE-OPEN to nominal while HOLDING the start pose** — this is the
   recovery action itself, and it is what the policy must learn to do before re-approaching.
4. Then it drives **straight to the grasp** at the normal constant velocity (no two-phase
   approach), and the usual grasp / firm / lift / hold follow unchanged.
5. The home -> hover move is **executed but NOT recorded**, so the demo's first frame IS the hover.
6. `dr_params.csv` labels every episode **`re-grasp-demo`** or **`standard`**.

**Smoke (mushroom, 15 eps, `--regrasp-prob 1.0`): 100 % success, all labelled, all within spec:**

| property | measured | spec |
|---|---|---|
| height above grasp pose | 63-118 mm (median 90) | 60-120 |
| xy offset from grasp | 4-39 mm (median 26) | <= 42 (3 cm radius) |
| start width | 16-78 mm | 10-80, randomized |
| width after re-open | 80 mm, all episodes | nominal |

**Pinch filtering still applies unchanged** (user check): `filter_pinch_episodes.py` runs on these
as-is — it keys off `priv_object_pos`, which they carry — and flagged 2 of 15 on the first smoke.

**Not implemented, and worth stating:** this teaches RESTART, not RETRY. There is no failure
DETECTION — the policy learns "from this state, re-open and grasp", not "that attempt failed, so
recover". The hover state is the trigger. Also the object is at its normal spawn pose rather than
displaced by a real failed grasp; the 3 cm xy scatter of the GRIPPER is a proxy for that.
Verification must be at TRAINING time against a matched no-hover-start baseline — collection
success says nothing about whether retry behaviour emerges.


**2026-08-29 — PER-OBJECT VERIFICATION (7 x 16 eps): 4 PASS, 3 REVIEW. The analytic squeeze rule
is NOT sufficient; the safe indentation has to be MEASURED per object, not predicted.**

User asked to stop the large collection and verify the synthesizer on every object before handing
the big run to the cluster agent. Done, each object judged on ITS OWN material
(PASS = sub-yield >= 80 % AND success >= 60 %):

| object | success | sub-yield | verdict |
|---|---|---|---|
| mushroom | 100 % | **100 %** | PASS |
| tomato | 100 % | **100 %** | PASS |
| tofu | 76.2 % | **100 %** | PASS |
| strawberry | 100 % | 94 % | PASS |
| cherry_tomato | 88.9 % | **56 %** | REVIEW (gentleness) |
| raspberry | 100 % | **19 %** | REVIEW (gentleness) |
| banana_chunk | **53.3 %** | 100 % | REVIEW (success) |

**A second unscaled constant was found and fixed on the way here.** Cutting the base squeeze alone
moved the cherry tomato only 5.8 % -> 6 % sub-yield. Measuring the executed grasp showed why: the
planner synthesized 25.0 mm but the gripper closed to 19.9 mm — 5.1 mm beyond plan, of which only
0.84 mm was the squeeze. The other ~4.3 mm was the FIRM phase (`FIRM_EXTRA_CLOSE_M` 2.0 mm +
`FIRM_WEAK_EXTRA_CLOSE_M` 2.5 mm), hard constants applied to EVERY soft grasp — 18 % of a 24.7 mm
object. Firm now uses the same material-aware budget, and the weakness threshold is 5 % of the
object's OWN yield instead of a flat 2000 Pa. That took the cherry tomato 6 % -> 56 %.

**But the analytic rule has hit its limit.** `d ~ K*(yield/E)*L` mispredicts in BOTH directions:

| object | budget as % of object | sub-yield | success | diagnosis |
|---|---|---|---|---|
| raspberry | 22.2 % | **19 %** | 100 % | budget still too LARGE |
| banana_chunk | 14.8 % | 100 % | **53.3 %** | budget too SMALL |

A single calibration constant cannot satisfy both: the raspberry needs less indentation than the
formula says, the banana chunk needs more. Contact geometry (curvature, contact-patch size) enters
the real stress and is not captured by `(yield/E)*L`.

**Recommended fix — MEASURE instead of predict (not yet implemented).** A short per-object
auto-calibration at collection start: sweep the squeeze over ~4 values on a handful of throwaway
episodes, read the resulting peak `priv_stress` (already recorded), and pick the largest squeeze
whose median stays under ~0.8x yield. That is fully automatic, adds no per-category constants,
costs a few minutes per category, and directly optimises the real trade-off — gentleness AND grip
reliability — instead of predicting one from an elastic formula that the elasto-plastic simulator
does not obey.

**Lesson (this bug class has now appeared TWICE — squeeze, then firm):** any absolute length or
stress constant applied across objects differing in BOTH size and material is suspect. And an
object passing tells you nothing about the others: the mushroom read 100 % sub-yield under both the
broken and the fixed rule.

**Status for the cluster handoff:** mushroom / tomato / tofu / strawberry are ready to collect now.
cherry_tomato, raspberry and banana_chunk should wait for the auto-calibration. The existing
250-episode mushroom set (96.5 % success, 99.6 % sub-yield, full material DR) is valid and is the
intended supplement.


**2026-08-29 — CAUGHT MID-COLLECTION: the size-only squeeze rule is GENTLE ON THE MUSHROOM AND
DAMAGING ON THE CHERRY TOMATO. Replaced with a MATERIAL-AWARE rule; mushroom set preserved.**

Routine status check on the running collection compared per-category gentleness:

| object | median stress | sub-yield |
|---|---|---|
| mushroom (done, 250 eps) | 0.58x yield | **99.6 %** |
| cherry_tomato (124 eps in) | **1.18x yield** | **5.8 %** |

The cherry tomato was being collected in exactly the past-yield regime the whole squeeze fix
existed to escape. **Killed the chain at 124/250 and discarded that run.**

**Cause: `--grasp-extra-close auto` scaled with SIZE ONLY.** For an indentation `d` over a
characteristic length `L`, stress goes as `sigma ~ E * d / L`, so staying under yield requires

    d  <=  K * (yield / E) * L

The squeeze must scale with the material's **yield/E** ratio, and that ratio varies **2.7x**
across our objects (tofu 2.5, raspberry 6.7, mushroom 7.5, strawberry 8.3, banana_chunk 10.0,
tomato 12.0, cherry_tomato 13.3). The cherry tomato is the worst case — the STIFFEST E (0.4 MPa)
combined with a LOW yield (30 kPa) — so a squeeze that is gentle on a mushroom drives it well past
yield. A size-only rule cannot express that.

**Fix:** `K = 0.455`, calibrated so the MUSHROOM's squeeze is unchanged at 1.94 mm — which keeps
the already-finished, validated mushroom set exactly reproducible and means it does NOT need
recollecting. Every other object is then derived from its own E and yield, so this remains a
zero-per-category-constant rule:

| object | size-only (old) | material-aware (new) |
|---|---|---|
| mushroom | 1.94 mm | **1.94 mm** (unchanged by construction) |
| cherry_tomato | 1.50 mm | **0.84 mm** |
| tomato | 2.78 mm | 1.74 mm |
| banana_chunk | 1.24 mm | 0.93 mm |
| strawberry | 1.97 mm | 1.77 mm |
| raspberry | 1.00 mm | 0.94 mm |
| tofu | 1.82 mm | **3.00 mm** (very soft, yield/E = 0.4 -> tolerates MORE squeeze; clipped) |

Note tofu moves the OTHER way: being soft with a relatively high yield it was being squeezed too
LITTLE, which costs grip reliability for no gentleness benefit.

**Method note — this is why per-category gentleness must be checked DURING collection, not after.**
The mushroom's 99.6 % looked like proof the recipe was right; it was proof the recipe was right
*for the mushroom*. A single validated object cannot certify a rule that depends on material.

Chain restarted at cherry_tomato (mushroom retained). 6 categories remaining.


**2026-08-28 — MY OWN BUG: three collection chains ran CONCURRENTLY for hours. Cause: a BRE
alternation in a `pgrep` pattern, so every "kill" was a no-op and every "relaunch" stacked.**

Checking collection progress showed impossible numbers (mushroom 237/250 "running" while
cherry_tomato showed 250/250 complete and raspberry 128/250 — all from one supposedly SEQUENTIAL
chain). `ps` showed **three `bigchain.sh` instances** alive at 17.8 h, 7.7 h and 7.3 h.

**Root cause:** every teardown used `pgrep -f "[b]igchain\|[c]ollect_demos_synth_v3"`. `pgrep`
takes an **ERE**, where alternation is `|`; `\|` is BRE syntax and is matched LITERALLY. So the
pattern looked for the literal string `bigchain|collect_demos_synth_v3`, matched nothing, killed
nothing — and each `nohup bash bigchain.sh` added another chain writing into the same dataset
directories. The three chains were also running THREE DIFFERENT code versions (pre-stress,
pre-material-DR, and current), so their outputs were not even mutually comparable.

**Damage:** none to a frozen dataset (nothing valid had completed), but ~18 h of GPU wasted and
every partial run had to be discarded. `data.pkl` is only written at completion, so the
in-progress sets were unusable regardless.

**Fixes:**
1. All chains killed by explicit PID.
2. Every partial / stale-code run dir from today deleted (8 mushroom + 1 raspberry).
3. **`bigchain.sh` now takes an exclusive `flock`** and refuses to start if another instance holds
   it — verified: a second launch prints "another bigchain is already running -- refusing to
   start". A stacked chain is now impossible regardless of whether a kill worked.

**Lessons, both worth keeping:**
- **`pgrep`/`pkill` take ERE.** `\|` silently matches nothing rather than erroring, so a kill can
  appear to succeed while doing nothing. Verify a kill with `ps` instead of trusting exit status —
  this is the same class of failure as the earlier `pkill` self-kills.
- **Long-running background chains need a lockfile, not just a kill-before-launch.** Kill-then-
  relaunch is only as reliable as the kill.


**2026-08-28 — PRE-FLIGHT before the frozen collection found TWO INERT DR BLOCKS. Material DR had
NEVER been applied by the v3 collector. Fixed, relaunched.**

User asked to double-check everything and confirm all DR params are recorded before committing the
large run. Both asks turned up real bugs:

1. **MATERIAL DR WAS NEVER APPLIED.** `DRConfig.sample_scene()` (which draws E / nu / rho /
   coup_friction) exists and `collect_demos_synth_v3.py` **never called it**. Every demo this
   collector has ever produced used the registry's NOMINAL material, and the `object_E`,
   `object_nu`, `object_rho`, `coup_friction` ranges in ALL SEVEN DR configs were dead text.
   That means **zero material diversity** in every dataset collected with v3 to date — a serious
   sim2real gap, since real produce varies in firmness far more than in shape.
   Fixed: `sample_scene()` is now called and E/nu/rho are baked onto the `ObjectEntry`.
2. **`coup_friction` was never passed to the sim.** `GenesisWorker` takes it (default 4.0) and the
   collector never supplied it, so the `[3.5, 4.5]` range was equally inert. Now passed per scene.
3. **Most DR draws were unrecorded.** `dr_params.csv` logged only `scene_scale` /
   `scene_bend_deg`; `twist`, `taper`, `rbf`, `axis_scale` (all APPLIED) and the material draws
   were absent, so a frozen dataset could not be reproduced or analysed along those axes. The CSV
   now carries all of them (31 columns).

**Known remaining limitation:** `object_yield` still cannot be randomized — `ObjectEntry` has no
yield field, so yield always comes from the registry material (pre-existing, noted in CLAUDE.md).

**Also added before freezing: `priv_stress` in `superset_soft_armfocus`**, so every episode carries
a per-episode gentleness record. Needed to (a) FILTER demos that exceed yield and (b) state the
sub-yield fraction of the ACTUAL dataset instead of a proxy sample. It is privileged-only and the
DPPO views are explicit key lists, so it does not touch the student (point-cloud) obs or any
converted view. The two 250-episode sets collected before this (mushroom, cherry_tomato) are
archived under `dataset/demos/_superseded_nostress/` — still valid demos, but lacking the stress
record and the material DR, so they are NOT part of the frozen set.

**Pre-flight verified on a 6-episode run:** all 31 DR columns populate; `mat_E` 2.607e5 inside the
mushroom's [2.0e5, 3.0e5]; `coup_friction` 4.435 inside [3.5, 4.5]; `priv_stress` present with
median 0.51x yield and **100 % sub-yield**; `rbf` reads 0 because no config sets `object_rbf`
(correct, not broken); all 7 experiments load with sensible auto params (squeeze 1.00-2.78 mm, yaw
30-69 deg); 100 GB disk free against a ~5 GB need.

**Note for interpreting the new runs:** material DR is ACTIVE for the first time, so per-category
success rates are not directly comparable to the earlier smoke numbers — the task is now genuinely
harder and more diverse.


**2026-08-28 — "GENTLENESS-AWARE" is defensible; "provably gentle" is NOT (yet). The sub-yield
regime restores a positive ranking trend: rho 0.00 -> +0.52, but only p = 0.085 at n = 12.**

User asked whether the synthesis can be called provably gentle, and failing that whether
"gentleness-aware" is safe. Verdict:

**SAFE to claim "gentleness-aware synthesis".** Justified on three independent grounds:
1. **By construction** — the objective explicitly contains stress terms (`-stress_top10`,
   `-w_peak * E * hi_1`, `-w_press * pressure`), the auto area selection enforces a HARD yield
   guard (`YIELD_SAFETY = 0.8`), and the refine round selects the widest holdable width
   (= gentlest) among distinct poses.
2. **By outcome** — at the adopted operating point the demos sit at median **0.56x yield** under
   full DR with 83 % sub-yield, and 100 % sub-yield in the fixed-scene test.
3. **By construction of the operating point** — the squeeze base was reduced 5 mm -> 2 mm
   specifically to move the demos below yield.

**NOT safe to claim "provably gentle" / "gentleness-optimal" / "minimizes damage."** The
within-object ranking is still weak:

| regime | Spearman rho | Pearson r | n | sub-yield |
|---|---|---|---|---|
| PAST yield (old squeeze 4.8 mm) | **0.000** (p=1.0) | -0.47 | 12 | 0 % |
| **SUB-yield (adopted, 1.9 mm)** | **+0.517** (p=0.085) | +0.22 (p=0.49) | 12 | **100 %** |

**The saturation explanation held up** — moving below yield recovers a positive rank correlation
where there was literally none. But rho = 0.52 at p = 0.085 is a TREND, not proof, and Pearson
+0.22 says the relation is monotone-ish with outliers rather than tight. A larger sweep (n = 40)
is running to settle whether it is real.

**What "provably gentle" would actually require** (recorded so the bar is explicit):
1. A significant within-object rank correlation in the operating regime (n >= 40; in progress).
2. The chosen grasp shown near-optimal among FEASIBLE alternatives w.r.t. true damage — not just
   correlated, but close to the achievable minimum.
3. Damage measured as PLASTIC deformation (permanent shape change / plastic work), which is the
   physically correct quantity for an elasto-plastic body; von Mises stress is the wrong axis
   above yield and only a proxy below it.

Items 2-3 are a project in themselves. Item 1 is cheap and running.


**2026-08-28 — LARGE-SCALE COLLECTION LAUNCHED (7 categories x 250 eps). One recipe change first:
the squeeze base 5 mm -> 2 mm, because 5 mm drove EVERY demo past yield.**

User asked whether synthesis is justified enough to freeze a dataset. Honest split:
- **Justified:** as a producer of successful, diverse, quality-filtered LIFTS — 75-100 %
  demonstrator success across 7 objects, align 0.85-0.96, pinches filtered, per-object materials,
  mesh randomization, every grasp parameter auto-derived.
- **NOT justified:** the GENTLENESS claim. Yesterday's controlled test showed the FEM objective has
  zero discriminative power at the operating point, because every grasp sat at 1.05-1.13x yield
  where the MPM saturates.

Since the dataset is meant to be frozen, the operating point had to be fixed BEFORE collecting —
otherwise "squeezing is the demos" bakes squeezing in permanently.

**Found the knob: it was the EXECUTED SQUEEZE, not the synthesized width.** Fixed-scene probe:

| `extra_close` | MPM peak (median) | sub-yield? | success |
|---|---|---|---|
| 0.0 mm | 0.68x | yes | 100 % |
| 2.0 mm | 0.91x | yes | 100 % |
| 4.8 mm (old auto) | 1.05-1.13x | **no** | 100 % |

Confirmed under FULL DR on the mushroom (n=12, mesh-cycled): **2 mm gives median 0.56x yield with
83 % of demos sub-yield and 86 % success, vs 4.8 mm at ~1.10x, ~0 % sub-yield, 80 % success.**
Half the stress, most demos now genuinely sub-yield, and success slightly BETTER.

**Adopted: the `--grasp-extra-close auto` base drops 5 mm -> 2 mm** (auto = 2 mm x smallest extent
/ 33 mm, clipped [1, 3]). Resolved per object: raspberry 1.00, banana_chunk 1.24, cherry_tomato
1.50, tofu 1.82, mushroom 1.94, strawberry 1.97, tomato 2.78 mm. The old 5 mm was tuned for grip
reliability before we could measure what it did to the material.

**Collection running:** mushroom, cherry_tomato, raspberry, tomato, banana_chunk, strawberry, tofu
— 250 episodes each, 8 envs, `scene_dr_every 1`, **random** mesh DR (NOT `--mesh-cycle`: cycling
is a coverage tool, random sampling is the correct DR for a real dataset; at 250 eps there are
~30+ scene rebuilds so the pool is covered anyway). Pasta bundle excluded per user.

**Nothing is hardcoded per category in the recipe.** The only per-object values are ones with no
alternative: the registry MATERIAL (E/rho/yield — physical fact), the task MPM `grid_density` /
`sim_substeps` (CFL stability, must scale with object size and stiffness), and the mesh pool
(inherent to the category). Every grasp parameter — area floor, width cap, yaw bound, squeeze — is
derived from the object at run time.

**Caveat carried into training:** the gentleness metric still does not discriminate ABOVE yield,
and 17 % of mushroom demos remain at/above it. The dataset is "successful, mostly-sub-yield lifts",
not "provably gentlest-possible lifts". Retargeting the objective at plastic strain remains open.


**2026-08-28 — CORRECTION: the rho = 0.84 surrogate validation is a SCENE-SIZE artefact. Under a
CONTROLLED test the FEM metric has ZERO rank correlation with simulator stress, because the MPM
material SATURATES at yield.**

User asked how the correlation was actually measured — same grasp in both, or something looser.
Answer: same grasp (planner synthesizes, sim executes that grasp), but **observational**: the
scene varied across the 10 episodes (scale 0.81-1.48, three mesh variants). Two follow-ups:

1. **Pairing verified.** The earlier number zipped filtered CSV rows to saved episodes by index.
   Re-derived the exact (batch, env) of every saved episode from the collector log and joined on
   that: **identical result, rho = +0.842.** So the pairing was right (the 8-25 mm object-position
   offsets are MPM settling drift, not mispairing).
2. **Confound check.** planner vs `scene_scale` rho = -0.67; MPM vs `scene_scale` rho = -0.89;
   partialling scale out left rho = +0.758 — which looked reassuring. **It was not.**

**CONTROLLED experiment (the one that should have been run first):** fix the scene entirely
(`--scene-dr-every 0`, one scale, one mesh) and sweep ONLY the commanded grasp width, which drives
indentation and hence predicted stress. n = 12, mushroom:

| commanded width | planner predicted | MPM measured |
|---|---|---|
| 20.4 mm | 14.6 kPa | 1.10 x yield |
| 29.1 mm | **44.7 kPa** | 1.06 x yield |
| 32.5 mm | 17.3 kPa | 1.11 x yield |
| 34.2 mm | **11.8 kPa** | 1.10 x yield |

**Planner spans 3.8x (11.8-44.7 kPa); the simulator is FLAT at 1.05-1.13 x yield. Spearman
rho = +0.000 (p = 1.0), Pearson r = -0.47.**

**Root cause — the linear-elastic vs ELASTO-PLASTIC mismatch flagged in the model doc, appearing
exactly where predicted.** Genesis MPM `ElastoPlastic` saturates von Mises at the yield surface;
past yield, squeezing harder produces plastic FLOW, not higher stress. Every grasp in our regime
is at or past yield, so the simulator's stress carries no information there, while the surrogate
(no yield model) keeps predicting higher stress into a regime where true stress cannot rise.

**What this means:**
- **rho = 0.84 must NOT be cited as validating the gentleness metric.** It shows only that both
  models know a smaller object is stressed more.
- **At the current operating point the FEM objective does not discriminate grasp gentleness.**
- **Von Mises stress is the wrong gentleness measure for an elasto-plastic body past yield** — the
  meaningful quantity is PLASTIC deformation (permanent shape change / plastic work), which
  neither the surrogate nor `priv_stress` currently reports.
- Fix direction: move the operating point BELOW yield (the sweep shows 34 mm predicts 11.8 kPa, so
  gentler grasps exist), and/or retarget the objective at plastic strain.

**Method lesson: an observational correlation across DR-varied scenes is not metric validation.**
Hold the scene fixed and vary the thing the metric is supposed to rank. n = 12, one object —
repeat on a second before the paper, but the mechanism is unambiguous.


**2026-08-27 (validation) — MEASURED the FEM surrogate against the MPM simulator: ranking rho
= +0.84, but the absolute stress is ~3x LOW and the "gentle" demos sit AT yield.**

User asked whether our FEM and rigid-soft coupling are correct and justified given that Genesis
ships its own. Verified in the submodule: Genesis DOES have a full FEM dynamics solver
(`fem_solver.py`, implicit + Newton + PCG) and TWO proper contact couplers (`sap_coupler.py`,
`ipc_coupler/`). **We use none of them** — our sim is Genesis MPM (`ElastoPlastic`), and the
`smgrasp` FEM is a separate quasi-static SCORING surrogate. Full comparison in
`docs/grasp_synthesis_model.md` §9b.

**The justification is cost and it is real:** the surrogate runs ~7k-35k times per grasp search
per env, and the E=1 normalization makes material DR a scalar multiply. A dynamic FEM+IPC solve
per candidate is orders of magnitude too slow. **But ours is not a contact model** — quasi-static,
contact set fixed on the UNDEFORMED mesh, normal-only/frictionless, linear-elastic against an
ELASTO-PLASTIC simulator (so it has no yield model at all, while "gentleness" is defined by yield).

**Measured the agreement** (the number that decides justification; only the PRE-fix `rho +0.10`
existed before, buried in a docstring). Needed two fixes first:
- `privileged: stress: true` was **silently ignored by the v3 collector** — `_privileged_obs_batch`
  never mirrored the stress field that `PolicyEnv` emits, so the config asked and nothing appeared.
  Now emitted (`priv_stress` (N,2) = [mean, top10]/yield), with `yield_stress` threaded through
  `execute_and_collect`.
- New `superset_soft_armfocus_stress.yaml` obs + `single_lift_mushroom_soft_armfocus_stress`
  experiment for validation runs.

**Result, 10 successful mushroom episodes: Spearman rho = +0.842 (p=0.002), Pearson r = +0.795
(p=0.006).** The surrogate RANKS grasps by the stress the simulator will actually produce — which
is exactly what a planner needs, so choosing grasps with it is justified.

**⚠ But the absolute calibration is not, and this corrects numbers reported all week:**

| | planner predicted | MPM measured |
|---|---|---|
| range | 6.8 - 18.8 kPa | 20.7 - 46.4 kPa |
| vs the mushroom's 40 kPa yield | ~17-47 % | **52-116 %, median ~100 %** |

1. **Every "stress as % of yield" figure I have reported came from the PLANNER and is ~3x too
   low.** Those were surrogate predictions, not simulator measurements.
2. **By the simulator's own measure the "gentle" demos sit AT the yield stress** (median peak
   ~1.0x). Do not claim sub-yield demonstrations without re-measuring per object.

Part of the gap is DEFINITIONAL: the planner's `stress_top10` masks out contact-adjacent elements
while the MPM figure is unmasked, so the planner deliberately excludes the highest-stress region.
The planner's unmasked `hi_1` is the like-for-like comparison and has NOT been checked. The
ranking result stands either way; the absolute claim does not.

**TODO before the paper:** (a) repeat this on every object, not just the mushroom; (b) compare
`hi_1` vs the MPM figure to separate definitional offset from genuine calibration error; (c) decide
whether the gentleness objective should be re-tuned now that the demos are known to sit at yield.


**2026-08-27 (mesh coverage) — `--mesh-cycle` + `--n-envs 2`: forcing EVERY mesh to be sampled
lowers two headline numbers. The earlier 100 %s were partly a sampling artefact.**

User asked for full mesh coverage rather than trusting a uniform draw. Two changes:
- **`--mesh-cycle`** walks `object_mesh_pool` in ORDER (round-robin), one mesh per scene rebuild,
  instead of sampling uniformly. A uniform draw only covers the pool *in expectation*, and an
  8-episode run rebuilds the scene once or twice — so it saw 1-2 of 4-5 meshes and a broken
  variant could sit unnoticed. Smoke/coverage use only; real collections should keep random DR.
- **`--n-envs 2`** (was 8) so a short run produces more batches, hence more scene rebuilds.

**Results with EVERY pooled mesh exercised:**

| object | success (full coverage) | meshes covered | previous (1-2 meshes) |
|---|---|---|---|
| raspberry | **100 %** | **5/5** | 100 % |
| tofu | **100 %** | 1 (no pool) | 96 % |
| strawberry | **100 %** | 1 (no pool) | 92 % |
| cherry_tomato | **89 %** | **4/4** | 89 % |
| banana_chunk | **86 %** | 1 (no pool) | 69-80 % |
| **mushroom** | **80 %** | **4/4** | **100 %** |
| **tomato** | **73 %** | **4/4** | **89 %** |
| pasta_bundle | 43 % | 1 (no pool) | 42-50 % |

**The mushroom's 100 % and the tomato's 89 % were partly SAMPLING ARTEFACTS** — both were measured
on runs that happened to draw only the nominal/easiest mesh. Exercising all four variants gives
80 % and 73 %. Raspberry and cherry_tomato are unchanged across full coverage, so those numbers
were real.

**Lesson: quote smoke numbers only from `--mesh-cycle` runs once a category has a mesh pool.**
A per-category success rate measured on an unknown subset of the pool is not comparable to one
measured on all of it. `docs/smoke_datasets.md` records the recipe per run so the two can be told
apart.


**2026-08-27 (object library) — 13 photo-derived fruit meshes added with PER-CATEGORY MESH
RANDOMIZATION; procedural placeholders retired. New `docs/smoke_datasets.md` history table.**

The cluster agent pushed 17 TripoSG meshes (cherry_tomato 6, raspberry 6, tomato 5) with its own
usability gate (topology + photo-silhouette consistency). Taking the gate's verdict and adding a
**category-appropriateness check it does not do** — volume within 0.5-2x the category median,
aspect < 1.6, euler 2:

| category | gate passed | we kept | dropped, and why |
|---|---|---|---|
| cherry_tomato | 6/6 | **4** | `4` and `6`: volume 0.42x / 0.30x the category median at aspect 1.83 / 1.97 — skinny outliers, not representative cherry tomatoes |
| raspberry | 6/6 | **5** | `6`: euler -1 |
| tomato | 4/5 | **4** | `2` already failed the agent's gate (euler 1) |

⚠ **The euler -1 on raspberry6 was MY bug, not the agent's** — the gate correctly reported euler 2,
and my preprocessing called `merge_vertices` + `fill_holes` on an already-watertight mesh and broke
it. Fixed to repair only what needs it; raspberry6 stays dropped pending a re-run.

Each kept mesh is uniformly scaled to its category's **already-validated** nominal extent (cherry
tomato 25 mm, raspberry 15.4 mm, tomato 60 mm — not new guesses), recentred, and registered.
Per-category `object_mesh_pool` in the DR config now samples the base mesh per scene. Each
category's BASE object points at the variant whose volume is closest to the category median (the
most representative member), since the base's `size` is what drives the auto yaw/squeeze rules.
The procedural placeholder meshes for cherry_tomato and tomato are **retired** — those categories
are now real scans.

**Smoke tests with mesh randomization live (8 episodes each, all-auto recipe):**

| category | success | meshes sampled |
|---|---|---|
| cherry_tomato | **89 %** | cherry_tomato1, cherry_tomato5 |
| raspberry | **100 %** | raspberry5 |
| tomato | **89 %** | tomato1, tomato5 |

(Only 1-2 meshes are sampled in an 8-episode run because the scene rebuilds once or twice; a full
collection exercises the whole pool.)

**New: `docs/smoke_datasets.md`, auto-generated by `gentle_manip.scripts.smoke_table`.** One row
per collection pairing the demonstrator success rate with the synthesis recipe that produced it
(area/width/yaw/squeeze/escalation/azimuth/nu), so any number can be traced back to its
configuration. Hand-maintained tables drift — regenerate instead. Gap found and fixed while
building it: `grasp_yaw_max_deg` and the FEM `nu` were **not** in the config snapshot, so runs
using the yaw bound were not distinguishable from runs without it. Both are recorded now.


**2026-08-27 (paper prep) — `docs/grasp_synthesis_model.md`: the synthesis model VERIFIED against
code, plus the v3.3 delta.** Written for paper writing: every statement checked against the
implementation, with **⚠ DO NOT CLAIM** markers wherever a natural claim would overstate the code.
Covers the FEM formulation (CST linear tets, E=1 normalization, inertia-relief bordered solve,
Schur-complement per-grasp solve), the contact/pad model (position control, real finger STL pad,
normal-only prescribed displacement, flat-plane push with a fillet taper), the E-linearity, the
holdability inequality, the full objective with every weight, the feasibility ladder, and what is
auto vs hand-set. Appendix A is the v3.3 diff (4 bug fixes, 6 new opt-in flags, measured effect of
each).

**Two model/description mismatches found while verifying — both matter for the paper:**
1. **Poisson ratio: every collection to date used ν = 0.33 for EVERY object.** `build_grasp_fem`
   passed no config, so `cfg.nu` fell back to `MetricConfig`'s "copper" default, while the
   materials declare ν 0.30-0.42 and the DR randomizes `object_nu` **for the MPM sim only**.
   Unlike E, **ν cannot be rescaled post-hoc**. Added `--grasp-nu auto` (uses the material value)
   but **defaulted to the historical 0.33** so existing runs stay reproducible. State ν = 0.33 in
   the paper, or re-run first.
2. **μ = 0.7 is one global constant**, not per-object and not randomized — same pad-object friction
   for tofu, mushroom and a wet tomato. (`coup_friction` in the DR configs is a different thing:
   MPM coupling friction in the simulator, not the planner's μ.)

Other items the paper must not overstate, all recorded in the doc: the FEM is a **planning
surrogate**, never run inside the sim loop and not calibrated against the MPM; it runs on a
**voxel-remeshed proxy** (~17 % thicker than the source on a thin body); the headline stress is a
**contact-masked top-10 %**, not a peak; FEM contact is **normal-only** (friction only via the
scalar holdability test); and `w_occ` — the only real occlusion measure — is **0 in every run**,
with occlusion controlled by the azimuth penalty and the hard yaw bound instead.


**2026-08-27 (pinch filter + tomato size) — TWO user catches, both fixed: the auto area floor still
admitted PINCHES, and the tomato was sized past the gripper's comfortable range.**

**1. Pinch / strange-grasp filter.** User flagged `banana_chunk .../26-08-27-qrp/.../ep0004_env3_
success_grasp.png`: **align 0.541, grip 0.83 N, width 35.1 mm** on a 33.7 x 35 x 20.4 mm object —
jaws nearly fully open, fingertips catching the top corner, contact patches tiny and on the upper
edge. A textbook pinch that the `area_min="auto"` floor let through, because when the whole
feasible pool is mediocre the "upper half by area" is still mediocre.

Fix: the auto selection now keeps the upper half by **BOTH contact area AND alignment**, then takes
the best score. `align` is the right second signal — it is already computed, and it discriminates
strongly (banana chunk: lifts averaged **0.83** vs **0.53** for failures). Both criteria are
POOL-RELATIVE medians, so this stays scale-free and adds no fitted constant; if the two together
empty the set it falls back to the area criterion alone rather than returning nothing.

Measured on the banana chunk, three runs (before / before / after):

| run | align med | align min | grasps < 0.6 | success |
|---|---|---|---|---|
| qym (before) | 0.82 | 0.41 | 2 | 12/16 |
| qrp (before, the flagged pinch) | 0.80 | **0.52** | **4** | 11/16 |
| **ofx (after)** | **0.86** | **0.60** | **1** | 11/16 |

Bad grasps cut without costing yield.

**2. Tomato size vs the GRIPPER's limit — a real hardware constraint, quantified.** The gripper
opens **88 mm** physically, and the planner bounds width at **79 mm**. So:

| tomato size | width needed | margin under 79 mm | demonstrator success |
|---|---|---|---|
| 6.5 cm (first guess) | ~65 mm | 14 mm | 57-62 % |
| **6.0 cm (adopted)** | ~60 mm | **19 mm** | **81-89 %** |
| 7.5 cm (real local size) | ~75 mm | **4 mm** | not viable |

**A realistically-sized 7-8 cm tomato is effectively out of range for this hand** — 4 mm of margin
leaves no room for the pads to indent, which is why the larger versions performed worst. 6 cm is
the user's call and it lifts the tomato from ~60 % to ~85 %. **If a realistic whole tomato is ever
needed, it is a gripper problem, not a synthesis problem.** The cherry tomato (2.5 cm) is the
realistic tomato category for this rig.

**Current cross-object state (8-episode smoke tests, all-auto + yaw/squeeze/align filters):**

| object | success | align med | align min | < 0.6 | stress % of yield |
|---|---|---|---|---|---|
| mushroom | **8/8 = 100 %** | 0.94 | 0.94 | 0 | 30 % |
| tomato (6 cm) | **13/16 = 81 %** | 0.96 | 0.90 | 0 | 31 % |
| cherry_tomato | **6/8 = 75 %** | 0.94 | 0.86 | 0 | 52 % |
| banana_chunk | 11/16 = 69 % | 0.86 | 0.60 | 1 | 43 % |
| pasta_bundle | 10/24 = 42 % | 0.85 | 0.46 | 2 | 46 % |

Alignment is now high everywhere (median 0.85-0.96) and sub-0.6 grasps are rare. **pasta_bundle at
42 % is the weakest and still has the worst alignment floor (0.46)** — it is the elongated one, and
the natural next suspect is the same thin/elongated regime that got the full banana parked, though
it is nowhere near as severe.


**2026-08-27 (occlusion + squeeze) — USER CAUGHT A REAL GAP: the 60 deg camera-azimuth bound is a
SHAPED PENALTY, and at 60 deg a SMALL object is COMPLETELY hidden by the gripper. Added a HARD
yaw bound and a size-scaled squeeze, both auto.**

User flagged `cherry_tomato .../26-08-27-pqs/videos/ep0004_env3_success.mp4` as showing ~90 deg yaw
with the object occluded. Audit of what actually executed:

- The azimuth bound **was applied and WAS holding** — max achieved azimuth across every run was
  60.0-60.6 deg, never near 90. So the mechanism works.
- **But it is a shaped penalty on the CLOSING-AXIS angle**, and it says nothing about whether the
  gripper BODY covers the object. Frame extraction from that video confirms the 25 mm tomato is
  fully hidden behind the fingers at the grasp and lift. **19-25 % of grasps sat exactly AT the
  60 deg bound** — i.e. the optimum was outside and they were parked at the most-occluding pose
  still allowed.
- **This matters for the POLICY, not just the video:** `single_lift.py` builds ONE camera,
  `cam_ext` at the calibrated L515 pose, and `superset_soft_armfocus` takes its point cloud from
  `cameras: ["cam_ext"]`. The render camera IS the observation camera.
- Occlusion risk scales INVERSELY with object size, exactly as the user predicted: over-60 deg rate
  was cherry tomato (25 mm) 25 %, mushroom (33 mm) 19 %, banana chunk (35 mm) 12 %, tomato (65 mm)
  **0 %**. A big object is only ever partially covered; a small one vanishes.

**Fixes — two new auto params, each on the descriptor that fits its physics:**
- `--grasp-yaw-max-deg auto` — a HARD structural bound (unlike the azimuth penalty), interpolating
  30 deg at 25 mm -> 75 deg at 65 mm on the object's **LARGEST** extent. Largest, because what
  matters is how much of the object's SILHOUETTE a finger can hide.
  **First attempt used the SMALLEST extent and that was wrong**: it handed the 60 mm-long,
  25 mm-thick pasta bundle the TIGHTEST bound (30 deg) and cost it half its yield (50 % -> ~17 %).
  Corrected to largest extent, pasta recovered to 44 %.
- `--grasp-extra-close auto` — the fixed 5 mm squeeze is 15 % of the 33 mm mushroom it was tuned on
  but 24 % of a 21 mm cherry tomato and 34 % of a 15 mm raspberry. auto = 5 mm x (smallest extent /
  33 mm), clipped [2, 6] mm, on the **SMALLEST** extent because that is the grasp direction. The
  mushroom is unchanged at 4.8 mm. (The separate `FIRM_EXTRA_CLOSE_M` = 2.5 mm is still a constant
  and is NOT scaled — a known remaining gap.)

**Result — occlusion fixed, no yield lost, squeeze gentle:**

| object | success before -> after | azimuth max before -> after | compression med |
|---|---|---|---|
| mushroom | 100 % -> **100 %** | 60.6 -> **52.5** | ~0 % |
| cherry_tomato | 81 % -> **80 %** | 60.5 -> **39.2** | **4 %** (max 18 %) |
| banana_chunk | 75 % -> **80 %** | 60.6 -> **38.3** | ~0 % |
| tomato | 62 % -> 57 % | 60.0 -> 60.1 (bound 75) | ~0 % |
| pasta_bundle | 50 % -> 44 % | (bound 69.4) | ~0 % |

The small objects that used to vanish now stay well inside the cone (cherry 60.5 -> 39.2 deg), and
nothing lost yield beyond noise. **Over-squeeze is NOT a problem at these settings** — the cherry
tomato sits at 4 % median compression (18 % worst), versus the 54 % the full banana needed.

**Caveat on the compression column:** it is `1 - width_at_peak / smallest nominal extent`, so it
goes NEGATIVE when the grasp is across a wider axis than the smallest one (pasta -148 %, banana
chunk -43 %). Those negatives are an artefact of the denominator, not real numbers — only the
non-negative values (cherry, mushroom) are meaningful as stated.

**2026-08-27 (later) — tomato meshed (12/15); `shape_consistency.py` CORRECTED TWICE in one
session — the first two versions of my own new gate were both wrong.**
tomato: 12/15 pass, 5 meshes in `obj_meshes/tomato/selected/`. Only tomato2 fails (0/3,
euler -15/1/0) — its calyx is DRIED, brown, papery sepals lying flat and thin against the
fruit (parallel sheets). tomato4's raised green calyx arms are rounded/tubular and stand
clear -> passed 3/3. **Thinness alone is not the predictor; PARALLEL PROXIMITY is.**
(Prediction was half right: called tomato2 and tomato4; tomato4 was wrong.)

**Correction A — the sorted-extent metric was wrong for OBLATE objects.** v1 compared
(2nd-largest / largest extent) against the photo's min/max silhouette aspect. For a
wide/wide/short object that ratio saturates near 1.0 regardless of flatness, so all three
tomato2 seeds were flagged at ~25% when they are in fact correct. TripoSG emits +X = image
right, +Y = image up, +Z = depth (verified: mushroom1/back_seed0 mesh X/Y = 0.918 vs photo
W/H = 0.918). The right in-plane test is **mesh X/Y vs photo W/H**, not sorted extents.

**Correction B — in-plane aspect ALONE cannot catch a depth hallucination.**
`cherry_tomato1_seed1` (bbox [0.52,0.58,1.90], a ROD from a round tomato) has in-plane
X/Y = 0.901 vs photo 1.009 — only 11% off, it PASSES the in-plane test. The rod extends
into DEPTH, which no silhouette test can see. Needs a second test: **depth ratio
Z / max(X,Y)**, which puts it at 3.27.

**Correction C — the depth test must be ONE-SIDED.** A first cut used
DEPTH_LO=0.35 and falsely flagged 18 of 24 shrimps plus the selected banana. For an
elongated or curled object max(X,Y) is the LENGTH and Z is the THICKNESS, so a flat shrimp
curl legitimately measures 0.20-0.30. HIGH depth is implausible for single-view
reconstruction; LOW depth is normal. Bounds now 0.02-2.5.

**Final state:** 114 candidates audited, exactly **1** silent failure —
`cherry_tomato1_seed1` (depth 3.27), correctly demoted; cherry_tomato1 now selects seed0.
The earlier shrimp4 re-selection is REVERTED: with the corrected in-plane metric seed1
scores 23.7% (not 50%), inside tolerance, so seed1 (euler 2) is chosen again over seed2.
**Lesson: a new validation gate needs its own calibration set before its verdicts are
trusted — mine produced 2 false positives and 1 false negative across three iterations,
and each was only caught by checking flagged meshes against their renders.**

**2026-08-27 — cherry_tomato + raspberry meshed; TWO corrections to the failure model and a
NEW GATE (`shape_consistency.py`).** Pass rates: cherry_tomato **16/18**, raspberry **12/18**
(one mesh per image; `selected/`). `obj_images/tomato/` is EMPTY — not run.

**Correction 1 — the mechanism is PARALLEL surfaces, not "thin detail" or "crevices".**
I predicted raspberry would be the worst object yet (druplets + hollow core + styles) and
cherry-tomato calyxes would fail like the strawberry. Both wrong. Raspberry scored 12/18
with only mild failures (euler 0/-2/1, never the shrimp's -64), and all three calyx-bearing
tomatoes passed 3/3. Why: druplets are CONVEX bumps whose inter-lobe valleys are V-shaped —
the surfaces DIVERGE. A tomato calyx lies flat against the fruit. The catastrophic cases
(shrimp tail fan, strawberry sepals, mushroom3's torn stem) are all roughly-PARALLEL sheets
separated by less than a voxel. Refined rule: **bumpy convex geometry is cheap; thin
parallel geometry is not.** The only cherry_tomato failures were image 5 — a thin curved
stem PROTRUDING, i.e. genuinely thin, not a calyx.

**Correction 2 — the section 6 gates cannot detect a geometrically WRONG mesh, and one
shipped.** All section 6 checks are TOPOLOGICAL. `cherry_tomato1_seed1` passed every one of
them (watertight, euler 2, genus 0, 12000 faces, positive volume, 1 component) while being a
**3.63:1 ROD generated from a photo of a round tomato, with 13x too little volume**
(bbox [0.52,0.58,1.90] vs its siblings' [1.91,1.84,1.87]). Caught only because the turntable
normalises by max vertex radius, so it rendered visibly small.

**NEW: `scripts/mesh_from_photos/shape_consistency.py`** — compares the mesh's silhouette
proportion (2nd-largest / largest extent) against the input photo's alpha-bbox aspect;
flags > 25% relative error. Calibration on known-good meshes: mushroom1/back_seed0 agrees to
1%, front_seed0 to 6%, so 25% is loose and still catches the rod by a mile. Audited all 99
candidates → exactly **2 silent failures**: `cherry_tomato1_seed1` (69% off) and
**`shrimp4_seed1` (50% off) — which had been SELECTED and delivered.** It was shrimp4's only
euler-passing seed, so the old ranking picked the least faithful of its three.

**`select_per_image.py` ranking changed: shape consistency is now the PRIMARY key, above the
section 6 gates.** Justification: genus > 0 does not block tetgen (closed + manifold +
self-intersection-free is what a tet mesher needs), whereas wrong shape is fatal. shrimp4
re-selected seed1 → **seed2** (shape err 14.2%, euler -7) over seed1 (50%, euler 2). Every
`selected/README.md` now carries a shape-err column. It is a SCREEN not a proof — it only
sees silhouette proportions, so a wrong mesh that preserves aspect ratio still passes.

Also: `.avif` added to the prep extension filter — `raspberry4/6` would have been silently
skipped, same class of bug as shrimp6's `.webp`. Filter now jpg/jpeg/png/webp/avif/bmp/tif/tiff.

**2026-08-27 (smoke test) — FOUR NEW OBJECTS through the all-auto recipe: cherry tomato, tomato,
banana CHUNK, pasta bundle. All synthesize; two needed fixes, both found and applied.**

**Meshes.** Procedural for the tomatoes and the pasta bundle (`scratchpad/make_meshes.py`:
oblate spheroids with optional lobes; a 7-strand bundle union'd and voxel-remeshed into one
solid), and the **banana chunk cut from the REAL banana scan** (two-plane cut through the thick
middle, scaled to 35 mm). Procedural was chosen over sourcing meshes online for reliability inside
the time box and because the nominal extent can be set exactly — **these are shape-class probes for
the SYNTHESIS pipeline, not calibrated food models. Real scans (e.g. via the TripoSG path already
used for the banana/shrimps) should replace them before any of these is used for real collection.**
All materials are literature-plausible, NOT measured.

| object | extents | E | yield | nominal source |
|---|---|---|---|---|
| cherry_tomato | 2.5 x 2.5 x 2.1 cm | 0.4 MPa | 30 kPa | procedural oblate |
| tomato | 6.4 x 6.5 x 4.9 cm | 0.3 MPa | 25 kPa | procedural, 5-lobed |
| banana_chunk | 3.4 x 3.5 x 2.0 cm | 0.25 MPa | 25 kPa | **cut from the real banana scan** |
| pasta_bundle | 6.0 x 2.5 x 2.5 cm | 0.12 MPa | 15 kPa | procedural 7-strand bundle |

**Results (8-episode smoke tests, all-auto recipe, unchanged otherwise):**

| object | demonstrator success | stress med | % of yield | align | min_pad |
|---|---|---|---|---|---|
| cherry_tomato | **13/16 = 81 %** | 17.3 kPa | 58 % | 0.94 | 19.6 mm2 |
| banana_chunk | **12/16 = 75 %** | 9.7 kPa | 39 % | 0.82 | 14.8 mm2 |
| tomato | **10/16 = 62 %** | 8.1 kPa | 32 % | 0.96 | 68.0 mm2 |
| pasta_bundle | **8/16 = 50 %** | 6.2 kPa | 41 % | 0.86 | 38.8 mm2 |

**BANANA CHUNK IS THE HEADLINE: 75 %, against the full banana's 0-12 %.** Same material, same
source mesh, cut compact (elongation **1.72** vs the full banana's **5.12**). This is INDEPENDENT
confirmation of the parked-banana diagnosis — the blocker was thin+elongated geometry falling
outside the contact model's small-strain validity, not anything about bananas. **If a banana
category is ever wanted, use chunks.**

**FIX 1 — pasta bundle: 0 % -> 50 %, caused by my material guess, not by shape.** At the first
guess (E 0.03 MPa, "cooked pasta") NOTHING lifted in 12 synthesized grasps, and the failure did
not look like the banana's: contact area was LARGE (52.9 mm2), align 0.76, stress only 31 % of
yield — i.e. every metric said the grasp was good and it still failed. That pattern says material,
not geometry. Refirmed to **E 0.12 MPa / yield 15 kPa** ("al dente" bundle, substeps 110 -> 220)
and it went to 50 %. **Lesson: an E chosen too soft produces grasps that look perfect on every
synthesis metric and simply squash out.** Worth remembering when adding any new soft object.

**OPEN — tomato at 62 % is the weak one.** Its failures are not low-contact (min_pad 57.5 on
failures vs 71.8 on lifts — both large) and align is high (0.93/0.96). The likely cause is SIZE:
at 6.5 cm the fruit needs 70-75 mm grasp widths against an 88 mm gripper opening, so there is very
little margin and the pads sit on a strongly curved surface. Not diagnosed further; if a large
round object is wanted, this is the case to look at.

**Also noted:** 8 real scanned SHRIMP meshes already exist in `obj_meshes/shrimps/selected/`
(shrimp1-8, TripoSG output, most watertight). Not tested here for time; shrimp is curved and
elongated (~0.24-0.35 thickness ratio) so it is a good future test of where the elongation limit
actually bites.


**2026-08-27 (resolved) — CONFIRMED against the ACTUALLY EXECUTED poses: the FEM contact model
cannot score ANY of the grasps that lift the banana. It is a VALIDITY-DOMAIN limit, not a bug.**

The previous entry left the mechanism unconfirmed because the probe pose was reconstructed rather
than executed. Redone properly: poses taken from the RECORDED episodes of the scripted run that
lifted 8/8 (ee_pos + ee_quat + gripper_width at the grasp instant, object orientation from that
env's `dr_params.csv` row), i.e. exactly what the simulator ran.

| status assigned to poses that ALL LIFTED | count |
|---|---|
| `degenerate` | 5/8 |
| `no_contact` | 3/8 |
| **`ok`** | **0/8** |

**Not one of the eight working grasps is scoreable.** All receive the infinite-penalty score. This
closes the question for good: no seeding, budget, width cap or area floor could ever have made CMA
pick them.

**The mechanism, and why it is NOT a defect to "fix" casually.** The executed grasps close to
**8.5 mm on an 18.6 mm-thick banana** — 54 % compression. `indent_from_width` evaluates contact
against the **NOMINAL, UNDEFORMED mesh**, so it sees the pads buried ~10 mm inside solid material
and stamps `degenerate` (past `max_indent` = 0.01 m, the small-strain validity limit). The real MPM
body simply deforms and conforms. The model is behaving CORRECTLY within its assumptions; the
banana's working grasps are outside those assumptions.

**This is why the compact objects are unaffected.** Mushroom / tofu / strawberry / raspberry lift at
92-100 % because their working grasps involve only a few mm of deformation on a ~30 mm body —
comfortably inside small strain. Only the banana (thin + needing large deformation to get grip) sits
outside. The synthesis pipeline is sound on 4 of 5 objects; it has a scope limit, not a corruption.

**Practical route that works TODAY, without touching the contact model:** the geometric heuristic
(object centre, top-down, closing perpendicular to the longest axis) lifts the banana **80 % at
26 mm with only 11 % compression** — usable, gentle demo data. Where the FEM metric is valid
(compact objects) use CMA+FEM; where it is not, use the heuristic. The open design question is how
to DECIDE between them automatically without inventing another fitted threshold: a thickness-vs-
`max_indent` test misfires (raspberry is 14.3 mm thick yet grasps fine at small indentation), so the
signal probably has to come from the search result itself, not from a shape statistic.

**A real contact-model fix would mean evaluating contact in the DEFORMED configuration** (iterate
deform -> re-check) rather than against the nominal mesh, and a large-deformation-capable stress
model, since the linear FEM's stress numbers are not trustworthy at 54 % strain either. That is a
substantial piece of work, not a tuning change.


**2026-08-27 (later still) — WHY the banana's good grasp is never chosen: the FEM metric marks it
INFEASIBLE. Seeding CMA with it does nothing; the defect is the CONTACT MODEL, not the search.**

Follow-up to the scripted-grasp result. Tried the obvious fix first — force the FIRST CMA start to
be the pose we know works (object centre, top-down, closing PERPENDICULAR to the longest axis),
always on and NOT gated on an elongation threshold (on a compact object the long axis is
near-arbitrary, so it is just one more sensible top-down seed; elongation for reference: mushroom
1.09, strawberry 1.23, raspberry 1.06, banana 5.12).

**It changed nothing — the banana run came back BIT-IDENTICAL** (env0 22632 Pa / 54.2 mm, same as
before the seed). CMA simply optimises away from the seed. So the search initialisation was never
the problem.

**Scoring the known-good grasp directly shows why** (banana flat at nominal pose, correct TCP z
built the same way the seed loop does):

| grasp | score | status | stress |
|---|---|---|---|
| long-axis centre, w = 18 / 22 / 26 / 30 / 34 mm | -1.3e8 | **`degenerate`** | inf |
| CMA winner, w = 79 mm | -47 393 | `ok` | 19.6 kPa |

**Every width of the grasp that lifts 80-100 % is stamped `degenerate` and given an
infinite-penalty score, so CMA can never select it.** No amount of seeding, budget escalation or
width capping could ever have found it — which retrospectively explains why none of those levers
moved lift success while they all moved synthesis feasibility.

`degenerate` comes from `width_grasp.indent_from_width`: it fires when a jaw indents deeper than
`max_indent` (default **0.01 m**), the small-strain validity limit of the FEM contact model. And
the successful scripted grasp compressed the banana from 18.6 mm to 8.5 mm — **~10 mm of total
indentation, exactly at that limit**. That is the shape of the problem: **for a THIN SOFT object
the grasps that actually hold sit outside the contact model's validity domain, while everything
inside it is too loose to lift.**

**UNRESOLVED / do not over-read.** Raising `max_indent` to 0.02 did NOT clear the status, and the
implied indentation works out to ~146 mm — larger than the banana itself. So the pose reconstructed
in the probe is probably NOT the pose that actually executed (the executor CLIPS tcp z into the
workspace, and the scripted run's raw z was below the table — see the earlier caveat). The exact
`degenerate` trigger therefore still needs confirming against the pose that really ran. What is
solid is the ranking result: the metric rejects the known-good family and accepts the
known-to-fail one.

**Status of the attempted fix: REVERTED.** The long-axis seed is provably inert here (bit-identical
output) and it perturbs `yaws[0]` for every other object for no benefit, so it was not kept. It
becomes the right change only AFTER the contact model can score such a grasp.

**Where to go next** — the work is in `smgrasp/width_grasp.py`, not in the search:
1. Confirm the `degenerate` trigger against the ACTUALLY EXECUTED pose (log the post-clip tcp from
   a scripted run and score exactly that).
2. The indent is measured to the EXTREME boundary point inside the rectangular pad footprint. On a
   CURVED body a lengthwise pad spans the crescent's curvature, so that extreme is far away and the
   model infers a huge burial. A first-contact/penetration-depth measure would not.
3. Only then re-add the long-axis seed.


**2026-08-27 (later) — DECISIVE: the banana failure is the SYNTHESIZED POSE, not the physics.
A hand-scripted centre grasp lifts it 80-100 %; CMA manages ~12 %.** User's proposed experiment:
script a grasp at the banana's centre with the pads aligned along its long axis (closing across
the ~18 mm width), sweep the width, and see whether it lifts. If it lifts, synthesis is the
problem; if not, the sim dynamics are. Implemented by monkeypatching `synthesize_grasp` to return
the scripted pose, leaving the ENTIRE v3.3 execution FSM (approach / close / firm / lift / hold /
success test) untouched.

| commanded width | lift rate | width at peak | compression |
|---|---|---|---|
| 12 mm | **100 %** (8/8) | 2.5 mm | 87 % |
| 15 mm | **100 %** | 5.5 mm | 70 % |
| 18 mm | **100 %** | 8.5 mm | 54 % |
| 22 mm | 89 % | 12.5 mm | 33 % |
| **26 mm** | **80 %** | 16.5 mm | **11 %** |
| 30 mm | 67 % | 20.5 mm | ~0 % |
| 34 mm | 30 % | 22.0 mm | ~0 % |

Verified as REAL lifts, not just saved episodes: object centre rises 11.2 mm -> 146.8 mm
(135 mm), 8/8 clear of the 50 mm bar.

**Conclusions:**
1. **The simulation can lift the banana reliably.** Every hypothesis about MPM contact / grid
   resolution / slip as the blocker is dead.
2. **CMA + the FEM gentleness metric are the blocker.** A one-line geometric heuristic (centre,
   pads along the long axis) beats the whole synthesis pipeline by 7x on this object.
3. **A GENTLE banana lift exists**: 26 mm -> 80 % success at 11 % compression. So the object is
   not intrinsically ungraspable, and the earlier "grasp->lift transition" framing was wrong --
   the transition is fine when the pose is right.
4. The trade-off is smooth and monotonic (tighter = more reliable, looser = gentler), which is
   exactly the knob the gentleness objective should be riding.

**Caveat on the readout:** the scripted TCP z was computed as `com_z + FINGER_TO_TCP_Z` (copied
from the collector's own fallback), and that constant is NEGATIVE (-0.0699), so the raw pose sits
below the table and the FEM scorer stamped every one `STATUS=table`. The executor clips z into the
workspace, so what actually ran was the clipped pose. **Do NOT read the `STATUS=table` output as
evidence that the table filter rejects good grasps** -- it reflects the bad z that was handed to
the scorer. (It does, separately, suggest the collector's fallback expression is suspect and worth
a look.)

**Next step (not yet implemented):** feed this heuristic into synthesis rather than replacing it --
seed CMA with the centre / long-axis-perpendicular pose at a width near the local cross-section,
and replace the fixed-45 mm fallback grasp with it. Both are minimal changes and both are strictly
better than what is there now.


**2026-08-27 — CROSS-OBJECT recipe: ALL-AUTO grasp params work on mushroom / strawberry /
raspberry / tofu with NO per-object tuning. Banana remains the sole outlier. Plus a REAL BUG:
every non-mushroom collection had been planning grasps with the MUSHROOM's material.**

**The bug first — `--grasp-E` defaulted to 3e5 (and density to 1000) for EVERY object and was
never derived from the material.** So every banana / strawberry / raspberry / tofu collection ever
run planned grasps with the mushroom's stiffness. This is not cosmetic: the FEM is linear in E
(sigma = E*sigma_1, F = E*F_1), so BOTH the predicted stress AND the GRIP FORCE were wrong per
object. On the raspberry (true E 1e5) the planner believed it had **3x the grip it actually had**
and reported 24.8 kPa where the truth is ~6 kPa; on the banana it assumed 20 % more grip than it
has, i.e. it was planning grasps too loose to hold. `--grasp-E/-density/-yield` now DEFAULT to the
object's registry material (explicit values still override) and the resolved values are printed at
launch. **Any stress number reported for a non-mushroom object before this date is wrong.**

**Two auto params, replacing the hand-set constants** (mushroom 20 / strawberry 15 / raspberry 4 /
banana 20 mm2 area floors were all guesses):
- `--grasp-width-max-mm auto` — 2.3 x the median LOCAL cross-section perpendicular to the long
  axis. Inert on compact objects, binds on elongated ones. Explicitly NOT the bbox, which ranks
  the banana largest/easiest when its graspable width is ~18 mm.
- `--grasp-area-min-mm2 auto` — search with NO hard floor, then keep the upper half of the
  feasible pool by worst-pad contact area. Contact area is the strongest predictor of whether a
  grasp LIFTS (banana, 76 grasps: min_pad 37.4 mm2 on lifts vs 24.4 on failures; 0/16 lifts below
  15 mm2), while stress does NOT discriminate. Crucially this only SELECTS among grasps already
  found, so unlike a raised hard floor it cannot force a squeeze (area_min 35 gave 2/8 feasible at
  32-38 kPa because area grows with indentation).
- **YIELD GUARD** (`YIELD_SAFETY = 0.8`): area and stress are coupled, so "largest area" alone
  over-squeezes small soft objects — measured on the raspberry at **165 % of its 15 kPa yield**.
  The auto rule now restricts to grasps under 0.8 x yield BEFORE ranking by area.

**Results — one recipe, five objects, no per-object parameters** (16 episodes each, full DR):

| object | demonstrator success | stress med | peak | yield | % of yield | min_pad | align |
|---|---|---|---|---|---|---|---|
| mushroom | **16/16 = 100 %** | 11.6 kPa | 20.0 | 40 | 29 % | 50.4 mm2 | 0.94 |
| tofu | **23/24 = 96 %** | 3.0 kPa | 6.0 | 20 | 15 % | 365 mm2 | 0.98 |
| strawberry | **22/24 = 92 %** | 10.8 kPa | 18.6 | 18 | 60 % | 44.1 mm2 | 0.92 |
| raspberry | **16/16 = 100 %** | 6.0 kPa | 7.7 | 15 | 40 % | 19.9 mm2 | 0.89 |
| banana | **0/16 = 0 %** | — | — | 25 | — | — | ~0.55 |

**Tofu needed NO special alignment or prior knowledge** — it came out at 96 % and the gentlest of
all (15 % of yield, align 0.98), straight from the same auto recipe. Strawberry sits at 60 % of
yield, the closest to bruising of the four that work; worth watching.

**The banana still fails, now for a understood reason.** Synthesis is fine (8/8) but the AUTO width
cap resolves to ~54 mm on the DR-scaled mesh — looser than the **40 mm** that was hand-validated —
so grasps drift back toward the long axis (align ~0.55) and none lift. The auto descriptor is
measured on the FEM mesh, which is voxel-remeshed and ~17 % thicker than the source for a thin
body, and the 2.3 coefficient was fitted to one object. For the banana specifically, pass
`--grasp-width-max-mm 40` explicitly.

**Recommended cross-object recipe** (v3.3 otherwise unchanged — this is the minimal diff):
```
--grasp-area-min-mm2 auto --grasp-width-max-mm auto    # + explicit 40 for the banana
```
E / density / yield now come from the object automatically.

**Open:** the banana is the one object where every lever tried (mesh fix, escalation, budget,
azimuth, width cap, all-auto) moves synthesis or plan quality but never lift success. Its failure
is at the grasp->lift transition. Do not collect banana data until that is understood.

**2026-08-26 — Literature scan on policy size-adaptation (items 17/18) →
`docs/size_adaptation_literature.md`.** Headline: **no published work does our controlled
study** (single parallel-jaw IL/diffusion policy measured for grasp-APERTURE adaptation
across object scale) — the width-vs-scale correlation looks like a novel diagnostic.
Closest precedent for the fix is **RMA-style privileged-vector regression**: Liang, Ellis &
Henriques (CVPR 2024, arXiv 2312.04670) infer object **mass and shape** from action+proprio
history on a manipulator; original RMA (arXiv 2107.04034) is the recipe. We are already
half-way there — `priv_object_dr_params=[scale,bend]` is consumed by the state teacher; the
missing half is the adaptation module for the point-cloud student.
**On the single-view worry (user):** do NOT build on full amodal shape completion — that
literature is explicit that single-view completion is uncertain and hallucinated geometry
breaks grasp planning. Reframe: we need ONE scalar (extent along the gripper closing axis),
which for a fixed external camera is usually *perpendicular to the camera ray* and hence
visible; and post-contact, gripper width + contact_force measure size directly. **Decisive
cheap first experiment: offline-regress `priv_object_dr_params[0]` from the ALREADY-RECORDED
demo point clouds.** That separates "student can't see size" from "student sees size but
ignores it" — which current evidence does not distinguish — before any architecture change.
If it sees but ignores, that is representation collapse (D2PPO dispersive loss, AAAI 2026).
Equivariant policies (EquiBot CoRL 2024) are the expensive option AND carry a caveat: SIM(3)
scales translations, but our gripper DoF is a separate normalised scalar — check the action
head. Scan is NOT systematic; escalate before any novelty claim in print.


**2026-08-27 — ⛔ BOTH "head drives width" variants FAIL end-to-end. Timing cannot be
delegated; only LEVEL can. The floor design is what survives.**
· SIGHTED head: offline MAE 0.6 mm, closed-loop **0.000** success (probe AND canonical eval) —
  it COPIES the current width (80→79.4, 28→28.6), so closure is never initiated; gripper
  creeps 80→54 mm.
· BLIND head (gripper width zeroed from its input): offline it genuinely LEARNS the ramp
  (`78 81 80 … 66 42 20 … 32 28 29 32`, MAE **3.9 mm** vs true, so vision+pose DO carry timing
  and level) — yet closed-loop it is **0.000** too, width stuck ~68 mm, lift onset detected in
  only 2% of episodes. That is DISTRIBUTION SHIFT: offline it is teacher-forced on DEMO arm
  trajectories; online the arm follows the POLICY and the visual cue it learned never fires.
CONCLUSION (twice-confirmed, two different failure modes): the closure DECISION must stay with
the policy — its width command is the only width signal trained closed-loop with its own
feedback. A feedforward head can supply the LEVEL, not the TIMING.
⇒ The floor `w_cmd = max(w_policy, w_level(scene))` is now the design of record. It is not a
hack: it is the only decomposition that survives both experiments, and the user's original
objection (per-step is the hard part) was the correct instinct. NOTE the level predictor
should be the per-EPISODE scalar aux head (rturn-style, r=0.82) — a constant target, so
copying is impossible by construction — not the trajectory head.
QUOTA: cancelled the 5 orphaned Config-2 checkpoint evals (sighted-head ckpts ⇒ guaranteed
0.000) left queued by their watchers.


**2026-08-27 — ⛔ WIDTH-HEAD-DRIVES-WIDTH FAILS: the head learned to COPY its proprio input,
so nothing ever initiates closure (0.00 success). Config 2 killed; blind variant launched.**
End-to-end test of Config 1 (`state_600_whead.pt`, head spliced over the width dim, banner
confirmed active): success **0.000** on BOTH the width probe and the canonical 200-episode
eval (succ 0.000 / ever 0.000), commanded width stuck at 53-55 mm — two independent
measurements agree, so the SIGHTED variant is conclusively dead. Mechanism proven by
a controlled sweep — fix the cloud/pose, vary ONLY the current gripper width fed in:

| current width in | head's next 4 out |
|---|---|
| 80.0 mm | 79.4 79.4 79.4 79.5 |
| 60.0 mm | 58.5 58.1 56.6 56.6 |
| 45.0 mm | 43.5 42.9 42.2 41.7 |
| 28.0 mm | 28.6 28.4 28.2 27.9 |

It is a COPIER (echo the input, decrement ~0.5 mm/chunk). The true demo trajectory is
`80 … 77.7 → 50.3 → 33.1 → hold` — a sharp DECISION to close, cued by the visual state. The
head never learned it because copying already minimises the MSE (the ramp is ~5 of ~200
frames). Driven by the head, the gripper crept 80→54 mm and never grasped.
**MY ERROR:** I claimed proprio "answers the per-step objection". It makes the head's
SUPERVISED task well-posed but simultaneously hands it the answer, so copying dominates and
the head cannot DRIVE the channel. The user's instinct that per-step is the hard part was
better than my response. It also VINDICATES the clamp the user called hacky: every viable
variant keeps the POLICY for timing and the head for level — that decomposition is
structural, not a hack.
ACTIONS: Config 2 (3 seeds) cancelled — width comes solely from the head there too, so it
shares the flaw exactly. New `network.width_head_blind` zeroes the gripper-width entries of
the proprio slice before the head, forcing it to infer level AND timing from vision+pose;
frozen retrain on lulkx@600 running (job 1726896). **BLIND HEAD RESULT: it DOES learn the ramp — vision+pose carry timing AND level.**
Full-trajectory prediction (val ep0, mm): blind `78 81 80 80 78 79 80 | 66 42 20 | 32 28 29
32 33` vs true `80 80 80 80 80 80 80 | 78 50 33 | 32 32 32 32 32` — MAE **3.9 mm** over 3
episodes (the sighted head's 0.6 mm was cheating). So the earlier failure was ENTIRELY the
copy shortcut, not missing information. WATCH: the blind head overshoots to 20 mm
mid-transition (13 mm tighter than the demo) — exactly the transient that would crush;
the canonical eval + width probe (jobs 1726965/66) will show whether it matters.
PROCESS NOTE (my bug): the FIRST 'blind' run was not blind — my sed patched `nc.` while the
trainer used `net_cfg.`, so it trained a SIGHTED head that I then masked at inference,
producing a spurious 22.1 mm MAE. Caught because val corr 0.999 was inconsistent with the
inference error; the trainer now PRINTS its resolved flags so this cannot recur silently.


**2026-08-27 — burial fix (d4aafeb) checked against OUR collections: no retro-action needed.**
The local agent's soft-body spawn-burial fix (rotation about the centroid drops an elongated
object's tip below the table) raised the question of whether our MUSHROOM data is affected —
the mushroom's centroid sits ~14.8 mm above its base with a ~20 mm max radius, so an extreme
rotation could in principle bury it ~5 mm. Checked empirically by success vs spawn tilt:
· v3.3 anchor (n=696): flipped 0.963 vs upright 0.929; tilt>30° 0.948 vs 0.916
· v3.4 smoke (n=64): flipped 1.000 vs upright 0.909
· tofu 650 (n=616, already spawn-z-fixed): 0.986 vs 1.000
Tilted/flipped spawns collect AS WELL OR BETTER everywhere, so burial never materially bit
the compact objects — matching their note. Their fix is raise-only (previously-correct spawns
untouched) and protects future elongated produce. Also adopting their second fallback marker:
a MISSING `<episode>_grasp.png` means synthesis failed → fallback grasp (the renderer returns
silently when the FEM has no contact), which is a cheaper check than the all-zero audit columns.


**2026-08-26 — Banana: mis-prepped MESH fixed + escalating CMA budget added. Synthesis
feasibility much improved; end-to-end demonstrator success only 0.38 -> 0.42, so the banana is
STILL NOT ready for collection — the bottleneck moved from synthesis to execution.**

**2026-08-27 — v3.4 smoke (yaw≤55° home-frame · n-grasp 30 · squeeze 3 mm): 93.8%, IDENTICAL
to the v3.3 anchor, with better contact on every axis.** `26-08-26-cdg`, 64 attempts, all
three flags verified in the launch line.

| | v3.3 anchor (clq, n=696) | v3.4 (cdg, n=64) |
|---|---|---|
| demonstrator success | 0.938 | **0.938** |
| CMA grasp width | 39.0 ± 9.3 mm | **43.7 ± 9.5 mm** |
| contact pad | 44 mm² | **51 mm²** |
| contact pressure | 52.3 kPa | **46.4 kPa** |
| align | 0.897 | **0.926** |
| fallback grasps | 2 | 0 |

READ: the +4.7 mm wider commanded width is the lever the REAL evidence pointed at (gentle
alzey's data commanded 34.1 mm vs crushing v3.3's 31.3), now combined with alzey's slower
30-step ramp — and it costs NOTHING in demonstrator success. Lower pressure on a bigger pad
is the gentleness signature we want. CAVEATS: n=64 is a smoke, and ±9.5 mm spread means
+4.7 mm is DIRECTIONAL, not precise; the tilt row is not a regression (the anchor predates
the tilt_deg column, so its 0.0° means UNMEASURED). v3.4 is the candidate recipe for the
next full mushroom collection; only the robot can confirm the gentleness claim.


**2026-08-27 — alzey probed at last: NOT genuinely size-adaptive (r=0.229) — its real-world
"gentleness adaptive to object size" is a LEVEL+RATE effect.** Prediction registered before
the run ("expect alzey also fairly flat, r<0.4; if r>0.6 my whole reading needs revision")
— confirmed. At-grasp corr 0.229 (succ-only 0.158), small-half 31.3 vs big-half 32.7 mm
(1.4 mm of "adaptation" against a 6.7 mm true spread). Comparison: afucm −0.04 (33.5/33.2),
lulkx 0.138 (30.2/30.1). So what distinguishes gentle-alzey from crushing-lulkx on the ROBOT
is that alzey commands ~2 mm WIDER and its data closed 39% SLOWER (30-step ramp) — level and
rate, not adaptation. This corroborates the variance analysis (mean is fine, spread kills)
and is exactly what the v3.4 smoke (n-grasp 30 + 3 mm squeeze + yaw 55) tests.
CONSEQUENCE: no trained policy in this campaign has ever exceeded r=0.30 on width (eqrth,
and that cost 0.13 success), so the frozen head's **0.888 at closure** is a step change
rather than an increment — provided the end-to-end eval shows success holds.


**2026-08-27 — WIDTH-TRAJECTORY HEAD implemented and both configs launched (user design:
6-DoF diffusion + standalone per-step width regression head).**
RATIONALE: nine probes showed the SAME encoder features give r~0.82 through a regression
head and r~0.1 through the diffusion path. Pose is genuinely multimodal (diffusion suits
it); width GIVEN the object is unimodal regression, and diffusion's mean-seeking collapses
it to a constant — which is exactly what over-squeezes (mean width is fine at −1.6 mm, but
one width for a 12-46 mm range leaves 30% of grasps >5 mm too tight). A clamp/substitution
at deploy was rejected as hacky by the user; this is the principled version.
IMPLEMENTATION (3dbc4e3, all flag-gated — existing runs bit-identical):
· `network.width_traj_head`: cond_encoded → horizon_steps widths. cond_encoded is
  [pointnet_feat ⊕ proprio] and proprio carries the CURRENT gripper width, so the head
  CONTINUES the closing ramp instead of jumping to the endpoint — this is what makes a
  per-step head well-posed (user's objection to the scalar version).
· `WidthHeadDiffusionModel`: width dim removed from the ε-loss via the existing per-dim
  weights (NO shape change, so every config/checkpoint still loads — chosen over literally
  emitting 6 dims to minimise bug surface); head target is `x_start[...,-1]`, i.e. the
  ground-truth chunk's own width column — no new label, no dataset rebuild.
· `eval_agent`: `GM_WIDTH_HEAD=1` splices the head over the sampled width dim. NECESSARY
  because production samples via `DiffusionEval`, NOT the training class — the model-class
  forward() override alone would never run at eval. Assignment ⇒ idempotent, so it cannot
  compound the way the additive gripper-offset bug did.
· GPU smoke test (`.agent_tmp/smoke_width_head.py`) passes 5/5: head shape, finite loss with
  the width component logged, splice idempotence, width weight exactly 0 with pose weights
  renormalised, baseline training path unchanged.
RUNS: **Config 1 RESULT (job 1726007): the frozen encoder DOES carry object size.** Fitting only
the head on lulkx@600 (encoder+denoiser frozen; 0.03% of params trainable) gives, AT THE
CLOSURE FRAME, **corr 0.888 / MAE 2.50 mm** — slightly BETTER than rturn's width-SUPERVISED
aux head (0.82 / 2.65 mm), on an encoder that never saw a width loss. So the head can be
RETROFITTED to our best policies with no retrain. CAVEAT worth keeping: the raw val corr is
0.999 / MAE 0.78 mm, but that is dominated by TRIVIAL steps (most of an episode is 'hold
~80 mm open', where predicting the current value scores near-perfectly); only the closure
number is decision-relevant. Checkpoint `state_600_whead.pt`; canonical eval + width probe
with GM_WIDTH_HEAD=1 running (jobs 1726079/80). **Config 2** = co-trained, 3 seeds (whead_s42/27/43,
jobs 1725987-89) on the shift9 dataset. NOTE Config 2 has the head but NOT paired-reg (the
class chain doesn't stack them today), so its baseline is **njhbz 0.805**, not avfnp 0.830;
merging both mechanisms is a small refactor if the head proves out.
QUOTA (user): killed all ckpt-500/600 evals of older families and dropped tofu seed 43
(tofu now 2 seeds: 42 + 27).


**2026-08-26 — Banana synthesis UNBLOCKED. Root cause was a MIS-PREPPED MESH, not the grasp
search; the general fix is ESCALATING CMA BUDGET on failure (`--grasp-escalate`, default 2).**
Two independent bugs, found in this order:

1. **The installed `banana.obj` was a 2.8x-oversized wedge, not a banana.** The raw scan aspect
   is `[0.66, 1.0, 0.225]` but the installed mesh measured **9.5 x 9.2 x 3.24 cm, 73 cm3**
   (aspect `[1.0, 0.97, 0.34]`). Cause: the stem cut removed ~35 % of the banana's LENGTH, so
   the long axis was no longer the longest bbox extent — `--target-extent 0.095` then landed on
   the **curve span** of the crescent and scaled the whole object up ~1.5x. **Lesson: never scale
   a curved/asymmetric object with `--target-extent` (longest-bbox) after a cut that shortens the
   long axis — use `--target-axis-extent <axis> <m>`, which names the axis explicitly.**
   Re-prepped from the `IMG20260824145616_seed0` scan: **9.5 x 7.07 x 1.86 cm, 26 cm3**,
   watertight, euler 2. Local graspable width **17.3 mm** (was 25.9). Registry size/`default_pos`
   and the task MPM settings updated with it (thin object -> grid density 250 -> **300**,
   substeps 240 -> **290**, so ~5.6 cells span the 1.86 cm thickness instead of 4.6).
   `prep_object_mesh.py` gained `--force-remesh` for scans that are watertight but the WRONG
   TOPOLOGY (this seed0 scan is genus 1 — the hooked stem tip closes into a handle). In the end
   the stem cut alone fixed the topology, and forcing the remesh made things worse (decimation
   broke watertightness), so the flag exists but was not needed here.

2. **The banana's feasible set is genuinely ~6x smaller than a mushroom's — it needs more
   SEARCH, not different search.** Instrumenting the status of every CMA candidate:
   banana **0.6 %** holdable vs mushroom **3.7 %**, with an otherwise near-identical rejection
   mix (no_contact ~40 %, table ~31-36 %, penetrate ~21-23 %). So it is not a distinct failure
   mode. Confirmed directly: at 4x budget (16 starts / 4000 fevals) standalone feasibility went
   **2/8 -> 8/8**. Implemented as `--grasp-escalate N` (default 2) in `collect_demos_synth_v3.py`:
   on synthesis failure, retry with n_starts and maxfevals both DOUBLED, up to N times.
   **Escalating on demand beats a per-object budget constant or a shape heuristic** — it costs
   nothing when the base budget already succeeds (mushroom / strawberry / raspberry synthesis is
   unchanged, and a run with no failures is bit-identical to before), and it needs no shape
   descriptor. A **bbox-derived** descriptor would be actively wrong here: the banana's bbox reads
   as a large easy object while its graspable local width is only 17 mm.

**Measured, collector under full DR (8 envs).** Two DIFFERENT metrics, and they disagree —
read them carefully:

*Synthesis feasibility* (does CMA return a grasp at all), first batch: old mesh 1-2/16; new mesh
alone 2/8; new mesh + escalation **4/8**, and the grasps are now proper local-width grasps
(w 30-35 mm, align 0.77-0.94, stress 15-21 kPa) instead of crescent spans.

*Demonstrator success* (does the episode actually lift and hold — the metric that matters for
the dataset), from each run's `stats.yaml`:

| run | config | success | attempts |
|---|---|---|---|
| 26-08-26-zbj | OLD mesh + medial seeding | 0.400 | 20 |
| 26-08-26-qqw | new mesh, no escalation | 0.381 | 21 |
| 26-08-26-fsl | new mesh, fixed 4x budget | 0.381 | 21 |
| **26-08-26-zuo** | **new mesh + escalation, area 20 mm2** | **0.421** | 19 |
| 26-08-26-hli | new mesh + escalation, area 10 mm2 | 0.211 | 38 |

**Escalation raises SYNTHESIS feasibility a lot but end-to-end demonstrator success barely at
all (0.381 -> 0.421, and the old mesh scored 0.400 — within noise).** The extra grasps it finds
mostly fail to EXECUTE. The bottleneck has therefore MOVED, not closed: from "CMA cannot find a
grasp" to "the grasp it finds does not hold in MPM". Do not read the 2/8 -> 4/8 feasibility jump
as a fix. Banana at ~0.42 remains far below mushroom/strawberry (~0.94-1.0) and is **not yet good
enough to collect a training set from** — a demonstrator this weak biases the dataset toward
whatever it happens to manage (see the near-perfect-demonstrator note in CLAUDE.md).

The area floor IS settled, on the end-to-end metric rather than on a quality argument: dropping
20 -> 10 mm2 raised first-batch feasibility to 6/8 but HALVED demonstrator success (0.421 ->
0.211), because the grasps the lower floor admits are the crescent spans (w 75-79 mm, align
0.51-0.69, one at 24.6 kPa against the banana's 25 kPa yield) that synthesize fine and then drop
the object. **Keep the area floor at 20 mm2.**

**FOLLOW-UP the same day — the banana's 0.42 was an ARTEFACT. Its real demonstrator success on
legitimate grasps is ~0.** User inspection of the `26-08-26-zuo` videos found (a) some clips have
no `_grasp.png`, (b) the top-down grasp "lifts the banana but completely crushes it", and (c) the
banana spawns partly BURIED for near-straight orientations. All three checked out, and together
they invalidate the earlier number.

1. **Missing `_grasp.png` == synthesis failed == FALLBACK grasp.** `finger_viz.render_grasp_pose`
   does `sig = grasp_stress_voigt(...); if sig is None: return False` — a SILENT no-write, no
   exception, so nothing appears in the log (zero "grasp viz failed" messages). A fallback grasp
   has no FEM contact, so `sig` is None. The missing PNG is a reliable fallback marker.
2. **The fallback grasp CRUSHES, and its episodes were being SAVED as successes.** The fallback is
   a fixed `w=0.045` top-down grasp regardless of object. On the banana (70 mm across the crescent)
   closing to 45 mm compresses it ~25 mm: it crushes AND lifts, so it passed the success test and
   entered the dataset. The code comment asserting such an episode "may not lift -> simply won't be
   saved" is FALSE for any object wider than 45 mm. **5 of the 8 saved `zuo` episodes were crushing
   fallbacks.** Fixed: fallback episodes are now DROPPED (`--keep-synth-failures` to restore the old
   behaviour), counted in `stats.yaml` as `episodes_fallback_dropped`.
3. **Soft-body spawn DR buried the banana** (`genesis_worker.py`). Rotation DR rotates MPM particles
   about their CENTROID and then shifts only in xy — no z re-seat. For a compact object that is
   harmless, but the banana's half-length is 4.75 cm, so a 45 deg pitch swings a tip ~3.4 cm below a
   centroid sitting only ~1.0 cm above the table. **Measured in `zuo`: object-centre z at t=0 was
   0.0071-0.0098 m when the flat half-thickness alone is 0.0093 m** — and a properly-resting ROTATED
   elongated object should sit HIGHER than that, not lower. Fixed by re-seating each env so its
   lowest particle returns to its pre-rotation resting height, clamped to RAISE-ONLY so correctly
   resting objects (and every previously collected object) are untouched.

**The consequence.** Re-running the exact `zuo` config with the fallback drop: across 3 batches
(24 spawns) there were 5 successful syntheses, **6 fallback-crush episodes dropped, and 0 genuine
grasp episodes saved**. Batch 1 skipped envs 3 and 6 — precisely the envs `zuo` had saved as
`ep0002_env3` / `ep0003_env6`. **So the banana's ~0.42 demonstrator success was almost entirely
crushing fallbacks; on legitimate synthesized grasps it is near zero.** The banana is much further
from usable than the earlier entry implied, and no banana dataset should be collected until a
synthesized grasp can actually lift it.

**Two levers tested and REJECTED for the banana:**
- *More CMA budget.* Escalating to x16 makes a fully-failing env ~31x base cost (1145+2290+4580+
  9160+18320 fevals) — one batch ran >20 min inside a single env and had to be killed. And a fixed
  4x budget gave the SAME demonstrator success as none (0.381 both). Budget raises SYNTHESIS
  feasibility, not lift success. Keep `--grasp-escalate 2`.
- *Loosening the camera-azimuth bound 60 -> 75 deg.* Gave 4/8 synthesis, IDENTICAL to 60, so the
  occlusion bound was never the binding constraint. It also admitted a 33.0 kPa grasp against the
  banana's 25 kPa yield. Not adopted.

**SAME DAY, third pass — the plans themselves were bad: CMA was grasping ALONG the banana's LONG
AXIS. New structural bound `--grasp-width-max-mm` fixes it.** User inspection of `26-08-26-tfi`:
"it grasps along the long axis of the banana". Confirmed from `dr_params.csv` — against a ~17 mm
local cross-section, the 5 synthesized grasps had widths **42.3 / 55.9 / 76.6 / 76.7 / 79.0 mm**
(median 76.6), i.e. 4 of 5 pressed the two crescent TIPS together instead of closing across the
body. None lifted.

**Why CMA preferred them, and why it was OUR bug.** An end-to-end grasp presents MORE pad contact
(`min_pad` 28-34 mm2) than the correct across-the-body grasp (23.8 mm2), and TWO terms we set
reward exactly that: the `area_min` floor (20 mm2, calibrated on a **33 mm mushroom cap**) and
`w_press` (grip / contact area). A proper grasp on a 17 mm band inherently yields small pad
contact, so the floor sat right at the edge of rejecting the CORRECT grasp while the long-axis
grasp cleared it comfortably. **This is the same size-calibration bug already fixed in
`filter_pinch_episodes.py` (thresholds must scale with object size), left unfixed in synthesis.**

**The mechanism that let it happen:** the width search bound was the hardcoded gripper max
`0.079`, and `_seed_width` measures the object's **global** cross-section along the closing axis —
95 mm for the banana's long axis, which CLIPS to 0.079, so every long-axis seed starts pinned at
max width and CMA stays there. (`_seed_width`'s docstring claims per-axis seeding avoids "biasing
every start toward the long axis"; on the banana the clip hides the difference and it does not.)

**Fix — `width_max`, a STRUCTURAL bound in the style of `roll_max` / `yaw_max_deg`** (plumbed as
`--grasp-width-max-mm`, scaled by the scene-DR scale like `--grasp-area-min-mm2`; default None =
0.079 = every existing object unchanged). A second leak was found and fixed with it: the width
REFINE scan used `min(1.6 * xb[6], 0.079)`, hardcoding the gripper max, so a 40 mm cap still
returned a 45.1 mm grasp until it was changed to respect `_w_hi`.

**Measured, 6 DR-matched poses (standalone, escalation ladder):**

| config | feasible | widths | stress kPa med/max | align med |
|---|---|---|---|---|
| baseline (<=79 mm, area 20) | 6/6 | 22-**79** mm | 10.8 / 15.5 | 0.69 |
| <=40 mm, area 20 | 6/6 | 34-40 mm | 12.1 / **27.5** (over yield) | 0.84 |
| **<=40 mm, area 10** | 6/6 | 25-40 mm | 13.3 / **16.1** | **0.87** |

**This REVERSES the earlier "keep the area floor at 20 mm2" conclusion, for a principled reason:**
the 20 mm2 floor was the only thing discouraging tiny-contact grasps while no width bound existed,
a job it was badly suited to. With the width capped structurally, the crescent grasps area-10
previously admitted are impossible by construction, so the floor can drop to what is physically
right for a 17 mm-wide object. Use **`--grasp-width-max-mm 40 --grasp-area-min-mm2 10`** for the
banana.

**In the collector under full DR (run `26-08-26-bvy`, 5 batches / 40 spawns):** synthesis
**26 ok / 6 failed (~81 %, was ~30 %)**, widths **23.3-44.0 mm (median 40.5), ZERO above 45 mm**,
align median 0.78. So the plan-quality problem the user identified is FIXED.

**But demonstrator success is still ~4/26 (~15 %) on genuine grasps.** The width cap corrected
WHAT is planned; the banana still usually fails to actually lift. The bottleneck remains the
grasp->lift transition, now on properly-oriented across-the-body grasps. Still not ready to
collect a banana dataset.

**Does it generalize? YES in kind — and the width cap IS auto-derivable** (`width_max="auto"`,
`--grasp-width-max-mm auto`). The descriptor is the median **LOCAL cross-section perpendicular to
the long axis** — the width an across-the-body grasp actually has to close on — times 2.3. This is
the descriptor the earlier bbox idea should have been: **the bbox ranks the banana LARGEST/easiest
(95 mm longest extent) when its graspable width is ~18 mm.**

| object | elongation | local x-sec (FEM mesh) | auto cap | UNCAPPED grasp width | effect |
|---|---|---|---|---|---|
| mushroom | 1.09 | 31.0 mm | 71.4 mm | 34.5 mm | inert, 2.1x headroom |
| strawberry | 1.23 | 33.7 mm | 77.4 mm | 41.7 mm | inert, 1.9x headroom |
| raspberry | 1.06 | 14.3 mm | 32.9 mm | 18.0 mm | inert, 1.8x headroom |
| **banana** | **5.12** | 20.9 mm | **48.1 mm** | **76.6 mm (median)** | **BINDS** — cuts 4 of the 5 tfi grasps |

So the cap is **inert by construction on compact objects** (safe to leave on) and binds only on
elongated ones. TWO caveats, both measured and both in the `local_cross_section` docstring:
1. The 2.3 coefficient is calibrated on ONE elongated object.
2. It runs on the **FEM** object, whose mesh is voxel-remeshed (`prepare_mesh`, voxel_div=14) and
   is therefore THICKER than the source for a thin body — the banana reads **20.9 mm vs 17.9 mm**
   on the raw mesh (~17 % inflation), so `auto` yields a **48.1 mm** cap rather than the 41.2 mm
   the raw mesh implies. 48 mm still binds hard, but it is LOOSER than the **40 mm that was
   hand-tuned and end-to-end verified**. Prefer an explicit `--grasp-width-max-mm` where a value
   has been validated; use `auto` for a NEW object.

**ALL banana runs on ONE consistent metric — the width cap did NOT raise demo yield.** Earlier
headline `success_rate` values are NOT comparable across runs because they counted crushing
fallbacks as successes. Recomputed from each run's `dr_params.csv` as successes among SYNTHESIZED
grasps only (`stress_Pa > 0`):

| run | config | spawns | synth | genuine ok | rate among synth | **genuine/spawn** | headline |
|---|---|---|---|---|---|---|---|
| zbj | old mesh + medial seeding | 24 | 3 | 0 | 0 % | 0 % | 0.400 |
| qqw | new mesh, no escalation | 24 | 4 | 2 | 50 % | 8.3 % | 0.381 |
| fsl | new mesh, fixed 4x budget | 24 | 8 | 3 | 37.5 % | 12.5 % | 0.381 |
| zuo | new mesh + escalation, area20 | 24 | 10 | 3 | 30 % | 12.5 % | 0.421 |
| hli | new mesh + escalation, area10 | 40 | 23 | 4 | 17.4 % | 10 % | 0.210 |
| **bvy** | **+ width cap 40, area10** | 32 | **26** | 4 | 15.4 % | **12.5 %** | (killed) |

**The width cap took synthesis 30 % -> 81 % but genuine successes per spawn stayed at 12.5 %,
identical to fsl and zuo.** At 3-4 successes per run these differences are noise. What the cap
bought is PLAN QUALITY (widths 23-44 mm vs a 76.6 mm median, no long-axis grasps, align 0.69 ->
0.78) — worth keeping, since it separates demos that grasp the banana correctly from demos that
squeeze its ends — but NOT more usable demos.

The fallback column explains the old headline numbers: **zuo's 0.421 was 3 genuine + 6 CRUSHING
episodes**, i.e. two thirds of its "successes" were the fallback grasp. bvy has 1 fallback success
because synthesis now rarely fails, so nothing masks the true rate any more.

**CONCLUSION: the banana sits at ~10-12 % genuine demonstrator yield regardless of the fix.**
Every lever tried this session — mesh correction, budget escalation, fixed 4x budget, azimuth
60->75, width cap — moved synthesis feasibility or plan quality, and NONE moved lift success.
That is consistent evidence the blocker is the **grasp->lift transition**, not the planner.
Do not spend more effort on synthesis for this object until that is understood.





**Negative result — medial-axis seeding does NOT help (`--grasp-medial-seeds`, default OFF).**
Seeding CMA from deep-interior medial points (each closing perpendicular to the local tangent,
width from the LOCAL cross-section instead of the global extent) was the hypothesis for elongated
objects. It fails on both counts: on the banana it gave **1 feasible grasp in 16** spawns (vs 0-2
for COM seeding) and cannot solve the corrected mesh at all, and it **regresses convex objects
with a slender sub-feature** — on a mushroom the medial axis runs down the STEM, producing exactly
the stem grasps the area floor exists to reject (align 0.93 -> 0.66, min_pad 13.8 -> 5.7 mm2,
stress 6.3 -> 10.5 kPa). Kept off by default and documented in the flag's help text so it is not
re-derived.

**Synthesis health check, all four objects** (6 DR-matched poses each: yaw full, pitch/roll +-45,
flip 0.25; collector-matched constraints — area floor 20 mm2, azimuth 60 deg, w_press 0.05,
w_peak 0.3 — with the escalation ladder):

| object | feasible | stress kPa (med / max) | align (med) |
|---|---|---|---|
| mushroom | 6/6 | 13.6 / 16.5 | 0.92 |
| strawberry | 6/6 | 4.9 / 5.6 | 0.95 |
| banana | see above (~4/8 under the collector) | 15-21 | 0.77-0.94 |

**Still open:** the banana's best grasps sit at 15-21 kPa against a **25 kPa** yield, so it is
close to bruising even when synthesis succeeds — the material yield is a literature guess, not
measured, and may need calibration before banana demos are trusted for gentleness. Raspberry
likewise runs near its yield. **Auto-tuning the area floor** (the same way budget is now
escalated) is a sensible follow-up but was not attempted.

**2026-08-27 — ⭐ THE CRUSHING IS A VARIANCE PROBLEM, NOT A LEVEL PROBLEM — which explains
why the +3 mm offset failed on the rig, and revives width adaptation with a precise
justification.** Offline check (`.agent_tmp/substitution_check.py`, rturn head on the s08
val split): per-episode over-squeeze = commanded − the demonstrator's own grasp width.

| regime | mean | worst | mean abs | **frac >5 mm too tight** | size-variance captured |
|---|---|---|---|---|---|
| CONSTANT (policy today, ~30.2 mm) | −1.6 mm | −15.8 mm | 5.7 mm | **30%** | 0% |
| SUBSTITUTE (head level) | +0.2 mm | −12.9 mm | 2.7 mm | **4%** | 67% |

True widths 31.8 ± 6.8 mm (range 11.8-46.0); head r=0.819, MAE 2.65 mm.
READS: (1) the policy's MEAN width is already right (−1.6 mm) — the pathology is SPREAD:
one width for a 12-46 mm range, so 30% of grasps are >5 mm too tight, worst −15.8 mm.
(2) A constant offset shifts the whole distribution and therefore CANNOT fix a spread — it
loosens the already-fine small objects while still crushing the large ones. That is exactly
the rig outcome, and it RETRACTS my earlier inference "if ±3 mm is inert, adaptivity is
probably inert too": the offset was inert BECAUSE it is a level shift. (3) Adaptivity is the
only mechanism that addresses spread, and substitution would cut the badly-crushed fraction
30% → 4% with a head that already generalizes OOD (r 0.80 on smallband).
NEXT (cheapest first): no-retraining SUBSTITUTION at deploy — the policy's own width command
supplies the TIMING (no stage predictor needed), the head supplies the LEVEL; the substituted
amount MUST be invisible to the policy's proprioception (the e4235c2 feedback lesson).
User's post-training idea (refine the width dim only) remains the retraining path if
substitution's head error proves too coarse.


**2026-08-27 — `--gripper-offset-m` TRIED ON THE RIG: does not rescue the v3.3 over-squeeze;
retrain is the path. Knob kept (fixed) for later use.** The no-retrain mitigation was tested on
the real robot and **did not help much (user)** — the decision is to wait for the retrain rather
than tune the offset further. Two things are worth keeping from the attempt:
1. **A REAL BUG in the knob was found and fixed** (e4235c2). First attempt: with
   `--gripper-offset-m 0.003` the gripper **walked open after the lift and dropped the object**
   (user). Cause: the offset biased only the COMMAND, so the measured width returned ~3 mm wider
   than anything in training — and the policy CONDITIONS on `gripper_width`. It read the wider
   value as "not closed yet / released", commanded wider, and the offset re-applied its bias on
   top: positive feedback. The offset is now **invisible to the policy** (the same amount is
   subtracted from the width fed to `policy.reset/push`), so proprioception stays on the training
   manifold while the robot holds `offset` wider; recording keeps the RAW obs. Even so, the
   corrected version did not fix the crushing.
2. **What that tells us about the diagnosis.** A constant open-up offset is exactly the right
   SHAPE of correction for a constant-width policy, so its failure is evidence that commanded
   width is not the whole story — the closing RATE (1.78 vs 1.28 mm/step) and/or the sim-vs-real
   material stiffness (E = 0.3 MPa, soft end of 0.3-3.0) are carrying more of the effect than the
   2.8 mm width gap alone. Worth remembering before attributing the retrain's outcome solely to
   the width distribution.
STATUS: knob stays in the tree at default 0 (inert). Worth re-trying **after** a retrain that
fixes the rate/extra-close, as a fine-tuning trim rather than a rescue — and it is also the
cheapest probe available if over-squeeze reappears on a future family. CAVEAT for whoever picks
it up: hiding width from the policy is only sound while the policy's width behaviour is
near-constant; on a genuinely size-adaptive policy it would corrupt the adaptation.

**Addendum (cluster agent) — the general rule I should have applied:**
**GENERAL RULE, worth applying to any future deploy knob:** a persistent BIAS on a commanded
channel that the policy also OBSERVES must be compensated in the observation, or it closes a
feedback loop. (Rate-limit style filters — smooth_alpha, max_pos_step_m — are exempt: they
add lag, not bias, and the policy is trained against controller lag.) Before shipping such a
knob, ask "does the policy observe this channel?" — I did not.


**2026-08-26 — shrimps: 8 images, one mesh PER IMAGE; euler gate 5/8 (10/24 candidates).**
`obj_images/shrimps/` holds 8 DIFFERENT objects, not views of one, so a new mode:
`scripts/mesh_from_photos/select_per_image.py` picks the best seed for each image and
writes `obj_meshes/shrimps/selected/<image>.{obj,report.json,mp4,gif}`. It ALWAYS selects
(ranked: passes gate, then |euler-2|, then winding-consistent, then watertight), because a
caller asking for "all of them" cannot have gaps; the verdict travels in `_selection.json`.

Per image: shrimp3 3/3, shrimp8 3/3, shrimp2 2/3, shrimp4 1/3, shrimp7 1/3, shrimp6 0/3,
**shrimp1 0/3 (best euler -32)**, **shrimp5 0/3 (best euler -64, up to 1969 floaters)**.

**Generalises the thin-structure finding:** the driver is not "appendages" specifically but
ANY geometry where two surfaces approach within a voxel of the ~512^3 grid —
(a) thin blades: shrimp1's splayed tail fan, (b) NEAR-CLOSED CURLS: shrimp5's tail almost
meeting its body, which fuses into a torus, (c) fine crevices in torn flesh. shrimp3/8,
which are smooth simple curls with well-separated ends, pass 3/3.

**Gate vs. visual quality diverge — do not read FAIL as unusable.** shrimp6 (euler 0,
genus 1) and shrimp1 (genus 17) look good; shrimp1's handles are confined to the tail fan
and its body is clean. shrimp5 is the only one that is genuinely bad (lumpy, torn, holed).
Since genus does not block tetgen, treat euler as triage, not rejection.

**Watermarked stock input is a real confound**, not just a licensing note: shrimp1's Alamy
watermark is tiled ACROSS the shrimp body and shrimp5/7/8 carry agency marks. Flagged to
the user for any paper/dataset use.

**prep_images.py gained** (a) `.webp/.bmp/.tif` support -- shrimp6.webp would have been
silently skipped; (b) `keep_largest_alpha()`, dropping non-largest foreground blobs BEFORE
the bbox crop, because stock agency banner bars are separate blobs that would otherwise
stretch the crop box and shrink the subject to a fraction of the frame; (c) scipy
`ndimage.label` replacing a pure-python flood fill that was far too slow at full res.

**Infra:** first submission died with `cudaErrorDevicesUnavailable` on n37... actually on
**n206** (the GPU was busy/unavailable); resubmitting with `--exclude=n206` succeeded on
n37. Node-specific, not a code fault. TODO: cheap `torch.cuda` probe before the 2 GB
pipeline load so a bad node fails in seconds.

**2026-08-26 — REAL RESULT (user): every v33b/v3.3 checkpoint OVER-SQUEEZES and crushes;
alzey remains the gentle, size-appropriate one. Diagnosed from the datasets.**
1. **PRIMARY — the commanded grip is tighter.** Commanded width at closure: v33b_shift9
   **31.3 ± 6.9 mm (p10 21.2)** vs alzey/afucm **34.1 ± 6.5 mm (p10 25.9)** — 2.8 mm tighter
   on average, 4.7 mm at the low end. CAUSE: our own small-object fix. Extending scale DR to
   [0.8, 1.5] filled the data with small mushrooms, which pulled the learned CONSTANT width
   down; width adaptation never transferred (every probe flat ~30 mm), so that tighter
   constant is applied to normal-size REAL mushrooms. This is the real-world bill for the
   sim small-object gain (avfnp 0.90/0.90) — a flat-width policy cannot have both.
2. **SECONDARY — closing is 39% faster** (user's hypothesis, confirmed): v3.3 closes at
   **1.78 mm/step (p90 2.19)** vs alzey **1.28 (p90 1.52)**, i.e. `--n-grasp 20` vs `30`. On
   a position-controlled real gripper that is a bigger commanded-minus-achieved gap per tick
   = harder driving at contact. Compounds (1).
3. **AMPLIFIER — sim material may be too soft.** Our mushroom is E=0.3 MPa, the SOFT end of
   the literature 0.3-3.0 MPa. A stiffer real mushroom yields far more stress for the same
   over-closure — which explains the INVERSION: v33b looks gentler in sim (28.1 vs alzey's
   33.9 kPa sustained) yet crushes in real. Sim stress is not a faithful real-gentleness
   proxy at this material setting.
ACTIONS: (a) shipped `--gripper-offset-m` (987e21d, default 0) — deploy-time open-up offset;
`--gripper-offset-m 0.003` on lulkx/avfnp tests the diagnosis on the rig with NO retraining;
(b) next collection: `--n-grasp 30`, `--grasp-extra-close 0.0025-0.003`, and either narrow
the scale range or accept the grip-margin cost of small-object coverage; (c) consider
re-measuring/raising the mushroom E.

**Same report — YAW/OCCLUSION: the bound HELD in data; ±60° is simply not tight enough.**
Yaw at closure (30° bins) is confined to **±90° with mass in ±60°** for v3.3 — and TIGHTER
than alzey's data, which reaches ±120°. So `--cam-azimuth-max-deg 60` did its job. Why the
robot still occludes: (i) ±60° already allows the gripper body to block the camera; (ii) yaw
is TASK-IRRELEVANT for a near-symmetric cap, so the policy's target is flat across the whole
permitted band and diffusion samples anywhere in it, occasionally at the occluding edge;
(iii) real clouds are slightly OOD, pushing the tail past anything sim showed. FIX: tighten
at COLLECTION (`--cam-azimuth-max-deg 30-40`) rather than expect the policy to learn a
preference it was never given. NOTE this failure is invisible to our sim eval (occlusion is
not scored) — same blind-spot class as the v33 real-slice bug.


**2026-08-26 — fallback-contamination cross-check (prompted by the local agent's banana
finding): ALL our collections are clean.** Their diagnosis — a blocked planner emits
DEFAULT top-down fallback grasps that keep "success rate" high while the audit columns
(stress/grip/align/min_pad) are all zero — is a check every collection should pass. Ran it:
tofu v10 0% · tofu v11 0% · mushroom argmax (cze) 0% · **mushroom anchor 26-08-25-clq
(njhbz's data) 2/696 = 0.3%** (success 0.938 → 0.941 excluding them, i.e. immaterial) ·
tofu 650 in progress 0% (and 160/160 successful so far). So no result to date is inflated
by fallbacks, and the tofu-policy failure remains purely a data-volume story. ADOPTED as a
standing post-collection check alongside the pinch rate.


**2026-08-26 — paired-reg file was NOT bias-corrected (user caught it); corrected variant
built + 3 seeds launched.** `paired_cube3_clouds.npz` dates from 2026-08-23, three days
before `shift_demo_clouds.py` existed, and its real half is uncorrected (real x-centroid
0.4218 vs sim 0.4393). So the shift9_preg family (mqlxj/avfnp/lulkx) trains on a MIXED
signal: co-train real slice corrected (+9 mm) while the paired regularizer aligns
UNCORRECTED real features to sim. Two readings, both plausible: (a) the regularizer aligns
the wrong real distribution and is therefore under-performing; (b) aligning across a ~9 mm
offset teaches encoder shift-INVARIANCE and is part of why this family is the campaign's
best. Test: `paired_cube3_clouds_shift9.npz` (real half +9 mm x; verified no zero-padding
so a plain shift is safe) → **preg9_s{42,27,43}** (jobs 1696934-36), identical to the
shift9_preg family in every other respect, so the ONLY delta is paired-file correction —
a clean 3v3 seed-matched A/B against mqlxj/avfnp/lulkx (0.770/0.830/0.820).


**2026-08-26 — ⭐ THE SMALL-OBJECT GRADIENT IS GONE (avfnp 0.90 small / 0.90 big) — and it
was solved WITHOUT width adaptation.** Width probes on the shift9_preg family (same
protocol/geometries as all nine earlier probes):

| policy | at-grasp corr | small-half ever | big-half ever | gap |
|---|---|---|---|---|
| afucm (foundation) | −0.04 | 0.57 | 0.77 | 0.20 |
| njhbz (shift9, plain) | 0.083 | 0.80 | 0.97 | 0.17 |
| lulkx (shift9+paired) | 0.138 | 0.83 | 0.93 | 0.10 |
| **avfnp (shift9+paired)** | 0.295 | **0.90** | **0.90** | **0.00** |

Small-object ever-success went 0.57 → 0.90 on the identical probe while commanded width
stayed essentially flat (0.295 at best, vs the data's 0.85). CONCLUSION for items 17/18/
18b/iter-4: the small-object failure mode that motivated the entire width-adaptation
thread is fixed by DATA QUALITY + DOMAIN ALIGNMENT (v3.3 synthesis · 4-mesh pool · scale
[0.8,1.5] · bias-corrected clouds · paired-feature consistency), NOT by teaching the
policy to size its grip. Width adaptation remains an open scientific question (the head
predicts at r≈0.8; no mechanism transfers that to the action) but is DEMOTED from the
critical path — it is not what small objects needed. NOTE avfnp is the sim+probe star yet
FAILS the real-obs hybrid row → deploy **lulkx/state_600** (0.820, gap 0.10, clean PASS).


**2026-08-26 — eval sim-server HANGS: 7 occurrences, pattern identified, watchdog added
(832dbe8).** Signature (prmaw@600, 5 shift9_preg evals, mqlxj@600 twice): the sim server's
FPS **decays monotonically** (17→12, 12.6→11.3) and then the log goes SILENT; the job then
sits idle until the 8 h wall clock with 0 videos written (a healthy eval writes 200, one
per episode, and an FPS line every ~40 s). Decay-then-stall points at RESOURCE GROWTH in
the long-lived genesis child rather than a random fault, and it correlates with multi-eval
contention on a node. Root cause NOT fixed (sim server = local agent's module; genesis is
known to leak, which is why GenesisProcess exists at all) — flagged for them with this
evidence. MITIGATION shipped in `dppo_eval.sbatch`: a watchdog kills the job after
`GM_EVAL_WATCHDOG_MIN` (default 25) minutes of sim-server log silence, converting a silent
8 h GPU burn into a fast FAILED that the standing monitors resubmit immediately. Cost of a
false positive is one cheap retry; a healthy eval's gap is ~40 s, so the margin is ~35×.


**2026-08-26 — real-obs gate on the shift9_preg family: lulkx/mqlxj PASS, avfnp (the best
sim score) FAILS on one hybrid row — the probe becomes a DEPLOY-SELECTION tool, not just a
poison detector.** Probed against `real_merged_shift9mm` (matching their training clouds).
- lulkx@600 PASS (descends on all four rows, grip pinned 80 mm) · mqlxj@400 PASS.
- avfnp@400 FAIL, but READ THE ROWS: it fails ONLY the diagnostic hybrid `sim proprio +
  REAL cloud` (cmd z 0.2033→0.2044, climbs); gripper is perfect (80 mm) everywhere and the
  DEPLOYMENT condition `REAL proprio + REAL cloud` descends 4.8 mm over the chunk —
  comparable to lulkx's 4.5 mm. So this is NOT the v33 poisoning signature (that was z
  0.225 + grip 44 mm on BOTH real-cloud rows); it is a milder visual-branch weakness that
  only surfaces when proprio is mismatched.
- **CORRECTION (same day, user question "they differ only by seed — why is one warned?"):**
  confirmed avfnp/lulkx differ ONLY by seed (27 vs 43); same data, model, everything. The
  failing row `sim proprio + REAL cloud` is OUT-OF-DISTRIBUTION BY CONSTRUCTION — no
  training sample or deployment step ever pairs sim proprio with a real cloud; it exists
  only to isolate modality. Two seeds extrapolating differently there is ordinary variance,
  NOT robot evidence. On the DEPLOYMENT row both descend (avfnp −4.8 mm, lulkx −4.5 mm).
  So the earlier "deploy lulkx over avfnp" ranking is RETRACTED: the gate is decisive for
  what it was built for (the poisoned family failed BOTH real-cloud rows catastrophically:
  z 0.225 + grip 44 mm) but must NOT rank healthy policies on an OOD row. Revised: both are
  deployment-healthy; avfnp/400 preferred on merit (0.830, small-object gap 0.00 vs lulkx's
  0.10), lulkx/600 the conservative alternative; deploy both if rig time allows.
  SUGGESTION for the local agent: make the probe's PASS/FAIL depend on the real+real row,
  with the hybrids reported as diagnostics only.
- **RESOLVED (upstream 0cb33f5): the FAIL was SAMPLING NOISE, not OOD extrapolation.**
  Diffusion sampling starts from random noise, so a single draw is a noisy verdict and a
  marginal policy flips PASS/FAIL between identical runs; the probe now averages
  `--n-samples` (default 8) and reports the spread (healthy: ±1-2 mm z, ±0.0 mm grip).
  Under the averaged probe **avfnp PASSES all four rows** and is staged for deploy
  alongside lulkx. My OOD argument was directionally right (don't rank on that row) but
  the actual mechanism was measurement noise — worth remembering: a stochastic policy
  needs a repeated-sample gate, and I should have run the probe more than once before
  reporting a FAIL. Their second finding matters too: the probe must be fed the real
  variant the policy TRAINED on (probing poisoned orkam with CORRECTED clouds made it
  PASS) — our shift9 probes did use shift9mm, so those verdicts stand.
- Deploy pairing for all shift9_preg runs: point_cloud_shift [0.009,0,0] ACTIVE.


**2026-08-26 — THREE FAMILIES REPORT: paired-reg on shift9 data is the campaign's BEST
(3 seeds 0.77/0.83/0.82); residual-width v2 FAILS; tofu-on-smoke-data fails (data volume).**
- **shift9_preg ×3 seeds (mqlxj/avfnp/lulkx, dataset v33b_shift9 + PairedReg w0.5)** —
  best-ckpt success 0.770 / **0.830** / 0.820, ever up to 0.865, sustained 25-29 kPa.
  Mean ≈0.807 with a TIGHT seed spread (±0.03) vs afucm 0.685 / njhbz 0.805 (1 seed).
  **Qualitatively new: these curves RISE with epochs (peaks at 400-600)** while every
  other family in the campaign peaks at 100 and decays — the paired consistency loss
  appears to act as a regularizer that keeps training productive. STACKING CONFIRMED:
  bias-corrected clouds (njhbz) + paired-feature alignment (alzey) > either alone; both
  are DOMAIN-ALIGNMENT mechanisms, and they compose.
- **resw2 = residual width v2 (nickq), the units-fixed mechanism: DECISIVE NEGATIVE** —
  0.170 @100 decaying to 0.03, sustained 44 kPa. Mechanistic read: the fix made the
  CLOSURE residual constant but left the APPROACH-phase residual scene-DEPENDENT (open
  command ≈80 mm fixed minus a per-episode anchor), i.e. it moved the hard part rather
  than removing it — and a head error there means the gripper is not fully open during
  approach (collision/push). Residual width actions are ABANDONED; with 18b, aux weights,
  FiLM and window weighting, that closes every mechanism tried for width adaptation.
- **tofu_preg ×3 seeds (zffwn/tygdq/puysh, v11 smoke 36 eps, pure sim): 0.09/0.045/0.07**
  — non-functional at every checkpoint, all seeds, curves still climbing at 600 (classic
  underfit-by-data-volume). 36 episodes vs afucm's 635; the demonstrator is fine (97.6%).
  Verdict: PIPELINE BRING-UP ONLY (tofu task/experiment/convert/train/eval path all work
  end-to-end) — the tofu policy question needs the 650 collection, still user-gated.
- OPERATIONAL: 5 shift9_preg evals were HUNG (sim server silent 3.5-5 h, same signature as
  the prmaw hang) — cancelled and resubmitted with a 25-min silence detector armed. Hang
  rate is now material (6 occurrences); if it recurs, add a watchdog to dppo_eval.sbatch.


**2026-08-26 — argmax+peak mushroom probe collection (user ask): pinch rate 7.8% → 2.0%
at equal success; width diversity survives argmax.** 100 eps, all-video
(`26-08-26-cze`), = the njhbz anchor recipe with EXACTLY two changes: diversity/jitter
OFF (pure argmax) + w_peak 0.3. Results: success 92.9% (anchor 93.8%); PINCH RATE 2/100 =
2.0% vs anchor 51/650 = 7.8% (4×); align 0.916 (p10 0.889), tilt 9.3° (p90 21.7° —
natural range on curved geometry); CMA width 41.9±9.6 mm — the width SPREAD is
scene-DR-driven, not jitter-driven, so argmax preserves it. Per-mesh: mushroom/m3 1.00,
m1 0.906, m2 0.792 (n=24, likely noise — watch). READ: the diversity machinery's main
output on mushrooms was pinches; argmax+peak is a candidate straight upgrade for the next
full mushroom collection (user to confirm from videos). PRECISION NOTE (from the local
agent's multi-object push): pure argmax needs ALL FOUR knobs zeroed (tol, jitter-deg,
jitter-pos, pitch-seed-deg — `_div_on` ORs them); this run left jitter-pos 0.003 +
pitch-seed 25 at defaults — near-argmax (tol=0 makes jitter acceptance ≈ never fire;
pitch seeding only shapes search init), results stand, but use the 4-zeros form going
forward. Cross-validation: their strawberry run (same recipe) → pinch 2.5% @ 93.8% —
argmax+peak now replicated on two objects.


**2026-08-27 — ANCHOR RUN DOCUMENTATION: njhbz (v33b_shift9) — full pipeline + args.**
The campaign's best sim policy (0.805/0.820 @300, sust 28.1; real-obs probe PASS @200/300).
1. COLLECT (job 1653982 → `dataset/demos/single_lift_mushroom_soft/26-08-25-clq`, 650 eps,
   93.8%, 3 NaN-guard discards): `collect_demos_synth_v3.py --experiment
   single_lift_mushroom_soft_abs_action_armfocus_realws_mm4_s08` (task
   single_lift_mushroom_soft · obs superset_soft_armfocus · action abs_pose_abs_gripper 10d
   · dr soft_orientation_realws_mm4_s08 = realws box x[0.29,0.48] y[±0.11] + 4-mesh pool
   [mushroom,1,2,3] + scale [0.8,1.5] + organic shape DR ±25°/±20°/±0.15/[0.95,1.15] + yaw
   180 / pitch-roll 45 / flips 0.25) with `--n-episodes 650 --n-envs 8 --scene-dr-every 1
   --maxfevals 1145 --seed 0 --n-home-to-pre 77 --n-grasp 20 --n-settle 1
   --grasp-extra-close 0.005 --cam-azimuth-max-deg 60 --grasp-jitter-deg 30
   --approach-xy-finish 0.45 0.75 --approach-speed 0.0024 --held-run-max 12
   --held-run-keep 10 --grasp-area-min-mm2 15 --grasp-w-press 0.05 --record-video 20`.
   NOTE: diversity selection ON (tol 0.3 default), w_align 2000 default, w_peak
   effectively 0 (legacy bug), no w_tilt — the tofu v11 strict-synthesis knobs are NOT in
   this anchor.
2. PINCH FILTER: `filter_pinch_episodes 26-08-25-clq` → `-filt`, 51/650 dropped → 599.
3. CONVERT: `convert_demos <clq>-filt/data.pkl --out dataset/dppo/single_lift_mushroom_soft_v33_7d
   --obs-keys ee_pos ee_quat gripper_width --point-cloud --derive-action
   abs_pose_euler_abs_gripper --derive-source-action abs_pose_abs_gripper` (gates: seam 0,
   dwell 0.193).
4. REAL SLICE: `shift_demo_clouds real_merged --shift 0.009 0 0` → `real_merged_shift9mm`;
   `convert_demos ... --derive-source-action delta_pose_delta_gripper_fast_rot
   --derive-lookahead 4` → `single_lift_mushroom_real_7d_cmd_shift9` (gate: t0 grip 79.8mm,
   z-lead 10.7mm=K4 design, grip tracking 0.1mm).
5. MERGE: `merge_npz_datasets soft_v33_7d real_7d_cmd_shift9 --out
   single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9` (654 eps, joint renorm).
6. TRAIN (job 1680274): dppo_pretrain hwo cfg pre_diffusion_pointnet (visual 512, mlp
   [1024]^3), 600 ep, save/100, `action_dim=7 seed=42
   experiment=single_lift_mushroom_soft_abs_action_armfocus_7d_realws`, PLAIN DiffusionModel.
7. DEPLOY PAIRING: point_cloud_shift [0.009,0,0] ACTIVE (mandatory).
8. PROBES (complete): real-obs PASS @200+@300; width probe @300: at-grasp corr 0.083 (flat ~31.6 mm, plain model as expected) BUT small-half ever-success **0.80** (big 0.97) — the best small-object result of any probed policy (afucm 0.57, dgvmu 0.67): small objects need positioning/data quality, not width adaptation — the campaign lesson in one number.
NJHBZ vs ALZEY (beyond the 9mm shift): sim slice hwo-v3 (1 mesh, [1.0,1.5], lerp
approach, close 30, no azimuth/anti-stem/stop-frames, unfiltered, 585) vs v3.3 (4-mesh,
[0.8,1.5], real-speed approach, close 20, azimuth 60, anti-stem, 10 stop frames,
pinch-filtered, 599); model PairedReg(w0.5, cube3 pairs) vs PLAIN — one-mechanism-each
siblings; their union = the shift9_preg_s{42,27,43} family now training. Deploy shift OFF
(alzey) vs ON (njhbz). TOFU v11 synth vs this anchor: argmax (tol/jitter 0) + w_align 3e4
+ w_peak 0.3 + w_tilt 1.5e5 + area 35 + no pinch filter — a much stricter flush synthesis;
do NOT port blindly to mushrooms (diversity is wanted on curved geometry).


**2026-08-27 — paired-reg families launched (user): tofu(v11 smoke, pure sim) ×3 seeds +
njhbz(shift9)+pairing ×3 seeds; verifier dwell gate shown NORMALIZATION-SENSITIVE.**
Six trainings (jobs 1692481-86, seeds 42/27/43 — cfg-default IS 42 so the requested
"original, 27, 42" collides; 43 = campaign alternate, flagged to user): (A) tofu_preg_s* —
njhbz-style recipe on `single_lift_tofu_soft_v11_7d` (36 eps from the v11 smoke; PURE SIM,
real data to be added later) + alzey's paired mechanism (PairedRegDiffusionModel,
paired_cube3_clouds.npz, w=0.5 — pairs are task-agnostic cube3 features); (B)
shift9_preg_s* — the v33b_shift9 dataset + the same pairing = both alignment mechanisms
stacked on the current best recipe, and the shift9 family's first seed-spread. Eval
watchers armed per run. GATE NOTE for the local agent: verify_derived_dataset's dwell[sim]
FAILed tofu at 0.455 while the mushroom v33 sim set passed at 0.193 — but in
normalization-independent units the two datasets are IDENTICAL (vector dwell 0.095 vs
0.094; median |d_xyz| 0.0104 vs 0.0109 derive-units): the metric is computed in each
dataset's own npz normalization and inflates when per-dim ranges differ (pure-sim tofu vs
merged mushroom). Override documented here; the verifier should compute dwell in
derive-space or physical units.


**2026-08-26/27 — v33b curves: shift9 WINS (njhbz 0.725 > afucm 0.685); the deprecated
orkam "recipe win" does NOT reproduce on clean data (lciml 0.595) — claim retracted.**
Standard-eval bests (vs afucm 0.685/24.0): njhbz (v33b+shift9 clouds) 0.725/0.740 @100
sust 38.0 (2/6 evals, curve filling) · gvqwa (aux0.5) 0.670/0.730 @100 · ezdzu (aux1.5)
0.660/0.735 @100 (stable curve) · lciml (plain) 0.595/0.675 @100. READS: (1) the poisoned
orkam's 0.715 was an artifact of the broken merge (its compressed normalization); v3.3
data alone does NOT beat afucm — RETRACTED; (2) the 9 mm cloud-bias correction (§4.1) is
what pushes past afucm (njhbz) — corrected real rows stop fighting sim rows in training;
(3) first dataset where aux width heads ≥ plain (gvqwa/ezdzu > lciml). njhbz@100 = TOP
new real candidate — REQUIRES probe_policy_real_obs + deploy with point_cloud_shift
ACTIVE. Also: qrbtr (window ×8) best 0.470 @200 — verdict mild NEGATIVE; resw2 (residual
v2, fixed units) training. Caveat on all: single seed, @100 peaks, noisy curves.


**2026-08-26 — rztss probe: FLAT (0.095) — root-caused to a UNITS BUG in my residual
transform; v2 (fixed) relaunched as s08_resw2.** The probe (residual add verified ACTIVE
in-log) measured commanded width flat at 27 mm. Forensics: the v1 relabeling subtracted a
DERIVE-space anchor from NPZ-normalized actions — round-trip consistent (training and eval
mirror the same wrong transform, so success was preserved) but the anchor never de-scened
the labels: training residual-at-closure stayed corr **1.000** with episode width (also
revealing: in sim the position-controlled gripper achieves its command exactly, so
commanded-at-closure is an affine of achieved width). rztss therefore learned its usual
scene-tracking-free constant — nothing structural ever happened. FIX (1500c44): two-stage
anchor (phys → derive-space → npz units, both sides); validated offline: residual-at-closure
std 0.004 npz-units (~0.1 mm), corr 0.032 — a pure constant, so commanded = const +
head(scene) now inherits the head's r≈0.8 GENUINELY by construction. Relaunched:
s08_resw2 (job 1689133, watcher with GM_RESIDUAL_WIDTH). Sibling probes: xqmxw (18b+small)
at-grasp 0.209 — feed-forward stays marginal. qrbtr (window ×8) curve filling (0.435/0.470
@100/200 — mild success tax). LESSON: any relabel/anchor transform needs an offline
label-statistics gate (corr(residual, scene) ≈ 0) BEFORE training — now part of the recipe.


**2026-08-26 — tofu v11 COMPLETE: 97.6% (97.9% per-attempt), tilt bounded (mean 3.8°,
p90 12°); recipe ready for the user's 650 gate.** v11 (`26-08-25-yhn`, 48 attempts) ==
v10 statistically (success/align/grip identical); its real additions are the w_tilt
bound + the tilt_deg audit column (v10's tilt was unmeasured — column predates it).
CORRECTION to the interim read: v10's align was already 0.958 — the spawn-z + peak/area
fixes did most of the quality work; v11 polices the residual tilt tail. Stress-number
"oddities" (user): explained — the plot shows contact-MASKED bulk top10 and tracks GRIP
(ep0008 1.98 N/6.4k vs ep0007 14.54 N/16.6k = deep squeeze, not artifact); REAL bug
found: scoring used --grasp-E default 3e5 (mushroom) → tofu Pa/N displayed ~6× inflated
(ranking E-invariant; holdability check inflated). 650-recipe recommendation: v11 flags
+ --grasp-E 50000 --grasp-density 1050 (+ optional deep-squeeze cap); trainings at gate:
afucm-recipe plain + (pending rztss probe) residual-width instead of aux w=2.0.


**2026-08-26 — REAL RESULTS (user): alzey ≈70%+ (item 16 POSITIVE in real) but
over-squeezes; qjzsf weak at workspace edge + crushes more.**
- alzey/state_200 (paired-feature encoder reg): ≥70% real success (not rigorously
  counted) — encoder domain-alignment survives the robot; at least afucm-class.
  OVER-SQUEEZE observed — and the sim gentleness metric PREDICTED it: state_200 is the
  harshest post-100 checkpoint (sustained 33.9 kPa, near the 40 kPa yield) vs 300
  (0.730/27.9) and 400 (0.675/22.9). RECOMMENDED: real-test state_300/400 for the
  squeeze trade — the first real validation of item 11 (rank on sustained, not success
  alone).
- qjzsf (real-only, 55 demos): edge-of-workspace weakness (teleop demos under-cover the
  box boundary; sim collections sample it uniformly) + more crushing (55 demos too few to
  learn the operator's width adaptation). Both are DATA-COVERAGE failures — consistent
  with the campaign-wide data-beats-architecture pattern.
- The over-squeeze is plausibly the real-world manifestation of the constant-width
  strategy (flat ~30 mm command → small mushrooms over-squeezed) — strengthens the
  rztss residual-width bet.


**2026-08-26 — ROADMAP CONSOLIDATION (user request): what is concluded, abandoned, ongoing, new.**
CONCLUDED & ADOPTED: v3.3 synthesis recipe (sim-validated; real trial via v33b after the
poisoning fix) · 4-mushroom mesh pool + scale [0.8,1.5] (items 6+18) · small-size data
coverage (2× small-scale success — the one width-related intervention that works) ·
regular MPM sampler everywhere · 10 stop frames (supersedes hold-tail aug) · anti-stem/
pinch + peak + tilt synthesis terms (tofu-driven, mushroom-applicable) · dataset+policy
gates (verify_derived_dataset, probe_policy_real_obs) as standing pre-train/pre-deploy
practice · gentleness ranking on sustained kPa.
CONCLUDED NEGATIVE (abandoned): gentle demos (item 10) · first-frame memory (item 12) ·
aux contact/pos heads (item 13) · h8/e4 alone (item 15; stop frames make h8 moot) ·
global gripper-dim loss ×3 (fix 2) · FiLM UNet head (fix 5, collapse) · width-aux head at
ALL weights 0.5-2.5 (no success gain; adaptation only via harmful gradient pressure) ·
18b feed-forward AS an adaptation mechanism (keeps success, width still flat) · episode-min
width as a probe metric (miss-closure artifact).
CONCLUDED SCIENCE: width IS predictable from the cloud (heads corr≈0.8 vs data 0.85,
OOD-generalizes with coverage) → the adaptation failure is PURELY policy learning ·
small-object failures are approach/centering precision, not width (item 17) · OOD is
asymmetric (big easy, small collapses without data) · sim eval is structurally blind to a
co-trained policy's real branch (v33 incident).
ONGOING: iter-4 width mechanisms — rztss (RESIDUAL width, the live bet: adaptation by
construction) + qrbtr (grasp-window loss ×8) training, xqmxw (18b+small data, peak
0.565@100 ≈ neutral so far) · v33b recovery ×4 (plain/aux0.5/aux1.5/shift9-plain — the
shift9 arm doubles as the perception-bias pairing A/B) · tofu v11 (tilt penalty) →
user 650 gate · REAL tests pending on user/local: alzey (item 16, top candidate), jtzqc
(camera DR), peikp (small-mushroom A/B), v33b winners after probe_policy_real_obs ·
bias-magnitude iteration (9 vs 12-13 mm, local agent).
NEW SINCE THE ORIGINAL ROADMAP: second object category (tofu, item 9 started) ·
perception-bias-corrected dataset family + deploy-pairing rule · residual-action
mechanism family (if rztss works, extendable beyond width) · per-object synthesis-knob
calibration as an explicit practice (area floor 15→35, w_tilt for flat faces).


**2026-08-26 — tofu v10: 97.6% (from 70.3%) — burial was the dominant failure driver;
v11 adds the approach-tilt penalty for grasp QUALITY.** v10 (1680342's predecessor
1678043, dataset `26-08-25-xhj`, 40 eps all-video, 0 aborts): spawn z 42 mm + `w_peak 0.3`
+ area floor 35 mm² → demonstrator success 63-70% → **97.6%**. User video review: no more
buried corners; remaining issue is QUALITY — some successes are tilted-gripper EDGE
contacts scoring nearly identically to flush grasps (edge 5682 Pa / align 0.919 vs nice
flush 5718 Pa / align 0.994). Why the metric can't see them: (a) the displayed/objective
stress is the contact-MASKED top10 — the visible red edge-stress stripes are exactly the
masked elements; (b) the FEM contact model presses a ROUNDED parabolic pad — a sharp pad
edge digging in is outside its vocabulary, so even unmasked p98 underestimates;
(c) `align` is orthogonal to approach PITCH (contacted faces still ⟂ closing axis).
CMA budget would NOT help — v10 is argmax, the metric's true optimum. Fix: wired
`--grasp-w-tilt` (scorer's existing `w_tilt·(1−cos_t)` — prices exactly the visible
approach tilt) + `tilt_deg` audit column in dr_params.csv (6a6fcd3). **v11 (1680342) =
v10 + w_tilt 1.5e5** (15° ≈ 5 kPa-equiv). User-flagged exemplars of the target family:
`26-08-25-xhj/videos/ep0020_env4, ep0035_env3, ep0036_env4` (near-vertical, align ≥0.99).


**2026-08-25/26 — ⛔ v33 DATASET POISONED (real slice un-derived) — local agent's real
deploy caught it; fix chain running.** The v3.3 doc's §3 merge command named
`dataset/dppo/single_lift_mushroom_real` — a real slice whose recorded DELTA actions were
written through as absolutes (never derived). Near-zero deltas decode to mid-range
absolutes, so on real-looking clouds the policy learned z→0.252 m / grip→44 mm; deployed
orkam did exactly that (climb + half-close). SIM evals stayed excellent (0.715 — sim
clouds never trigger the poisoned mapping), so the cluster-side gates (seam, dwell) could
not catch it. afucm/s08/all other co-trains UNAFFECTED (properly derived slice; verified
via normalization z-ranges: afucm 0.239 m vs v33's broken 0.438 m).
LESSONS: (1) a co-train dataset gate must check the REAL slice's commanded-vs-achieved
consistency (t0 gripper ~80 mm open; |cmd z − ach z| median <1 cm) — now part of the fix
chain and any future merge; (2) sim eval cannot validate the real-slice health of a
co-trained policy — the poisoned mapping is invisible to sim clouds; the local agent's
offline real-cloud probe (real cloud in, action out) is the cheap pre-deploy check.
Fix-slice verification: my chain gate PASSED (t0 gripper 79.8 mm vs poisoned 44; z lead 10.7 mm = the K4 design lead; gripper tracking 0.1 mm); the local agent's canonical `verify_derived_dataset.py` passes seam, flags dwell 0.429 — the KNOWN-BENIGN real-teleop value (doc: 'the real slice carries 0.42 and works'; the 0.20 threshold is sim-calibrated). CLUSTER-AGENT POST-MORTEM (after reading docs/v33_real_slice_bug.md): two own mistakes beyond the doc's wrong dataset name — (1) followed the handoff §3 verbatim without cross-checking the foundation table's own recipe (real slice = real_merged + derive+K4, recorded in THIS devlog); (2) gated only the new sim conversion, not the reused real slice or the merged norms (the 0.438 m z-max anomaly was in a file the chain created; the foundation's lead gate on the real slice would have read 249 mm). Rule: EVERY slice of a merge gets verify_derived_dataset.py, reused or not. §4.1 executed too: shift9mm real variant chain live → `..._v33b_shift9` + one plain training for the deploy-pairing A/B (v33b runs = real_merged UNCORRECTED → deploy with point_cloud_shift OFF; shift9 run → shift ACTIVE). Pre-deploy rule going forward: run `probe_policy_real_obs.py` on any co-trained checkpoint before recommending deployment. FIX (chain live): re-convert `single_lift_mushroom_real_merged` WITH
`--derive-source-action delta fast_rot --derive-lookahead 4` → gate → re-merge →
`..._noos_cmd_v33b` → retrain v33b_plain / v33b_aux0p5 / v33b_aux1p5. **orkam / engcz / kjljs are DEPRECATED** (2026-08-26, user): remaining eval jobs cancelled, experiments.csv statuses set `deprecated-poisoned-real-slice`; their checkpoints are UNSALVAGEABLE for real deployment — the broken real-branch mapping is learned into the weights (a visual-conditional behavior, not a decode issue) AND the merged normalization itself is contaminated (real-slice ranges compress every sim channel), so no inference-time patch exists; retraining (v33b) is the only fix. Their partial sim curves (orkam 0.715 @200 etc.) are kept ONLY as v3.3-recipe sim evidence, clearly marked deprecated.


**2026-08-26/27 — MULTI-OBJECT SYNTHESIS PROBE (user): banana / strawberry / raspberry on the
njhbz recipe with PURE ARGMAX + peak term. Strawberry excellent, raspberry needed a substeps
fix, BANANA BLOCKED (planner search scaling, diagnosed).** Recipe = njhbz collect args with
`--grasp-diversity-tol 0 --grasp-jitter-deg 0 --grasp-jitter-pos 0 --grasp-pitch-seed-deg 0`
(pure argmax — note ALL FOUR must be 0, the code's `_div_on` gate ORs them) plus
`--grasp-w-peak 0.3`, per-object area floor, everything else identical.
NEW ASSETS (`gentle_manip/scripts/prep_object_mesh.py`, new reusable tool: plane cut → largest
component → watertight repair → uniform scale → recentre):
`banana.obj` 17×16.5×5.8 cm (stem AND its flat cut face removed at the y=0.53 neck, watertight,
from `obj_meshes/banana1`); `strawberry.obj` 4.0×3.8×3.25 cm (calyx cut at y=0.15 then
**morphological opening on the voxel fill** — a plain plane cut leaves leaf stubs that splay
BELOW the cut; opening ×2 strips them at ~1% volume cost — remeshed watertight). Registered
with new materials (banana E 0.25 MPa/yield 25 kPa; strawberry E 0.15 MPa/yield 18 kPa) + task,
DR and experiment configs. MPM per size: banana grid 180 (its 420 cm³ would carry 6.5k
particles at the mushroom's density), strawberry grid 320, raspberry grid 600.

| object | eps | demo success | PINCH | stress_top10 (yield) | grip | align | min_pad | width |
|---|---|---|---|---|---|---|---|---|
| strawberry | 40 | 93.8% | **1/40 = 2.5%** | 8.3 kPa p90 12.6 (18) | 1.86 N | 0.944 | 33.4 mm² | 49.0 mm |
| raspberry | 24 | 100% | **4/24 = 16.7%** | 11.5 kPa p90 15.7 (15) | 0.34 N | 0.900 | 7.1 mm² | 18.2 mm |
| banana | — | BLOCKED (all FALLBACK) | n/a | audit all zero | — | — | — | — |

Reading the two that worked: the **strawberry is the clean case** (flush centred envelops,
stress less than half its yield, min_pad 2× the 15 mm² floor). The **raspberry grasps 100% but
pinches 6× more often** (16.7%) and runs at 11.5 kPa against a 15 kPa yield — p90 15.7 is AT
yield, i.e. the gentlest holdable grasp on a 1.5 cm berry is already a bruising one. Its
min_pad (7.1 mm²) sits just above the 4 mm² floor I set, so the floor is doing little; a berry
this small may simply need a larger floor or a different end effector rather than a better
search. The 4 flagged episodes are genuine (vert +0.4…+7.7 mm — TCP ABOVE the berry centre —
at widths 1.4–4.7 mm vs the 8.7 mm median), not filter artefacts.

**PINCH-FILTER BUG FOUND AND FIXED (thresholds were absolute, now size-scaled).** The filter's
−5 mm vertical / 25 mm width / 15 mm horizontal tests were tuned on a 33 mm mushroom, so on the
15 mm raspberry they flagged **22/24 = 91.7%** — nearly all false positives, since vert −4.5 mm
and width 8.7 mm are a perfectly good envelop at that size. `filter_pinch_episodes.py` now
resolves the object's nominal extent from the run's own experiment config and scales the three
soft thresholds by `extent / 33 mm` (the vert>0 dangling test stays absolute — it is a sign
test). Raspberry then reads 16.7% and the strawberry is unchanged at 2.5%. This is the second
time absolute thresholds have misfired on a small object (the 0.9-scale mushroom2 batch was the
first) — any new geometric gate should be written scale-relative from the start.

**Strawberry is the good case**: flush centred envelops (visual check clean), stress less than
half its yield, min_pad 2× the 15 mm² floor, and area floor 15 + w_peak 0.3 held the pinch rate
to 2.5%. **RASPBERRY: the grid-600 task config is MPM-stable but RIGID-solver unstable** —
"Invalid constraint forces causing 'nan'" killed 5/5 batches (0 episodes saved) even though
synthesis itself produced 40 clean grasps. CFL supports ~325 substeps so the config's 350 is
marginal; new `single_lift_raspberry_soft_stable.yaml` raises it to 560 (the cluster's original
config left untouched). **BANANA IS BLOCKED — an ELONGATED-OBJECT limit of the planner, not a size or
physics problem.** Eliminated one at a time: holdability is fine (FEM reaches 15 N at 4 mm
indent vs the 2.79 N needed, at 8.7 kPa); not the area floor; not the azimuth bound; not the
pitch-seed fan; and NOT the object size — the user re-specified the banana at **9.5 cm nominal
extent with scale DR [0.9, 1.1]** (asset regenerated: 9.5 × 9.2 × 3.24 cm, 73 cm³, **69.5 g**,
required grip only 0.49 N — a far more realistic banana than the first 17 cm / 398 g cut) and
it STILL fails on every episode. Instrumenting `score_finger_grasp` over 400 random in-bounds
candidates explains it: the CMA xy box is 1.2× the object's world bbox, so banana
half_xy = 60×62 mm vs strawberry 26×24 mm — a ~6× larger area — while the banana's footprint
is a thin curved band filling ~23% of its box (strawberry ~60%) AND its feasible yaw is coupled
to the local tangent, which varies along the arc (a round berry accepts almost any yaw). Random
hit rate is 0 for BOTH objects, so the planner depends entirely on its seeds + local descent;
that works when the object fills its box and fails when it does not. 24 starts × 6000 evals
recovered only 2 real grasps in 16 attempts. Every saved banana episode is a DEFAULT top-down
fallback — **spot these by the all-zero audit columns (stress/grip/align/min_pad = 0)**, the
demonstrator "success rate" stays high and is meaningless. FIX (not done, needs a code change):
seed the search along the object's MEDIAL AXIS with tangent-aligned yaw instead of seeding the
bbox centre. Applies to any elongated produce (banana, carrot, chilli, bean).

**RASPBERRY: two independent silent failures, both fixed** (`single_lift_raspberry_soft_stable.yaml`;
the cluster's original config is untouched). (1) grid 600 + 350 substeps NaN'd the RIGID solver
("Invalid constraint forces") on 5/5 batches — CFL supports ~325 so 350 was marginal; 560 fixes
it. (2) With that fixed it still saved 0/24 across 21 batches **with no error at all**: the
collector lifts `LIFT_HEIGHT = 0.20 m` above the grasp, but the config's MPM domain stopped at
z = 0.15, so the berry left the grid on every lift and the success check silently failed 160/160.
Ceiling raised to 0.27 (and the success band widened to match a 20 cm lift) → 8/8 on the first
batch. Lesson for any new object: **check `mpm_bounds` z against grasp_z + LIFT_HEIGHT** — the
failure mode is a silent 0% save rate, not a crash.

**2026-08-26 — local-agent response to the v33 post-mortem: the handoff doc was the origin;
both cluster findings actioned.** Owning the first cause plainly: **`v3.3_synth.md` §3 named
`dataset/dppo/single_lift_mushroom_real` in the merge command** — a stale prebuilt npz that had
never been derived. The cluster agent followed the handoff as written, so the poisoning
originates in my doc, not in their execution. §3 is now **corrected in place** with a visible
banner: it builds the real slice **from the demo pkl with `--derive-source-action` + K4 every
time** (never reuse a prebuilt real npz), points at the shift9mm variant for the corrected
build, and adopts the cluster's rule that **every slice of a merge is gated, reused or not**,
plus the merged file (its action ranges are a cheap tell — v33's z max read 0.438 m where no
sim collection exceeds 0.235 m).
Second finding accepted: **the dwell gate false-positived on correctly-derived real data**
(0.429 vs my sim-calibrated 0.20). `verify_derived_dataset.py` now takes `--source
sim|real|mixed` with ceilings 0.20 / none / 0.35 — under `--source real` dwell is REPORTED, not
failed, because human teleop legitimately pauses AND it does not discriminate there (poisoned
0.51 vs correct 0.43); the discriminating checks for a real slice are derivation and lead
(poisoned gripper 44 mm / lead 249 mm vs correct 79.8 mm / ~11 mm). Re-verified: the poisoned
slice still fails on derivation+lead under `--source real`. A gate that cries wolf on every
valid real dataset is how the next real defect gets waved through, so this mattered.
Agreed with the rest of the post-mortem: deprecating orkam/engcz/kjljs is right — the broken
mapping is learned into the weights as a visual-conditional behaviour AND the merged
normalization is contaminated, so there is no inference-time patch. The v33b + v33b_shift9
pairing A/B is exactly the §4.1 experiment.

**2026-08-26 — item-18 iter 4 LAUNCHED (residual width + grasp-window weighting) + tofu
v9→v10.** Following the predictability study (heads corr≈0.8 → policy-learning problem),
two mechanisms implemented (519e973) and launched on the s08 dataset, afucm recipe +
aux w0.5, eval on [0.8,1.5]: **rztss** = RESIDUAL WIDTH actions (dataset relabels action
dim −1 as command − episode width in action units; eval adds the width head's prediction
back via GM_RESIDUAL_WIDTH — adaptation guaranteed by construction; transform validated
offline: closure residual ≈ −13 mm = squeeze+compression offset) · **s08_winw** =
width-dim ε-loss ×8 on chunks overlapping the closing/hold window (~58% of steps; fixes
pyzpl's global-weighting bluntness), queued on quota. Also running: **xqmxw** = s08_18b
(aux0.5+feed on the s08 data). TOFU: spawn z 0.016→0.042 (8bcb1e6; user spotted tilted
spawns burying corners — worst-case half-diagonal 36.4 mm at scale 1.4); collector gained
`--grasp-w-peak` / `--grasp-w-area` flags (635f9c4) — the §11.7 peak term (unmasked p98,
0.3) existed but a legacy-default bug forwarded 0 in EVERY run to date, so corner-grasp
spikes were invisible to the objective (user's hypothesis, confirmed in code). Smoke v9
cancelled unrun (user); **v10** (1678043, queued) = v8 argmax recipe + spawn fix +
w_peak 0.3 + area floor 35 mm² (tofu-calibrated: failure pads 38.8 mm² sat above the
mushroom-stem-calibrated 15) — tests burial + corner grasps in one go.


**2026-08-26 — WIDTH PREDICTABILITY STUDY (user question: is width predictable from the
cloud at all?): YES — corr ≈0.8 vs the 0.85 data ceiling → the failure is PURELY policy
learning.** Offline eval of the trained width heads (predictions at 0/15/30% episode
timesteps, denormalized mm; `.agent_tmp/eval_width_head.py`, job 1670446):
dgvmu in-domain corr 0.785 / MAE 2.8mm / R² 0.60; bcvrt 0.736/3.0/0.54; rturn (s08-trained)
0.819/2.7/0.67 in-domain AND 0.804/2.4/0.53 on the smallband OOD. The [1.0,1.5]-trained
heads clamp below range (OOD R²<0) — generalization tracks data coverage, as expected.
CONCLUSIONS: (1) the 1024-pt cloud carries the size signal — perception is NOT the
bottleneck; (2) the executor discards information that is accurate to ~2.5mm at its own
conditioning interface (even when explicitly appended, 18b) — hypothesis: the diffusion
ε-loss barely separates adaptive from constant width (a few normalized units among
hundreds), so the constant is a near-minimum. NEXT-STEP CANDIDATES (not launched):
**residual width actions** (relabel gripper dim as residual from the head's prediction —
adaptation becomes architectural, policy learns only corrections; convert-time transform +
small inference change), grasp-window-only gripper loss weighting, conditioning dropout.


**2026-08-26 — width-adaptation probes COMPLETE across all 9 policies (figure:
`docs/figures/width_at_grasp_all_2026-08-26.png`); the mechanism question is answered.**
New probes (s08 arms on [0.8,1.5]; bcvrt on the standard range): bcvrt (18b feed-forward)
at-grasp corr **0.09** — flat ~33 mm at every scale: even FEEDING the width prediction into
the denoiser conditioning does not create adaptation (it only avoids the aux-pressure
success cost). peikp (plain+small-data) **0.01** — zero adaptation, yet the best success of
its cohort. rturn (aux1.5) 0.07; dxrxd (aux2.5) 0.19 (succ-only 0.36 — heavy gradient
pressure bends behavior slightly, echoing eqrth).
**CONSOLIDATED CONCLUSION (9 policies, 2 data regimes, 4 mechanisms):** diffusion policies
here do NOT imitate the demonstrator's width adaptation (data r=0.85) under ANY tested
mechanism (aux loss 0.5-2.5, loss reweighting, FiLM, feed-forward conditioning). They pick
ONE data-appropriate operating width (s08-trained arms ~27 mm vs [1.0,1.5]-trained ~33 mm —
the constant tracks the data distribution) and rely on positioning; success gains come from
DATA COVERAGE (small-scale complement: 0.13-0.22 → 0.45), and the residual size gradient is
approach/centering precision (item 17). Width adaptation as a target is hereby DEPRIORITIZED;
the open levers for small objects are perception (object-region point budget) and centering.
s08_18b (aux0.5+feed on the s08 data, job 1664288) still trains — tests whether 18b's
no-cost property + small data stack; prediction: ≈peikp success, flat width.


**2026-08-25 (night) — tofu smoke v3 (1653490): 63.5% demonstrator success; diagnosed as
EDGE-PINCH grasps; v4 relaunched on the full v3.3 recipe.** 40/40 episodes + all videos at
`dataset/demos/single_lift_tofu_soft/26-08-25-zeo/videos/`. Failure signature from
dr_params.csv: failures have HALF the pad contact area (25.4 vs 59.1 mm²) and much higher
contact pressure (91 vs 63 kPa) at similar width/grip — the classic cube edge/corner-pinch
mode (grasp_synthesis notes: the cube is the multi-optimal hard case), which v3.3's
anti-pinch terms (`--grasp-area-min-mm2 15 --grasp-w-press 0.05`) exist to demote. The v3
smoke predates the v3.3 merge, so it ran without them. v4 smoke (1654259) = full v3.3
recipe on tofu (anti-pinch + continuous approach + 20-step close + 10 stop frames), 40 eps
all-video, same gate: user reviews videos before the 650 run.
**UPDATE (same night) — smoke iteration series & ROOT CAUSE:** v4 (full v3.3, 1654259)
62.5% — anti-pinch improved contact (fail pad 25→39 mm², pressure 91→62 kPa) but success
flat → pinch was a symptom. v5 (2 mm squeeze, 1655154) 56.3% — WORSE, so squeeze direction
is mushroom-like (more grip = fewer drops); yield-through ruled out. Failure STILLS
(videos_failed/*_grasp.png) showed the real mode: TILTED corner/edge grasps high on the
block, align 0.74-0.79. dr_params align-vs-success on v4: align≥0.9 → 100% (25/25),
<0.85 → 26%. Cause: the collector's `--grasp-align` default 2000 (deliberately lowered
from the metric's 3e4 for mushroom pose diversity) lets the flat-faced cube's degenerate
tilted-optima family through — §11.7's cube pathology exactly. v6 smoke (1655383): v4
recipe + `--grasp-align 30000` → 64.5%, align distribution UNCHANGED (prediction ≥85%
FALSIFIED — the knob didn't bite). v7 (1655807, jitter 30°→8°): 67.8%, align STILL
unchanged (p10 0.685 vs cos8°≈0.99 expected) — jitter isn't the tilt source either.
Remaining mechanism: `--grasp-diversity-tol 0.3` picks RANDOMLY among all grasps within
30% of best score, and the cube's genuinely-tilted CMA-ES optima sit inside that window
(diagonal grasps score decent FEM stress, offsetting the align penalty). v8 (1655995):
diversity-tol 0 + jitter 0 → pure argmax with w_align 3e4 — align p10 must approach 1.0
BY CONSTRUCTION; if success still ~68%, align was a confound and iteration stops.
**FINAL STATUS (investigation closed pending user/local-agent):** v8 (argmax, no
diversity/jitter, w_align 3e4, 1655995) = 70.3% — best of the series but align p10 still
0.72: in some scenes the TILTED grasp is the genuine FEM argmax even at w_align 3e4
(scene-level: batches at align 0.90-0.94 succeed 88-100%, batches at 0.73-0.84 succeed
25-38%). Spawn-tilt hypothesis falsified (corr(spawn tilt, align)=0.016 over 192 attempts).
Solid facts: align≥0.9 → 97-100% success in EVERY variant; the entire failure mass is
align<0.85; anti-pinch/squeeze/jitter/align-weight knobs individually insufficient.
ROOT FIX (proposed, NOT implemented): finger_grasp.synthesize_grasp needs the §11.7-style
canonical flush-seed round-2 (exists in plan_width_grasp, absent here) so a flush candidate
is always evaluated per scene — local agent's module, coordinate before changing.
Series: v3 63.5% (-zeo) · v4 62.5% (-dpg) · v5 56.3% · v6 64.5% (-ter) · v7 67.8% (-cvu)
· v8 70.3% (-bud). All-video dirs for user review; recommended interim recipe = v8's.
Tofu 650 + trainings remain GATED. Note: seed-0's 8 scene draws
all landed ≥1.0 scale (4% chance, verified the [0.8,1.4] range IS active — batch-1 draw
1.182 matches the new range's transform of the old 1.318 draw).


**2026-08-25 (night) — OVERNIGHT PLAN (user, before sleep): v3.3 synth campaign on upstream
arrival.** The local agent is finishing an improved grasp synth (v3.3). Standing order for the
cluster agent: (1) periodic watcher on origin/master (armed); (2) on push: merge properly,
READ `docs/v3.3_synth.md` (documents the new-synth collection + training procedure);
(3) collect 650 demos with the new synth using ALL 4 mushrooms (mm4 mesh pool), the NEW
size range ([0.8, 1.5] — needs an mm4+s08 DR variant), and train with the width-aux
supervision at MULTIPLE weights (informed by attempt-1/2: w=0.5 healthy, w=2.0 harmful,
18b feed-forward @0.5 matched baseline 0.675 — so likely 0.5 / 1.5 [+ feed variant]).
Tofu 650+training remains GATED on the user's video OK (gate now on smoke v10, see the tofu series entry).
EXECUTED: v3.3 push merged clean (8abfeb6); campaign chain LAUNCHED per doc §1-5 — 650-ep mm4_s08 collection (v3.3 recipe verbatim incl. anti-stem + pinch filter), convert with the CORRECTED gates (within-episode seam; dwell <0.20 by design), merge_npz_datasets with the real 55 → `..._noos_cmd_v33`, then 3 arms: v33_plain / v33_aux0p5 / v33_aux1p5 (width-aux weights per the user's overnight ask), eval on the standard `_7d_realws` experiment vs afucm. Collect sbatch gained EXTRA_ARGS passthrough for the new flags.
**COMPLETED (2026-08-26 early):** collection 26-08-25-clq 650 eps / 93.8% demonstrator
(per-mesh 0.90-0.98, scale<1.0 at 0.947, 3 NaN-guard discards / 87 batches); pinch filter
dropped 51 (7.8%, above the 2-5% expectation — wide-cap variants add rim-grasp chances) →
599 kept; gates PASSED (0 within-episode seam jumps, dwell 0.193 < 0.20); merged with the
real 55 → `noos_cmd_v33`. Trainings RUNNING: orkam (v33_plain, 1663145) · engcz
(v33_aux0p5, 1663146) · kjljs (v33_aux1p5, 1663147), eval watchers armed.


**2026-08-25 — PRACTICE CHANGE: MPM sampler `regular` is now the GLOBAL default
(user directive), incl. demo collection; tofu E 4e3→5e4.** The first tofu smoke
(1653283, cancelled) collapsed into a particle pile — two compounding causes:
(1) collections never pinned `GM_MPM_SAMPLER`, so on aarch64 they ran genesis's
`random` sampler (under-connected particles) while evals were pinned `regular` — a
silent train/eval sampler mismatch that existed for ALL prior collections (tolerated
by the stiff mushroom, fatal for soft tofu); `scene_builder` now defaults to
`regular` everywhere (env override still possible). (2) tofu E=4 kPa is too soft at
our grid resolution — raised to 50 kPa (firm/momen tofu; still 6× softer than the
mushroom), yield 20 kPa unchanged. Also: collections apply NO material DR (registry
values verbatim — only SimBackend servers sample object_E), and the tofu experiments
initially inherited the mushroom's E range [2e5,3e5] for eval servers → new
`dr/soft_orientation_realws_tofu.yaml` (E [3e4,8e4], ν [0.28,0.38]). Smoke relaunched
(1653451). NOTE: the running s08 smallband collection (1651703) still uses the
pre-change random sampler — CONSISTENT with the afucm base dataset it complements;
the first regular-sampler mushroom collection will need a comparability check.


**2026-08-25 — SECOND OBJECT CATEGORY: 3 cm tofu block (item-9 groundwork).**
New object `"tofu"` (registry): the item-1 3 cm cube geometry as a genuinely SOFT body —
tofu material preset E=4 kPa / yield 20 kPa (vs mushroom 300/40) — subdivided cube mesh
`assets/objects/tofu.obj` so shape DR + the CMA-ES SDF work unchanged. Task
`single_lift_tofu_soft` mirrors the mushroom task exactly (band/hold/rewards/resolution;
stress yield auto-injected from the material); experiments
`single_lift_tofu_soft_abs_action_armfocus_{realws,7d_realws}` are the mushroom twins
(same realws DR, scale [1.0,1.5] → 3-4.5 cm). Plan (user): 40-ep ALL-VIDEO smoke
(job 1653283) → user verifies videos → 650-ep collection → two PURE-SIM trainings
(afucm recipe + eqrth aux-width w=2.0 recipe). Watch-item: the adopted 5 mm
grasp_extra_close was tuned on the stiff mushroom; on 4 kPa tofu the same squeeze
deforms far more — the smoke videos are the check.


**2026-08-24 (late) — width-at-grasp probes on the item-18 / fix arms (figure:
`docs/figures/width_at_grasp_2026-08-24.png`, 5 policies side by side).**
Metric: commanded width at the grasp→lift transition (EE-z min → first frame risen >2 cm;
onset found in 300/300 episodes) — replaces episode-min width, which miss-closures inflate.
References: demo data r=0.85 · afucm −0.04 · prmaw +0.24.

| policy | at-grasp corr | succ-only | small/big-half width | ever small/big |
|---|---|---|---|---|
| dgvmu (aux w0.5) | −0.02 | 0.05 | 30.4 / 30.1 mm | 0.67 / 0.77 |
| eqrth (aux w2.0) | +0.30 | **+0.51** | 33.5 / 35.0 mm | 0.63 / 0.80 |
| pyzpl (grip-loss ×3) | +0.04 | 0.14 | 32.5 / 32.5 mm | 0.57 / 0.80 |

Reads: (1) dgvmu — the executor completely ignores the size the head provably encodes
(converged aux loss, flat ~30 mm commands): the cleanest possible motivation for 18b.
(2) eqrth — heavy aux weight DOES push adaptation into behavior (+0.51 succ-only, the
highest policy corr measured yet) but at the 0.13 success cost; 18b aims for this
adaptation without the gradient-pressure cost. (3) pyzpl — loss re-weighting creates no
signal, as expected. (4) Notable: dgvmu small-half ever-success 0.67 vs afucm's 0.57 on
the identical probe — the aux head helps small objects WITHOUT width adaptation,
consistent with encoder size-awareness aiding approach/centering (item 17's dominant
failure mode). Next: same probe on 18b (bcvrt) best checkpoint when its curve lands.


**2026-08-24 (later) — mesh-pool DR (`object_mesh_pool`) + 4-mushroom smoke collection;
lesson: the v3 collector MIRRORS SimBackend's scene DR.**
New DR knob `object_mesh_pool` (dr_config.py): per-scene-rebuild uniform pick of the base
mesh from a list of registry names, same cadence as size/shape DR, deform applied ON TOP of
the pick; audited as `mesh_variant` (scene_params + dr_params.csv column). Configs:
`dr/soft_orientation_realws_mm4.yaml` + experiment `..._armfocus_realws_mm4` (= adopted
realws setup + pool [mushroom, mushroom1, mushroom2, mushroom3]). No-op guarantee for all
existing configs verified (pool defaults None → identical control flow).
**LESSON (bug caught at launch):** patching `SimBackend._apply_scene_dr` alone is NOT
enough for collections — `collect_demos_synth_v3.py` has its own mirrored `_apply_scene_dr`
(bakes scale into the exported mesh for the CMA-ES SDF; deform dir `gm_synth_deform_*` vs
SimBackend's `gm_deform_*` — the prefixes in the Genesis spawn log are the tell). First
smoke launch (1639566) was collecting nominal-mushroom-only; caught in the ~45 s startup
check by the deform-dir prefix, cancelled, collector patched (same pool pick + audit
column), relaunched as 1643324. Any future scene-DR extension must touch BOTH paths
(CLAUDE.md "keep the two in sync" applies to scene DR, not just privileged obs).
Smoke recipe: adopted hwo/v3 recipe, N_EPISODES=150 N_ENVS=8, SCENE_DR_EVERY=1.
**Second lesson from the same smoke (relaunch 1643324, FAILED batch 5):** a rare scene
draw can NaN the rigid solver at reset-settle (`Invalid constraint forces` — precedent
job 1300574, v2/nominal-mesh, so NOT mesh-pool-specific; batch 5 drew mushroom3 @ scale
1.490). Fix (01d81ab): the v3 collector now retries reset/settle up to 3× with a freshly
rebuilt scene (new DR draw) instead of dying; persistent failures still raise. All 4 pool
meshes had collected fine in batches 1-4 before the crash. Third launch (1647974) died the same way at the SAME batch-5 scene (seed-identical DR
stream: mushroom3 @ scale 1.490) but mid-episode AFTER a clean settle — the scene is
systematically unstable, GPU nondeterminism only moves where it blows. Fix 2 (b272928):
batch-level guard — on any mid-execution solver NaN, discard the batch, rebuild with a
fresh DR draw, continue (5 consecutive aborts still raise).
**Fourth launch COMPLETED (1649397): 150/150 episodes, demonstrator success 91.5%
(matches the nominal-only recipe's 91-94% — the 4-mesh pool costs nothing), 111 min.**
Dataset: `dataset/demos/single_lift_mushroom_soft/26-08-24-rnh/` (dr_params.csv has the
mesh_variant column). Per-mesh demonstrator success: mushroom 0.938 (n=48) ·
mushroom1 0.896 (48) · mushroom2 0.950 (40) · mushroom3 0.875 (32). The guard fired
2×/21 batches, BOTH on mushroom3 at scale ≥1.3 (1.308, 1.360; plus the pre-guard
1.490×2) while mushroom3 ≤1.27 and every other mesh at any scale never blew up —
mushroom3 (wide-flat cap) at ≥1.3× is the only unstable region; options if it matters
for the full collection: per-mesh scale cap, or accept ~10% discarded batches.
Partial dirs from the dead attempts (26-08-24-qjm, 26-08-24-ivq) left on disk, ignorable.

**2026-08-24 (later) — 3 TripoSG mushroom scans (MIT-licensed generator) normalized into assets (item 6).**
`obj_meshes/{mushroom1,mushroom2,mushroom3}/clean.obj` (~6 k verts each, unit-scale,
y-up with stem along +y) converted to the nominal `assets/objects/mushroom.obj`
convention and added to the registry:

- rotation (x,y,z)→(x,z,−y): stem +y → −z, so cap-up at +z like the nominal (both
  orientations verified by 3-view point projections before and after);
- uniform scale to the nominal's MEAN extent 33.2 mm (per-mesh factors ≈0.0177-0.0189);
- origin convention matched: xy bbox center at 0, z origin 42.7 % above the bbox bottom
  (nominal z-span −14.8..+19.9 mm; variants land −13.4..−15.3 .. +17.9..+20.5 mm);
- faces copied verbatim (no vn/vt in the sources), METERS, headers document the transform.

Written: `gentle_manip/assets/objects/mushroom{1,2,3}.obj`; registry entries
`"mushroom1"/"mushroom2"/"mushroom3"` (mushroom material, `object_type="soft"`,
`default_pos=(0.47,0,0.016)`, sizes 32.3×32.2×35.1 / 31.8×31.9×35.9 / 35.7×32.6×31.3 mm
from measured extents; m3 is a genuinely wide-flat specimen). Verification figure:
`docs/figures/mushroom_variants_2026-08-24.png` (x-z projection, nominal + all three,
same mm axes). Registry import + mesh-path existence checked. Next step for item 6:
run collection/eval over the variant set (spawn-z sanity per mesh first — same check
that caught the 1.8× OOD floor clip).

**2026-08-24 (later) — banana1 / strawberry1: the euler gate's real failure mode is
THIN APPENDAGES, and the handles are GENERATED, not a cleanup artifact.**
Pass rates: mushroom1 9/12, banana1 **1/6**, strawberry1 **0/3**.

`scripts/mesh_from_photos/genus_trace.py` (new) walks euler through every cleanup
stage and sweeps decimation ratios. Measured on the LARGEST COMPONENT, before any
decimation:

| mesh | pre-decimation | after decimation (2x-10x staging, 12k or 20k) |
|---|---|---|
| strawberry photo_seed0 | watertight, **genus 36** | genus 13-15 |
| banana ..5616_seed0 | watertight, **genus 1** | genus 1, identical at every setting |
| mushroom1 back_seed0 (control) | **genus 0** | genus 0 |

**Conclusions:**
1. TripoSG GENERATES the handles. Decimation does not add them — for the strawberry it
   REMOVES them (36 -> 14). Decimation ratio/target has no measurable effect on genus.
   No amount of cleanup tuning will fix this; do not go looking again.
2. The handles live in **thin appendages** at the ~512^3 occupancy-grid limit: the
   strawberry's curling calyx sepals and the banana's thin stalk. Two thin surfaces
   passing within a voxel fuse into a tunnel. The mushroom is chunky with no thin
   parts, which is exactly why it scores 9/12. Predicted (and to be checked) that
   mushroom3, whose stem is torn into a thin sheet/fin, fails more than mushroom2.
   NOT the achenes — the strawberry's seed pits survive decimation as clean dimples.
3. **CORRECTION to the earlier entry: "staged decimation" is probably NOT what fixed
   mushroom1.** Staging shows zero effect on genus in all three traces. The original
   failure was `euler=4, comps=2` (two closed components), and the operative fix was
   the **post-decimation floater pass** added in the same patch. Staging is harmless
   but appears to do no work; do not credit it.
4. **genus > 0 does NOT block FEM.** tetgen needs closed + manifold +
   self-intersection-free; a genus-11 watertight manifold tetrahedralises fine. The
   spec's "a fruit should be genus 0" is a PLAUSIBILITY check for spotting artifacts
   (and it works — it caught real ones), not a solver requirement. Options for these
   objects: crop the calyx/stalk (the part that is not grasped) to reach genus 0, or
   downgrade euler to a warning and gate on self-intersections instead.
   **Known gap:** `self_intersections` is in the spec's section 6 report schema but is
   NOT yet computed by `postprocess.py` (no pymeshlab on aarch64; would need an
   rtree broadphase + exact tri-tri test).

**Also: `rembg` handles a hand-held object far better than expected** — the operator's
hand was removed cleanly from both banana frames. Residue is a small grey smudge at the
grip point (soft-alpha 0.113 vs mushroom's ~0.05). Where the hand OCCLUDES the object
(banana view 1's stem) the geometry is simply invented, same class of problem as the
mushroom underside. `u2net_human_seg` is available in the installed rembg if explicit
hand subtraction is ever needed.

**2026-08-24 (later still) — mushroom2/mushroom3 CONFIRM the thin-structure diagnosis.**
Pass rates across five objects: mushroom2 **9/9**, mushroom1 9/12, mushroom3 **3/9**,
banana1 1/6, strawberry1 0/3. mushroom2 vs mushroom3 is a near-controlled comparison
(same species, same rig, minutes apart, both clean mattes); the only material difference
is mushroom3's stem being torn into a thin fin, and yield drops 9/9 -> 3/9. Within
mushroom3 it degrades by view with stem thinness: U-notch 2/3, thin spike 1/3, broad fin
0/3. The prediction made before running was correct. Selected meshes: mushroom2
`IMG20260824150816_seed0`, mushroom3 `IMG20260824150710_seed0` (its cap is clean in every
candidate — only the stem carries handles). New: `scripts/mesh_from_photos/write_readme.py`
generates a per-object README + promotes the chosen candidate; `mesh_from_photos_object.sbatch`
runs any object end-to-end. **Also found: an ODD euler (mushroom3 ..0710_seed1, euler=1)
means non-orientable / not-truly-manifold, NOT handles — trimesh's `is_watertight` only
checks 2-faces-per-edge. The `genus` field assumes orientability and is meaningless for
odd euler; use `is_winding_consistent`.** And the SKEWER is reconstructed as a rod in all
three mushroom2 view-3 seeds — occlude rig hardware at capture time.

**Heuristic gap worth knowing:** `_prep_report.json` called all of banana/strawberry
"clean". A hand attached to the fruit is ONE connected component, does not touch the
border, and barely moves the soft-alpha fraction, so no automated check fires. The
section 3 matte review must stay a human visual gate.

**2026-08-24 — Item-17 fix arms launched in parallel (user directive: fixes #2 and #5
alongside item 18; all on the afucm base; 1-2 iterations allowed).**
- **item 18** aux grasp-width head: `item18_w0p5` (1626204) / `item18_w2p0` (1626209) —
  per-episode min-width label computed at dataset load, head on the shared cond feature.
- **fix #2** gripper-dim loss upweight: `fix2_gripw3` (1626293) —
  `WeightedAuxDiffusionModel`, epsilon-MSE dim weights [1,1,1,1,1,1,3] (mean-normalized).
- **fix #5** FiLM conditioning: `fix5_film` (1626309) — `PointNetDiffusionUNet` head
  (Unet1D FiLM-conditions each residual block on the fused feature) replacing concat-MLP.
Acceptance test for all: canonical sweep + width-probe re-run on each best checkpoint
(target: corr(cmd width, scale) → 0.85; small-bin ever-success up from 0.32-0.48;
figures pattern as in docs/figures/width_probe_2026-08-24/). Bug found en route: the
item-12 patch had leaked `use_first_frame_context` into PointNetDiffusionUNet.__init__
(shared anchor string) — UNet was unconstructible; fixed. Run dirs:
logs/dppo/dppo-pretrain/single_lift_mushroom_simreal_realws_noos_cmd/<ids in monitor logs
log_item18_*, log_fix2gripw3, log_fix5film>; slurm logs by job id.

**2026-08-24 — Photo→mesh asset pipeline stood up (TripoSG).** New
capability: photographs of a real object → clean watertight decimated mesh for
`assets/objects/`. Scripts `scripts/mesh_from_photos/{prep_images,generate,postprocess,turntable}.py`,
env `envs/triposg_arrhenius`, outputs `obj_meshes/<obj>/`. Full subpage:
`docs/mesh_from_photos.md`.

**Practice change — Tencent's Hunyuan 3D generator is BANNED on this cluster and in
this project (licence, not technical): its community licence excludes the EU from its
territory and the restriction reaches the model's OUTPUT.** Arrhenius is in Sweden, so
any generated mesh would contaminate `assets/objects/` and every derived artifact
(datasets, figures). Its checkout has been removed from `third_party/` (2026-08-25,
user directive); never re-clone it or download its weights here.
Adopted alternative: **TripoSG (VAST-AI), MIT for both code and weights.** TRELLIS
(also MIT) was evaluated and rejected on aarch64 grounds — `spconv` (a hard dependency,
used as the sparse tensor container itself) has no ARM wheel at any version, and
neither does `xformers`; `flash-attn` is sdist-only.

**Cost of that swap, recorded so it is not rediscovered:** TripoSG is SINGLE-IMAGE.
It has no analogue of TripoSG2mv's `run_multi_image`, so multiple photos give
multiple independent meshes to compare, not one fused reconstruction. If multi-view
fusion becomes necessary, the options are (a) build `spconv`+`flash-attn` from source
for aarch64/sm_90 for TRELLIS, or (b) run TripoSG2mv on non-EU hardware.

**aarch64 (GH200) wheel findings — reusable.**
- `pymeshlab` has NO aarch64 Linux wheel at any version: <2025 is x86-only, 2025.x
  requires `manylinux_2_35` and Arrhenius glibc is 2.34. (The `pymeshlab<2025`
  constraint in `envs/sim_arrhenius` therefore does not actually make it installable
  there either.) Use `fast-simplification` (quadric decimation) + `manifold3d`
  (watertight repair) instead — both have aarch64 wheels.
- `open3d`, `spconv*`, `xformers`: no aarch64 Linux wheels. `vtk`, `rtree`,
  `pymeshfix`, `manifold3d`, `fast-simplification`, `pyfqmr`: yes.
- **Arrhenius GPU nodes DO have outbound network** (unlike Alvis). HF weights can be
  downloaded inside the job; no login-node pre-staging step is needed.

**Two traps that each cost a job.**
1. `TripoSGPipeline.__call__` has `use_flash_decoder=True` by DEFAULT, and that path
   imports `diso` (sdist-only CUDA ext, no aarch64 wheel). `flash_extract_geometry`
   swallows the ImportError in a bare `except`, returns `(None, None)`, and the
   failure surfaces minutes later as `AttributeError: 'NoneType' has no attribute
   'astype'`. Pass `use_flash_decoder=False` → `hierarchical_extract_geometry` +
   skimage marching cubes. No CUDA build needed anywhere in this pipeline.
2. **CORRECTED (was wrong in the first version of this entry): there is NO
   login-node kill.** The `exit 144, no output` I attributed to a login-node limit
   was self-inflicted: `pkill -f "<pattern>"` matches the *pkill command's own shell
   cmdline*, so it kills the shell that runs it. Reproduced deliberately:
   `bash -c 'pkill -f "zzz_unique_marker"; echo SURVIVED'` → exit 144, no output.
   Never `pkill -f` a pattern that appears in your own command line.
   The real login-node issue is CONTENTION, not a limit: loading a 2.1M-face GLB
   there did not finish in 110 s, while a GH200 node does load+split in 2.3 s. So
   still run postprocessing through SLURM — but for throughput, not because it is
   killed. (This account can only submit to the `gpu` partition.)

**Perf note — CORRECTED.** An earlier version of this entry claimed
`trimesh.split()` is minutes-slow at 2M faces and had been replaced with a
`scipy.sparse.csgraph.connected_components` pass. Both halves were wrong: the scipy
patch never actually applied (the editing command was killed by the `pkill` bug
above, so the file kept the original `split()`), and the original `split()` measures
**2.3 s** at 1.87M faces on a GH200 node. No optimisation was needed or made.
What IS verified: decimate BEFORE hole-filling/manifold repair — repair costs
minutes at 2M faces and milliseconds at 12k. And decimate in STAGES (<=10x per pass): a single
155:1 jump from 1.9M to 12k tears the surface — results came back non-watertight with
handles (euler -8) and a second component shed. Staged [186k, 18.6k, 12k] + a second
floater pass gives euler 2.

**First object done — `mushroom1`, 12 candidates (4 views x 3 seeds), 6-13 s each on one
GH200.** Selected `back_seed0`: 11994 faces, watertight, euler 2, genus 0, single
component. Delivered unscaled (no `measurements.json`; longest axis = 1.903129 in
TripoSG's normalised frame). Two results worth keeping:
- **Seed variance is TOPOLOGICAL.** 3 of 12 came back watertight but with genuine
  handles (genus 2-5) and were rejected by the euler gate. The spec's "three seeds per
  object" is load-bearing, not ceremony.
- **The four views disagreed on volume by 2.3x** (back 2.12 vs right 0.93) — and the
  cause was SEGMENTATION, not the model. Each mesh reproduced its own input silhouette
  faithfully; the skewer's white tape flag survived `rembg` in front/left/right and got
  reconstructed as a protruding fin, stretching the silhouette and squashing the object.
  Only `back` had a clean matte. **Practice: the §3 matte review gate is mandatory, and
  rig hardware must be occluded at capture time.**
- **Underside confirmed invented** (`obj_meshes/mushroom1/_underside_check.png`): smooth
  featureless dome, no gills/annulus/rim, because all four views are equatorial. This is
  the region that sets the grasp contact patch, so treat the lower hemisphere as fiction.
  **Requested from user: a photo from below** (obtainable — the object hangs from a
  skewer), plus a top view and two 45-degree obliques.

**2026-08-24 — Item 17 width-probe results (full numbers).** Instrumented 60-episode evals
(12 geometries each, per-step command dumps; artifacts: `.agent_tmp/{prmaw,afucm}_width_ep*.npz`,
`<run>/eval/width_probe/`; slurm 1624552/1624553):

| | corr(min cmd width, obj_scale) | mean MIN COMMANDED width: below-median-scale eps → above-median (policy rows); small vs large scale bins (data row) | ever_success: below- vs above-median scale |
|---|---|---|---|
| training data (both realws collections) | **0.85** | 35.5 → 50.4 mm | demonstrator flat ~0.92 |
| afucm/state_400 | 0.27 | 20.3 → 23.4 mm | 0.57 / 0.77 |
| prmaw/state_200 | 0.44 | 16.7 → 21.6 mm | 0.47 / 0.83 |

(Reading: episodes split at the MEDIAN object scale; each half's value = mean of the
policy's minimum commanded gripper width per episode — its chosen grip. Commanded widths
sit below achieved ones because absolute commands squeeze past the surface; the
DIFFERENCES carry the adaptation signal: policies move 3-5 mm across sizes where the
demos move 15 mm.)

Supporting findings: training scale distribution only mildly thin at 1.0-1.1 (17% vs 20%
uniform; same skew in both collections — same seed-0 scene sequence) and CANNOT explain
the failure (the least-represented bin 1.2-1.3 at 11.5% performs BEST); demonstrator
success flat across scale (no selection bias). FIGURES: `docs/figures/width_probe_2026-08-24/{width_vs_scale_scatter,width_histograms,success_vs_scale}.png`
— the scatter shows demos (gray, r=0.85 trend) vs policy successes (green: wide 22-36 mm
spread at every scale, mild lower-envelope trend, systematically BELOW the demo line) and
failures (red ×, clustered at near-closed widths = closed-on-air). Caveat: the probe's
seeded geometry draw covered scales 1.0-1.33 only — 1.4-1.5 behavior unmeasured.
REFINEMENT (per-bin): success-only widths are NEAR-CONSTANT ~27-29 mm at every scale
(afucm 27.3/29.1/28.0); the low small-bin means came from FAILED episodes closing on air
(11-15 mm) — a positioning-miss symptom. Verdict + reranked fixes in item 17; item 18
(aux grasp-width head) remains proposed as an encoder size-awareness aid.

**2026-08-24 — Post-campaign tail (all concluded).** Width probes for item 17
(widthprobe_prmaw / widthprobe_afucm — instrumented 60-ep evals, 12 geometries each,
per-step command dumps → commanded-width-vs-scale correlation) DONE, results in the item-17
verdict. prmaw state_600 eval DONE (retried after an 8h hang/TIMEOUT on n58, job 1645708):
0.410 / 0.445 ever / 0.465 in-band, sustained 18.7 kPa, peak 49.8 — curve declines after
the 200 peak (0.54), best ckpt unchanged; prmaw row final. Everything else concluded; latest verdicts: item 10 gentle = NEGATIVE
(0.54 peak @200, sustained 25.1 kPa — 0.15 success cost, no stress benefit vs afucm's
0.685/24.0); hold-tail = rescues h8 across seeds (ymbve 0.68, udvpq 0.67 vs jjjjy 0.04)
but mildly hurts healthy configs (ht_afucm 0.49-0.53 vs 0.685); ptpii (item 12) curve
complete, negative stands. User is real-testing the sim shortlist today.

**2026-08-24 — Item 12 first verdict: first-frame context HURTS as implemented.**
`ptpii` (attempt 3): 0.315/0.38/0.265/0.31/–/0.28 vs baseline 0.685 — success halved.
Suspected: the constant 512-d context feature doubles the visual conditioning width
(diluting the fixed-size MLP) or acts as a static distractor. Negative result recorded;
possible follow-ups if revisited: smaller context projection (e.g. 64-d bottleneck),
FiLM-style modulation instead of concat, or gating. Not pursued for now.

**2026-08-24 — Hold-tail augmentation study launched (user request).** New repo transform
`gentle_manip/dppo/augment_hold_tail.py`: +10 frames replicating the final state+command
at every episode end — teaching "after reaching, keep commanding the same pose" (the
post-arrival behavior was OOD in all demos; suspected driver of hold-phase drops and
jjjjy's h8/e4 failure). Datasets `*_n5_ht_cmd` / `*_noos_ht_cmd` (normalization verbatim —
replicated rows change no min/max). Six runs: {wclac(n5), afucm(noos), jjjjy(noos, h8/e4)}
× seeds {42, 43}, canonical sweeps, results to the canonical table.

**2026-08-24 — Monitoring post-mortem (missed failed launch) + fixes.** The item-12
training (gzjkf) died 59 s after submission (dataset-init ordering bug) and went UNNOTICED
for hours despite the standing health monitor DETECTING it — root cause: the monitor ran
as a background shell whose per-line alerts land in an output file that only notifies on
process EXIT; detection worked, the delivery channel was broken. The retry then failed
AGAIN at the first validation pass (val_dataset missed the new flag — overrides must be
applied to BOTH train_dataset and val_dataset). Fixes adopted: (1) the SLURM
failure-detector now runs through the notification-per-line channel (every
FAILED/TIMEOUT/OOM emits an immediate alert); (2) every submission is verified to survive
startup (~45 s in-queue check); (3) lesson recorded: a +train_dataset.X override almost
always needs its +val_dataset.X twin. Item 12 attempt 3 running with both flags.

**2026-08-23 — Allocation items 7/10/12/13/14/15/16 launched (cluster agent).**
All on the afucm setup (realws sim+real co-train baseline 0.685@400) unless noted:
- **15**: horizon 8 / execute 4 training + sweep (config-only).
- **7**: OOD size+shape eval configs (scale [1.5,1.8], bend [25,40], twist [20,35], taper
  [0.15,0.3], axis [1.15,1.3]; pose/material unchanged) — evals of afucm/nmbtz/qjzsf
  (realws-OOD) and vdmtb (std-OOD) peak checkpoints.
- **10**: gentler-grasp collection (extra_close 2.5 mm — HALF, not zero, per conclusion 11)
  chained through commanded-euler convert → real co-train merge → train → sweep; rank vs
  afucm on success AND sustained stress (item 11).
- **14**: training-time camera-pose DR (per-sample rigid cloud perturbation ≤0.5 cm/axis,
  ≤5°, centroid pivot) — `cloud_pose_jitter_*` dataset knobs, training running.
- **12**: first-frame context token — training running (`gzjkf`). DESIGN NOTE: the
  episode's FIRST cloud is encoded by the SAME PointNet (shared weights) and its feature
  concatenated into the conditioning: `[cur_cloud_feat ⊕ first_frame_feat ⊕ state]`.
  Dataset looks up each sample's episode-first frame; the venv snapshots the reset cloud
  and republishes it per step; the anchor frame is exempt from item-14 jitter. This is the
  goal-image-conditioning PATTERN applied to the initial frame (a known lightweight trick,
  not a named method) — a FIXED, non-evolving context. Unlike an RNN it has zero
  recurrence (BC's random-window batching unchanged, no hidden-state management, no
  forgetting/drift) but also cannot integrate information over time or refresh if the
  object moves after frame 0. Escalation path if it shows signal: temporal transformer
  over a longer obs window, RNN as comparison — the flag-gated plumbing (model
  `use_first_frame_context`, dataset `first_frame_context`, venv `first_point_cloud`
  modality) is reusable for both.
- **13** (possible-now variant): masked aux losses — aux_valid mask lets sim rows supervise
  the object-pos head while real rows contribute zero aux gradient; aux-carrying merged
  dataset + training. Real GT labels can drop into the same mask later.
- **16**: PairedRegDiffusionModel — cosine consistency between the policy PointNet's
  features of the 4148 paired real/sim cube3 steps (object-agnostic encoder alignment),
  w ∈ {0.1, 0.5}, both training.
All arms get canonical eval sweeps; results roll into the sections above as they land.

**2026-08-23 — Real-data-amount ablation launched (for real-robot testing).** afucm's
recipe (realws sim 585 eps + plain-concat real, union norm) with the FIRST N real demos,
N ∈ {1, 5, 10, 20, 30} (nested subsets — deterministic split order), 1 seed each, full
eval sweeps on the realws box. Curve endpoints already measured: nmbtz (N=0, 0.71) and
afucm (N=50, 0.685). Datasets `single_lift_mushroom_simreal_realws_n{N}_cmd`. Sim curves
are expected flat (~0.65-0.71 — real data is free in sim); the deliverable is the USER'S
real-robot ranking over N, i.e. how many real demos the pipeline actually needs.
`pull_run.sh` now ships the complete deploy kit per run (checkpoint + config/ snapshot +
the run's own normalization.npz) so each N-arm pulls ready-to-deploy.

**2026-08-23 — DP3 harness integration + two eval-comparability bugs (user-caught).**
DP3 checkpoints now evaluate through the canonical harness (`eval_dp3_harness.py` +
`dp3_eval.sbatch`). The user noticed from renders that DP3 evals' MPM particles looked
randomly sampled where DPPO's looked regular — real bug: `GM_MPM_SAMPLER=regular` is
pinned in `dppo_eval.sbatch` but was missing from the DP3 launcher (genesis silently
falls back to the random sampler → different physics discretization). Second bug found
while verifying: a venv horizon that isn't exactly `policy_steps × act_steps` skips the
per-batch truncation auto-reset and shifts the DR RNG stream → pose scenarios diverge
from batch 1 on (DP3's 37×8=296 vs DPPO's 75×4=300). Both fixed; batch-0 scenario params
verified bit-identical across stacks. **Eval-protocol rules adopted:** (1) any new eval
launcher must replicate ALL of `dppo_eval.sbatch`'s pinned GM_* env (sampler, substeps
discipline) — diff the sim-server logs' `[scene_builder] GM overrides` line when in
doubt; (2) real-demo-trained policies are evaluated with the `_realws` experiments (the
real workspace box is their data's home turf), sim-trained policies with the experiment
matching their collection's DR.
**2026-08-23 — Item 1 real probe dataset recorded + verified** (user-recorded, local-agent
checks): **`dataset/demos/single_lift_cube3_real/26-08-23-oso`** — 5 teleop episodes / 4,148
steps of the 3 cm red cube placed right below the arm. Confirmed setup: armfocus obs at record
time (fingerprint 0.93–0.94, matches the mushroom foundation band), delta fast_rot actions
(±0.75 speed clip), 30 Hz, per-episode RGB mp4s (`videos/`) via the new `--record-rgb` knob,
and paired RGB|cloud videos rendered (`videos_paired/`, via
`gentle_manip/visualization/paired_rgb_cloud_video.py`). Content: ep1/2/5 real grasps (close
at z≈0.3–1 cm, width settling 37 mm, lifted holding; ep1 also contains a genuine slip+retry),
ep3/4 are air-closes (no cube grasp — width 17–21 mm at z≈8–9 cm). Sim twin staged: `cube3`
mesh/registry/task (true 3 cm, rigid). TWO RECORDER BUGS found via the pairing and fixed with
regression tests: (1) RGB frames were not masked by the idle trim (video ran ahead of the data
wherever the operator paused); (2) a DISCARDED episode's frames leaked as a ghost prefix into
the next episode's video — this was the dominant desync (ghost contains real motion). Legacy
videos from before the fixes align exactly by END-ANCHORING (the save side flushes at the save
tick); the paired renderer does this automatically.

**2026-08-25 — ⛔ v33 REAL SLICE IS BROKEN: undrived delta actions. All v33 policies must be
retrained.** Full write-up: [v33_real_slice_bug.md](v33_real_slice_bug.md). Deploying
`orkam/state_200` on the real arm made it climb in +z with the gripper part-closed; `state_400`
and `kjljs/state_100` behave identically. Root cause is NOT the deploy wiring (verified
correct), NOT the checkpoint, and NOT the v3.3 recipe: the real slice merged into
`single_lift_mushroom_simreal_realws_noos_cmd_v33` was **never derived** — the demos' recorded
DELTA actions were written into the 7d absolute dataset as if already absolute. A delta of ≈0
decodes to the MIDPOINT of each absolute range, so that slice teaches, for any real-looking
cloud, `z = 0.252 m` (midpoint; demos achieve 0.096) and `gripper = 44 mm` (midpoint; demos
hold 80). The robot reproduced those numbers exactly. Confirmed cluster-side from
`downloaded_runs/orkam/normalization.npz` (action z max 0.75 → 0.438 m; no sim collection
exceeds 0.235 m). **afucm is unaffected** (its merged z max 0.072 → 0.239 m — properly derived
slice), which is exactly why afucm works and every v33 checkpoint does not.
**Fix:** re-convert with `--derive-source-action delta_pose_delta_gripper_fast_rot
--derive-lookahead 4`, re-merge, retrain (command in the doc). **The v3.3 recipe has not yet
been fairly tested** — this failure says nothing about it.
**Two gates added, both verified:** `gentle_manip/scripts/verify_derived_dataset.py` (dataset:
derivation/lead/seam/dwell — flags the broken slice on 4 counts, passes a good one; run on every
convert output) and `examples/sim2real_diagnose/probe_policy_real_obs.py` (policy: real vs sim
vs hybrid observations, exits non-zero if the policy climbs or closes at t0 — afucm PASS,
orkam/kjljs FAIL). **Method lesson (generalizes):** simulated evaluation is structurally blind
to a co-trained policy's real branch — orkam scored 0.715 in sim vs afucm's 0.685 while being
non-functional on real input. Sim ranking doesn't merely fail to transfer here; it cannot see
brokenness at all. Second lesson: a policy that mispredicts on demos **from its own training
set** is a data bug, not a model bug — that one-line check redirected the whole investigation.

**2026-08-25 — v3.3 synthesis READY FOR CLUSTER (recipe + handoff: [v3.3_synth.md](v3.3_synth.md)).**
v3.3 = v3.2 with settle rolled back 6→1 (user), + **approach speed compensation** (per-env
duration = profile arc-length / 0.0024 m/step; speed band 2.73–3.48 → 2.42–2.55 mm/step,
real median 2.37 — the fixed-duration speed∝distance artifact is gone), + **anti-stem/pinch
planner terms** (`--grasp-area-min-mm2 15` worst-pad area floor scaled by scale², +
`--grasp-w-press 0.05`): the FEM metric PREFERS stem grasps (5.4 vs 10.0 kPa — more CMA-ES
would worsen them); the stem grasp was a 96th-pct outlier on min-pad-area (8 vs 49 mm²) and
pressure (114 vs 37 kPa), and enabling the dormant v4 terms removed stem grasps entirely
(min pad 3.6→17.7 mm², 16/16 success, visual check clean on fully-flipped mushrooms).
New DR/experiment `_mm4_s08` (4-mushroom pool × scale [0.8,1.5], items 6+18): per-mesh
success 75–92 %, small scales 100 %. Pinch filter criterion refined to vert-primary (the
absolute width rule misflagged a whole 0.9-scale slim-mushroom2 batch of correct envelops).
Fixes en route: np.trapz→trapezoid (NumPy 2); seam gate must diff WITHIN episodes.
Smokes: 26-08-25-zrg (50 ep, 86.2 %), -uix (16, speed verify), -vqg (16, full v3.3, 100 %).

**2026-08-24/25 — v3.2 synthesis + 200-demo quick verification (run `bsipf`; ROUGH PICTURE
ONLY, user-curtailed at state_200).** v3.2 = v3.1 + real-style CONTINUOUS approach
(`--approach-xy-finish 0.45 0.75`: xy smoothstep finishing early, z linear — no via-point,
no stop; speed guard caps peak xy at 3.2 mm/step), azimuth 45→60 + jitter 20→30 (wider
yaw/pitch/roll), 10 trailing stop frames (`--held-run-max 12 --held-run-keep 10`, the fleli
hold-deficit fix), and the NEW pinch post-filter (`filter_pinch_episodes.py` — flags
dangling/rim grasps via TCP-vs-object geometry at hold; the user-flagged pinch video was
the top outlier; 9/200 = 4.5 % dropped). Collection `26-08-24-cvz` (200 eps, 90.5 %) →
filtered 191 → kinematics vs real: hover-at-alignment 92 mm (real 84), xy-align frac 0.49
(real 0.60), rot 33.5° (real 30). Trained afucm-twin arch, 1200 ep/save 200 on 191+55.

| run | log location (…/dppo-pretrain/) | best ckpt | success | ever | in-band | sustained | peak | remark |
|---|---|---|---|---|---|---|---|---|
| bsipf | `single_lift_mushroom_simreal_realws_noos_cmd_v32/bsipf` | state_200 (only one evaluated) | 0.055 | 0.140 | 0.175 | 16.0 kPa | 49.4 kPa | v3.2, 191 demos; eval stopped after state_200 (user) — state_400–1200 UNevaluated |

Honest read: weak but **not conclusive** — state_200/1200 is only 17 % through its cosine
cycle (fleli's fraction-matched point is state_100: 0.00/0.41), and 191 demos vs 500. Still,
ever-rate 0.14 vs fleli-state_100's 0.41 suggests slower take-off, not just less training.
hold_failure_gap 0.035 (= fleli) — the stop-frame fix is NOT yet confirmed effective at this
early checkpoint. Sustained stress 16 kPa is the lowest recorded (n=11 successes; likely
weak-grasp artifact, not gentleness). Checkpoints exist for later eval; the real verdict
belongs to the cluster-scale rerun (500+ demos, matched fractions). ALSO ADOPTED for that
rerun (measured, user-prompted): **approach speed compensation** — fixed 77-step approach
makes speed ∝ spawn distance (corr 0.91 in sim vs 0.29 in real; real moves at ~constant
2.4 mm/step) → per-env `dur_i = dist_i / v_ref` (the per-env FSM supports it; not yet
implemented). Gate lesson: the euler-seam pre-flight check must diff WITHIN episodes
(traj_lengths) — concatenated diffs cross episode boundaries and false-trip on diverse end
poses (v32 within-episode jump: 0.016 = seam-free; boundary: 1.131).

**2026-08-24 — Offset-corrected paired real variant (`26-08-23-oso-offset`) validates the
bias fix.** The cube3 real clouds shifted by the implemented `point_cloud_shift` [0.009,0,0]
(proprio untouched, zero-pad preserved) → `dataset/demos/single_lift_cube3_real/26-08-23-oso-offset`,
re-compared against the same sim twin: full-cloud chamfer **14.8 → 8.7 mm**, arm segment
13 → 6.5–10.4 mm, object region 25 → ~16.6 mm (= the physical placement offset, correctly
untouched by a perception fix). Residual arm bias +3.9 mm x: the NN-displacement estimator
attenuates under shape noise, so the TRUE bias is likely ~12–13 mm — if the shift is ever
recalibrated, try ~0.012–0.013 (one more measure-shift iteration would pin it). Multi-view
paired videos (offset real | sim): `dataset/demos/single_lift_cube3_rigid/26-08-23-oso-offset/`.
For item 16: the cluster agent can build a second paired npz from the offset variant if they
want the consistency loss to see bias-corrected real clouds.

**2026-08-24 — v3.1 overnight campaign RESULTS (items 2+5 test, run `fleli`) + the missing
STOP-signal finding.** Training-results table (local protocol: 200 eps, seed 42, realws
experiment, scene_group 4, per-episode video; best ckpt = best EVER success; both rows
evaluated on THIS machine for apples-to-apples — afucm's cluster number was 0.685):

| run | log location (…/dppo-pretrain/) | best ckpt | success | ever | in-band | sustained | peak | remark |
|---|---|---|---|---|---|---|---|---|
| afucm | (cluster) `single_lift_mushroom_simreal_realws_noos_cmd/afucm` — local re-eval `downloaded_runs/afucm/eval/2026-08-24_10-11-08` | state_400 | 0.575 | 0.650 | 0.66 | 24.7 kPa | 50.9 kPa | baseline: foundation co-train (hwo-recipe sim + 55 real noos) |
| fleli | `single_lift_mushroom_simreal_realws_noos_cmd_v31/fleli` | state_200 | 0.265 | 0.610 | 0.65 | 30.1 kPa | 53.0 kPa | v3.1 demos (item-2 human-matched grasp event + item-5 azimuth-45); state_100: 0.00/0.41; **evals stopped after state_200 (user call)** — 300–600 unevaluated |

Reading: by state_200 (of 600) the v3.1 policy REACHES the band on par with afucm (ever
0.61 vs 0.65, in-band 0.65 vs 0.66) but does not HOLD: success lags ever by 0.345 (afucm:
0.075), hold_failure_gap 0.035 vs 0.010 (state_100: 0.10). The failure is at the STOP, not
the grasp. **Stop-signal audit** (user hypothesis confirmed): every sim episode ends with
EXACTLY 4 held stop frames — `_trim_long_holds` keep=4 collapses the whole hold phase —
vs 6 in the real demos; one action-chunk of "stop at lift height" supervision. hwo carries
the same 4 (and afucm still holds), so thin stop supervision alone isn't sufficient as an
explanation, but it is the obvious deficit to fix first. **Adopted next step (user):**
increase kept stop frames (e.g. `HELD_RUN_KEEP` 4 → ~10, or exempt the final hold from
trimming), augment the policy, retrain. Secondary observation: v31 sustained stress is
HIGHER than afucm (30.1 vs 24.7 kPa) despite identical squeeze parameters — worth a look
when the stop fix re-runs. Campaign details: [item2_demo_kinematics.md](item2_demo_kinematics.md).

**2026-08-24 — real_lab.yaml `point_cloud_shift` set to the measured bias [0.009, 0, 0].**
The item-1 arm-segment bias (~9 mm −x in every real cloud) is now cancelled at the source
for all future real recording AND deployment (applied to the static cam_ext extrinsic).
Deploy note: sim-trained/co-trained policies should benefit (deploy clouds now align with
the sim training distribution); the real demo slices recorded BEFORE this (merged 55,
cube3 probe) keep their baked-in unshifted clouds — a ~9 mm intra-dataset inconsistency in
mixed training, negligible vs the noise but worth remembering. Cheap real A/B if in doubt:
toggle the shift to 0 in real_lab.yaml and compare a few afucm episodes.

**2026-08-24 — Item 2 kinematics analysis + the v3.1 synthesis update (overnight campaign,
in progress).** Full report: [item2_demo_kinematics.md](item2_demo_kinematics.md). Real
merged 55 vs hwo 650, pose-space at 30 Hz: the hwo recipe already MATCHES human speed
almost exactly (translation 2.20 vs 2.22 mm/step; rotation, approach depth, close-from-
full-open, lift speed all matched) — speed is NOT the remaining data-side lever. The real
differences cluster at the GRASP EVENT: humans hover 6 steps before closing (scripted 2),
close 40 % faster (21 vs 34 steps), rotate less (30° vs 50° from home), stay vertical
(tilt 2.0° vs 7.4°), and squeeze ~4 mm deeper (settle 30.9 vs 35.3 mm — deliberately NOT
copied: fights gentleness + would confound vs afucm; recorded as a slip-robustness lever).
**v3.1 implemented** (`collect_demos_synth_v3.py`, defaults inert): `--n-settle` (hover)
and `--cam-azimuth-max-deg` (item-5 occlusion bound via the FEM planner's shaped azimuth
penalty + camera-perp seed fan — one knob serves occlusion AND the rotation match).
v3.1 recipe = hwo + `--n-settle 6 --n-grasp 20 --cam-azimuth-max-deg 45`; smoke-verified
(hover 6, close 25, rot 43°). New npz-level merge tool `gentle_manip/dppo/merge_npz_datasets.py`
(per-source denorm → concat → joint renorm) builds mixed sim+real datasets whose sources
need different derivations. Overnight run: 500-ep realws collection (`26-08-24-ndr`) →
7d-euler commanded conversion → +55 real (noos, afucm setup: big net 600 ep) → checkpoint
sweep vs afucm under the same local protocol. Results table to follow.

**2026-08-23 — Item 1 gap analysis: the real-sim cloud difference decomposes into a ~9 mm
perception bias + placement offset.** Full report: [item1_cube3_simreal_gap.md](item1_cube3_simreal_gap.md).
On the paired cube3 datasets (below): full-cloud chamfer 14–18 mm/frame. The proprio-pinned
ARM segment shows a systematic real→sim displacement of **+9 mm in x** (8.0–10.8 across all
5 eps, y/z ≤2 mm) = the real rig's perception bias along the cam_ext ray (L515 depth
over-read / extrinsic xy residual — actionable via the existing `point_cloud_shift` knob or
recalibration). The OBJECT segment is displaced +25 mm x: the 9 mm bias plus ~16 mm of
by-eye placement offset (protocol fix: register the cube by jogging the TCP onto it). One
rigid translation explains 43 % of the whole gap (14.8 → 8.4 mm); the residual is diffuse
(L515 noise; real cube renders 58–70 pts vs sim 92 — grazing-angle dropout). Object-point
detail: the real top face is thin (9–16 pts vs sim 27) and reads ~2 mm lower (both domains
read the 31 mm top low at the L515's near-edge-on elevation); the real x-extent flutters
28–51 mm between episodes where sim is a constant 51 mm (unstable silhouette); after
removing the 25 mm translation the object chamfer drops to roughly the noise floor — the
difference is POSITION + SPARSITY, not shape, i.e. exactly the nuisance variation item 16's
paired feature-consistency loss should absorb. z is healthy
(top face within ~2 mm; the historical 6–11 mm table-z offset is absent here). Bonus
findings: armfocus clouds are ~93 % arm with NO far-field table in either domain; rigid sim
replayed an accidental 7 cm cube push to 3 mm (ep1); the real servo's ROTATION tracking lags
(ep4 drifts to ~8–15° during fast yaw) — an execution-side gap already mitigated by the
rate-limit bounds. Paired videos now render 3 views per side (`--render-only` re-render;
RGB|cloud renderer likewise upgraded).

**2026-08-23 — Paired real–sim twin of the cube3 probe (for item 1 analysis + item 16
regularization).** New committed generator `gentle_manip/scripts/replay_real_to_sim_paired.py`:
replays a real recording's delta actions open-loop in sim with the cube at the real
first-frame TCP xy, the sim home Cartesian-matched to the real first frame, and obs/action
processing taken from the real run's own `config.yaml` (the baked authority) — producing a
demo-schema pkl paired STEP-FOR-STEP with the real one. Output (data, not committed — upload
separately): **`dataset/demos/single_lift_cube3_rigid/26-08-23-oso`**, the sim twin of
`single_lift_cube3_real/26-08-23-oso`; per-episode `match_report.yaml` + proprio-overlay PNGs
+ real|sim rolling cloud mp4s live beside the pkl. Pairing quality across all 5 episodes
(`match_report.yaml`): EE err 1.1–1.8 mm mean / ≤10.2 mm max, quat 1.0–1.6° mean (ep4 3.1°:
its fast late-episode yaw teleop shows the real servo's rotation lag vs sim IK, drifting to
~8–15° in the last quarter — positions still ≤1.8 mm; the weakest channel of the pair),
gripper ≤0.6 mm, cloud NN 13.3–16.3 mm mean (includes the known real L515 noise + ~6–11 mm
table-z extrinsic offset, not just replay drift). Integrity verified: actions bit-identical
to the real pkl, all obs shapes match, sim clouds always full 1024 pts. Sim nominal home
landed within ~1 mm of the real home — home_offset correction is tiny. This is the data for
allocation item 16 (paired-feature encoder regularization, cluster agent) and the substrate
for item 1's data-difference analysis. Upload: rsync both `26-08-23-oso` dirs to the cluster
repo under the same `dataset/demos/` relative paths.

**2026-08-23 — Work allocation & sequencing adopted** (user): items 1–14 above; local agent
on real-vs-sim data analysis (cube probe + demo matching) first, occlusion integration and
missed-grasp robustness later; cluster agent on the afucm ablation (done, flat), OOD (done, asymmetric), gentler
grasp, memory, aux-on-real, camera DR; generalist last. Cross-references into existing work
noted in the table (azimuth bound and retry already built; dwell gate and FEM-margin
calibration flagged as prerequisites for items 2 and 10).

**2026-08-23 — REAL-ROBOT shortlist deployment (user-run).** afucm/state_400 ~75% success
(best); qjzsf/state_1000 second; nmbtz/state_500 (sim-best) worst → co-training helps in
real, pure sim still gapped, sim rankings invert across data regimes (conclusions 12; the
co-training open question is resolved). All three deployed via the `deploy_real.sh` entries
(armfocus obs; rate-limit clamp active).

**2026-08-22 → 08-23 — Local agent: v4/v4.1/v5 grasp-synthesis line + the dwell stall.**
v4 delivered (pinch 0.57→0, honest benchmark, min-jerk/Bézier trajectory); v4.1 shelf lift
measured and REJECTED (10–15 pts demonstrator success for −17% sustained stress, largely an
IK-whip artifact; the width release is pure cost); rate-bounded absolute actions + azimuth
occlusion bound + retry-on-slip shipped with gates; grasp-synth benchmark scale bug found +
fixed; `execute_offset` retired (FEM margin vs MPM). Payoff validation: v5 (min-jerk,
preshape) 0/60 → v6 (no preshape) 0-across-a-2×2 despite passing both pre-flight gates →
DWELL identified (user hypothesis, quantified 32% vs 2%) → v7 (linear) 0 → **0.66**
(rot6d state_600) / 0.50 (7d-commanded), independently re-deriving this log's commanded-
derivation fix en route. New third gate proposed: dwell fraction. Remaining gap to 0.76 =
this log's firmness (extra_close 5 mm) + big-net + early-checkpoint levers, now adopted.
Details: `grasp_synthesis_v41_review.md`, `grasp_synthesis_v4_plan.md`,
`grasp_synthesis_v4_algorithm.md`.

**2026-08-20 → 08-23 — Action-space ablation campaign + follow-ups.** Parts A/B/C
(DPPO vs DP3 × abs vs delta × real/sim data), two derivation-bug hunts (euler seam,
fixed-point stall), commanded-target derivation, hwo recipe forensics (R1/R2), the realws
workspace box, 8-run co-training matrix, DP3 canonical-harness adapter + retrains with
full checkpoint retention. Details: the three subpages above. In flight as of this entry:
DP3 harness evals (4 existing ckpts), DP3 Part A retrains (12-ckpt), DP3-on-sim arm
(train + sweep).

### 2026-08-27 — Width adaptation has a LIVE tracker: `docs/width_adaptation.md`

Per user mandate, the width campaign now keeps a single live log+plan updated AS experiments
launch/land, not retrospectively: `docs/width_adaptation.md` (status, number table, experiment
ledger, decisions+reasons, ranked queue, reviewer questions). Analysis detail stays in
`docs/width_predictability.md`; bugs stay in the DEVLOG bug ledger.

### 2026-08-26/27 — CEILING ANALYSIS: the failure is NOT perceptual (likely a paper contribution)

Prompted by the user asking the right question: "is the width predictable from point cloud at
all? If the cloud hides too much, no algorithm can rescue it."

**Demonstrator side (pure CSV, no model).** Across every 650-episode collection, `dr_params.csv`
gives `scene_scale` and the chosen `width_mm` on the same row:

| object | corr(width, scale) | width sd | residual after scale | corr(w,align) | corr(w,yaw) |
|---|---|---|---|---|---|
| mushroom (n=650) | **+0.85** | 6.3 mm | 3.2 mm | +0.27 | -0.03 |
| tofu (n=614) | **+0.79** | 7.1 mm | 4.3 mm | -0.06 | -0.08 |

So the demonstrator's width is ~0.85-determined by SIZE ALONE, and orientation barely enters.
My earlier worry that CMA-ES stochasticity caps learnability was WRONG.

**Orientation hypothesis (user's) TESTED AND NOT CONFIRMED.** For a cube the extent along the
closing axis is s*(|cos|+|sin|), a 41% swing over a 45deg wedge, so tofu width "should" depend
strongly on yaw. Folding yaw into the 0-45deg wedge gives corr ~ -0.08. Reason: CMA-ES's
alignment term picks FLUSH FACE grasps nearly every time, so the demonstrator never enters the
diagonal regime. The geometry is real; the demonstrator suppresses it. (Consequence for the
paper: a "regress width + top-down grasp" baseline is NOT refuted by yaw geometry on our data --
find a different argument, or just run the baseline.)

**Perception side (Step 0, job 1728536; join PROVEN at corr 0.9998 via the width_mm cross-check).**
Frozen lulkx encoder, fresh head, t=0 cloud, mushroom:

    cloud -> object SCALE : corr 0.739
    cloud -> grasp WIDTH  : corr 0.597

Predicting SIZE beats predicting WIDTH, as expected: width carries ~3.2 mm of demonstrator
residual that the head must absorb as noise.

**THE CEILING.** corr(cloud->size) 0.739 x corr(size->width) 0.85 = **~0.63** is the maximum any
width head can reach on this encoder. Our heads measure 0.60-0.67 — i.e. AT the ceiling. My
earlier INFERENCE that corr(cloud->size) >= 0.78 was too optimistic; measured, it is 0.739.

**Therefore the nine head variants were chasing ~0.03 of headroom.** The real loss is:

    0.63 available  ->  0.474 delivered at grasp  (and success 0.867 -> 0.250)

which is entirely a CONTROL problem: turning a decent estimate into a commanded width without
destroying the grasp. All remaining GPU on the mushroom side goes to the shift/floor mechanism,
not to better heads.

CAVEAT: 0.739 is through the POLICY's frozen encoder, so it is a LOWER BOUND on what the cloud
contains — an encoder trained with size supervision could exceed it. That is exactly what the
aux-width retrain (1728356) tests, so it stays alive.

**Tofu is harder on both axes** (predictability 0.79 vs 0.85, residual 4.3 vs 3.2 mm) against a
much narrower tolerance band — consistent with the user's instinct even though the yaw mechanism
they proposed is not the cause. Tofu Step 0 running (1728554, gadkf@300 encoder).

**Not adopted (yet):** switching the level head's target from width to scale. Principled and the
numbers favour it (0.739*0.85 = 0.63 vs 0.597 direct), but it buys ~5% on a quantity that is not
the bottleneck and adds a fitted linear map to the deploy path.

**From `docs/size_adaptation_literature.md` (other agent's scan) — two items that change our read:**
1. §2c "a grasp is itself a measurement": at contact, gripper width + contact force DIRECTLY
   encode object size. This reframes our phase analysis — vision corr collapses 0.667 -> 0.097 at
   closure onset because of occlusion, but that is exactly when PROPRIOCEPTION becomes a direct
   size measurement. We deleted that channel (`width_head_blind=True`) because a sighted head
   "copied" it. The copying was diagnosed as cheating only because we asked for the WRONG TARGET
   (per-step width, where copying trivially wins). Vision and proprioception are complementary IN
   PHASE; a single-frame head cannot exploit that, an RMA-style HISTORY module can.
2. §4 flags "final closed gripper width as an aux MSE target" as UNVERIFIED — that is precisely
   our item-18 head. No external precedent to lean on; the defence must come from our ablations.

### 2026-08-26/27 — BUG LEDGER for the width-adaptation campaign (read before launching evals)

User mandate under a 2-day deadline: "remember all bugs encountered, we have no time to waste".
Every one of these cost queue time or invalidated a result. Guards added where possible.

**B1. NORM defaulted to the WRONG dataset (worst one — silently invalidates a whole eval).**
`dppo_eval.sbatch` line 64: `NORM=${NORM:-.../single_lift_mushroom_soft_pcd/normalization.npz}`.
Any checkpoint from another lineage gets its actions denormalized with the wrong min/max — the
run COMPLETES and reports a plausible success rate, so nothing looks wrong. Ranges differ a lot:
v33b_shift9 action[-1] = [-0.718, 0.818] (span 1.536) vs soft_pcd [-0.500, 0.000] (span 0.500).
Cost: the entire first margin sweep + both shift arms + the baseline probe (5 jobs).
GUARD ADDED: the sbatch now reads the checkpoint's own `.hydra/config.yaml`, compares `env:` to
the dataset NORM points at, and REFUSES to start on a mismatch.
NOTE: job 1728066 (floor margin 0 -> corr 0.474 / success 0.250) DID set NORM correctly, so that
headline result stands.

**B2. Three consecutive launches, one missing env var each.** `EVAL_EXPERIMENT`, then
`SIM_EXPERIMENT`, then `CFG_DIR` — each failing fast but costing a queue round-trip, because I
reconstructed the invocation piecemeal instead of copying the whole working one.
RULE: copy the full working command from a job that ran, change ONLY the experiment variable.
(The sbatch defaults are stale in general: `single_lift_mushroom_soft_eval.yaml` no longer exists
and `CFG_DIR`'s default `single_lift_mushroom_soft_pcd` is not a valid cfg dir.)

**B3. Diagnosed ramp shape from a SUBSAMPLED print.** The "mode averaging" theory came from
`[::12]` sampling; at consecutive frames the truth is a smooth 15-frame ramp and there is no
bimodal decision. Built and ran the whole discretisation fix on a false premise.
RULE: never infer a temporal shape from a subsampled print, and always carry a ground-truth
control computed the SAME way (the `true=14` middle-band count is what exposed it).

**B4. Validated a one-moment quantity on a phase-POOLED average.** Level-head val corr 0.743
pools over t and is dominated by the long approach phase; at the moment the latch actually fires
it is 0.667, and at closure onset/mid-episode it collapses to 0.097/0.030 (occlusion).
RULE: validate a quantity AT THE MOMENT it is used.

**B5. "It can only loosen, so it is safe" — wrong.** The floor cannot create a CRUSH mode, from
which I concluded it could not hurt success. Loosening is exactly how you DROP an object:
success 0.867 -> 0.250. RULE: enumerate the failure mode the mechanism CREATES, not just the one
it avoids.

**B6. Episode-MIN width is not a safe adaptation metric** (user caught). A policy that goes OOD
and closes in mid-air scores a small min with no grasp behind it, inflating apparent adaptation.
AT-GRASP (EE-z min -> first frame risen >2cm) is primary; both are now reported side by side with
a `lift%` column (at-grasp is NaN, never silently backfilled) and a `gapMIN` column that flags
miss-closures directly. `.agent_tmp/probe_report.py`.

**B7. The width-dump writer was an ad-hoc patch that vanished**, leaving previously published
probe numbers unreproducible (including the "~0.1 baseline" quoted for weeks).
FIXED: `GM_WIDTH_DUMP` is now a permanent, mm-denominated feature of `dppo/eval_agent.py`,
recording commanded width AND ee_z, flushed per batch plus an `atexit` flush (reset() alone only
flushes the PREVIOUS batch, so the final one was being lost). Baseline is being re-measured with
identical code rather than carried over.

**B8. `ep_min` shared across train/val splits.** One dict keyed by episode start; both splits
start at s0=0, so val overwrote train targets. Impact measured as negligible (corr 0.743 vs
0.748) but fixed.

**B9. Killed a disproved arm without launching its replacement** — nothing trained for 2h against
a hard deadline. RULE: kill and relaunch in the SAME action.

**B10. Absolute-vs-difference affine conversion (shift mean) — killed both shift arms.**
`GM_WIDTH_SHIFT_MEAN_MM` is an ABSOLUTE width, but I converted it with the DELTA scale factor
`kk = 4/(0.088*(a_hi-a_lo))`, dropping the affine OFFSET. 34.32mm mapped to +1.016 instead of
-0.351, so every episode got a uniform -8mm squeeze (clipped) -> 0.000 success on alpha=0.5 AND
alpha=1.0. An absolute width converts through the FULL affine map; only a difference converts by
the scale alone. (The margin knob and the dump were correct — margin IS a difference, and the dump
already used the full inverse affine.)
GUARD ADDED: a ROUND-TRIP check in `eval_agent.py` — invert the conversion and compare to the
input, raise if it differs by >0.01mm. NOTE: my first attempt was a RANGE check, which does NOT
work here — 0-88mm maps to [-1.367, +1.237], so the buggy +1.016 sits comfortably inside any
plausible bound. Only the round-trip catches it (it decodes +1.016 back to 80.54mm vs 34.32 input).
LESSON: for any two-space conversion, verify by INVERTING it, not by bounding it.

**B11 (positive).** The degenerate-eval watchdog the user proposed caught B10 in 30 episodes
instead of a 2h burn, on its first night. `GM_EVAL_MIN_SUCCESS` (0.05) / `GM_EVAL_MIN_EPISODES`
(20) in `dppo_eval.sbatch`. False-positive math is in the comment there: a true-0.30 policy shows
0/20 with p=0.08%; a true-0.10 policy is killed ~12% of the time. Never raise the threshold near a
rate that matters.

**B12. `verify_derived_dataset`'s DWELL gate is NOT DISCRIMINATING — it fails the PRODUCTION
dataset.** Building an align-filtered dataset, the gate reported `dwell 0.451 -> FAIL` and I nearly
concluded the filtering had broken it. CONTROL (job 1729239): the same verifier on the
known-good `single_lift_mushroom_soft_v33_7d` — the dataset that trained lulkx to 0.820 — reports
**0.449, also FAIL**. Seam passes on both. The DEVLOG's recorded "dwell 0.193" for that same
dataset must come from a different metric definition or an older verifier.
CONSEQUENCES: (a) the gate carries NO signal in its current form and has been giving false
assurance — anything that "passed" it recently deserves re-checking; (b) it needs its threshold
recalibrated or the metric fixed. TODO, not done tonight.
LESSON: when a gate fails, run it on a KNOWN-GOOD input before believing it. Without that control
a valid experiment would have been abandoned on a broken check.

**B13. `envs/rpc.py` WHITELISTS the step-info keys — a new per-step signal vanishes silently.**
Wiring a contact-triggered width controller, `contact_force` was computed correctly in
`genesis_worker` (unconditionally, for soft AND rigid), surfaced by `sim_backend`, and emitted by
`policy_env` — then DROPPED crossing the sim-server boundary, because `rpc.py` ships only an
explicit list (`success`, `obj_z`, `stress_*`). No error, no warning; the controller simply never
fired. FIX: add the key to BOTH sides (server `resp[...]` ~L151 and client rebuild ~L262).
RULE: **any new per-step signal needs an explicit rpc entry or it disappears without error.**
Same silent-drop class as B1 (the normalization default).
CAUGHT BY: a deliberate no-op guard in the controller (raise if no contact after 8 steps). Cost 3
minutes instead of two 40-minute probes reporting a FALSE NEGATIVE on the most promising mechanism.
LESSON REINFORCED: every new mechanism ships with an assertion that it is actually running.

**B14. `collect_demos_synth_v3.sbatch` had NO `EXTRA_ARGS` passthrough — flags silently ignored.**
It builds its arg list from NAMED env vars only, so any flag without a dedicated variable is
dropped without warning. Two raspberry collections ran believing they used
`--grasp-area-min-mm2 / --grasp-width-max-mm / --grasp-yaw-max-deg` and in fact used DEFAULTS.
(The v1 sbatch has the passthrough; v3 never did.) FIXED: passthrough added.
CONSEQUENCE: the raspberry SMOKE's 95.2% success is not a measurement of the intended recipe — it
used defaults AND the mushroom-material bug. Treat it only as evidence raspberry is graspable.
Third instance tonight of the same class (B1 NORM default, B13 rpc info whitelist): **a value is
accepted, silently discarded, and the run completes looking healthy.**
RULE: after launching with new flags, ECHO THE RESOLVED COMMAND and grep for them — do not assume
a variable was consumed.

Earlier, same class: the `--gripper-offset-m` feedback bug (additive bias on an observed channel),
the residual-width two-space unit bug, and the fake "blind" run (sed patched `nc.` while the
trainer read `net_cfg.`). All three are unit/plumbing mismatches that produced plausible numbers.
STANDING RULE for this campaign: any mm<->normalized conversion is done in ONE place, prints its
resolved value in BOTH units, and is sanity-checked against a known span before a long job runs.

### 2026-08-26 (late) — Floor: adaptation WORKS, but an over-prediction bias crashes success

**Result (lulkx@600, width-probe protocol, 60 eps):**

| metric | base | + latched floor (margin 0) |
|---|---|---|
| width/size corr AT GRASP | ~0.1 | **0.474** (succ-only 0.693) |
| commanded width small-half vs big-half | flat | 38.5 vs 41.9 mm |
| success | 0.867 | **0.250** |

So the decomposition DOES deliver adaptation — the first mechanism of nine that ever has.
Adaptation and success are being TRADED, not jointly lost, which is a far better position
than any previous arm reached.

**My "it can only loosen, so it is safe" claim was WRONG.** It cannot create a new CRUSH mode,
and I concluded from that it could not hurt success. Loosening is exactly how you DROP an
object: the floor is a hard max, so any over-prediction holds the gripper too wide and the
grasp slips. I reasoned about one failure mode and ignored the one the mechanism creates.

**Phase analysis (job 1728362) INVERTED my follow-up hypothesis.** I assumed the latch (fired
on the episode's first act()) sampled the head at its worst moment. The opposite is true:

    phase                          corr  bias_mm  RMSE_mm  P(over>2mm)  P(over>5mm)
    t=0 (what the latch uses)     0.667      3.0      5.8         0.58         0.35
    closure onset                 0.097      1.0      7.6         0.49         0.32
    mid-episode                   0.030      1.2      7.9         0.51         0.35

At t=0 the external camera sees the object UNOCCLUDED; by closure onset the gripper and arm
are wrapped around it and the size signal is gone. Latching early is already correct, and the
"latch later" fix I was about to run would have made it worse. (Same occlusion explains the
user-observed drift on lift frames: corr 0.03 there.)

**The aggregate val corr 0.743 is misleading** — it pools over t, and approach is ~77 of ~120
steps, so it is dominated by frames where the object is visible. Per-phase corr is the honest
number. LESSON: for any quantity used at ONE moment of the episode, validate it AT THAT MOMENT,
never on a phase-pooled average.

**Root cause: bias, not timing.** The head over-predicts by +3.0 mm and is >2 mm too wide in
58% of episodes. That is the whole success loss.

**Fixes (both running):**
1. `GM_WIDTH_FLOOR_MARGIN_MM` sweep 2/4 mm (1728367/8) — brackets the +3 mm bias empirically.
2. **Quantile level head** (1728381/2, tau=0.10/0.25): pinball loss instead of MSE. The floor's
   error is ASYMMETRIC (over-prediction drops the object; under-prediction is harmless), so the
   conditional MEAN is the wrong estimator — the tau-quantile over-predicts only ~tau of the
   time, conservative by CONSTRUCTION rather than by a hand-tuned constant. `TAU` env var in
   `.agent_tmp/train_level_head.py`; the trainer now reports bias_mm and P(over>2mm), not MSE.

**Also fixed:** `ep_min` was one dict keyed by episode start shared across train and val — both
splits start at s0=0, so val silently overwrote train targets. Re-fit gives corr 0.743 vs 0.748,
i.e. negligible, so the running evals stay valid; fix kept.

**Process:** three consecutive eval launches failed on one missing env var at a time
(EVAL_EXPERIMENT, then SIM_EXPERIMENT, then CFG_DIR) because I reconstructed the invocation
piecemeal instead of copying the whole working one from job 1728066. Copy the working command,
then change only what the experiment varies.

### 2026-08-26 — Width adaptation: the mode-averaging diagnosis was WRONG; it is PHASE vs LEVEL

**Deadline:** user set a hard 2-day limit (venue). Target: adaptive width AND success >= 0.7.

**I diagnosed mode averaging and was wrong.** The "mushy 66/42/20 vs true 78/50/33" that
convinced me was an ARTIFACT OF PRINTING EVERY 12th FRAME. A fine-grained ramp check
(job 1728131, `.agent_tmp/wbins_traj_check.py`) shows the truth:

    TRUE closure zoom : 75 73 71 69 66 64 62 59 57     <- SMOOTH ~15-frame ramp
    MSE  head         : 60 56 62 51 51 49 50 46 44     <- tracks it, but ~15 mm EARLY
    BINS head (K=64)  : 66 66 66 66 66 66 66 66 66     <- flat: WORSE
    middle-band frames: true=14  MSE=15  BINS=23  |  MAE: MSE 3.9mm  BINS 4.5mm

The closure is a smooth ramp, NOT a bimodal step, so there is no bimodal decision to average
away — and the MSE head's middle-band count MATCHES ground truth. Discretisation (action
tokenisation, K=64 + CE) is the textbook mode-averaging fix and it made every metric worse.
**Line dropped**; the co-trained bins retrains (1728126/7) were cancelled on this evidence.

**LESSON (methodological, reusable): never diagnose a temporal shape from a subsampled print.**
A smooth N-frame ramp viewed every 12th frame is indistinguishable from a step with mushy
intermediates. Any claim about ramp shape/timing must come from consecutive frames, and must
carry a ground-truth control computed the same way (the middle-band count here) — without the
`true=14` column I would have "confirmed" mode averaging from the BINS run too.

**The real failure: PHASE vs LEVEL.** A per-step width head must emit two things at once —
the closure PHASE (where in the ramp am I) and the LEVEL (how tight for THIS object). One head
cannot supply both from our inputs:
  * SIGHTED (proprio width visible): phase is trivially the current width -> learns
    "predict ~= current width", copies its input (80->79.4, 28->28.6), zero adaptation.
  * BLIND (proprio width zeroed): level roughly right, but no phase signal -> ramp fires
    ~15 mm early, 0.000 closed-loop, lift onset in 2% of episodes.
This single mechanism explains EVERY per-step head result we have collected.

**Consequence: the floor is principled, not a hack.** Phase belongs to the policy (its width
command is the only width signal ever trained closed-loop); level belongs to a point-cloud
head (r~0.89 through an MLP vs r~0.1 through the diffusion path). `w_cmd = max(w_policy,
w_level)`, latched per episode, assigns each to its rightful owner and only ever LOOSENS, so
it cannot create a new crush mode — which is what protects the success gate.

**New knob:** `GM_WIDTH_FLOOR_MARGIN_MM` (+ `GM_WIDTH_NORM`) in `dppo/eval_agent.py` —
`w_cmd = max(w_policy, w_level - margin)`. Given in MILLIMETRES and converted inside the
adapter on purpose: the floor lives in npz-normalized action units, and hand-converting at the
call site is exactly how the residual-width bug happened. Sweep running at margin 0/2/4 mm
(1728066, 1728305, 1728306) on lulkx@600 (level head val corr 0.748).

**lulkx is not overfitting** (user asked): 100..600 = 0.575/0.735/0.810/0.805/0.810/0.820 —
flat from 300 on, 600 marginally best, all within +-0.027 SE. Keeping 600 as the retrofit base.

**Process lesson:** I cancelled the first Config-2 runs (1725987-9) for a sound reason (the copy
test had just shown a sighted head useless) but launched no replacement, leaving nothing
training for 2h against a hard deadline. When killing a disproved arm, launch its replacement
in the SAME action.


### 2026-08-27 — Freeze-after-closure is a NULL result, and the contact stop's gain is a LEVEL effect

**Result (canonical, 200 eps, lulkx@600):**

| arm | success | peak | SUSTAINED | vs baseline |
|---|---|---|---|---|
| baseline | 0.820 | 53065 | 28060 | — |
| freeze eps=1mm | 0.825 | 52467 | 27217 | **-3%** |
| freeze eps=2mm | 0.825 | 52814 | 27082 | **-3.5%** |
| freeze eps=3mm | 0.815 | 52633 | 26954 | **-4%** |
| contact stop 1.5N | 0.810 | 51053 | 21241 | -24% (NOT deployable) |

**Why the freeze failed — the trigger was aimed at the wrong failure mode.** `GM_WIDTH_FREEZE_MM`
latches when the command RISES above its running minimum. That was designed against *CFG's*
failure (re-opening -> lift-then-drop). The BASELINE does the opposite: measured post-grasp drift
(`.agent_tmp/drift_check.py`, 30 episodes x 3 runs) is **median -1.40mm, 63% DECREASING vs 13%
increasing** — it keeps squeezing. A rise-triggered latch almost never fires, which is exactly why
all three eps values (1/2/3mm) give indistinguishable results. **LESSON: a mechanism debugged against one policy's
failure mode must have that failure mode re-verified on the policy it is being applied to.** Same
class of error as assuming the contact stop would transfer.

**The bigger correction — the contact stop never adapted, it just gripped WIDER.**
Its at-grasp width is ~32.1mm vs baseline ~30.3mm (**+1.8mm**) and its slope is 0.05 mm/mm, i.e.
NO width adaptation. So the entire -24% sustained-stress gain is a **LEVEL effect**, not a
stopping-the-squeeze effect and not adaptation. This matters because a level shift needs no sensor
at all: it is reproducible by simply commanding a constant offset. It also independently reproduces
the alzey-vs-lulkx story (alzey commands ~2.8mm wider and is the gentlest policy on the real robot).

**ADOPTED:** `GM_WIDTH_OFFSET_MM` in `dppo/eval_agent.py` — constant absolute offset on the
commanded width, converted with the DELTA scale factor (no affine term) and guarded by the same
round-trip assertion that caught B10. Testing +2/+3mm (jobs 1738755/6). If ~-20% sustained stress
lands at ~0.80 success, this is the deployable gentleness mechanism: **no sensor, no trigger logic,
one number**, and it is directly testable on the real rig against the over-squeeze the user reported.

**Open question this raises for the paper:** if a constant offset captures most of the gentleness
gain, then width ADAPTATION is worth only the residual — the part a constant cannot cover, which is
exactly the wide-size-range case (§4c). That is an honest and much simpler story than the one we
started with, and it is falsifiable by the offset sweep.

**B15 (2026-08-27, LOUD — cost 2 min).** Relaunching a lulkx eval without
`GM_EXTRA_OVERRIDES="action_dim=7 env_name=<v33b_shift9>"` builds a 10-d action model against a
7-d checkpoint: `size mismatch ... mlp_mean.layers.2.weight [28] vs [40]` (28 = 7x4 horizon,
40 = 10x4). **Failed instantly and unmistakably**, which is the contrast worth recording: the
same class of omission on NORM (B1) was SILENT and cost a whole sweep. A shape check is a free
guard; a scale check is not. When cloning a launch, copy the FULL override set — the sbatch echoes
`extra=...` on every run, so diff that line against the run you are replicating.
Second gotcha: `GM_EXTRA_OVERRIDES` contains a SPACE, so it cannot ride inside
`--export=ALL,VAR=...` (sbatch splits on commas and the shell on spaces) — export it into the
environment and use a bare `--export=ALL`. Same family as the arms.txt/commas issue.

### 2026-08-27 — GRIP LEVEL EXPLAINS ~95% OF GENTLENESS. Adaptation explains ~none of it.

Sustained stress regressed on AT-GRASP WIDTH, fitted within protocol groups:

| group | n | Pa per mm | r | R2 |
|---|---|---|---|---|
| canonical 200ep (baseline, contact x3, freeze x2) | 6 | **-3468** | -0.988 | **0.976** |
| probe 60ep (all mechanisms) | 7 | -1659 | -0.940 | 0.884 |
| **floor margin family alone (ONE mechanism, 4 levels)** | 4 | **-1714** | -0.974 | **0.949** |

**The mechanism does not matter; the resulting LEVEL does.** Contact stop, freeze, floor and
baseline all sit on one curve. In particular the FLOOR's gentleness is fully accounted for by the
mean width it produces (R2 0.949 within its own family) — **its width ADAPTATION (slope 0.48) buys
no measurable stress reduction beyond the level shift.** That is the strongest evidence yet that we
were optimising the wrong variable for four days.

**Two slopes differ 2x** (-3468 canonical over a 2.3mm span vs -1714 probe/floor over 8mm) =>
diminishing returns: the first mm of extra width is worth more than the eighth. Do not extrapolate
the canonical slope past ~3mm.

**PRE-REGISTERED PREDICTION for the running offset arms (1738816/7), recorded BEFORE they land:**
+2mm -> sustained 20,700-24,200 (-14% to -26%); +3mm -> 17,700-22,500 (-20% to -37%). Success stays
~0.80 IF level is the only thing changing (contact 1.5N holds 0.810 at +1.5mm wider). A large
success DROP at +2/3mm would falsify "level is sufficient" and mean the contact stop's timing
matters after all.

**THE DECISIVE EXPERIMENT, and it is already running.** floor m8 reaches width 33.66 / stress 21840
/ succ 0.750 (probe). A +3mm constant offset lands at ~33.6 — THE SAME MEAN WIDTH with NO adaptation
(slope ~0.17 vs the floor's 0.48). Canonical arms for both are in flight (1734915 floor m8,
1738817 offset+3). **If they match on stress AND success, width adaptation contributes nothing at
mushroom's size range and the shippable answer is one constant.** If the floor wins, adaptation
earns its place. Either outcome is a clean, publishable answer to the question the campaign opened
with — and it is a MATCHED-MEAN comparison, which nothing we ran before this was.

**B16 (2026-08-27, LOUD, caught by the degenerate watchdog in 15 min).** A UNIFORM constant offset
on the commanded width (`GM_WIDTH_OFFSET_MM`) gives **0/20 success** at +2mm and +3mm. NOT a units
bug — the dump confirms the magnitude was applied exactly (traj max 80.00 -> 82.00 at +2mm,
-> 83.00 at +3mm). The failure is **CLOSED-LOOP FEEDBACK**: the policy observes `gripper_width` in
its own proprio, so shifting EVERY command from t=0 — including the ~80mm open-gripper approach —
puts the observation out of distribution before the grasp begins. The gripper then never closes
(at-grasp 40.4mm / 46.4mm vs baseline 30.6; episode-end width 78.3/83.0 vs baseline 27.6).

**LESSON (generalises to the real robot): you cannot open-loop shift a channel the policy OBSERVES.**
This is the same trap as the earlier `--gripper-offset-m` feedback bug. The contact stop escaped it
only because it clamps LATE (at contact), leaving the approach in-distribution.

**FIX — a CONSTANT FLOOR, not a shift:** `GM_WIDTH_FLOOR_CONST_MM`, `w_cmd = max(w_policy, W_min)`.
The approach (~80mm) is far above W_min so the clamp does not bind until the policy closes past it;
the trajectory stays in-distribution until the grasp itself. Converted with the FULL AFFINE map
(absolute width, not a delta — the B10 distinction) and round-trip checked against the dump's
inverse so the two cannot drift.

**This also gives the campaign its cleanest experiment.** The constant floor is the exact CONTROL
for the vision floor: identical clamp, constant vs per-object PREDICTED level. Matching the mean
at-grasp width and comparing success + stress isolates ADAPTATION from LEVEL — the confound in
every comparison we have run so far. The matching constant is solved on the baseline's real
at-grasp distribution (clamping is nonlinear, so it is not simply the target mean).

**Conversion validated against B10's ground truth (2026-08-27).** The new absolute-width affine map
gives 32.84mm -> -0.3953 and 34.62mm -> -0.3426 (0.0296 norm/mm). Extrapolated to B10's documented
case, 34.32mm -> **-0.3515**, against the -0.351 B10 records as correct. An EXACT match from an
independent route. Note the sign: absolute widths in this range are NEGATIVE in normalized space,
which is exactly why B10's delta-scale error (+1.016) was so destructive and why a plain range
check could not catch it. Prefer validating a conversion against a known-correct external value
over a self-consistency round-trip — the round-trip cannot detect a wrong target SPACE (B16).

### 2026-08-27 — LEARN THE LEVEL (user steer: no hardcoded hacks)

User: *"a constant level per object sounds like a hack, if it can be learned from point cloud then
it may sound better... any hardcoded hacks will not receive welcome from reviewers."* Right, and it
identifies a real handicap in the campaign: **every level head we have evaluated was fitted
POST-HOC on a FROZEN encoder (corr 0.75) while a supervised encoder reaches 0.927.** We have been
judging "can the level be predicted?" using the worst version of the predictor we could have built.

Launched two arms on lulkx's exact recipe (paired reg 0.5, abs 7d, alzey obs; 600 epochs, ~3h):
- **A `wsup_aux` (1739398)** — `aux_grasp_width_weight=1.0`, level supervision in the objective.
- **B `wsup_feed` (1739399)** — A + `feed_width_pred=true`: the width head's DETACHED prediction is
  concatenated onto the denoiser conditioning. **No clamp, no constant** — the policy learns to use
  a predicted level. Training-only heads, so ZERO deployment cost and no new sensor.

The constant floor is retained as the ABLATION ("why not one number?"), which is the control a
reviewer demands, not the deliverable. Note `PairedRegDiffusionModel` already inherits
`AuxDiffusionModel`, so this needed no new code — only overrides. `+train_dataset.aux_grasp_width`
carries its `+val_dataset` twin (a missing twin silently validates on a different feature set).

**Monitoring lesson (2026-08-27).** My arming check for the width-supervision runs grepped stdout
for `loss_grasp_width` and fired "supervision INERT" on BOTH runs — a FALSE ALARM. `_aux_log` is
consumed by `wandb.log` only (`train_diffusion_agent.py:88-90`) and is never printed. The runs were
in fact correctly configured. **Verify a knob against the RESOLVED HYDRA CONFIG (or a parameter
count), not against log text you have not confirmed is emitted.** Positive confirmation here:
tebvy/ixjgp both show `aux_grasp_width_weight: 1.0` + train AND val `aux_grasp_width: true`, ixjgp
adds `feed_width_pred: true`, and the param counts differ by **exactly 1024** = 1 extra input dim x
the 1024 first hidden layer. A false "inert" alarm is cheap; the opposite error (a silent inert
knob reported as a result) is the B1 failure and is not.

**Supervision confirmed active by THREE independent routes (2026-08-27)**, after the false-alarm
above — worth recording as the pattern to reuse when a knob's own log line is unavailable:
1. **resolved hydra config** — `aux_grasp_width_weight: 1.0`, train AND val `aux_grasp_width: true`
2. **parameter count** — ixjgp exceeds tebvy by exactly 1024 (= 1 `feed_width_pred` input dim x the
   1024 first hidden layer)
3. **loss differential vs the un-supervised twin** — lulkx (no aux) 0.7304/0.2690/0.1810/**0.1577**
   vs tebvy 0.7800/0.3076/0.2201/**0.1964**: consistently ~+0.039 at epoch 4, which IS the width
   MSE term. An inert override would reproduce lulkx's curve exactly.
Route 3 is the strongest because it observes the knob's EFFECT rather than its declaration.

**Planned verification at 600 epochs:** measure corr(predicted episode-min width, true) on val. The
post-hoc frozen-encoder head reached **0.75**; a supervised encoder should approach the **0.927**
ceiling measured by the predictability analysis. That number decides whether the learned floor has
been handicapped all along, which is the question the user's "don't hardcode it" steer raises.

### 2026-08-27 — Floor frontier at 40 geometries + protocol-matching lesson

| arm | success | SUSTAINED | vs baseline | mean at-grasp |
|---|---|---|---|---|
| baseline | 0.905 | 29,734 | — | 30.47mm (sd 4.50) |
| **floor m8** | **0.760** | **22,804** | **-23%** | 34.04mm |
| floor m6 | 0.705 | 19,700 | -34% | 35.46mm |
| floor m4 | 0.575 | 18,047 | -39% | 36.96mm |

m8 is the shippable operating point; earlier reporting leaned on m4 (-0.33 success) and understated
the mechanism.

**LESSON — a matched-mean comparison must also match the PROTOCOL.** My first constant-floor
controls ran the eval cfg default (**5 distinct geometries**) while the adaptive floor arms passed
`n_episodes=200 scene_group_size=1` (**40 geometries**), and the constants were solved on
PROBE-protocol (60ep) means. Both errors point the same way as B1: the run completes and reports a
believable number that answers a different question. Caught by diffing the resolved `extra=` line
between the two launches — **the sbatch echoes every override, so diff that line against the run
you are matching BEFORE trusting a comparison.** Relaunched at 40 geometries with constants solved
on the 40-geometry baseline distribution (33.34/35.15/36.82 -> means 34.04/35.46/36.96, exact).

### 2026-08-27 — [CORRECTED BELOW] THE LEVEL HEAD WAS HANDICAPPED (corr 0.667 -> 0.850 at latch)

Width-head correlation on the VAL split, one point per EPISODE (the label is per-episode; per-step
rows are not independent):

| head | corr | 95% CI |
|---|---|---|
| post-hoc fit on a FROZEN encoder (every floor result to date) | **0.75** | — |
| **tebvy — supervised IN the objective, epoch 100/600** | **0.933** | [0.892, 0.959] |
| ixjgp — same + `feed_width_pred`, epoch 100/600 | 0.897 | [0.837, 0.936] |

**At one sixth of training the supervised head already reaches the 0.927 predictability ceiling.**
Say "reaches", not "exceeds": 0.933 vs 0.927 is well within the CI and the two were measured under
different protocols. **Every floor arm in this campaign clamped on a 0.75 head when 0.93 was
available.** The information was in the cloud; we never trained the encoder to retain it. This
directly answers the user's steer (*"if it can be learned from point cloud that may sound better"*)
— it can, and cheaply: aux heads are training-only, so ZERO deployment cost and no new sensor.

**Do NOT over-claim.** (a) Prediction quality is not policy quality — the head knowing the level
says nothing yet about the policy ACTING on it; ixjgp (prediction fed to the denoiser, no clamp) is
the arm that tests that. (b) In-distribution only: earlier analysis had cloud->size falling to
0.44-0.59 on UNSEEN shapes, which is exactly the regime the multi-category generalist lives in.
(c) n=65 val episodes.

**Endgame evals once 600 lands:** (1) ixjgp plain = the no-clamp learned design; (2) tebvy + floor
at the m6/m8 margins = the floor on a head that can actually predict; (3) both against the
matched-mean CONSTANT floors already running. That is the full adaptation-vs-level answer with the
learned side finally at full strength.

### 2026-08-27 — ⚠ CORRECTION: the 0.933 width-head figure was PROPRIO LEAKAGE

corr(pred, true grasp width) BY EPISODE PHASE, supervised head (tebvy@100), 65 val episodes:

| phase | corr | mean abs err | what is happening |
|---|---|---|---|
| **0.0 (latch moment)** | **0.850** | 0.079 | genuine cloud-based prediction |
| 0.2 | 0.827 | 0.085 | still genuine |
| 0.4 | 0.621 | 0.126 | **occlusion dip** — gripper descending over the object |
| 0.6 | 0.565 | 0.118 | occlusion dip |
| 0.8 | **0.998** | 0.010 | **LEAKAGE — not prediction** |
| 1.0 | **0.998** | 0.010 | **LEAKAGE** |

**The late-phase 0.998 is the head reading the label off its own input.** The conditioning feature
is [pointnet_feat (+) state] and `state` includes the CURRENT GRIPPER WIDTH; by phase 0.8 the
gripper is already AT the episode's minimum width, which IS the label. This is precisely why
`width_head_blind` exists in this codebase.

**RETRACTED:** "corr 0.933, reaches the 0.927 ceiling" (reported earlier today). That figure was a
MEDIAN ACROSS PHASES and therefore contaminated by leaked late samples.

**The correct, deployment-relevant number is the correlation AT THE LATCH MOMENT.** The floor arms
use the legacy latch = the episode's first `act()` (t=0), where every episode's gripper sits at the
same open width, so proprio carries NO per-episode information and the prediction is genuinely from
the cloud:
- frozen post-hoc head at t=0: **0.667**
- supervised head at t=0: **0.850**
A real improvement that should make the floor clamp better, but NOT the ceiling.

**LESSON — a per-episode label that also appears in the observation stream leaks.** Any correlation
measured after the policy has ACTED on the quantity being predicted is contaminated. Measure a
predictor only at (or before) the moment its output is USED. Corollary: the two contradictory latch
comments in `eval_agent.py` ("latch at t=0, object unoccluded" vs "t=0 is the head's WORST moment,
object far away") are now settled by measurement — **t=0 is the BEST honest moment (0.850)** and the
mid-episode occlusion dip (0.57-0.62) is real, matching upstream a15f88c's occlusion audit.

**ARM C `wsup_blind` (1742221) — remove the leakage from TRAINING, not just from measurement.**
The leakage above is not only a measurement artifact: the aux head is supervised at EVERY step
against the episode-MIN width, and in the back half of an episode the gripper is already AT that
width. **The head can satisfy its loss by copying proprio instead of learning size from the cloud**
— a shortcut that dilutes exactly the encoder incentive the auxiliary objective exists to create.
That is a plausible reason the honest t=0 correlation is 0.850 rather than near the 0.927 ceiling.

New flag `aux_width_blind` (`pointnet_diffusion.py`) zeroes the gripper-width entry of each
per-step proprio slice before the AUX width head — reusing the mask `width_head_blind` already
applies to the trajectory head, but as a SEPARATE flag so the trajectory head is untouched.
Default False, so every existing run stays bit-identical. Prints its resolved state at build
(an earlier "blind" run in this repo was silently SIGHTED — same trap).

Three arms now: **A tebvy** (sighted aux), **B ijxgp** (sighted aux + feed_width_pred),
**C wsup_blind** (blind aux). Compare on the honest t=0 correlation, then on closed-loop success
and stress. If C >> A at t=0, the shortcut was real and blinding is the right default for any
auxiliary head whose label also appears in the observation stream.

### 2026-08-27 — CONSTANT FLOOR (canonical protocol): -41% sustained stress at 0.745 success

| arm (canonical, 200 eps) | success | ever | peak | SUSTAINED | vs baseline |
|---|---|---|---|---|---|
| baseline lulkx | 0.820 | 0.865 | 53065 | 28060 | — |
| contact stop 1.5N (NOT deployable) | 0.810 | 0.855 | 51053 | 21241 | -24% |
| **constant floor 32.84mm** | **0.745** | 0.785 | 49052 | **16439** | **-41%** |

**A single constant beats the contact stop on stress (-41% vs -24%) with no sensor at all**, at
0.745 success — above the user's 0.7 bar. This is the simplest possible mechanism: one clamp,
`w_cmd = max(w_policy, 32.84mm)`, deployable on the real rig today.

It also roughly validates the pre-registered width->stress prediction: the fit said -3468 Pa/mm and
the clamp moves mean at-grasp ~+2.9mm, predicting ~18.0k; measured 16.4k. The LEVEL model of
gentleness holds quantitatively.

**This is exactly the "hack" the user warned about**, and that is the point: it sets the bar the
LEARNED version (tebvy/ixjgp/neoca + floor, or the no-clamp feed_width_pred design) must clear to
justify its complexity. If a learned per-object level cannot beat one number, the honest paper
reports the constant as the method and the learned level as a negative result.

⚠ Protocol note: this is the CANONICAL protocol (5 geometries). The adaptive-vs-constant
MATCHED-MEAN comparison is the 40-geometry set (1739549/50/51 vs 1734914/5), still running. Do not
cross-compare the two protocols — the baselines differ (0.820 vs 0.905).

### 2026-08-27 — Generalist eval was BLOCKED; three fixes to make the cond/uncond ablation possible

The 3-object generalist trained fine (uncond `xaqnb` 350 epochs, val 0.0012) but **could not be
evaluated**. Three separate gaps, each of which would have produced a wrong answer rather than an
error:

1. **`category_embed` was never wired into the eval path.** The dataset supplies it in TRAINING
   (`StitchedSequencePointCloudCategoryDataset`), but `eval_agent` never built it, so a
   category-conditioned checkpoint had no way to run closed-loop. Added `GM_CATEGORY=<object>` ->
   `category_embedding.embed()` -> broadcast over the batch (identity is static per episode and
   each eval run is ONE object, so no per-step cost). `category_embedding.py` is genesis-free
   precisely so the harness can import it — the design anticipated this; the harness side was
   simply never written. **It HARD-FAILS if the network wants an embedding and GM_CATEGORY is
   unset** — otherwise the conditioned policy would run on a silently absent/garbage conditioning.
2. **`dppo_eval.sbatch` hardcoded `REPO`** to the main checkout, so evaluating the worktree's
   checkpoint would have `cd`'d to the MAIN repo and run the main `eval_agent` — i.e. WITHOUT the
   category wiring, looking like a normal run. Now `REPO=${GM_REPO:-<main>}` (worktree copy).
3. **No 7d raspberry experiment existed.** Raspberry's experiment declares
   `action: abs_pose_abs_gripper` (10d rot6d) while the generalist is 7d; the eval sim server
   derives its ActionPipeline FROM THE EXPERIMENT, so a 7d policy evaluated with it is a
   mismatch. Created `single_lift_raspberry_soft_abs_action_armfocus_7d_realws.yaml` — diff vs the
   10d original is EXACTLY one line (the action config).
   Reassuring: the 3-object dataset BUILD already converted raspberry 10d->7d
   (`--derive-action`/`--derive-source-action`), so the TRAINING data was consistent all along;
   only the eval-side config was missing.

Smoke-tested the embedding first (dim 21; one-hot mushroom 0 / raspberry 1 / tofu 10; registry
sizes 33/30/15mm; pairwise L2 1.55-1.65 so categories are distinct) rather than discovering a
KeyError two hours into a GPU eval.

Uncond arm launched on all three objects (1743043/4/5, 60-ep probes). Cond arm follows when its
training finishes.

### WHERE RUN ARTIFACTS LIVE (2026-08-27) — read this before hunting for a result

⚠ **The 3-object generalist lives in the WORKTREE `../gm_generalist`, NOT the main checkout.**
That split is the single easiest thing to trip over: the main repo also holds an OLDER 2-object
generalist (`single_lift_generalist_mushroom_tofu/{waslz,xktwc}`), so opening the main repo finds a
plausible-looking but SUPERSEDED run.

| artifact | path |
|---|---|
| mushroom trainings (tebvy/ixjgp/neoca) | `logs/dppo/dppo-pretrain/single_lift_mushroom_simreal_realws_noos_cmd_v33b_shift9/<id>/` |
| 3-obj generalist (xaqnb uncond, lbjbv cond) | **`../gm_generalist/`**`logs/dppo/dppo-pretrain/single_lift_generalist_3obj/<id>/` |
| per-run record | `<run>/.hydra/config.yaml` (AUTHORITATIVE resolved config), `<run>/config/` (env snapshot), `<run>/EXPERIMENT.md`, `<run>/checkpoint/state_N.pt` |
| evals | `<run>/eval/<EVAL_SUBDIR>/` -> `summary.json`, `episodes.csv` (per-episode success/stress + the DR params ACTUALLY applied), `render/batchNN_envM.mp4` (one clip per episode) |
| width dumps (input to every adaptation metric) | `.agent_tmp/<GM_WIDTH_DUMP tag>_widthcmd_b*.npz` — per-step commanded width (mm) + EE-z (m) |
| slurm | `logs/slumr_logs/<jobid>.{out,err}` + `<jobid>_<name>_simserver.log` |
| analysis tools | `.agent_tmp/`: `decompose_width.py` (slope+CI), `level_vs_stress.py`, `phase_vs_width.py`, `eval_width_head.py`, `pick_const_slope.py` |

**`EXPERIMENT.md` is written at LAUNCH; its Final summary is filled at the END.** For a run still in
flight, `.hydra/config.yaml` is the only trustworthy statement of what is actually running — several
of today's bugs (B15, the protocol mismatch, the category-embed wiring) were caught by reading it
rather than the logs.

**Upstream ef19428 — ν was 0.33 for EVERY object in every collection to date** (`build_grasp_fem`
fell back to MetricConfig's copper default; materials declare 0.30-0.42). `--grasp-nu auto` added,
DEFAULTED to 0.33 so existing runs stay reproducible. **Impact on the width campaign: small for
mushroom** (declared 0.35 vs 0.33 used) and our width-vs-size numbers come from RECORDED widths
regardless. Larger unknown for tofu/raspberry — a third caveat on the 3-object generalist data,
alongside raspberry's over-squeeze (fixed 5mm = 34% of a 15mm object, upstream a15f88c) and the
procedural-material caveat on the 4 new objects. μ = 0.7 is also one global constant, not per-object.

Also landed: `docs/grasp_synthesis_model.md` (written for paper writing, every claim checked against
the implementation, with **"DO NOT CLAIM" markers** where a natural claim would overstate the code)
and `docs/paper/{preview_icra.tex,refs.bib,related_work.md}`. Use these when writing up the width
results — the DO-NOT-CLAIM discipline is exactly what today's retractions (the 0.933 leakage, the
"baseline is 43% adaptive" 12-geometry artifact) argue for.

### 2026-08-27 — ⭐ MATCHED-MEAN VERDICT: the adaptive floor does NOT beat a constant

40 geometries, 200 eps, lulkx@600. Each pair holds the MEAN at-grasp width fixed; the ONLY
difference is whether the clamp level is PREDICTED per object (slope 0.48) or CONSTANT (slope ~0.17).

| pair | ADAPT succ | CONST succ | d succ | ADAPT sust | CONST sust | d stress |
|---|---|---|---|---|---|---|
| m8 <-> cfloor 33.34 | 0.760 | **0.775** | **-0.015** | 22804 | **20590** | **+10.8%** |
| m6 <-> cfloor 35.15 | **0.705** | 0.630 | +0.075 | 19700 | 19157 | +2.8% |
| m4 <-> cfloor 36.82 | **0.575** | 0.520 | +0.055 | 18047 | 16221 | +11.3% |

**The success difference is INCONSISTENT IN SIGN (-0.015..+0.075) while the constant is
CONSISTENTLY lower on sustained stress (2.8-11.3% better in all three).** At m8 — the shippable
operating point — the constant wins on BOTH.

⚠ **RETRACTED, same day, one pair later:** after seeing ONLY the m6 pair I told the user
"adaptation is not a gentleness mechanism, it is a SUCCESS mechanism." The m8 pair reverses the
sign. **Do not draw a conclusion from the first pair of a designed comparison** — the whole point of
running three operating points is that one pair cannot distinguish an effect from noise. This is the
same failure mode as the 12-geometry "baseline is 43% adaptive" artifact: concluding before the
designed sample was complete.

**THE CAVEAT THAT KEEPS THIS OPEN:** the adaptive arm clamps on the POST-HOC FROZEN head — corr
**0.667** at the latch moment. The supervised head reaches **0.850** and has NEVER been run in the
floor. So this is a negative result for the CURRENT PREDICTOR, not for adaptation in principle. The
decisive re-run is tebvy/neoca + floor at m6/m8 against these same constants.

**Honest status if the supervised head does NOT change this:** the shippable gentleness mechanism is
a CONSTANT floor (canonical: -41% sustained stress at 0.745 success; 40-geo m8-matched: 0.775 succ /
20590 sust vs baseline 0.905 / 29734), and learned per-object level adaptation is a NEGATIVE RESULT
at mushroom's ~19mm size range. It would remain necessary only ACROSS categories (15mm raspberry vs
65mm tomato), which one constant cannot serve — that is where the generalist work has to carry it.

### 2026-08-27 — NOT OOD, NOT DATA AMOUNT: the generalist fails to adapt where it has the MOST data

User hypothesis: the flat width fit might be an artifact of evaluating on object sizes the training
data never covered (some plotted points do sit near the demo slope). Tested directly by comparing
eval sizes against the training collections' own `scene_scale`, then re-fitting INSIDE the training
range:

| | train size range (successful demos, geometries) | eval range | % eval OOD | slope overall | **slope IN-RANGE** |
|---|---|---|---|---|---|
| mushroom | 27.1-49.2mm (653 demos, **87 geometries**) | 33.0-43.7mm | **0%** | +0.12 | **+0.12** |
| tofu | 24.2-41.7mm (614 demos, **76 geometries**) | 24.0-35.7mm | 8% | +0.34 | **+0.37** |
| raspberry | 15.1-19.9mm (**284 demos** across 3 dirs) | (eval pending) | — | — | — |

**HYPOTHESIS REJECTED.** Mushroom has ZERO out-of-range eval episodes — the eval range is strictly
INSIDE the training range — and restricting the fit changes the slope not at all. Tofu moves within
noise. **The policy does not adapt precisely where it has the most data.**

**Data amount is not the explanation either:** 653 mushroom demos over 87 DISTINCT geometries
spanning a WIDER range than we evaluate. And the demonstrator's own slope in that very data is
**+1.09** (tofu +1.06, raspberry +1.04) — the signal is unambiguously present in what the policy
trained on. Combined with the earlier predictability result (cloud->size 0.927 with a supervised
encoder), all three "the data is at fault" explanations are now closed:
NOT coverage, NOT amount, NOT perceivability.

The points near the demo slope are per-geometry means scattering around a FLAT line, not an
adapting subpopulation: mushroom R2 = 0.01, i.e. object size explains 1% of width variance. The
scatter is large (sd 4.4mm) but UNCORRELATED with size — the policy varies its grip a lot, just not
with the object.

(Count correction: raspberry is 284 successful demos across 26-08-27-{abp,yrp,yxu}, not the 61 my
first pass reported from a single directory.)

### 2026-08-27 — THE SHARPEST STATEMENT YET: same variance as the demos, none of the information

Per-episode grasp width, demonstrator (`dr_params.csv` `width_mm` vs `scene_scale`) vs the
3-object generalist, measured the same way for both. Figures:
`docs/figures/demo_vs_policy_width.png` (3 panels, demo + policy scatter per object) and
`docs/figures/generalist_width_vs_size.png`.

| object | | slope | sd around fit | total sd | size explains |
|---|---|---|---|---|---|
| mushroom | demo | 1.08 | 4.33 | 8.01 | **71%** |
| | policy | 0.11 | 7.90 | 7.91 | **0%** |
| tofu | demo | 1.12 | 4.68 | 7.56 | **62%** |
| | policy | 0.31 | 7.72 | 7.80 | **2%** |
| raspberry | demo | 0.96 | 2.73 | 3.44 | **37%** |
| | policy | -0.31 | 6.89 | 6.91 | **0%** |

**The policy reproduces the demonstrations' SPREAD while discarding their STRUCTURE.** On mushroom
the total sd is nearly identical (7.91 vs 8.01mm) yet size explains 71% of the demo's variance and
**0%** of the policy's. On raspberry the policy's spread (6.91mm) is **2x the demonstrator's**
(3.44mm) — it varies more than was ever demonstrated, on the most fragile object.

**This is a better problem statement than "the policy does not adapt".** It says the deficit is not
missing variance or an unlearnable behaviour: the variance is already there, it just carries no
information. The task is to CONVERT noise into size-tracking, not to add a new capability.

**"Near-constant width" was WRONG and is retracted** (user caught it from the figure): mushroom
at-grasp is 28.4 +- 7.9mm, ranging 7-50mm. Two distinct claims were conflated — "low variance"
(FALSE) and "variance uncorrelated with size" (TRUE). Only the second is supported. Use
"varies widely but not with size", never "near-constant".

**Corollary for the fix:** reducing sampling variance alone would yield a tight CONSTANT, not an
adaptive grasper — unless the CONDITIONAL MEAN already tracks size, in which case averaging the
noise away would expose it. That is exactly what `.agent_tmp/policy_stochasticity.py` (job 1748993)
measures: K samples of the SAME observation, within-obs sd vs across-obs sd.

### 2026-08-27 — IDEA (documented, NOT yet tested): tighten the demo target by ALIGN filtering

User's suggestion, from reading `docs/figures/demo_vs_policy_width.png`: the demonstrator has width
outliers too, so filtering them might help. Quantified on existing data (`.agent_tmp/demo_outliers.py`,
`.agent_tmp/align_filter_sweep.py`) — the hypothesis holds, with one methodological correction.

**The width outliers are POORLY-ALIGNED grasps, not random noise:**

| object | outlier align vs inlier | corr(\|resid\|, align) | align explains of the leftover |
|---|---|---|---|
| mushroom | 0.788 vs 0.926 | **-0.59** | 38% |
| raspberry | 0.563 vs 0.879 | **-0.74** | **69%** |
| tofu | 0.867 vs 0.929 | +0.08 | 13% |

**⚠ FILTER ON `align`, NEVER ON THE WIDTH RESIDUAL.** Filtering on the width residual is filtering
on the TARGET VARIABLE — it biases the learned distribution toward whatever we decided the answer
should be and is not defensible to a reviewer. `align` is an INPUT-SIDE quality criterion and is
legitimate. Both remove roughly the same episodes; only one is honest.

**Operating points (align threshold -> data kept, residual sd, slope):**

| object | threshold | kept | residual sd | slope |
|---|---|---|---|---|
| mushroom | >= 0.90 | 85% (555) | 4.33 -> **2.78mm** | 1.08 (unchanged) |
| **raspberry** | >= 0.80 | 84% (238) | 2.73 -> **1.29mm** | 0.96 -> **1.14** (toward ideal) |
| tofu | >= 0.85 | 97% | 4.68 -> 4.56 (little) | 1.10 |

Raspberry gains most — a **53% tighter target for 16% of the data** — and it is our worst object
(success 0.583, the ONLY one whose SUSTAINED stress exceeds yield, 1.13x). Mushroom is a good trade
too, and note that set is ALREADY align-filtered once (`26-08-25-clq-alignfilt`), so this is a
SECOND pass. Tofu barely responds cheaply and its sweep is NON-MONOTONIC (0.90 -> 5.00, worse than
0.85's 4.56), which suggests its collections have differently-shaped align distributions — inspect
before filtering tofu.

**UNPROVEN, and the reason this is filed as an idea rather than a plan:** a tighter target is
plausibly easier to learn, but it is NOT established that it converts into size-tracking. The policy
currently gets **0%** of its width variance explained by size; if the bottleneck is CONDITIONING
rather than target noise, a cleaner target changes nothing. Cheapest test is to fold the align
filter in alongside the aux width supervision (dataset rebuild is minutes; the retrain shares a GPU
slot), so the two candidate fixes are evaluated together rather than serially.

### 2026-08-27 — IDEA (documented, not tested): aux/pseudo-condition on METRIC SIZE, not width

User's proposal. It is better supported than the width-aux we are currently training, on three
independent grounds:

1. **Literature precedent.** `docs/size_adaptation_literature.md` already contains it — line 126
   ("regress `priv_object_dr_params[0]` (scale) from the observation"), lines 137-139 ("an
   adaptation module regressing the privileged size vector; feed its output to the student.
   Precedent #3/#4. **Also covers item 18's aux-head idea, since the auxiliary target is the
   privileged scale rather than an invented one**"). And lines 163-165 warn that "final closed
   gripper width as an auxiliary" was asserted by a search summary but is **NOT in the cited
   paper** — i.e. the WIDTH target (tebvy/ixjgp/neoca) has the WEAKER citation of the two.
2. **Predictability.** cloud->SIZE 0.927 vs cloud->WIDTH 0.771. Width conflates object size with
   the demonstrator's squeeze margin and with `align`; size is a pure perceptual quantity.
3. **Cross-category meaning.** For the generalist the target must be **METRIC size (mm)** =
   `scene_scale x registry nominal extent`, NOT scale: a scale of 1.2 is 40mm on a mushroom and
   18mm on a raspberry, so scale is not comparable across categories (the user made exactly this
   point earlier: *"if metric size can be used for prediction that will be the best"*).

**Cost: a dataset rebuild, NOT a re-collection.** The dppo npz carries only
`actions/point_cloud/rewards/states/terminals/traj_lengths` — no privileged params. But
`dr_params.csv` has `scene_scale` per episode and `dppo/add_category_embed.py` is an existing
template for grafting a per-episode label onto a built npz from per-source episode counts.

**AUX vs PSEUDO-CONDITION — the distinction that matters:** an auxiliary loss shapes the ENCODER but
does not force the policy to USE the estimate. That is exactly the gap we are sitting in: the width
head reaches corr 0.850 while the policy's own width stays **0%** size-explained. The
**pseudo-condition** variant (`feed_width_pred`-style: detached prediction concatenated onto the
denoiser conditioning) is the one that forces usage. `ixjgp` already tests that MECHANISM — swapping
the label from width to metric size is a one-line change once the label exists.

**B17 (2026-08-27) — TWO reference/normalization mistakes in one analysis. Both SILENT.**

**B17a — compared the policy against the RAW DEMO CSV instead of its TRAINING TARGET.**
⚠ **MY FIRST EXPLANATION OF THIS WAS ALSO WRONG and is retracted:** I claimed the gap was a
"gripper-offset compensation" implied by the `shift9` in the lineage names. **There is NO gripper
offset anywhere in this pipeline.** `shift9` is the ~9mm POINT-CLOUD x-bias correction
(`shift_demo_clouds.py --shift 0.009 0 0`, DEVLOG item 17) and it explicitly leaves
"proprio/actions/zero-padding untouched". I saw a name and a similar magnitude and connected them
without checking — the exact failure this bug class is about. The user caught it.

**THE REAL CAUSE (verified in code):** `dr_params.csv` `width_mm` is `g["x"][6]*1e3` = the CMA-ES
**PLANNED** grasp width. Execution then deliberately closes further: `--grasp-extra-close` (5.0mm
in these collections) + the FIRM phase's `FIRM_EXTRA_CLOSE_M` (2.5mm) when the grip reads weak. So
the recorded ACTION is the plan minus ~7.5mm, BY DESIGN:

| object | CMA-ES plan | plan - 7.5mm | actual train target | resid |
|---|---|---|---|---|
| mushroom | 40.0 | 32.5 | 31.8 | +0.7 |
| tofu | 42.2 | 34.7 | 32.9 | +1.8 |
| raspberry | 15.6 | 8.1 | 8.8 | -0.7 |

**Corollary — the raspberry over-squeeze, quantified:** a FIXED 7.5mm squeeze on a 15.6mm plan is
**48% compression**. That is precisely what upstream a15f88c's size-scaled
`--grasp-extra-close auto` fixes.

Reported to the user: "the policy squeezes 12-15mm
tighter than the demonstrator". **ACTUAL, against what it was trained on: -3.4mm (mushroom),
-5.6mm (tofu), and +4.7mm WIDER on raspberry.** The claim was ~3x overstated and reversed in sign
for raspberry.

| object | demo CSV | TRAIN TARGET | policy | true gap |
|---|---|---|---|---|
| mushroom | 40.0 | **31.8 +- 6.9** | 28.4 | **-3.4** |
| tofu | 42.2 | **32.9 +- 7.2** | 27.3 | **-5.6** |
| raspberry | 15.6 | **8.8 +- 3.0** | 13.5 | **+4.7 (WIDER)** |

**B17b — decoded three source datasets with a FOURTH dataset's normalization.** While checking
B17a I used the generalist's `action_min/max` to decode the per-object source npzs, giving tofu
"21.1mm" (true 32.9). Caught only because 21.1mm was implausible against a 42.2mm demo. **Every
dataset has its own normalization; a borrowed one is silently wrong, not an error.** Same family as
B1 (NORM default) and B10 (delta scale factor on an absolute value) — this repo's single most
recurrent bug class is *the right arithmetic applied to the wrong reference frame*.

**GUARD ADDED: `.agent_tmp/width_utils.py`.** `train_target_widths(env)` loads the normalization
sitting NEXT TO the npz it reads, so the pair cannot be mismatched; `demo_csv_widths()` carries a
docstring stating it is NOT the training target. `plot_demo_vs_policy.py` now imports it and plots
BOTH references — solid = demonstrator physical, dotted = TRAIN TARGET, with the measured offset
printed per panel.

**What the corrected figure shows that the wrong one hid:** the policy's flat line CROSSES the
training target — too WIDE for small objects, too TIGHT for large ones. That is the
constant-vs-proportional signature, and it was invisible while the reference curve sat 7-11mm too
high (everything looked uniformly "too tight").

**RULE: before quoting any policy-vs-demonstration gap, state WHICH reference — raw demo CSV or
training target — and decode with that dataset's OWN normalization.**

### 2026-08-27 — PRE-FLIGHT AUDIT of queued/planned work (user: "no more time to waste")

Two real bugs found in work that had NOT yet run, both silent:

**AUDIT-1 — `eval_width_head.py` (job 1750154, CANCELLED before it ran) would have re-reported the
LEAKED metric.** It medians the head's prediction ACROSS the episode; phases 0.8-1.0 leak (the
gripper is already AT the episode-min width, so the head reads the label off its own proprio:
corr 0.998, err 0.010). It would have printed ~0.93 again — the exact number retracted hours
earlier. **FIXED:** now reports the HONEST correlation at t=0 (the latch moment, where every
episode's gripper is equally open so proprio carries no per-episode signal), and prints the leaked
variant separately, explicitly labelled, so the two can never be confused again.

**AUDIT-2 — `launch_supervised_floor.sh` hardcoded the architecture flags.** For `neoca` (blind
aux) it would have omitted `aux_width_blind=true`, feeding the width head gripper-width input it
was TRAINED TO HAVE ZEROED. **FIXED:** derives flags from the run's OWN `.hydra/config.yaml`,
matching `eval_width_explain.sh`.

**GENERAL RULE this exposes — which flags are self-guarding and which are not:**

| flag | effect if omitted at eval |
|---|---|
| `feed_width_pred` | changes `input_dim` (+1) -> **shape mismatch -> LOUD** (safe) |
| `aux_grasp_width` | head missing from state_dict -> **LOUD** (safe) |
| **`aux_width_blind`** | only MASKS an input; no shape change -> **SILENT, wrong predictions** |

**A flag that changes TENSOR SHAPES protects itself. A flag that only changes BEHAVIOUR does not —
those must always be read from the run's own resolved config, never hardcoded at the call site.**
This is the same lesson as B15 (shape mismatch = loud) vs B1 (wrong NORM = silent), and it is now
the deciding criterion for whether a launch script may hardcode anything.

**Cleared in the same audit:** ixjgp/tebvy plain evals use the correct lineage NORM (each
checkpoint's own train env matches, and the sbatch's B1 pre-flight guard would refuse otherwise);
protocol is `n_episodes=200 scene_group_size=1` = 40 geometries, matching the baseline
(`slope_base` 0.905) they will be compared against; `gen3c_mushroom` points at an existing
checkpoint with the category wiring present in the WORKTREE eval_agent.

**AUDIT-3 (found by the job failing, NOT by my audit) — the generalist eval never passed
`category_embed_dim`.** `gen3c_mushroom` died with `size mismatch: [1024, 593] vs [1024, 572]` —
exactly the 21 embedding dims. I wired `GM_CATEGORY` into `eval_agent` (so the embedding is BUILT)
but never told the eval-side network builder to allocate the input for it.

**The audit gap:** I audited the jobs I had just submitted and did NOT re-audit `gen3c`, which was
queued BEFORE the audit practice started. **An audit must cover everything currently queued, not
just what was submitted since the last audit.**

Saved by the loud-failure property: `category_embed_dim` changes `input_dim`, so it is
self-guarding (see the flag table above). Had it been a behaviour-only flag it would have run and
produced a plausible wrong ablation.

**FIXED:** `.agent_tmp/eval_generalist.sh` now derives EVERY architecture flag
(`category_embed_dim`, `aux_grasp_width`, `feed_width_pred`, `aux_width_blind`) from the run's own
resolved config, and picks the per-object experiment. All three generalist launch paths
(`eval_width_explain.sh`, `launch_supervised_floor.sh`, `eval_generalist.sh`) now share this
derive-from-config discipline; no launch script hardcodes an arch flag any more.

### 2026-08-27 — VARIANCE REDUCTION IS NOT THE LEVER: the policy is DETERMINISTIC given its obs

`.agent_tmp/policy_stochasticity.py` (job 1752518), 12 samples of each of 24 real observations,
built on the EVAL inference path (`DiffusionEval`, not the training class — the first attempt died
in 37s because `PairedRegDiffusionModel.p_mean_var` has no `deterministic` kwarg):

- **WITHIN-observation width sd = 0.01mm** (max 0.08) -> the policy returns the SAME width for the
  same observation. **There is no sampling noise to reduce.** DDIM / temperature / sample-averaging
  would change NOTHING. This closes the user's hypothesis that reducing variance might expose a
  buried size signal.

⚠ **The reported ACROSS-observation sd (0.02mm) is INVALID — my phase choice was wrong.** I sampled
at phase 0.15 (mid-approach) where every episode's gripper is still OPEN and commanding the same
thing, so that number is the open-gripper command's spread, not the grasp width's. The determinism
result stands (it is a within-observation comparison, phase-independent); the across-observation
number must not be quoted.

**WHAT THE VALID HALF IMPLIES — a sharper problem statement.** The policy is deterministic given
its observation, yet its at-grasp width still varies **sd 7.9mm** across episodes. Therefore that
variance IS a function of the observation. It is **not noise — it is MIS-ATTRIBUTION**: a
deterministic observation->width map that keys on features which are not object size (size explains
0%). So the fix cannot be smoothing, averaging, or any variance-reduction trick; it must change
WHAT THE MAPPING ATTENDS TO. That is exactly what the four-arm ablation
(lulkx / tebvy / ixjgp / neoca) is testing, and it is now the ONLY live route to
"grasp width explainable by object size".

### 2026-08-27 — ⭐ THE HEAD KNOWS, THE POLICY IGNORES IT (feed_width_pred is a negative result)

Four-arm ablation, identical except the mechanism (seed 43, 600 epochs, paired-reg 0.5, same data).
PLAIN evals, no clamp, 40 distinct geometries.

**Policy behaviour — the target metric:**

| arm | slope | 95% CI | %demo | R2 | success | SUSTAINED |
|---|---|---|---|---|---|---|
| lulkx (no aux) | 0.17 | [0.00, 0.33] | 15% | 0.09 | 0.905 | 29734 |
| **ixjgp** (aux + `feed_width_pred`) | **0.20** | [0.06, 0.33] | 18% | 0.17 | **0.795** | 28668 |

**Head quality — honest t=0 (latch-moment) correlation:**

| head | corr | 95% CI |
|---|---|---|
| frozen post-hoc | 0.667 | — |
| tebvy (aux) | 0.755 | [0.626, 0.844] |
| **ixjgp** (aux + fed) | **0.810** | [0.705, 0.880] |

**THE FINDING: the head predicts grasp width from the cloud at r=0.81, that prediction is
concatenated onto the denoiser's conditioning, and the POLICY STILL DOES NOT USE IT** (slope 0.20 vs
baseline 0.17, CIs overlapping; R2 0.17). Not a perception problem, not an information-availability
problem, not a plumbing problem. **The diffusion policy declines to condition its width on an
explicit, accurate width estimate placed directly in its input.**

Cost: **-0.11 success** (0.905 -> 0.795) for +0.03 slope. The one positive: ixjgp's slope CI
excludes zero where the baseline's includes it, and R2 doubled — a real but small effect.

**SECONDARY FINDING, and it makes `neoca` the key remaining arm: tebvy's head got WORSE with
training — 0.850 @epoch100 -> 0.755 @600.** Consistent with the proprio-shortcut hypothesis: as
training proceeds the head increasingly satisfies its loss by COPYING the gripper width (readable at
late phases) rather than learning size from the cloud. `neoca` (aux_width_blind) is the only arm
that cannot take that shortcut, so it is the cleanest test of whether a genuinely cloud-derived
size estimate propagates any further into behaviour.

### 2026-08-27 — ⭐⭐ AUXILIARY WIDTH SUPERVISION IS CLOSED. It makes a WIDER CONSTANT, not an adapter.

Complete four-arm ablation. Identical except the mechanism (seed 43, 600 epochs, paired-reg 0.5,
same dataset). PLAIN evals, no clamp, 40 distinct geometries.

| arm | intercept | slope | 95% CI | R2 | success | ever | SUSTAINED |
|---|---|---|---|---|---|---|---|
| lulkx (no aux) | 23.6 | **0.17** | [0.00, 0.33] | 0.09 | **0.905** | 0.940 | 29734 |
| tebvy (aux only) | 29.7 | 0.05 | [-0.08, 0.19] | 0.01 | 0.755 | 0.760 | 26490 |
| ixjgp (aux + `feed_width_pred`) | 23.4 | 0.20 | [0.06, 0.33] | 0.17 | 0.795 | 0.850 | 28668 |
| neoca (blind aux) | 31.3 | 0.04 | [-0.11, 0.18] | 0.01 | **0.605** | 0.785 | 26991 |

**EVERY aux arm made adaptation WORSE.** And the mechanism is legible in the INTERCEPT: aux arms sit
at 29.7 / 31.3mm vs the baseline's 23.6 while their slopes flatten to ~0.05. **Auxiliary width
supervision does not teach sizing — it converts the policy into a WIDER CONSTANT GRIPPER.** Lower
stress, lower success, less adaptation: the LEVEL effect again, arriving as a side effect of an
objective aimed at something else.

**Blinding did NOT rescue it** — `neoca` was the best remaining hypothesis (remove the proprio
shortcut, force genuinely cloud-derived size) and produced the WORST success (0.605) with the
flattest slope (0.04). **The shortcut was not the problem.**

**`feed_width_pred` only repairs the damage the aux loss causes** (0.05 -> 0.20), landing back at the
untouched baseline (0.17) at a cost of -0.11 success.

**CONCLUSION: the head predicts grasp width from the cloud at r=0.81 (honest, latch-time), that
estimate can be concatenated onto the denoiser's conditioning, and NONE of it changes the policy's
behaviour. Auxiliary supervision — sighted, blind, or fed forward — is CLOSED as a route to
size-explainable width.**

**WHAT REMAINS UNTESTED (and why it is the next lever):** the diffusion loss on the width dim is
dominated by the long OPEN-GRIPPER APPROACH; the handful of frames where width actually encodes the
grasp decision contribute almost nothing to the gradient. No amount of extra INPUT fixes a LOSS that
barely rewards using it. `pointcloud_dataset.py` already implements **`width_window_weight`**
(per-chunk up-weighting of the width dim near closure) and it has NEVER been run. That is a
LOSS-shaping fix rather than a conditioning fix, and it is the only untried lever that addresses the
diagnosis directly.

### 2026-08-27 — WHY width does not adapt: the width pathway is only ~15% VISUALLY SENSITIVE

User asked whether the failure is (a) size unreadable from the cloud at grasp time (occlusion) or
(b) a vision-proprioception SHORTCUT (cf. "When would Vision-Proprioception Policies Fail in
Robotic Manipulation?"). Tested by INTERVENTION, not inference.

**First, the metric was validated** (`docs/figures/atgrasp_metric_validation.png`): per-episode
traces with the extraction window shaded. The extracted value sits at the closure plateau, gripper
DOWN, BEFORE the lift — the metric is honest. Caveat visible in env 4: some episodes keep closing
AFTER the grasp (34mm -> 12mm during hold), which at-grasp does not capture — that is what the
SUSTAINED stress metric is for.

**CLOUD-SWAP ABLATION (`visual_ablation.py`) — swap the whole object's cloud, keep proprio:**

| phase | lulkx (within-category) | generalist (CROSS-category) |
|---|---|---|
| 0.0-0.2 (approach) | **0.06-0.12mm** | 0.15mm |
| 0.4 | 0.23mm | 0.29mm |
| **0.5 (closure)** | **1.68mm** (max 6.3) | **2.18mm** (max 11.7) |
| 0.6 | 1.42mm | 1.35mm |

Correct sizing would need **~10mm** (demonstrator slope 1.08 over a ~10mm size difference).
Measured: **<=2.2mm ~ 15%** — the same order as the slope deficit (0.17/1.08 = 16%).
**The width pathway barely consults vision.** This reconciles everything: the encoder CAN predict
size (r=0.81 honest), the estimate CAN be fed to the denoiser (ixjgp), and neither matters because
the width output does not read visual features.

**It is WEAK VISUAL DEPENDENCE, not occlusion.** Sensitivity is LOWEST during the approach when the
object is fully visible (0.06mm) and RISES at closure when the fingers occlude it (1.68mm) — the
OPPOSITE of the occlusion prediction. The policy never learned to use vision for width; it did not
lose it to the fingers.

**REFUTED (my hypothesis): between-category width is NOT EE-height inference.** All three objects
are grasped at the SAME height (mushroom 6.0mm, tofu 6.3, raspberry 5.7) and
corr(width, EE-z) = **-0.000** across objects.

⚠ **OPEN, and the single-step test CANNOT settle it:** these are ONE-STEP counterfactuals, but
closure is a ~30-step CLOSED-LOOP ramp in which each command is anchored to the current gripper
width (proprio). A 1-2mm per-step visual correction, INTEGRATED over the ramp, could well produce
the generalist's 15mm between-category gap. So "2mm swap response" does NOT prove vision is unused
between categories — the script's own printed interpretation overstated this. Settling it needs a
CLOSED-LOOP swap (run full episodes with a mismatched cloud), ~40min of eval, deliberately NOT run
tonight because it refines the story without changing the next action.

**WHAT THIS IMPLIES FOR THE FIX:** if the problem were occlusion, better viewpoints or earlier
latching would help. It is not. It is weak visual dependence, so the levers are the ones that FORCE
it: **`cond_dropout_prob`** (randomly drop the visual conditioning in training — already implemented
for the CFG work, standard remedy for a policy ignoring a modality) or a width path that
STRUCTURALLY cannot see proprio. Neither has been tried against the width objective.

### 2026-08-27 — ⚠ RETRACTED: three failed attempts to measure the width pathway's VISUAL SENSITIVITY

User asked WHY between-category width sizing works while within-category does not — occlusion, or a
vision-proprioception shortcut? Three offline probes were attempted; **all three are invalid and
none of their numbers should be used.**

1. **Cloud-swap** (`visual_ablation.py`) — swap another episode's cloud, keep proprio.
   **CONFOUNDED (user caught it):** the swapped cloud carries a different OBJECT POSITION, a
   different ARM/GRIPPER configuration and a different viewpoint, while proprio still describes the
   original pose. The policy received a physically incoherent, off-distribution observation, so the
   measured dwidth conflated size with pose mismatch. **RETRACTS the "1.68mm / ~15% visually
   sensitive" figure** and everything built on it.
2. **Fixed-phase sampling** — every probe sampled at a fixed FRACTION of the episode. Closure onset
   is at phase 0.45-0.55 (median 0.49) and episodes run 179-302 steps, so a fixed fraction lands on
   the APPROACH for some episodes and mid-closure for others. Symptom: the size-scaling run reported
   **79.98mm at every scale** (gripper still fully open — nothing to be sensitive with). The same
   error made the stochasticity test's ACROSS-observation number (0.02mm) meaningless, and may have
   contaminated the "occlusion dip" phase curve.
3. **Object-point scaling at closure onset** — mask = points within 5cm laterally of the EE and
   below it. At closure that region is mostly FINGERS and TABLE: measured "object extent" came out
   **68mm for a 33mm mushroom**, and the >=30-point filter left **n=1-2 episodes**. Meaningless.

**LESSON: an offline probe of a CLOSED-LOOP policy needs its sampling point defined by the
BEHAVIOUR (per-episode closure onset), not by episode fraction — and any "object" mask built from
EE-relative geometry is contaminated by the gripper exactly when the grasp happens.** Doing this
properly requires privileged object segmentation (available in sim, absent from the student cloud)
or a CLOSED-LOOP eval with a manipulated cloud. Not a quick probe.

**STILL STANDING (none of it depends on these probes):** policy slope **0.17** vs demonstrator 1.08
at 40 geometries (200-episode closed-loop evals); width head **r=0.81** honest at the latch moment;
the four-arm ablation (aux / aux+fed / blind-aux all fail to change behaviour and produce WIDER
CONSTANTS); the constant floor (-41% sustained stress at 0.745 success).

**OPEN: the mechanism behind between- vs within-category sizing is UNRESOLVED.**

### 2026-08-27 — the width head OVERFITS: train r=0.98 vs val r=0.75-0.81 (user challenged the number)

User asked whether r=0.81 holds on VALIDATION or is memorization, and how the head was trained.

**How it was trained:** NOT a post-hoc probe — an auxiliary MSE inside policy training.
Label = the episode's MIN gripper width (`states[:,-1].min()` per episode) broadcast to every step;
head reads the shared conditioning feature `[pointnet_feat (+) state]`; `MSE x 1.0` added to the
diffusion loss, on the TRAIN split. `val.npz` is forward-only under `torch.no_grad()` during
training, so no gradients flow from it.

**Train vs val at t=0 (`.agent_tmp/head_overfit_check.py`):**

| arm | TRAIN r | VAL r | gap | shuffled-label control |
|---|---|---|---|---|
| tebvy | **0.986** | 0.755 | +0.232 | -0.006 |
| ixjgp | **0.989** | 0.810 | +0.180 | +0.008 |
| neoca (blind) | **0.977** | 0.800 | +0.177 | +0.022 |

**SUBSTANTIAL OVERFITTING — the user's concern was right.** R2 goes 0.97 (train) -> ~0.60 (val).
The script's own printed line ("small train-val gap => real generalization") is TOO GENEROUS and is
overridden here: a 0.18-0.23 correlation gap is not small.

**What survives:** (a) the correlation is REAL — shuffled-label controls sit at ~0.00, so the metric
is sound; (b) it is GENUINELY VISUAL — `neoca`'s head is BLIND to gripper width by construction and
still reaches **0.800 val**, as good as the sighted heads, which rules out proprio leakage as the
source (this is the proprio-only control, obtained for free).

**CLAIM WEAKENED:** "an ACCURATE size estimate handed to the policy does not change its behaviour"
becomes **"a MODERATELY accurate estimate (val R2 ~0.6 in-distribution, and earlier work put
cloud->size at 0.44-0.59 on UNSEEN shapes) does not change it."** A policy ignoring a noisy signal
is a weaker indictment than one ignoring a clean signal.

**But it does not rescue the mechanism:** across the three arms there is NO ordering between head
quality and policy adaptation — ixjgp has the BEST head (0.810) and slope 0.20; tebvy has the WORST
(0.755) and slope 0.05. If head accuracy drove policy sizing, that ordering would exist.

### 2026-08-27 — USER HYPOTHESIS SUPPORTED: the shortcut explains the SPREAD, not just the flat slope

User: *"the policy instead learns a multi-modal behavior based on proprioception — the demonstrator
shows different grasp widths at similar pose (the true reason is object size but ignored) — so the
shortcut also causes the VARIANCE of grasp width for objects of the same size."*

**Quantitative support, from numbers already in hand:**

| quantity | sd |
|---|---|
| demo width, MARGINAL | 6.93mm (npz) / 8.01mm (dr_params collection) |
| demo width given PROPRIO (k=20 NN at the closure decision) | **6.67mm = 96% of marginal** |
| demo width given SIZE (regression, dr_params) | **4.33mm = 62% of marginal** |
| **POLICY width, marginal** | **7.91mm** |

**Proprio carries almost no width information** (`proprio_multimodal.py`): episodes with
near-identical ee_pos+ee_quat at the closure decision still show 91-96% of the marginal spread
(k=10/20; small-k numbers are biased LOW by the sd estimator and should not be quoted).
**Size does** (62%). So a proprio-only learner can do no better than the MARGINAL — and the policy's
spread IS the marginal (7.91 vs 8.01). The shortcut therefore explains BOTH failures at once:
no size tracking AND the ~8mm spread.

⚠ **INVALID CONTROL, discarded:** I tried a same-estimator k-NN in SIZE space using a cloud-derived
size proxy (lateral extent of low points in the first frame). The proxy reads **176mm mean for a
33-49mm mushroom** and correlates **-0.077** with width — it measures the TABLE. The size figure
above therefore comes from the dr_params regression, a DIFFERENT dataset and estimator; the two
should not be quoted as a matched comparison.

**ONE REFINEMENT to the hypothesis:** the policy is DETERMINISTIC given its observation (within-obs
sd 0.01mm), so it is not stochastically sampling a multimodal conditional — it is a DETERMINISTIC
proprio->width map whose outputs happen to span the marginal. Same spread, different mechanism, and
it matters: a stochastic policy could be sharpened by lowering sampling temperature; a deterministic
one cannot. This is consistent with the earlier finding that variance reduction is not a lever.

**THE RUNNING SIZE SWEEPS TEST THIS DIRECTLY:** within a pinned pose context proprio is nearly
constant across scales, so a proprio-driven width will be FLAT across the sweep while a
cloud-using one will track. That is the cleanest form of the test and needs no proxy at all.

### 2026-08-27 — ⚠⚠ RETRACTED: "the policy does not inherit the demonstrator's gentleness"

Upstream 21603e0 validated the CMA-ES FEM surrogate against MPM. It RANKS grasps well
(Spearman rho +0.842) but its ABSOLUTE scale is **~3x low**: planner 6.8-18.8 kPa vs MPM
20.7-46.4 kPa on the same 10 mushroom episodes, **MPM median ~100% of yield**.

**OUR COMPARISON MIXED THE TWO MEASUREMENT SYSTEMS.**
- DEMO stress (11,351 Pa = "0.28x yield") came from `dr_params.csv` `stress_Pa`, which the collector
  fills from `g["stress_top10"]` — and that key is produced by `smgrasp/lift_stress.py`, i.e. **the
  FEM PLANNER**, not the rollout. (The collector *also* computes a `_stress_top10(vm)` from MPM
  von Mises, but only for the internal FIRM decision; it never reaches the CSV.)
- POLICY stress (sustained 28,060 / peak 53,065) comes from the eval harness reading **MPM
  `von_mises_stress`** via `policy_env.py`.

So "**demonstrator 0.28x yield vs policy peak 1.33x — the policy does not inherit the
demonstrator's gentleness**", which §4d of the plan proposed as THE PAPER'S PROBLEM STATEMENT, is
**a planner number compared against a simulator number. RETRACTED.**

**Corrected, both on MPM:**

| | MPM stress | x yield |
|---|---|---|
| demonstrator (upstream, n=10) | 20.7-46.4 kPa | 52-116%, median ~100% |
| policy SUSTAINED | 28.1 kPa | 0.70x |
| policy PEAK | 53.1 kPa | 1.33x |

The policy's SUSTAINED stress sits INSIDE the demonstrator's range. Only its PEAK exceeds the demo
maximum. Upstream also notes part of the gap is definitional (the planner masks contact-adjacent
elements; the like-for-like unmasked comparison is **not yet checked**), so even the 3x is
provisional.

**SCOPE — what is and is not affected:**
- **AFFECTED:** every DEMONSTRATOR stress figure taken from `dr_params.csv` (`stress_Pa`), including
  the 0.28x-yield claim and the "demos are gentle, the policy is not" framing.
- **UNAFFECTED:** all POLICY stress from eval `summary.json` (MPM) — the -41% constant floor, the
  four-arm stress column, raspberry's above-yield SUSTAINED, and every between-arm gentleness
  ranking. Those compare MPM against MPM.

**LESSON (this is the wrong-reference-frame class again, 4th instance today):** two numbers with the
same NAME ("stress, % of yield") came from two different MEASUREMENT SYSTEMS. Before comparing any
two quantities, state which system produced each. A shared unit is not a shared scale.

### 2026-08-27 — ⭐ CONTROLLED SIZE SWEEP (user's design): no adaptation, and a SELECTION-BIAS trap caught

User designed the clean test: vary ONLY object size, closed-loop, with object pose / arm home /
yaw PINNED per sub-env for the whole run (`GM_FIXED_SCALES`, `GM_FIXED_POSE`, `GM_FIXED_YAW_DEG`
added to `sim_backend.py`, default OFF). Tofu, generalist xaqnb, 5 scales x 3 pose contexts x 3
repeats = 45 episodes, every one rendered, observations dumped.

**RESULT — all episodes (the honest view):**

| pose context | slope | width change over the 18mm span |
|---|---|---|
| yaw 0 (face-on) | +0.063 | +1.1mm |
| yaw 45 | -0.218 | -3.9mm |
| yaw 90 | +0.534 | +9.6mm |
| **pooled per-episode** | **+0.101** | — |
| *demonstrator* | *1.08* | *+19.4mm* |

**Even with pose PINNED, there is no demonstrator-like size tracking.** That answers the question
the sweep was built for: pose variation was NOT masking real sensitivity.

**⚠ THE TRAP — success-only analysis shows FALSE adaptation.** Restricting to successful episodes
gives slope **+0.856 (yaw 0)** and **+0.897 (yaw 90)** — near the demonstrator's 1.08. It is
SELECTION BIAS, a collider: size -> success <- width. Proof, from the failures:

| size | ALL width | SUCCESS | FAIL |
|---|---|---|---|
| 24.0mm | 29.9 +- 3.6 | **25.8** (n=3) | **31.9** (n=6) |
| 28.5mm | 34.2 +- 5.1 | **29.9** (n=3) | **36.4** (n=6) |
| 42.0mm | 33.5 +- 5.3 | 33.5 (n=9) | — |

The ALL-episode distribution is flat across sizes; at SMALL sizes the too-wide draws FAIL, so
conditioning on success keeps the narrow ones and manufactures a slope. **This validates the
existing choice in `decompose_width.py` to use ALL episodes** — a success-only pipeline would have
reported adaptation that does not exist.

**NEW, and it matters for shipping:** small objects fail by being gripped **TOO WIDE** (31.9mm on a
24mm object), i.e. they are DROPPED, not crushed. **The constant floor only ever WIDENS the grip, so
it would make small-object success WORSE.** Ship it for mushroom-sized objects only; it is the wrong
mechanism for a multi-category setup containing raspberries.

**DESIGN NOTE for the next sweep:** the within-scale repeat sd is **3.28mm** (MPM is not
bit-reproducible), so with 3 repeats each cell mean carries ~+-1.9mm — enough to rule out a
demonstrator-like slope (19.4mm expected) but NOT to estimate a small slope precisely. The scale
count (5 vs 10) is not the limiting factor; REPEATS are. Use >=6 repeats per cell if the goal is a
precise slope rather than a demonstrator-vs-flat verdict.

### 2026-08-27 — 10-SCALE SWEEP: the policy IS size-sensitive, ~30% of the demonstrator (5 scales is NOT enough)

Same controlled design, tofu / generalist xaqnb, pose pinned per sub-env. Comparing scale counts:

| pose context | 5 scales (45 eps) | **10 scales (90 eps)** |
|---|---|---|
| yaw 0 (face-on) | +0.063 | **+0.259** (r +0.53) |
| yaw 45 | **-0.218** | **+0.147** (r +0.31) |
| yaw 90 | +0.534 | **+0.491** (r +0.72) |
| mean | +0.13 | **+0.30** |
| noise floor (within-scale repeat sd) | 3.28mm | 2.81mm |

**5 SCALES IS NOT ENOUGH** — at 5 the contexts disagreed in SIGN (yaw 45 came out -0.218, noise);
at 10 all three are positive and the mean roughly doubles. The extra 45 episodes change the ANSWER,
not just the error bars. **Use 10 scales.** (Earlier note said repeats were the binding constraint;
that was based on the 5-scale run alone and is superseded — scale count matters too.)

**REFINEMENT TO THE CAMPAIGN'S CONCLUSION, in the policy's favour:** with pose properly controlled
the generalist shows **consistent positive size sensitivity, ~0.15-0.49 (mean 0.30) = ~30% of the
demonstrator's 1.08** — not the ~0.1 the uncontrolled 40-geometry measurements suggested. Pose
variation WAS masking part of the signal. The policy is **UNDER-RESPONSIVE by ~3x**, not size-blind.
Correct the earlier "no size tracking / flat" framing to "under-responsive by ~3x".

**Selection bias reappears, milder:** success-only gives 0.48-0.61 vs all-episode 0.15-0.49. Same
collider (size -> success <- width), smaller inflation because failures are rarer (67/90 succeed).

**ANALYSIS BUG found and fixed (no re-run needed):** `analyze_sweep.py` binned sizes with
`np.round(S,2)` but tested membership as `|S - bin| < 1e-6` against the RAW values. Exact for
tofu-5's scales (24.0, 28.5, ...), broken for tofu-10's (26.001 vs bin 26.0) -> empty cells and NaN
slopes. Fixed by rounding once and grouping on the rounded array; the tofu-5 numbers reproduce
exactly, confirming the fix changed nothing else.

### 2026-08-27 — CLARIFICATION (user caught it): "uncertain" is NOT "zero"

User asked whether the controlled sweep contradicted the earlier "the tofu generalist is not width
adaptive". It does not — **the numbers agree; my WORDING was wrong.**

| arm | uncontrolled (full DR, per-geometry) | controlled sweep (pose pinned, 10 scales) |
|---|---|---|
| xaqnb / tofu | **0.31** CI [-0.14, 0.76], R2 0.13, verdict "uncertain" | **+0.30** (0.26 / 0.15 / 0.49) |
| lulkx / mushroom | **0.17** CI [0.00, 0.33] | **-0.04** (-0.24 / +0.04 / +0.08) |
| xaqnb / mushroom | 0.12 CI [-0.49, 0.74] | (running, 1758351) |

**TOFU: controlling the DR changed NOTHING** — 0.31 -> 0.30. The positive result was always in the
data; the uncontrolled CI merely spanned zero at 12 geometries. **I then spoke about that arm as
though it were FLAT** ("the generalist does not track size within category"). The metric's own
verdict word is "uncertain", which means *cannot be distinguished from zero* — NOT *is zero*.
Conflating the two is how a wide-CI point estimate of 0.31 got reported as no adaptation.

**MUSHROOM: controlling made it slightly WORSE** (0.17 -> -0.04), so pose variation was not hiding
sensitivity there; if anything the uncontrolled number flattered it.

**RULE going forward:** report a slope as (point estimate, CI) and say "indistinguishable from zero"
when the CI spans it — never "flat", "no adaptation", or "size-blind", which assert the point
estimate IS zero. Three pose contexts give the sweep mean an SE of ~0.10, so tofu's +0.30 is ~2.9
sigma from zero: suggestive, not decisive, and still ~3.5x under the demonstrator's 1.08.

### 2026-08-27 — PROPRIO-SHORTCUT ARMS (user's idea, from GAP: arXiv 2602.12032)

User pointed at "When would Vision-Proprioception Policies Fail in Robotic Manipulation?"
(Lu, Xia, Wu, Lu, Hu). Read the paper; two things matter for us.

**1. Their Table 1: VISION-ONLY BEATS VISION+PROPRIO CONCATENATION on nearly every task.**

| task | vision-only | concat | GAP |
|---|---|---|---|
| Meta-World pick-place | 91.8 | 78.4 | 94.2 |
| RoboSuite threading | 43.6 | 33.2 | 53.0 |
| real: press button | 18/20 | 12/20 | 20/20 |
| real: lift lid and pour | 9/20 | 5/20 | 15/20 |

**2. GAP is a GRADIENT modulation, not dropout (user corrected me on this).** Eq 5:

    w_s^{j+1} = w_s^j - lambda * (1 - rho) * eta * grad_{w_s} L_BC

`w_s` = parameters of the PROPRIOCEPTION CHUNK `phi_s`, which in their architecture is "an encoder
and a temporal transformer". `rho` = motion-transition phase indicator, obtained by Change Point
Detection on the motion representation {dp, dtheta, dg} to segment motion-CONSISTENT phases, then an
LSTM predicting per-timestep rho supervised by those segment indices. High rho -> gradient damped.

**KEY ARCHITECTURAL FACT FOR US: we have no `w_s` to damp.** `_cond_encoded` RAW-CONCATENATES the
state (`parts = [feat, state]`) with no proprio encoder, so GAP's mechanism is not expressible until
one exists. The standard input-side trick (`p*a + p.detach()*(1-a)`) does NOT help: it scales the
gradient w.r.t. p, and p is an INPUT, so no parameter update changes.

**AND OUR ADVANTAGE (the user's point): our demonstrator is SCRIPTED, so rho is KNOWN.** No CPD, no
LSTM — the dataset already computes the closing/hold window for `width_window_weight`, and we reuse
it as a binary rho.

**Four arms launched on the GENERALIST (xaqnb's exact recipe, seed 42, 350 epochs, paired-reg 0.5;
each differs in ONE thing):**

| arm | job | mechanism | eval-side change |
|---|---|---|---|
| A vision-only | 1758631 | whole proprio zeroed | yes (`GM_BLIND_PROPRIO`) |
| B width-blind | 1758632 | only the gripper-width channel zeroed | yes (`GM_BLIND_GRIPPER_WIDTH`) |
| C GAP | 1758633 | `proprio_encoder` + Eq-5 damping in the known grasp window | none (gradient-only) |
| D control | 1758634 | same encoder, `gap_damp=false` | none |

**C needs D.** The encoder is an architecture change GAP requires; without D we could not tell
damping from the extra capacity.

**Implementation, verified by smoke test before spending 16 GPU-hours** (`.agent_tmp/smoke_arms.py`):
encoder gradient damped inside the window (0.135 vs 0.230 undamped); **forward value identical, max
diff 0.00e+00** (Eq 5 is gradient-only, so inference is untouched); D undamped (0.204); A/B ablations
exactly as intended. Straight-through form: `h = s*h + (1-s)*h.detach()` with `s = lambda*(1-rho)`.

**Gotcha worth recording:** C/D need `aux_grasp_width` + `normalization_path` on the DATASET, because
that is the only path that builds the grasp-window mask. The model's aux WEIGHT stays 0 (no aux loss
— that objective was shown harmful earlier today); the label is used solely to locate the window.
Without it, C would have trained with `gap_damp=True` and no window to damp in, and produced a clean
null that looked like a real negative result.

### 2026-08-28 — upstream 8b8f4db: von Mises SATURATES past yield. Our SUSTAINED metric survives; PEAK does not.

Upstream retracted its own rho=+0.84 surrogate validation: it was a SCENE-SIZE confound (planner vs
scale rho -0.67, MPM vs scale rho -0.89). A CONTROLLED test (fixed scene, sweep only grasp width)
gives **rho = 0.000** (Pearson -0.47). Root cause: Genesis MPM is ELASTO-PLASTIC, so von Mises
saturates at the yield surface — past yield, squeezing produces plastic flow, not higher stress.
Their conclusion: *von Mises is the wrong gentleness measure past yield; plastic deformation is.*

**OUR DATA SHOWS THE SAME SATURATION, and it splits our two metrics cleanly:**

| arm | SUSTAINED | x yield | PEAK | x yield |
|---|---|---|---|---|
| baseline lulkx | 29734 | 0.74 | 53470 | 1.34 |
| floor m8 | 22804 | 0.57 | 51737 | 1.29 |
| floor m6 | 19700 | 0.49 | 50615 | 1.27 |
| floor m4 | 18047 | 0.45 | 49709 | 1.24 |
| constant floor 32.84 | 16439 | 0.41 | 49052 | 1.23 |

**SUSTAINED spreads 45% across mechanisms (29.7k -> 16.4k) and sits at 0.41-0.74x yield — BELOW the
surface, still informative. PEAK spreads only 8% (53.5k -> 49.1k) and is pinned at 1.23-1.34x yield
— the saturated regime.** The 8% peak spread is not a gentleness difference; it is the metric
running out of range.

**CONSEQUENCES FOR OUR CLAIMS:**
- **SURVIVE** (all rest on SUSTAINED, below yield): the constant floor's -41%, the floor margin
  frontier, the four-arm stress column, every between-arm gentleness ranking. Choosing
  `stress_top20_ttop20` as the primary metric turns out to have been load-bearing.
- **DO NOT USE for gentleness**: `stress_max_tmax` (PEAK). "The policy peaks at 1.33x yield" means
  "past yield", full stop — the MAGNITUDE is not interpretable, so it cannot rank arms.
- Raspberry's SUSTAINED at 1.13x yield is ALSO in the saturated regime -> that one number is
  suspect, unlike the mushroom/tofu sustained figures.

**Method lesson (upstream's words, and it is the third instance tonight):** *an observational
correlation across DR-varied scenes is not metric validation.* Same family as our success-only
selection bias and the 12-vs-40-geometry artefact: a confound that varies WITH the quantity of
interest manufactures a correlation. The remedy in every case was the same — fix everything else and
sweep one variable, which is exactly what the user's controlled size sweep does for width.

### 2026-08-28 — 2x2 COMPLETE: the OBJECT drives size sensitivity, not multi-object training

Controlled sweeps (pose pinned per sub-env, 10 scales over each object's own DR range, 90 eps):

| | mushroom | tofu |
|---|---|---|
| **specialist** (lulkx) | **-0.04** (-0.24/+0.04/+0.08) | — |
| **generalist** (xaqnb) | **+0.06** (+0.11/-0.04/+0.11) | **+0.30** (+0.26/+0.15/+0.49) |

- **Same policy, different OBJECT: +0.06 -> +0.30** (~2.2 sigma on 3 pose contexts). Real.
- **Same object, different TRAINING REGIME: -0.04 -> +0.06** (~0.9 sigma). **Indistinguishable
  from noise.**

**MY HYPOTHESIS THAT MULTI-OBJECT TRAINING INDUCES SIZE RESPONSIVENESS IS NOT SUPPORTED.** The tofu
result is about TOFU: a cube's silhouette scales legibly in a point cloud, while a mushroom's
rounded cap (plus stem) is harder to size. **The "+0.30 / under-responsive by 3x" characterisation
therefore applies to tofu specifically, NOT to the policy in general** — the same policy is at +0.06
on mushroom.

**Consequence for the final pipeline: do NOT expect the multi-object generalist setup to deliver
width adaptation by itself.** Whatever fixes this has to be a mechanism (the four proprio-shortcut
arms are the current test), not a data-composition change.

Figure: `docs/figures/size_sweeps_width_vs_size.png` (4 panels: the 2x2 plus the 5-scale
under-sampling check).

### 2026-08-28 — PROPRIO ABLATION IS NOT VIABLE HERE: A and B are 0/21. GAP is the only live arm.

| arm | val loss @350 | mushroom success |
|---|---|---|
| baseline xaqnb | 0.0012 | 0.883 |
| **A vision-only** (all proprio zeroed) | 0.0056 (4.7x) | **0.00 — killed by the degenerate watchdog at 21 eps** |
| **B width-blind** (gripper-width channel only) | 0.0029 (2.4x) | **0.00 — same** |
| C GAP (encoder + Eq-5 damping) | 0.0014 | (running) |
| D encoder control | 0.0009 | (running) |

**Removing proprio does not expose the shortcut — it BREAKS THE POLICY.** The val losses predicted
it (4.7x / 2.4x worse) and the evals confirmed it: zero successes.

**The surprise is B.** I argued a policy emitting ABSOLUTE width targets should not need its CURRENT
width. It does. Most likely the gripper-width channel carries **PHASE** information — where the
episode is in approach/close/lift — so deleting it destroys closure TIMING, not just width choice.
That also fits the earlier observation that the policy's width command tracks its own previous
width: it is using that channel as a clock.

**CONSEQUENCE: "the policy over-relies on proprio" cannot be fixed by REMOVING proprio in this
setup.** Proprio is load-bearing for basic function. This is why the GAP paper's vision-only result
(which BEAT concatenation on their tasks) does not transfer: their Meta-World / RoboSuite tasks
differ in exactly the relevant way — we use absolute pose+width commands against a
scripted-demonstrator phase structure, so the policy has no other source of phase.

**GAP is therefore the RIGHT SHAPE of intervention for us and the only live arm**: it keeps proprio
available at INFERENCE and damps only its LEARNING inside the grasp window (Eq 5 is gradient-only,
verified forward-identical). Its val loss (0.0014) sits next to baseline, so nothing is broken.
D (encoder, no damping) isolates the encoder's capacity from the damping.

Watchdog note: the degenerate-eval guard killed both dead arms at 25 min instead of 90, saving
~2 GPU-hours. That guard has now paid for itself three times today (B16, A, B).

### 2026-08-28 — ALL FOUR PROPRIO ARMS FAIL CLOSED-LOOP (0/21). Val loss does NOT predict success.

| arm | val loss @350 | mushroom success | why |
|---|---|---|---|
| baseline xaqnb | 0.0012 | **0.883** | — |
| A vision-only | 0.0056 (4.7x) | 0.00 | proprio removed -> policy loses its CLOCK |
| B width-blind | 0.0029 (2.4x) | 0.00 | same, from ONE channel |
| C GAP (encoder + damping) | 0.0014 | 0.00 | see below |
| **D encoder CONTROL** | **0.0009 (BETTER than baseline)** | **0.00** | **the encoder alone** |

**Load-mismatch hypothesis CHECKED AND REJECTED.** `DiffusionEval` falls back to
`load_state_dict(strict=False)`, so a key mismatch would be SILENT and would explain D exactly.
Verified (`.agent_tmp/check_encoder_load.py`): encoder keys present in both checkpoint and eval
model, **0 missing, 0 unexpected**. The failures are real.

**THE FINDING: BC validation loss is a weak predictor of closed-loop success — D is the clean
demonstration.** D has LOWER val loss than the baseline and ZERO success. Val loss measures
single-step action prediction on demonstration states; success needs 300 steps of stability under
the policy's OWN state distribution, where representation changes compound. We have been reading
val loss as a health check all session; it is not one.

**Consequences:**
- A/B: proprio is LOAD-BEARING (the phase/clock signal), so the shortcut cannot be removed by
  ablation in this setup. Substantive.
- C/D: inserting ANY learned transform between raw proprio and the denoiser destabilises closed-loop
  control, INDEPENDENT of GAP's damping — **D is what proves this**, and it is exactly why the
  control arm was run.
- **GAP therefore remains UNTESTED**: its prerequisite (a proprio encoder) is not free here.

**If pursued, the fix is near-IDENTITY INITIALISATION of the proprio encoder** so the policy starts
at baseline behaviour and departs only as training justifies — standard when inserting a module into
a working architecture, and it would make D a true no-op control. Cost: ~4h retrain + 1.5h sweep for
C and D. NOT launched — four consecutive arms have now failed and the deadline is close; the
alternative is to stop mechanism-hunting and write up.

**Guard worth adding regardless:** `strict=False` on checkpoint load converts a structural mismatch
into a plausible wrong answer (same family as the wrong-NORM and wrong-reference-frame bugs). An
assert that no INFERENCE-path key is missing would keep the aux-head tolerance without the silence.

### 2026-08-28 — WHY C/D FAILED (init, not architecture) — and why our GAP arm was NOT a faithful GAP test

**The 0/21 was diagnosed to a bad initialisation, and the bug hunt cleared the code.** Ruled out, in
order: port mismatch (train vs eval `_cond_encoded` **bit-identical, max diff 0.000e+00**), silent
key drop (`strict=False`; **0 missing, 0 unexpected**), EMA not tracking the new params
(**0.0002 relative on the encoder**; D fails with EMA *and* raw weights). All rejected.

**Cause: a RANDOMLY-INITIALISED module inserted into the proprio path.** Proprio is the policy's
closure CLOCK (established by arms A/B: removing it -> jitter at the grasp pose, user's own video
read). A random encoder scrambles that clock at step 0, the policy learns to disregard it, vision
cannot supply phase, and it converges to "approach, never close" — exactly the observed behaviour,
confirmed offline on VALIDATION demo states (D flat at ~80mm where the demo goes 80 -> 36.7 -> 33).

**FIX, verified:** residual encoder `h = state + MLP(state)` with the final layer ZERO-initialised
-> **exact pass-through at init (max diff 0.000e+00)**. NOTE the first attempt — identity weights on
both linears — was NOT a pass-through (**2.493**, barely better than random's 3.276) because
`Mish(x) != x`. The smoke test caught that before it cost a 4h run.

**⚠ AND A DEEPER PROBLEM (user asked "are you using the same setup as the paper?"): NO.**
GAP Eq 2 is `â = (psi_s(f_s) + psi_v(f_v)) . W_share + b` — proprio and vision have SEPARATE chunks
and SEPARATE HEADS whose outputs are SUMMED, which makes the proprio contribution SEPARABLE, so
damping `w_s` genuinely reduces its influence. **Ours CONCATENATES `[pointnet_feat (+) proprio]`
into ONE shared trunk, so damping a bolted-on encoder does nothing: the trunk downstream is free to
compensate.**

| | GAP | ours |
|---|---|---|
| fusion | separate heads psi_s, psi_v, **SUMMED** | **CONCATENATED** into one trunk |
| w_s | encoder + temporal transformer (whole pathway) | a bolted-on 2-layer MLP |
| rho | continuous, CPD + LSTM | binary, known grasp window |

**CONSEQUENCE: our "GAP arm" cannot test GAP.** Even a clean run would only show whether damping a
bolted-on encoder shifts width — the mechanism GAP relies on is absent. A faithful test needs the
two-branch summed-head denoiser, which changes the BASELINE too (~10h for baseline + C + D + sweeps).
**Recorded as scoped future work.** My earlier withdrawal of the lambda point stands and is
unaffected: D has NO damping and failed identically, so damping never explained D.

**METHOD CORRECTION with session-wide reach: epsilon-prediction val loss is NOT an action-quality
measure.** D posted the BEST val loss of any arm (0.0009 vs baseline 0.0012) while never closing.
The loss is dominated by predicting injected noise, and width sits at its open value ~85% of every
episode, so a total failure on that dimension barely moves it. **Val loss was used as a health check
all session; it is not one.** Closed-loop success is the only health signal that counts.

### 2026-08-28 — VENDORED GAP + the three faithful arms; a per-split phase-file bug caught by an assert

Per user instruction ("just try to use their code", "do not self reimplement", "make sure their
things are minimally touched"), GAP is now VENDORED at `third_party/GAP/` (clone of
GeWu-Lab/GAP, `.git` removed) with `third_party/GAP/VENDORED.md` recording the mechanism AS
IMPLEMENTED IN THEIR CODE plus the 5 places their code differs from their paper text. We run
their CPD (ruptures PELT + their `CostDirection`, pen=4) + LSTM phase estimator unmodified.

**Three arms, one variable each, all on xaqnb's exact recipe (seed 42, 350 ep, paired-reg 0.5):**

| arm | job | rho source | isolates |
|---|---|---|---|
| C' `gapC_faithful` | 1762867 | our known grasp window (wide) | GAP with ground-truth phase |
| D' `gapD_control`  | 1762868 | — (GAP off, encoder on) | the encoder ALONE |
| E  `gapE_theirphase` | 1763013 | THEIR CPD+LSTM (sparse) | faithful GAP replication |

Their rho is SPARSE — `min 0.000 max 0.998 mean 0.0075, frac>0.5 0.0076` over 1248 trajs — it
marks MOTION TRANSITIONS, not a wide window. E is therefore the arm that actually reproduces
their method; C' tests our own prior about where the phase is.

**BUG (caught at startup by an assert I had added, cost ~2 min x2, zero silent corruption):**
`AssertionError: phase length 254340 != total steps 28042`. A phase file is computed from ONE
npz split, but I passed it to BOTH `+train_dataset.gap_phase_json` AND `+val_dataset.gap_phase_json`.
The generalist splits 90/10: **train 1248 trajs / 254340 steps, val 138 / 28042**. The val
dataset's assert fired.

- **First fix was WRONG** — I sliced `_ph[:max_n_episodes]`, but `max_n_episodes` defaults to
  10000, so the slice was a no-op and the job failed identically. The arithmetic named the real
  cause: 254340 + 28042 = 282382 and 28042 is 9.9% of it = the train/val split.
- **Correct fix:** pass the phase to `train_dataset` ONLY. Verified safe — the network's only
  consumer of `in_grasp_window` (`pointnet_diffusion.py:323`) is gated on
  `self.training and proprio_dropout_prob > 0`, and E leaves dropout at the 0.0 default; the GAP
  gradient hook reads `batch_train.conditions` only. The assert message now names the split cause.

**LESSON (reusable, and it generalises past GAP):** any per-timestep side-channel file (phase,
weights, labels) is **split-specific** — it must be handed only to the dataset built from the
same npz. Assert `len(side_channel) == sum(traj_lengths)` at load; without it, a 9x misalignment
would have indexed rho into the wrong timesteps and produced a plausible-but-meaningless GAP arm.
This is the [[wrong-reference-frame-bug-class]] pattern (right arithmetic, wrong reference) caught
LOUDLY for once, and it is the argument for keeping such asserts even when they cost a relaunch.

### 2026-08-28 — ABSORBING upstream f7bb394 (sub-yield restores rho +0.52): what it changes for OUR metrics

Upstream re-ran the surrogate validation in the SUB-YIELD regime after moving the squeeze base
5mm -> 2mm: Spearman rho **0.000 (past yield) -> +0.517 (sub-yield, p=0.085, n=12)**, 100% sub-yield.
Their verdict: **"gentleness-aware" is defensible; "provably gentle" is NOT.** n=40 sweep running.

**Consequences for our side (read against our 2026-08-28 saturation entry):**

1. **Our SUSTAINED metric is VINDICATED, and now for a stated reason.** `stress_top20_ttop20_mean`
   puts our arms at **0.41-0.74x yield** — inside exactly the sub-yield band where upstream now
   shows von Mises still carries ranking signal. Our policy-vs-policy sustained comparisons stand.
2. **Our PEAK metric stays RETRACTED.** 1.23-1.34x yield is above the saturation point; upstream's
   item 3 states plainly that von Mises is "the wrong axis above yield". No change — but the reason
   is now sharper than "it saturates".
3. **NEW future-work item, adopted from their item 3: damage should be measured as PLASTIC
   DEFORMATION** (permanent shape change / plastic work), not von Mises, for an elasto-plastic body.
   Every stress number in this DEVLOG is a proxy. This does not invalidate our relative rankings
   (all arms measured on the same proxy, in the sub-yield band) but it does cap what we may CLAIM.
4. **Claim discipline, inherited:** we may write "gentleness-aware"; we may NOT write "provably
   gentle", "gentleness-optimal", or "minimizes damage". Add to the plan's section 8 (WHAT WE WILL
   NOT CLAIM).
5. **Operating-point drift to watch (NOT yet an issue):** their adopted squeeze base is now 2mm,
   while our current generalist data was collected at 5mm extra squeeze (see the collector
   version/squeeze confound note). The final collection will therefore sit at a GENTLER operating
   point than the interim data our method development runs on. Our deliverable this phase is THE
   METHOD, not the checkpoint, so this is tolerable — but any absolute stress number quoted from
   current data must be labelled with its squeeze base, and width-vs-size SLOPES should be
   re-measured on the final data before they go in the paper.

**No merge performed** — the merge remains held per user instruction; this was read via
`git show f7bb394` on a fetched ref, working tree untouched.

### 2026-08-28 — ARM F: their GAP codebase end-to-end (only the RGB+depth branch swapped for a cloud)

User: *"I meant running with their code base, is it one of the experiment? Because I am curious if
architecture matters. Only need to adapt the RGB+depth branch to pnt cld."* C'/D'/E graft GAP's
gradient rule onto OUR point-cloud diffusion policy; **F runs THEIR stack** — their `WorkSpace`
trainer, `BCPolicy`, `ProEncoder`, `DeterministicMLP` head, AdamW, CosineAnnealingLR, GAP rule, and
their CPD+LSTM phase — on our data. If GAP helps there and not here, architecture is the reason.

**Job 1763402** (`logs/gap_arch/armF_full`); smoke 1763378 passed end-to-end in 68 s.
Code: `gentle_manip/dppo/gap_arch/{gm_cloud_adapter,run_gap_arch}.py`, launcher
`.agent_tmp/gap_arch.sbatch`. Proven params are READ FROM THEIR YAMLS at runtime, not retyped:
lambda 0.3, horizon 9, n_obs_steps 5, batch 128, lr 3e-4, hidden_dim 512, epoch 101, seed 1,
DeterministicMLP head, AdamW(wd 1e-4).

**Exactly ONE edit to their tree** (`gap/gap.py`, `.orig` kept + diffed): `lambda = self.cfg.lambda`
-> `lambda_ = self.cfg['lambda']`. **Their gap.py does not parse as published** — `lambda` is a
Python reserved word. Everything else is injected by monkeypatch. Two environment stubs, both
raising if ever called, neither touching training math: `torchvision` (module-level resnet18 import;
F builds no ResNet) and `diffusers` (their head.py imports DDPM/DDIM schedulers, used only inside
`ConditionalUnet1D`/`TransformerForDiffusion` — verified NOT by `DeterministicMLP`). Stubbed rather
than installed because installing either into the shared aarch64 venv risks pulling a different
torch build while C'/D'/E are training in it.

**THREE PROPERTIES OF THEIR CODE, found by reading it and then CONFIRMED empirically by the smoke:**

1. **Their `'pro' in name` filter damps the VISION branch too.** `SpatialProjection` is bound to
   `self.projection`, and "projection" contains "pro". Smoke output:
   `GAP damps 52/112 encoder tensors (4,504,960/8,821,376 = 51.1%)`, broken down as
   `{'imgencoder': 2, 'proencoder': 50}` — the 2 are the visual projection's weight+bias.
   Almost certainly unintended by the authors, but it IS their code, so F reproduces it.
   **Our C'/D'/E match `proprio_encoder` only — truer to the PAPER, less true to the CODE.**
   The runner prints the damped tensor list every run so this is never silently assumed.
2. **Their `CosineAnnealingLR` is stepped PER BATCH with `T_max=cfg.epoch`**, so it is not a decay
   at all — it is a CYCLIC LR of period `2*epoch` BATCHES. The smoke's lr_trace (T_max=2) shows the
   period-4 cycle exactly: `[3e-4, 1.5e-4, 0, 1.5e-4, 3e-4, ...]`. At epoch=101 the full run cycles
   every 202 batches (~1000 cycles over training). Reproduced faithfully; `lr_trace.npy` is saved.
3. **Their validation set is a SUBSET of their training set** — train `demo_range=[0, demo_num]`,
   valid `[0, int(0.1*demo_num)]`. So their `snapshot_valid` selects on contaminated loss. Left
   untouched; **we will pick F's checkpoint by CLOSED-LOOP sim eval, not by their valid loss** —
   consistent with the already-established lesson that eps-prediction val loss does not predict
   closed-loop success.

**A fourth, relevant to eval:** their `BCPolicy.get_action` does `if img.shape[0] != 1:
img = img.unsqueeze(0)` — it assumes a SINGLE env. The canonical harness runs num_envs=5, so F's
eval adapter must call `head.get_action(encoder(...))` directly rather than that wrapper, which
would otherwise reshape a 5-env batch into nonsense.

**Bug caught in MY adapter before it ran (not theirs):** indexing the NpzFile inside the
per-trajectory loop (`d["point_cloud"][s:s+L]`) re-decompresses the FULL 3.1 GB member on every one
of 1248 iterations. Arrays are now materialized once; per-trajectory slices are views, and a strict
subset (their 10% valid set) is copied so it does not pin the whole buffer.

**Caveat to state when reporting F:** their `epoch: 101` was tuned on 100-500 demos; our 1248-demo
set makes an epoch ~2.5x larger, so F gets ~200k gradient steps vs ~79k in their hammer setup.
Epoch count is their proven parameter and is kept; the COMPUTE is therefore not matched to theirs.

### 2026-08-28 — upstream 0210ba1 (material DR NEVER applied): we have the mismatch too, but it is BOUNDED

Upstream's pre-flight found `DRConfig.sample_scene()` was never called by `collect_demos_synth_v3`,
so **every demo it ever produced used the registry NOMINAL material** — E/nu/rho/coup_friction
ranges in all seven DR configs were dead text. Fixed upstream; collection relaunched.

**VERIFIED ON OUR OWN DATA (not assumed).** `dr_params.csv` for all three generalist sources —
mushroom `26-08-17-hwo` (688 rows), tofu `26-08-25-yhn`, raspberry `26-08-27-abp` — carries **no
material column at all** (no mat_E/mat_nu/mat_rho/coup_friction); only `scene_scale` and
`scene_bend_deg`. So our generalist training set has **zero material diversity**.

**AND THE EVAL DOES RANDOMIZE MATERIAL.** `episodes.csv` shows mat_E 2.107e5-2.871e5,
mat_nu 0.321-0.371, mat_rho 952-1083 (mat_yield constant 4e4 — matching upstream's known
`ObjectEntry` limitation). That is the [[rigid-demo-eval-dr-mismatch]] class recurring on a new
axis: **train has no material DR, eval has +-15% E.**

**MEASURED IMPACT — bounded, and NOT significant in the operating regime.** Success by mat_E
tertile across 618 evals carrying material DR:

| sample | low-E | high-E | difference |
|---|---|---|---|
| a FAILING policy (zffwn, succ 0.00-0.09, n=1200) | 0.108 | 0.008 | **-0.100 +-0.029 (significant)** |
| **STRONG policies (succ >= 0.30, 14 runs, n=1855)** | **0.885** | **0.855** | **-0.030 +-0.031 (NOT significant)** |

My first cut used the six most recent files, which all happened to be `zffwn` — a floor-level
policy, where a marginal difficulty shift is amplified and two of three tertiles were at 0.000.
**The strong-policy pooled test is the relevant one**, and per-run differences scatter both ways
(+0.092 to -0.275), consistent with noise.

**CONSEQUENCES (stated precisely):**
1. **Arm-vs-arm comparisons STAND.** Every arm carries the identical handicap and the effect is
   bounded at ~0.03 success — far below the differences we act on (e.g. floor m8's -0.145).
2. **Absolute success numbers are within noise of unbiased**; no re-measurement is warranted.
3. **We may NOT claim robustness to object firmness.** The training data contained none and we
   never tested it. Add to plan section 8.
4. **The final collection will be genuinely harder** (material DR active for the first time), so
   current success rates are NOT a baseline the final pipeline should be held to.
5. Our stress-vs-yield fractions are all against a CONSTANT yield (4e4) — the yield axis has never
   been randomized in either training or eval, so it is a fixed reference, not a source of spread.

### 2026-08-28 — CORRECTION: C' is NOT a phase-guided arm. GAP's per-BATCH max destroys a dense phase.

I described C' as "GAP with our known grasp window as rho". **That description is wrong in effect**,
and the arithmetic says so before the run finishes.

GAP's coefficient is `1 - lambda * max(phase)` where the max is over the **whole batch**
(`phase_p = torch.max(batch['phase']).item()`). With batch_size 128:

| arm | phase signal | batch-max behaviour | effective coefficient |
|---|---|---|---|
| **C'** | BINARY grasp-window flag, **52% of steps** (log: "width-window loss weight 0.0 on 52% of steps"); per-SAMPLE higher still, since the flag is `.any()` over the 4-step horizon | `1 - 0.48^128 = 1 - 1e-41` | **0.700 on EVERY batch, every epoch 0-50** |
| **E** | THEIR CPD+LSTM rho, sparse (per-sample mean 0.0076, frac>0.5 0.0077) | mean 0.612, p10 0.000, p90 0.998 | **varies: mean 0.817, sd 0.124, range [0.70, 1.00]; 77% of batches modulated, 17% essentially unmodulated** |

**So C' is a UNIFORM 30% gradient damping of the proprio encoder for epochs 0-50 — arithmetically
just a 0.7x learning-rate multiplier on that submodule. It contains no phase guidance whatsoever.**
E is the only genuine phase-guided replication we have, which retroactively justifies the work to
get it running.

**THIS IS A REAL METHODOLOGICAL CRITIQUE OF GAP AS IMPLEMENTED, not a bug in our port.** Taking the
max over a batch of 128 means the coefficient saturates to its floor unless the phase signal is
SPARSE. Their own rho is sparse (0.77% of samples above 0.5), so their implementation works for
them; ANY dense or wide phase definition silently degenerates to a constant. A per-SAMPLE
coefficient would not have this property. Worth stating if GAP is discussed in the writeup.

**The four arms are still a well-formed set — with corrected labels:**
- **D'** — encoder, NO damping (control)
- **C'** — encoder + UNIFORM 0.7 damping, epochs 0-50 (*not* phase-guided)
- **E**  — encoder + GENUINE phase-guided damping (their rho, their estimator)
- **F**  — their entire architecture (BCPolicy + DeterministicMLP + their trainer)

C' -> E now isolates exactly "does the PHASE GUIDANCE matter, given the same mechanism and the same
lambda?" — a cleaner question than the one I originally set up, obtained for free.

### 2026-08-28 — arm F eval path built and SMOKE-TESTED before the run finished

`gentle_manip/dppo/gap_arch/eval_gap_arch.py` + cfg `.../eval_gap_arch.yaml` (forked from
`eval_diffusion_pointnet.yaml`; diff is exactly `_target_`, the `model:` block, and
`cond_steps 2 -> 5`). Same EvalSpec/venv/metrics as every other arm, per hard requirement #1.

**Offline smoke (job 1763977) against the 2-epoch smoke checkpoint — ALL PASSED:**
loaded 9,493,695 params (n_obs_steps 5, horizon 9); `act()` -> (5,4,7); actions inside [-1,1]
(Tanh head); **a wrong history length was REJECTED loudly**; deterministic to `max|a1-a2| = 0.0`.

**Two traps found and closed while wiring it, both of the silent-wrong-answer kind:**
1. **History SPACING.** Their policy needs 5 frames at ONE-env-step spacing. Buffering across
   `act()` calls would space them `act_steps`(=4) env steps apart — a train/eval mismatch that
   looks like "the architecture failed". Fix: `cond_steps: 5` so the VENV stacks them at the right
   granularity (the same mechanism our arms already rely on), plus an assert in `GapArchPolicy.act`.
2. **NORMALIZATION guard silently skips for F.** `dppo_eval.sbatch` derives the expected dataset
   from the checkpoint's `.hydra/config.yaml`; arm F has none, so the `-f` test fails and the guard
   is bypassed — while the venv still uses `normalization_path` for obs/action (de)normalization.
   Fix: `GapArchEvalAgent.__init__` re-asserts it against `task.name` stored in the snapshot's own
   cfg. **A guard that is skipped rather than failed is worse than no guard**, because the log line
   confirming it was never printed and its absence is easy to miss.

**THE SLOPE NEEDS THE WIDTH DUMP — it does NOT come from episodes.csv.** `decompose_width.py`
regresses `width_cmd_mm` / `ee_z_m` from `.agent_tmp/<tag>_widthcmd_b*.npz`, written by the policy
adapter. `GapArchPolicy` now writes them, with the mm conversion COPIED character-for-character
from `eval_agent._DiffusionPolicy._flush_dump` (verified by a comment-stripped string compare:
normalized -> derive space via action_min/max -> mm x88). Without `GM_WIDTH_DUMP`, arm F yields
success + stress but NO slope — i.e. nothing on this phase's target metric.

**Launch recipe (once F finishes):**
```
GM_WIDTH_DUMP=armF \
GM_WIDTH_NORM=$REPO/dataset/dppo/single_lift_generalist_3obj/normalization.npz \
CKPT=<run dir>  NORM=$REPO/dataset/dppo/single_lift_generalist_3obj/normalization.npz \
CFG_DIR=$REPO/gentle_manip/dppo/cfg/single_lift_mushroom_simreal_realws_noos_cmd_v32 \
CFG_NAME=eval_gap_arch  SIM_EXPERIMENT=single_lift_mushroom_soft_eval \
GM_EXTRA_OVERRIDES="model.snapshot=<run dir>/snapshot_100.pth" \
sbatch gentle_manip/scripts/arrhenius/dppo_eval.sbatch
```
Then the slope: add `<tag>=<eval dir>` to `.agent_tmp/arms.txt` and run `decompose_width.py`.

**Second eval smoke (job 1764143) covering the dump — PASSED:** `width_cmd_mm (12,5)` in
[54.2, 57.3] mm, `ee_z_m` in [-0.002, 0.269] m, both physically plausible for an 88 mm gripper
(the VALUES are meaningless — a 2-epoch checkpoint on random input — the PLUMBING is what was
verified, including the 0-88 mm range assert).

### 2026-08-28 — PRE-REGISTERED predictions for arm F (written BEFORE it finished; epoch 81/101)

Recorded in advance so the reading is not fitted afterwards — the same discipline as the
matched-mean floor experiment.

**[SUPERSEDED — see the retraction entry below: this check scores 1.000 while a trivial copier scores 0.998, so it is uninformative. Repaired by an autoregressive rollout.]**

**First check is OFFLINE, not the sim eval** (`.agent_tmp/armF_closure_check.py`): does F's
commanded gripper width track the demonstrator's closing ramp on the 138 HELD-OUT val.npz
trajectories it never trained on? This is asked first because the campaign's hardest-won lesson is
that **a regression/eps loss does not measure action quality** — arm D had the BEST val loss of any
arm and never closed (0/21). F's train loss is ~7e-5, which by itself means nothing.

**Priors:** F's head is a UNIMODAL deterministic Tanh MLP, not a diffusion model. Mode-averaging
was already RULED OUT as the blocker here (closure is a smooth ramp, not a multimodal choice), so
a regressor should be *able* to represent closure. F also gets ~200k gradient steps vs their ~79k.

**The three outcomes and what each licenses:**

| outcome | reading | consequence |
|---|---|---|
| **1. F closes, success in a normal band** | their architecture works on our data; GAP's effect is then measurable INSIDE their stack | compare F(GAP) against a GAP-off F to isolate GAP within their architecture; architecture is not the blocker |
| **2. F never closes** (travel ratio < 0.6, like arms A-D) | the "approach, never close" failure is NOT specific to our diffusion policy — it reproduces in a completely different architecture on the same data | strong evidence the cause is the DATA/objective or the proprio-vision balance, not our model. Would also retire "our architecture is the problem" |
| **3. F closes but its width-vs-size slope stays flat (~0.1-0.2, like lulkx's 0.17)** | **the most informative outcome**: size-blind grasping survives a total change of architecture, head, trainer and loss | the defect is architecture-INDEPENDENT -> it lives in the data/objective. That would make "more/better architecture" a dead end and point the whole phase at conditioning or supervision instead |

**Prediction (stated, so it can be wrong):** outcome 3 is most likely. Every mechanism tried so far
lands on one curve where grip LEVEL explains 95-98% of stress and size explains ~1% of width; that
pattern has survived CFG, floors, aux heads, encoders and a 2x2 object/training swap, which is the
signature of a data property rather than a model property.

**Guard against over-reading F either way:** its compute is not matched to theirs (2.5x their
gradient steps), its LR schedule is their per-batch cyclic one, and their valid split is
contaminated — so F is a test of ARCHITECTURE on our data, NOT a reproduction of their paper's
numbers, and must not be reported as one.


### 2026-08-28 — RETRACTED: my arm-F "closure is learned" check was UNINFORMATIVE (a copier scores 0.998)

The offline closure check I pre-registered returned `corr(pred,true) = 1.000` (p10 1.000, p90
1.000), travel 48.6 vs 48.5 mm, and printed "VERDICT: ramp reproduced — closure is learned".
**That verdict is WITHDRAWN. The check cannot distinguish learning from copying.**

**Measured, not argued (job 1765211).** A trivial copier — "command exactly the width you currently
observe" — scores **corr 0.9982 (p10 0.9981, p90 0.9988), mean |error| 0.48 mm** on the same 30
held-out episodes. Arm F scored 1.000. **F beat a no-op baseline by 0.002 correlation.**

**Why the check was flawed, and it was foreseeable:**
1. The policy OBSERVES its current gripper width, actions are ABSOLUTE, and closure is a smooth
   ramp — so `action[i] ~= observed_width[i]` and copying is near-optimal. **This exact leakage is
   already in this DEVLOG** ("feed 80 mm -> predicts 79.4; feed 28 -> 28.6... a copier cannot DRIVE
   the channel"), documented for the width head. I wrote a check vulnerable to the same thing.
2. It was TEACHER-FORCED: ground-truth width fed at every step, which is precisely the condition
   under which a copier looks perfect and the closed-loop failure (arms A-D) is invisible.

**LESSON (generalises past this arm): any predictive check on a channel the policy also OBSERVES
must be scored against the copy baseline, not against zero.** A high correlation means nothing until
the trivial predictor's score is known. Cheap rule: report `corr_policy` next to `corr_copier`
always; if the gap is not large, the check is uninformative by construction.

**Repair (job 1765229, `.agent_tmp/armF_autoreg_width.py`):** AUTOREGRESSIVE rollout — the observed
gripper channel is replaced by the policy's OWN previous command, so a copier stalls at the open
width while a driver ramps down. Arm pose and cloud stay ground-truth, so it isolates the width
channel and is NOT a full closed loop (only the sim eval is). The action-space -> obs-space affine
is validated first by round-tripping true actions against the achieved next-step width, because
that conversion is the B10/B17 trap.

**Unchanged:** F trained cleanly (2:00:30, final loss 5.0e-5) and the pre-registered outcome table
still stands — it simply cannot be adjudicated by the retracted check.

### 2026-08-28 — arm F: DRIVES closure (repaired check), and a CONFIG DEFAULT change for eval geometry

**1. Autoregressive width rollout (job 1765229) — the repair of the retracted check.**
The affine chain was validated first: commanded width vs ACHIEVED next-step width `mean|d| = 0.30 mm`.
Then, with the observed gripper channel replaced by the policy's OWN previous command:

| | policy | demonstrator |
|---|---|---|
| travel (max-min) | **64.8 mm** | 48.6 mm |
| final / min width | **15.2 mm** | 31.4 mm |

travel ratio **1.33**. **This check DISCRIMINATES** (a copier stalls near the open width at ratio
~0), so the conclusion "arm F drives the width channel, it is not a copier" is supported — unlike
the retracted teacher-forced version.

**DO NOT yet read the 16 mm over-closure as a finding.** In this rollout the arm pose and point
cloud stay GROUND-TRUTH while the width diverges, so proprio drifts off-distribution and errors
compound; over-travel is an expected artefact of that setup. It is a FLAG to check against the
closed-loop eval, not a measurement. (If it survives closed-loop, it is the over-squeeze failure
mode this whole campaign started from — v33b over-squeezing real mushrooms.)

**2. CONFIG DEFAULT CHANGED: `eval_gap_arch.yaml` scene_group_size 4 -> 1.** Caught while F's first
eval was already running: at 200 eps / 5 envs = 40 batches, `sgs=4` yields **10 distinct
geometries**, but the width-slope rule requires **>=40** (a 12-geometry sample reversed three
verdicts in this campaign). Cancelled 1765317 at ~10 min, relaunched as 1765398 with sgs=1.

**Verified against the established arms rather than assumed** — their `summary.json` reports
`scene_group_size: None`, which is unusable, so the geometry count was recounted from
`episodes.csv`: every `lulkx/eval/slope_*` arm has **40 batches / 40 unique obj_scale**. F now
matches that protocol exactly, and the same protocol will be used for C'/D'/E so all four arms are
comparable. The default lives in the CONFIG (with the rationale inline) rather than in an env var,
so it cannot be forgotten on the next launch.

**Also documented in that config:** do NOT set `EVAL_SUBDIR` with it — the sbatch derives its subdir
as `dirname(dirname(CKPT))`, which assumes `<run>/checkpoint/state.pt`; arm F's flat
`<run>/snapshot_N.pth` layout would send the output to `logs/gap_arch/eval/...`, outside the run
dir. Pass `logdir=` in `GM_EXTRA_OVERRIDES` instead. (`eval_base` itself handles the flat layout
correctly — it returns `p.parent` when the parent is not named `checkpoint`.)

**Eval-protocol note for the four-arm table:** xaqnb's `gen3u_*` baselines are **60-episode**
screens (success 0.883 mushroom / 0.867 tofu / 0.583 raspberry). Our own rule says 60-ep probes run
optimistic by 0.05-0.10 on strong arms, so F's 200-ep number must NOT be differenced against them
directly; either re-run xaqnb at 200 or label the protocol gap explicitly.

### 2026-08-28 — PRACTICE CHANGE (user mandate): every eval records commands + observations

**User:** *"if recording command and obs is useful, you should really remember to do it for all
future runs."* Implemented as a DEFAULT, not a habit: `dppo_eval.sbatch` in BOTH the main repo and
the `gm_generalist` worktree now sets `GM_WIDTH_DUMP`, `GM_OBS_DUMP` and `GM_WIDTH_NORM`
automatically (tag `<EVAL_SUBDIR>_<jobid>`). `GM_OBS_DUMP_CLOUD=1` remains opt-in (clouds are
large); `GM_DUMP_OFF=1` disables.

**Why this was costing us:** aggregate metrics cannot diagnose BEHAVIOUR. The arm-F approach-offset
was spotted only because the user watched render frames, and diagnosing it needed a SECOND eval
(job 1765652) launched purely to capture obs. The width slope is likewise computed from dumps, not
from `episodes.csv`.

**Trap for any NEW policy adapter:** the dump lives in the *Policy* class, not the harness, so a new
adapter starts with none — arm F's `GapArchPolicy` initially had neither the width nor the obs dump,
which would have made it uncomparable on this phase's target metric. Adding a Policy adapter now
means adding both dumps.

### 2026-08-28 — GAP's proven config uses a DETERMINISTIC head, not diffusion (user asked)

Their code ships BOTH heads — `cfgs/policy/head/dmm.yaml` (`DeterministicMLP`, Tanh MLP, MSE) and
`cfgs/policy/head/diffunet.yaml` (`ConditionalUnet1D`, diffusion, `num_diffusion_iters: 5`). But the
GAP entrypoint chains `gap.yaml -> policy: bc_policy -> head: dmm`, so **their proven GAP setting is
the NON-diffusion head**, which is what arm F ran.

**User's inference — a deterministic head is unfriendly to our multimodal data — fits the evidence
better than my covariate-shift explanation.** Measured on held-out data, F's one-step teacher-forced
mapping is FAITHFUL: slope(pred~true) = 1.00, variance preserved, bias 1.2 mm in x (2% of the
62.4 mm spread). So the mapping is not shrunk. But one-step prediction gets the full history, which
DISAMBIGUATES which approach the arm is already committed to; closed-loop from the start, an MSE
regressor averages competing valid approaches and commands something between them — a systematic
offset, exactly the render artefact observed.

⚠ **Correction to an earlier note in this DEVLOG:** "mode averaging — premise refuted (closure is a
smooth ramp)" was established for the WIDTH channel only. It does NOT extend to the APPROACH POSE,
where multiple valid grasp azimuths genuinely exist in our CMA-ES demos. Mode averaging is live again
for pose.

**Test (arm F2): same everything, `head: dmm -> diffunet` — THEIR OWN diffusion head, their own
config.** If the offset disappears, the head is the cause and "DP matters for our multimodal data"
is demonstrated inside their codebase; if it persists, the cause is elsewhere.

### 2026-08-28 — ARM F FAILS CLOSED-LOOP: 0/20, auto-killed as degenerate. Approach POSE, not closure.

`armF_mushroom_canon` was killed by the degenerate guard: **success 0.00 over 20 episodes**
(batches 1-4 all 0.00; one batch reached `ever=0.20`, i.e. it briefly lifted then dropped, so the
pipeline is functional, not flailing). **The user spotted the cause first, from the render frames:
the approach position is consistently offset from the object in -x.**

**ADAPTER BUGS RULED OUT** (each checked, not assumed):
| candidate | status |
|---|---|
| normalization mismatch | guard PASSED (`normalization OK: single_lift_generalist_3obj`) |
| obs history length | asserted at every step; `cond_steps=5` = their `n_obs_steps` |
| history ORDER / padding | `genesis_venv.py:114` left-pads with the EARLIEST obs (matches training's clamp-to-start); `_stacked` takes the last 5, newest last |
| one-step action mapping | faithful on held-out data: slope(pred~true) **1.00**, variance preserved, x bias **1.2 mm** (2% of the 62.4 mm spread) |
| can it close the gripper at all | YES — autoregressive rollout drives closure (travel ratio 1.33) |
| shift9 / point-cloud x bias | real-backend only; the generalist set is unshifted, so sim train/eval agree |

**So the failure is not "approach, never close" (arms A-D) — F CLOSES, but approaches the WRONG
PLACE.** That distinction matters: it points at the approach POSE distribution, not the width
channel.

**Leading explanation — the user's, and it fits every measurement:** their GAP default head is
`DeterministicMLP` (Tanh MLP + MSE), NOT diffusion. Our CMA-ES demos contain MULTIPLE VALID grasp
azimuths per scene. One-step teacher-forced prediction is faithful because the observed history
already reveals which approach the arm is committed to; closed-loop FROM THE START that context is
absent, and an MSE regressor commands the AVERAGE of the valid modes — a pose that belongs to none
of them. A constant single-axis offset is exactly what mode-averaging over a roughly bimodal
approach distribution looks like.

**PRE-REGISTRATION check:** this is closest to outcome 2 ("F never closes"), but the pre-registered
wording was wrong in detail — F fails on APPROACH POSE, not on closure. Recording the mismatch
rather than reinterpreting the prediction to fit.

**DECISIVE TEST — arm F2 (job 1765733), already running:** identical to F except
`head: dmm -> diffunet`, i.e. THEIR OWN `ConditionalUnet1D` diffusion head, one config line, their
config, their code. If the offset vanishes and success recovers, "a deterministic head cannot
represent our multimodal approach distribution" is demonstrated INSIDE their codebase. If it
persists, the cause is elsewhere and the deterministic head is exonerated.

⚠ **Do NOT yet report "their architecture fails on our data".** F confounds head (deterministic) with
everything else in their stack. F2 separates them, and it is the comparison that licenses any claim.

### 2026-08-28 — LESSON: a dependency STUB can break OTHER packages. Five jobs lost to it.

Arm F2 (their diffusion head) took **five failed jobs** to launch, all traceable to ONE decision of
mine: stubbing `torchvision` instead of installing it.

| attempt | failure | real cause |
|---|---|---|
| 1765733 | `diffusers.DDPMScheduler stub called` | my `try: import diffusers / except: stub` **silently** swallowed a real import error and degraded |
| 1765758 | `ValueError: torchvision.__spec__ is None` | bare `ModuleType` stubs have no `__spec__`; diffusers probes via `importlib.util.find_spec` |
| 1765789 | `Could not import module 'PreTrainedModel'` | `transformers` (INSTALLED, 5.16.1) imports torchvision — **my fake broke it**, so diffusers' loader chain failed |
| 1765830 | `cannot import name 'HfFolder'` | their pinned `diffusers==0.11.1` needs old `huggingface_hub`; ours is 1.28.0 and `transformers` REQUIRES >=1.5.0, so downgrading was not safe |
| 1765875 | — | **fixed**: real `torchvision==0.21.0` (the build paired with torch 2.6.0), `--no-deps`, abort-if-torch-moves. torch identical before/after. All stubs now inert. |

**THE LESSON:** a stub is not a local decision. Substituting a fake module for a real dependency
changes the import graph for **every other package that depends on it** — here `transformers`, which
was installed and working, and which diffusers needs. Each subsequent error looked like a NEW
problem and was actually the same one.

**RULES ADOPTED:**
1. Prefer installing the REAL, version-matched package with `--no-deps` over stubbing, and verify the
   critical pin (here torch) is byte-identical before/after. Stub only a leaf nothing else imports.
2. **Never let `except: <install stub>` swallow the reason.** Print the exception and REFUSE to run
   when the stub cannot satisfy the requested mode (`--head diffunet` now hard-fails on a stub).
3. **Test under the deployment environment, not a clean one.** My first diffusers check passed
   because it ran with NO stub present — it validated a configuration we never run. The re-test
   reproduces the runner's stub environment exactly. Same class of error as the teacher-forced
   closure check: verifying under conditions that differ from the real ones.

`third_party/GAP/VENDORED.md` difference 11 updated: we run `diffusers 0.35.1` + `torchvision
0.21.0` (torch 2.6.0), not their `0.11.1`/`0.16.0` (torch 2.1.0). Behaviourally equivalent for our
use — their scheduler is fully pinned by explicit constructor args, and the resulting betas match
exactly: `[0.101294, 0.279544, 0.473635, 0.724052, 0.999]`, timesteps `[4,3,2,1,0]`.

### 2026-08-28 — ARM F2 LAUNCHED (job 1765979): their DIFFUSION head, one line changed from F

F2 = arm F with `head: dmm -> diffunet`, i.e. THEIR `ConditionalUnet1D` (5 diffusion iters) instead
of THEIR `DeterministicMLP`. Same encoder, same trainer, same GAP rule, same data, same proven
params. Smoke passed (loss 1.106 -> 0.984; GAP damping identical at 52/112 tensors, confirming the
encoder is untouched).

**WHY:** F's failure is a **-28.8 mm median lateral miss** (mean -33.8, sd 19.1, NEGATIVE in 20/20
episodes, lowest ee_z 3.1 mm so the descent is correct). Offline the same policy has a 1.2 mm bias
at slope 1.00 — so this is a CLOSED-LOOP behaviour, not a frame bug. The user's explanation: a
unimodal Tanh+MSE head cannot represent our multimodal approach-pose distribution and commands a
pose belonging to no mode.

**CAVEAT TO STATE WITH ANY F2 RESULT — it is not a clean single-variable test.** Their diffusion
head is **56,892,167 params**, against F's ENTIRE model at 9,493,695. Capacity moves with the head.
So a positive F2 licenses "their diffusion head fixes it", NOT "diffusion per se fixes it".
A capacity-matched control would be needed for the stronger claim.

**Seed-matched context (the strongest evidence so far):** xaqnb (our PointNet DIFFUSION policy)
scores **1.00 on seeds 4200128 and 4200129 — the exact seeds where F scored 0.00**. So the task and
data are learnable; what differs is the policy. xaqnb vs F confounds encoder+head+trainer+loss,
which is precisely what F2 separates.

### 2026-08-28 — ARM F's REAL DEFECT: the grasp position IGNORES the object. corr(ee_x, obj_dx) = 0.087

**CORRECTION FIRST.** I reported F's miss as "median -28.8 mm in x, 20/20 negative". **That number is
WITHDRAWN as a precise figure** — it used a point-cloud proxy for the object (median x of the
lowest-20% z points) that is CONTAMINATED: at the grasp the gripper itself is at table height and
contributes low-z points, dragging the estimate toward the arm. The t=0 variant is worse still —
with the arm at home, the lowest-z points are dominated by the TABLE, not the object, which is why
it produced an implausible -106 mm. Neither proxy localises the object. The qualitative observation
(a consistent negative x offset, which the USER spotted in the renders) stands; the magnitude did not.

**GROUND TRUTH instead.** `episodes.csv` records `obj_dx`, the object's actual x displacement.
Paired with the obs dump over the same 20 episodes:

| quantity | value |
|---|---|
| `obj_dx` range | **-174.6 to -12.2 mm** (sd 46.5 mm) — the object really does move a lot |
| ee_x at grasp | median 350.0 mm (home 454.4 mm) |
| **corr(ee_x@grasp, obj_dx)** | **0.087** |

**F grasps at a nearly FIXED x regardless of where the object is.** That single number explains
0/20 success completely, and it is far stronger than the offset framing: the failure is not a biased
approach, it is an approach that does not use the object's position at all.

**This is the PROPRIO-SHORTCUT pathology on the POSE channel** — the same failure mode this whole
campaign has been chasing on the WIDTH channel ("the head learns to COPY the observed width"; a
copier cannot DRIVE the channel). Corroborating: the commanded x is essentially the CURRENT x
(delta -1.91 mm during approach, **+0.11 mm at the grasp** = fully stalled), while the arm drifts
-92.7 mm by accumulating those small deltas until they vanish.

**AND IT IS EXACTLY THE PATHOLOGY GAP EXISTS TO FIX** — vision-proprioception policies failing by
over-relying on proprioception. Arm F runs GAP (lambda 0.3, epochs 0-50) and still exhibits it in
the extreme, which is itself a result about GAP's efficacy at our scale.

**METHOD LESSON (third instance today):** a derived proxy needs validation against ground truth
BEFORE its numbers are quoted. The cloud-based object estimate looked reasonable, produced a
plausible -28.8 mm, and was wrong. Ground truth (`obj_dx`) was in `episodes.csv` the whole time.
Same family as the copy-baseline lesson: always ask what the trivial/known-correct reference gives.

**OPEN — needs the same measurement on a policy that WORKS:** xaqnb's eval (1765506) was launched
before dumps became default-on, so it has a width dump but NO obs dump; corr(ee_x, obj_dx) cannot be
computed for it yet. Any future eval gets it automatically. Expectation: a working policy shows a
high correlation. Without that pairing, the 0.087 is descriptive of F, not yet a contrast.

### 2026-08-28 — ⚠ RETRACTED: "arm F ignores the object" (corr 0.087). THE PAIRING WAS BROKEN.

**The user rejected the conclusion from a render (`armF_diag20/render/batch02_env1.mp4`), saying the
arm plainly approaches different positions depending on the object. The user was right.**

**How it was caught:** validate the dump<->episodes.csv pairing against a column whose answer is
KNOWN. The arm's initial pose must equal home + `home_d{x,y,z}`:

| check | measured r | required |
|---|---|---|
| ee_x@t0 vs home_dx | **-0.217** | ~ +1.0 |
| ee_y@t0 vs home_dy | +0.141 | ~ +1.0 |
| ee_z@t0 vs home_dz | +0.110 | ~ +1.0 |

Standard deviations match EXACTLY (11.1 vs 11.1 mm) => right values, WRONG ORDER. A permutation
error, not noise. **Everything derived from that pairing is withdrawn:** `corr(ee_x, obj_dx)=0.087`,
"grasps at a nearly fixed x", and the "proprio shortcut on the pose channel" story built on it.

**ALSO WITHDRAWN — every cloud-derived object position.** `object_focus` keeps points near the EE,
so at t=0 the above-table points are THE ARM: `cloud_obj_x` mean 473.9 sd **10.7** mm against the
arm's home 454 mm, while the true `obj_dx` sd is 46.5 mm. So the -28.8 mm and -106 mm "gaps" were
both measuring the gripper against itself.

**WHAT STILL STANDS (independent of the pairing):**
- F's closed-loop success is **0/20** (auto-killed as degenerate) — from the harness, not my analysis.
- Commanded x minus CURRENT x is -1.91 mm on approach and **+0.11 mm at the grasp** — per-episode,
  no cross-file pairing involved. The policy does stop commanding motion.
- Offline one-step mapping is faithful: slope 1.00, bias 1.2 mm.
- **xaqnb scores 1.00 on seeds where F scores 0.00** — both from harness logs.
- The user's render observation: the approach is visibly offset, and it DOES vary with the object.

**FIX:** the obs dump now records `dump_batch` as an explicit pairing key. The deeper rule, which I
violated: **never join two data sources on an ASSUMED correspondence — validate the join against a
column with a known answer first.** `home_d{x,y,z}` is the perfect probe and cost one command.

**This is the third proxy failure today** (copy-baseline corr, cloud object position, this join), all
the same shape: a derived quantity that looked plausible, quoted before being checked against a
known-correct reference.


### 2026-08-28 — `docs/CHECKLISTS.md` created (user request)

A read-BEFORE-acting companion to this DEVLOG: the DEVLOG records what happened, CHECKLISTS records
what to do so it does not happen again. Sections scaffolded for launching training / evaluation /
analysis / common practices (to be filled in as the user directs); the **Common Mistakes** section
is populated now with this session's errors, each as a short paragraph with the artifact that
proves it (job ids, dump paths, eval dirs).

Recorded in persistent memory as a standing instruction: read it at the start of a task — before a
launch, before an eval, and before quoting any number — not afterwards.

### 2026-08-28 — ADOPTED PRACTICE: the "teaser eval" (user's term and design)

A cheap degeneracy screen on a MID-training checkpoint so a dead run can be killed from footage
instead of running to completion. **15 episodes = 3 batches x 5 envs, `scene_group_size=1` (fresh
geometry every batch), EVERY episode rendered, actions+observations+clouds dumped.** Same checkpoint
epoch across arms so teasers are mutually comparable. It is a SCREEN, not a measurement.

First use: jobs 1766403/4/5 on C'/D'/E at `state_150` (of 350) ->
`<run>/eval/teaser_e150/`. Full protocol in `docs/CHECKLISTS.md` section 2.1.

**Caught while setting it up:** the eval config does NOT carry training-time architecture flags.
C'/D'/E trained with `proprio_encoder: true`; without
`+model.network.proprio_encoder=true` the eval builds a DIFFERENT network from the checkpoint.
All three teasers were held until the log printed `[PointNetDiffusionMLP] proprio_encoder=True`.
Generalised into the checklist: an eval must be verified to rebuild the TRAINED architecture.

**Not teasered:** arm F2 at epoch ~10/101 — too early; an undertrained diffusion policy could look
degenerate and trigger a false kill. Its teaser waits for ~epoch 50.

### 2026-08-28 — TEASER RESULT: C' and D' are DEGENERATE and were KILLED. E WORKS. GAP's phase guidance earns its place.

Teasers at `state_150` (15 eps, 3 batches, sgs=1, all rendered). **The user read the footage first:
C'/D' approach the object correctly but NEVER CLOSE; E looked reasonable.** The numbers agree:

| arm | mechanism | success | ever | sustained |
|---|---|---|---|---|
| **C'** `smyoy` | encoder + UNIFORM 0.7 damping | **0.000** | 0.000 | n/a |
| **D'** `ylbjq` | encoder, NO damping (control) | **0.000** | 0.000 | n/a |
| **E** `bwmcy` | encoder + their SPARSE phase-guided rho | **0.800** | 0.800 | 22,641 |

**C' (1762867) and D' (1762868) CANCELLED at 4h36m**, freeing two GPUs. Checkpoints
`state_{50,100,150}.pt` retained as evidence.

**IT IS NOT A BUG — and the proof is E itself.** All three teasers ran the SAME eval config,
normalization, checkpoint-loading path, `+model.network.proprio_encoder=true` override, sim
experiment and seeds. A harness or adapter defect would have taken E down too. E grasps at 0.800.

**THE RESULT:**
- **D' (encoder, no damping) -> degenerate.** Adding the proprio encoder ALONE breaks closure,
  reproducing the earlier A-D "approach, never close" failure at 0/21.
- **C' (encoder + uniform 0.7 damping) -> degenerate.** Blunt damping does not rescue it.
- **E (encoder + their sparse phase-guided damping) -> 0.800.** The modulation does something real.

**This is a POSITIVE result for GAP's mechanism**, and it is the arm that only exists because the
split-specific phase-file bug was found and fixed this morning — the faithful variant turned out to
be the one that matters.

⚠ **CONFOUND, not yet separable:** E differs from C' in BOTH targeting AND average strength (C' is
0.700 on every batch; E averages 0.817 and sits near 1.0 most of the time). "Phase targeting matters"
and "gentler damping matters" both explain the contrast. A uniform-0.817 arm would separate them.

⚠ **SCOPE:** C'/D' are shown degenerate **at epoch 150**, not at 350. Accepted because the earlier
proprio arms held this failure mode to their FINAL checkpoints and val loss provably does not predict
closure — but the claim must be stated at epoch 150, not extrapolated.

### 2026-08-28 — E's MULTI-OBJECT teaser: strong on mushroom, weak on raspberry (the user's recommendation paid off immediately)

Same checkpoint (`state_150`), same protocol (15 eps / 3 batches / sgs=1 / all rendered):

| object | demonstrator width | E success | E ever | sustained | xaqnb reference (60-ep, FINAL ckpt) |
|---|---|---|---|---|---|
| mushroom | ~33 mm | **0.800** | 0.800 | 22,641 | 0.883 |
| tofu | ~42 mm | **0.467** | 0.533 | 9,289 | 0.867 |
| raspberry | ~15 mm | **0.133** | 0.133 | 15,957 | 0.583 |

**The user's recommendation to teaser MORE THAN ONE OBJECT paid off on its first use:** a
mushroom-only screen would have reported 0.800 and hidden a 6x spread across objects. Adopted into
`docs/CHECKLISTS.md` 2.1.

**NOT degenerate anywhere** — raspberry is weak but non-zero (2/15), so the screen's verdict is
"let it finish", not "kill".

**Do NOT difference these against the xaqnb column.** Those are 60-episode runs on a FINAL
checkpoint; these are 15-episode screens at epoch 150/350. Screen-vs-measurement AND mid-vs-final —
comparable VERDICTS only, never numbers. The canonical 200-ep/40-geometry eval is what settles it.

**Hypothesis worth testing later, not now:** success tracks object SIZE more than training-set share
(raspberry is both the smallest at ~15 mm and the least represented at 200 demos vs tofu's 587 —
yet tofu, the LARGEST, is mid-pack, so share alone does not explain the ordering). Relevant to the
width-vs-size question this phase is about.

### 2026-08-28 — BASELINE at the matched protocol: the generalist SUCCEEDS (0.915) with ZERO width adaptation (slope 0.02)

`xaqnb/eval/gen3u_mushroom_200geo40` — 200 episodes, **40 distinct geometries**, sgs=1, mushroom.

| metric | value |
|---|---|
| success | **0.915** (ever 0.920) |
| SUSTAINED stress | 35,167 |
| peak | 54,673 |
| **width slope** | **0.02, 95% CI [-0.21, +0.25], R2 0.00, 2% of demonstrator** |

**THE HEADLINE FOR THIS PHASE: a policy can be highly successful and completely size-blind.** The CI
includes zero, so there is NO detectable width adaptation; the upper bound rules out anything above
~23% of the demonstrator's 1.08. The mushroom SPECIALIST (lulkx) measured 0.17 [0.00, 0.33] on the
same protocol, so the generalist is, if anything, LESS adaptive — consistent with the earlier 2x2
finding that size sensitivity is driven by the OBJECT, not by multi-object training.

**This is now the reference every GAP arm must be read against**, and it reframes what a "win" is:
beating 0.915 on success is NOT the goal (that ceiling is already reached); moving the slope off
0.02 without losing success is.

**METHOD NOTE — short-probe bias is NOT reliably signed.** The 60-episode screen of this same
checkpoint gave 0.883; the 200-episode run gives **0.915**. Our protocol note says 60-ep probes run
OPTIMISTIC by 0.05-0.10 on strong arms; here it was PESSIMISTIC by 0.03. So a short probe cannot be
corrected by assumption — only matched protocols are comparable. (Also: the old `gen3u_mushroom`
had 12 geometries and was never valid for a slope at all.)

### 2026-08-28 — CAVEAT on reading arm E: GAP's window is a fixed EPOCH RANGE, so the modulated FRACTION differs

`modulation_starts=0, modulation_ends=50` in their `WorkSpace.train()` is an absolute epoch range,
not a fraction. Our runs differ in length, so the share of training that is actually modulated does
too:

| run | epochs | GAP window | fraction of training modulated |
|---|---|---|---|
| their proven setup | 101 | 0-50 | **50%** |
| arm F / F2 | 101 | 0-50 | **50%** |
| **arm E (the one that WORKS)** | **350** | 0-50 | **14%** |

**Consequences.** (1) E is not running GAP at their proven proportion — it gets ~1/4 the relative
dose, and still rescues an arm that is degenerate without it (D' = 0.000), which if anything
strengthens the mechanism claim. (2) Any statement of the form "E used their proven GAP settings"
must say *their proven lambda and epoch window*, NOT their proven proportion. (3) A dose-response
arm (window scaled to 0-175 = 50% of 350) is the clean follow-up if the mechanism is worth pursuing.

Same trap as their per-batch `CosineAnnealingLR` with `T_max=cfg.epoch`: a hyperparameter defined in
absolute units behaves differently when the run length changes. **Check every "proven" setting for
whether it is absolute or relative before porting it to a differently-sized run.**

### 2026-08-28 — CAPACITY-MATCHED control for the head comparison (F3): measured, feasible at 1.12x

User asked whether to raise F2's capacity. **Direction is the opposite** — F2 is already the LARGE
arm, and its weak teaser (0.067 at epoch ~48/101) is better explained by being half-trained. Adding
capacity would deepen the confound, not resolve it.

Measured head sizes (job 1768510), encoder is 8,821,376 params in every case:

| head | head params | x MLP | TOTAL model | vs arm F total |
|---|---|---|---|---|
| DeterministicMLP (arm F) | 672,319 | 1.0x | **9.49 M** | — |
| ConditionalUnet1D [256,512,1024] (arm F2) | 56,892,167 | 84.6x | 65.7 M | **6.9x** |
| ConditionalUnet1D [64,128,256] | 7,502,471 | 11.2x | 16.3 M | 1.7x |
| **ConditionalUnet1D [16,32,64]** | **1,825,127** | 2.7x | **10.65 M** | **1.12x** |

**Proposed arm F3 = F2 with `down_dims=[16,32,64]`** — total model within 12% of arm F, because the
shared ENCODER dominates and the head ratio barely moves the total. Config-only (`down_dims` is
already a `ConditionalUnet1D` constructor kwarg), so their code stays untouched.

**Reads:** F3 grasps where F never did => capacity excluded, HEAD TYPE (unimodal MSE vs diffusion)
is the cause. F3 fails like F => capacity was doing the work.

Held until F2 finishes and E's canonical eval lands — a control should not be designed against a
half-trained 15-episode screen.

### 2026-08-28 — VERDICT ON GAP: it rescues a degenerate arm, but does NOT beat the baseline and does NOT touch width adaptation

Canonical protocol, mushroom, 200 episodes, **40 distinct geometries**, sgs=1 — E vs the matched baseline:

| | success | ever | SUSTAINED | peak | intercept | **width slope** | R2 |
|---|---|---|---|---|---|---|---|
| xaqnb baseline | **0.915** | 0.920 | 35,167 | 54,673 | 27.3 mm | 0.02 [-0.21, +0.25] | 0.00 |
| **E (GAP, their CPD+LSTM rho)** | 0.840 | 0.915 | **30,970** | 53,784 | 28.4 mm | **0.03 [-0.18, +0.23]** | 0.00 |

**1. GAP DOES NOT IMPROVE WIDTH ADAPTATION.** 0.02 -> 0.03 with near-identical CIs and R2 0.00 both.
The defect this phase exists to fix is completely untouched.

**2. GAP COSTS SUCCESS: -0.075** (0.915 -> 0.840) against the plain generalist.

**3. ITS -12% SUSTAINED STRESS IS A PURE LEVEL EFFECT, NOT ADAPTATION.** E's intercept is 1.1 mm
WIDER (28.4 vs 27.3). At the campaign's measured -1,714 to -3,468 Pa/mm, 1.1 mm predicts ~3,300 Pa;
the observed drop is 35,167-30,970 = **4,197 Pa** — same order. E sits on the SAME level-vs-stress
curve as contact-stop, freeze, floor and baseline. Confirmed by the `ever - success` gap: **0.075 for
E vs 0.005 for the baseline** — E lifts about as often as the baseline SUCCEEDS, then drops more,
the signature of a looser grip.

**THE TWO-SIDED CONCLUSION:**
- **POSITIVE for GAP as a mechanism.** It rescues an arm that is degenerate without it: D' (encoder,
  no damping) = 0.000 and C' (uniform damping) = 0.000, both at epoch 150, while E reaches 0.840.
  Phase-guided modulation demonstrably does something real.
- **NEGATIVE for GAP as a solution to OUR problem.** The proprio-encoder + GAP direction produces a
  policy that is WORSE than the plain generalist on success and IDENTICAL on size-blindness. The
  encoder introduces a pathology; GAP repairs it; the net result is below where we started.

**IMPLICATION FOR THE PLAN: close the proprio-encoder/GAP line.** It answered its question — proprio
over-reliance is real and modulation fixes the degeneracy it causes — but it does not advance
"grasp width explainable by object size", and it cannot beat 0.915/0.02. Effort should go to
mechanisms that change what the policy CONDITIONS ON (metric-size conditioning, perceived-size in
the category-embedding slot), not to how its proprio gradients are scaled.

**Caveats retained:** E ran GAP at their proven lambda/window but only 14% of its 350 epochs were
modulated (vs their 50%); and C'/D' are shown degenerate at epoch 150, not 350.

### 2026-08-28 — ABSORBED upstream 0ab70b1 (concurrent collection chains): no damage to us, but the final dataset slips ~18 h

Upstream found three `bigchain.sh` collection chains running concurrently for hours: every teardown
used `pgrep -f "[b]igchain\|[c]ollect_demos_synth_v3"`, but **`pgrep` takes an ERE**, so `\|` was
matched LITERALLY, every kill was a no-op, and each relaunch stacked another chain into the same
dataset dirs — running three DIFFERENT code versions (pre-stress, pre-material-DR, current). Fixed
with an exclusive `flock` on `bigchain.sh` plus kills by explicit PID; all partial/stale run dirs
deleted.

**Impact on us:**
1. **No data damage.** Nothing valid had completed and `data.pkl` writes only at completion, so no
   frozen set is affected. Our three generalist sources are untouched.
2. **SCHEDULE: the final collection is ~18 h later than it looked.** Our phase deliverable is THE
   METHOD, so this does not block method work — but any plan that assumed the final dataset soon
   should be re-timed.
3. **We are NOT exposed to the same failure.** Everything here goes through `sbatch`/`scancel`, and
   today's C'/D' cancellations were verified against `sacct` rather than trusting an exit status.
   No `pgrep`-based teardown exists in our path.

**Their two lessons are general and belong in `docs/CHECKLISTS.md` section 1 (launching) when we
fill it in** — noted here so they are not lost:
- `pgrep`/`pkill` take ERE; `\|` silently matches NOTHING rather than erroring, so a kill can report
  success while doing nothing. **Verify a kill by observing state (`ps`/`sacct`), never by exit code.**
- **Long-running background chains need a lockfile, not just kill-before-launch** — kill-then-relaunch
  is only as reliable as the kill.

### 2026-08-28 — ⭐ THE LOSS BUDGET: the size signal is 2% of the objective, and 99.9% of it is a PROPRIO COPY

User asked whether the DEMO and/or the WIDTH ACTION SPACE is the problem. **Measured on the actual
generalist training set** (`dataset/dppo/single_lift_generalist_3obj/train.npz`, 1248 trajs,
254,340 steps; raw npz + `normalization.npz`, no model involved) — three numbers that settle it.

**1. The width action is a near-exact copy of the policy's OWN observed gripper width.**

    regress commanded width (action dim 6, mm) on observed gripper width (state dim 7, mm)
      all frames      R2 = 0.9960   slope 1.0009   intercept -0.61 mm
      post-open only  R2 = 0.9886

**2. Where the copy fails is 14.5% of frames, and NOT where the size decision is.**

| phase | share of frames | copy \|err\| mean | p99 |
|---|---|---|---|
| open (80 mm hold) | 47.5% | 0.54 mm | 0.54 |
| ramp (closing) | 14.5% | **3.26 mm** | 12.76 |
| plateau (the grasp level) | 38.0% | **0.58 mm** | 0.61 |

On the PLATEAU — the frames that carry the size-dependent answer — the copier is accurate to
**0.39 mm rms against a 10.60 mm between-episode signal**, i.e. it captures **99.90% of the
plateau-level energy**. The residual the cloud would have to explain is 0.1%.

**3. The loss budget.** Fraction of the total squared-error energy of the whole 7-dim action target:

| component | share |
|---|---|
| width dim, overall | 31.6% |
| **width dim, BETWEEN-EPISODE level (the size signal)** | **2.07%** |
| ...of which NOT already explained by the proprio copy | **0.1%** |
| **net gradient pressure to read size from the cloud** | **~0.003% of the BC loss** |

**THE DIAGNOSIS: the near-flat slope is not a failure, it is the objective's optimum.** A
size-blind copier is within ~0.003% of the loss-minimising width policy on this data. Nothing on
the INPUT side (aux head, `feed_width_pred`, category embed, GAP) and nothing at INFERENCE (CFG,
floors, contact stop) changes that fraction — which is exactly the pattern every arm has shown.

**AND IT EXPLAINS THE A/B FAILURE.** The gripper-width channel is simultaneously the shortcut AND
the closure clock, so deleting it (arms A/B, 0/21) removes the clock. The shortcut cannot be
removed from the OBSERVATION; it has to be removed from what the TARGET encodes.

**WHY THE ACTION SPACE IS THE ROOT CAUSE (user's hypothesis, supported).** With ABSOLUTE width
commands + own width in obs, the width channel is a TRAJECTORY-TRACKING problem — 94% of its
variance is the object-independent ramp/hold shape — with a proprioceptive shortcut, instead of a
DECISION problem. The single frame per episode where size actually enters (where the ramp stops)
carries no meaningful loss weight.

**THE PROPERTY A FIX MUST HAVE, and the offline test for it.** Relabel the width target so it is
**per-episode CONSTANT from t=0** (the eventual grasp width), and let a rate-limited controller
execute the ramp. Measured on this dataset:

    During the OPEN phase (47.5% of all frames):
      R2(eventual grasp level | observed gripper width) = 0.0000

So a constant target channel has **NO proprio shortcut over 47% of frames**, and 100% of its
variance IS the size signal (vs 2%). The slope then becomes, by construction, the accuracy of a
cloud->width regression — already measured at r = 0.78-0.81 (`docs/width_predictability.md`).

**NOT the same as residual-width v2 (nickq, DECISIVE NEGATIVE).** That relabelled the EXISTING dim
as `command - episode anchor`, which made the APPROACH-phase residual scene-dependent (open command
80 mm minus a per-episode anchor) so the gripper was not reliably open during approach. The
constant-target form keeps dims 0-6 byte-identical and ADDS the target as a separate channel, so
the approach command is untouched.

**PRE-REGISTERED PREDICTION on the one untried lever.** `width_window_weight` (implemented at
`pointcloud_dataset.py:102-110`, still NEVER run) up-weights the width dim on chunks overlapping
the closing/hold window (~52% of steps). The arithmetic above predicts it is **NOT sufficient**: it
raises the 2% but its window is dominated by the PLATEAU, where the copier is already accurate to
0.39 mm. Expect a small slope change at best. It is one config flag, so it is still worth running
as a cheap falsification of this analysis — recorded here BEFORE the run so the reading is not
fitted afterwards.

**METHOD NOTE:** every number here comes from the raw dataset npz, no policy, no eval, no join
across files — the class of check that has repeatedly been decisive in this campaign, and the
cheapest one available.

### 2026-08-28 — DELTA-WIDTH arm launched (user's "last resort"): absolute 6d pose + DELTA gripper

**Job 1769974 `dgrip_gen`.** xaqnb's recipe verbatim (seed 42, 350 epochs, paired-reg 0.5, USUAL
network size) on a relabelled dataset; the ONLY change is the gripper action space.

**Rationale (user's):** with an ABSOLUTE width the policy can satisfy its loss by COPYING the
observed width — **86% of demo steps repeat the previous command** — and a copier cannot DRIVE
closure. As a DELTA, "hold" is one value and "close" is a distinct act.

**What was built.**
- `dataset/dppo/single_lift_generalist_3obj_dgrip/` — relabelled from the generalist set. Pose dims
  UNTOUCHED; gripper column becomes a per-step delta, `gripper_delta_scale = 0.0035 m` (max demo
  delta is 3.07 mm, so 0% of steps clip).
- `configs/action/abs_pose_euler_delta_gripper.yaml` + experiment
  `single_lift_mushroom_soft_abs_action_armfocus_7d_realws_dgrip` (diff vs its absolute twin is
  exactly the `action:` line).
- Shared sim/real code: `ActionConfig.gripper_delta/_scale`, a delta branch in
  `ActionPipeline._process_absolute` (**scale factor ONLY — no affine**, the B10 trap),
  `PolicyEnv` pushes the flag to the backend as it does `rate_limit`, and BOTH `SimBackend` and
  `RealBackend` accumulate onto their running target BEFORE the rate limiter. All behind a flag
  defaulting to false, so every existing config is bit-identical.

**Validation gates, all passed.**
| gate | result |
|---|---|
| delta decode | 0 -> 0, +-1 -> +-3.5 mm; absolute mode and pose dims unchanged |
| existing pipeline tests | **23/23 pass** (no regression on shared sim/real code) |
| round-trip vs SHIPPED normalization, both splits | **max 0.00000 mm**, 0% clipped |
| `delta[0]` semantics | = first COMMAND - first MEASURED width (the backend seeds its target from the MEASURED width at reset; using 0 would offset every episode) |
| paired regularizer | operates on POINT CLOUDS only, never actions -> action-space agnostic, comparability with xaqnb preserved |

**BUG FOUND AND FIXED IN MY OWN RELABELLING** — see `docs/CHECKLISTS.md` 5.1: val was normalized
with its own range while `normalization.npz` shipped train's. The first round-trip missed it because
it validated each split against its own constants. Now both splits are decoded with the SHIPPED file.

⚠ **THE RISK THIS ARM CARRIES: 86% of delta targets are exactly ZERO.** The policy can minimize loss
by emitting ~0 forever and never closing — the mirror of the copy-proprio failure. **Teaser at
`state_150`** (15 eps / 3 batches / sgs=1, all rendered) rather than discovering it at 350.

⚠ **EVAL MUST use the `_dgrip` experiment.** Pairing this checkpoint with the absolute pipeline would
decode the gripper through the wrong map entirely (affine into [0,88] mm instead of +-3.5 mm/step).

### 2026-08-28 — F2 VERDICT: the diffusion head fixes their TOTAL failure (0.000 -> 0.200), but their stack is still far behind ours

Canonical protocol, mushroom, 200 episodes, 40 distinct geometries:

| arm | stack | head | success | ever | SUSTAINED | peak |
|---|---|---|---|---|---|---|
| xaqnb baseline | OURS | PointNet diffusion MLP | **0.915** | 0.920 | 35,167 | 54,673 |
| E (GAP) | OURS | same + proprio encoder | 0.840 | 0.915 | 30,970 | 53,784 |
| **F2** | **THEIRS** | **ConditionalUnet1D (diffusion)** | **0.200** | 0.305 | **42,394** | 57,386 |
| F | THEIRS | DeterministicMLP (Tanh+MSE) | **0.000** | — | — | — |

**1. THE USER'S HYPOTHESIS IS CONFIRMED IN DIRECTION.** F -> F2 is ONE config line
(`head: dmm -> diffunet`) inside THEIR codebase, same encoder/trainer/data/GAP settings, and it
takes success from **0.000 to 0.200**. A unimodal MSE head cannot represent our multimodal
approach-pose distribution; a diffusion head can. The user reached this from the render frames
before any number existed.

**2. THE HEAD WAS *A* BLOCKER, NOT *THE* BLOCKER.** 0.200 vs our 0.915 means most of the gap is
elsewhere in their stack (ResNet-shaped encoder replaced by our PointNet, their temporal
transformer, their per-batch cyclic LR with `T_max=cfg.epoch`, their contaminated valid split,
horizon 9 vs our 4).

**3. F2 IS THE HARSHEST-GRIPPING ARM MEASURED: sustained 42,394** (vs baseline 35,167, E 30,970).
Where it does grasp, it grips hardest — so it is worse on gentleness as well as success.

⚠ **CONFOUND STILL OPEN:** F2 has **6.9x** arm F's TOTAL parameters (65.7M vs 9.5M) — head type and
capacity moved together. Arm F3 (`down_dims=[16,32,64]`, total 10.65M = 1.12x arm F) is staged and
config-only; it would separate them.

⚠ **NOT a reproduction of their paper:** their 101-epoch budget on our 1248-demo set is ~2.5x their
original gradient steps, and their GAP window covers 50% of F/F2's training but only 14% of E's.

**CONCLUSION FOR THE PLAN: their architecture is not a route worth pursuing for us.** Our own
generalist is 4.6x better on success and gentler. The transferable finding is the NEGATIVE one about
deterministic heads on multimodal demonstration data — which is a paper-worthy observation in its
own right, and it was the user's call, not a metric's.

### 2026-08-28 — ⚠ SIZE-BLINDNESS IS OBJECT-SPECIFIC: tofu ADAPTS (slope 0.26, CI excludes 0), mushroom does NOT

Canonical protocol, 200 episodes, **40 distinct geometries**, both objects, both arms:

| object | arm | success | SUSTAINED | **width slope** | 95% CI | %demo |
|---|---|---|---|---|---|---|
| mushroom | xaqnb baseline | 0.915 | 35,167 | 0.02 | [-0.21, +0.25] | 2% |
| mushroom | E (GAP) | 0.840 | 30,970 | 0.03 | [-0.18, +0.23] | 3% |
| **tofu** | **xaqnb baseline** | **0.785** | 12,645 | **0.26** | **[+0.06, +0.46]** | **24%** |
| **tofu** | **E (GAP)** | **0.705** | 10,957 | **0.22** | **[+0.04, +0.41]** | **21%** |

**THE HEADLINE CORRECTION: "the policy is size-blind" is TRUE FOR MUSHROOM, FALSE FOR TOFU.** On
tofu both arms show statistically detectable adaptation — the CI EXCLUDES zero — at ~a quarter of the
demonstrator. This CONFIRMS, at the rigorous 40-geometry protocol, the earlier 2x2 finding that
**size sensitivity is driven by the OBJECT, not by multi-object training** (previously tofu +0.30 vs
mushroom +0.06 at a weaker protocol).

**GAP's verdict is unchanged and now holds on BOTH objects:** 0.02 vs 0.03 (mushroom), 0.26 vs 0.22
(tofu) — GAP does not change size-awareness either way. E again trades success for a gentler grip on
tofu too (-0.08 success, -13% sustained), the same level effect as on mushroom.

**TWO CANDIDATE EXPLANATIONS, not yet separated:**
1. **Real:** a cube's graspable extent scales directly with `obj_scale`, so the mapping the policy
   must learn is simple and learnable. A mushroom has a cap and a stem, and shape DR (bend/twist/
   taper/axis_scale) deforms it, so scale and graspable extent decouple.
2. **MEASUREMENT:** our x-axis is `obj_scale x nominal`, a PROXY, not the measured graspable extent.
   For a cube it is nearly exact; for a mushroom it is noisy, and noise in x FLATTENS a fitted slope.
   So part of mushroom's 0.02 may be attenuation, not blindness.
**These make different predictions and the test is cheap: measure the actual graspable extent per
episode (from the cloud or the mesh) and refit.** Until then, do not claim the policy is
size-blind in general — only that it is on mushroom, under a proxy x-axis.

**FIGURE BUG FIXED:** `E_vs_baseline_width_vs_size_tofu.png` inherited the subtitle "Both arms are
flat: GAP does not make the policy size-aware" hardcoded from the mushroom figure — FALSE for tofu.
Both figures now read "GAP vs baseline: the slopes match — GAP does not change size-awareness",
which is what the data supports on both objects.
Figures: `docs/figures/E_vs_baseline_width_vs_size_{mushroom,tofu}.png`.

### 2026-08-28 — delta-width teaser blocked by the WORKTREE/MAIN split (5th instance), then gated

The dgrip teaser failed instantly: the eval runs with `GM_REPO=<worktree>` and the `_dgrip`
experiment + action config + the whole `gripper_delta` code path existed only in the MAIN repo.
Training never surfaced it (pure supervised on the npz; the ActionPipeline is not involved).

**Ported SURGICALLY, not by file copy** — the two copies already diverge (`sim_backend.py` by 49
lines for the main-only `GM_FIXED_SCALES`/`GM_FIXED_POSE` sweep knobs), so a wholesale copy would
have dragged unrelated changes into the worktree.

**Gated before relaunching:** job 1771216 loads the `_dgrip` experiment FROM THE WORKTREE and checks
the decode — `action mode: absolute | gripper_delta: True | scale: 0.0035`, `-1/0/+1 -> -3.5/0/+3.5
mm`. Only on PASS does the teaser fire (1771217). **The near-miss worth naming: if the config had
been present but the CODE absent, the gripper would have decoded through the ABSOLUTE affine into
[0,88] mm and produced a plausible WRONG result rather than an error.**

Rule added to `docs/CHECKLISTS.md` 5.2.

**Also recorded — on the user's fallback idea (recollect ~200 episodes with delta commands):** the
86% zeros come from the DEMONSTRATOR'S PHASE STRUCTURE (the scripted FSM holds a constant target
through approach/grasp/firm/lift/hold), NOT from the relabeling, which is exact (round-trip
0.00000 mm). Re-collecting with delta COMMANDS would reproduce the same physical trajectory and the
same sparsity. What would actually change the data is a collector that INTERPOLATES / rate-limits
each target change across several steps, making the deltas dense — a different demonstration STYLE,
not a re-encoding. Worth deciding deliberately before spending ~7 h, especially with the collector
mid-rework upstream.

### 2026-08-28 — DELTA-WIDTH teaser @state_150: the 86%-zero hazard did NOT materialise; it OVER-CLOSES instead

`dtjze/eval/teaser_e150` — 15 eps / 3 batches / sgs=1, all rendered, `_dgrip` experiment verified
to decode the gripper as a delta before the run.

| metric | delta-width @ep150 | demonstrator | reference (200 ep) |
|---|---|---|---|
| success | **0.333** (ever 0.333) | 0.94 | xaqnb 0.915 / E 0.840 |
| SUSTAINED stress | 19,457 | — | xaqnb 35,167 |
| **width travel / episode** | **31.5 mm** (min 29.0, max 35.2) | ~48 mm | — |
| **min width reached** | **13.2 mm** | ~31 mm | — |

**1. THE PRE-REGISTERED RISK DID NOT HAPPEN.** 86% of delta targets are exactly zero, so the feared
failure was "emit ~0 forever, never close". Instead the policy drives the gripper decisively in
EVERY episode — minimum travel across 15 episodes is 29 mm, none stalled. **The delta
parameterization does make closure an ACT rather than a level to hold**, which is exactly the
mechanism the user proposed.

**2. THE NEW FAILURE IS OVER-CLOSING: it closes to 13.2 mm where the demonstrator stops at ~31 mm
— roughly 18 mm too tight.** Corroborated by `ever == success` exactly (0.333): nothing lifts then
drops, so objects are either grasped or missed/crushed outright, unlike the baseline's lift-then-drop
signature. The low sustained stress (19,457) therefore reflects FAILING TO HOLD rather than gentle
grasping, and must not be read as a gentleness win.

**3. IMPLICATION FOR THE USER'S FALLBACK (recollect ~200 eps with delta commands).** The failure is
OVER-ACTION, not under-action, so a native-delta collection is unlikely to fix it: the deltas would
be IDENTICAL (the relabeling is exact, round-trip 0.00000 mm). Over-closing points at the
INTEGRATION running too far. Two cheaper levers first:
   - the existing `rate_limit` dgrip is 0.005 m/step and currently NEVER BINDS (our scale is 0.0035);
     tightening it would directly cap the accumulated closure;
   - a collector that INTERPOLATES/rate-limits target changes would make the deltas dense — a
     different demonstration STYLE, which is the version of the user's idea that would actually
     change the data.

**VERDICT: not degenerate — let it finish to 350 and re-check.** 0.333 is a mid-training,
15-episode SCREEN, not a number to compare against 200-episode results.

### 2026-08-28 — LITERATURE ANCHOR for the loss budget + DELTA is NOT the fix (measured) + the changepoint is ALSO copy-exact

User asked: has nobody hit this in the literature; how is the gripper action usually represented; is the
point cloud the problem. Three answers, two of them measured on our own data.

**1. IT IS A NAMED FAILURE MODE — the COPYCAT problem / causal confusion in imitation learning.**

| paper | venue | the nuisance variable |
|---|---|---|
| [Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979) (de Haan, Jayaraman, Levine) | NeurIPS 2019 | any observed correlate of the expert action |
| [Fighting Copycat Agents in BC from Observation Histories](https://arxiv.org/abs/2010.14876) (Wen et al.) | NeurIPS 2020 | the PREVIOUS ACTION, recoverable from an obs history |
| [Keyframe-Focused Visual Imitation Learning](https://arxiv.org/abs/2106.06452) (Wen et al.) | ICML 2021 | fix = up-weight expert ACTION CHANGEPOINTS |
| [Resolving Copycat Problems via Residual Action Prediction](https://arxiv.org/abs/2207.09705) (Chuang et al.) | ECCV 2022 | fix = predict the action RESIDUAL |

**OUR INSTANCE IS THE DEGENERATE-EASY CASE.** The literature's copycat needs an observation HISTORY
from which the previous action can be inferred. We hand it over directly: the action IS the absolute
gripper width and `gripper_width` (the achieved result of the previous command) is a raw obs channel.
So the standard remedy — "condition on the most recent observation only" — does not apply to us; the
leak is IN the most recent observation.

**2. `width_window_weight` NOW HAS A WORSE PROGNOSIS, and the reason is a NEW measurement.**
The ICML-2021 fix is to up-weight expert action CHANGEPOINTS. Measured on our data, the changepoint
(the one frame per episode where the closing ramp stops) is where the copier is MOST accurate:

| frames | share of dataset | copy \|err\| |
|---|---|---|
| ramp (closing) | 14.5% | **3.26 mm** (servo lag — object-INDEPENDENT) |
| plateau (the grasp level) | 38.0% | 0.58 mm |
| **CHANGEPOINT (ramp stops)** | **0.49%** | **0.41 mm** |

**There is NO frame at which the copier fails on the size-dependent quantity.** By the time the
command settles, the achieved width has caught up, so the copy is exact exactly where the decision
is. The copy's only error is the ramp's servo lag, which carries no size information. This
STRENGTHENS the pre-registered null for `width_window_weight` (its window is ramp+plateau) and it
also rules out the keyframe variant of the same idea. Loss re-weighting cannot work here: there is
no frame to re-weight toward.

**3. DELTA WIDTH IS NOT THE FIX EITHER — measured counterfactual on the same dataset.**

| representation | between-episode share of action energy |
|---|---|
| absolute width (current) | 2.07% |
| **delta width** | **2.22%** |

Delta buys 0.15 points, and it makes the target WORSE-posed: `delta = 0.000 mm` on the open phase and
`-0.004 mm` on the plateau — identical for every episode regardless of object — so the grasp LEVEL is
never supervised at all, only recoverable by INTEGRATING the ramp (corr(ramp-mean delta, final level)
+0.747; corr(ramp length, final level) -0.384). RAP (ECCV 2022) works in driving because the residual
removes the leak; for a rate-limited aperture ramp it removes the SIGNAL. **Absolute-vs-delta is not
the axis. TRAJECTORY-vs-TARGET is.**

**4. HOW THE FIELD REPRESENTS THE GRIPPER — and why this has gone unreported.** The dominant
convention is BINARY open/close, in which case there is no aperture to adapt and this failure mode
cannot be observed. [UMI](https://arxiv.org/abs/2402.10329) (Chi et al., RSS 2024) is the main work
arguing for continuous width: *"In contrast to the binary open-close action used in prior works, we
found commanding gripper width continuously significantly expands the range of tasks doable by
parallel-jaw grippers."* Two things they do are directly relevant to us:
- They get force regulation from **HARDWARE** — soft series-elastic fingers, so commanded width
  deformation implicitly sets grasp force — rather than from the policy predicting the right width.
  That is a hardware answer to the problem we are fighting in software, and it is a live option
  (cf. the GelSight premise question in `width_adaptation.md` §0).
- They represent EE proprioception as a **RELATIVE TRAJECTORY**, not absolute pose. Their stated
  reason is calibration-free deployment/tracking robustness, but it has the same de-shortcutting
  side effect — and note they did NOT do this for the gripper channel, which stays absolute width.

**5. IS THE POINT CLOUD THE PROBLEM? NO — by our own numbers** (`docs/width_predictability.md`):
cloud->width r = 0.771 (well-trained head, frozen standard encoder) / 0.776 (aux-supervised encoder);
cloud->scale 0.927. Delivered slope-equivalent is 0.02-0.20. **The genuine perception limit is
CROSS-SHAPE, not size:** leave-one-mesh-variant-out drops cloud->width to 0.23-0.41. So the cloud is
adequate for within-distribution sizing and is a first-order problem only for the unseen-object goal.

**NOVELTY CHECK (for the paper):** the earlier scan's claim stands after re-searching —
no published controlled study measures whether a parallel-jaw IL policy adapts APERTURE to object
size. The framing this session supports is sharper and is a contribution in itself: *continuous-width
grippers inherit a copycat pathology that binary grippers structurally cannot exhibit, and it is
invisible to BC validation loss.*

### 2026-08-28 — PLAN: TARGET-PARAMETERISED WIDTH (8th action dim). Pre-flight PASSED, prediction pre-registered.

The fix implied by the loss budget, with the two load-bearing checks already measured on
`dataset/dppo/single_lift_generalist_3obj/train.npz` (1248 eps) BEFORE any GPU time.

**THE CHANGE.** Add action dim 7 = the episode's FINAL commanded width, **constant from t=0**.
Dims 0-6 stay byte-identical. Execute `width = max(dim6, dim7)`: dim6 (which the copycat solves,
and that is fine) supplies the closure CLOCK; dim7 supplies the LEVEL.

**PRE-FLIGHT 1 — the execution rule is label-consistent, so there is NO train/execute mismatch.**
Because the demo command decreases monotonically to its plateau, `max(dim6_label, dim7_label) ==
dim6_label` identically:

| check | result |
|---|---|
| command RE-OPENS after closing | 5 / 1248 (0.40%) |
| plateau level != episode MIN | 2 / 1248 (0.16%) |
| **`max(dim6,dim7)` reproduces the demo command EXACTLY** | **99.84% of episodes** |

This is the property EVERY previous adaptive arm lacked: CFG, floors, contact stop and freeze are
all TEST-TIME overrides the policy never saw in training (-> distribution shift -> the success
collapses this campaign kept measuring). Here the rule is an identity on the training data.

**PRE-FLIGHT 2 — the gradient pressure, in the units the MSE actually sees.**

| | today (7-dim) | proposed (8-dim) |
|---|---|---|
| dim carrying the size signal | dim6 level, 2.07% of action energy | **dim7, 5.1%** |
| share of THAT explained by the proprio copy | 99.9% | **0% (R2 = 0.0000 over the 47.5% open phase)** |
| **net pressure to read size from the cloud** | **~0.003%** | **~5.1% (~1700x)** |

**NOT residual-width v2 (nickq, DECISIVE NEGATIVE).** That RELABELLED dim6 as `command - anchor`,
making the APPROACH command scene-dependent so the gripper was not reliably open on approach. This
ADDS a channel and leaves the approach command untouched.

**PRE-REGISTERED PREDICTION (written before the run).**
- slope **0.5-0.9** — upper-bounded by the head's cloud->width r ~= 0.78 and by the demonstrator's
  own 3.2 mm residual, which an imitator cannot beat.
- success **NOT materially below 0.915**, because the execution rule is label-consistent.
- **FALSIFIER:** slope < 0.2 despite a 5.1% gradient share => the trunk genuinely cannot route
  cloud-derived size to the action head, and the remaining answer is ARCHITECTURAL separation
  (a width head that does not see `gripper_width`), not loss or action-space shaping.

**Known risk, stated:** dim7 error is consequential in both directions (over-predict -> loose ->
drop; under-predict -> extra squeeze). Head MAE ~2.8 mm against the collector's ~8 mm squeeze
budget, and the demonstrator's own width residual is 3.2 mm — so the error is roughly at
demonstrator noise, but it is the binding constraint and should be reported per size bin.

**Cost:** dataset RELABEL only (no re-collection) ~10 min; `action_dim 7->8` + eval adapter; ~4 h
train + ~1.5 h canonical eval (200 eps / 40 geometries).

**FREE SUPPORTING MOVE, measured but never run** (`docs/width_predictability.md`): dropping the
bottom-20%-`align` mushroom demos raises corr(width, scale) 0.841 -> 0.933 AND lowers demo stress
10%, with retention uniform across scale (72-83% per bin, all 4 mesh variants kept). It sharpens the
conditional the diffusion policy must fit, which is the same disease attacked from the data side.

**EXPLICITLY NOT WORTH GPU TIME NOW** (all measured nulls or closed lines): `width_window_weight`
and keyframe up-weighting (no frame exists where the copier fails on the size-dependent quantity),
delta-width actions (2.22% vs 2.07%), further aux/conditioning heads, GAP/proprio-encoder.

### 2026-08-28 — CLOSING SPEED IS NOT THE PROBLEM. NON-TERMINATION IS — and it has a heavy tail.

User asked whether closing too fast is a problem. **Measured, and the premise is inverted:** the
policy closes SLOWER per step than the demonstrator and is nowhere near the rate limit. What it does
not do is STOP. Policy = xaqnb canonical mushroom (200 eps / 40 geometries,
`.agent_tmp/canon_mushroom_200geo40_1767027_widthcmd_b*.npz`); demonstrator = the training npz
(mushroom+tofu-scale episodes, n=1041), absolute env steps.

| | demonstrator | policy |
|---|---|---|
| closing rate while closing | **-1.74 mm/step** | **-1.29 mm/step** (26% SLOWER) |
| rate limit (`abs_pose_euler_abs_gripper.yaml`) | 5.00 mm/step | 5.00 mm/step (3.9x headroom) |
| closure ONSET | step 95 | step 82 |
| **onset -> SETTLED (within 0.5 mm of final)** | **26 steps** | **179 steps (7x)** |
| flat fraction of the last 25% of the episode | **1.00** | 0.87 |
| further closure after step 120 | 0 (dead flat) | median -0.9, **p90 +21.3, p99 +41.4 mm** |
| final commanded width | 32.8 mm | **27.3 mm** |

**IT IS A TAIL, NOT A DRIFT.** The 2.7 mm mean extra closure is driven by ~10% of episodes that keep
squeezing **20-40 mm** past the point where the demonstrator has already stopped. The median episode
is fine. This is the quantified version of the anecdote already in this DEVLOG ("some episodes keep
closing AFTER the grasp, 34mm -> 12mm during hold") and it is where the SUSTAINED-stress gap lives.

**THIS IS THE CLOSED-LOOP SIGNATURE OF THE COPYCAT RULE.** `a = w_obs - 0.61 mm` (R2 0.996 on the
demos) has NO FIXED POINT — it keeps commanding a slightly tighter width forever until a finger is
physically blocked, and on a soft MPM body it never is. The demonstrator has a HARD STOP (its
plateau is dead flat, 1.00); the learned rule has none. So the dataset diagnosis and the closed-loop
behaviour agree, from two independent measurements.

**CONSEQUENCE FOR THE PLAN — the same single change fixes this too.** `width = max(dim6, dim7)`
imposes a hard floor: once dim6 reaches dim7 the executed width CANNOT creep further. One change
addresses (i) the flat slope, (ii) the runaway tail, (iii) the part of the sustained-stress gap that
tail causes. **Guard to add:** dim7 is predicted per step, so LATCH it (median of dim7 over the first
K steps after closure onset, or a running min-clamp) — reuse the latching machinery from the floor
arms. Report the p90/p99 extra-closure statistic as an acceptance criterion, not just the mean.

**AND A WARNING THAT FOLLOWS DIRECTLY: do NOT slow the demonstrator's closure to be "gentler".** The
copier's ONLY error is the servo lag during the ramp (3.26 mm). Closing more slowly shrinks that lag
toward zero, which makes the shortcut MORE exact and the size signal even less learnable. Gentleness
from a slower ramp would be bought at the cost of adaptation.

**NOT fixed by dim7, and probably not worth acting on:** onset is 13 steps early (82 vs 95). Small,
and the cloud-swap ablation shows visual sensitivity is lowest during approach anyway. Flagged, not
prioritised.

### 2026-08-28 — 3x-WIDTH teaser @state_300: healthy, and it looks like another LEVEL shift

`ddgrl/eval/teaser_e300` — 15 eps / 3 batches / sgs=1. Eval architecture verified from its own
resolved config (`mlp_dims = [3072,3072,3072]`) before reading anything; without that override the
eval would have built a 1024-wide net against a 3072-wide checkpoint.

| | 3x width @ep300 (15 eps) | 1x baseline (200 eps) |
|---|---|---|
| success | 0.800 (ever 0.867) | 0.915 (ever 0.920) |
| SUSTAINED | 21,218 | 35,167 |
| at-grasp width | **32.5 mm** (sd 3.9) | **28.2 mm** (sd ~4.4) |

**Not degenerate — let it finish.** The interesting hint: it grips **~4.3 mm WIDER** with
correspondingly lower stress, and `ever - success` is 0.067 vs the baseline's 0.005. That is the
signature of the SAME LEVEL EFFECT every mechanism in this campaign has produced (wider grip -> less
stress -> more drops), not of size-awareness. **Extra capacity appears to have moved WHERE it grips,
not made it size-aware** — but 3 geometries cannot give a slope, so the canonical 200-ep/40-geometry
eval decides.

**Supporting prior:** the 3x net reached train 0.0005 / val 0.0009 vs the baseline's 0.0007 train —
7.6x the parameters bought almost no fitting gain, suggesting the 1x net already fits these
demonstrations and capacity is not the binding constraint. Stated as a PREDICTION: eps-loss has
repeatedly failed to predict closed-loop behaviour here (arm D had the best val loss and never
closed), so the slope is what settles it.

**Protocol caveat:** teaser is `state_300` (not the `state_150` used for C'/D'/E), 15 episodes, 3
geometries — a screen, not comparable to 200-episode numbers in either direction.

### 2026-08-29 — upstream 691c565 + 5f95c7a: GENTLENESS NEEDS MATERIAL, NOT JUST SIZE — this is the answer to the "why not scripted top-down?" reviewer question

Upstream found a size-only squeeze rule is gentle on mushroom (**0.58x yield, 99.6% sub-yield**) and
**damaging on cherry tomato (1.18x yield, 5.8% sub-yield)** — same rule, opposite outcome. Physics:
for indentation `d` over characteristic length `L`, `sigma ~ E*d/L`, so sub-yield requires

    d <= K * (yield/E) * L

**`yield/E` varies 2.7x across our objects** (tofu 2.5, raspberry 6.7, mushroom 7.5, strawberry 8.3,
banana 10.0, tomato 12.0, cherry_tomato 13.3) and is **NOT observable from a point cloud**.
5f95c7a extends the same budget to the FIRM phase, which dominated on small stiff objects (4.5 mm on
a 24.7 mm cherry tomato = 18%, dwarfing its 0.84 mm base squeeze — which is why fixing the base
alone moved sub-yield only 5.8% -> 6%).

**THIS ANSWERS THE REVIEWER QUESTION the user raised ("why not just top-down grasp at the floor
width?").** Geometry gives `L`. It cannot give `yield/E`. So a vision-only geometric floor is
calibrated for ONE material and is necessarily too loose (drops) or too tight (crushes) on others —
and upstream MEASURED that failure on a real object. What a learned policy can do instead is
associate visual appearance with material response, learned from demonstrations generated under a
material-aware rule.

**PRE-REGISTERED PREDICTION for the scripted baseline (user approved building it):** a vision-only
top-down grasp with a geometric size-only floor will be ACCEPTABLE ON MUSHROOM (the material its
calibration implicitly targets) and MATERIALLY WORSE on tofu and raspberry. The learned policy should
degrade less across the three. If the scripted baseline instead matches the policy everywhere, that
is a more important finding and we need it before a reviewer supplies it.

⚠ **DATA NOTE:** the mushroom portion of our generalist set is unchanged by the new rule
(1.94/2.43 mm vs 2.0/2.5), so it stays valid. **Our TOFU and RASPBERRY demos were collected under the
OLD fixed rule**, and tofu has the LOWEST yield/E (2.5) of any object — the most likely to have been
over-squeezed. Check their recorded stress before reusing those portions in the bigger collection.

### 2026-08-29 — ⚠ THE SCRIPTED VISION-ONLY BASELINE BEATS THE LEARNED POLICY ON MUSHROOM. This is a real problem for the framing.

User asked the reviewer's question directly ("why not just top-down grasp at the floor width?") and
approved building the control. Built: `gentle_manip/dppo/scripted/` — object position, narrow-axis
width and yaw from the **t=0 point cloud only** (STUDENT INFO: cloud + proprio, no privileged pose,
no registry, no category), then a fixed phase machine emitting absolute top-down pose targets.
Calibrated ON MUSHROOM deliberately (+18 mm parallax, visible->true 1.23, 2 mm squeeze).

**Canonical protocol, 200 episodes, 40 distinct geometries:**

| object | LEARNED generalist | SCRIPTED vision-only |
|---|---|---|
| **mushroom** | 0.915 / sustained 35,167 | **0.940 / sustained 28,934 (-18%)** |
| tofu | 0.785 / 12,645 | (200-ep run in flight) |
| raspberry | **0.583** | 0.467 (15 ep, 5/15 perception fallbacks) |

**ON MUSHROOM — the object the real-robot work is built on — A NO-LEARNING PIPELINE WINS ON BOTH
SUCCESS AND GENTLENESS.** That must be answered before submission; "it degrades on small objects" is
only a partial answer.

**Where the scripted baseline DOES break, and why — measured, not argued.** Its commanded width on
raspberry is **median 8.9 mm, range 4.0-34.9 mm on a ~15 mm object**: with only **4-7 object points**
in a 1024-point `object_focus` cloud (vs ~75 for a mushroom) the geometric estimate is unstable, and
5/15 episodes could not detect the object at all. **A hand-written geometric pipeline needs enough
points to measure the object; the learned policy extracts usable structure from the SAME sparse
cloud where explicit geometry cannot.** That is a defensible contribution claim and it is testable.

⚠ **TWO RETRACTIONS OF MY OWN REASONING, both made before checking:**
1. I claimed raspberry's initial 0/20 supported the MATERIAL argument (`d <= K*(yield/E)*L`). It did
   not — the estimator fell back on **15/15** episodes; it never detected the object. That was a
   perception failure OF MY BASELINE. With thresholds fair to small objects it reaches 0.467.
2. I reached for the interpretation that suited the narrative before verifying the mechanism. The
   material argument may still be true, but THIS experiment does not evidence it.

⚠ **`ScriptedTopDownPolicy` shipped with NO dumps — the third Policy adapter to hit that documented
trap** (the dump lives in the Policy class, not the harness), which cost the ability to diagnose a
200-episode run that had already been spent. Dumps + LOUD fallback counting added afterwards.

**SCOPE — what this does NOT say.** Singulated objects, flat table, clean simulated depth, top-down
grasps only. Clutter, occlusion, non-top-down grasps and real-robot transfer are all untested, and
they are where a learned policy would be expected to earn its place. **That is the experiment the
paper now needs**, and it is a better use of effort than anything remaining on width adaptation.

### 2026-08-29 — PHASE CLOSE-OUT: the consolidated table. NOTHING beat the plain baseline.

All canonical: 200 episodes, 40 distinct geometries, sgs=1, per-episode video, dumps on.

**SUCCESS by object**

| arm | mushroom | tofu | raspberry |
|---|---|---|---|
| **xaqnb baseline (plain generalist)** | **0.915** | **0.785** | **0.583** (60-ep) |
| E — GAP, their CPD+LSTM rho | 0.840 | 0.705 | — |
| 3x MLP width (7.6x params) | 0.705 | 0.485 | 0.465 |
| DELTA width | 0.680 | 0.550 | 0.360 |
| F2 — their stack, diffusion head | 0.200 | — | — |
| F — their stack, deterministic head | 0.000 | — | — |
| C'/D' — proprio encoder +- uniform damping | 0.000 (ep150) | — | — |
| **SCRIPTED vision-only (no learning)** | **0.940** | 0.760 | 0.467 |

**SUSTAINED STRESS (mushroom / tofu)**: baseline 35,167 / 12,645 · E 30,970 / 10,957 ·
3x 24,975 / 8,937 · delta 25,742 / 9,682 · F2 42,394 / — · **scripted 28,934 / 9,184**.

**WIDTH SLOPE (mushroom, 40 geometries)**: baseline 0.02 [-0.21,0.25] · E 0.03 [-0.18,0.23] ·
3x 0.21 [+0.02,+0.40] · delta 0.25 [-0.17,0.66] (at-grasp sd 13.3 mm vs baseline 4.4).
On TOFU both baseline and E DO adapt: 0.26 [+0.06,+0.46] and 0.22 [+0.04,+0.41].

**CONCLUSIONS**
1. **No learned variant beat the plain recipe on any object.** Every mechanism that lowered stress
   did so by gripping wider or looser and dropping more — the same LEVEL effect throughout.
2. **Size-blindness is object-specific**, not universal: real adaptation on tofu, none on mushroom.
   Unresolved whether mushroom's 0.02 is genuine or attenuation from the `scale x nominal` proxy
   x-axis — the cheap test (measure true graspable extent, refit) is still not done.
3. **GAP:** mechanism real (rescues 0.000 arms) but loses to baseline and never touches slope.
4. **Their architecture:** the deterministic head is a genuine blocker (0.000 -> 0.200 with their
   own diffusion head), but their stack still lands 4.6x below ours.
5. **⚠ A NO-LEARNING VISION-ONLY BASELINE BEATS THE LEARNED POLICY ON MUSHROOM** (0.940/28,934 vs
   0.915/35,167) and is competitive-and-gentler on tofu (0.760/9,184 vs 0.785/12,645). It loses only
   on raspberry (0.467 vs 0.583), where 4-7 object points make its geometric estimate unstable
   (commanded width median 8.9 mm, range 4.0-34.9 on a ~15 mm object).

**WHAT THE PAPER MUST NOW ANSWER**, and it is not width adaptation: on singulated objects on a flat
table with clean depth, explicit geometry is at least as good and gentler. The learned policy's
demonstrated edge is the SPARSE-CLOUD regime. Everything else a learned policy should buy —
clutter, occlusion, non-top-down grasps, real-robot transfer — is UNTESTED. That is where the
remaining effort belongs.

### 2026-08-29 — ⚠ CORRECTION: the scripted baseline does NOT beat the learned policy. My v1 was given the answer.

The user caught three things wrong with the v1 baseline: it must be top-down ALWAYS, it should be a
simple width estimate then grasp, and **it must not use mesh information**. All three were right, and
the third was a genuine leak I introduced.

**THE LEAK.** v1 used `VIS_TO_TRUE = 1.23`, derived as (mushroom nominal **33 mm** from the MESH
REGISTRY) / (visible 26.8 mm). That is privileged information the learned policy never gets. v1 also
searched yaw for the object's narrowest axis — orientation OPTIMISATION, not a plain top-down grasp.

**v2 removes both.** Fixed top-down yaw; width = the extent it MEASURES along the gripper's closing
axis, minus a 2 mm squeeze. Nothing from the registry.

| object | LEARNED | v1 (mesh factor + yaw search) | **v2 (honest vision-only)** |
|---|---|---|---|
| mushroom | **0.915** / 35,167 | 0.940 / 28,934 | **0.395** / 13,706 |
| tofu | **0.785** / 12,645 | 0.760 / 9,184 | **0.070** / 5,137 |
| raspberry | **0.583** | 0.467 | ~0.04 (killed degenerate) |

**MECHANISMS, all read off the dumps rather than inferred:**
- **tofu 0.070** — commanded width median **42.2 mm**. A 30 mm cube on a diagonal presents
  sqrt(2)x30 = **42.4 mm** to a FIXED axis. With yaw fixed the gripper measures the diagonal and
  opens to it, so it never closes on a face. A fixed top-down grasp cannot handle a box at arbitrary
  yaw — aligning the jaws with a face is exactly what the yaw search had been doing.
- **mushroom 0.395** — commanded **37.9 mm** on a ~33 mm object: a fixed axis measures a wider extent
  than the narrowest one, and without the (illegitimate) 1.23 factor the result is still too wide to
  grip. Stress 13,706 confirms it is barely touching.
- **raspberry ~0.04** — width now CORRECT (median 15.3 mm on a ~15 mm object) but `X_BIAS_M = 0.018`
  is a parallax constant calibrated on mushroom/tofu. Parallax scales with the object's own radius,
  so an 18 mm correction on a 15 mm object overshoots past the object entirely.

**SO THE REVIEWER CHALLENGE IS MUCH WEAKER THAN I REPORTED.** My earlier entry ("A NO-LEARNING
BASELINE BEATS THE LEARNED POLICY ON MUSHROOM") stands only for a baseline that was handed the mesh's
nominal size and allowed to optimise orientation. **Corrected: a genuinely vision-only, mesh-free,
fixed-top-down grasp is 2.3x-11x worse than the learned policy on every object tested.**

⚠ **WHAT IS STILL UNTESTED, and it is the fair middle case:** yaw search (legitimate — it is
geometric, no mesh) WITH measured-only width (no mesh factor). v1 confounded the two advantages;
v2 removed both. That arm would separate "orientation selection" from "privileged size", and it is
the honest strongest form of the reviewer's baseline.

**METHOD LESSON:** I calibrated three constants (`X_BIAS_M`, `VIS_TO_TRUE`, `SQUEEZE_M`) on one
object and reported the baseline as vision-only. Two of them silently encoded that object's scale.
**A constant fitted per-object is privileged information wearing a different hat** — the user caught
what I did not.

### 2026-08-30 — π0.5 PREP: the wrist camera was mounted BACKWARDS, and the smoke test could not have caught it

Building the two-view collection for the π0.5 baseline (external-only vs external+wrist), the
wrist camera passed its smoke test — `(2,480,640,3)` uint8, and the view changed *more* than the
external one when the arm moved (14.87 vs 9.85 mean |pixel change|). **It was pointing at the empty
background above the table.** Only opening `cam_wrist_t25.png` revealed it: dark sky, a sliver of
gripper, no table and no mushroom, while `cam_ext_t25.png` showed arm + table + mushroom correctly.

**ROOT CAUSE — an axis-convention mismatch at the Genesis API boundary.** Genesis
`camera.set_pose(transform=)` takes an **OpenGL** pose: `T_to_pos_lookat_up` computes
`lookat = pos - T[:3, 2]`, so the camera looks along **−z**, +y up. `EE_T_CAM_WRIST`
(`xarm7_config.py`) is a calibrated **OpenCV** extrinsic: +z forward, +y down — the convention
`RealBackend` uses and `depth_to_pointcloud` requires. Genesis's own `camera.extrinsics` property
is the proof, converting back with exactly the flip we were missing:
`res = transform.copy(); res[..., :3, 1:3] *= -1`. Passing the OpenCV matrix raw therefore aimed
the camera along `−z_cam` = straight up.

**FIX** (`genesis_worker.py`, at the `set_pose` call — the Genesis boundary, so `wrist_cam_T` keeps
the OpenCV semantics it shares with `RawObs.camera_extrinsics` and the real backend):
`_T[..., :3, 1:3] *= -1.0` (self-inverse). This also makes the extrinsic the worker reads back
correct, so wrist depth would back-project correctly too, not just render correctly.

**THE METHOD LESSON, which is the durable part.** The assertion I wrote (`view changed > 2.0`)
**cannot fail the way the code actually breaks**: a camera bolted on backwards still moves with the
arm. It was a test of the re-pose plumbing that I read as a test of the camera. Replaced with three
checks that are geometric rather than incidental:
1. **round-trip** — the extrinsic Genesis reports back must equal `world_T_ee @ EE_T_CAM_WRIST`
   (rotation error < 1°). Catches a convention error in either direction, and catches using the TCP
   where the calibration wants the gripper base link.
2. **aim** — the forward axis `world_T_cam[:3, 2]` must have `z < −0.5` at the top-down home pose.
3. **content** — ≥25% of wrist depth pixels within 1 m: it sees the table, not the sky.

Generalises past cameras: **name the convention explicitly whenever a pose crosses a library
boundary** (OpenCV vs OpenGL, wxyz vs xyzw, which link a calibration is relative to). This is the
axis-convention sibling of the B1/B10/B17 wrong-reference-frame class — right arithmetic, wrong
frame — and it was caught only by looking at the artifact. Recorded in `docs/CHECKLISTS.md` §5.3.

**Near-miss worth stating:** had this shipped, the wrist variant of the π0.5 comparison would have
trained on 250 episodes of sky. It would still have *converged*, and it would have "shown" that the
wrist view does not help — a clean, plausible, entirely fabricated negative result.

*Evidence: job 1796508 (PASSED with the camera backwards), `.agent_tmp/rgb_smoke/`; fix + the three
strengthened checks submitted as job 1796548 (result pending at the time of writing).*

### 2026-08-30 — π0.5 COLLECTION LAUNCHED (250 eps, ext+wrist RGB). Two more traps caught pre-launch

Following the wrist-camera fix above, three further checks before committing ~4 h of GPU time.

**TRAP 1 — the collector had its OWN copy of the hardcoded-empty RGB bug.** `sim_backend.py`'s
`rgb_images={}` was fixed on 2026-08-29, but `collect_demos_synth_v3._state_to_raw_obs` builds its
own `RawObs` and had the identical hardcoded `{}`. Fixing one did not fix the other. RGB rendering
is now DERIVED from the obs config (`_want_rgb = obs_config.images is not None`) rather than a CLI
flag, so the render and the recorded keys cannot disagree — a flag is a thing you forget.
*Lesson: when a bug lives in a duplicated builder, grep for the duplicates before calling it fixed.*

**TRAP 2 — the run would have died AT THE MERGE, after hours of collection.** Measured, not
guessed: two 640×480 RGB streams cost **448 MB per episode — 99.3% of the entire episode** — i.e.
**100.9 GB for 250 episodes**, while `_merge_shards` loads every shard into RAM at once against a
**102 GB** job allocation. The collection would have completed, then OOMed writing `data.pkl`.

Fixed with JPEG q95 at shard-write (`gentle_manip/utils/image_codec.py`, `--image-quality`,
default 95, `0` = raw escape hatch). Measured on a re-run smoke:

| | raw | jpeg q95 |
|---|---|---|
| 4-episode `data.pkl` | 1.61 GB | 0.06 GB |
| per episode | 403 MB | 15 MB |
| **projected, 250 eps** | **100.9 GB** | **3.8 GB** |
| mean abs pixel error vs raw | — | **0.46 / 255 (ext), 0.39 / 255 (wrist)** |

26.8× smaller at ~0.18% pixel error, full resolution retained. Lossy is confined to pixels that a
VLA resizes to 224×224 anyway; actions, proprio, point cloud, privileged labels, DR params and
seeds are untouched, and `convert_demos.py` already excludes `image_*` keys, so the DPPO student
path is unaffected.

**CHECK 3 — REPRODUCIBILITY DEMONSTRATED, not assumed (the user's one stated requirement).** The
raw and JPEG smokes ran at the SAME `--seed`, which turns them into a free experiment: all 24
initial-condition columns of `dr_params.csv` — `cma_seed`, object dx/dy/roll/pitch/yaw/flip, home
offset, `scene_scale`, `scene_bend_deg`, `mesh_variant`, twist/taper/rbf/axis_scale, material,
friction — matched **exactly** across the two independent runs. (Outcome columns may still drift:
MPM on GPU is not bit-deterministic in the rollout. Initial conditions are what reproducibility
means here.)

**LAUNCHED: job 1796751** — 250 episodes, 8 envs, `--seed 0`, `maxfevals 1145`, `scene_dr_every 1`,
the `cdg` v3.4 recipe (`n_grasp 30`, `--grasp-extra-close 0.003`, `--grasp-area-min-mm2 15`,
`approach_speed 0.0024`, `cam_azimuth_max_deg 60`), video for ALL 250 episodes.
Experiment `single_lift_mushroom_soft_pi05` — asserted against the parsed config objects to differ
from the proven collection recipe by EXACTLY `wrist_camera` + the two RGB keys; action, DR, point
cloud and privileged labels are byte-identical, so this set stays comparable to the existing ones
AND still trains the DPPO student.

**ARTIFACTS:** run dir `dataset/demos/single_lift_mushroom_soft/26-08-30-<id>/` (`data.pkl`,
`dr_params.csv`, `config.yaml`, `stats.yaml`, `videos/`); slurm logs
`logs/slumr_logs/1796751.{out,err}` + `1796751_collect.log`. Smokes kept for comparison at
`.agent_tmp/pi05_smoke{,_jpeg}/`.

**FRAMING TO PRE-REGISTER (unchanged, and it matters):** the real rig has NO wrist camera any more,
so external-only is the deployable variant and the wrist variant is a sim-only upper bound. Our
data is one scripted behaviour, one object, one instruction — a π0.5 loss says more about that data
regime than about π0.5. Say so wherever the numbers appear.

### 2026-08-30 — openpi/π0.5 IS FEASIBLE ON THE GH200 NODES AT ITS EXACT PINS (checked, not assumed)

The standing worry about a JAX-based VLA on this cluster is aarch64: the GH200 nodes are aarch64
(glibc 2.34) and several ecosystems ship x86-only CUDA wheels. Checked openpi's actual pins against
what PyPI and the PyTorch index really hold, before spending any time vendoring.

| requirement | openpi pin | aarch64 CUDA availability | verdict |
|---|---|---|---|
| `jax[cuda12]` | `==0.5.3` | `jaxlib`, `jax-cuda12-pjrt`, `jax-cuda12-plugin` 0.5.3 all ship `manylinux2014_aarch64` cp311/cp312 | ✅ exact pin works |
| `torch` | `==2.7.1` | **not** on the cu126 index (only 2.9.0/2.9.1); **is** on **cu128**: `torch-2.7.1+cu128-cp312-cp312-manylinux_2_28_aarch64.whl` | ✅ exact pin works, via cu128 |
| driver | CUDA 12.8 needs ≥570 | measured **580.159.04** on the GH200 | ✅ |
| python | openpi ≥3.11, lerobot (PyPI) ≥3.12 | jax 0.5.3 + torch 2.7.1 both have cp312 aarch64 | ✅ use **3.12** |

**So no version overrides are needed** — unlike `envs/dppo_arrhenius`, which had to override the
dppo fork's `torch==2.4.0`. The new env follows the repo's established per-arch pattern: a
`[[tool.uv.index]]` entry pinning torch to the aarch64 CUDA index, here **cu128** rather than the
cu126 the other arrhenius envs use.

⚠ **TRAP for whoever sets this up:** PyPI's own `torch-2.7.1-...-manylinux_2_28_aarch64.whl` is
**99 MB vs 821 MB for x86_64** — it is a **CPU-ONLY** build. Installing torch for aarch64 from
plain PyPI silently yields no CUDA, and everything "works" until it is mysteriously slow. The
size ratio is the quickest tell. Always take aarch64 torch from the cuXXX index.

Not yet done, and deliberately not started without a check-in: vendoring openpi into `third_party/`
and creating `envs/openpi_arrhenius`. The 250-episode collection (job 1796751) is the active work.

**PROVENANCE GAP FOUND AT LAUNCH (job 1796751), mitigated in-place.** The collector stamps
`git_commit` into `config.yaml` — it recorded **74e14df**, which is the commit BEFORE the wrist-pose
flip, the RGB passthrough and the JPEG codec. Checking out that commit does not reproduce this
dataset. Since the changes are deliberately still uncommitted (the user pushes themselves), the run
dir was made SELF-CONTAINED instead: `PROVENANCE.md` + `uncommitted_changes.patch` (tracked files)
+ `uncommitted_status.txt` + `new_files/` holding the four untracked files that `git diff` cannot
capture (`image_codec.py` and the three configs). 40 kB, and the dataset now carries what builds it.

*Generalises: `git diff` does NOT include untracked files, so a "snapshot the diff" reproducibility
habit silently omits exactly the NEW files a new feature adds — the most load-bearing ones.*

**TODO (deferred, do NOT edit the collector while job 1796751 runs):** `--image-quality` is not in
the collector's `config.yaml` control block, so the snapshot does not record how the images were
encoded (it IS in `data.pkl`'s meta as `image_encoding`/`image_quality`). Add it to the control
dict after the collection lands. Deferring because the CMA-ES `ProcessPoolExecutor` workers may
re-import the module, and a cosmetic snapshot field is not worth any risk to a 4.5 h run.

### 2026-08-30 — ⚠ WRONG ACTION SPACE + WRONG DR: π0.5 collection cancelled at 21/250 and relaunched

**The user caught it.** I forked the π0.5 experiment from `..._mm4_s08` — the recipe the most
recent mushroom *collections* used — reasoning that matching the collection recipe kept the dataset
comparable. Wrong reference. `mm4_s08` carries **`abs_pose_abs_gripper` (10-dim rot6d)** and a
**4-mesh pool at scale [0.8,1.5]**, while every recent *training* run — including the generalist
`ddgrl` we are fixed on — uses **`abs_pose_euler_abs_gripper` (7-dim euler)** with
**`soft_orientation_realws`** (single mesh, scale [1.0,1.5]).

**Why "just re-derive 7d from the 10d recording" would NOT have rescued it.** My first instinct was
that rot6d→euler is a lossless re-encoding of the same rotation, so `convert_demos --derive-action`
could fix it at conversion. Reading `abs_pose_euler_abs_gripper.yaml` killed that: the 7d config
carries **`euler_frame_offset_deg: [180,0,0]`**, and without it a top-down grasp's roll sits exactly
on the ±π seam of `as_euler` — the encoded roll sign-flips between consecutive frames (18–27% of
transitions in every derived abs dataset), trains fine to a low loss, and decodes to a ~180°-wrong
wrist roll → **~0% eval success** (run `oppsu`, `docs/debug_partC_euler_action_anomaly.md`). The
two action spaces are not two spellings of the same thing.

**Two independent defects, one root cause.** The action space AND the DR were both inherited from
the wrong lineage by a single act of forking the wrong file. Had only the action been wrong the
eval harness would likely have failed loudly; the DR would have failed *silently*, giving π0.5 a
different object size/shape distribution than the DPPO baseline it is meant to be compared against
— a confound invisible in every downstream number.

**Corrected:** `single_lift_mushroom_soft_pi05` is now a fork of
`single_lift_mushroom_soft_abs_action_armfocus_7d_realws` (ddgrl's experiment), asserted in code to
differ by EXACTLY `wrist_camera` + the two `image_*` keys — action, DR, augmentation, point cloud
and privileged labels all identical. Job **1796751 cancelled at 21/250**; relaunched as
**1797457**. The aborted run dir carries an `ABORTED.md` saying why it must not be used.

**RECORDED AS FIXED SETUP — `docs/CHECKLISTS.md` §0** (user standing decision): action (7-dim euler
absolute, with the frame offset), proprio (quaternion), obs (`superset_soft_armfocus`) and DR
(`soft_orientation_realws`) are settled, with the concrete parameter values written out. What may
change: network architecture, data composition, training hyperparameters, and the OBJECT SET
(expected to grow — the one DR-adjacent field allowed to move, and only deliberately, since it
shifts the size distribution every width/gentleness number is measured against). New work forks the
reference and asserts the delta against the PARSED config objects.

*METHOD LESSON, and it is the same shape as the `VIS_TO_TRUE` mesh leak: I picked a reference
because it was NEAR the work (the collection recipe) rather than because it was the thing the
result must be COMPARABLE TO (the training reference). "Which config did the last similar job use?"
is the wrong question; "which config does the baseline I am comparing against use?" is the right
one.*

**⚠ SAME-DAY CORRECTION to the entry above — I was wrong that derivation could not rescue it.**
`collect_demos_synth_v3._invert_actions_absolute` **hardcodes rot6d**: the collector records 10-dim
absolute actions no matter what the experiment's `action:` says, and EVERY demo set on disk
(mushroom, tofu, raspberry) is `action_dim: 10`. The 7-dim euler the policies train on has always
been produced at CONVERSION — `.agent_tmp/build_3obj_generalist.sh` builds the generalist dataset
with `--derive-action abs_pose_euler_abs_gripper.yaml --derive-source-action
abs_pose_abs_gripper.yaml`, and `actions/derive.py` applies `euler_frame_offset_deg` there. So
deriving 7d from a 10d recording is not a risky workaround, it is THE pipeline; the `oppsu` failure
came from deriving with a config that lacked the offset, not from deriving at all.

**What this changes:** the relaunch was justified by the **DR** error alone — that one was real,
and it is the dangerous kind because it fails silently (a different object size/shape distribution
than the baseline being compared against). The action-space half of my reasoning was wrong, and the
running job records 10-dim exactly like every baseline set, which is correct. `docs/CHECKLISTS.md`
§0 has been corrected accordingly: do not try to make a collection record 7-dim, and do not read
`action_dim: 10` in a shard as a bug — what must be right is the DERIVE step.

*LESSON: I read one config file's warning comment and generalised it into a claim about the whole
pipeline without checking how existing datasets were actually built. The evidence that would have
corrected me — every demo set being action_dim 10 — was one command away and I ran it only after
the relaunch.*

### 2026-08-30 — π0.5 ADAPTED WITH ZERO CHANGES TO openpi (config + CLI only)

User constraint: *"the best is no internal code needs to be changed, only config modification, and
maybe mild adaptation for our evaluation env"*. Achieved in full — `third_party/openpi` is a clean
checkout at `215abfb` with nothing edited. The trick is to make OUR data match a config they
already ship, rather than adding a config to their tree.

| piece | how | where it lives |
|---|---|---|
| env | `uv sync` from **openpi's own uv.lock** on an aarch64 node | `third_party/openpi/.venv` |
| dataset | emit libero's exact feature names: `image` / `wrist_image` / `state` / `actions` (+ `task`) | `gentle_manip/pi05/convert_to_lerobot.py` (OURS) |
| training | stock `pi05_libero` + CLI overrides (`--data.repo-id`, `--batch-size`, ...) | no file at all |
| inference | `LiberoOutputs` slices to the first **7** dims = exactly our action dim | unchanged |
| eval | `Pi05EvalPolicy` driving the canonical `run_eval` | `gentle_manip/pi05/eval_policy.py` (OURS) |

**Why `pi05_libero` fits us without editing it** — checked in their source, not assumed:
`LiberoInputs` passes `state` and `actions` through with **no hardcoded dimension** (openpi pads to
the model dim), so our 8-dim state and 7-dim action are fine. `LiberoOutputs` returns
`data["actions"][..., :7]` — libero's action dim happens to equal ours, so it is correct for us
verbatim. `repo_id` is a tyro field on `DataConfigFactory`, so the dataset is a pure CLI override.

**FEASIBILITY, measured:** jax 0.5.3 (openpi's exact pin) has `manylinux2014_aarch64` cp311/cp312
wheels for `jaxlib` / `jax-cuda12-pjrt` / `jax-cuda12-plugin`; verified on the GH200 —
`jax.devices()` → 4 `CudaDevice`s, matmul runs. ⚠ **openpi's `torch==2.7.1` resolves to a CPU-ONLY
aarch64 wheel** (99 MB vs 821 MB on x86_64). Harmless here: openpi's JAX path uses torch only for
the LeRobot dataloader, all compute is JAX. Worth knowing before someone "fixes" it.

**ACTIONS — derived, not recorded.** The LeRobot `actions` are the 7-dim euler absolute set derived
with the generalist's exact recipe (`--derive-action abs_pose_euler_abs_gripper`,
`--derive-source-action abs_pose_abs_gripper`, lookahead 1). Validated on real episodes: decoding
the derived actions back through `ActionPipeline` reproduces the commanded targets to **0.000 mm
position / 0.038° max rotation / 0.000 mm gripper**, with **0/237 roll seam crossings**. That
round-trip is the check `oppsu` failed.

**TWO CAMERA VARIANTS FROM ONE DATASET.** `--cameras ext_wrist` (both views) vs `--cameras ext`
(wrist_image zero-filled — openpi's own idiom for a missing camera, what they do for
`right_wrist_0_rgb`). Stated honestly: `LiberoInputs` hardcodes the left-wrist mask to True, so the
ext-only variant still spends image tokens on a blank frame rather than masking it off. It is a
fair "no wrist information" ablation, not a free lunch. The real rig has no wrist camera, so `ext`
is the deployable variant and `ext_wrist` a sim-only upper bound.

**⚠ QUALITATIVE FINDING — NEIGHBOURING ENVS ARE VISIBLE IN cam_ext.** Rendering the recorded RGB to
video (`gentle_manip/pi05/visualize_rgb.py`) shows two more XArms on the horizon at lift height.
Cause: soft/MPM scenes must use the per-env bound-camera path with `env_separate_rigid=False` (the
rasterizer cannot separate MPM geometry per env), so at `ENV_SPACING = 2.5 m` each env's camera
sees its neighbours. **The point cloud has never been affected** — it is cropped to
`[0.2,-0.215,0.004]–[0.71,0.215,0.45]` — which is exactly why this never surfaced before. RGB has
no crop. It is consistent between train and eval so it does not bias the π0.5-vs-DPPO comparison,
but it is a sim artifact absent on the real rig, and the neighbours MOVE (dynamic distractors).
Options: accept+document / crop the RGB to the workspace / collect at `num_envs=1` (~8× slower).
Recommended: accept and document, since both arms of the comparison see it.

*Method note: this was found only by RENDERING THE RECORDED OBSERVATION STREAM and looking at it —
not by any assertion. The collector's own `videos/` come from a separate free-flying camera and
would never have shown it.*

### 2026-08-30 — ⚠ TWO TODOs BEFORE THE SERIOUS MULTI-OBJECT RGB COLLECTION (user, 2026-08-30)

Both found by LOOKING at the rendered observation streams. The current 250-episode mushroom
collection (job 1797457) KEEPS RUNNING and is fine as a smoke/plumbing dataset — the user's
decision. These must be fixed before the multi-object collection that real comparisons rest on.

**TODO 1 — WRIST CAMERA IS INSIDE THE GRIPPER.** `xarm7_config.EE_T_CAM_WRIST` is an **identity
placeholder**, not a calibration: the file's own TODO says *"must be replaced with calibrated
transform"* and *"NOT used by the current rig (no wrist camera)"*. With identity the camera sits at
the gripper BASE-LINK origin looking along +tool-z at fingertips 171 mm away — i.e. embedded in the
gripper body, which is exactly what the wrist frames show (fingers filling the top and bottom
edges). ⚠ **I earlier described this as "the calibrated OpenCV extrinsic" — WRONG, and the user
caught it.** (The separate OpenCV→OpenGL flip earlier today was still necessary and correct; it
turned a backwards camera into a forward one, but cannot fix a placeholder pose.)

*Fix:* adopt the calibrated matrix CLAUDE.md already documents but which never reached the code —
translation `[+0.07132349, -0.00272051, -0.16624549]`, optical axis ≈ +tool-z. That puts the camera
**337 mm behind the fingertips and 71 mm off-axis: outside the gripper, looking past the jaws**,
which is what the user asked for ("a bit outside of the gripper in the axis direction"). If that
matrix is judged stale, fall back to a simple standoff along −tool-z. Either way, re-run the three
geometric checks from `.agent_tmp/test_rgb_obs.py` (extrinsic round-trip, forward axis, near
geometry) AND look at a frame — the round-trip check passes for ANY self-consistent pose, including
a placeholder, so it cannot catch this class on its own.

**TODO 2 — NEIGHBOURING ENVS VISIBLE IN cam_ext.** At `ENV_SPACING = 2.5 m`, soft/MPM scenes must
use the per-env bound-camera path with `env_separate_rigid=False` (the rasterizer cannot separate
MPM geometry per env), so each env's external camera sees its neighbours on the horizon — and they
MOVE, so they are dynamic distractors. The point cloud was never affected (cropped to
`[0.2,-0.215,0.004]–[0.71,0.215,0.45]`), which is why this never surfaced before RGB.

*Options, best first:*
1. **Backdrop occluder** — add a wall/plane fixture behind the workspace, outside the robot's
   reach. Occludes the neighbours AND makes the scene more real-lab-like (a real rig has a
   background), so it reduces the sim2real gap rather than just hiding an artifact. Cheap.
2. Raise `ENV_SPACING` — neighbours shrink but stay in frame. Partial.
3. Collect RGB at `num_envs=1` — clean, ~8× slower. Only if 1 proves insufficient.

*Method note applying to BOTH: neither was caught by any assertion. Both were caught by rendering
the recorded observation stream to video and looking at it. The collector's own `videos/` come from
a separate free-flying camera and would never have shown either one.*

### 2026-08-30 — π0.5 TRAINS ON OUR DATA (smoke PASS), OOM was a SLURM allocation bug, full run chained

**SMOKE PASS (job 1797898):** restored `pi05_base` params (12.5 GiB) in 4.2 s → **10/10 steps at
2.2 s/it** → blocking save 5.65 s → *"Save Finalize is done on all hosts"* → checkpoint finalized as
`.../pi05_smoke/9`, `rc=0`. End to end on our data with **openpi unmodified**.

**THE OOM WAS OURS, NOT openpi's.** The previous attempt (1797884) trained all 10 steps and was then
**OOM-killed during the checkpoint save**, leaving a `9.orbax-checkpoint-tmp-0` that never
finalized. Orbax stages params (12.5 GiB) + train_state (37.5 GiB) into **HOST** memory to write,
and the job had SLURM's default per-CPU allocation of ~102 GB — while the node has ~485 GB. Adding
**`#SBATCH --mem=0`** fixed it outright. Checkpoint size is set by MODEL size, not dataset size, so
this does not regress on the 250-episode run.

⚠ **Note the failure shape: it trains fine and dies at SAVE.** A long run would have burned hours
and produced nothing. Smoke tests must therefore run far enough to WRITE A CHECKPOINT — a smoke
that stops at "loss went down" would have passed this and taught us nothing.

*GPU-side fallbacks, from openpi's README, if ever needed (host RAM was the issue here, not GPU):*
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (now set), `--fsdp-devices 4`, or disable EMA. Full fine-tune
needs >70 GB and the GH200 has 97.8 GB, so it fits on one GPU; LoRA (>22.5 GB) is not required.

**FULL RUN CHAINED (survives the session — SLURM dependencies, not a watcher):**
| job | what | depends on |
|---|---|---|
| 1797457 | collection, 250 eps → `26-08-30-lyr` | — |
| **1797916** | convert BOTH variants + norm stats → `dataset/lerobot/gm/mushroom250_{ext_wrist,ext}` | afterok 1797457 |
| **1797917** | fine-tune `pi05_mushroom250_ext_wrist` | afterok 1797916 |
| **1797918** | fine-tune `pi05_mushroom250_ext` | afterok 1797916 |

Training: stock `pi05_libero` + CLI overrides only, `--batch-size 32 --num-train-steps 12000
--save-interval 2000`, 24 h wall. Steps/batch are a JUDGEMENT CALL, not a tuned value — 250 eps ×
~220 frames ≈ 55k frames, so 12k steps at batch 32 is ~7 epochs. `save-interval 2000` means usable
checkpoints exist even if the wall clock runs out. Revisit once the first loss curve is visible.

`afterok` means a FAILED collection silently cancels the whole chain rather than training on
partial data — the intended behaviour, but check `squeue`/the run dir rather than assuming
training started.

### 2026-08-30 — π0.5 EVAL RUNS THROUGH THE CANONICAL HARNESS (smoke PASS, plumbing only)

`gentle_manip/pi05/eval_harness.py` + `eval_policy.py` route a fine-tuned π0.5 checkpoint through
`gentle_manip.evaluation.run_eval` over the SAME `GenesisMultiStepVecEnv` + `serl_sim_server`
bridge every DPPO/DP3 eval uses (CLAUDE.md hard requirement #1), so π0.5 and DPPO face identical
scenarios, seeds, DR and metrics. Verified on the 10-step smoke checkpoint (job 1798196):
5 episodes, per-episode video (750 frames each), `summary.json` + `episodes.csv`, **success 0.000 —
which is the CORRECT result for 10 training steps.** This validated the path, not the policy.

**Modelled on `eval_dp3_harness.py`**: DP3 also normalizes internally, and solves it by building the
venv with IDENTITY normalization so the venv's normalize/unnormalize become no-ops. Same trick here,
so π0.5 emits actions directly in the ActionPipeline [-1,1] space — the space
`derive_action_set` wrote into the LeRobot dataset. Train and eval agree by construction.

**FIVE integration defects, each one layer deeper, all in OUR code (openpi untouched):**
1. **norm stats looked up under the wrong asset id.** `create_trained_policy` resolves
   `assets/<asset_id>/norm_stats.json` with `asset_id = assets.asset_id or repo_id`; passing the
   STOCK `pi05_libero` config looked for `physical-intelligence/libero`. The checkpoint stores them
   under OUR repo_id. Fixed by inferring repo_id from the checkpoint's own `assets/` tree, so it
   cannot be passed inconsistently with the checkpoint being loaded.
2. **`--view student` has no images.** Added an ADDITIVE `pi05: [images]` view (teacher/student
   untouched) → `[ee_pos, ee_quat, gripper_width, image_cam_ext, image_cam_wrist]`, no point cloud
   (π0.5 ignores it, and dropping it avoids a per-step depth render).
3. **the sim server never set `render_rgb_obs`.** Now DERIVED from the view
   (`obs_cfg.images is not None`), not a new CLI flag, so render and requested keys cannot
   disagree. `None → False` for every existing experiment ⇒ current runs byte-identical.
4. **`PolicyEnv` needs `rgb_shape` when `images` is set** and the server passed None. Now derived
   from the task's `scene_spec` cameras, asserting every image camera agrees (the obs space
   declares ONE (H, W)).
5. **the venv does not expose raw `ee_pos`.** It packs proprio into a stacked `state` array in
   PROPRIO_VIEW order — the same 8-dim vector the LeRobot `state` feature was written with — with
   an n_obs_steps axis on every modality. The policy now takes the LAST obs step.

*None of these would have been found by reading code; each needed a run. Worth the five iterations
now, on a throwaway checkpoint, rather than after a multi-hour fine-tune.*

**Known cost, stated so it is not mistaken for a property of π0.5:** openpi's `Policy.infer` takes
ONE observation, so `act()` loops over the 5 envs. A 5-episode eval took ~10 min. The canonical
200-episode protocol will be slow; batching inference is the obvious optimisation if it matters.

### 2026-08-30 — LARGE-SCALE v4 COLLECTION LAUNCHED (6 categories x 500 eps) + a reproducibility fix v4 was missing

Merged the local agent's v4 work (`origin/master` 12c0dd9) into `integrate-all-2026-08-29`. **No
conflicts** — the anticipated raspberry clash did not occur because they had already merged our
branch (e8fbb20). `collect_demos_synth_v3.py` was auto-merged with edits from BOTH sides; verified
by grep that ours survived (RGB passthrough, `_want_rgb`, `image_quality`, `encode_images`,
`dataset_idx`, `cma_seed`) alongside theirs (`regrasp`, `episode_type`).

**⚠ v4 WAS MISSING THE CSV<->DATASET JOIN, and it is not a cosmetic gap.** v4 writes a
`dr_params.csv` row for EVERY attempt, inline, before the save loop — with no `dataset_idx`. And a
`success=1` row still may not be saved: the `n_episodes` cap truncates the last batch, and v4
silently drops "succeeded-by-crushing" fallback demos (`total_fallback_dropped`). So "the successes,
in order" is WRONG, and the DR parameters could not be matched to the episodes they produced. This
is exactly the defect the user had me fix in v3 ("Make sure the recorded stuff are reproducible"),
about to be baked into 3000 episodes. Ported the v3 mechanism (buffer rows → stamp `dataset_idx` in
the save loop → write after, −1 for unsaved) and added the `scan_metric` column that guardrail #3
requires. Both **APPENDED** to the header so no existing column index shifts. Validated on a
6-episode post-patch run: `dataset_idx` contiguous from 0, `scan_metric`=p98.

*Process note: the six smokes were launched BEFORE the patch, so none of them exercised it — a
separate tiny run was needed. Also: I edited v4 while three smokes were still running. They were
unharmed (ProcessPoolExecutor forks rather than re-imports), but it was an avoidable risk I had
explicitly declined to take earlier in the day with v3. Don't edit a module that running jobs may
re-import.*

**SMOKE VERIFICATION (16 eps, `--mesh-cycle`, per category — guardrail #1). 5/6 match or beat the
handoff's expectations; sub-yield meets or beats target EVERYWHERE:**

| category | exp succ | got | exp sub-yield | got | peak top10/yield (med/max) |
|---|---|---|---|---|---|
| raspberry | ~100% | 100% | ~88% | 87.5% | 0.81 / 1.16 |
| cherry_tomato | ~75% | 70.8% | ~80% | **87.5%** | 0.82 / 1.20 |
| tomato | ~80% | **50%** | ~100% | 100% | 0.35 / 0.91 |
| tofu | ~65% | **87.5%** | ~100% | 100% | 0.42 / 0.77 |
| strawberry | ~45% | 45.0% | ~94% | **100%** | 0.32 / 0.74 |
| banana_chunk | ~40% | 40.0% | ~100% | 100% | 0.40 / 0.71 |

tomato is the one deviation (50% vs ~80% success) but its sub-yield is perfect — a THROUGHPUT
difference, not a data-quality one (~1000 attempts for 500 eps ≈ 18 h, inside the 48 h allocation).
Material DR confirmed live (2–4 distinct `mat_E` per smoke — "silently inert" was a real past bug).

**LAUNCHED (500 eps/category, user's number; the handoff specifies 250):** raspberry 1800346,
cherry_tomato 1800347, tomato 1800348, tofu 1800349, strawberry 1800350, banana_chunk 1800351.
Recipe = handoff verbatim + `--regrasp-prob 0.2` (user) + `--scan-metric p98`, no manual
`--closure-gain`. NO pasta_bundle; mushroom NOT recollected (the 26-08-28-jgr 250-set stands).
`-t 48:00:00`, `--mem=0`. Wall-clock estimated from the mushroom run's measured 1.06 min/attempt:
raspberry ~9 h → banana_chunk ~22 h.

**Still open after collection (handoff §Filtering):** `filter_pinch_episodes.py` + drop episodes
with `priv_stress` top10 ≥ 1.0 (expect ~12% on raspberry and cherry_tomato, ~0 elsewhere — the
smokes' 87.5% sub-yield on both predicts exactly that).

### 2026-08-30 — π0.5 BASELINE: FIRST NUMBERS (20-ep screen). Underpowered; the real finding is lift-then-drop

Both variants fine-tuned 12000 steps on the 250-episode mushroom set (batch 32, 1.9 it/s, ~1.7 h
each; checkpoints finalized at `11999`, rc=0). Screened through the CANONICAL harness, 20 episodes,
num_envs 5.

| variant | success | ever | SUSTAINED stress | succ eps |
|---|---|---|---|---|
| ext+wrist | **0.400** | 0.65 | 23,782 | 8/20 |
| ext-only (DEPLOYABLE — no wrist on the real rig) | **0.150** | 0.45 | 22,033 | 3/20 |

**THE WRIST GAP IS NOT SIGNIFICANT AT n=20.** 8/20 vs 3/20 → two-proportion z ≈ 1.84, p ≈ 0.07;
95% CIs ~[0.19,0.64] vs ~[0.03,0.38], heavily overlapping. Per-batch success swung 0.00–0.80 in
BOTH arms (5 eps/batch). State it as "directionally favours the wrist, underpowered", never as a
result. Mid-run I called the early 0.40-vs-0.00 split "striking"; batch 3 reversed it (0.00 vs
0.40). *A 2-batch lead at 5 eps/batch is noise — do not narrate partial evals.*

**THE ROBUST OBSERVATION IS THE `ever` >> `success` GAP** (0.65→0.40 and 0.45→0.15, both arms):
π0.5 REACHES the success height band and fails to HOLD it for the required 30 steps. That is
lift-then-drop — a weak/slipping grasp, not a failure to find the object. Same signature this
project logged for CFG earlier (§0-CFG: "the 0.33 ever-minus-success gap is the lift-then-drop
signature"). It is the most actionable thing in these numbers.

**Against the DPPO plain baseline (~0.905 on mushroom), π0.5 is far behind — but the caveats are
load-bearing and were PRE-REGISTERED**: 12k fine-tune steps (openpi's own `pi05_libero` uses 30k),
250 episodes of ONE object, ONE scripted behaviour, ONE instruction, and a 20-episode screen.
A loss in this regime says more about our data than about π0.5. Do not report it as "π0.5 is worse"
without those qualifiers, and preferably not before the canonical 200-episode protocol.

**NOT RUN: the canonical 200-episode eval** (~6.7 h/variant at the current per-env inference rate)
— it would contend with four running v4 collections for a constrained GPU-minutes budget. Do it
after collection, and consider a longer fine-tune first.

*Artifacts:* `third_party/openpi/checkpoints/pi05_libero/pi05_mushroom250_{ext_wrist,ext}/11999`
and their `eval/26-08-30-172510/` (summary.json, episodes.csv, per-episode video).

### 2026-08-30 — ⚠ CORRECTION: the π0.5 screening numbers were measured on ONE GEOMETRY

Setting up the width probe exposed that `EvalSpec.scene_group_size` defaults to **0 = a single
fixed object geometry for the whole eval** (only pose/orientation vary per batch), and
`gentle_manip/pi05/eval_harness.py` never overrode it. So the screen I reported —
**ext+wrist 0.400 / ext-only 0.150** — ran all 20 episodes at `obj_scale` **1.407**, one shape,
one size. Verified directly: `episodes.csv` has exactly ONE distinct `obj_scale`.

**What this does and does not invalidate.** The two variants still faced IDENTICAL scenarios, so
the ext-vs-wrist comparison is internally consistent (and was already not significant, p≈0.07).
What is void is any comparison to numbers measured under the canonical 40-geometry protocol —
notably DPPO's ~0.905 on mushroom. Do NOT put 0.400 next to 0.905 in a table: they are different
protocols, which is precisely the B1 error class this project keeps re-learning.

**Fixed:** `--scene-group-size` added to the π0.5 harness, **default 1**, and it now prints the
resulting distinct-geometry count at startup. `docs/CHECKLISTS.md` §3.1 makes "check `obj_scale`
has >1 distinct value in episodes.csv before believing any size-related number" an explicit step.

*LESSON: a default that silently narrows the experiment is worse than a missing argument. I chose
`EvalSpec()` defaults deliberately "to stay canonical" — but the canonical trio is
(n_episodes, num_envs, seed); `scene_group_size` is NOT part of it and defaults to the degenerate
value. Check what a default actually IS, rather than trusting that a config named canonical is
canonical in every field.*

### 2026-08-30 — WIDTH PROBE: binned design added (user request), method written into CHECKLISTS §3.1

`gentle_manip/pi05/width_probe.py` + `configs/dr/wprobe_{1p000,1p125,1p250,1p375,1p500}.yaml` +
matching experiments. Pins `object_scale` to 5 levels across the DR range, each asserted in code to
differ from the reference experiment by `object_scale` ALONE; `scene_group_size=1` keeps shape and
material varying so every bin yields several DISTINCT GEOMETRIES at a FIXED size.

**Why binned beats random draws here:** (a) LEVERAGE — slope SE ∝ 1/sd(x), so pinning the extremes
reaches the precision of ~40 random draws clustered mid-range with fewer geometries; (b)
DE-CONFOUNDING — under random DR, size and shape co-vary, and pinning size isolates the size term.
It buys precision, NOT immunity from the 40-geometry rule: a borderline CI at low k is UNRESOLVED,
not negative, and `width_probe.py` prints that warning itself when k < 40.

π0.5 now writes the SAME dump format as the DPPO probe (`GM_WIDTH_DUMP` →
`.agent_tmp/<tag>_widthcmd_b*.npz` with `width_cmd_mm` + `ee_z_m`), so one analysis serves every
policy. Launched on ext+wrist: jobs 1805040-1805044, 5 bins x 15 eps.

### 2026-08-30 — NOTE: π0.5 CANNOT be trained on the REAL demos — they carry no RGB

Checked every real-demo pkl (`single_lift_mushroom_real*`, incl. the 9 mm-x-shift-corrected
`..._real_merged_shift9mm`, 55 episodes, 7-dim actions recorded natively). Observations are
`[ee_pos, ee_quat, gripper_width, point_cloud]` — **no image keys anywhere**. They were collected
for a point-cloud student, so RGB was never recorded.

π0.5 is image-conditioned, so the real-data path is blocked until real demos are RE-RECORDED with
RGB. That is feasible whenever the rig is next available — the L515 (`cam_ext`) already streams
colour, and `record.py` goes through `PerceptionPipeline`, so it needs only an obs config with an
`images:` block (the `superset_soft_armfocus_rgb` fork is the template). No wrist camera exists on
the rig any more, so a real π0.5 would be the EXT-ONLY variant — which is the deployable arm we are
already measuring in sim.

*Do not "solve" this by rendering pseudo-images from the point cloud: the appearance would match
neither the real camera nor the sim training distribution, and a policy trained on it could not be
deployed against a real RGB stream in any meaningful sense.*

### 2026-08-30 — "the mushroom is black in eval but white in the demos" — record camera ≠ policy input

User asked whether the π0.5 comparison is fair given the object looks different. Checked three
frames from the same collection/eval:

| source | mushroom |
|---|---|
| collection VIDEO (free-flying record camera, close) | white/light speckled ball, prominent |
| eval VIDEO (record camera, farther back) | small dark speckled blob |
| **training OBSERVATION `image_cam_ext` — what π0.5 actually consumes** | **small DARK low-contrast object**; frame mean 46/255 |

**Verdict: fair on this axis.** The "white" is the COLLECTION RECORD CAMERA, which is not the
policy's input and is framed differently in collection vs eval. The policy's actual input has
always shown a small dark object, consistent between training and eval (identical `cam_ext` entry,
same task config). *Generalises: `videos/` and `render/*.mp4` come from a separate free-flying
camera. NEVER reason about train/eval appearance from them — decode `image_*` from the dataset and
dump the eval obs instead.* (Same trap as 2026-08-30's wrist camera: the collector's videos would
never have shown that bug either.)

**Two honest caveats, recorded rather than smoothed over:**
1. Compared training-obs against eval-VIDEO, not eval-OBS. Identical camera specs make them match
   by construction, but it is not directly measured. Dump eval obs RGB if certainty is needed.
2. **There IS a real observation-level train/eval difference**: `cam_ext` sees neighbouring parallel
   envs (the leakage TODO), so the BACKGROUND depends on `num_envs` — collection ran **8**, the
   screening eval ran **5**, the width probe runs **8**. The probe matches training; the screening
   eval did not. Another reason those screening numbers are soft, on top of the single-geometry bug.

**Worth carrying into the writeup:** in the external view the object is small, dark and low-contrast.
Consistent across train and eval so it does not bias the comparison, but a plausible contributor to
π0.5's weak absolute numbers — and exactly what a close, bright wrist view should help with, which
is the hypothesis the ext-vs-ext+wrist arms test.

### 2026-08-30 (night) — OVERNIGHT PLAN, left durable in case the session dies

**RUNNING (21 jobs).** v4.1 collections, 500 eps each, 13 objects: 7 food (raspberry,
cherry_tomato, tomato, tofu, strawberry, banana_chunk, mushroom) + 6 primitives (cylinder, sphere,
lamp, cuboid, ellipsoid, torus). Every 500-run is gated `afterok` on its own 16-ep `--mesh-cycle`
smoke, and smokes now record FULL renderings (upstream rule c03f9f3; my earlier smokes used
`--record-video 4` — a violation). Recipe verbatim: p98, `--regrasp-prob 0.2`,
`--grasp-extra-close auto`, no manual `--closure-gain`. 500 (not the handoff's 250) so all 13
objects weigh equally in the generalist.

Plus the π0.5 low-data pair: 25 mushroom + 25 tofu RGB episodes, **wrist camera 12 cm outside the
gripper + BLACK backdrop** (user-approved from `.agent_tmp/obs_fixed_mushroom`).

**THEN, unattended, in order:**
1. Verify each collection (schema incl. `dataset_idx`/`scan_metric`, per-object sub-yield, success
   vs the handoff table). Recover `data.pkl` with `scripts/merge_shards.py` for anything cut at its
   walltime — shards survive, only the final merge is lost.
2. **Document the final dataset composition** (per object: run dir, episodes, success, sub-yield).
3. `.agent_tmp/build_generalist13.sh` — convert each slice with the generalist's derive flags
   (7d euler target ← 10d rot6d source), merge, then launch the ×3 paired-reg generalist.
   * **normalization is JOINT and AFTER the merge** (user requirement). `merge_npz_datasets`
     already does this; the build now ASSERTS it — merged actions must SPAN [-1,1] (fitting inside
     is not enough; per-slice normalization would also fit) and the joint stats must differ from at
     least one slice's.
   * **EPOCHS ARE SOLVED, NOT COPIED.** ddgrl = 350 epochs × (254,340/128) = **695,461 gradient
     steps**. The 13-object set is ~5× larger, so 350 epochs would be ~5× the budget — a different
     experiment. Solve `n_epochs` from the realised `train.npz`; scale `save_model_freq` by the same
     ratio (projected 69 epochs / save every 10). Recompute at build time: per-object success rates
     differ, so the realised count is not 13×500×199.
   * paired term uses `paired_cube3_clouds_shift9.npz` (9 mm corrected; ddgrl used the UNcorrected
     one). ⚠ residual **−8.4 mm x / +7.1 mm y** remains — the regulariser is being asked to equate
     clouds that still differ. Flagged, not silently shipped.
   * NO real co-training (real exists only for mushroom).
4. π0.5 low-data: convert 50 demos → two FULL fine-tunes (ext-only, ext+wrist) at openpi's
   documented small-dataset budget (20k @ batch 64) → controlled width-size probe
   (`GM_FIXED_SCALES` + `GM_FIXED_POSE` + `GM_FIXED_YAW_DEG`, CHECKLISTS §3.2).
5. **THEN LoRA** (user-approved): `gentle_manip/pi05/train_lora.py`, same two variants, as the
   COMPARISON to the full fine-tune — 50 demos is exactly where a full fine-tune of a 3B VLA
   overfits. LoRA is impossible via openpi's CLI (`freeze_filter` is an nnx Filter object, and
   setting only the variant strings would add adapters while freezing NOTHING), so the config is
   built in our code from their classes and handed to their unmodified `main()`.

**OPEN QUESTIONS FOR THE MORNING (do not silently resolve):** the paired-cloud residual above;
and whether the dark, low-contrast object in `cam_ext` is depressing π0.5's absolute numbers — the
backdrop fixed the background, not the object's own lighting.

### 2026-08-31 — ⚠ RASPBERRY v4.1 500-ep: 55% OF SAVED DEMOS EXCEED YIELD, and it is SIZE-DEPENDENT

`26-08-30-iqe`, 500/500 saved, 99.0% demonstrator success — but **sub-yield 223/500 (44.6%)**,
median peak top10 **1.02×** yield, max 1.28. The 16-ep smoke said 87.5% sub-yield and the handoff
expected ~88%; its filtering step budgets "≤12% on raspberry". We have 55.4% over-yield.

**IT IS A SIZE EFFECT, NOT NOISE.**

| stratum | over-yield |
|---|---|
| small third (`scene_scale` < 0.93) | **85.6%** |
| large third (`scene_scale` > 1.12) | **19.5%** |

`corr(peak, scene_scale) = -0.524`, `corr(peak, grasp width_mm) = -0.621`,
`corr(peak, closure_cmd_mm) = +0.468`, `corr(peak, mat_E) = +0.267`. Re-grasp and standard episodes
are equally affected (54.3% vs 55.7%), so the `--regrasp-prob` start state is not the cause. The
v4.1 p98 closure rule simply commands too firm a grasp on SMALL raspberries — and raspberry is
already the smallest category (~1.5 cm), so scale 0.93 is a ~1.4 cm berry.

**⚠ DO NOT "FIX" THIS BY FILTERING.** Dropping the 277 over-yield episodes leaves 223 that are
SIZE-BIASED toward large berries — which would corrupt precisely the width-vs-size analysis this
dataset exists to support (§3.1/§3.2), and would do so invisibly. A size-biased slice is worse than
a smaller unbiased one.

**WHY THE SMOKE MISSED IT.** 16 episodes, and `--mesh-cycle` cycles MESHES, not SCALES — the draw
happened to land on kinder (larger) samples: smoke median peak 0.81 vs 1.02 over 500. *A 16-episode
smoke cannot see a stratified failure that only appears in one third of the size range.* The
guardrail is still worth keeping (it caught torus's walltime), but it certifies "runs and is
schema-clean", not "the data is gentle across the DR range".

**THIS ALSO QUESTIONS THE p98-vs-masked DECISION.** Upstream chose p98 over `masked` specifically
because "the raspberry saves demos at only 56% sub-yield" under masked. At 500 episodes, p98 on
raspberry gives 44.6% sub-yield — i.e. p98 is behaving on the full run the way masked did on the
smoke that disqualified it. Both metrics were compared on 16-episode runs; that sample cannot
resolve a size-stratified effect either.

**OPEN — needs a decision, not a silent workaround:** (a) re-collect raspberry with a size-aware
closure floor, (b) drop raspberry from the generalist, or (c) keep it and report the dataset's
sub-yield honestly per object. Every other finished object is clean: tomato 99.8%, tofu 100.0%,
mushroom 99.8%.

### 2026-08-31 — RASPBERRY: pinch-filtered and KEPT (user decision), with the residuals documented

User's call: filter and keep, rather than re-collect or drop. `26-08-30-iqe` → `-iqe-filt`.

| | before | after pinch filter |
|---|---|---|
| episodes | 500 | **307** (193 dropped) |
| sub-yield | 44.6% | **60.9%** |
| peak top10/yield (median) | 1.02 | 0.94 |

**The filter finds real artifacts.** 81.3% of DROPPED episodes were also over-yield vs 39.1% of
KEPT — pinches (object dangling from the fingertips, near-minimum width) and crushes are largely
the same episodes, which is what the user saw in the rendered videos.

**TWO RESIDUALS THAT MUST TRAVEL WITH THIS DATASET:**
1. **120 of the 307 survivors (39.1%) still exceed yield.** Filtering mitigates, it does not fix.
2. **The kept set is SIZE-SKEWED.** Small raspberries are both likelier to be pinched AND likelier
   to be crushed, so filtering removes them preferentially: small-third share **32.0% → 14.0%**,
   median `scene_scale` 1.027 → 1.105. Acceptable as a mild distribution shift for GENERALIST
   TRAINING; **NOT acceptable for a raspberry width-vs-size claim** — 14% small examples cannot
   support a size slope, so raspberry is EXCLUDED from per-object width-size conclusions (§3.1/3.2).
3. Raspberry therefore contributes **307** episodes where every other object contributes 500 —
   unequal weighting, to be stated wherever the dataset is described.

**⚠ BUG FOUND AND FIXED IN `filter_pinch_episodes.py`:** it copied the source `dr_params.csv`
verbatim into the filtered run dir, so `dataset_idx` still indexed the UNFILTERED episode order
(0..499) against a 307-episode `data.pkl`. Every DR-param↔episode join on a filtered dataset would
have paired the WRONG rows — silently, because the stale indices are all individually valid. Now
remapped (kept rows renumbered 0..n-1, dropped marked -1) and verified contiguous. Same broken-join
class as the v3/v4 `dataset_idx` fixes; that makes three occurrences, so **treat "does the join
survive this transformation?" as a standing check for any script that subsets episodes.**

### 2026-08-31 — PRIMITIVES SWITCHED TO MUSHROOM MATERIAL (upstream 38b46a0/fdbc320)

Upstream added additive `single_lift_prim_*_mush_soft_abs_action_armfocus` variants (mushroom
material) and the instruction is to collect THOSE, not the plain `prim_*` (tofu material). Their
measurements: cylinder & sphere **53% → 100%** success, lamp 57% (geometry-limited, collect
anyway), cuboid/ellipsoid/torus smokes still running on their side, "torus only if wall-clock
acceptable".

**Cancelled all six tofu-material primitive jobs** (prim_lamp had ~18/500 — discarded, wrong
material) and relaunched the `_mush` variants: smoke (16 eps, `--mesh-cycle`, FULL renderings) →
500-run gated `afterok`. Diff confirmed the `_mush` experiments differ from the plain ones by
exactly `task` and `dr` (mushroom material + its DR ranges); recipe verbatim otherwise.

**torus is deliberately NOT auto-released.** Its tofu-material smoke measured 20% success →
32.6 h projected for 500 eps, which is why its walltime had already been raised to 44 h. Upstream's
own caveat is "only if wall-clock acceptable", so its 500-run is HELD until I measure the `_mush`
smoke's rate — an `afterok` gate checks the exit code, not whether the job can finish in time.

### 2026-08-31 — WIDTH-SIZE PROBING, CONTROLLED: the earlier 0.50 slope may have been a POSE ARTIFACT

First fully-controlled probe (`GM_FIXED_SCALES` + `GM_FIXED_POSE` + `GM_FIXED_YAW_DEG`, 5 sizes x
8 eps, all 5 dumps written after the final-batch flush fix):

| model | probe | slope | 95% CI | R2 | verdict |
|---|---|---|---|---|---|
| 250-demo mushroom, ext+wrist | **UNcontrolled**, 4 bins | **0.50** | [0.26, 0.73] | 0.81 | ADAPTS |
| 50-demo mixed, ext+wrist | **CONTROLLED**, 5 bins | **0.04** | [-0.05, 0.13] | 0.14 | NO ADAPTATION |

**These two are NOT comparable — they differ in TWO ways at once** (training data 250-mushroom vs
50-mixed, AND probe controls). I will not attribute the drop to data size on this evidence.

**The likelier reading is that the 0.50 was an artifact.** §3.2's whole point is that pose/yaw leak
into the size slope when unpinned; the uncontrolled run had pose free to vary across only 4 usable
size bins, which is exactly the configuration where a chance pose-size correlation manufactures a
slope. The controlled run, with pose pinned per sub-env, finds nothing.

**Disambiguating run launched:** the SAME 250-demo checkpoint under the FULL controls. If it also
comes back ~0, the 0.50 was pose leakage and must be retracted; if it stays ~0.5, the difference is
genuinely data size and the low-data regime loses size tracking. Either way the uncontrolled number
does not stand on its own.

*Method note: this is the second time an uncontrolled width measurement produced a confident-looking
slope that a controlled one erased — the first was the 12-vs-40-geometry reversal (DEVLOG
2026-08-27). Width slopes are unusually good at manufacturing false positives.*

### 2026-08-31 — the controlled 250-demo probe's 0.000 was a CAMERA MISMATCH; π0.5 pipeline verdict

`EE_T_CAM_WRIST` changed at 08-30 23:01 (identity → 12 cm outward). The 250-demo model's data
(15:02) and training (17:19) both predate it, so the controlled probe at 08-31 11:48 showed it a
wrist view it had never seen: **0.000/40**, against 0.425 on the uncontrolled probe at 22:20 the
night before — which ran BEFORE the change. Not a result about controls, data size, or adaptation.

**What this costs:** the 250-vs-50 comparison is unavailable without re-collecting and retraining
(~7 h), and the 0.50 slope is now unreproducible because the configuration that produced it no
longer exists. User's call (2026-08-31): don't rebuild it — the pipeline is proven, move on.

**WHAT STANDS (data and eval share the 12 cm camera, so internally valid):**
- π0.5 trains and evaluates end-to-end on our data with **openpi completely unmodified** —
  stock `pi05_libero` + CLI overrides for training; our adapters only for conversion, norm stats
  and the canonical-harness eval.
- 50-demo (25 mushroom + 25 tofu), ext+wrist, fully-controlled probe: success 0.225,
  width slope **0.04 [-0.05, 0.13]**, R² 0.14 → **no size adaptation**.
- The `ever == success` equality in that run (0.225/0.225) differs from the 250-demo model's
  `ever` >> `success` lift-then-drop signature — worth a look once there is a comparable pair.

**Retracted:** the 250-demo width slope 0.50 and the ext-vs-wrist gap derived from the uncontrolled
probes. Both were measured under a camera that no longer exists, on top of the pose-control caveat.

---

### 2026-08-31 — POLICY SELECTION UNDER NON-DOMINANCE: adopt a damage-rate CONSTRAINT, not a weight

**Question (user):** two policies, neither Pareto-dominating the other — safety says pick the
gentler, performance says pick the more successful. Fixed weight? But the right weight depends on
the material. Should the criterion be sub-yield on `top20_ttop20`?

**ANSWER: yes to the statistic, no to the weight.** Written up as **CHECKLISTS §3.3**, adopted as
the standing selection criterion and the intended paper presentation format. User is taking it to
colleagues, so §3.3 is explicitly open for refinement.

**Criterion: maximize success subject to `damage_rate ≤ ε`,** where
`damage_rate = mean(stress_top20_ttop20 / mat_yield ≥ 1.0)` over episodes, aggregated across
objects by **max**, not mean.

Why a constraint rather than `success − λ·stress`:
- λ has units (success per Pa) — a value tuned on mushroom cannot transfer to tofu or raspberry,
  which is the very material-dependence it is supposed to abstract away;
- yield is a PHYSICAL boundary (elastic/recoverable below, plastic/permanent above), so exceedance
  is a binary with a non-arbitrary cutoff. ε is then one dimensionless, auditable number.

**NEW MEASUREMENT — peak stress is empirically dead as a ranking axis.** Across the 200-episode
mushroom evals on disk, `stress_max_tmax / mat_yield ≥ 1.0` in **91–100% of episodes for EVERY
policy**. Zero discrimination. The plan's "no PEAK comparisons (past saturation)" rule was argued
from the 1.23–1.34× yield figure; this is the measured form of it. SUSTAINED damage rate spans
5.0–66.0% over the same runs, i.e. it discriminates by an order of magnitude more.

**NEW MEASUREMENT — the mean hides the tail.** lulkx `slope_base` (0.905, mean sust/Y 0.69) and
luqsl `state_249_eval235` (0.900, 0.66) are indistinguishable on success and mean stress but differ
by **10 points of damage rate** (25.0% vs 15.5%, ±6.0 / ±5.0 at n=200). Reporting the mean was
averaging away the only axis that separated them.

**A SHARPER JUSTIFICATION FOR BINARIZING than "it's simpler".** Above yield the MPM model has no
plasticity, so the MAGNITUDE of an over-yield von-Mises value is an elastic extrapolation past the
regime where the constitutive law holds — it carries no physical information. Only the FACT of
exceedance does. So thresholding at yield is not a coarsening of a good continuous signal; it is
the correct use of a signal that is only valid sub-yield. This is the same footing as the standing
"von Mises is a proxy valid sub-yield only" caveat, now turned from a limitation into a method.

⚠ **The §3.3 illustration table is NOT a ranking** — its five rows sit on different eval protocols
(`slope_base` vs `eval235`, two of them `dppo-finetune`). Kept because it demonstrates the metric's
discrimination; labelled in place so it cannot be misread as a result. A real ranking needs all
arms on one protocol.

**TODO:** add `dmg_rate` + CI as a default eval-summary column so it is produced at eval time
rather than re-derived per analysis.

### 2026-08-31 — LoRA at 50 demos: 0.000 success, strictly worse than full fine-tuning

Controlled width probes 1852065 / 1852066 landed. All four arms on the SAME probe (40 episodes,
`GM_FIXED_SCALES` + `GM_FIXED_POSE` + `GM_FIXED_YAW_DEG`, checkpoint step 19999):

| arm | success | ever | sustained stress |
|---|---|---|---|
| full-FT ext | 0.025 | 0.025 | 17,333 |
| **full-FT ext+wrist** | **0.225** | 0.225 | 18,820 |
| LoRA ext | **0.000** | 0.000 | — |
| LoRA ext+wrist | **0.000** | 0.000 | — |

**VERIFIED NOT A PIPELINE DEFECT** (checked before reporting, because a hard 0.000 usually is one):
- LoRA training CONVERGED — loss 0.093 → 0.0008 over 20k steps, both arms;
- `action_dim=7`, `action_horizon=10` probed correctly at eval — no action-space mismatch;
- **`norm_stats.json` byte-identical between the LoRA and full-FT checkpoints** for both repo_ids
  (`cmp` on `*/19999/assets/gm/lowdata50_*/norm_stats.json`) — the assets-dir trap is ruled out;
- same `repo_id`, same step count, same probe → the only difference is LoRA vs full FT.
- the arm is NOT frozen: LoRA emits a 44–50 mm gripper swing per episode and lifts 0.085 m mean.
  It moves, closes, lifts — and never carries the object.

**READING.** Train loss 0.0008 on 50 demos with 0.000 success is memorization that does not
transfer. The mechanistic story: LoRA freezes the base and applies low-rank updates, but our target
action space (**7-D absolute euler**) is not the space π0.5 was pretrained on; moving the action
expert to a new absolute space plausibly needs full-rank capacity. Fitting the 50 training
trajectories is achievable low-rank; generalizing to new geometries and poses is not.

**STATISTICS — only ONE of the two comparisons is conclusive.** ext+wrist 9/40 vs 0/40 is real
(p<0.005). ext 1/40 vs 0/40 is NOT separable at n=40; both are floor. Do not report "LoRA is worse
on both cameras" — report it for ext+wrist, and say ext is uninformative because the full-FT
baseline there is itself at the floor.

**CAVEAT that limits how far this generalizes:** this is the 50-demo low-data regime by design.
It does NOT establish that LoRA fails at 250 demos or on a delta-action space; it establishes that
LoRA does not substitute for full FT *here*. The LoRA arm is finished; no further runs planned.

### 2026-08-31 — The RGB scene was DARK because of a wall SHADOW, not because the wall blocked light

**User report:** the pi0.5 scene renders very dark; "the wall behind blocks the scene light".

**Actual mechanism — a shadow, and from the SIDE wall, not the back one.** Genesis' default
`VisOptions` is ONE `DirectionalLight(dir=(-1,-1,-1))` (light ARRIVING from +x+y+z),
`ambient_light=(0.1,0.1,0.1)`, `shadow=True`. A wall of height h at y=+1.2 therefore casts a
shadow band over `y in [1.3-h, 1.3]` at table height. At **h=1.5 that band is [-0.20, +1.30]**,
which swallows the entire workspace (`|y| <= 0.30`). The BACK wall (x=-0.55) is irrelevant: the
light comes from +x, so its shadow falls away from the robot.

Measured on the shipped RGB dataset (`26-08-30-edq`), whole episodes, every 5th frame:

| stream | mean | p95 | %<32 |
|---|---|---|---|
| cam_ext BEFORE | 31.7 | 90.0 | 57.7% |
| cam_wrist BEFORE | 41.2 | 90.0 | 59.3% |

**p95 pinned at exactly 90 in every stream** — even the brightest surfaces reached only 35% of
range, so this was global underexposure ON TOP of the shadow. Both had to be fixed.

**FIX 1 — walls 1.5 m -> 0.9 m.** At h=0.9 the shadow band is [+0.40, +1.30] and clears the
workspace. Occlusion is not weakened: cam_ext (x=0.989, VFOV 46 -> HFOV 59) sees the back wall
at 1.54 m where the **frame top is z=0.774**, so 0.9 m covers the full frame with 0.13 m margin;
and the side walls first enter frame at x=-1.13, i.e. already behind the back wall.

**FIX 2 — three-point lighting, scoped to backdrop scenes** (`scene_builder._backdrop_lighting`):
key light unchanged, plus a weaker y-opposite fill and a near-vertical top light, ambient
0.1 -> 0.35. Returns `{}` for non-backdrop scenes, so no point-cloud experiment changes. (Depth is
lighting-invariant, so this is belt-and-braces — but scoping means the claim needs no argument.)

**RESULT** (6-episode smoke, job 1855381, same measurement):

| stream | mean | p95 | %<32 | %>=250 |
|---|---|---|---|---|
| cam_ext AFTER | 73.6 (+42) | 254 | 10.0% (-48) | 5.99% |
| cam_wrist AFTER | 104.0 (+63) | 254 | 7.3% (-52) | 6.06% |

**The clipping is on the WHITE ARM, not the object** — checked, because +6% saturated pixels
would otherwise be a regression. Wrist-camera centre crop (where the mushroom sits):
**86 -> 245 distinct intensity levels**, %>=250 only **1.11%**, %<32 53.7% -> 21.3%. Nearly 3x the
tonal resolution on the thing the policy has to see. The MPM particle structure is now visible.

**Occlusion re-verified BY LOOKING, not assumed** (the standing lesson from the wrist-camera bug):
the backdrop region of cam_ext is uniformly dark with no bright robot blobs. A first automated
check reported "19% of upper rows > 100" — that was the WHITE ARM passing through the crop, i.e.
my test region was badly chosen, not a leak.

**One residual, benign:** a dark moving shape at frame left (temporal std 4.7, vs 5.6 in the
arm-sweep region — so it genuinely moves). It is DARK (mean 51) where a lit neighbour robot would
be bright, it sits just above the table line i.e. ON the wall, and geometry says neighbours at
y=+-2.5 are occluded by the back wall. It therefore reads as OUR OWN ARM'S SHADOW cast on the
backdrop — legitimate scene content that the real rig would also have. Stated as a reading, not a
proof: I verified the occlusion geometry, not the identity of the blob.

**ARTEFACTS:** `docs/figures/backdrop_lighting_before_after.png`,
`docs/figures/pi05_obs_AFTER_lit/` and `pi05_obs_BEFORE_dark/` (4 episodes each, the RECORDED
observation streams via `pi05/visualize_rgb.py`, not the collector's render camera).

⚠ **THE EXISTING RGB DATASETS WERE COLLECTED DARK.** `26-08-30-edq` (mushroom) and `26-08-30-vyi`
(tofu) — and therefore the 50-demo lowdata models and both LoRA runs — are all pre-fix. Any RGB
policy trained on them and evaluated under the new lighting is a train/eval MISMATCH, exactly the
class that invalidated the 250-demo model when the wrist camera moved. Re-collect before comparing
across the fix; do not mix.

### 2026-08-31 — Generalist-12 round: setup APPROVED and frozen (CHECKLISTS §1.1)

Full resolved setup written to **CHECKLISTS §1.1** as the reference record. Summary of the
decisions taken this session:

- **12 objects**, torus dropped (wall-clock), pasta_bundle never collected. All slices pinch+NaN
  filtered; **normalization applied AFTER the merge**, asserted in code.
- **Architecture MATCHES `ddgrl` exactly** — `[3072]x3`, `visual_feature_dim 512`,
  `category_embed_dim None`. **RETRACTED:** my earlier description of this round as "x3 network
  size vs ddgrl" was wrong — ddgrl IS already `[3072]x3`; the "x3" referred to ddgrl's size
  relative to the older `[1024]x3` baseline. Nothing is being scaled up.
- **Gradient budget held equal to ddgrl at 695,461 steps**, NOT epoch count. ddgrl's `train.npz`
  is 1,248 episodes / 254,340 transitions / 203.8 mean length -> 1,987 steps/epoch x 350. The
  12-object set is ~4.5x larger, so ~79 epochs. Matching EPOCHS instead would have given this run
  4.5x ddgrl's optimisation — epochs are not comparable across dataset sizes.
- **`save_model_freq = 8`** (user decision). ~10 ckpts/seed, ~4.8 GB over 3 seeds. Rejected
  `save_freq = val_freq` (~17 GB) as too expensive. Accepted consequence: the val minimum can sit
  up to 4 epochs (~5% of schedule) from the nearest checkpoint, so the val-min epoch must be
  reported alongside the chosen checkpoint's epoch. Revisit this number if the dataset grows.
- **Seeds 42, 27, 321**; **2 checkpoints each** (closest-to-val-min, later on a near tie; and
  last) = 6 evals, all on one protocol.

**NEW FINDING that motivated the checkpoint-pair design.** `ddgrl` uses `val_freq: 10` but
`save_model_freq: 50` — so its own val minimum at **epoch 280 has NO checkpoint** (saved set is
50,100,...,350). Copying the reference verbatim would have made "the checkpoint closest to the val
minimum" approximate by construction. Worth knowing for any re-analysis of ddgrl itself: its
best-by-val checkpoint is not on disk, and 250/300 are the only nearby options.

**Why evaluate val-min AND last:** both alzey and ddgrl finished PAST their val optimum (+16% and
+11% above their minima). Holding ddgrl's gradient budget inherits a schedule that also ends past
its optimum, so the pair measures directly whether that overtraining costs success or gentleness.

**Backdrop + 3-point lighting stays RGB-only** (user, 2026-08-31): "keep this only for
rgb-required case". Verified scoping — `backdrop: true` appears in exactly 2 of 32 task configs
(`single_lift_{mushroom,tofu}_soft_pi05rgb`); `_backdrop_lighting` returns `{}` for every other
scene and the walls are not built at all. The point-cloud generalist path is untouched.

### 2026-09-01 — Generalist-12 LAUNCHED: 3 seeds, provenance-guarded

**Dataset built and verified** — `dataset/dppo/single_lift_generalist_12obj/`:
train **4,931 trajs / 950,741 transitions**, val **551 trajs / 105,940 transitions**, `action_dim=7`,
range exactly [-1,+1], and **all 12 slices differ from the joint `action_min`** (i.e. normalization
was genuinely recomputed AFTER merging, not inherited from one slice). Build job 1857892, 15m29s.

**Resolved schedule** (solved from the realised `train.npz`, not assumed):
950,741/128 = 7,428 steps/epoch -> **n_epochs 94**, **val_freq 3**, **save_model_freq 8** (user).
**698,200 gradient steps vs ddgrl's 695,461 — within 0.4%.**

| seed | job |
|---|---|
| 42 | 1858149 |
| 27 | 1858150 |
| 321 | 1858151 |

**PROVENANCE GUARD (`​.agent_tmp/verify_round_provenance.py`) — user requirement, no earlier round
may leak in.** Runs THREE times: login node before submit, inside the build job, inside the
launcher before sbatch. Two independent checks: (1) the 12 collections, pinned by exact path
derived from each SLURM job's own log rather than a glob; (2) the MERGE INPUT LIST, which is where
an earlier round would physically enter.

**It caught two real things.** The merge list on disk was **STALE with only 9 entries** (generated
before the last 3 primitives were filtered). And the mushroom directory holds `26-08-25-clq-filt`
and `26-08-26-cze-filt` from EARLIER rounds beside this round's `26-08-30-urg-filt` — mushroom has
**46 run dirs**, tofu 14, raspberry 5, so a glob would have silently mixed rounds.

**Paired regulariser verified BY CONTENT, not filename:** `paired_cube3_clouds_shift9.npz` differs
from the uncorrected file by **+9.03 mm in x**. Note **both ddgrl AND alzey used the UNCORRECTED
`paired_cube3_clouds.npz`** — the two files sit side by side, so this was a live trap. Swapping in
the corrected file is the single intended change vs ddgrl; architecture, weight and budget match.

**THREE BUGS CAUGHT BEFORE THEY RAN — all silent-failure class:**
1. `build_generalist12.sh` passed `EXTRA_OVERRIDES`, but `dppo_pretrain.sbatch` reads
   **`GM_EXTRA_OVERRIDES`**; and it placed `--export` AFTER the script path, where sbatch treats it
   as a script ARGUMENT, not an option. Together these would have dropped EVERY override —
   architecture, paired-reg model, action_dim, seed — yielding a plain `[1024]x3` `DiffusionModel`
   that looked like a successful run and would have been compared against ddgrl as if matched.
2. My build sbatch called `envs/sim/.venv/bin/python` (the **x86 login-node venv**) on an
   **aarch64** GPU node -> `Exec format error`, dead in 25 s. Cluster is dual-arch; compute-node
   work must go through `uv run --project envs/*_arrhenius`.
3. The generalist hydra cfg is NOT in the main repo — it lives in the **`gm_generalist` worktree**
   (`.../cfg/single_lift_mushroom_simreal_realws_noos_cmd_v32`, `config_name pre_diffusion_pointnet`),
   discovered from ddgrl's `.hydra/hydra.yaml` `config_sources`. The base cfg there is
   `[1024,1024,1024]` + plain `DiffusionModel`; ddgrl's `[3072]x3` + paired reg came ENTIRELY from
   overrides. So a dropped override would not error — it would quietly train the wrong model.

**Walltime checked:** ddgrl did 300 epochs in 3h44m -> 695k steps ~= 4.4 h. Same gradient budget and
batch size here, so GPU work is identical; only dataloading grows (950k vs 254k transitions).
sbatch limit 22 h, ample. `--mem=0` retained (the earlier OOM was at checkpoint SAVE, not training);
~11 GB of point clouds resident at 950k x 1024 x 3 float32.

**Filtering completed the set:** prim_cuboid 500->489 (11 pinch), prim_lamp 500->**405** (95 pinch,
19% — the geometry-limited shape, consistent with its 57% demonstrator success at smoke).

**NEXT:** verify all 3 survive startup with the overrides ACTUALLY applied (model target, mlp_dims,
seed, normalization) — a dropped override is invisible in the result. Then per seed evaluate 2
checkpoints: closest to the val-loss minimum (later one on a near tie) and the last, all 6 on one
canonical protocol (`scene_group_size=1`, `record_batches=null`, dumps on), reported per §3.3
(success + damage rate with CI, max over objects).

**STARTUP VERIFIED from the RESOLVED config** (2026-09-01 00:15). Run IDs:

| run | seed | job | model | mlp_dims | paired npz | w | epochs | save/val | action_dim |
|---|---|---|---|---|---|---|---|---|---|
| `kklef` | 42 | 1858149 | PairedRegDiffusionModel | [3072]x3 | shift9 | 0.5 | 94 | 8 / 3 | 7 |
| `ctzhi` | 27 | 1858150 | PairedRegDiffusionModel | [3072]x3 | shift9 | 0.5 | 94 | 8 / 3 | 7 |
| `dgxtd` | 321 | 1858151 | PairedRegDiffusionModel | [3072]x3 | shift9 | 0.5 | 94 | 8 / 3 | 7 |

All three loaded `States (950741, 8)` (8-dim quat proprio) and `Actions (950741, 7)`.

⚠ **LESSON — a startup check that reads the sbatch's ECHO proves nothing.** My first monitor
reported "STARTED ... model=PairedRegDiffusionModel mlp_dims=[3072,3072,3072] seed=42" for all
three within a minute. That was a FALSE POSITIVE twice over: the `.out` file contains only the
sbatch's `[sbatch] pretrain: ... overrides=...` echo, so (a) the grep for `epoch|loss` matched
`train.n_epochs=94` inside that echo rather than any training progress, and (b) every "confirmed"
setting was read back out of the string that was PASSED, not the config that was BUILT. Since the
whole risk here is overrides being silently DROPPED (bug #1 above), reading them back from the
command line is circular. **Verify from `<run>/.hydra/config.yaml` — the resolved config Hydra
actually instantiated — and from `<run>/run.log`, never from the launcher's own echo.**

### 2026-09-01 — FIRST LAUNCH FAILED IN 2m05s: matching the GRADIENT BUDGET broke the LR SCHEDULE

All three seeds died at init, identically:

```
File "third_party/dppo/util/scheduler.py", line 55, in __init__
    assert warmup_steps < first_cycle_steps
AssertionError
```

**Root cause — a direct consequence of the epoch rescaling, and a trap for any future run that
holds the gradient budget instead of the epoch count.** The LR schedule is expressed in **EPOCHS**,
not gradient steps. `first_cycle_steps: ${train.n_epochs}` auto-tracks and so silently became 94,
but `warmup_steps: 100` is a LITERAL and did not. ddgrl ran 350 epochs, so 100 was a legal 28.6%
warmup; at 94 epochs the same literal exceeds the whole run.

**FIX — scale warmup by ddgrl's FRACTION, not to whatever is merely legal.** 100/350 = 28.6% ->
`round(94 * 100/350)` = **27** (28.7% of training), so the cosine shape matches ddgrl rather than
just clearing the assert. Derived in the launcher and asserted (`assert wu < ep`) so it cannot
silently regress if the dataset size changes again.

**GENERAL LESSON (record for the paper's method section too): when you hold the GRADIENT BUDGET
constant and let the epoch count fall, EVERY hyperparameter expressed in epochs silently changes
meaning.** Here that was warmup (crashed loudly, cheap) — but `save_model_freq` and `val_freq`
are the same class and fail SILENTLY, just producing a different checkpoint/validation density.
Both were already scaled deliberately; the audit rule is: **enumerate every epoch-denominated
knob and rescale or re-decide each one explicitly.** For this round that set is
{`n_epochs`, `warmup_steps`, `val_freq`, `save_model_freq`}, and `first_cycle_steps` which
auto-tracks.

Failed run dirs `kklef`/`ctzhi`/`dgxtd` deleted (0 checkpoints, died at init) and `experiments.csv`
reconciled, so the relaunch gets clean IDs.

**RELAUNCH SUCCEEDED (2026-09-01 00:27), warmup=27.** Verified by REAL training progress plus the
resolved `.hydra/config.yaml` (not the launcher echo — see the lesson above):

| run | seed | job | warmup | epochs | paired npz | epoch-1 train loss | s/epoch |
|---|---|---|---|---|---|---|---|
| `ydvlr` | 42 | 1859102 | 27 | 94 | shift9 | 0.1126 | 166 |
| `cvzth` | 27 | 1859103 | 27 | 94 | shift9 | 0.1129 | 175 |
| `fyetc` | 321 | 1859104 | 27 | 94 | shift9 | 0.1138 | 181 |

94 epochs x ~175 s = **~4.6 h**, against 4.4 h predicted from ddgrl's measured rate (300 epochs in
3h44m at the same batch size). The two agreeing is independent evidence the gradient budget really
is matched rather than merely arithmetically equal — different dataset size, same wall-clock per
gradient step.

Seed spread at epoch 1 is 0.1126-0.1138 (~1%), i.e. the seeds differ as expected and are not
accidentally identical.

### 2026-09-01 — Generalist-12 TRAINING DONE: NO OVERFITTING at matched gradient budget

All 3 seeds COMPLETED (4h15m / 4h40m / 4h45m), 94 epochs, 12 checkpoints each.

| run | seed | val min | @ep | final | vs min | train final | train/val gap |
|---|---|---|---|---|---|---|---|
| `ydvlr` | 42 | 0.00050 | 93 | 0.00050 | **+0.0%** | 0.00040 | **1.25x** |
| `cvzth` | 27 | 0.00050 | 93 | 0.00050 | **+0.0%** | — | — |
| `fyetc` | 321 | 0.00050 | 93 | 0.00050 | **+0.0%** | — | — |
| ddgrl (3obj) | — | 0.00090 | 280/350 | 0.00100 | +11.0% | 0.00050 | 2.00x |
| alzey | — | 0.00250 | 450 | 0.00290 | +16.0% | 0.00090 | 3.22x |

**HEADLINE: val loss fell MONOTONICALLY to the final epoch on all three seeds — no overfitting
point exists in this schedule.** ddgrl's val minimum sat at 80% of its run and it finished +11%
above it; alzey +16%. Ours finish AT their minimum with a train/val gap of 1.25x versus ddgrl's
2.00x and alzey's 3.22x. **At the SAME gradient budget, the 3.7x larger dataset (950,741 vs
254,340 transitions) does not reach its overfitting point.** That is a data-scale result, not a
schedule artifact — the optimisation is identical by construction.

**IMPLICATION FOR THE SCHEDULE:** 94 epochs may UNDER-train this dataset. ddgrl's budget was
chosen to match ddgrl; since val is still improving at the end, a longer run is a live option and
should be considered before treating these numbers as the ceiling for 12 objects.

**CONSEQUENCE FOR THE USER'S CHECKPOINT SPEC — degenerate, and handled explicitly.** "Closest to
val-min" and "last" are the SAME checkpoint (`state_94`) on every seed. Running one checkpoint
twice and reporting two results would be meaningless, so the second arm is the **EARLIEST
checkpoint statistically tied with the minimum** (`state_80`; the tie set runs ep78-93). That
answers the useful form of the question — did the final ~15% of training buy anything in success
or gentleness? **This is a DEVIATION from the requested selection, forced by the data; it must be
labelled as such wherever these results are reported.**

⚠ **Precision caveat:** losses are logged to 4 decimals, so the 0.00050 plateau from ep78 may be
partly display quantization. This STRENGTHENS treating ep78-93 as tied, but "val min at 93" is
precise only to the logged resolution.

**EVAL GUARD FIRED CORRECTLY (first 6 submissions, 1863689-94, all FAILED in 5 s):**
```
[sbatch] NORMALIZATION MISMATCH — refusing to run.
  checkpoint trained on env : single_lift_generalist_12obj
  NORM points at dataset    : single_lift_mushroom_soft_pcd
```
`dppo_eval.sbatch` defaults `NORM` to the mushroom-only dataset. Decoding 12-object-normalized
actions with mushroom min/max would have produced silently WRONG commands and a plausible-looking
success number. **The guard refused instead of skipping** — the correct design (contrast §5.2's
"a guard that SKIPS is worse than no guard"). Fixed by passing
`NORM=dataset/dppo/single_lift_generalist_12obj/normalization.npz`; resubmitted as 1863751-56.

**Eval protocol (all 6 identical):** `canon_mushroom_200geo40` — 200 episodes, 5 envs,
`scene_group_size=1` (40 distinct geometries), `record_batches=null`, dumps on. This is the exact
protocol the 3-object generalists were measured on, so the numbers are directly comparable to
ddgrl. **SCOPE LIMIT:** mushroom only. ddgrl was also evaluated on tofu and raspberry, and §3.3's
max-over-objects aggregation needs a per-object breakdown — that extension is NOT yet run.

**EVAL ATTEMPT 2 ALSO FAILED (1863751-56, 2m49s) — the eval cfg rebuilds the WRONG ARCHITECTURE.**

```
RuntimeError: Error(s) in loading state_dict for PointNetDiffusionMLP:
  size mismatch for mlp_mean.layers.0.weight: copying a param with shape [3072, 572]
  from checkpoint, the shape in current model is [1024, 572]
```
`Number of network parameters: 2890828` at eval vs **20902988** at training — it built `[1024]x3`.

**The eval config `eval_diffusion_pointnet.yaml` hardcodes `mlp_dims: [1024, 1024, 1024]` with the
comment "big net — must match the afucm-twin training cfg". That comment is FALSE for every
[3072]x3 run**: ddgrl's own eval `.hydra/config.yaml` shows `[3072,3072,3072]`, i.e. it passed the
override. Training-time architecture lives ONLY in the training overrides; the eval cfg does not
inherit it (CHECKLISTS §2.1 already warns about this for `proprio_encoder` — the same class).
Fixed with `GM_EXTRA_OVERRIDES="model.network.mlp_dims=[3072,3072,3072]"`; resubmitted 1863942-47.

**Why this one was lucky:** a WIDTH mismatch is unloadable, so it crashed. A architecture flag that
changes behaviour without changing tensor shapes (e.g. `proprio_encoder`) would build a different
network, load cleanly, and return a plausible success number. **Verify the eval's
`Number of network parameters` against the training log's before trusting any eval result.**

**RUNNING TALLY of silent-failure-class problems caught this round — all would have produced
plausible-looking results:** (1) `EXTRA_OVERRIDES` vs `GM_EXTRA_OVERRIDES` + `--export` after the
script path -> every override dropped; (2) epoch-denominated `warmup_steps` after rescaling epochs
to hold the gradient budget; (3) eval `NORM` defaulting to the mushroom-only dataset; (4) eval cfg
rebuilding `[1024]x3` instead of the trained `[3072]x3`. Only (2) and (4) crashed; (1) and (3)
were caught by a guard or by reading the resolved config.

### 2026-09-01 — GENERALIST-12 RESULTS (6 canonical mushroom evals, all COMPLETED)

Protocol identical for all six and for the 3-object references: `canon_mushroom_200geo40`
(200 eps, 5 envs, `scene_group_size=1` -> 40 geometries). Eval arch verified at 20,902,988 params
= the trained `[3072]x3`.

**PER ARM (n=200 each)**

| run | seed | ckpt | succ% | ever% | sust/Y | dmg% |
|---|---|---|---|---|---|---|
| cvzth | 27 | ep80 | 72.0 | 77.5 | 0.45 | 9.0 |
| cvzth | 27 | ep94 | 72.0 | 79.0 | 0.51 | 16.5 |
| fyetc | 321 | ep80 | 71.5 | 77.5 | 0.45 | 9.5 |
| fyetc | 321 | ep94 | 72.0 | 77.5 | 0.50 | 13.0 |
| ydvlr | 42 | ep80 | 73.5 | 76.5 | 0.52 | 15.0 |
| ydvlr | 42 | ep94 | 74.5 | 80.0 | 0.48 | 10.5 |

**POOLED (n=600) vs the 3-OBJECT REFERENCES**

| arm | n | succ% | sust/Y | dmg% |
|---|---|---|---|---|
| 12obj early (ep80) | 600 | 72.3 ± 3.6 | 0.47 | **11.2 ± 2.5** |
| 12obj last (ep94) | 600 | 72.8 ± 3.6 | 0.50 | **13.3 ± 2.7** |
| ddgrl 3obj | 200 | 70.5 ± 6.3 | 0.46 | 11.0 ± 4.3 |
| bwmcy 3obj | 200 | **84.0 ± 5.1** | 0.68 | **24.0 ± 5.9** |

**1. THE 12-OBJECT GENERALIST MATCHES ddgrl ON MUSHROOM — it does not beat it.** 72.3-72.8% vs
70.5%, difference ~2 pts against CIs of ±3.6/±6.3. Damage rate 11.2-13.3% vs 11.0%:
indistinguishable. **The defensible claim is generalisation WITHOUT DEGRADATION — 4x the object
categories at no measurable cost on the reference object — not an improvement.** Do not report
this as "the bigger dataset helped" on mushroom.

**2. THE LAST 15% OF TRAINING BOUGHT NOTHING MEASURABLE.** ep80 -> ep94: success +0.5 pts (noise),
damage +2.1 pts with overlapping CIs, and the per-seed DIRECTION is inconsistent (dmg 9.0->16.5,
9.5->13.0, but 15.0->10.5). **This matters because VAL LOSS WAS STILL IMPROVING over that span
(0.0006 -> 0.0005).** Val-loss gains at this magnitude do NOT translate into eval success or
gentleness — so "val is still falling, train longer" is not by itself an argument for a longer run.
That tempers the under-training implication recorded earlier: the schedule may be under-trained by
val, but the extra epochs are not visibly paying off in the metrics we care about.

**3. §3.3's DAMAGE-RATE CONSTRAINT DOES REAL WORK HERE — the winner depends entirely on eps:**
* eps=10%: **NOTHING admissible** (best is 12obj early at 11.2%)
* eps=15%: {12obj early, 12obj last, ddgrl} -> winner **12obj last** (72.8% succ, 13.3% dmg)
* eps=25%: bwmcy becomes admissible -> winner **bwmcy** (84.0% succ, 24.0% dmg)

**bwmcy has the HIGHEST success of anything measured (84.0%) and is excluded at eps<=15% because it
damages 24% of grasps.** That is exactly the trade a scalarised "score" would have hidden, and it
is the strongest concrete evidence so far that the constraint form is the right presentation.

⚠ **NOISE FLOOR — single-seed damage rate at n=200 is NOT reliable.** Across the six evals the
damage rate spans **9.0-16.5%**, a 7.5-point spread, while each individual CI is only ±4-5. That
spread EXCEEDS the ep80-vs-ep94 effect. Pooling 3 seeds (n=600) tightens to ±2.5. **Any future
damage-rate comparison must pool seeds or it will manufacture differences.** ddgrl and bwmcy are
single-seed n=200 here, so their ±4.3/±5.9 bars overlap much of the field — the bwmcy-vs-rest gap
survives that, the 12obj-vs-ddgrl difference does not.

**SCOPE LIMIT (unchanged):** mushroom only. §3.3's max-over-objects aggregation needs tofu and
raspberry (ddgrl has both) — NOT yet run. A 12-object policy judged on one object is a weak test,
and the per-object breakdown is the obvious next step.

### 2026-09-01 — First-frame OBJECT-CROP labelling (idea assessed, labelled, NOT trained)

**User idea:** condition the policy on an object embedding taken from the FIRST frame (crop the
cloud below a fixed height), because once the object is between the fingers its points merge with
the gripper's. Gated by the user: no training until the labelling is reviewed.

**PRIOR ART — the idea EXISTS and was tested NEGATIVE.** `first_frame_context`
(`dppo/pointcloud_dataset.py`) + `use_first_frame_context` (`dppo/pointnet_diffusion.py`) encode
the episode's first cloud with the SAME PointNet backbone and concatenate its 512-d feature.
DEVLOG item 12, run `ptpii`: **success 0.380 vs 0.685 baseline — HALVED**, "NOT taken to real",
filed under CONCLUDED NEGATIVE. **The user's variant differs in exactly one way: an OBJECT-ONLY
CROP rather than the whole frame** — which plausibly targets the failure mechanism (a 512-d
episode-constant vector over the whole scene is a high-capacity nuisance channel the denoiser can
use as an episode ID). The DEVLOG's own listed follow-ups were feature-side ("bottlenecked
context, FiLM, gating"); an input-side crop is not among them.

⚠ **RETRACTED from my own earlier analysis:** I argued an embedding might beat `ixjgp`'s 1-d
`feed_width_pred` because 1 dim in 572 is ignorable. Item 12 fed **512 dims** and made things
much WORSE. The bandwidth argument does not survive.

**t=0 GEOMETRY (measured, 4931 episodes).** `ee_z` at the first frame is BIMODAL:
79.2% at ~0.198 m (home) and **20.8% at 0.066-0.142 m** — matching `regrasp_prob 0.2`, i.e. the
retry-aware episodes start hovering ABOVE the object, exactly as the user warned. Minimum EE
origin **0.0661 m**, so a 10 cm crop catches the EE in ~11% of episodes, 7 cm in ~1%, 5 cm in none.
User set the crop at **6 cm** (TCP sits just below the finger ends, so fingers dip below the TCP
only under significant tilt).

**LABELLING RESULT** (`gentle_manip/scripts/label_first_frame_object.py`, writes
`first_frame_object.npz` per slice; figures via `plot_first_frame_object.py` in
`docs/figures/first_frame_object/`):

| finding | number |
|---|---|
| empty crops | **0 / 4931** — the crop always finds the object |
| episodes with "outliers" | 79 / 4931 (**1.6%**); 0 in cherry_tomato/prim_ellipsoid/prim_sphere/raspberry |
| **TRUNCATED (object cut by the 6 cm ceiling)** | **tomato 256/445 = 57.5%**, prim_cylinder 74/437 = 16.9%, prim_lamp 4 |
| point-count spread | raspberry mean 15.8 (**min 1**) vs tomato 163.7 — 10x |

**THE OUTLIER FILTER IS PROBABLY WRONG — it removes OBJECT points, not contamination.**
Largest-voxel-component ranking kept 19 points and rejected 25 in prim_lamp ep128 (voxel count,
not point count, decides). Validating against `aux_object_pos`: 44/79 kept the cluster FARTHER
from the true COM — **but that criterion is CONFOUNDED**: the cloud is a single-view SURFACE, so
its centroid is offset from the volumetric COM by ~the object radius (both clusters sit 20-50 mm
away), letting a 3-8 point cluster near the centre win by geometry. Better reading: both clusters
are the SAME object, severed by 1 cm voxel connectivity at a thin/occluded waist — supported by
mushroom 12/12 and prim_cuboid 3/3 having the SMALL cluster closer to the COM (a finger would be
farther), and by the prim_lamp clusters sitting 3.0-3.5 cm apart against a 5.2 cm object.
**Recommendation: drop the outlier step or coarsen the merge; it is deleting real points.**

⚠ **METHOD ERROR I made and corrected:** the first validation compared RAW cloud coordinates
against `aux_object_pos` without noticing it is normalized to [-1,1] like the states — giving
1.0-1.6 m "distances" and a meaningless 54/79 figure. De-normalize with `obs_min/obs_max[:3]`
first. **Any aux_* array in a converted dppo npz is normalized; never compare it to raw clouds.**

**OPEN FOR THE USER:** truncation is the blocking issue, not outliers. At 6 cm more than half of
tomato episodes lose the object's top, so its "object embedding" would systematically encode a
partial object; raspberry can be a ONE-POINT cloud. Both bound what this conditioning could do.
NOT TRAINED — awaiting review.

### 2026-09-01 — Per-object eval sweep: 12/36 died on MPM/rigid NaN; retried at 2x substeps

Sweep = 11 objects + pasta_bundle (OOD) x 3 seeds, `state_94`, protocol identical to
`canon_mushroom_200geo40`. Outcome: 17 completed, 6 running, **13 FAILED**.

**Client symptom was `ConnectionError: socket closed mid-message` — a RED HERRING.** As CHECKLISTS
§0 warns, that is what a dead sim SERVER looks like from the client. The real error is only in
`logs/slumr_logs/<jid>_<subdir>_simserver.log`:
```
GenesisWorker 'reset' failed in subprocess:
genesis.GenesisException: Invalid constraint forces causing 'nan'.
    Please decrease Rigid simulation timestep.
```
**Always read the SIM-SERVER log, never the client traceback, for an eval that dies mid-episode.**

**One "failure" was cosmetic:** `strawberry` seed 42 wrote **200 episodes + 200 clips**
(success 0.675, ever 0.730, sustained 14,390) and only failed at TEARDOWN. Its data is valid and
is being used. So 12 genuine failures, not 13 — **check for `summary.json` before discarding a
FAILED eval.**

**Failure pattern is per-OBJECT, not per-seed:** tomato / cherry_tomato / banana_chunk /
pasta_bundle failed 3/3. tomato died after ~12 resets (3 min); the others 28-34 min in. So it is
an early STOCHASTIC instability, not a config that never starts.

⚠ **NO simple substeps ordering explains it** — tomato has the LOWEST `sim_substeps` (175) and
cherry_tomato among the highest (330), both 3/3 failures, while raspberry (350) and the prims
(190) all pass. And the exception comes from the RIGID solver, not MPM. So "raise substeps" is a
mitigation whose mechanism is not established, not a diagnosis.

**RETRY: 12 evals relaunched at 2x substeps** via `GM_SIM_SUBSTEPS` (honoured by
`scene_builder`, so no config edit): tomato 175->350, cherry_tomato 330->660,
banana_chunk 290->580, pasta_bundle 220->440. Subdir suffixed `_ss<N>` so the fidelity is visible
in the path and can never be silently compared against the 1x runs. Walltime raised to 8 h (2x
substeps ~= 2x slower).

⚠ **REPORT THIS CAVEAT:** these 4 objects are evaluated at a DIFFERENT sim fidelity than their own
COLLECTION used, and than the other 8 objects. Per-object substeps already vary by design
(175-350) and each object is reported separately, so the cross-object table survives — but the
train/eval fidelity mismatch for these 4 must be stated wherever their numbers appear.

**ALTERNATIVE HYPOTHESIS NOT YET EXCLUDED:** collection ran 500 episodes/object at 1x substeps
with no such crash, using the CMA-ES scripted demonstrator. The eval differs by running a LEARNED
policy, which can drive the arm into configurations the demonstrator never visits. If the retries
still fail, the cause is more likely policy-induced arm/table interaction than integrator
fidelity, and the fix would be elsewhere.

### 2026-09-01 — Per-object eval RESULTS (7/13 objects) + which objects failed at what substeps

**12-object generalist `state_94`, 200 eps / 40 geometries per seed, mean ± SD over seeds
{42,27,321}. Eval seed 42 / num_envs 5 / n_episodes 200 on EVERY eval (verified identical,
including the 3-object references ddgrl/bwmcy/xaqnb) — so within-object cross-seed and
cross-policy comparisons are scenario-identical.**

| object | seeds | success % | sust/yield | damage % | yield kPa |
|---|---|---|---|---|---|
| prim_cylinder | 3 | 83.7 ± 1.8 | 0.59 | **19.2 ± 2.5** | 40 |
| prim_sphere | 3 | 77.8 ± 2.1 | 0.49 | 11.2 ± 3.8 | 40 |
| mushroom | 3 | 72.8 ± 1.4 | 0.50 | 13.3 ± 3.0 | 40 |
| strawberry | **2** | 69.0 ± 2.1 | 0.60 | **26.6 ± 2.9** | 18 |
| tofu | 3 | 59.7 ± 8.5 | 0.31 | **0.7 ± 1.2** | 20 |
| prim_lamp | 3 | 47.2 ± 1.5 | 0.35 | 4.2 ± 2.0 | 40 |
| raspberry | 3 | **24.7 ± 10.5** | 0.22 | 9.7 ± 4.6 | 15 |

Unweighted mean success **62.1%** (range 24.7-83.7). **§3.3 max-over-objects damage rate = 26.6%
(strawberry)** — the number the constraint binds on. Mushroom alone reads 13.3%, so
**mushroom-only reporting understates this policy's damage by half.** Strongest concrete
argument yet for max-over-objects rather than a single-object headline.

**Success and gentleness are NOT aligned across objects:** prim_cylinder has the BEST success
(83.7%) and near-worst damage (19.2%); tofu is mid-success (59.7%) and essentially never damages
(0.7%). Ranking objects by success inverts the gentleness ranking.

⚠ **strawberry is 2 seeds, and its numbers EXCLUDE one NaN-stress episode.** NaN >= 1 evaluates
False, so a NaN row silently counts as SAFE in a damage rate — always mask NaN explicitly.
(Only strawberry had any: 1 row in 1 seed.)

⚠ **MY ERROR, corrected:** I earlier reported "strawberry seed 42 completed with 200 episodes,
success 0.675". That was `ls ... | head -1` picking a DIFFERENT run's directory — 0.675 is
`cvzth` (seed 27). Strawberry seed 42 genuinely failed and has no data. **Never identify a run by
the first glob match.**

**EVALS THAT FAILED, with substeps (revisit later; user deferred):**

| object | task sim_substeps | 1x result | retry substeps | retry result |
|---|---|---|---|---|
| tomato | 175 | FAILED 3/3 (died 3 min) | **350** | **RUNNING 58+ min, 0 exceptions -> FIXED** |
| cherry_tomato | 330 | FAILED 3/3 (34 min) | 660 | **FAILED 3/3 again, same NaN** |
| banana_chunk | 290 | FAILED 3/3 (34 min) | 580 | FAILED (2/3 same NaN) |
| pasta_bundle (OOD) | 220 | FAILED 3/3 (28 min) | 440 | pending |
| strawberry | 260 | FAILED 1/3 | — | not retried (2 seeds suffice for now) |
| prim_cuboid, prim_ellipsoid | 235 | running at 1x | — | — |

**The 2x-substeps mitigation SPLIT the failures, which is the informative part:**
* **tomato (175 = the LOWEST substeps in the set) is FIXED by 350** — it was genuinely
  under-integrated.
* **cherry_tomato (330, already among the highest) still NaNs at 660** — doubling an
  already-high value changes nothing, so its mechanism is NOT integrator fidelity.
I was wrong to apply a blanket 2x to all four; tomato at 175 and cherry_tomato at 330 were never
the same problem. **Do not escalate substeps further for cherry_tomato/banana_chunk.**

**LEADING HYPOTHESIS for the remaining failures:** collection ran 500 eps/object at 1x substeps
with the CMA-ES scripted demonstrator and never crashed; the eval runs a LEARNED policy that
reaches configurations the demonstrator never visits. The exception comes from the **RIGID**
solver ("Invalid constraint forces causing 'nan'"), not MPM — consistent with arm/table
constraint violation rather than soft-body integration. Next step is the obs/action dumps in the
moments before the NaN, not more substeps.

⚠ **Retries run at DIFFERENT sim fidelity than their own collection and than the other objects.**
Subdirs carry `_ss<N>`; state this wherever those numbers appear.

**OUTLIER FILTER REMOVED (user, 2026-09-01): "what you rejected are part of the lamp. The
rejection filter is unnecessary if a height crop works, the rest are all object pnts."**

Confirmed from three independent directions and now the default:
1. **EE height** — across all 79 flagged episodes the EE at t=0 has MEDIAN **19.8 cm**, 14 cm
   above the 6 cm ceiling; only 2/79 sit below 8 cm. **No gripper point can be inside the crop**,
   so the rejected clusters were never finger contamination.
2. **Geometry** — the two components sat 3.0-3.5 cm apart on a 5.2 cm prim_lamp: inside one object.
3. **User inspection** of the rotating 3D renders.

The largest-connected-component step was severing ONE object at a thin/occluded waist and
deleting real points. `--outlier-filter` is retained to reproduce the old behaviour but is OFF.

⚠ **The reasoning that led me astray:** two well-separated clusters in a 2D projection LOOK like
object-plus-contamination. The 2D panels collapse exactly the axis that disambiguates them, and
my `aux_object_pos` check was confounded (single-view SURFACE centroid is offset from the
volumetric COM by ~the object radius, so a 3-point cluster near the centre "wins"). **The cheap
decisive check was the EE height — a number I already had — not a clustering metric.** Reach for
the measurement that rules out the hypothesis outright before building a discriminator.

**Rotating 3D renders** (`gentle_manip/scripts/video_first_frame_object.py`,
`docs/figures/first_frame_object/rotating/`) are what made this judgeable: 360 deg orbit, grey =
above crop, blue = kept, red = rejected, green star = EE at t=0, orange plane = the ceiling.
Keep this tool for any future point-cloud labelling question -- 2D projections were actively
misleading here.

### 2026-09-01 — TOMATO: the object the policy lifts BEST is the one it DESTROYS

tomato eval completed (3/3 seeds, retried at 350 substeps). It inverts the headline:

| object | n | success % | sust/Y | damage % | yield kPa |
|---|---|---|---|---|---|
| **tomato** | 3 | **91.8 ± 1.6** | **0.98** | **70.5 ± 2.0** | 25 |
| prim_cylinder | 3 | 83.7 ± 1.8 | 0.59 | 19.2 ± 2.5 | 40 |
| prim_sphere | 3 | 77.8 ± 2.1 | 0.49 | 11.2 ± 3.8 | 40 |
| mushroom | 3 | 72.8 ± 1.4 | 0.50 | 13.3 ± 3.0 | 40 |
| prim_cuboid | 3 | 71.0 ± 3.0 | 0.52 | 14.3 ± 4.2 | 40 |
| strawberry | 2 | 69.0 ± 2.1 | 0.60 | 26.6 ± 2.9 | 18 |
| prim_ellipsoid | 3 | 60.0 ± 5.6 | 0.38 | 8.8 ± 1.2 | 40 |
| tofu | 3 | 59.7 ± 8.5 | 0.31 | **0.7 ± 1.2** | 20 |
| prim_lamp | 3 | 47.2 ± 1.5 | 0.35 | 4.2 ± 2.0 | 40 |
| raspberry | 3 | 24.7 ±10.5 | 0.22 | 9.7 ± 4.6 | 15 |

**1. tomato: 91.8% success at a mean sustained stress of 0.98x YIELD.** The policy grips it at
essentially exactly the bruising threshold and exceeds it in 70.5% of episodes.

**2. §3.3 max-over-objects damage jumps 26.6% (strawberry) -> 70.5% (tomato).** Under the
constraint form this policy is inadmissible at any sensible eps. **Mushroom alone reads 13.3% —
the single-object headline understates the true damage by MORE THAN FIVEFOLD.** This is the
strongest evidence so far that the max-over-objects aggregation is the right presentation and that
mushroom-only reporting would have been actively misleading.

**3. success and damage CORRELATE +0.59 across the 10 objects.** On this policy, being better at
the task means gripping harder. That is the opposite of what a success-only evaluation implies and
belongs in the paper's problem statement.

⚠ **CONFOUND — NOT YET EXCLUDED, and it bears directly on the headline.** tomato is the object
re-run at **350 substeps, 2x its collection's 175** (its 1x eval crashed on the rigid-solver NaN).
Higher substeps integrate MPM stress more accurately, so tomato's stress is NOT measured on the
same footing as the other nine objects, nor as its own training data. Finer integration may resolve
stress peaks the 1x runs smooth over, which would inflate tomato's damage RELATIVE to everything
else. tomato is also the LOWEST-fidelity object by default (substeps 175, grid 180) — suspicious
in itself. **Distinguishing "tomato is genuinely crushed" from "tomato is measured differently"
needs a 1x tomato eval — exactly the run that crashes.** Top follow-up; do not treat 70.5% as
settled, and do not quote it without this caveat.

Still missing: cherry_tomato / banana_chunk / pasta_bundle (unresolved NaN), strawberry seed 42.

### 2026-09-02 — REAL DEPLOYMENTS ANALYZED; red-cube pairing lands at 8.9mm; VLA trained + teased

**1. zdwii91 (cotrained) real deploy: "hesitant closing" — the mode-switch diagnosis REFINED.**
Closure dynamics from the deploy recordings vs both training sources:

| source | 70->45mm closure | re-open events |
|---|---|---|
| sim-only deploy (cvzth80) | 9 steps, monotone | 0 |
| REAL teleop demos | 11 steps, monotone | 0 |
| cotrained deploy (zdwii91) | **126 steps (p90 347)** | **median 6, max 50** |

Both demo sources close FAST and MONOTONE — there are no two closing styles to switch between.
The hesitation is oscillation at the CLOSE/DON'T-CLOSE decision (re-opens at replan boundaries).
Proposed mechanism: real observations select the real-demo mode (domain match as mode selector),
but real is 2.5% of training -> weakly learned -> successive replans land on different sides of
the commit boundary. Consistent: approach fine, sim eval fine, hesitation only at closure on real.
**RECOLLECTING SIM DEMOS TO MATCH GRIPPER BEHAVIOUR WOULD NOT HELP — the closing behaviour already
matches (9 vs 11 steps; width level 39.6 vs 35.0mm).** Cost if wanted anyway: ~1.5-2 days.
Levers that target the mechanism: (a) OVERSAMPLE the real slice (retrain only, ~4.5h/seed);
(b) stronger paired encoder reg (see item 3); (c) deploy-side width latch to break oscillation.

**2. cvzth80 (sim-only) real deploy: oversqueeze IS sim2real, perception-side, with a stacked
physics term.** It COMMANDS 20-28mm min width on real vs ~40mm in its own sim evals — the policy
asks for ~12mm deeper closure under real observations (residual cloud bias + noisier mid-grasp
occlusion shifts the closed-loop feedback). On top: sim MPM resistance stops fingers near the
surface even when commands run past it, masking over-commanding a real position-controlled
gripper executes faithfully.

**3. RED-CUBE PAIRED REPLAY (3cm cube @ x=0.45,y=0):** `dataset/demos/single_lift_red_cube_simtwin/`
— 388 steps, step-paired. **ee_err 1.4mm mean / 5.7 max, quat 0.82 deg, gripper 0.2mm; cloud_nn
8.9mm mean / 12.2 p95.** Recorded WITH point_cloud_shift [0.009,0,0] active, so 8.9mm is the
RESIDUAL gap after the standing correction — independently reproducing cube3's ~8.4mm residual on
a fresh object/day. **The residual perception bias is SYSTEMATIC**, and is the right magnitude to
account for finding 2. Usable as paired validation data for encoder feature-discrepancy probing.
⚠ THREE bugs fixed in `replay_real_to_sim_paired.py` en route, worst first: it decoded actions
with `rec_cfg["action"]` (teleop DELTA) while newer recorders store `record_action` (7d ABSOLUTE)
— absolute targets integrated as deltas walked the arm 70mm/293mm off. **Decode with
`record_action` when present.** Also: paired-RGB recordings need images dropped (or --with-rgb)
for the sim twin, and the sim obs recorder must not index real-only obs keys.

**4. pi0.5 REAL VLA: TRAINED (30k steps, 18h — node slowed to ~4s/it) + sim teaser done.**
Checkpoints kept: 10000/20000/29999 (42 GB each; save_interval 2000 but keep_period 5000 means
only multiples of 5000 SURVIVE — and 5000/15000 are never written since saves land on 2000s).
**Teaser (15 eps, mushroom, generic prompt, mask active): success 0.000 — AS FRAMED, this is a
real->sim transfer number, NOT a performance measure.** Plumbing all passed: no NaNs, commands in
range, gripper actuates, and the CLIPS show a COHERENT approach — the arm descends and positions
over the object, it just never commits to grasp on sim renders. Pipeline validated end-to-end;
the real test is deployment (`deploy_real_pi05.py`).

**5. mlupe (base @1.5x budget, 8dp val): TIMEOUT at epoch 103/136** (node ran at ~420s/epoch vs
the seeds' 179 — walltime sized on the fast rate). The 8dp data still answers the question:
**val genuinely falls past ep91 (-13.4% over ep92-103; min 0.00059666 @ep99)** — the 91-epoch
budget IS undertrained by val — with a possible turn at ep102 (+3%). Whether that val gain buys
ANY success/gentleness is testable from the existing `state_96` (one eval) — the ep80-vs-94
evidence says no. PENDING user call: finish the run (~14h) vs evaluate state_96 (~2.5h).
