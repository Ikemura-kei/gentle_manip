# CHECKLISTS — read this BEFORE launching, evaluating, or reporting

Companion to `docs/DEVLOG.md`. The DEVLOG records *what happened*; this file records *what to do
so it doesn't happen again*. It is written to be read at the START of a task, not after.

**How to use:** before a launch, an eval, or before quoting any number in a message, walk the
relevant section. The Common Mistakes section is the highest-value part — every entry cost real
GPU time or produced a wrong conclusion that had to be retracted.

---

## 1. Launching training
*(to be filled in)*

## 2. Evaluation

### 2.1 TEASER EVAL — screen a mid-training checkpoint for degeneracy (standing practice)

**Purpose (user, 2026-08-28):** catch a completely degenerate policy EARLY, from footage, so a dead
run can be killed instead of burning its full wall-clock. It is a SCREEN, not a measurement — never
quote a teaser number as a result.

**Protocol:**
| knob | value | why |
|---|---|---|
| checkpoint | roughly MID-training (e.g. `state_150` of 350) | early enough to save time, late enough to be fair |
| `N_EPISODES` | **15** | = 3 batches x 5 envs |
| `NUM_ENVS` | **5** | canonical sub-env count |
| `SCENE_GROUP_SIZE` | **1** | fresh geometry EVERY batch -> 3 distinct objects, so a pass is not one lucky scene |
| `record_batches` | `null` | **every episode rendered** — the footage IS the deliverable |
| dumps | on (default), `GM_OBS_DUMP_CLOUD=1` | actions + observations kept so any question can be answered without re-running |

**Use the SAME checkpoint epoch across arms** so the teasers are mutually comparable.

**Verify before trusting it:** the eval must rebuild the TRAINED architecture. Training-time
architecture flags are NOT in the eval config — C'/D'/E were trained with
`proprio_encoder: true`, so the eval needs `+model.network.proprio_encoder=true` or it silently
builds a different network. Wait for the `[PointNetDiffusionMLP] proprio_encoder=True` line before
reading any result. Same for `normalization OK` and `width DUMP active`.

**Do NOT teaser a run that is too early** (e.g. <15 % trained): an undertrained policy can look
degenerate and trigger a false kill. Wait until roughly mid-training.

**Reading it:** 0/15 with the arm never approaching, or never closing, or grasping nowhere near the
object = degenerate, worth killing. Non-zero success, or plausible approach+closure in the clips =
let it finish and judge on the canonical 200-episode eval.

**Teaser on MORE THAN ONE OBJECT (user, 2026-08-28).** For a multi-object policy, screen on a FEW
objects, not just mushroom — a policy can be fine on the object it sees most and degenerate on
another, and the teaser is cheap enough to catch that. Pass the object's own eval experiment to BOTH
`SIM_EXPERIMENT` and `EVAL_EXPERIMENT` (e.g. `single_lift_tofu_soft_abs_action_armfocus_7d_realws`)
and name the subdir per object (`teaser_e150_<object>`). Keep the SAME checkpoint epoch across
objects so they are comparable.
**When the generalist covers many objects, do NOT teaser every one — pick a few** (ideally spanning
the size range, e.g. raspberry ~15 mm / mushroom ~33 mm / tofu ~42 mm demonstrator width).
*Note: the per-object experiment configs live in the `gm_generalist` WORKTREE, so these evals need
`GM_REPO=<worktree>`; the main repo does not have the raspberry 7d variant.*

## 3. Analysis
*(to be filled in)*

## 4. Common practices — analysis methods, eval settings
*(to be filled in)*

---

## 5. COMMON MISTAKES

Each entry: what happened → how it was caught → the rule. Grouped by kind. Dates are when the
mistake was made, not when it was recorded.

### 5.1 Measurement and analysis

**Joining two data sources on an ASSUMED correspondence (2026-08-28).** I paired per-batch obs
dumps to `episodes.csv` by (file index, env index) and computed `corr(ee_x@grasp, obj_dx) = 0.087`,
then concluded arm F "grasps at a fixed x, ignoring the object". The user rejected it from a render
(`logs/gap_arch/armF_full/eval/armF_diag20/render/batch02_env1.mp4`) showing the arm plainly moving
to different positions. The join was a permutation error: the arm's initial pose must track
`home_dx` at r≈+1.0 and gave **−0.217**, with *identical* standard deviations (11.1 vs 11.1 mm) —
right values, wrong order. **Rule: never join on an assumed key. First validate the join against a
column whose answer is known** (`home_d{x,y,z}` vs the dumped `state_phys[0]` is the standard probe
here, and costs one command). The obs dump now writes an explicit `dump_batch` key.
*Evidence: `.agent_tmp/armFdiag_obs_b*.npz`, `…/armF_diag20/episodes.csv`.*

**Quoting a derived proxy before validating it (2026-08-28).** I estimated object x as the median x
of the lowest-20 % z cloud points and reported a "−28.8 mm lateral miss", then "−106 mm" from a t=0
variant. Both were wrong: `object_focus` keeps points near the end-effector, so those points were
**the arm**. Proof: `cloud_obj_x` had sd **10.7 mm** while the true `obj_dx` sd is **46.5 mm** — the
proxy barely varied while the object moved a lot. **Rule: a derived proxy must be validated against
ground truth before its numbers are quoted. Ground truth was in `episodes.csv` (`obj_dx`) the whole
time.** *Evidence: same dumps; `episodes.csv` DR columns.*

**A check with no discriminative power (2026-08-28).** My offline "does arm F reproduce the closure
ramp?" test returned `corr = 1.000` (p10 1.000) and I pre-registered it as the decisive first
result. A trivial copier — "command exactly the width you currently observe" — scores **0.998** on
the same episodes. F beat a no-op by 0.002. The policy *observes* the channel it predicts, actions
are absolute, and closure is a smooth ramp, so copying is near-optimal; the check was also
teacher-forced, which is exactly when a copier looks perfect. **Rule: for any predictive check on a
channel the policy also OBSERVES, report `corr_policy` beside `corr_trivial_baseline`. A high
correlation means nothing until the trivial predictor's score is known.**
*Evidence: job 1765211; `.agent_tmp/armF_copy_baseline.py`.*

**Wrong units in my own diagnostic (2026-08-28).** I reported arm F's per-axis bias as "−7.30 mm in
x". The conversion used `action_min/max` from `normalization.npz`, which maps into the
ActionPipeline's *normalized* space, not metres. The true value is **−1.20 mm** (2 % of the 62.4 mm
x spread) — an order of magnitude smaller, and the difference between "possible frame bug" and
"negligible". This is the repo's recurring reference-frame bug class (B1/B10/B17) committed inside
the tool meant to detect it. **Rule: an ABSOLUTE value converts with the full affine map, a DELTA
with the scale factor only; sanity-check every magnitude against a known physical quantity (here:
the 330 mm workspace span) before reporting.** *Evidence: job 1765626.*

**Normalizing each split with its OWN statistics (2026-08-28).** Relabelling the generalist
dataset to a DELTA gripper, my script computed the normalization range PER SPLIT, normalized each
split with its own values, then wrote only TRAIN's range into `normalization.npz`. Train's delta
range is `[-0.878, +0.643]`; val's is `[-0.843, 0.000]` (val contains no opening motions at all), so
val's actions were encoded in a scale the shipped file does not describe. **My round-trip check
missed it because it validated each split against ITS OWN constants** — internally consistent,
externally wrong. Training would have been fine, but the val loss would have been computed on
mis-scaled targets, i.e. a silently wrong monitoring signal. **Rule: normalization statistics come
from TRAIN and are applied to every split; validate every split by decoding it with the SHIPPED
normalization file, never with values recomputed in the same script.** Same shape as the
split-specific phase-file bug: consistency-with-itself is not consistency-with-what-ships.
*Evidence: `.agent_tmp/relabel_delta_width.py`, `.agent_tmp/fix_val_norm.py`.*

**Overstating a small-sample correlation (2026-08-28).** `r = 0.087` at n = 20 has a 95 % CI of
about **[−0.37, +0.51]** — consistent with moderate tracking. I stated it as "ignores the object"
with no interval. **Rule: report n and a CI with every correlation, and do not phrase a null result
as a positive claim about absence.**

### 5.2 Protocol

**Comparing across protocols / too few geometries (2026-08-28).** Arm F's eval launched with
`scene_group_size: 4`, which at 200 episodes / 5 envs yields only **10 distinct geometries**; the
width-slope rule in this project requires **≥ 40** (a 12-geometry sample previously reversed three
verdicts). Caught ~10 min in and relaunched. The requirement was verified by *counting* unique
`obj_scale` in the established `lulkx/eval/slope_*` arms (40 batches / 40 geometries) because
`summary.json` reports `scene_group_size: None` and is unusable for this. **Rule: fix the protocol
in the CONFIG with the rationale inline, and verify the delivered geometry count from the OUTPUT,
not the input.** *Evidence: job 1765317 (cancelled) → 1765398; `eval_gap_arch.yaml`.*

**A guard that SKIPS is worse than no guard (2026-08-28).** `dppo_eval.sbatch` refuses to run on a
normalization mismatch by reading the checkpoint's `.hydra/config.yaml` — but arm F has no such
file, so the `[ -f ]` test fails and the guard is **silently bypassed** while the venv still uses
`normalization_path`. The confirming log line simply never appears, and its absence is easy to
miss. **Rule: when adding a new policy/eval path, re-assert every guard that keys off a
conventional artifact the new path lacks, and make the guard PRINT on success.**

**Assuming a new adapter inherits harness features (2026-08-28).** The width dump lives in the
*Policy* class, not the harness, so `GapArchPolicy` started with none — arm F would have produced
success and stress but **no width slope**, the metric the whole phase is about, discovered only
after a 200-episode eval. **Rule: a new Policy adapter must implement the dumps; dumps are now
default-on in both copies of `dppo_eval.sbatch` (`GM_DUMP_OFF=1` disables).**

**A side-channel file is SPLIT-SPECIFIC (2026-08-28).** The GAP phase JSON, computed from
`train.npz`, was passed to both `train_dataset` and `val_dataset`. The splits are 90/10 (1248 trajs
/254 340 steps vs 138/28 042), so ρ would have indexed the wrong timesteps. **Caught loudly** by an
`assert len(side_channel) == sum(traj_lengths)` added at load. **Rule: any per-timestep side channel
(phase, weights, labels) belongs only to the dataset built from the same npz — and assert it.**
*Evidence: jobs 1762955, 1762967.*

**A change to shared `gentle_manip/` code must be applied to BOTH the main repo AND the
`gm_generalist` worktree (2026-08-28, 5th instance of this class).** I added the delta-gripper path
(`ActionConfig`, `ActionPipeline`, `PolicyEnv`, `SimBackend`, `RealBackend`) plus two config YAMLs to
the MAIN repo only. Training was unaffected — it is pure supervised on the npz and never touches the
ActionPipeline — so nothing surfaced until the EVAL, which runs with `GM_REPO=<worktree>` and died on
`FileNotFoundError: .../gm_generalist/.../single_lift_..._dgrip.yaml`. **The dangerous near-miss: had
only the CONFIG been missing-but-present and the CODE absent, the pipeline would have silently
decoded the gripper through the ABSOLUTE affine into [0, 88] mm instead of +-3.5 mm/step, and
produced a plausible wrong result instead of an error.**
**Rules:** (a) after editing anything under `gentle_manip/` (actions, envs, perception, configs),
port it to the worktree in the same session; (b) port SURGICALLY, not by copying files — the two
copies already diverge (e.g. `sim_backend.py` differs by 49 lines for the `GM_FIXED_SCALES` sweep
knobs that live only in main); (c) gate the run on a check that the OTHER repo can load and decode
the new thing, and do not launch if it fails.
*Evidence: jobs 1771158 (failed), 1771209/1771216 (gate), 1771217.*

**The dump<->episodes.csv join by (batch, env) is BROKEN, systematically (2026-08-28/29).** The
`home_d{x,y,z}` probe (the arm's initial pose MUST track its recorded home offset at r~+1.0) returns
**-0.22 on `armFdiag` and -0.19 on the scripted baseline's dump** — two different eval paths, same
failure, standard deviations matching so the values are right and the ORDER is wrong. Any analysis
joining a per-batch dump to `episodes.csv` on `(batch, env)` is therefore unsafe until the dump
carries an explicit per-episode key (scenario seed or episode index). **Rule: run the `home_dx`
probe before ANY such join, and prefer analyses that stay INSIDE one dump file** (e.g. validating an
object estimator against where the arm actually grasped, which needs no join at all and gave
r=+0.997..0.999 over 255 episodes).

### 5.3 Environment and tooling

**A dependency stub is not a local decision (2026-08-28).** I stubbed `torchvision` rather than
installing it. It broke `transformers` (installed and working), which `diffusers` needs, and
produced five failed jobs whose errors each looked like a new problem
(`DDPMScheduler stub called` → `torchvision.__spec__ is None` → `Could not import PreTrainedModel` →
`cannot import HfFolder`). Fixed by installing real `torchvision==0.21.0` with `--no-deps` and an
abort-if-torch-changes check. **Rule: prefer the real, version-matched package with `--no-deps`,
verifying the critical pin is byte-identical before/after; stub only a leaf nothing else imports.**
*Evidence: jobs 1765733, 1765758, 1765789, 1765830, 1765875.*

**Silent fallback that degrades capability (2026-08-28).** `try: import diffusers / except: install
stub` swallowed the real import error, so a run requesting the diffusion head proceeded on a stub
and raised mid-run. **Rule: never let an `except` install a degraded substitute silently — print the
exception, and REFUSE to run when the substitute cannot satisfy the requested mode.**

**Verifying in a clean environment instead of the deployment one (2026-08-28).** My diffusers check
passed in a bare interpreter; the runner installs a fake `torchvision` first, under which it fails.
The test validated a configuration we never run. **Rule: reproduce the runtime's stubs/env exactly
in any verification job.** Same shape as the teacher-forced closure check: verifying under
conditions that differ from the real ones.

**Editing a config by index splicing without re-parsing (2026-08-28).** I forked an eval YAML by
cutting between `model:` and `shape_meta:` — but `shape_meta` *precedes* `model`, so the result had
duplicated blocks. Caught only by parsing the YAML back and asserting on the loaded object. **Rule:
after generating or editing a config, LOAD it and assert on the parsed structure; a file that
parses without error can still be wrong.**

### 5.4 Reporting

**Claiming work was done before doing it (2026-08-28).** I told the user the DEVLOG launch recipe
"now requires `GM_WIDTH_DUMP`/`GM_WIDTH_NORM`" before adding them; a check showed it did not.
**Rule: verify the file actually contains the change before saying it does.**

**Generalising a finding beyond its evidence (2026-08-28).** "Mode averaging refuted — closure is a
smooth ramp" was established for the **width channel only**; I carried it into reasoning about the
**approach pose**, where multiple valid grasp azimuths genuinely exist. **Rule: record the channel /
regime a conclusion was established on, and re-open it when moving to another.**

**Pre-registering an outcome table, then having to note the prediction was wrong in detail
(2026-08-28).** Predicted outcome 2 for arm F as "never closes"; F *does* close and fails on
approach pose instead. The pre-registration was still worth it — it made the mismatch visible
rather than reinterpretable. **Rule: pre-register, and when reality differs, record the mismatch
instead of adjusting the prediction to fit.**
