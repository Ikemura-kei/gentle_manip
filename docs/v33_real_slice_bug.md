# The v33 real-slice bug: undrived delta actions, and how to catch this class of failure

**Status (2026-08-25):** root-caused, blocking. Every policy trained on
`single_lift_mushroom_simreal_realws_noos_cmd_v33` (`orkam`, `kjljs`, any sibling) is
non-functional on the real robot and must be retrained after re-converting the real slice.
The v3.3 *synthesis recipe* is not implicated — it has not yet had a fair trial.
Two reusable gates were added; both reproduce the diagnosis in seconds without hardware.

---

## 1. Symptom

Deploying `orkam/state_200` on the real arm: from home the end-effector climbed in +z and the
gripper closed part-way, instead of descending toward the object. Recording:
`dataset/real_deploy/orkam200/` (8 steps, aborted).

Decoding the recorded commands: x/y tracked the object correctly (0.45 → 0.41 toward an object
at x≈0.406), but z rose 0.200 → 0.229 m and the gripper was commanded to **44 mm** at step 4,
while every demonstration holds **80 mm** open through the whole approach. `state_400` behaved
identically, as did `kjljs/state_100` (a v33 sibling with an aux grasp-width head).

## 2. Diagnosis

**Step 1 — reproduce offline.** Feed the checkpoint the recorded deploy observation and decode
its predicted chunk. The failure reproduced exactly, so it is a property of the policy, not of
the robot, the controller, or the deploy wiring (which was independently verified: obs config,
action config, normalization dims, network shape, checkpoint keys all correct).

**Step 2 — control.** The same probe on `afucm/state_400` (the ~75 % real performer) on the
*same* observation: descends, gripper 80 mm. So it is specific to v33.

**Step 3 — isolate the modality.** Build hybrid observations that swap proprioception and
point cloud between a simulated and a real frame:

| observation | orkam/200 | afucm/400 |
|---|---|---|
| sim proprio + sim cloud | descends, 80 mm | ok |
| real proprio + sim cloud | descends, 80 mm | ok |
| sim proprio + **real cloud** | **68 mm** | ok |
| real proprio + **real cloud** | **z 0.225, 44 mm** | ok |

The real **cloud** triggers it with either proprioception. Point density was ruled out: a
*denser* simulated cloud (120 low points vs the real frame's 104) still behaves correctly.

**Step 4 — the tell.** The policy mispredicts on real frames **that are in its own training
set**: fed the first frame of the 55 real co-train demos it commands 46–61 mm, where the demo
holds 80 mm. A policy does not fail on its own training data unless the training *targets* are
wrong. That moved the search from the model to the dataset.

**Step 5 — the dataset.** In `dataset/dppo/single_lift_mushroom_real` (the real slice merged
into v33):

| channel | commanded (dataset) | achieved (demos) | midpoint of the absolute range |
|---|---|---|---|
| z | median **0.252 m** | median 0.096 m | **0.252 m** |
| gripper | median & max **44.0 mm** | median 79.8 mm | **44.0 mm** |

The commanded values sit exactly on the **midpoint of each absolute range**. That is the
signature of raw *delta* actions being written through as if they were *absolute*: a delta of
≈0 (the arm is not being asked to move on that axis this step) decodes to the middle of the
range. Median commanded z = 0.2515 m is `(0 + 1)/2 × 0.497 + 0.003`; median commanded gripper
= 44 mm is `(0 + 1)/2 × 0.088`.

**Conclusion.** The real demos record *delta* actions and were never run through the
delta→absolute derivation. The resulting slice teaches: *"on a real-looking point cloud,
command z = 0.25 m and gripper = 44 mm"* — which is precisely what the robot did.

## 3. Scope

- **Affected:** anything trained on `single_lift_mushroom_simreal_realws_noos_cmd_v33`
  (`orkam`, `kjljs`, siblings). Confirmed on the cluster side, not just locally: v33's merged
  normalization (shipped inside `downloaded_runs/orkam/normalization.npz`) carries the broken
  slice's ranges — action z max 0.75 → **0.438 m**, while no sim collection ever exceeds
  0.235 m — and matches the local broken slice on 6 of 7 dimensions.
- **Not affected:** `afucm` and the earlier shortlist. afucm's merged z max is 0.072 → 0.239 m,
  i.e. its real slice was derived properly. This is why afucm works and v33 does not.
- **Secondary effect worth noting:** because the broken slice's action ranges are far wider
  than the simulated slice's, joint normalization is dominated by it, compressing the useful
  range of every channel for the sim half as well.

## 4. Fix

Re-convert the real demos **with** derivation, then re-merge and retrain:

```bash
uv run --project envs/dppo python -m gentle_manip.dppo.convert_demos \
  dataset/demos/single_lift_mushroom_real_merged \
  --out dataset/dppo/single_lift_mushroom_real_7d \
  --obs-keys ee_pos ee_quat gripper_width --point-cloud \
  --derive-action        gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml \
  --derive-source-action gentle_manip/configs/action/delta_pose_delta_gripper_fast_rot.yaml \
  --derive-lookahead 4
```

`--derive-source-action` must name the config the demos were **recorded** with (delta fast_rot
for the real teleop sessions — check the recording's own `config.yaml`, which is authoritative);
`--derive-lookahead 4` supplies the commanded lead that slow teleop lacks. Then re-merge with
`merge_npz_datasets` and retrain. **Run the gate below on the output before training.**

### 4.1 Do the perception-bias correction in the same pass (recommended)

The real slice is being rebuilt anyway, which is the cheap moment to also remove the rig's
measured ~9 mm perception bias from the stored clouds — simulated clouds are unbiased by
construction, so correcting the real ones brings the two halves of a co-training set into
agreement instead of ~9 mm apart. Source the corrected demos from:

```bash
uv run --project envs/sim python -m gentle_manip.scripts.shift_demo_clouds \
  dataset/demos/single_lift_mushroom_real_merged --shift 0.009 0 0
# -> dataset/demos/single_lift_mushroom_real_merged_shift9mm  (clouds +9 mm x; proprio,
#    actions and zero-padding untouched; shift + source + commit recorded in its config.yaml)
```

then convert *that* run with the command above. Produce **both** variants and keep them as
separate datasets so the comparison is possible.

**⚠ Deployment pairing rule — this is the part that goes wrong silently.** A policy trained on
corrected clouds must be deployed with `point_cloud_shift: [0.009, 0, 0]` **active** in
`real_lab.yaml`; a policy trained on uncorrected clouds must be deployed with it at **zero**.
Either pairing is self-consistent; a mismatch reintroduces the full bias and there is no error
message. Note that afucm's ~75 % was measured with the shift OFF on uncorrected data — a
consistent pairing — so it remains a valid baseline either way.

**Bookkeeping requirement:** every run must state in its `EXPERIMENT.md` (and in the
`experiments.csv` description) which real variant it used — `real_merged` or
`real_merged_shift9mm` — otherwise the two families become indistinguishable after the fact and
the deploy pairing cannot be checked.

**Open question worth resolving while you are at it:** the +9 mm figure comes from a
nearest-neighbour displacement estimate, which attenuates under shape noise; the residual after
correction suggests the true bias may be ~12–13 mm (see
[item1_cube3_simreal_gap.md](item1_cube3_simreal_gap.md)). One more measure-shift iteration on
the paired cube3 data would pin it before a large collection is committed to a value.

## 5. The two gates (added to the repo)

### 5.1 Dataset gate — `gentle_manip/scripts/verify_derived_dataset.py`

Run on every `convert_demos` output before training:

```bash
uv run --project envs/dppo python -m gentle_manip.scripts.verify_derived_dataset \
  dataset/dppo/<name> --demos dataset/demos/<task>/<run>
```

Checks four failure modes, all silent in training loss *and* in simulated evaluation:
**derivation** (commanded values pinned at the range midpoint while the demos' achieved values
sit elsewhere), **lead** (commanded must lead achieved — but by millimetres, not centimetres),
**seam** (±π wraps on the euler dim, diffed *within* episodes only), **dwell** (near-identical
consecutive actions, excluding the intended trailing stop frames). Verified behaviour:

```
broken real slice : derivation[z] FAIL (0.252 = midpoint, achieved 0.096) ·
                    derivation[gripper] FAIL (44 mm vs 80 mm) · lead FAIL (249 mm) ·
                    dwell FAIL (0.51)                                    -> FAIL
good sim slice    : all ok                                               -> PASS
```

### 5.2 Policy gate — `examples/sim2real_diagnose/probe_policy_real_obs.py`

Run on every checkpoint before it touches the robot:

```bash
uv run --project envs/dppo python examples/sim2real_diagnose/probe_policy_real_obs.py \
  --ckpt downloaded_runs/<run>/checkpoint/state_N.pt \
  --normalization downloaded_runs/<run>/normalization.npz \
  --real dataset/demos/single_lift_mushroom_real_merged \
  --sim  dataset/demos/single_lift_mushroom_soft/<a v3.x collection>
```

Prints the four rows of the hybrid table above and exits non-zero if the policy climbs or
closes the gripper at episode start. Current verdicts: `afucm/state_400` **PASS**;
`orkam/state_200` **FAIL**; `kjljs/state_100` **FAIL**.

## 6. Lessons

1. **Simulated evaluation cannot validate a co-trained policy's real branch.** It never
   presents a real observation. orkam scored *better* than afucm in simulation (0.715 vs 0.685)
   while being non-functional on real input. This is a sharper form of the known
   "sim ranking does not transfer": here the sim-best checkpoint is not merely worse in
   reality, it is broken, and the sweep is structurally blind to it.
2. **A policy that mispredicts on its own training data is a data bug, not a model bug.**
   That check is one line and immediately redirected this investigation.
3. **Hybrid observations localize the modality** at zero cost — swapping proprioception and
   cloud between domains told us within one run that the visual branch was implicated.
4. **Midpoint-valued commands are the fingerprint of an unconverted delta dataset.** Worth
   checking whenever an absolute-action dataset is built from a delta-recorded source.
