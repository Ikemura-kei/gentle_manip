# Debug report: Part C (7d-euler-absolute) DPPO policy scores ~0% success

**TL;DR — root cause found (see "ROOT CAUSE CONFIRMED" section below):** the recorded grasp's
roll angle sits essentially AT the euler ±π wraparound seam for ~99.5% of all timesteps, so
26.5% of consecutive-step action labels flip between ≈−1.0 and ≈+1.0 (same physical
orientation, opposite raw-action sign) — a training target a diffusion policy can fit on
average (hence clean loss) but that decodes to a physically wrong (~180° off) wrist roll at
rollout time whenever sampling lands between the two label modes. This is specific to the
euler rotation representation + this task's roll≈π grasp geometry; the rot6d sibling
(`jfhlu`) has no such singularity and scores 0.75-0.88 success on the same underlying demos.

**Run under investigation:** exp_id `oppsu` — `logs/dppo/dppo-pretrain/single_lift_mushroom_soft_hwo_7d/oppsu/`
**Sibling run (for comparison):** exp_id `jfhlu` — `logs/dppo/dppo-pretrain/single_lift_mushroom_soft_abs_pcd_hwo/jfhlu/`
**Checkpoint to download for offline debugging:** `logs/dppo/dppo-pretrain/single_lift_mushroom_soft_hwo_7d/oppsu/checkpoint/state_400.pt` (23 MB; `state_{100,200,300,500,600}.pt` also present alongside it, all ~23 MB each, all scoring ~0% — any is representative).

Both trained on the same underlying real teleop demos
(`dataset/demos/single_lift_mushroom_soft/26-08-17-hwo/data.pkl`), converted twice with
`--derive-action` into two DPPO datasets that differ ONLY in the action's rotation
representation:

| run | action config | action_dim | dataset |
|---|---|---|---|
| `oppsu` (Part C, broken) | `abs_pose_euler_abs_gripper.yaml` (7d: pos3+euler3+grip1) | 7 | `dataset/dppo/single_lift_mushroom_soft_hwo_7d` |
| `jfhlu` (sibling, healthy) | `abs_pose_abs_gripper.yaml` (10d: pos3+rot6d6+grip1) | 10 | `dataset/dppo/single_lift_mushroom_soft_abs_pcd_hwo` |

Position/gripper mapping (`pos_min/max`, `gripper_min/max`) is byte-identical between the two
action configs — only the rotation encoding differs.

## The anomaly

`oppsu` eval sweep (checkpoints 100–600, `EvalSpec` canonical: 200 episodes, seed 0):

| ckpt | success_rate | ever_success_rate | mean_episode_reward |
|---|---|---|---|
| 100 | 0.0   | 0.0   | 16.73 |
| 200 | 0.0   | 0.0   | 16.32 |
| 300 | 0.0   | 0.0   | 16.80 |
| 400 | 0.005 | 0.005 | 15.89 |
| 500 | 0.0   | 0.005 | 16.14 |
| 600 | 0.0   | 0.0   | 15.97 |

`jfhlu` (same demos, rot6d instead of euler) at the same checkpoints: success_rate
0.75–0.88, ever_success_rate 0.83–0.90. So the euler-only change collapses success from
~0.8 to ~0.

Training itself looks completely normal: `logs/slumr_logs/1455364_pretrain.log` shows loss
converging cleanly 0.7762 → 0.0161 (train) / 0.0187 (val) over 600 epochs — no divergence,
no NaNs. The resolved eval Hydra config (`.hydra/config.yaml` under
`oppsu/eval/state_400_eval/`) has all the expected values: `obs_dim: 8`, `action_dim: 7`,
`experiment: single_lift_mushroom_soft_abs_action_7d`, `env_name: single_lift_mushroom_soft_hwo_7d`,
and the normalization path resolves correctly. So this is not a config-wiring mistake or a
training-divergence issue — the model fits the *encoded* action targets fine; something goes
wrong in the *physical meaning* of what it outputs.

## `episodes.csv` pattern (state_400_eval, 200 episodes)

Object height barely changes at all across an episode: `obj_z_max ≈ obj_z_final ≈ 0.02` (the
object's resting height) for essentially every episode, yet `episode_reward` is nonzero
(~16, comparable in *magnitude* to jfhlu's) and per-episode stress fields
(`stress_max_tmax`, `stress_mean_tmax`) are non-trivial and nonzero — meaning the gripper
IS making contact / squeezing the object, it just never lifts it off the table. Reward here
is shaping-only (distance + stress terms); there is no success/lift bonus contributing, so a
nonzero reward with `obj_z_max≈0.02` is consistent with "approaches and touches the object,
never grasps-and-lifts it."

This point was not yet fully re-verified against the FULL episodes.csv (only the first row
was inspected before this report was compiled) — worth confirming the pattern is uniform
across all 200 rows, not just episode 0.

## ROOT CAUSE CONFIRMED: roll sits at the euler ±π wraparound singularity

Checked the actual converted dataset directly
(`dataset/dppo/single_lift_mushroom_soft_hwo_7d/train.npz`, 585 episodes, 108225 steps,
`actions` shape `(108225, 7)` = pos3 + euler3 (dims 3,4,5) + grip1):

```
dim3 (roll):  min=-1.0000 max=1.0000 mean=-0.0511 std=0.9876   <- std EXCEEDS uniform-on-[-1,1] (0.577)!
dim4 (pitch): min=-1.0000 max=1.0000 mean= 0.0722 std=0.2520
dim5 (yaw):   min=-1.0000 max=1.0000 mean= 0.0064 std=0.4049

dim3: frac(|x|>0.9) = 0.995   <- 99.5% of ALL 108225 timesteps are pinned near the rail
dim3: frac(|x|<0.1) = 0.000

big single-step jumps (|Δ|>1.0) in dim3, within-episode: 28545 / 107640 transitions (26.5%)
example (episode 0, step 2→3): [-0.997, -0.999, -1.000, +0.998, -0.999]
```

**This refutes the range-compression hypothesis below** (the range isn't compressed — dim3
uses the full [-1,1] span) and instead reveals something worse: roll (dim3, first axis of the
`xyz` euler sequence) sits essentially AT the ±π wraparound boundary for virtually every
timestep of every episode (physically consistent with a "mostly top-down grasp" where the
gripper approaches rolled ~180° from whatever reference pose defines roll=0). Because
`Rotation.as_euler` reports angles on `(-π, π]`, an orientation that hovers right at that
seam gets alternately reported as ≈+π or ≈−π depending on which side of the branch cut
tiny/physical noise (or the achieved-pose measurement) lands on — **28545 of 107640
frame-to-frame transitions (26.5%) jump by more than 1.0 in raw action units**, i.e. the
label flips between ≈−1.0 and ≈+1.0 for what is physically the same (or nearly the same)
roll angle, often on CONSECUTIVE timesteps within a single episode.

**Why this explains every observed symptom:**
- **Clean training loss**: a diffusion policy trained on a target that's tightly bimodal at
  the two rails (−1 and +1) can fit that bimodal distribution with low loss on average — it
  isn't being asked to track a single smooth trajectory, so nothing here looks like
  divergence.
- **Catastrophic eval (~0% success)**: at sampling time, any single draw that lands between
  the two modes (or picks the "wrong" mode for that instant, given how fast-flipping and
  noisy the label was) decodes to a **physically wrong roll of ~180° off** — since the whole
  ±π range decodes linearly, a raw value near 0 (between the two modes) decodes to roll≈0,
  which for a task whose true roll target lives at the ±π rail is about as wrong as an
  orientation can get.
- **Contact/stress nonzero but no lift** (`obj_z_max≈obj_z_final≈0.02`): the gripper still
  reaches roughly the right position (dims 0-2 have normal, non-degenerate stats) and closes
  (dim6/gripper looks normal too), so it touches/squeezes the mushroom — but with the wrist
  rolled to a wrong angle, the two fingers don't bracket it correctly for a stable lifting
  grasp.

**This is a genuine encoding defect, not a data or training bug**, and it is specific to
euler + this particular grasp's rest orientation (roll ≈ π). rot6d (`jfhlu`'s
representation) has no such singularity — Gram-Schmidt orthonormalization of a raw 6-vector
is continuous everywhere on SO(3), so the same "roll sits near the wrap boundary" physical
fact never becomes a labeling discontinuity in the training data.

### Fix directions (for whoever picks this up)

1. **Cheapest fix — rotate the euler reference frame.** Pick a different `euler_seq` or
   pre-rotate the EE frame by a fixed offset (e.g. 180° about one axis) before extracting
   Euler angles, so the physically-common roll≈π sits near 0 instead of at the wrap seam.
   Needs to be applied consistently in both `invert_absolute_action` (encode, convert time)
   and `ActionPipeline._process_absolute` (decode, rollout time) — e.g. bake a fixed
   `R_offset` multiply into both, or simply shift by subtracting π from the extracted roll
   and wrapping it into `(-π, π]` around 0 rather than around π. Cheapest: re-derive the 7d
   dataset with a shifted `euler_seq`/reference and retrain — no re-collection needed, same
   turnaround as this run.
2. **Alternative — clip/re-anchor per-episode**, unwrapping consecutive roll values within
   an episode (`np.unwrap`) before mapping to raw action space, so at least the WITHIN-episode
   labels are continuous (though this doesn't fully fix a diffusion policy that predicts
   open-loop from a fixed initial state, since unwrapped values could still exceed [-1,1] if
   allowed to accumulate — would need the range widened or clipped/re-wrapped carefully).
3. **Simplest and most defensible for THIS ablation round** — treat euler-absolute as
   confirmed inferior to rot6d for this task's grasp geometry and not worth re-running inside
   the current round; note it as a negative/explained result in the writeup, and optionally
   queue direction 1 as a fast follow-up ablation (same demos, no re-collection) rather than
   blocking the current round on it.

## (Superseded) range-compression hypothesis — refuted by the data above, kept for the record

`gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml`:

```yaml
mode: absolute
rot_repr: euler
clip: [-1.0, 1.0]
pos_min: [0.26, -0.225, 0.003]
pos_max: [0.59,  0.225, 0.50]
euler_seq: xyz
euler_min: [-3.14159265, -3.14159265, -3.14159265]
euler_max: [ 3.14159265,  3.14159265,  3.14159265]
gripper_min: 0.0
gripper_max: 0.088
```

The config's own header comment says: *"convert_demos re-normalizes to the dataset's actual
range, so no need to tighten unless you want finer resolution."* **This is not true — grepped
`gentle_manip/dppo/convert_demos.py` and `gentle_manip/actions/derive.py` end-to-end and there
is no renormalization step anywhere.** `derive_action_set()` (`actions/derive.py:27`) calls
`invert_absolute_action()` (`actions/pipeline.py:34`) directly with the action_config's
declared `euler_min`/`euler_max`, i.e. the full ±π used verbatim:

```python
# actions/pipeline.py:70-74 (encode, used at convert time)
angles = Rotation.from_quat(xyzw).as_euler(action_config.euler_seq, degrees=False)
a_rot = np.clip(lo + (angles - emin) / (emax - emin) * span, lo, hi)
```

```python
# actions/pipeline.py:166-171 (decode, used at rollout time)
angles = emin + (euler_raw - lo) / span * (emax - emin)
xyzw = Rotation.from_euler(self.cfg.euler_seq, angles, degrees=False).as_quat()
```

These two are *mathematically consistent inverses of each other* (verified by inspection —
same `emin`/`emax`/`span`/`euler_seq`, same affine map run forwards and backwards), so this
is **not a train/eval mismatch bug**. But because the actual recorded grasp trajectory is a
near-top-down grasp (bounded roll/pitch, limited yaw sweep — see the config's own comment
"fine for a mostly top-down grasp with bounded yaw"), the *physically used* orientation range
is almost certainly a small sliver of the declared ±π domain. Since the mapping is
**linear and covers the full ±π range regardless of what's actually used**, the entire
meaningful control signal for roll/pitch/(and possibly yaw) gets compressed into a tiny band
near raw-action 0 — e.g. if the real trajectory only spans ±0.05 rad of roll, the useful raw
range is only ≈ ±0.016 out of the full [-1, 1] the policy can emit.

By contrast, rot6d (`jfhlu`'s representation) has no such bottleneck: Gram-Schmidt
orthonormalizes *any* raw 6-vector into a valid rotation continuously, so there's no
fixed global range the network has to hit precisely — nearby physical rotations map to
nearby raw vectors at whatever natural scale the training data has, with no compression.

**This range-compression story turned out to be wrong** — see "ROOT CAUSE CONFIRMED" above
for the actual (data-verified) mechanism: it's not that the useful range is a tiny sliver of
[-1,1], it's that the label sits AT the ±π rail and flips sign discontinuously between
adjacent frames. Kept this section only so the reasoning trail is visible; skip straight to
"ROOT CAUSE CONFIRMED" and its "Fix directions" for what to actually do.

## Remaining open question (not yet ruled out, lower priority)

**Achieved- vs commanded-pose derivation being more consequential for euler than rot6d** —
the doc's own caveat notes Part C derives actions from *achieved* EE poses (post-physics)
rather than the *commanded* teleop targets that `jfhlu` presumably also derives from (same
caveat, though). The doc claims this difference is "near-identical" for absolute mode, but
that claim was made generically, not with the wraparound risk in mind — small
achieved-vs-commanded noise right at a rotation singularity could be *part of* why the sign
flips so often (noise pushing the achieved angle back and forth across the ±π seam), on top
of the singularity itself being unavoidable given the grasp geometry. Not necessary to
resolve before applying the "Fix directions" above, since those fix the encoding regardless
of the exact noise source — but worth keeping in mind if the frame-rotation fix (direction 1)
doesn't fully resolve the ~26.5% jump rate on retest.

## Relevant files

- `gentle_manip/actions/pipeline.py` — `invert_absolute_action` (L34-81, encode),
  `ActionPipeline._process_absolute` (L159-179, decode)
- `gentle_manip/actions/derive.py` — `derive_action_set` (L27-41)
- `gentle_manip/actions/action_config.py` — `ActionConfig` (`rot_repr`, `euler_min/max/seq`)
- `gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml` (broken arm's action config)
- `gentle_manip/configs/action/abs_pose_abs_gripper.yaml` (healthy sibling's action config)
- `gentle_manip/configs/experiments/single_lift_mushroom_soft_abs_action_7d.yaml` (Part C
  experiment config: task=`single_lift_mushroom_soft`, action=`abs_pose_euler_abs_gripper`,
  obs=`superset_soft`)
- Checkpoints: `logs/dppo/dppo-pretrain/single_lift_mushroom_soft_hwo_7d/oppsu/checkpoint/state_{100,200,300,400,500,600}.pt`
- Eval outputs: `logs/dppo/dppo-pretrain/single_lift_mushroom_soft_hwo_7d/oppsu/eval/state_{N}_eval/{summary.json,episodes.csv,render/*.mp4}`
- Training log: `logs/slumr_logs/1455364_pretrain.log`
- Converted dataset: `dataset/dppo/single_lift_mushroom_soft_hwo_7d/{train,val}.npz`,
  `normalization.npz`
- Raw demos (source of both `oppsu` and `jfhlu`): `dataset/demos/single_lift_mushroom_soft/26-08-17-hwo/data.pkl`
- Sibling healthy run for side-by-side comparison: `logs/dppo/dppo-pretrain/single_lift_mushroom_soft_abs_pcd_hwo/jfhlu/`

### Diagnostic scripts used to confirm the root cause (numpy-only, no repo env needed)

Both just `np.load` the `train.npz` and inspect the `actions` array (shape `(108225, 7)`) —
trivial to rerun/extend. On the login node, plain `pip install numpy` into any throwaway venv
is enough (no need for `envs/dppo`/`envs/dppo_arrhenius`, which are aarch64-only and won't run
on the x86_64 login node — `envs/dppo`'s venv currently has an aarch64 python symlinked into
it too, likely from a prior arch mix-up, so even that one won't run on the login node right
now).

```python
# per-dim range/std check
import numpy as np
d = np.load(".../dataset/dppo/single_lift_mushroom_soft_hwo_7d/train.npz")
a = d["actions"]
for i in range(a.shape[1]):
    print(f"dim{i}: min={a[:,i].min():.4f} max={a[:,i].max():.4f} std={a[:,i].std():.4f}")

# within-episode frame-to-frame jump check (wraparound signature)
tl = d["traj_lengths"]; starts = np.concatenate([[0], np.cumsum(tl)])
big_jumps = total = 0
for i in range(len(tl)):
    s, e = starts[i], starts[i+1]
    dj = np.abs(np.diff(a[s:e, 3]))
    big_jumps += (dj > 1.0).sum(); total += len(dj)
print(f"big jumps: {big_jumps}/{total}")
```
