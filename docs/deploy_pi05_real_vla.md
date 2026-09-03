# Deploy the pi0.5 REAL VLA on hardware (2026-09-02)

The VLA fine-tuned on all 141 real teleop demos (7 objects, single external camera, wrist masked
off). Run `pi05_real7_ext`, exp `pi05_real7_ext`. Kept checkpoints: 10000, 20000, **29999** (final).

## 1. Download the checkpoint (on your LOCAL machine)

openpi checkpoints are NOT handled by pull_run.sh (that is for DPPO run dirs) — use scp. The
checkpoint is SELF-CONTAINED: openpi loads norm stats from `<ckpt>/assets/`, so there is no
separate normalization file to fetch (unlike the DPPO deploy).

**Skip `train_state/` — it is the 30 GB optimizer state, needed only to RESUME training, never for
inference.** Deployment needs `params/` (12 GB) + `assets/` + `_CHECKPOINT_METADATA`:

    REMOTE=arrhenius1.hpc.arrhenius.naiss.se   # or your ssh alias
    CK=/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip/third_party/openpi/checkpoints/pi05_libero/pi05_real7_ext/29999
    mkdir -p pi05_real7_ext/29999
    rsync -avzP --exclude 'train_state' "$REMOTE:$CK/" pi05_real7_ext/29999/
    # ~12 GB. Full copy (with train_state) is ~42 GB if you ever need to resume.

## 2. Deploy

    uv run --project envs/dp3 python -m gentle_manip.scripts.deploy_real_pi05 \
        --checkpoint <local>/pi05_real7_ext/29999 \
        --prompt "pick up the mushroom gently" \
        --max-pos-step-m 0.01 \
        --record dataset/real_deploy/pi05_real7_e29999

## Which part of the deploy stack is pi0.5-specific

`scripts/deploy_real_pi05.py` is the ONLY pi0.5 piece. Everything below the policy is shared with
the DPPO deploy:

| layer | pi0.5-specific? | what |
|---|---|---|
| `Pi05RealPolicy` (in deploy_real_pi05.py) | **YES** | wraps openpi `create_trained_policy`; builds the `observation/{image,wrist_image,state}` + `prompt` dict from RealBackend obs; returns the (n_action_steps, 7) chunk |
| `masked_wrist.patch()` | **YES** | called in `Pi05RealPolicy.__init__` BEFORE the policy is built — zeros the left-wrist slot AND sets image_mask=False, matching training. Without it openpi feeds a black frame as a VALID view (silent train/serve mismatch). |
| obs plumbing: `image` = resize_with_pad(cam_ext, 224); `wrist_image` = zeros; `state` = [ee_pos, ee_quat, gripper_width] (8-dim) | **YES** | mirrors `pi05/eval_policy.py._one` exactly, so the real obs is built the way training saw it |
| `run_deploy_loop` (deploy_real.py) | shared | receding-horizon loop, keyboard controls (k restart / SPACE re-home / q quit), recording, `--max-pos-step-m` safety cap |
| `PolicyEnv` + `RealBackend` | shared | the XArm7/RealSense stack, identical to DPPO deploy |
| `ActionConfig` abs_pose_euler_abs_gripper | shared | 7-dim euler-absolute decode — SAME action space as the generalist |

Contrast with the DPPO deploy: NO point cloud (pi0.5 is RGB), NO `point_cloud_shift` (that
corrects a depth-cloud bias, irrelevant to RGB), NO separate `--normalization` (bundled in the
checkpoint's assets/).

## Prompt — this is the experiment knob

The model saw 6 phrasings, two of which say the literal word "object". Pass an OBJECT-NAMED one for
the standard case, or a GENERIC one to test acting without an object name:
  * named:   "pick up the mushroom gently" / "lift the tomato up from the table carefully" / ...
  * generic: "pick up the object from table gently" / "lift the object up from table gently"
There is NO default — state it explicitly. `--prompt` is required.

## Safety + expectations

* First run: keep `--max-pos-step-m 0.01` (per-step position cap) and a hand near the e-stop.
* This is 141 real demos, single view — small for a VLA. The sim teaser (real->sim transfer, not a
  performance measure) showed a COHERENT approach but no grasp commit on sim renders. Real behavior
  is the actual test; treat the first deployment as validation, not a finished policy.
* If it approaches but does not close (the DPPO generalist's hesitation), that is worth recording —
  the closing-commitment question is exactly what the horizon-16 generalist arms are probing.
