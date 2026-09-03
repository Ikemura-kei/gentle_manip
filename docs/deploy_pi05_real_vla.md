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

## LOCAL setup on the robot box (added 2026-09-03, verified end-to-end)

`envs/dp3` is Python 3.8 and CANNOT host openpi (needs >=3.11). Run in openpi's own venv with the
repo on PYTHONPATH. `third_party/openpi` is gitignored, so it must be cloned locally first:

    git clone https://github.com/Physical-Intelligence/openpi.git third_party/openpi
    cd third_party/openpi && git checkout 215abfb
    uv sync --no-install-package evdev        # evdev (lerobot->pynput) fails to build against
                                              # this box's kernel headers; teleop-only, unused
    uv pip install --python .venv/bin/python \
        pyrealsense2==2.54.2.5684 xArm-Python-SDK "opencv-python>=4.8" scipy

Then: `bash gentle_manip/scripts/deploy_pi0.5.sh` (or the explicit command in it).

### Two gotchas that cost an evening

1. **`--repo-id gm/real7_ext` is REQUIRED.** Norm stats live at
   `<ckpt>/assets/gm/real7_ext/norm_stats.json` (the TRAINING repo id); without it openpi looks
   under the config default `physical-intelligence/libero` and raises FileNotFoundError. Also
   note the pinned 215abfb `create_trained_policy()` takes `norm_stats=`, NOT `repo_id=` — the
   deploy script now tries `repo_id=` and falls back to loading the stats itself, so it works
   against both openpi versions.
2. **Import order is load-bearing (SEGFAULT).** On this torch + jax-CUDA build,
   `import openpi.policies.policy` (torch) followed by `import openpi.training.checkpoints`
   (orbax) segfaults the process; the reverse order is fine. Minimal repro:
   `python -c "import openpi.policies.policy, openpi.training.checkpoints"` -> SIGSEGV.
   `Pi05RealPolicy.__init__` therefore imports `openpi.training.checkpoints` FIRST, before
   masked_wrist (which pulls libero_policy -> policy.py) and before policy_config.

3. **Policy interface mismatch.** `Pi05RealPolicy` exposed only `act(obs)`, but the shared
   `run_deploy_loop` drives policies as `reset(obs)` / `push(obs)` / `predict()` — it died with
   `TypeError: reset() takes 1 positional argument but 2 were given` the moment the robot was
   connected. Added those three methods as a thin wrapper over `act()` (pi0.5 is stateless per
   inference, so they just track the latest obs).

4. **Silent mid-episode re-homes (`max_episode_steps`).** `PolicyEnv` auto-resets at a fixed
   horizon (default **200 steps**), which re-homes the arm and re-opens the gripper. The deploy
   loop is not told, so the recording continues as ONE episode and it looks like a spontaneous
   reset. Seen in `pi05_real7_e29999` ep3: instantaneous snaps to home at steps ~195 and ~390
   while the policy was still commanding a pose ~200 mm away, gripper 30 -> 80 mm each time.
   Fixed by passing `max_episode_steps=10**9` (the DPPO deploy already did).

### Verified offline before touching the robot

Policy loads, and on a recorded mushroom frame it returns a (10, 7) chunk whose first action
tracks the demo (gripper 0.814 vs 0.814, z -0.982 vs -0.980).
