# pi0.5 VLA on REAL data — 7 objects, single external camera (2026-09-02)

Job 1917308. One command does convert -> norm stats -> full fine-tune:

    sbatch gentle_manip/scripts/arrhenius/pi05_real_vla.sbatch
    # knobs: REPO_ID (gm/real7_ext), EXP_NAME (pi05_real7_ext), STEPS (30000), BATCH (64), SKIP_CONVERT

## Data — PINNED

`dataset/transfer/real_paired_7obj_2026-09-01`, **141 episodes / 32,837 frames**, 7 objects,
ONE external camera at 480x640 (letterboxed to 224 by openpi's own `resize_with_pad`).

| object | eps | frames | | object | eps | frames |
|---|---|---|---|---|---|---|
| grape | 20 | 5,107 | | padron_pepper | 20 | 4,710 |
| mushroom | 20 | 5,069 | | strawberry | 20 | 4,346 |
| tofu | 21 | 4,955 | | cherry_tomato | 20 | 3,904 |
| tomato | 20 | 4,746 | | | | |

Scale reference: the sim low-data study was ~11k frames, the 250-demo run ~50k. This sits between.
Actions are ALREADY 7-dim euler-absolute (no derive step, unlike sim's 10-dim rot6d recording);
proprio is the same quaternion 8-dim as the generalist.

## Language labels (user, 2026-09-02) — `--phrasings real`

Round-robin per episode:

1. `lift the {obj} up gently`
2. `pick up the {obj} gently`
3. `pick the {obj} up from the table carefully`
4. `lift the {obj} up from the table carefully`
5. `lift the object up from table gently`      <- literal "object"
6. `pick up the object from table gently`      <- literal "object"

**5 and 6 are IN TRAINING on purpose** (~1/3 of episodes). They teach the policy to act on an
UNNAMED object, which is what makes a generic-prompt evaluation possible: prompt for "the object"
and see whether it still grasps, including one whose name it was never given.

## THE WRIST SLOT IS MASKED OFF — the key decision (user, 2026-09-02)

openpi's `LiberoInputs` treats the two spare camera slots DIFFERENTLY:

```
"right_wrist_0_rgb": zeros      image_mask: np.False_   <- masked OFF (pi05)
"left_wrist_0_rgb":  wrist      image_mask: np.True_    <- HARDCODED TRUE
```

So zero-filling the LEFT slot — **what our sim `--cameras ext` runs did** — feeds the model an
all-black frame on a channel it is told is VALID. Pretraining never paired a real base view with a
black wrist view.

⚠ **This RETRACTS the strength of an earlier claim.** I cited the sim ext-only result
(0.025 success vs ext+wrist's 0.225) as the prior against single-view. That gap partly measures
the black-frame artifact, not purely the absence of wrist information. It is weaker evidence than
presented, and this model does not share the defect.

`gentle_manip/pi05/masked_wrist.py` subclasses `LiberoInputs` to zero the left slot AND set
`image_mask=False` — exactly openpi's own treatment of the absent right wrist. Wired by swapping
the MODULE ATTRIBUTE (`LeRobotLiberoDataConfig.create()` resolves it at call time), so
**no openpi file is edited**. `--no-mask-wrist` reproduces the old behaviour for comparison.

⚠ **`masked_wrist.patch()` MUST also run at INFERENCE** (sim teaser + real deploy) or the model
sees a different mask than it trained with.

## Recipe

Stock `pi05_libero`, **FULL fine-tuning**, 30k steps, batch 64, save every 2k.
LoRA deliberately NOT used: in sim at 50 demos it gave **0.000** success vs full FT's 0.225, while
train loss converged to 0.0008 either way — a converged LoRA loss is not evidence it learned.
Config `name` is left UNCHANGED (assets_dirs = assets_base_dir / config.name; renaming breaks
norm-stats lookup). The run is identified by `exp_name`.

## Validation plan

1. **Sim teaser on mushroom — a PLUMBING CHECK, not a performance measure.** A real-trained VLA
   scored in sim measures real->sim transfer; a low number would NOT mean the policy is bad. What
   it does check: actions in range, no NaNs, gripper actuates, server round-trip clean.
2. **Real deployment — the actual test.** Needs a pi05 deploy script (`Policy.infer` against
   `RealBackend`); `deploy_real_dppo.py` is DPPO-specific and `eval_harness.py` is sim-only.
   NOT YET IMPLEMENTED.

## Known risk

141 episodes is small for a VLA, and single-view remains the harder configuration even with
correct masking. Treat the first model as pipeline validation rather than a deployable policy.
