# GAP — vendored reference copy

Source: https://github.com/GeWu-Lab/GAP  (project page https://gewu-lab.github.io/GAP/)
Paper: "When would Vision-Proprioception Policies Fail in Robotic Manipulation?"
       Lu, Xia, Wu, Lu, Hu — arXiv 2602.12032

Vendored 2026-08-28 as a REFERENCE for porting the Gradient Adjustment with Phase-guidance (GAP)
mechanism into our diffusion policy. `.git` removed so it is a plain directory, not a nested repo.

## The mechanism, as ACTUALLY IMPLEMENTED (gap/gap.py, after loss.backward())

    phase_p = torch.max(batch['phase']).item()        # scalar, MAX over the batch
    coeff_p = 1 - lambda * phase_p                    # NOT lambda*(1-rho) as printed in Eq 5
    if modulation_starts <= epoch <= modulation_ends:
        for name, parms in self.policy.encoder.named_parameters():
            if 'pro' in name:                         # proprio-encoder params ONLY
                parms.grad *= coeff_p
    optimizer.step()

`lambda: 0.3` (gap/cfgs/gap.yaml).

## Differences from the paper text that matter

1. Code uses `1 - lambda*rho`; Eq 5 prints `lambda*(1-rho)`. The code gives FULL learning outside
   transitions; the printed form would damp everywhere by lambda.
2. Damping is applied only inside an EPOCH WINDOW (modulation_starts/ends), not for the whole run.
3. rho is a per-BATCH SCALAR (max over the batch), not per-sample.
4. Selection is by PARAMETER NAME ('pro' in name) within one encoder — there is NO requirement for
   the two-branch summed-head structure of Eq 2. (An earlier note in this DEVLOG claimed our
   concatenated architecture could not support GAP; that claim is WITHDRAWN — see the entry.)
5. They ship a diffusion head (cfgs/policy/head/diffunet.yaml), so diffusion policies are in scope.

## Difference 6 (found 2026-08-28): `gap/gap.py` DOES NOT PARSE as published

Line 85 is `lambda = self.cfg.lambda`. `lambda` is a Python reserved word, so the file raises
SyntaxError on import. This is the ONLY edit made to their tree:

```python
lambda_ = self.cfg['lambda']   # VENDOR EDIT: `lambda` is a Python keyword,
coeff_p = 1 - lambda_ * phase_p    # so gap.py as published does not parse.
```

`gap/gap.py.orig` is the untouched original; `diff` confirms nothing else changed. It also means
the TRAINING half of their release cannot have been run as published, whereas `gap/dataset.py`
(CPD + LSTM) and `costdirection.py` both parse and we execute them verbatim.

## Difference 7: their `'pro' in name` filter also damps the VISUAL branch

`ImgEncoder.projection` / `DepthEncoder.projection` (`SpatialProjection`) match the substring
'pro'. Measured on our arm-F model: **52/112 encoder tensors, 4,504,960/8,821,376 params (51.1%)**,
split `{'imgencoder': 2, 'proencoder': 50}`. The paper describes modulating the proprioception
branch; the code additionally damps each visual branch's projection head. Arm F reproduces the
CODE's behaviour; our C'/D'/E arms match `proprio_encoder` only, i.e. the PAPER's description.

## Difference 8: `CosineAnnealingLR` is stepped per BATCH, with `T_max = cfg.epoch`

`WorkSpace.train()` calls `self.lr_scheduler.step()` inside the batch loop while the scheduler was
built with `T_max=cfg.epoch`. The result is not a decay over training but a CYCLIC LR of period
`2*cfg.epoch` BATCHES. Verified: with T_max=2 the trace is `[3e-4, 1.5e-4, 0, 1.5e-4, 3e-4, ...]`.

## Difference 9: the validation split is a SUBSET of the training split

`train demo_range=[0, demo_num]`, `valid demo_range=[0, int(0.1*demo_num)]` — same `data_path`.
`save_snapshot("valid")` therefore selects on partly-memorised data.

## Difference 10: `BCPolicy.get_action` assumes a single environment

`if img.shape[0] != 1: img = img.unsqueeze(0)` corrupts any batched (multi-env) rollout. Their
inference path is single-env; a batched evaluator must call
`head.get_action(encoder(img, proprio, ...))` directly.

## Difference 11: diffusers version (only relevant to the `diffunet` head)

Their `environment.yml` pins `diffusers==0.11.1` with `torch==2.1.0+cu121`; our aarch64 venv runs
`torch 2.6.0+cu126`, so we installed `diffusers 0.35.1` + real `torchvision 0.21.0` with `--no-deps` (verified: torch identical
before and after — three jobs were training in that venv).

**Behaviourally equivalent for our use.** They construct the scheduler with fully explicit arguments
(`head.py:252`: `num_train_timesteps=5, beta_schedule='squaredcos_cap_v2', clip_sample=True,
prediction_type='epsilon'`) and use only `.timesteps`, `.step()`, `.add_noise()`. The schedule is
determined by those arguments, not by the library version. Verified on our install:
`timesteps [4,3,2,1,0]`, `betas [0.101294, 0.279544, 0.473635, 0.724052, 0.999]`.
