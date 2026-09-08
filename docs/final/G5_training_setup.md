# G5 — G3 + mean⊕max cloud pooling (cluster, 2026-09-08)

The cluster's run for this round. **G5 = G3 exactly, plus one encoder change: the point-cloud
feature is pooled as `concat(max, masked mean)` instead of `max` alone.**

Baseline chain: G2 `ttukt` (recipe v5, consistency ablated) → **G3** (horizon 4 → 16 executing 4,
hold tail 10 → 22) → **G5** (+ pooling). G3/G4 run locally and are specified in
`docs/final/G3_G4_training_setup.md`; the shared setup reference is
`docs/final/G1_G2_training_setup.md`.

| | G2 `ttukt` | G3 (local) | **G5 (this run)** |
|---|---|---|---|
| action horizon | 4 | **16** | **16** |
| executed steps | 4 | 4 | 4 |
| hold tail | 10 | **22** | **22** |
| cloud pooling | max | max | **concat(max, masked mean)** |
| paired real–sim | 0.5 | 0.5 | 0.5 |
| encoder consistency | 1e-8 (ablated, log-only) | 1e-8 | 1e-8 |

Everything else is untouched: lr 1e-4 cosine → 1e-5, batch 128, AdamW wd 1e-6, EMA 0.995 from
epoch 1, seed 42, `d435i_noise_train` BC augmentation with the ±[5, 3.5, 1.5] mm cloud offset,
1 M gradient-step target, warmup 5 %, checkpoints every 20 epochs, val every 5.

## 1. Why pooling, when the local analysis argues against it

**The counter-argument is real and is recorded here rather than glossed.**
`G3_G4_training_setup.md` §2 drops pooling on measured grounds: width is recoverable from the cloud
at **corr ≈ 0.8 against a data ceiling of 0.85**, so max-pooling is *not* destroying size
information, and a pooling change "targets a bottleneck shown not to exist". That measurement stands
and this run does not dispute it.

**So G5 is explicitly NOT a width fix.** The justification is different and weaker, and should be
read that way: pooling is being tested as *general representation quality for a small backbone*, not
as a cure for the cherry-tomato width bias.

- [Equivariant vs. Invariant Layers](https://arxiv.org/abs/2306.05553) (ICML 2024) sweeps backbones ×
  pooling and finds complex pooling most helps **simple** backbones — ours is a 3-layer MLP PointNet
  — that pooling choice can matter **more than backbone width and depth**, and that **pairwise
  pooling combinations** significantly improve a fixed backbone. mean⊕max is exactly such a pair.
- R3D keeps **structured N×C tokens** rather than collapsing to one global vector, and traced DP3's
  "lightweight PointNet is enough" result to a **BatchNorm artefact** — with LayerNorm (which we
  already use) stronger encoders beat PointNet. mean⊕max is the cheapest step in that direction that
  needs no extra points.
- R3D's pairing rule blocks the alternatives: at **1024 points** a wider encoder is predicted to
  *underperform*, and more points is a **re-collection decision** — our demos store clouds already
  FPS-sampled to 1024 with no depth retained. Pooling is the one encoder change available on the
  current dataset.

**User's framing, recorded:** run it to try, accepting the argument against it. A null result is a
useful outcome — it closes the last cheap encoder-side lever and moves the remaining explanation for
small-object failure onto approach precision (what G3 targets) and data balance.

## 2. What changed in the code

`gentle_manip/dppo/pointnet_diffusion.py::PointNetEncoderXYZ` gains `pooling: str = "max"`.
**The default is unchanged and bit-identical** — verified `torch.allclose(..., atol=0)` against the
previous max-only path — so every existing config, including G3/G4, is unaffected.

With `pooling="meanmax"`:

```
feat   = mlp(x)                                  # (B, 1024, 256)
pooled = max(feat, dim=1)                        # (B, 256)  ← byte-identical to the default path
valid  = (|x|.sum(-1) > 0)                       # (B, 1024, 1) mask out padded points
mean   = (feat * valid).sum(1) / valid.sum(1)    # (B, 256)
out    = final_projection(cat[pooled, mean])     # Linear(512 → 512)
```

Two deliberate choices:

- **The max branch is left exactly as-is**, so G5 differs from G3 only by the *added* channels. If it
  regresses, the added mean is attributable; nothing else moved.
- **Padded points are masked out of the mean.** The perception pipeline pads short clouds up to
  `max_points`, so some frames carry exact-`(0,0,0)` points (measured: 61 of 119,716 val frames,
  median 43 of 1024 where present, clustered at episode start/end — see
  `G1_G2_training_setup.md` §9). **Max is naturally immune** to those slots; a mean is not, and an
  unmasked mean would be silently diluted — a bug that would present as "mean pooling didn't help".
  Verified: on a frame with 300 padded points the masked and naive means differ by 0.29; on an
  unpadded frame they are identical.

Cost: encoder 175,104 → 306,176 parameters (+131,072, from `Linear(256→512)` becoming
`Linear(512→512)`). Total model ~2.89 M → ~3.02 M. Permutation invariance verified.

## 3. Dataset — the hold tail must move with the horizon

G5 uses **`dataset/dppo/single_lift_generalist_soft_v5_tail22`**, built on the login node with
`gentle_manip/dppo/augment_hold_tail.py <src> <dst> 12` (appends 12 replicated end-frames per
episode, ~8 % more data, normalization copied verbatim, `hold_tail_k` recorded for provenance).

This is not optional at horizon 16. Chunks never run off the end of an episode, so the number of
chunks lying wholly inside the hold tail is `K − horizon + 1`: at horizon 16 with the original tail
of 10 the policy would see **zero** all-hold chunks and never learn to commit to stillness. Tail 22
reproduces today's counts *exactly* (1,057,346 training chunks, 39,501 all-hold, 3.7 %), because both
depend only on `K − horizon`. **Rule from G3: keep `tail − horizon = 6`.** Full derivation in
`G3_G4_training_setup.md` §3.

## 4. Reading the result

**Compare against G3, not against G2** — G5 and G3 share the horizon and tail change, so their
difference isolates pooling. G5-vs-G2 would conflate three changes.

Teasers are **diagnosis, not ranking**: at n = 20 the spread (5) sits inside 1 σ (2.8), established by
the 2026-09-08 checkpoint sweep. What the per-episode videos and `signals/` plots show is *mechanism* —
approach placement, chosen width, retry behaviour, orientation wander. Canonical evaluation is
100 episodes (`EvalSpec`'s fixed value; the 200 in `eval_diffusion_pointnet.yaml` is the deviation).

Outcomes and what each would mean:

- **G5 ≈ G3** — the expected result given §1's counter-evidence. Closes the last cheap encoder-side
  lever; the remaining explanations for small-object failure are approach precision and data balance.
- **G5 > G3** — pooling carries general representation value beyond width, as the ICML sweep predicts
  for simple backbones. Would make structured tokens / attention pooling worth the implementation
  cost, and would raise the point-count question (a re-collection decision).
- **G5 < G3** — the added mean channels hurt. Most likely cause to check first is whether the mask is
  behaving on real data, not just the synthetic test.

⚠ A `pooling=meanmax` checkpoint carries a **different encoder shape**. Evaluating or deploying it
requires the same `+model.network.pointnet.pooling=meanmax` override; a mismatch is a state-dict load
error (loud, unlike the `predict_epsilon` trap in `G3_G4_training_setup.md` §4, which fails silently).

## 5. Run record

(filled at launch — job id, run id, wandb id, resolved EPOCHS, wall clock, results)
