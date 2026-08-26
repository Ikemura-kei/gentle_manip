# Is grasp width predictable? — ceiling analysis for width adaptation

**Status:** 2026-08-27. Deep-dive subpage, linked from `docs/DEVLOG.md`. These numbers are
method-independent: they bound what ANY width-adaptation mechanism can achieve, so they stay
valid whichever mechanism we ship. Companion: `docs/size_adaptation_literature.md`.

## The question

The policy commands a near-constant width across a 12–46 mm object range. Before engineering
around that, we must know whether the information is there at all: *if the point cloud hides
object size, no algorithm can rescue it.*

Decomposed into two independently measurable links:

```
   point cloud  --(A)-->  object size  --(B)-->  demonstrator's grasp width
```

## (B) demonstrator side — pure CSV, no model

`dr_params.csv` records `scene_scale` and the chosen `width_mm` per episode.
Script: inline in the DEVLOG session; see `.agent_tmp/probe_corr.py` for the eval-side analogue.

| object | n | corr(width, scale) | Spearman | width sd | residual after scale |
|---|---|---|---|---|---|
| mushroom | 653 | **+0.841** ±0.077 | +0.872 | 6.3 mm | 3.2 mm |
| tofu | 614 | **+0.791** ±0.079 | +0.822 | 7.1 mm | 4.3 mm |

Orientation barely enters: corr(width, align) = +0.27 / −0.06, corr(width, folded yaw) = −0.03 /
−0.08. **Tofu is harder on both axes** — less predictable and larger residual — against a much
narrower tolerance band.

### Orientation hypothesis: TESTED, NOT CONFIRMED
For a cube the extent along the closing axis is `s·(|cosθ|+|sinθ|)`, a 41% swing over a 45°
wedge, so tofu width "should" depend strongly on yaw. Folding yaw into that wedge gives corr
≈ −0.08. Cause: CMA-ES's alignment term picks **flush face grasps** nearly every time, so the
demonstrator never enters the diagonal regime. The geometry is real; the demonstrator suppresses
it. *Consequence:* a "regress width + top-down grasp" reviewer baseline is NOT refuted by yaw
geometry on our data — argue differently, or just run the baseline.

## (A) perception side — Step 0

Frozen policy encoder + fresh head, t=0 cloud (object unoccluded), 85/15 episode split.
Script: `.agent_tmp/step0_size_from_cloud.py` (jobs 1728536 mushroom, 1728554 tofu).

| cloud @ t=0 → | mushroom | tofu |
|---|---|---|
| object **scale** | **0.739** | **0.842** |
| grasp **width** | 0.597 | 0.573 |

**Predicting SIZE beats predicting WIDTH** on both objects, and by more on tofu (+0.27) than
mushroom (+0.14): regressing width forces the head to absorb the demonstrator residual as noise.
Tofu's size is easier to see (a cube reads cleanly; mushroom carries bend/twist/taper DR).

### Join safety
`scene_scale` lives in the CSV, clouds in `data.pkl`, and only ordering links them — the silent
misalignment class that cost three results on 2026-08-26. The join is **proven, not assumed**:
`width_mm` vs each episode's achieved min gripper width must agree if the order is right.
Measured corr **0.9998** (mushroom) / **0.9992** (tofu); the script ABORTS below 0.9. The constant
~9.5 mm offset is the collector's extra squeeze (`grasp_extra_close` + firm phase), and its near-
perfect correlation also confirms that extra squeeze is effectively CONSTANT — so conclusions
drawn on `width_mm` transfer to the achieved width the policy actually imitates.

## The two ceilings — DO NOT CONFLATE (I did, initially)

Two different quantities, two different bounds:

1. **Ceiling on predicting the demonstrator's width** ≈ corr(cloud→size) × corr(size→width)
   = 0.739 × 0.841 ≈ 0.62 (mushroom). Measured direct cloud→width is 0.597 — consistent, which
   supports size mediating most of the relationship. Our heads reach 0.60–0.67: **at ceiling.**
2. **Ceiling on the DELIVERED metric** corr(commanded width at grasp, true scale). A policy can
   only know scale to corr(cloud→size), so this ceiling is **0.739** (mushroom) / **0.842**
   (tofu) — *not* 0.62.

I originally quoted 0.63 for both and said "we are at the perception ceiling". That was wrong for
the delivered metric. The honest gap is:

    delivered 0.474   vs   achievable 0.739     (mushroom, latched floor)

so the CONTROL gap is ~0.27, larger than the ~0.16 I first reported. The conclusion is unchanged
and strengthened: the loss is in control, not perception.

## Known flaws in this analysis (read before citing)

1. **Selection effect, mushroom.** Filtering to demonstrator-successful episodes raises
   corr(width, scale) from **0.708 → 0.841** (+0.133). Tofu is unaffected (+0.000; 614/616
   succeeded). Which is correct depends on the question: the policy trains ONLY on saved
   successful demos, so 0.841 is right for "what the policy is asked to imitate", while 0.708 is
   right for "how determined is width by size in general". **Report both; never the 0.85 alone.**
2. **Mild nonlinearity.** Spearman exceeds Pearson on both objects (0.872 vs 0.841; 0.822 vs
   0.791), so the linear residual (3.2 / 4.3 mm) slightly OVERSTATES irreducible error.
3. **Random split, not geometry holdout.** `size_adaptation_literature.md` §3 Step 0 specifies
   holding out geometries; we split episodes at random, so 0.739 / 0.842 measure interpolation
   within seen mesh variants, not generalisation to a new shape. Mushroom has 4 variants — a
   leave-one-variant-out rerun is the honest version and is cheap. **Not yet done.**
4. **`scene_scale` is a DR multiplier, not the physical extent along the closing axis**, and it
   does not exist at real deployment. Two consequences: (a) corr(width, scale) UNDERSTATES how
   determined width is by true geometry, so the ceilings above are conservative; (b) any
   scale-targeted head must be retargeted to **metric extent in mm** (= scale × nominal extent)
   to transfer to real objects.
5. **Collections are not independent replications.** Several mushroom rows are re-runs of the
   same recipe/seed (e.g. `26-08-16-btd` and `-nyl` are identical to 3 decimals). The 14-row
   table is ~4 distinct conditions, not 14 samples.
6. **Frozen encoder ⇒ lower bound.** (A) is measured through an encoder trained for policy
   objectives; a size-supervised encoder could exceed it (what run 1728356 tests).
7. **Single-frame conditioning.** Step 0 repeats the t=0 observation `cond_steps` times rather
   than using a real history, so it does not measure what a history-based (RMA-style) module
   could extract — which matters because vision and proprioception are complementary in phase
   (vision 0.667 at t=0 → 0.097 at contact; proprioception only informative AFTER contact).

## Tooling flaw found and fixed

`GM_WIDTH_DUMP` recorded only `traj[:, 0, -1]` — the first step of each action chunk — but
`act_steps=4`, so 3 of every 4 executed width commands went unrecorded, and both the at-grasp and
min statistics could miss the true extremum. Fixed to record all executed steps. Probes launched
before the fix share the flaw **uniformly across baseline and arms**, so their relative comparison
stands while absolute widths may be off by 1–2 mm.

## Target choice: METRIC EXTENT, not scale (decided 2026-08-27, user raised it)

Step 0 shows scale is more predictable than width (0.842 vs 0.573 on tofu), which tempted a
scale-targeted head. **Rejected.** `scene_scale` is CATEGORY-RELATIVE: 1.2 means a different
physical size for mushroom than for tofu, it is undefined for a real object, and the project goal
is a multi-category generalist. Its higher correlation is partly an artifact of predicting an
easier but less useful quantity.

**Use the object's extent along the CLOSING AXIS at the grasp pose, in millimetres.** For a
parallel-jaw grasp that scalar is the only geometry that matters (`size_adaptation_literature.md`
§2c), so the "hard to define 3D size" problem does not arise — we never need a full bbox.

Two facts make it practical:
1. **We already have it.** CMA-ES sets `width_mm` to just contact the object, so it IS that
   extent plus a near-constant offset — verified here: achieved width correlates with `width_mm`
   at 0.9992–0.9998 with a constant ~9.5 mm gap. "Predict width in mm" already IS "predict metric
   extent", and is already category-general.
1b. **The definition problem is SOLVED upstream (1fdb06f, local agent).**
   `smgrasp.finger_grasp.local_cross_section(obj)` returns the median cross-section perpendicular
   to the long axis in metres — "the width a proper across-the-body grasp actually has to close
   on". Their key point: a BBOX would be wrong in exactly the wrong direction (the banana's bbox
   makes it the LARGEST object at 95 mm while its graspable width is 17.9 mm). Measured nominals:
   mushroom 28.5, strawberry 30.5, raspberry 13.2, banana 17.9 mm. So the level-head target should
   be `scale x local_cross_section(nominal)`: metric, category-general, mesh-computable, and it is
   what the demonstrator is closing on. This is the retargeting flagged in flaw #4, now concrete.

2. **The nominal is a TRAINING-time constant only.** Target = `scale x nominal_extent_mm`, known
   per category in sim. Within a category this is a linear rescale of scale, so it inherits the
   same learnability (0.842), but the head OUTPUTS mm — so inference and real deployment need no
   nominal at all.

## Making the demos MORE PREDICTABLE from the student's view (user's proposal, 2026-08-27)

Rather than only learning a hard-to-predict demonstrator, make the demonstrator easier to predict
*without* losing success or gentleness. Two findings.

### 1. The residual is not noise — it is `align`

| model of width | mushroom R2 | tofu R2 |
|---|---|---|
| scale only | 0.708 | 0.626 |
| **+ align** | **0.818** | **0.818** |
| + align + pose | 0.822 | 0.819 |

Width is jointly determined by object size AND how flush the grasp is (a tilted grasp cuts a
different cross-section). **`align` is a property of the grasp pose, which the STUDENT chooses** —
so a width head predicting at t=0, before any pose is committed, is structurally blind to half of
what determines the answer. This may matter more than any loss function tried in item 18:
condition the width head on the policy's OWN sampled pose chunk (the diffusion head already
produces it). NOT YET TESTED — strongest remaining architectural idea.

### 2. Filtering by contact quality: a win-win on MUSHROOM (not tofu)

Filter on GRASP QUALITY, never on the residual itself (filtering on residual is circular and
would bias the very statistic being reported).

| mushroom filter | n | corr(w,scale) | mean stress |
|---|---|---|---|
| none | 653 | +0.841 | 12437 |
| drop top 20% pressure | 523 | +0.906 | 11889 |
| **drop bottom 20% align** | 523 | **+0.933** | **11191** |

| tofu filter | n | corr(w,scale) | mean stress |
|---|---|---|---|
| none | 614 | +0.791 | 8610 |
| drop bottom 20% align | 494 | +0.758 | 8169 |

Dropping the worst-aligned 20% of mushroom demos makes width MORE predictable (+0.09) AND the
demos GENTLER (-10% stress) on 80% of the data: low-align grasps are simultaneously the
unpredictable ones and the harsh ones. Tofu gains nothing in predictability (its align
relationship is different) though pressure-filtering still lowers stress.

**Coverage check (the failure mode that would invalidate it):** retention is uniform across scale
— 72/83/79/83/80% in bins [0.80,0.95)...[1.40,1.55) — and all 4 mesh variants survive. The filter
does NOT preferentially remove small objects.

**Why this attacks the disease, not the metric:** if width given an observation is near
deterministic in the data, the conditional the diffusion policy fits is sharp, so its mean-seeking
has far less to average over. That is a cleaner mechanism than any inference-time correction.

**PREDICTION RECORD (honesty):** I predicted filtering would help TOFU and cost mushroom its
gentleness. Both wrong — the CSV settled it in seconds. Same for the yaw-geometry hypothesis
earlier. Cheap CSV checks before expensive runs have paid for themselves repeatedly tonight.

**Recipe if adopted** (NOT yet run — a new dataset build + merge is exactly where the v33
poisoning incident happened, so it needs the pre-flight gates): filter the sim collection by
`align >= p20`, reconvert with `--derive-action`/`--derive-source-action`, re-merge with the
UN-poisoned real slice, run `verify_derived_dataset.py`, then retrain.

## CORRECTION 2026-08-27: the ceiling was under-estimated (under-trained head)

Job 1728668 fit a 128-unit MLP for 300 epochs with weight decay on the SAME frozen features, data
and split where Step 0's aux head (40 epochs, no decay) scored 0.597 for cloud->width:

    Step 0 aux head (40 ep, no WD)   corr 0.597
    MLP     (300 ep, WD 1e-4)        corr 0.771

Two consequences:
1. **0.771 EXCEEDS the "ceiling" of 0.739 x 0.841 = 0.62 quoted above**, confirming flaw #4: the
   cloud carries width information BEYOND uniform scale (true extent, shape), so a size-mediated
   product is NOT a valid upper bound. Treat those numbers as ESTIMATES UNDER A MEDIATION
   ASSUMPTION, never as ceilings.
2. **Head fits reported here are training-limited, not feature-limited.** Any conclusion of the
   form "we are at ceiling, stop improving the head" drawn from the 0.597/0.667 figures is void.

The `align` decomposition still stands (it is a property of the demonstrator's data, not of a
head fit), but the idea it suggested — conditioning on the student's own POSE — is refuted:
V 0.771 / P 0.574 / V+P 0.748. Pose is not align; align needs contact geometry.

## The demonstrator is NOT an existence proof for the student (user correction, 2026-08-27)

Tempting argument, and WRONG: "CMA-ES achieves 11.3 mm of aperture range at 94% success, so the
information must be there." **The demonstrator sees the FULL MESH** (complete geometry + an FEM
contact model); the student sees ONE partial view from a fixed L515, cropped, 1024 points, with
the arm in frame. They do not have the same information, so the demonstrator bounds nothing about
what the student can do.

The valid argument is the DIRECT measurement on the student's own observation:

    single-view cloud @ t=0 -> object size    0.739 (mushroom) / 0.842 (tofu)
    single-view cloud @ t=0 -> grasp width    0.771 (well-trained head)
    ...vs DELIVERED                           0.336 baseline / 0.511 floor

That headroom is measured on exactly the input the deployed policy has — no mesh, no completion.
Consistent with `size_adaptation_literature.md` §2c: single-view amodal completion is unreliable
and we must not build on it, but the scalar we need (extent along the CLOSING AXIS) is typically
perpendicular to the camera ray and therefore in the visible silhouette; the occluded dimension is
largely the one we do NOT need.

## Bottom line

The point cloud is NOT hiding the information: size is recoverable at 0.739 (mushroom) / 0.842
(tofu) even through an encoder never trained for it. Width prediction is at its ceiling. The
entire remaining loss is control — converting a good estimate into a commanded width without
destroying the grasp.
