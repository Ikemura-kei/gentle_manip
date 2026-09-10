# One policy, many fragile objects — where we are and what we are claiming

*Working summary for discussion, 2026-09-09. Numbers below are as measured; impressions are
labelled as impressions.*

## The goal

**A single policy that handles many different deformable and/or fragile objects safely** — one
network, many object classes, no per-object training, no tactile sensing, and no crushing the food.

This is a step up from our previous work ([arXiv 2510.25405](https://arxiv.org/abs/2510.25405),
Ikemura et al.), which trained a **single-object** policy per object: vision-only, stress-penalised
RL, zero-shot sim2real, 36.5 % stress reduction on tofu. That established that vision alone can be
gentle. It did not address one policy covering many objects, and did not claim generalisation to
unseen ones.

## The finding

A diffusion policy trained on **real teleop demonstrations of 7 objects only** — cherry tomato,
cube, grape, mushroom, padron pepper, tofu, tomato; 130 episodes — **attempts and often succeeds on
objects far outside that set**, including a pen: elongated and thin, a shape class with no
representative in training. Grasps are qualitatively gentle, and it appears to stop closing at the
right moment.

Configuration (`lyslr`): ImageNet-pretrained ResNet-18 with GroupNorm, 224 px, horizon 16 executing
4, random shift + photometric augmentation, temporal ensembling at inference.

**Observed limitations, from the same sessions:**
1. Minimal retry behaviour. Expected — the demonstrations contain no recovery at all, so retry is
   extrapolation.
2. Grasp pose is not always the best available one.
3. Some objects still get over-squeezed.

A plausible mechanism for the gentleness, from watching it: the policy stops closing when both
fingers *visually* contact the object. That cue is a projection from one external camera, so it is
sometimes false — which would predict failures clustering at particular object placements rather
than uniformly. Testable from the recorded runs.

## Why this might be new

Each neighbouring result exists; the intersection does not, as far as we have found.

| | multi-object | generalises to unseen | fragile / gentle | no tactile |
|---|---|---|---|---|
| BC-Z, RT-1 | yes | yes | no | yes |
| ChicGrasp (2025) | one class | no | yes | yes |
| our prior paper | no | no | yes | yes |
| **this** | **yes** | **yes** | **yes** | **yes** |

## The pipeline, and the framing

Human demonstrations are the bottleneck: they are expensive, and they cover neither enough objects
nor the *best* grasp for each object — a teleoperator picks a grasp that works, not the one that
minimises stress on that particular geometry.

So the pipeline is: **procedural grasp synthesis targeting deformable and fragile objects → sim
demonstrations at scale → a generalist policy**, with no human collection effort.

**The framing this gives the paper: the RGB real-demo policy is the bar, and it is surprisingly
high. Our goal is to beat it without collecting human demonstrations.** That is a sharper claim
than "our pipeline works", and it is falsifiable.

### The pipeline also fixes retry, and does it without demonstrating recovery

Limitation 1 above — the real-demo policy barely retries — is a coverage problem, not a capacity
one. Every human demonstration starts from the same home pose with the gripper wide open, so a
*failed* grasp puts the robot somewhere the policy has never been: mid-air, gripper half closed,
object not where it should be. Retry is extrapolation, and it shows.

Synthesised demonstrations do not have that shape. The collector starts episodes from
**randomised poses AND partially-closed gripper widths** — `START_W_MIN = 0.02`, so the gripper
begins anywhere from 20 mm to fully open, re-opening at `GRIP_SPEED = 0.0022` m/step, which is the
measured real teleop closing speed (2.20 mm/step over 141 episodes and 7 objects). Start pose and
width are sampled per environment from the DR config's start modes.

The consequence is that **the state a failed grasp leaves you in is already inside the training
distribution as a valid starting state.** The policy does not need recovery demonstrations to
retry; it needs only to recognise where it is, and from there it has seen thousands of trajectories
that end in a successful grasp.

This is worth stating as a design property rather than a side effect: it is a case where synthetic
demonstrations are not merely cheaper than human ones but **structurally better**, because a human
teleoperator cannot practically be asked to begin a thousand demonstrations from a thousand
awkward half-closed mid-air poses.

Current sim-side status: the best configuration (mean⊕max pooling + horizon 16 + temporal
ensembling) roughly triples the hardest object over the previous baseline — cherry tomato 10/20 and
9/20 replicated, against 3/20 — while matching the best baseline on mushroom (18/20) at lower
stress. Several other ideas were negatives and are recorded as such.

## What is not yet established

**Why it generalises.** We believe pretrained features. The literature disagrees with itself:
[R3M](https://arxiv.org/abs/2203.12601) and VC-1 report large gains from pretrained
representations, while [Hansen et al., ICML 2023](https://arxiv.org/abs/2212.05749) find a
from-scratch net with **random shift augmentation** competitive with R3M and MVP — and Diffusion
Policy's own real-robot config uses **random initialisation**. `lyslr` has pretrained weights *and*
fine-tuning *and* augmentation, so the credit is unassigned. **One 2 h run with random init settles
it**, and either outcome is worth reporting.

**Gentleness in the real world.** Every stress number we have is sim-measured. For a paper about
gentle manipulation, real evidence is needed — bruising, deformation, or force.

**Novel-object success rate.** Currently an impression. The paper needs N objects × M trials with
success and damage per object.

## Next experiments, in order

1. **Random-init arm** — decides the attribution question. 2 h 15 m, no new data.
2. **Systematic novel-object evaluation** — a table, not an impression. Robot time.
3. **A real gentleness measure** — closes the loop on the project's actual mission.
4. **Recovery demonstrations** — the known fix for limitation 1, and a collection change rather
   than a model change.
