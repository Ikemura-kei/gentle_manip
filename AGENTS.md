# AGENTS.md — Operating Principles for Autonomous Work in This Repo

High-level rules for how an agent should work in this repo, distilled from real
incidents this session. Complements `CLAUDE.md` (project architecture, experiment
discipline, conventions) — this file is about *how to operate*, not *what the
project is*. Read `CLAUDE.md` first for project specifics; read this for the
process discipline that applies across any experiment/session.

## 1. Always save video for anything that generates a trajectory

**Every process that runs a policy or a scripted demonstrator through the sim —
canonical eval, ad-hoc rollout checks, RLDG-style rollout collection, CMA-ES demo
collection — must save one video clip per episode/attempt, no exceptions.** This
was already a hard rule for canonical eval (`CLAUDE.md` Experiment Discipline #2)
but was missed twice in one session for adjacent tools: a new rollout-collection
script shipped with a `save_video` config field that was never actually wired to
anything (silent no-op), and demo-collection launches omitted `--record-video`.
Both had to be stopped mid-run and restarted once caught, wasting real compute.

**Before trusting ANY new data-generating script or config**, verify the video
actually lands on disk with a real smoke test — don't assume a config flag does
what its name says. A "record_video: true/false" field that's merely *read* but
never passed into the actual recorder call is worse than no field at all, because
it looks correct at a glance.

Failed/unsuccessful attempts are as worth recording as successes — they're often
the more informative videos when debugging why something isn't working.

## 2. Guard GPU/CPU resources actively, don't just hope

- **Check `nvidia-smi --query-compute-apps` headroom before stacking a new
  parallel job**, not after. This session ran up to 5-6 lightweight jobs
  concurrently (sim servers + trainers) on a single GPU without issue specifically
  because headroom was checked before each addition, not assumed.
- **Every backgrounded sim/training process must be launched process-group-safe**
  (`setsid ... & disown`) and **killed process-group-safe**
  (`kill -TERM -- -$PGID`, never a bare `kill <pid>`) — Genesis subprocesses and
  DPPO trainers spawn children that a plain `kill` leaves orphaned, silently
  holding GPU memory.
- **After stopping any job, verify cleanup**: `ps aux | grep <pattern>` shows
  nothing, `nvidia-smi --query-compute-apps` is empty or matches only what's
  still intentionally running. Don't move to the next step on faith.
- **Don't `uv sync` an environment that manages a dependency manually outside its
  `pyproject.toml`** (this repo: `envs/dppo`'s torch) while a job is actively
  running in it — it risks silently swapping a CUDA build for a CPU one mid-run.
  If a new dependency is needed only for a helper script, precompute into a
  cache from a *different*, safely-syncable environment instead.

## 3. Keep rolling — autonomous operation is the default, not the exception

When given a green light to work unattended (explicit "keep going while I'm
away," or standing project instructions to that effect):
- **Don't pause for confirmation on routine experiment progression** — plateau
  detected → stop cleanly → eval → record results → move to the next queued
  step, all without asking. Only stop for a genuine decision point: a result
  that contradicts the plan, an ambiguous next-direction choice with real
  tradeoffs, or something destructive/irreversible.
- **Batch independent work in parallel** when resources allow (see #2) rather
  than serializing everything — this session ran 3-5 sim/training jobs at once
  routinely, cutting wall-clock time substantially versus one-at-a-time.
- **Write findings down as they land, not in a final summary.** Update the
  project's working log and persistent memory incrementally, after each
  meaningful result — not just at the end. A session can be interrupted (host
  restarts happened this session) or run long enough that "summarize once at
  the end" loses everything since the last checkpoint.
- **Treat a stale/duplicate background-task notification as exactly that** —
  cross-check current process/file state before redoing work a notification
  describes; don't assume every notification reflects the current moment.

## 4. Verify infrastructure before trusting it, especially newly-built pieces

- **Smoke-test any new script/pipeline at small scale before scaling up.** Every
  piece of new infrastructure this session (VLM embedding cache, rollout
  collector, checkpoint-resume) was smoke-tested at a tiny scale first, which
  caught real bugs (a `transformers` API version mismatch, a missing
  `shape_meta` entry causing a live `KeyError`, an unwired video flag) before
  they could waste a multi-hour run.
- **A config flag that "looks like" it should do something needs to be traced to
  where it's actually consumed**, not assumed correct because the name is
  descriptive. Grep for where a flag is read, not just where it's declared.
- When copying a config file as a template for a new variant (new category, new
  conditioning mode, etc.), **check every block that references dimensions,
  keys, or paths specific to the old variant** — `shape_meta.obs` missing a new
  observation key is exactly the kind of bug a naive copy-paste leaves behind.

## 5. Resource cleanup and correctness beat speed, but don't gold-plate

Prefer fixing a real gap (checkpoint-resume support, a broken video flag)
properly over working around it repeatedly. But don't build more generality than
the current experiment needs — a fixed-seed, opt-in flag defaulting to the old
behavior (as done for `record_raw`, `category_embed_dim`, `--embed-source`) is
the right shape: zero risk to every existing caller, real capability for the new
one.

---

Project-specific conventions (task naming, canonical eval protocol, directory
structure, the `RawObs` sim/real boundary, etc.) live in `CLAUDE.md`. Current
experiment status and findings live in `docs/cross_category_specialist_log.md`.
This file is deliberately narrow — process discipline only.
