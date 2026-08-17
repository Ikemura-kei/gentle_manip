"""Full canonical 100-episode eval campaign (2026-08-17): compares the RLDG+VLM
combined generalist (kdcee/checkpoint/state_330.pt) against each category's own
solo specialist, across ALL 9 held-in + 4 zero-shot categories, using the
6-metric protocol (success_rate + 4 top-5%-vertex stress metrics + the combined
0.5*SR + 0.5*gentleness score -- see gentle_manip/evaluation/{harness,metrics}.py).

Held-in (9) = the categories actually merged into the current generalist
(banana, cherry, grape, kiwi, mushroom, pasta_bundle, raspberry, shrimp,
tomato -- egg_boiled/strawberry never made it into the merge, see
logs/fragile25_specialist/generalist_stdout.log). Zero-shot (4) = blackberry,
scallop, dumpling, gelatin. watermelon is EXCLUDED (reproducible MPM
divergence in its DR range, see project_generalist_12plus4_campaign.md memory
-- do not add it back without first fixing that).

Specialists only exist (and are only evaluated) for the 9 held-in categories --
zero-shot categories have no specialist by definition.

Idempotent: skips a (experiment, category) pair whose result json already
exists, so this is safe to re-launch after an interruption.

Usage:
    uv run --project envs/dppo python -m gentle_manip.scripts.run_full_eval_campaign
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gentle_manip.scripts.run_fragile25_specialist import RESULTS_DIR  # noqa: E402
from gentle_manip.scripts.run_fragile25_final_eval import eval_one  # noqa: E402
from gentle_manip.scripts import run_fragile25_specialist as specialist_mod  # noqa: E402

HELD_IN = ["banana", "cherry", "grape", "kiwi", "mushroom", "pasta_bundle",
          "raspberry", "shrimp", "tomato"]
ZERO_SHOT = ["blackberry", "scallop", "dumpling", "gelatin"]

GENERALIST_CHECKPOINT = ("/home/yif/Documents/KTH/git/robosuite_mog_private/dppo/log/"
                         "dppo-pretrain/single_lift_fragile25_generalist_pcd/kdcee/"
                         "checkpoint/state_330.pt")
N_EPISODES = 100
PORT = 5580

OUT_DIR = REPO / "logs" / "full_eval_campaign"
GEN_OUT = OUT_DIR / "generalist"
SPEC_OUT = OUT_DIR / "specialist"
GEN_OUT.mkdir(parents=True, exist_ok=True)
SPEC_OUT.mkdir(parents=True, exist_ok=True)


def run_generalist() -> None:
    for cat in HELD_IN + ZERO_SHOT:
        out = GEN_OUT / f"{cat}.json"
        if out.exists():
            r = json.loads(out.read_text())
            print(f"[campaign] generalist/{cat}: already done, "
                 f"success_rate={r.get('success_rate')}", flush=True)
            continue
        role = "held-in" if cat in HELD_IN else "zero-shot"
        t0 = time.time()
        r = eval_one(cat, role, GENERALIST_CHECKPOINT, port=PORT, n_episodes=N_EPISODES)
        r["elapsed_s"] = time.time() - t0
        out.write_text(json.dumps(r, indent=2))
        sm = r.get("summary") or {}
        print(f"[campaign] generalist/{cat} ({role}): success_rate={r['success_rate']} "
             f"combined={sm.get('combined_sr_gentleness')} ({r['elapsed_s']:.0f}s)", flush=True)


def run_specialist() -> None:
    for cat in HELD_IN:
        out = SPEC_OUT / f"{cat}.json"
        if out.exists():
            r = json.loads(out.read_text())
            print(f"[campaign] specialist/{cat}: already done, "
                 f"success_rate={r.get('success_rate')}", flush=True)
            continue
        spec_json = REPO / "logs" / "fragile25_specialist" / f"{cat}.json"
        checkpoint = json.loads(spec_json.read_text())["checkpoint"]
        cfg_dir = specialist_mod.write_configs(cat, port=PORT)
        t0 = time.time()
        r = specialist_mod.eval_specialist(cat, cfg_dir, checkpoint, port=PORT)
        r["category"] = cat
        r["checkpoint"] = checkpoint
        r["elapsed_s"] = time.time() - t0
        out.write_text(json.dumps(r, indent=2))
        sm = r.get("summary") or {}
        print(f"[campaign] specialist/{cat}: success_rate={r['success_rate']} "
             f"combined={sm.get('combined_sr_gentleness')} ({r['elapsed_s']:.0f}s)", flush=True)


def main() -> None:
    print(f"[campaign] START -- {len(HELD_IN)} held-in + {len(ZERO_SHOT)} zero-shot categories, "
         f"{N_EPISODES} episodes each, generalist ckpt={GENERALIST_CHECKPOINT}", flush=True)
    run_generalist()
    run_specialist()
    print("[campaign] DONE — all generalist + specialist evals complete", flush=True)


if __name__ == "__main__":
    main()
