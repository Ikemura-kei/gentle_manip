"""Convert recorded sim demos into HIL-SERL RLPD demo transitions.

SERL's learner loads demos via --demo_path: a pickle of a flat list of transition
dicts {observations, actions, next_observations, rewards, masks, dones} that it inserts
into the demo replay buffer (RLPD samples 50/50 demo/online). This converts our episode
format ({"episodes": [{"observations": {k: (T, ...)}, "actions": (T, 7),
"rewards"?: (T,)}]}) into that.

IMPORTANT: RLPD demos need REWARDS and must be in the TEACHER's obs space. So collect
them IN the mushroom teacher env (state_privileged obs) with per-step reward logged —
e.g. drive the scripted/teleop policy through gentle_manip.serl.gym_env.SimGymEnv (the
env's reward is available each step) and dump episodes with a "rewards" array. If an
episode has no "rewards", this fills 0.0 and warns (fine only for a format smoke-test,
NOT for real RLPD).

    uv run --project envs/serl python -m gentle_manip.serl.convert_demos \
        --demo dataset/demos/mushroom/<file>.pkl --out demos_serl/mushroom.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def episodes_to_transitions(episodes: list) -> list:
    transitions, missing_reward = [], 0
    for ep in episodes:
        obs, act = ep["observations"], np.asarray(ep["actions"], dtype=np.float32)
        rewards = ep.get("rewards")
        keys = list(obs.keys())
        T = len(act)
        for t in range(T - 1):
            o = {k: np.asarray(obs[k][t], dtype=np.float32) for k in keys}
            no = {k: np.asarray(obs[k][t + 1], dtype=np.float32) for k in keys}
            if rewards is not None:
                r = float(np.asarray(rewards)[t])
            else:
                r = 0.0
                missing_reward += 1
            done = t == T - 2                       # last transition of the episode
            transitions.append(dict(
                observations=o,
                actions=act[t],
                next_observations=no,
                rewards=r,
                masks=1.0 - float(done),
                dones=done,
            ))
    if missing_reward:
        print(f"WARNING: {missing_reward} transitions had no reward -> filled 0.0. "
              f"RLPD needs real rewards; collect demos through the teacher env.")
    return transitions


def main() -> None:
    ap = argparse.ArgumentParser(description="Sim demos -> HIL-SERL RLPD transitions")
    ap.add_argument("--demo", type=Path, required=True, help="our demo pickle (episodes)")
    ap.add_argument("--out", type=Path, required=True, help="output SERL transitions pickle")
    args = ap.parse_args()

    data = pickle.load(open(args.demo, "rb"))
    episodes = data["episodes"] if isinstance(data, dict) else data
    transitions = episodes_to_transitions(episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(transitions, f)
    print(f"wrote {args.out}: {len(transitions)} transitions from {len(episodes)} episodes")


if __name__ == "__main__":
    main()
