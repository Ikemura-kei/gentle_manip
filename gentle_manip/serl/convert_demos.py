"""Convert recorded sim demos into HIL-SERL RLPD demo transitions.

SERL's learner loads demos via --demo_path: a pickle of a flat list of transition dicts
{observations, actions, next_observations, rewards, masks, dones} it inserts into the
demo replay buffer (RLPD samples 50/50 demo/online). This subsets a recorded SUPERSET
demo to a view (drop the modalities that view doesn't use) and converts.

    uv run --project envs/serl python -m gentle_manip.serl.convert_demos \
        --demo dataset/demos/mushroom/<run>/data.pkl \
        --experiment mushroom_lift --view teacher --out demos_serl/mushroom_teacher.pkl

The demo pickle is written in envs/sim (numpy 2.x); envs/serl is numpy 1.x, so it is read
with a compat unpickler (numpy._core -> numpy.core). obs stays a dict here — train_serl
flattens it to {"state": vec} at load, per the view's key order.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# gentle_manip (genesis-free: Experiment) via sys.path — not an envs/serl dependency.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read numpy-2.x pickles under numpy 1.x (numpy._core -> numpy.core)."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def _load(path) -> dict:
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


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
                observations=o, actions=act[t], next_observations=no,
                rewards=r, masks=1.0 - float(done), dones=done,
            ))
    if missing_reward:
        print(f"WARNING: {missing_reward} transitions had no reward -> filled 0.0.")
    return transitions


def main() -> None:
    ap = argparse.ArgumentParser(description="Sim demos -> HIL-SERL RLPD transitions")
    ap.add_argument("--demo", type=Path, required=True, help="recorded demo pickle (superset obs)")
    ap.add_argument("--out", type=Path, required=True, help="output SERL transitions pickle")
    ap.add_argument("--experiment", default=None, help="subset to this experiment's --view first")
    ap.add_argument("--view", default="teacher", help="obs view to keep (with --experiment)")
    args = ap.parse_args()

    data = _load(args.demo)
    if args.experiment:
        from gentle_manip.experiment import Experiment, subset_demo
        exp = Experiment.load(args.experiment)
        view = exp.view_obs(args.view)
        data = subset_demo(data, view)
        print(f"subset to view '{args.view}': obs keys {view.obs_keys()}")
    episodes = data["episodes"] if isinstance(data, dict) else data
    transitions = episodes_to_transitions(episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(transitions, f)
    print(f"wrote {args.out}: {len(transitions)} transitions from {len(episodes)} episodes")


if __name__ == "__main__":
    main()
