"""Audit a recorded demo set: is EVERY per-step pose delta inside the action config's rate limit?

This is Part D's acceptance check for the v5 dataset. The bound is enforced twice upstream
(bound_scaled_schedule + the per-step clamp), but the dataset is what the policy trains on, so the
dataset is what gets audited — measuring the artifact, not trusting the mechanism.

Decodes each recorded ABSOLUTE action exactly as ActionPipeline would (the same math a deployed
policy's output goes through), then measures consecutive-step deltas in the delta-`scales`
convention (world-frame rotvec). Reports per-dimension max ratios and the count of violating steps.

    uv run --project envs/sim python -m gentle_manip.scripts.audit_demo_rate_bound \
        dataset/demos/single_lift_mushroom_soft/<run> \
        --experiment single_lift_mushroom_soft_abs_action_robust
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gentle_manip.actions.pipeline import ActionPipeline          # noqa: E402
from gentle_manip.experiment import Experiment                    # noqa: E402

DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "dgrip"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="demo run dir (reads shard_*.pkl / data.pkl)")
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args()

    exp = Experiment.load(args.experiment)
    ac = exp.action_config
    if ac.mode != "absolute" or ac.rate_limit is None:
        raise SystemExit(f"experiment's action config has mode={ac.mode!r}, "
                         f"rate_limit={ac.rate_limit} — nothing to audit")
    lim = np.asarray(ac.rate_limit, np.float64)
    pipe = ActionPipeline(ac)

    shards = sorted(args.run.glob("shard_*.pkl")) or [args.run / "data.pkl"]
    worst = np.zeros(7)
    viol = np.zeros(7, dtype=int)
    n_steps = n_eps = 0
    for sh in shards:
        with open(sh, "rb") as f:
            payload = pickle.load(f)
        for ep in payload["episodes"]:
            acts = np.asarray(ep["actions"], np.float32)
            cmd = pipe.process(acts)                       # (T, 8) pos+quat+grip — decoded targets
            n_eps += 1
            for t in range(1, len(cmd)):
                dp = np.abs(cmd[t, :3] - cmd[t - 1, :3])
                Rp = Rotation.from_quat(cmd[t - 1, [4, 5, 6, 3]])
                Rc = Rotation.from_quat(cmd[t, [4, 5, 6, 3]])
                dr = np.abs((Rc * Rp.inv()).as_rotvec())
                dg = abs(float(cmd[t, 7] - cmd[t - 1, 7]))
                d = np.concatenate([dp, dr, [dg]])
                ratio = d / lim
                worst = np.maximum(worst, ratio)
                viol += (ratio > 1.0 + 1e-4).astype(int)   # float32 decode noise allowance
                n_steps += 1

    print(f"audited {n_eps} episodes, {n_steps} steps against rate_limit={list(lim)}")
    print(f"{'dim':>7} {'max ratio':>10} {'violations':>11}")
    ok = True
    for i, name in enumerate(DIMS):
        flag = "" if viol[i] == 0 else "  ***"
        if viol[i]:
            ok = False
        print(f"{name:>7} {worst[i]:10.3f} {viol[i]:11d}{flag}")
    print("\nPASS — every step inside the bound" if ok else "\nFAIL — the dataset violates the bound")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
