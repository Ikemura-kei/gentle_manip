"""Algorithmic retry-keypoint labeling for ReTVL (arXiv 2606.24633), substituting for the
paper's human annotators. We have privileged full-trajectory sim data unlike their
real-robot setup, so a robust heuristic on the recorded gripper_width channel is a
defensible substitute: our fast-reattempt collection produces a clean
open -> close(fail) -> reopen -> close(success) gripper signature for every
recovered_from_slip=True episode (verified by inspection: gripper_width drops from
~0.08 to a partial close, returns to ~0.08 on the failed-attempt release, then drops
again for the successful second attempt). The retry keypoint = the start of the FINAL
close event (where "corrective behavior begins", matching the paper's definition),
handling >1 retry generically in case an episode has more than one reopen/reclose cycle.
"""
from __future__ import annotations

import numpy as np

GRIPPER_OPEN_FRAC = 0.95     # fraction of episode-max gripper_width counted as "open"
CLOSE_DROP_FRAC = 0.90       # must drop below this fraction of max to count as a real close


def find_close_starts(gripper_width: np.ndarray) -> list[int]:
    """Indices where the gripper transitions from open -> closing (start of a grasp
    attempt). Robust to noise via a two-threshold (open/close) state machine, not a
    single crossing (avoids spurious re-triggers from small oscillation)."""
    gw = np.asarray(gripper_width).flatten()
    gmax = gw.max()
    open_thresh = GRIPPER_OPEN_FRAC * gmax
    close_thresh = CLOSE_DROP_FRAC * gmax

    starts = []
    state = "open" if gw[0] >= open_thresh else "unknown"
    pending_start = None
    for t in range(1, len(gw)):
        if state in ("open", "unknown") and gw[t] < open_thresh and pending_start is None:
            pending_start = t  # candidate: started dropping from open
        if pending_start is not None and gw[t] < close_thresh:
            starts.append(pending_start)
            state = "closed"
            pending_start = None
        elif pending_start is not None and gw[t] >= open_thresh:
            pending_start = None  # false alarm, bounced back to open without real close
        if state == "closed" and gw[t] >= open_thresh:
            state = "open"
    return starts


def retry_keypoints(gripper_width: np.ndarray) -> list[int]:
    """One keypoint per retry (close-attempt after the first). Empty list if the episode
    never re-closed after opening (i.e., a clean first-attempt success -- shouldn't happen
    for a recovered_from_slip=True episode, but handled gracefully)."""
    starts = find_close_starts(gripper_width)
    return starts[1:]  # first close = the (failed) first attempt, not a retry


def label_episode(obs: dict) -> list[int]:
    return retry_keypoints(obs["gripper_width"])


if __name__ == "__main__":
    import pickle
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "dataset/demos_retry_v2/single_lift_banana_soft/26-08-25-qkc/data.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    eps = d["episodes"]
    n_with_keypoint = 0
    n_multi = 0
    for i, e in enumerate(eps):
        kps = label_episode(e["observations"])
        tag = "recovered" if e.get("recovered_from_slip") else "?"
        print(f"ep {i:3d} T={len(e['actions']):4d} {tag:10s} retry_keypoints={kps}")
        if kps:
            n_with_keypoint += 1
        if len(kps) > 1:
            n_multi += 1
    print(f"\n{n_with_keypoint}/{len(eps)} episodes got >=1 retry keypoint "
         f"({n_multi} had >1 keypoint)")
