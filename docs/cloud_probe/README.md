# Pinned-yaw cloud probe — G6 `vnhnr` state_20 (2026-09-11)

Sim point clouds from the diagnostic in which the object's yaw is pinned to **0 / 45 / 90 deg**, one
per sub-env, so the policy's wrist rotation can be compared against the object's actual orientation.
Recorded for comparison against a real deploy recording of the same objects.

    cloud_sim_banana_env{0,1,2}.mp4   banana_mush,     pinned yaw 0 / 45 / 90
    cloud_sim_can_env{0,1,2}.mp4      can_lying_mush,  pinned yaw 0 / 45 / 90
    sim_cloud_banana_topdown.png      all three banana envs, frame 2

Each video is top-down (x-y) beside a side view (x-z). Grey = the full 1024-point cloud, blue = the
"object band" (z 15-120 mm, x 0.30-0.55 m), orange = the band's principal axis, which is what the
grasp yaw has to close across.

## Reproducing, and doing the real side

`render_cloud.py` takes either source and draws them identically, so the two are comparable:

    python render_cloud.py sim  <eval obs dump>.npz          <out_prefix>
    python render_cloud.py real <real_deploy shard_0000>.pkl <out_prefix>

The sim dump comes from an eval run with `GM_OBS_DUMP=<tag> GM_OBS_DUMP_CLOUD=1
GM_OBS_DUMP_DIR=<dir>`; the pinning itself is `GM_FIXED_POSE=1 GM_FIXED_YAW_DEG="0,45,90"` on the
sim server.

## What state_20 showed

| object | env 0 (yaw 0) | env 1 (yaw 45) | env 2 (yaw 90) | success |
|---|---|---|---|---|
| can_lying_mush | EE yaw -47.7 deg | +53.1 deg | never closed cleanly | 2/3 |
| banana_mush    | no clean close   | +38.5 deg | +6.0 deg          | 0/3 |

Both objects fail at the 90 deg pose, and the banana barely rotates there at all. Measured object
axes in the cloud are 3.4-3.6x elongated with 250-390 of 1024 points on the object, so the object is
resolved clearly -- perception is not the limit in sim.

**This is epoch 20 of 100 and is not evidence about the finished policy.** The pinned poses also
carry per-env XY offsets (GM_FIXED_POSE does not zero them, despite its docstring), so the three
envs differ in position as well as yaw.
