# External camera (D435i, `cam_ext`): drift check, extrinsic validity, recalibration

All tools live in `gentle_manip/diagnostics/`, run in the **deploy env**
(`uv run --project envs/deploy python -m gentle_manip.diagnostics.<tool>`), stream the same colour
640×480 @ 30 Hz as the deploy backend (**a centre crop of the 16:9 sensor — realsense-viewer shows more**),
and never write `xarm7_config.py` themselves.

**Fixed references** (`dataset/camera_calibration/reference/`):
- `aruco_ref.npz` / `aruco_ref.png` — the pinned reference frame: the table ArUco (5×5 dict, id 1, 80 mm, stuck
  on the wooden board near the arm base) as pixel corners + camera-frame corners, the TCP-measured corner
  `ref_tcp` (corner 1 = the marker's lower-left corner in the image = (0.1613, 0.0829, 0.0136) m), and the
  **park pose** `park_pos_m` = (0.3793, 0.0829, 0.350) m, top-down. Pinned 2026-09-05.
- The park pose is where the arm stands for every camera measurement: board out of view, ArUco and table visible.

## 1. Drift check (camera only, ~10 s, no robot motion)

Answers "did the camera move since the reference was pinned?" — independent of any extrinsic.

```bash
uv run --project envs/deploy python -m gentle_manip.diagnostics.drift_check
```
Reads: per-corner drift in **pixels, converted to mm lateral at the marker range** (the number to judge;
noise ≈ 0.5 px ≈ 0.6 mm), plus PnP camera-frame corners (depth along the ray is ±5 mm at 0.73 m — only
meaningful if ≫ 5 mm) and a rotation estimate. Saves `reference/drift_<stamp>.png` (blue = reference,
green = now). The arm need not be parked, but the marker must be unoccluded. **You decide the threshold**;
if it is exceeded → procedure 3, then `--pin`.

## 2. Extrinsic validity (robot to park pose + camera, ~1 min)

Answers "is the extrinsic currently in `xarm7_config.py` still right?" against two external truths.

```bash
uv run --project envs/deploy python -m gentle_manip.diagnostics.extrinsic_check --move     # drives to the park pose (30 mm/s, ESC stops)
uv run --project envs/deploy python -m gentle_manip.diagnostics.extrinsic_check            # arm already parked
```
Reads: the TCP-measured ArUco corner through the extrinsic (camera − robot, mm) and the table plane (tilt,
expect 0°; height, expect 13.8 mm = board top; 10-frame median cloud beyond 0.33 m, RANSAC 4 mm — 8 mm merged
the bare table into the board plane and faked ~1° of tilt). **Noise floor, measured 2026-09-05 right after
adoption: corner ±4 mm, tilt ±0.2°, height ±0.2 mm.** Tens of mm / degrees means the extrinsic is stale →
procedure 3. Note the bare table beyond the board reads ~5 mm low at 1–1.4 m (depth scale error), so only
the board plane is a valid height reference.

## 3. Recalibration (what was done on 2026-09-05)

**Operator prerequisites**
1. Clamp the ChAruco board (6×6, 20 mm squares, 15 mm markers, DICT_4X4) in the gripper and **close the
   gripper by hand** (the tools never command the gripper). Rigid is all that matters; the exact pose is not.
2. Clear the table except the ArUco. Nothing above the gripper (the first move is straight up).
3. Move the arm (teach mode / web UI) to roughly the park x/y at ~0.30 m height. Keep the web UI or e-stop
   within reach; the live window's **ESC** stops the arm at any time (`set_state(4)`).
4. Dry run first (default): `... calib_replay` — lists the 13 poses (outlier #7 of the 09-03 round is dropped
   automatically), skips any outside `EE_BOUNDS`.

**Run**
```bash
uv run --project envs/deploy python -m gentle_manip.diagnostics.calib_replay --live --table-check --table-z 0.0138
```
Sequence: lift to `--start-z` 0.35 → reference frame + ArUco (saved as `reference_<stamp>_aruco_cam.npy`) →
13 poses (30 mm/s, 5 s dwell, board captured at each) → back to the start pose → `calib_select` (robust subset
solve) → table check. Undetected poses are skipped. Output: `charuco_hand_eye_<stamp>.npz` + `_selected.npz`.

**Correct with external truth**
```bash
uv run --project envs/deploy python -m gentle_manip.diagnostics.extrinsic_correct \
    --selected dataset/camera_calibration/eye-to-hand/charuco_hand_eye_<stamp>_selected.npz \
    --aruco-cam dataset/camera_calibration/eye-to-hand/reference_<stamp>_aruco_cam.npy \
    --ref 1 0.1613 0.0829 0.0136 --table-z 0.0138
```
Rotates the solve so the table plane is level and at 13.8 mm, shifts x/y onto the TCP-measured corner, prints
residuals at each stage and the matrix to paste. Expect: tilt → 0, height → 13.8, corner → (0, 0, ≈±2) mm;
hand-eye self-consistency worsens by 1–2 mm (09-03: 0.94→1.76, 09-05: 1.74→3.91). If the corner's z residual
after the plane fix is ≫ 3 mm, RGB (ArUco) and depth (plane) disagree — re-measure the corner.

**Adopt + re-pin**
1. Paste the printed matrix into `WORLD_T_CAM_EXT` in `gentle_manip/robot/xarm7_config.py`.
2. `... extrinsic_check` (arm still parked) — corner and plane residuals must be small.
3. `... drift_check --pin` — today's frame becomes the drift reference (the old one is kept as
   `aruco_ref_superseded_<stamp>.npz`). If the ArUco was re-measured, update `ref_tcp` in `aruco_ref.npz`.
4. Check the real-deploy crop: with the camera ~0.75 m from the board, ~0.5 % of board points sit above the
   18 mm `z_min` (0 % above 20 mm); the outlier filter should remove them — confirm in the live cloud viewer.
5. DEVLOG entry; commit `xarm7_config.py` + the reference bundle.
