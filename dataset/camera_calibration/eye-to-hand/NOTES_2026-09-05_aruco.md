# 2026-09-05 — table ArUco reference (camera moved; old WORLD_T_CAM_EXT obsolete)

Marker: 5x5 dict (DICT_5X5_50), id 1, 80 mm, stuck on the wooden board (13.8 mm), board rotated so the
marker sits at the BACK of the board near the arm base. Detected with `diagnostics/aruco_check.py` at
640x480 (note: the D435i 4:3 color modes are a centre CROP of the 16:9 sensor — HFOV 55.6 vs 70 deg).

Corner indices follow the marker's own orientation (cv2 order), as labelled in
`aruco_check_2026-09-05-17-11-17.png`; the marker is upside-down in the image.

| idx | label | pixel (u,v) | camera PnP (m) |
|---|---|---|---|
| 0 | top-left | (456, 205) | [0.1503, -0.0469, 0.6841] |
| 1 | top-right | (385, 205) | [0.0703, -0.0467, 0.6823] |
| 2 | bottom-right | (378, 179) | [0.0687, -0.0839, 0.7531] |
| 3 | bottom-left | (442, 179) | [0.1487, -0.0840, 0.7549] |

## Robot-frame ground truth (TCP touch, web UI, ~2-3 mm accuracy)

Only corner **1** (image-frame bottom-left, pixel 385,205) was reachable; corners 2/3 are too close to
the base, 0 was not measured.

| idx | robot-frame TCP (m) |
|---|---|
| 1 | (0.1613, 0.0829, 0.0136) |

z 13.6 mm == board top (13.8 mm): consistent. Use: translation check of any extrinsic at this point
(`aruco_check.py --ref 1 0.1613 0.0829 0.0136`); together with the table plane (tilt + height) it fixes
5 of 6 DOF of a correction, yaw comes from the hand-eye solve.

## Replayed round 2026-09-05-17-22-02 (calib_replay.py, 13 poses, --start-z 0.35, outlier #7 of the 09-03 round dropped)

- calib_select: HORAUD on 6 poses, 12/13 inliers, median 1.85 mm; 4/5 solvers within 4 mm; camera at
  (0.768, 0.002, 0.290) m, 28.8 deg down. Table check (raw): tilt 1.71 deg, height 7.7 mm (expect 13.8).
- `extrinsic_correct.py` (plane fitted once on a 10-frame median cloud, 181k inliers): plane fix -> tilt 0,
  height 13.8; ArUco corner 1 error (-7.9, -2.6, -9.7) -> (-7.9, -2.6, +1.9) -> after x/y shift (0, 0, +1.9) mm.
  Correction vs raw solve: 1.71 deg, 13.2 mm. RGB-PnP corner vs depth plane disagree by 1.9 mm in z (independent).
- Cost: board-in-gripper self-consistency 1.74 -> 3.91 mm median (max 7.1). On 09-03 it was 0.94 -> 1.76.
- Crop: board-plane inliers through the corrected matrix: z median 13.8, p95 15.3, p99 17.0 mm; 0.46 % above
  18 mm, 0 % above 20 mm (camera now 0.37-1.43 m from the board -> more depth noise than on 09-03).
- Result: `charuco_hand_eye_2026-09-05-17-22-02_corrected.npz` (T = corrected, T_hand_eye = raw). NOT yet in
  xarm7_config.py.
