#!/usr/bin/env python3
"""Diff every SHARED sim/real setting and justify each intentional difference.

The sim/real boundary is RawObs: everything above it (perception, actions, bounds) is shared
code and MUST be configured identically, or a policy trained in sim sees a different world at
deploy. This audit diffs the two config sets field by field and classifies each difference as
OK (a documented, real-only or sim-only concern) or MISMATCH (a genuine parity bug).

    uv run --project envs/sim python examples/sim2real_diagnose/parity_audit.py
"""
import sys
from pathlib import Path
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.actions.action_config import ActionConfig
from gentle_manip.robot import xarm7_config as cfg

_C = Path(__file__).resolve().parents[2] / "gentle_manip" / "configs"
SIM_OBS, REAL_OBS = "superset_soft_armfocus_board", "point_cloud_1cam_armfocus"
TASK = "single_lift_cube3_soft_board"

# differences that are CORRECT by design, with the reason
EXPECTED = {
    "outlier_voxel_size": "REAL-ONLY: denoises D435i flying pixels; sim clouds are clean so it "
                          "would be a no-op there anyway",
    "outlier_min_neighbors": "REAL-ONLY: pairs with outlier_voxel_size",
    "privileged": "SIM-ONLY: privileged fields are training labels; real deploy runs task=None",
    "quat_noise_std": "shared jitter; check the VALUES match",
}
rows, bad = [], 0

def cmp(name, a, b, note=""):
    global bad
    if a is None or b is None:  # None coerces to NaN, and NaN != NaN — compare identity first
        same = (a is None) and (b is None)
        rows.append((("ok " if same else "MISMATCH"), name, str(a), str(b),
                     note or EXPECTED.get(name, "")))
        if not same:
            globals()['bad'] = globals()['bad'] + 1
        return
    try:                       # numeric sequences compare with a tolerance...
        same = bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-9))
    except (TypeError, ValueError):
        same = a == b          # ...everything else (str lists, None, bool) exactly
    tag = "ok " if same else ("OK*" if name in EXPECTED else "MISMATCH")
    if tag == "MISMATCH":
        bad += 1
    rows.append((tag, name, str(a), str(b), note or EXPECTED.get(name, "")))

so = ObsConfig.from_dict(yaml.safe_load((_C / "obs" / f"{SIM_OBS}.yaml").read_text()))
ro = ObsConfig.from_dict(yaml.safe_load((_C / "obs" / f"{REAL_OBS}.yaml").read_text()))
sp, rp = so.point_cloud, ro.point_cloud

print(f"SIM obs : {SIM_OBS}\nREAL obs: {REAL_OBS}\nTASK    : {TASK}\n")
for f in ("cameras", "crop_min", "crop_max", "max_points", "focus_z_lo", "focus_r_ee",
          "focus_arm_weight", "pixel_sample_n", "depth_min", "depth_max"):
    cmp(f, getattr(sp, f), getattr(rp, f))
cmp("outlier_voxel_size", sp.outlier_voxel_size, rp.outlier_voxel_size)
cmp("outlier_min_neighbors", sp.outlier_min_neighbors, rp.outlier_min_neighbors)
cmp("quat_noise_std", so.quat_noise_std, ro.quat_noise_std)
cmp("include_joint_pos", so.include_joint_pos, ro.include_joint_pos)
cmp("privileged", so.privileged is not None, ro.privileged is not None)

# geometry / bounds — one constant each, so these are parity by construction
t = yaml.safe_load((_C / "tasks" / f"{TASK}.yaml").read_text())
cmp("board_thickness (sim) vs measured real 0.0138", t["board_thickness"], 0.0138)
cmp("object spawn xy (sim) vs real placement", list(t["object_spawn_xy"]), [0.30, 0.0])
cmp("EE_BOUNDS_MIN z", cfg.EE_BOUNDS_MIN[2], 0.0139, "shared constant, clipped by BOTH backends")
X = np.asarray(cfg.WORLD_T_CAM_EXT)
cmp("cam_pos (task) vs WORLD_T_CAM_EXT", list(np.round(t["cam_pos"], 8)), list(np.round(X[:3, 3], 8)))
cmp("cam_up (task) vs -WORLD_T_CAM_EXT[:,1]", list(np.round(t["cam_up"], 8)),
    list(np.round(-X[:3, 1], 8)))
cmp("cam_fov vs measured D435i VFOV", t["cam_fov"], 43.15)

w = max(len(r[1]) for r in rows) + 2
print(f"{'':9}{'field':<{w}}{'SIM':<26}{'REAL/MEASURED':<26}note")
for tag, n, a, b, note in rows:
    print(f"[{tag:^7}] {n:<{w}}{a:<26}{b:<26}{note}")
print(f"\n{bad} genuine mismatch(es).  OK* = intentional, justified above.")
