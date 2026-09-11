"""Stage 4: register one prepped mesh (registry entry + task/dr/experiment yaml trio),
following docs/final/adding_new_objects.md sections 3-4 and the bs_cube / bs_* precedent
exactly (same DR block, same experiment template, same header convention).

Additive only: appends a new OBJECT_MAP.update({...}) block before `def get_object_def`, never
touches an existing entry. Idempotent: re-running with the same name overwrites that object's
own generated files (registry insertion is guarded against duplicate keys).

Usage: called as a library from the batch driver (batch_expand.py); see __main__ for a
single-object CLI example.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PY = REPO_ROOT / "gentle_manip" / "assets" / "registry.py"
TASKS_DIR = REPO_ROOT / "gentle_manip" / "configs" / "tasks"
DR_DIR = REPO_ROOT / "gentle_manip" / "configs" / "dr"
EXP_DIR = REPO_ROOT / "gentle_manip" / "configs" / "experiments"

BOARD_THICKNESS = 0.0138
CAM = dict(fov=43.15, pos=[0.78032539, -0.00362529, 0.28947945],
           lookat=[0.25932406, 0.01186746, 0.01380000], up=[-0.46799099, -0.01495762, 0.88360665])


def already_registered(name: str) -> bool:
    txt = REGISTRY_PY.read_text()
    return re.search(rf'^\s*"{re.escape(name)}"\s*:\s*ObjectDef', txt, re.M) is not None


def append_registry_entry(name: str, material_key: str, extents_m: tuple[float, float, float],
                          default_pos_z: float, source_note: str) -> None:
    if already_registered(name):
        raise ValueError(f"{name!r} already in OBJECT_MAP -- pick a new name (additive-only registry)")
    txt = REGISTRY_PY.read_text()
    sx, sy, sz = extents_m
    block = (
        f"\n# {source_note}\n"
        f"OBJECT_MAP.update({{\n"
        f'    "{name}": ObjectDef("{name}", MATERIALS["{material_key}"], object_type="soft",\n'
        f"                    size=({sx:.5f}, {sy:.5f}, {sz:.5f}), default_pos=(0.47, 0.0, {default_pos_z:.4f}),\n"
        f'                    mesh_path=str(_OBJ_DIR / "{name}.obj")),\n'
        f"}})\n"
    )
    marker = "\ndef get_object_def(name: str) -> ObjectDef:"
    assert marker in txt, "registry.py structure changed, update the insertion marker"
    txt = txt.replace(marker, block + marker, 1)
    REGISTRY_PY.write_text(txt)


def spawn_z(h: float, diag: float, s_max: float = 1.6) -> float:
    """docs/final/adding_new_objects.md OBJECT-SPECIFIC spawn_z formula. s_max=1.6 matches the
    widened object_scale DR upper bound below (2026-09-08, user: stronger size randomization)."""
    return max(BOARD_THICKNESS + (h / 2) * s_max + 0.020,
               BOARD_THICKNESS + (diag / 2) * s_max + 0.005)


def grasp_gate_dist(diag: float) -> float:
    """Heuristic (no closed-form in the docs): half-diagonal + a ~45mm finger-reach margin,
    clamped to the observed bs_* range [0.06, 0.11]. Validated per-object by the smoke test,
    not treated as exact -- adjust if dev_synth shows a mismatch (§6 of adding_new_objects.md)."""
    return float(min(0.11, max(0.06, diag / 2 + 0.045)))


TASK_TEMPLATE = """# [task] {name} (Objaverse-LVIS {category}, object-expansion batch): {material_key} material, board rig.
# Used by: collect_demos_synth_v4 / training / eval via the experiment yaml
# Status: active
object_name: "{name}"
object_type: "soft"
sim_substeps: {substeps}
mpm_grid_density: {grid_density}
object_spawn_z: {spawn_z:.4f}   # board + max(h/2*1.6+20mm, diag/2*1.6+5mm); h={h_mm:.1f}mm diag={diag_mm:.1f}mm (s_max=1.6 matches the widened object_scale DR)
success_z_min: 0.175
success_z_max: 0.275
hold_steps: 30
success_scale: 0.2
board_thickness: 0.0138
board_size: [0.6, 0.7, 0.0138]
board_center: [0.41, 0.0]
board_color: [0.647, 0.608, 0.514, 1.0]
finger_color: [0.1, 0.1, 0.1, 1.0]
cam_fov: {cam_fov}
cam_pos: {cam_pos}
cam_lookat: {cam_lookat}
cam_up: {cam_up}
rewards:
  stress: {{cap: 1.5, mean_weight: 0.2, scale: 0.2, top10_weight: 0.8}}
  dist_to_obj: {{decay: 8.0, scale: 0.1}}
  lift: {{grasp_gate_dist: {grasp_gate:.3f}, lift_target: 0.16, scale: 0.1}}
"""

DR_TEMPLATE = """# [dr] {name}: realws pose/shape/material DR. WIDENED size/shape/material bands (2026-09-08,
# user: stronger randomization than the bs_/prim_ batch -- more size up/down, per-axis geometry
# variation e.g. cylinder diameter vs height, wider E/nu). object_scale [0.6,1.6] (was [0.9,1.2]);
# object_axis_scale [0.6,1.6] on a random x/y/z axis each scene, so a cylinder-like mesh gets
# independently thinner/fatter or shorter/taller draws, not just uniform resize. Grasp-cap margin:
# the mesh is pre-scaled at prep time so min-width * 1.6 (this s_max) stays under the 79mm planner
# cap (gentle_manip/scripts/object_expansion/geom_filter.py target_min_width=0.040).
# Used by: the {name} experiment yaml
# Status: active
object_pos_x: [0.30, 0.46]
object_pos_y: [-0.12, 0.12]
object_nominal_xy: [0.47, 0.0]
robot_init_pos_xyz: 0.02
object_yaw_deg: 180
object_pitch_roll_deg: 45
object_flip_prob: 0.25
object_flip_deg: [160, 180]
coup_friction: [3.5, 4.5]
object_scale: [0.6, 1.6]
object_axis_scale: [0.6, 1.6]
object_bend_deg: [-4, 4]
object_twist_deg: [-4, 4]
object_taper: [-0.05, 0.05]
object_E: [1.5e5, 5.0e5]
object_nu: [0.28, 0.42]
object_rho: [700, 1300]
start_modes: {{home: 0.6, in_air: 0.15, above_object: 0.15, mid_approach: 0.1}}
disturbance_prob: 0.1
"""

EXP_TEMPLATE = """# [experiment] {name}: 7d euler abs action, arm-focus cloud, realws DR, D435i noise.
# Used by: collect_demos_synth_v4 / DPPO training + eval / deploy
# Status: active (object-expansion batch)

task: single_lift_{name}_soft
action: abs_pose_euler_abs_gripper_z15
dr: soft_orientation_realws_{name}
augmentation: d435i_noise

obs: superset_soft_armfocus_board
views: {{student: [point_cloud], teacher: [privileged]}}

rl: {{batch_size: 256, discount: 0.99, max_episode_steps: 400, random_steps: 300, training_starts: 5000}}
"""


def write_configs(name: str, category: str, material_key: str, h_m: float, diag_m: float,
                  thin_axis_m: float) -> dict:
    substeps, grid_density = (235, 250) if thin_axis_m >= 0.008 else (470, 500)
    sz = spawn_z(h_m, diag_m)
    gg = grasp_gate_dist(diag_m)
    task_txt = TASK_TEMPLATE.format(
        name=name, category=category, material_key=material_key, substeps=substeps,
        grid_density=grid_density, spawn_z=sz, h_mm=h_m * 1000, diag_mm=diag_m * 1000,
        cam_fov=CAM["fov"], cam_pos=CAM["pos"], cam_lookat=CAM["lookat"], cam_up=CAM["up"],
        grasp_gate=gg)
    dr_txt = DR_TEMPLATE.format(name=name)
    exp_txt = EXP_TEMPLATE.format(name=name)
    (TASKS_DIR / f"single_lift_{name}_soft.yaml").write_text(task_txt)
    (DR_DIR / f"soft_orientation_realws_{name}.yaml").write_text(dr_txt)
    (EXP_DIR / f"single_lift_{name}_soft_abs_action_armfocus_7d_realws.yaml").write_text(exp_txt)
    return {"spawn_z": sz, "grasp_gate_dist": gg, "substeps": substeps, "grid_density": grid_density}


def register(name: str, category: str, material_key: str, extents_m: tuple[float, float, float],
            default_pos_z: float, source_note: str) -> dict:
    sx, sy, sz = extents_m
    diag = math.sqrt(sx * sx + sy * sy + sz * sz)
    append_registry_entry(name, material_key, extents_m, default_pos_z, source_note)
    cfg = write_configs(name, category, material_key, h_m=sz, diag_m=diag, thin_axis_m=min(sx, sy, sz))
    return cfg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--material", default="soft_shape")
    ap.add_argument("--extents-mm", type=float, nargs=3, required=True)
    ap.add_argument("--default-pos-z", type=float, required=True)
    ap.add_argument("--source-note", default="")
    args = ap.parse_args()
    ext_m = tuple(x / 1000 for x in args.extents_mm)
    cfg = register(args.name, args.category, args.material, ext_m, args.default_pos_z,
                   args.source_note or f"# {args.name} (Objaverse-LVIS {args.category})")
    print(cfg)
