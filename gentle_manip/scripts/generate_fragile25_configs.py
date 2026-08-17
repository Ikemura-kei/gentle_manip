"""One-time generator for the fragile-food 25-category campaign's task / DR /
experiment YAML triples (gentle_manip, 2026-08-13). Mirrors
single_lift_mushroom_soft.yaml + food_shape_mushroom_soft_easy.yaml +
single_lift_mushroom_soft_easy.yaml (this session's validated "_easy" toy-
task recipe) for every object in the 25-category roster, substituting only
object_name / sim_substeps / mpm_grid_density / the per-object narrow
material-E/nu/rho band (computed from each object's OWN registered nominal,
not copy-pasted from mushroom's).

Run once:
    uv run --project envs/sim python -m gentle_manip.scripts.generate_fragile25_configs
Writes (skips any that already exist, use --overwrite to force):
    gentle_manip/configs/tasks/single_lift_<obj>_soft.yaml
    gentle_manip/configs/dr/food_shape_<obj>_soft_easy.yaml
    gentle_manip/configs/experiments/single_lift_<obj>_soft_easy.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gentle_manip.assets.registry import get_object_def  # noqa: E402

CFG = REPO / "gentle_manip" / "configs"

# 20 train + 5 zero-shot test (test objects still need task/DR/experiment
# configs -- the eval harness spawns and scores them -- they just skip
# collection/specialist-training/rollout in later phases).
TRAIN = ["tofu", "mushroom", "shiitake", "fish_raw", "beef_raw", "blueberry",
        "raspberry", "grape", "avocado", "kiwi", "sponge", "egg_boiled",
        "strawberry", "peach", "banana", "tomato", "chicken_breast", "shrimp",
        "cheese", "pasta_bundle"]
TEST = ["blackberry", "scallop", "dumpling", "gelatin"]   # watermelon dropped: reproducible MPM
                                                          # divergence in its DR range (2026-08-17)
ROSTER = TRAIN + TEST

TASK_TEMPLATE = '''# [task] Soft-MPM {obj} single-lift: object + reward (dist/lift/stress) + success band + sim dynamics.
# Used by: experiments (task:) -> DP3/DPPO, demo collection
# Status: experimental
#
# Fragile-food 25-category campaign (2026-08-13). Mirrors single_lift_mushroom_soft.yaml's
# validated structure (Config C reward shaping/success-band convention) -- only object_name
# and the MPM stability params (sim_substeps/mpm_grid_density, from registry.py's
# sim_substeps_override when set) differ per object.
success_z_min: 0.175
success_z_max: 0.275
hold_steps: 30
object_name: "{obj}"
object_type: "soft"

sim_substeps: {substeps}
mpm_grid_density: {grid_density}

success_scale: 0.2

rewards:
  stress:
    scale: 0.2
    cap: 1.5
    mean_weight: 0.2
    top10_weight: 0.8
  dist_to_obj:
    scale: 0.1
    decay: 8.0
  lift:
    scale: 0.1
    grasp_gate_dist: 0.079
    lift_target: 0.16
'''

DR_TEMPLATE = '''# [dr] TOY/EASY variant of food_shape for SOFT (MPM) {obj} -- fragile-food
# 25-category campaign (2026-08-13). Mirrors food_shape_mushroom_soft_easy.yaml's
# narrow-DR recipe (validated: mushroom-soft-easy scored 75.0%, the best specialist
# result of the whole session). Material E/nu/rho band is narrowed around THIS
# object's own registered nominal (not mushroom's) -- {e_nominal:.3g} Pa +/-7%.
# Used by: experiments (dr:) single_lift_{obj}_soft_easy
# Status: experimental
object_pos_xy: 0.02
robot_init_pos_xyz: 0.01
object_yaw_deg: 180
object_pitch_roll_deg: 8

object_scale:      [0.95, 1.05]
object_bend_deg:   [-2.0, 2.0]
object_twist_deg:  [-2.0, 2.0]
object_taper:      [-0.02, 0.02]
object_axis_scale: [0.97, 1.03]

object_E:  [{e_lo:.4g}, {e_hi:.4g}]
object_nu: [{nu_lo:.3g}, {nu_hi:.3g}]
object_rho: [{rho_lo:.4g}, {rho_hi:.4g}]
coup_friction: [3.5, 4.5]
'''

EXPERIMENT_TEMPLATE = '''# [experiment] TOY/EASY SOFT (MPM) {obj} lift -- fragile-food 25-category campaign
# (2026-08-13), mirroring single_lift_mushroom_soft_easy's validated recipe.
# Used by: serl_sim_server.py; grasp_synthesis collect_demos_synth_v2.py; DP3/DPPO
# Status: experimental
task: single_lift_{obj}_soft
action: delta_pose_delta_gripper
dr: food_shape_{obj}_soft_easy
augmentation: l515_noise

obs: superset_soft
views:
  teacher: [privileged]
  student: [point_cloud]

rl:
  random_steps: 200
  training_starts: 400
  batch_size: 256
  discount: 0.92
  utd_ratio: 2
  critic_lr: 1.0e-4
'''


def generate_one(obj: str, overwrite: bool) -> None:
    od = get_object_def(obj)
    m = od.material
    substeps = od.sim_substeps_override or 220
    grid_density = od.mpm_grid_density_override or 250

    task_path = CFG / "tasks" / f"single_lift_{obj}_soft.yaml"
    dr_path = CFG / "dr" / f"food_shape_{obj}_soft_easy.yaml"
    exp_path = CFG / "experiments" / f"single_lift_{obj}_soft_easy.yaml"

    if task_path.exists() and not overwrite:
        print(f"[generate_configs] {obj}: task config exists, skipping (--overwrite to force)")
    else:
        task_path.write_text(TASK_TEMPLATE.format(obj=obj, substeps=substeps,
                                                   grid_density=grid_density))
        print(f"[generate_configs] {obj}: wrote {task_path}")

    if dr_path.exists() and not overwrite:
        print(f"[generate_configs] {obj}: DR config exists, skipping")
    else:
        dr_path.write_text(DR_TEMPLATE.format(
            obj=obj, e_nominal=m.youngs_modulus,
            e_lo=m.youngs_modulus * 0.93, e_hi=m.youngs_modulus * 1.07,
            nu_lo=max(0.05, m.poisson_ratio * 0.97), nu_hi=min(0.49, m.poisson_ratio * 1.03),
            rho_lo=m.density * 0.98, rho_hi=m.density * 1.02))
        print(f"[generate_configs] {obj}: wrote {dr_path}")

    if exp_path.exists() and not overwrite:
        print(f"[generate_configs] {obj}: experiment config exists, skipping")
    else:
        exp_path.write_text(EXPERIMENT_TEMPLATE.format(obj=obj))
        print(f"[generate_configs] {obj}: wrote {exp_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--objects", nargs="+", default=None,
                    help="subset to (re)generate, default: the full 25-object roster")
    args = ap.parse_args()

    objects = args.objects or ROSTER
    for obj in objects:
        generate_one(obj, args.overwrite)
    print(f"\n[generate_configs] DONE — {len(objects)} object(s). "
         f"TRAIN={len(TRAIN)} TEST={len(TEST)}")


if __name__ == "__main__":
    main()
