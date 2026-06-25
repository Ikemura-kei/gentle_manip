"""Grasp / penetration parameter sweep with the Genesis viewer.

Builds a single-env scene with the viewer on and auto-grasps the object once per
value of a swept parameter, so you can watch the fingers vs the object and read a
penetration / calibrated-width / held result for each value. Edit the SWEEP block
below to choose the parameter, range, and step count; everything else stays at
the FIXED values.

Run (rigid red cube is the interesting case):
    MUJOCO_GL=glfw uv run --project envs/sim python examples/grasp_tuner.py

A small pygame window also opens (focus it): SPACE pauses, ESC quits early.
A results table prints at the end. Needs a display.

================================ EDIT ME ================================="""

# Which knob to sweep, and over what range. One of:
#   runtime (one build, fast):   "friction", "flim", "grip_close", "depth",
#                                "kp_outer", "kp_finger", "kp_inner"
#   rebuild (one build per step): "tc" (constraint_timeconst), "substeps"
SWEEP_PARAM = "flim"
SWEEP_MIN = 15
SWEEP_MAX = 50
SWEEP_STEPS = 5
SWEEP_LOG = True          # True: geometric spacing (good for kp/flim); False: linear

# Held constant for every grasp in the sweep (the swept one is overridden).
FIXED = {
    "object": "red_cube",
    "object_type": "rigid",     # "rigid" | "soft"
    "friction": 1.0,            # rigid<->rigid friction [0.01, 5]; set on robot AND object
    "flim": 1.0,               # grip dof force range (<=0 keeps URDF default)
    # Gripper KP per linkage role (each shared by the symmetric left+right joints).
    # All 6 joints share one target angle (mimic emulation); only the stiffness differs.
    "kp_outer": 10000.0,         # outer/drive knuckles (drive_joint, right_outer_knuckle)
    "kp_finger": 10000.0,        # finger joints at the pads (left/right_finger_joint)
    "kp_inner": 10000.0,         # inner knuckles (left/right_inner_knuckle_joint)
    "grip_close": 0.02,          # close-target width (0 = full close = worst-case penetration)
    "depth": 0.006,             # fingertip z at the bottom of the descent
    "tc": 0.0001,                 # constraint_timeconst (contact stiffness)
    "substeps": None,           # None = scene default (6); finer => tc floor 2*dt/substeps drops
}
CLOSE_RAMP_STEPS = 15           # ramp the gripper open->target over this many steps (slow close)
HOLD_STEPS = 25                 # extra steps to dwell on the lifted pose (so you can see it)

# ========================================================================="""
import os

if os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
    os.environ["MUJOCO_GL"] = "glfw"

import argparse
import dataclasses

import numpy as np
import pygame

from gentle_manip.assets.registry import get_object_def
from gentle_manip.envs.genesis_worker import GenesisWorker
from gentle_manip.robot.xarm7_sim import _np
from gentle_manip.tasks.single_lift import SingleLiftTask

DOWN = np.array([[0.0, 1.0, 0.0, 0.0]], np.float32)   # gripper pointing down (wxyz)
REBUILD_PARAMS = {"tc", "substeps"}


def sweep_values():
    if SWEEP_PARAM == "substeps":
        vals = np.unique(np.round(np.linspace(SWEEP_MIN, SWEEP_MAX, SWEEP_STEPS)).astype(int))
        return [int(v) for v in vals]
    space = np.geomspace if SWEEP_LOG else np.linspace
    return [float(v) for v in space(SWEEP_MIN, SWEEP_MAX, SWEEP_STEPS)]


def build(params, show_fps):
    spec = SingleLiftTask(
        {"object_name": params["object"], "object_type": params["object_type"]}
    ).scene_spec
    if params["substeps"]:
        spec = dataclasses.replace(spec, sim_substeps=int(params["substeps"]))
    # Gripper KP is set per linkage-role at runtime (below); build uses cfg defaults.
    w = GenesisWorker(spec, 1, settle_steps=40, show_viewer=True,
                      constraint_timeconst=float(params["tc"]), show_fps=show_fps)
    return w


def main():
    ap = argparse.ArgumentParser(description="Grasp parameter sweep (config at top of file).")
    ap.add_argument("--show-fps", action="store_true",
                    help="show Genesis per-step FPS logging (off by default)")
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_mode((460, 80))
    pygame.display.set_caption("grasp sweep — focus me: SPACE pause, ESC quit")
    print(__doc__.split("====")[0])

    values = sweep_values()
    obj_w = float(get_object_def(FIXED["object"]).size[1])
    rebuild = SWEEP_PARAM in REBUILD_PARAMS
    print(f"sweep {SWEEP_PARAM}: {values}   (object={FIXED['object']} width={obj_w*100:.1f}cm, "
          f"{'rebuild/step' if rebuild else 'runtime'})\n")

    flags = {"quit": False, "paused": False}

    def poll():
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    flags["quit"] = True
                elif e.key == pygame.K_SPACE:
                    flags["paused"] = not flags["paused"]

    results = []
    worker = None
    try:
        for val in values:
            if flags["quit"]:
                break
            params = dict(FIXED); params[SWEEP_PARAM] = val

            if worker is None or rebuild:
                if worker is not None:
                    worker.close()
                worker = build(params, args.show_fps)
            w = worker
            robot, grip = w.robot.robot, w.robot.grip_dofs
            n = len(grip)
            lf, rf = robot.get_link("left_finger"), robot.get_link("right_finger")
            cube = w.handle.objects[0]

            w.reset()
            for ent in (robot, cube):
                try:
                    ent.set_friction(float(params["friction"]))
                except Exception as e:
                    print(f"set_friction failed: {e}")
            if float(params["flim"]) > 0:
                robot.set_dofs_force_range(np.full(n, -params["flim"]), np.full(n, params["flim"]), grip)
            # Per-role gripper KP (grip dof order = [drive(L-out), L-fing, L-inn,
            # R-out, R-fing, R-inn]); each role shared by its symmetric L/R pair.
            for role, idx in (("kp_outer", (0, 3)), ("kp_finger", (1, 4)), ("kp_inner", (2, 5))):
                robot.set_dofs_kp(np.full(2, float(params[role]), np.float32),
                                  [grip[idx[0]], grip[idx[1]]])

            c = _np(cube.get_pos())[0]
            cx, cy = float(c[0]), float(c[1])
            z = float(params["depth"])

            def run(steps, zz, g, tag, g_end=None):
                # g_end set => ramp the gripper command g->g_end across `steps` (slow close).
                for i in range(steps):
                    poll()
                    while flags["paused"] and not flags["quit"]:
                        poll(); w.handle.scene.step()
                    if flags["quit"]:
                        return
                    gc = g if g_end is None else g + (g_end - g) * (i + 1) / steps
                    w.robot.apply_target(np.array([[cx, cy, zz]], np.float32), DOWN,
                                         np.full(1, gc, np.float32))
                    w.handle.scene.step()
                    la = _np(lf.get_AABB()).reshape(-1, 2, 3)[0]
                    ra = _np(rf.get_AABB()).reshape(-1, 2, 3)[0]
                    lp, rp = _np(lf.get_pos())[0][1], _np(rf.get_pos())[0][1]
                    pg = (ra[0, 1] - la[1, 1]) if lp < rp else (la[0, 1] - ra[1, 1])
                    gw = float(w.robot.read_state()["gripper_width"][0])
                    print(f"  {SWEEP_PARAM}={val!s:>9} [{tag:7}] gw={gw:.4f} pad_gap={pg:.4f} "
                          f"pen~{max(0, (obj_w - pg) / 2) * 1000:4.1f}mm   ", end="\r", flush=True)

            run(40, z + 0.10, 0.08, "approach")
            run(60, z, 0.08, "descend")
            run(CLOSE_RAMP_STEPS, z, 0.08, "close", g_end=float(params["grip_close"]))  # slow close
            run(60, z, float(params["grip_close"]), "settle")                            # let it settle
            gw_close = float(w.robot.read_state()["gripper_width"][0])
            run(70, z + 0.13, float(params["grip_close"]), "lift")
            run(HOLD_STEPS, z + 0.13, float(params["grip_close"]), "hold")
            cz = float(_np(cube.get_pos())[0][2])
            held = cz > 0.06
            results.append((val, gw_close, cz, held))
            print(f"\n[done] {SWEEP_PARAM}={val!s:<10} gw_close={gw_close:.4f}  lift_z={cz:.3f}  "
                  f"held={'YES' if held else 'no'}")
    finally:
        if worker is not None:
            worker.close()
        pygame.quit()

    print(f"\n===== sweep results ({SWEEP_PARAM}) =====")
    print(f"{SWEEP_PARAM:>12}  gw_close  lift_z  held")
    for val, gw, cz, held in results:
        print(f"{val!s:>12}  {gw:7.4f}  {cz:6.3f}  {'YES' if held else 'no'}")


if __name__ == "__main__":
    main()
