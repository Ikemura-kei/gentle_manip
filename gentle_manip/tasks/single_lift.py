from __future__ import annotations

import numpy as np

from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.scenes.scene_spec import CameraEntry, FixtureEntry, ObjectEntry, SceneSpec
from gentle_manip.tasks.base_task import BaseTask


class SingleLiftTask(BaseTask):
    """Lift one object to a target height and hold it there for hold_steps steps.

    Success: object_center_z > initial_z + lift_height, sustained for hold_steps
    consecutive steps.
    """

    def __init__(self, task_cfg: dict) -> None:
        super().__init__(task_cfg)
        self.lift_height: float = float(task_cfg.get("lift_height", 0.15))
        self.hold_steps: int = int(task_cfg.get("hold_steps", 30))
        self.object_name: str = str(task_cfg.get("object_name", "tofu"))
        self.object_type: str = str(task_cfg.get("object_type", "soft"))  # "soft" | "rigid"
        # optional wrist camera (VLA baselines want a base + wrist view); OFF by default
        self.wrist_camera: bool = bool(task_cfg.get("wrist_camera", False))
        self.backdrop: bool = bool(task_cfg.get("backdrop", False))

        # Success: either an ABSOLUTE object-center z-band [min, max] (ruler-checkable on
        # the real robot — the center must sit in this height window), or, if unset, the
        # legacy RELATIVE "lifted lift_height above the initial z". Held hold_steps either way.
        z_min = task_cfg.get("success_z_min")
        z_max = task_cfg.get("success_z_max")
        self.success_z_min = float(z_min) if z_min is not None else None
        self.success_z_max = float(z_max) if z_max is not None else None

        # MPM sim params — configurable so a stiff soft body (mushroom) can raise
        # substeps for CFL stability without touching the rigid-cube defaults. The
        # mushroom uses "Config C" (see materials.py / CLAUDE.md): substeps=210,
        # mpm_grid_density=250 at E=0.3 MPa.
        self.sim_substeps: int = int(task_cfg.get("sim_substeps", 80))
        self.mpm_grid_density: float = float(task_cfg.get("mpm_grid_density", 300.0))
        self.cam_fov: float = float(task_cfg.get("cam_fov", 43.15))  # D435i VFOV (measured)
        # cam_ext (depth -> point cloud) extrinsic. Default = the 2026-09-05 D435i WORLD_T_CAM_EXT
        # (pos + optical axis to the board plane); task cfgs carry the same values (+ cam_up).
        self.cam_pos: tuple = tuple(task_cfg.get("cam_pos", (0.78032539, -0.00362529, 0.28947945)))
        self.cam_lookat: tuple = tuple(task_cfg.get("cam_lookat", (0.25932406, 0.01186746, 0.01380000)))
        # Camera ROLL. pos+lookat leave rotation about the optical axis free and Genesis defaults
        # to up=(0,0,1), which is ~3 deg off the real D435i mounting. Set cam_up = -world_T_cam[:3,1]
        # to reproduce a calibrated extrinsic exactly. None => Genesis default.
        _up = task_cfg.get("cam_up")
        self.cam_up: tuple | None = tuple(_up) if _up is not None else None
        # BOARD (2026-09-03): the real rig has a 13.8 mm board covering the workspace, and the
        # object rests on IT, not on the table. `board_thickness` > 0 adds it; the object's spawn z
        # must be raised by the same amount. Size defaults to cover the whole point-cloud crop so
        # no board EDGE falls inside it (the real board is ~0.45 x 0.60 m, wider than the crop).
        self.board_thickness: float = float(task_cfg.get("board_thickness", 0.0))
        self.board_size: tuple = tuple(task_cfg.get("board_size", (0.60, 0.70, 0.0138)))
        self.board_center: tuple = tuple(task_cfg.get("board_center", (0.42, 0.0)))
        # RGBA 0-1; None = Genesis default. Sampled from the real board in the cam_ext RGB
        # (median #a59b83) so renders match the hardware for paper figures.
        _bc = task_cfg.get("board_color")
        self.board_color = tuple(_bc) if _bc is not None else None
        _fc = task_cfg.get("finger_color")
        self.finger_color = tuple(_fc) if _fc is not None else None
        # optional spawn-height override (m); None => registry default_pos z. Used to clear the
        # MPM domain padding at coarse grid_density (see ObjectEntry.spawn_z).
        _sz = task_cfg.get("object_spawn_z")
        self.object_spawn_z: float | None = float(_sz) if _sz is not None else None
        _sxy = task_cfg.get("object_spawn_xy")
        self.object_spawn_xy = tuple(float(v) for v in _sxy) if _sxy is not None else None
        # MPM domain. The default is the mushroom-tuned box below and MUST stay exactly that, so
        # existing tasks are untouched. It is configurable because domain volume and grid density
        # trade off directly: cost ~ volume x density^3, so TIGHTENING the box is what buys the
        # resolution a small object needs. A 1.5 cm raspberry spans only ~4 cells at the mushroom's
        # 4 mm cells; shrinking the domain lets its density rise without a runaway cost.
        # Genesis insets the usable region by 3 cells on every face (boundary_padding = 3*dx), so a
        # coarser grid raises the effective floor — pair any change with object_spawn_z.
        # ONE shared MPM box for every object (2026-09-04, D435i board rig). Sized as the UNION of every
        # realws DR spawn box (x [0.29,0.48], y +-0.11/0.12, all scales/orientations, lifted to
        # success_z_max) + Genesis's 3*dx boundary padding + 5 mm, verified by sampling 3000 spawns per
        # object: 0 % clipped for all 20 objects. The per-task boxes it replaces clipped 93-100 % of
        # spawns for banana/banana_chunk/cherry_tomato/pasta_bundle/tomato at the new x=0.29 spawn
        # (a clipped particle is silently killed -> the "5 % grasp yield" failure). 1.36 M cells at
        # grid 300, same cost as the previous default. A task may still override, but should not need to.
        _mb = task_cfg.get("mpm_bounds")
        self.mpm_bounds: tuple = (tuple(tuple(float(v) for v in side) for side in _mb) if _mb
                                  else ((0.21, -0.19, -0.03), (0.56, 0.19, 0.35)))

        self._initial_z: np.ndarray | None = None
        self._success_counter: np.ndarray | None = None

    @property
    def scene_spec(self) -> SceneSpec:
        # Sim params + camera are the values validated in the dev prototype
        # (examples/gs_sim_backend_dev.py): the grasp reproduces only with this
        # dt/substeps/mpm_bounds/grid_density. One external camera matches the real
        # single-camera rig (cam_ext at the calibrated WORLD_T_CAM_EXT pose).
        # BACKDROP (opt-in, task cfg `backdrop: true`). Occludes the NEIGHBOURING PARALLEL ENVS
        # that otherwise appear on cam_ext's horizon (see fixtures.add_fixtures for why MPM scenes
        # cannot separate envs in the renderer). They MOVE, so they are dynamic distractors in the
        # RGB observation; the point cloud was never affected because it is cropped.
        # Walls sit OUTSIDE the point-cloud crop (x<=0.71, |y|<=0.215), so every point-cloud
        # experiment stays bit-identical. Opt-in: only RGB/VLA tasks should enable it.
        _fixtures = [FixtureEntry(fixture_type="table")]
        if self.board_thickness > 0.0:
            # A Box fixture is placed by its CENTRE, so z = thickness/2 puts its top face at
            # exactly `board_thickness` above the table (= the real 13.8 mm surface the gripper
            # touch measured at 13.9-14.1 mm).
            _bt = float(self.board_thickness)
            _fixtures.append(FixtureEntry(
                fixture_type="chopping_board",
                pose=(self.board_center[0], self.board_center[1], _bt / 2.0),
                params={"height": _bt,
                        "size": (self.board_size[0], self.board_size[1], _bt),
                        **({} if self.board_color is None else {"color": self.board_color})}))
        if self.backdrop:
            # HEIGHT 0.9 m, NOT 1.5 (2026-08-31). Genesis' single default light is
            # DirectionalLight(dir=(-1,-1,-1)) — i.e. it comes from +x+y+z — and shadows are on,
            # so the +y wall casts a shadow band over y in [1.3-h, 1.3] at table height. At
            # h=1.5 that is [-0.20, 1.30], which covers the whole workspace (|y|<=0.30): the
            # scene rendered DARK. At h=0.9 the band is [0.40, 1.30] and clears it.
            # 0.9 m is still ample for OCCLUSION: cam_ext (pos x=0.989, VFOV 46 -> HFOV 59)
            # sees the back wall at 1.54 m, where the frame top is z=0.77. So 0.9 covers the
            # full frame with margin, and the side walls only enter frame at x=-1.13, i.e.
            # already behind the back wall. Verified geometrically AND by looking at the
            # recorded observation frames (see docs/figures/pi05_obs_*.mp4).
            _H = 0.9
            _fixtures += [
                FixtureEntry(fixture_type="backdrop", pose=(-0.55, 0.0, _H / 2),
                             params={"size": (0.02, 4.0, _H)}),      # behind the robot
                FixtureEntry(fixture_type="backdrop", pose=(0.3, -1.2, _H / 2),
                             params={"size": (4.0, 0.02, _H)}),      # side wall -y
                FixtureEntry(fixture_type="backdrop", pose=(0.3, 1.2, _H / 2),
                             params={"size": (4.0, 0.02, _H)}),      # side wall +y
            ]
        return SceneSpec(
            objects=[ObjectEntry(name=self.object_name, object_type=self.object_type,
                                 spawn_xy=self.object_spawn_xy,
                                 spawn_z=self.object_spawn_z)],
            fixtures=_fixtures,
            cameras=[
                CameraEntry(
                    name="cam_ext",
                    # default = calibrated L515; task-cfg cam_pos/cam_lookat override (camera study)
                    pos=self.cam_pos,
                    lookat=self.cam_lookat,
                    up=self.cam_up,
                    # Genesis fov is VERTICAL (camera.f = 0.5*res[1]/tan(fov/2), height-based).
                    # MATCHED TO THE D435i 2026-09-03: measured live colour intrinsics at 640x480
                    # are fx=607.01 fy=606.96 (HFOV 55.59, VFOV 43.15, DFOV 66.77). fov=43.15
                    # reproduces f=606.94 — within 0.01% of the real focal length.
                    # RESIDUAL, not expressible in Genesis: the real principal point is
                    # (322.56, 246.58) while Genesis forces (res/2) = (320, 240). The 6.6 px
                    # y-offset is ~0.6 deg of pointing — a known small sim/real backprojection
                    # difference, not something fov can absorb.
                    fov=self.cam_fov,
                ),
            ] + ([
                # OPTIONAL WRIST CAMERA (task cfg `wrist_camera: true`, 2026-08-29). Its pose is a
                # PLACEHOLDER — GenesisWorker re-poses it every step per env from
                # world_T_ee @ EE_T_CAM_WRIST, mirroring RealBackend. fov 58 ~ the D405's VFOV.
                # Added for VLA baselines that expect a base + wrist view; OFF by default so every
                # existing dataset and policy is unaffected.
                CameraEntry(name="cam_wrist", pos=(0.4, 0.0, 0.4),
                            lookat=(0.4, 0.0, 0.0), fov=58.0),
            ] if self.wrist_camera else []),
            finger_color=self.finger_color,
            sim_dt=1.0 / 30.0,
            sim_substeps=self.sim_substeps,
            # z-floor -0.02 (was -0.012): genesis pads the MPM domain inward by ~0.012, so
            # -0.012 gave a padded floor of exactly 0.0 = ZERO clearance. The mushroom rests
            # with its base at z~0, and its lowest particle can land a hair below 0 (seen on
            # aarch64/GH200: min z = -3.6e-5, 36um under the floor -> build crash). x86 rounded
            # to >=0 and passed. Dropping the floor gives real clearance + headroom for DR
            # pose tilts that dip a corner lower. Only extends the (empty) domain below the
            # plane; plane collision + physics unchanged.
            #
            # 2026-08-16: still not enough margin -- 4 overnight grasp-synthesis collections
            # (soft_orientation DR: full-range pose/shape/scale) crashed on the SAME exception
            # a few hours in (min z = -8.37e-3 vs the -8e-3 padded floor, ~370um over -- bigger
            # violation than the 36um one above, wider DR ranges push particles further). Added
            # 2mm of margin on ALL SIX faces (not just z) as a general safety buffer -- cheap
            # (only extends the empty domain) and z is not provably the only direction that can
            # be violated by some DR combination we haven't hit yet.
            # (the literal above is now self.mpm_bounds' default — overridable per task, see __init__)
            mpm_bounds=self.mpm_bounds,
            mpm_grid_density=self.mpm_grid_density,
        )

    def reset(self, sim_feedback: SimFeedback) -> None:
        super().reset(sim_feedback)
        num_envs = sim_feedback.object_center.shape[0]
        self._initial_z = sim_feedback.object_center[:, 2].copy()
        self._success_counter = np.zeros(num_envs, dtype=np.int32)

    def is_success(self, sim_feedback: SimFeedback, raw_obs: RawObs) -> np.ndarray:
        if self._initial_z is None or self._success_counter is None:
            return np.zeros(sim_feedback.object_center.shape[0], dtype=bool)

        obj_z = sim_feedback.object_center[:, 2]
        if self.success_z_min is not None:
            in_target = (obj_z >= self.success_z_min) & (obj_z <= self.success_z_max)  # absolute band
        else:
            in_target = obj_z > (self._initial_z + self.lift_height)                   # relative
        self._success_counter = np.where(in_target, self._success_counter + 1, 0)
        return self._success_counter >= self.hold_steps
