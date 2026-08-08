"""Sim side of the RawObs boundary: Genesis (in a subprocess) -> RawObs.

Implements the Backend protocol PolicyEnv consumes, with num_envs = N parallel
Genesis envs. Genesis itself lives in a GenesisProcess child (memory-leak fix);
SimBackend owns the policy-facing logic: it interprets actions as deltas and
accumulates them into a target EE pose + gripper, exactly as RealBackend does
(but batched), so sim and real share action semantics. Observations always
reflect the read-back state, never the command.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from gentle_manip.domain_randomization.dr_config import DRConfig
from gentle_manip.envs.genesis_process import GenesisProcess
from gentle_manip.envs.raw_obs import RawObs
from gentle_manip.envs.sim_feedback import SimFeedback
from gentle_manip.robot import xarm7_config as cfg
from gentle_manip.scenes.scene_spec import SceneSpec


class SimBackend:
    def __init__(
        self,
        spec: SceneSpec,
        num_envs: int,
        config: Optional[dict] = None,
        *,
        use_subprocess: bool = True,
        show_viewer: bool = False,
        render_cameras: bool = True,
        record_camera: bool = False,
    ) -> None:
        config = config or {}
        self.num_envs = int(num_envs)

        # State-based teacher: drop scene cameras so the worker skips the (expensive)
        # per-env depth render. Observation-only — does NOT touch the physics, so the
        # dynamics (and an off-policy replay buffer) stay consistent. Callers set this
        # from ObsConfig.needs_cameras().
        # record_camera keeps ONE camera built (for on-demand RGB clips of behaviour)
        # even when obs needs none — read_state still skips its per-step depth render
        # (worker render_obs_cameras=False below), so the teacher stays render-free.
        import dataclasses
        if not render_cameras and spec.cameras and not record_camera:
            spec = dataclasses.replace(spec, cameras=[])
        elif not render_cameras and spec.cameras and record_camera:
            spec = dataclasses.replace(spec, cameras=spec.cameras[:1])   # one clip cam
        self.render_cameras = bool(render_cameras)

        robot_overrides = config.get("robot", {})
        sim_cfg = config.get("sim", {})
        self._dr = DRConfig.from_dict(config.get("dr"))         # sim-only domain randomization
        self._gripper_max = float(
            robot_overrides.get("default_gripper_width", cfg.DEFAULT_GRIPPER_WIDTH)
        )
        self._rng = np.random.default_rng(config.get("seed", 0))

        # Per-scene DR (material/friction/size/shape) — resampled by rebuilding via
        # process.restart(). scene_dr_every>0 -> reset() re-randomizes the whole scene every N
        # resets ("full DR"); that needs restart, so force the subprocess backend.
        self._nominal_scale = float(spec.objects[0].scale) if spec.objects else 1.0
        self._deform_dir = None            # temp dir for shape-DR deformed meshes
        self._applied_scene: dict = {}     # size/shape/coup DR actually applied (for eval CSV)
        self._scene_dr_every = int(sim_cfg.get("scene_dr_every", 0))
        self._reset_count = 0
        self._in_scene_dr = False          # recursion guard (randomize_scene ends in reset())
        self._nominal_spec = spec          # camera-adjusted, PRE-DR base (rebuild from this)
        spec, coup = self._apply_scene_dr(spec)   # FULL scene DR at launch (material + size + shape)
        self._spec = spec
        if self._scene_dr_every > 0:
            use_subprocess = True          # relaunch-based scene DR needs GenesisProcess.restart
        worker_kwargs = dict(
            settle_steps=int(sim_cfg.get("settle_steps", 30)),
            settle_max_steps=int(sim_cfg.get("settle_max_steps", 200)),
            settle_vel_thresh=float(sim_cfg.get("settle_vel_thresh", 0.005)),
            coup_friction=float(coup if coup is not None else sim_cfg.get("coup_friction", 4.0)),
            robot_overrides=robot_overrides,
            # A record-only camera is built but not depth-rendered each step.
            render_obs_cameras=bool(render_cameras),
        )
        if use_subprocess:
            # Training path: Genesis in a child process (GPU-memory-leak fix). A
            # viewer can't run under the headless egl child, so it's ignored here.
            self.process = GenesisProcess(spec, num_envs, **worker_kwargs)
            self.process.start()
        else:
            # Debug/teleop path: Genesis in-process so the 3D viewer can open.
            # Import locally so the subprocess path never imports genesis in-parent.
            from gentle_manip.envs.genesis_worker import GenesisWorker
            self.process = GenesisWorker(spec, num_envs, show_viewer=show_viewer, **worker_kwargs)

        # Accumulated command targets (seeded on reset).
        self._target_pos = np.zeros((num_envs, 3), dtype=np.float64)
        self._target_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_envs, 1)).astype(np.float64)
        self._target_gripper = np.full(num_envs, self._gripper_max, dtype=np.float64)
        self._last_state: Optional[dict] = None

    # ── Backend protocol ──────────────────────────────────────────────────────
    def reset(self, object_dxy=None, home_offset=None, object_euler=None, **kwargs) -> RawObs:
        # Full DR: every N resets, re-randomize the WHOLE scene (material+size+shape) by relaunching
        # the genesis child instead of a plain reset. Guarded so randomize_scene's own reset() (and
        # explicit object_dxy overrides, e.g. eval/demo-replay) don't trip it.
        if (self._scene_dr_every > 0 and not self._in_scene_dr and object_dxy is None):
            self._reset_count += 1
            if self._reset_count % self._scene_dr_every == 0:
                return self.randomize_scene()

        # Explicit object_dxy (num_envs, 2) places the object at a chosen offset from
        # its default pose (e.g. to match a recorded demo's cube); otherwise per-reset DR.
        if object_dxy is not None:
            object_dxy = np.asarray(object_dxy, dtype=np.float32).reshape(self.num_envs, 2)
        else:
            object_dxy = self._dr.sample_object_dxy(self._rng, self.num_envs)

        # Per-env reset home-pose jitter (sim-only DR), unless explicitly provided.
        if home_offset is not None:
            home_offset = np.asarray(home_offset, dtype=np.float32).reshape(self.num_envs, 3)
        else:
            home_offset = self._dr.sample_home_offset(self._rng, self.num_envs)

        # Per-env object spawn orientation (yaw/pitch/roll) DR, unless explicitly provided.
        if object_euler is not None:
            object_euler = np.asarray(object_euler, dtype=np.float32).reshape(self.num_envs, 3)
        else:
            object_euler = self._dr.sample_object_euler(self._rng, self.num_envs)

        # Record the per-env randomization actually applied this reset (for eval audit/CSV).
        self._last_reset_dr = {"object_dxy": object_dxy, "home_offset": home_offset,
                               "object_euler": object_euler}

        state = self.process.reset(object_dxy, home_offset, object_euler)
        self._last_state = state
        # Seed targets from the actual reset pose so the first deltas are relative
        # to where the arm really is.
        self._target_pos = state["ee_pos"].astype(np.float64).copy()
        self._target_quat = state["ee_quat"].astype(np.float64).copy()
        self._target_gripper = state["gripper_width"].astype(np.float64).copy()
        # Cache scene DR params ([scale, bend_deg]) for the episode; used by privileged obs.
        sp = self.scene_params()
        self._episode_dr_vec = np.array(
            [sp.get("scale", 1.0), sp.get("bend_deg", 0.0)], dtype=np.float32
        )
        return self._build_raw_obs(state)

    def material_params(self) -> dict:
        """The resolved material of the (first) object — E/nu/rho/yield, ObjectEntry overrides
        applied over the registry default (same resolution as SceneBuilder). Constant per
        scene; recorded per eval episode for the audit trail."""
        from gentle_manip.assets.registry import get_object_def
        e = self._spec.objects[0]
        m = get_object_def(e.name).material
        return {"E": float(e.youngs_modulus if e.youngs_modulus is not None else m.youngs_modulus),
                "nu": float(e.poisson_ratio if e.poisson_ratio is not None else m.poisson_ratio),
                "rho": float(e.density if e.density is not None else m.density),
                "yield": float(m.von_mises_yield_stress)}

    def render_rgb(self, all_envs: bool = False):
        """RGB frame(s) or None — behaviour clips / eval video. all_envs=False -> env-0 (H,W,3);
        all_envs=True -> all envs (N,H,W,3) for per-trajectory video. Works for both the subprocess
        (GenesisProcess.render) and in-process (GenesisWorker.render_rgb) backends, so video
        survives the relaunch-based scene DR (which needs the subprocess)."""
        fn = getattr(self.process, "render", None) or getattr(self.process, "render_rgb", None)
        return fn(all_envs) if fn is not None else None

    def scene_params(self) -> dict:
        """The object SIZE + SHAPE DR actually applied to the current scene ('scale',
        'bend_deg', 'twist_deg', 'taper', 'rbf' for whatever is randomized) — {} if none.
        Constant per scene; recorded per eval episode alongside material."""
        return dict(self._applied_scene)

    def _apply_scene_dr(self, spec: SceneSpec):
        """Sample FULL scene-level DR — object CATEGORY (cross-category, if configured) +
        material (E/ν/ρ + coupling friction) + object size/shape — and bake it into `spec`;
        return (new_spec, coup_friction_or_None). Works from the REGISTRY nominal mesh +
        NOMINAL scale + ABSOLUTE material values, so it's idempotent — always call it on
        self._nominal_spec so rebuilds don't chain. Records the applied category/size/shape/
        coup in self._applied_scene (material is auditable via material_params).

        Category sampling (DRConfig.object_category_pool) happens FIRST, before material/
        shape are sampled, because those two calls take the resolved category's ObjectDef as
        a fallback for any field DRConfig itself leaves unset (its own per-category
        material_dr_mult/shape_dr_ranges — see dr_config.py). Swapping `o.name` (+
        `object_type`, from the new category's registry entry) is enough for scene_builder to
        re-resolve mesh/material from the NEW category via its existing None-fallback
        (ObjectEntry.mesh_path/youngs_modulus/etc. are None on the nominal entry, so they
        already fall back to the registry lookup by name) — mesh_path is explicitly reset to
        None on a category switch so a stale override never survives onto a different object.
        """
        import dataclasses
        import numpy as _np
        from gentle_manip.assets.registry import get_object_def

        self._applied_scene = {}
        if not spec.objects:
            return spec, None

        o = spec.objects[0]
        category_name = self._dr.sample_category(self._rng)
        if category_name is not None:
            cat_def = get_object_def(category_name)
            o = dataclasses.replace(o, name=category_name, object_type=cat_def.object_type,
                                    mesh_path=None)
        else:
            cat_def = get_object_def(o.name)

        mat = self._dr.sample_scene(self._rng, category=cat_def)         # E/nu/rho/yield/coup
        shp = self._dr.sample_shape_scale(self._rng, category=cat_def)   # scale + bend/twist/taper/rbf
        has_sim_override = (cat_def.sim_substeps_override is not None
                            or cat_def.mpm_grid_density_override is not None)

        if not mat and not shp and category_name is None and not has_sim_override:
            return spec, None   # nothing sampled/overridden at all (no category pool, no absolute DR)

        updates, applied = {}, {}
        if category_name is not None:
            applied["category"] = category_name
        for key, field_name in (("E", "youngs_modulus"), ("nu", "poisson_ratio"), ("rho", "density")):
            if key in mat:
                updates[field_name] = float(mat[key])
        if "scale" in shp:
            updates["scale"] = float(self._nominal_scale * shp["scale"])
            applied["scale"] = updates["scale"]
        shape = {k: shp[k] for k in ("bend", "twist", "taper", "rbf", "axis_scale", "axis_scale_ax")
                 if k in shp}
        nominal_mesh = cat_def.mesh_path                          # from the resolved category, no chaining
        if shape and nominal_mesh is not None:                   # mesh object only; boxes -> size only
            import tempfile
            from gentle_manip.assets import mesh_deform
            if self._deform_dir is None:
                self._deform_dir = tempfile.mkdtemp(prefix="gm_deform_")
            updates["mesh_path"] = str(mesh_deform.save_deformed(nominal_mesh, shape, self._rng, self._deform_dir))
            for k, deg in (("bend", "bend_deg"), ("twist", "twist_deg")):
                if k in shape:
                    applied[deg] = float(_np.rad2deg(shape[k]))
            for k in ("taper", "rbf", "axis_scale"):
                if k in shape:
                    applied[k] = float(shape[k])
            if "axis_scale_ax" in shape:
                applied["axis"] = "xyz"[int(shape["axis_scale_ax"])]
        coup = mat.get("coup_friction")
        if coup is not None:
            applied["coup_friction"] = float(coup)
        self._applied_scene = applied
        objects = list(spec.objects)
        objects[0] = dataclasses.replace(o, **updates) if updates else o
        scene_updates = {}
        if cat_def.sim_substeps_override is not None:
            scene_updates["sim_substeps"] = cat_def.sim_substeps_override
            applied["sim_substeps"] = cat_def.sim_substeps_override
        if cat_def.mpm_grid_density_override is not None:
            scene_updates["mpm_grid_density"] = cat_def.mpm_grid_density_override
            applied["mpm_grid_density"] = cat_def.mpm_grid_density_override
        spec = dataclasses.replace(spec, objects=objects, **scene_updates)
        return spec, coup

    def randomize_scene(self, tries: int = 6) -> RawObs:
        """Re-randomize the WHOLE scene (material + coupling friction + object size/shape) and
        REBUILD via GenesisProcess.restart (kill + respawn the genesis child — a single sim at a
        time). Deterministic if the caller reseeds self._rng first (eval). An unlucky combo can
        make the settle blow up (solver NaN); we resample + rebuild up to `tries` times so full DR
        stays robust. Returns the reset obs; a plain reset() if no scene DR is configured."""
        if not self._dr.has_scene_dr():
            return self.reset()
        if not hasattr(self.process, "restart"):
            raise RuntimeError("scene DR needs the subprocess backend (use_subprocess=True)")
        self._in_scene_dr = True
        try:
            for attempt in range(tries):
                spec, coup = self._apply_scene_dr(self._nominal_spec)   # resample from nominal
                self._spec = spec
                self.process.restart(spec, coup_friction=coup)
                try:
                    return self.reset()                                # settle (guard set)
                except RuntimeError as e:                              # unstable scene -> resample
                    if attempt == tries - 1:
                        raise
                    print(f"[dr] unstable rebuilt scene {self._applied_scene} (attempt "
                          f"{attempt + 1}/{tries}), resampling: {str(e).splitlines()[-1][:80]}", flush=True)
        finally:
            self._in_scene_dr = False

    def set_auto_scene_dr(self, enabled: bool) -> None:
        """Enable/disable the periodic (every-N-resets) AUTO scene-DR relaunch WITHOUT touching
        the RNG stream. Used to freeze auto scene DR during a fixed-seed eval: the eval harness
        drives its own deterministic per-group randomize_scene(), so the training server's auto
        relaunch must not also fire mid-eval (it would rebuild geometry + consume RNG, breaking the
        eval's apples-to-apple determinism). Saves the configured cadence on the first disable and
        restores it on re-enable. This only gates the counter branch in reset(); it changes no
        _rng draw, so a normal reset is byte-identical whether auto DR is on or off."""
        if not enabled:
            if not hasattr(self, "_scene_dr_every_saved"):
                self._scene_dr_every_saved = self._scene_dr_every
            self._scene_dr_every = 0
        else:
            if hasattr(self, "_scene_dr_every_saved"):
                self._scene_dr_every = self._scene_dr_every_saved
                del self._scene_dr_every_saved

    def step(self, scaled_action: np.ndarray) -> RawObs:
        action = np.asarray(scaled_action, dtype=np.float64).reshape(self.num_envs, -1)

        if action.shape[-1] == 8:
            # ActionPipeline absolute mode: pos(3) + quat_wxyz(4) + gripper(1), ready
            # to set directly (no accumulation). Still clip to the workspace box for
            # safety even though ActionPipeline already mapped into pos_min/pos_max.
            pos, quat, grip = action[:, :3], action[:, 3:7], action[:, 7]
            self._target_pos = np.clip(pos, cfg.EE_BOUNDS_MIN, cfg.EE_BOUNDS_MAX)
            neg = quat[:, 0] < 0
            quat = quat.copy()
            quat[neg] = -quat[neg]                                  # keep w >= 0
            self._target_quat = quat
            self._target_gripper = np.clip(grip, 0.0, self._gripper_max)
        else:
            # ActionPipeline delta mode (default): dpos(3) + drot(3) + dgripper(1),
            # accumulated onto the running target.
            dpos, drot, dgrip = action[:, :3], action[:, 3:6], action[:, 6]

            # Translation: accumulate then clip to the workspace box (per env).
            self._target_pos = np.clip(
                self._target_pos + dpos, cfg.EE_BOUNDS_MIN, cfg.EE_BOUNDS_MAX
            )

            # Orientation: compose the delta rotation (base-frame premultiply), batched.
            q = self._target_quat
            R_cur = Rotation.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]]))
            xyzw = (Rotation.from_rotvec(drot) * R_cur).as_quat()       # (B, 4) xyzw
            wxyz = np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])
            neg = wxyz[:, 0] < 0
            wxyz[neg] = -wxyz[neg]                                      # keep w >= 0
            self._target_quat = wxyz

            # Gripper: accumulate then clip to [0, open width].
            self._target_gripper = np.clip(self._target_gripper + dgrip, 0.0, self._gripper_max)

        state = self.process.step(
            self._target_pos.astype(np.float32),
            self._target_quat.astype(np.float32),
            self._target_gripper.astype(np.float32),
        )
        self._last_state = state
        return self._build_raw_obs(state)

    def get_sim_feedback(self) -> Optional[SimFeedback]:
        if self._last_state is None:
            return None
        s = self._last_state
        extra = {} if s["von_mises_stress"] is None else {"von_mises_stress": s["von_mises_stress"]}
        if s.get("object_quat") is not None:
            extra["object_quat"] = s["object_quat"]           # (N, 4) wxyz, rigid only
        if s.get("contact_force") is not None:
            extra["contact_force"] = s["contact_force"]       # (N,) Newtons, rigid only
        if hasattr(self, "_episode_dr_vec"):
            extra["object_dr_vec"] = self._episode_dr_vec     # (2,) [scale, bend_deg], episode const
        return SimFeedback(
            ee_pos=s["ee_pos"],
            gripper_width=s["gripper_width"],
            object_center=s["object_center"],
            extra=extra,
        )

    def close(self) -> None:
        self.process.stop()
        if self._deform_dir is not None:            # remove temp shape-DR meshes
            import shutil
            shutil.rmtree(self._deform_dir, ignore_errors=True)
            self._deform_dir = None

    # ── internal ──────────────────────────────────────────────────────────────
    def _build_raw_obs(self, state: dict) -> RawObs:
        raw = RawObs(
            ee_pos=state["ee_pos"],
            ee_quat=state["ee_quat"],
            gripper_width=state["gripper_width"],
            joint_pos=state["joint_pos"],
            joint_vel=state["joint_vel"],
            depth_images=state["depth_images"],
            rgb_images={},                              # MVP: depth/point-cloud only
            camera_intrinsics=state["camera_intrinsics"],
            camera_extrinsics=state["camera_extrinsics"],
            tactile_images={},                          # sim has no tactile
        )
        raw.validate()
        return raw
