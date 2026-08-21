"""Evaluate the SCRIPTED GRASP-SYNTHESIS policies (v2 SDF vs v3 FEM gentleness) under the shared
canonical eval harness — reported IDENTICALLY to the learned policies (same EvalSpec, fixed scenario
sequence, summary.json + episodes.csv with per-episode DR params + stress, env-cfg snapshot,
per-trajectory video). Algorithm name: "scripted_policy" (algo_variant: grasp_synth_{sdf,fem}).

The grasp is synthesized PER ENV at reset from the true object pose (priv_object_pos + priv_object_rot6d)
and the sim's ACTUAL (shape+size DR) mesh — the SAME 7-DOF TCP grasp `[tx,ty,tz,roll,pitch,yaw,width]`
the demo collectors produce — then executed by the shared approach->grasp->firm->lift->hold FSM
(identical for both synthesizers, incl. the soft stress-based firm robustness). So the ONLY variable
between the two runs is the SYNTHESIS: geometric SDF cost (v2) vs FEM gentleness metric (v3). Because
scenarios are seed-driven, this stays apples-to-apple with the learned-policy eval AND with each other.

Needs a SOFT teacher-view server that carries object ORIENTATION + stress and full DR ranges (so the
harness can drive per-group scene DR = size/shape/material). Run each synth SEQUENTIALLY, SAME seed:

    uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server \\
        --experiment single_lift_mushroom_soft_grasp_eval --view teacher \\
        --num-envs 5 --render-rgb --subprocess --port 5583
    uv run --project envs/sim python -m gentle_manip.scripts.eval_grasp_synth \\
        --synth fem --port 5583 --n-episodes 200 --seed 0 --scene-group-size 2
    #  then kill+relaunch the server (fresh geometry RNG) and rerun with --synth sdf, SAME seed.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "grasp_synthesis"))

import collect_demos_synth_v3 as gsv3          # noqa: E402 — SDF synth helpers + firm constants
from grasp_traj import (GraspTrajectory, PhaseSchedule, SCHEDULE_V3,  # noqa: E402
                        SCHEDULE_V4)
from smgrasp import finger_grasp as fg          # noqa: E402


# ── Grasp-objective PROFILES ──────────────────────────────────────────────────
# THE bug this fixes: the benchmark used to build grasp_kw WITHOUT w_align / diversity / jitter /
# pitch-seed, so `plan_finger_grasp` fell back to its strict defaults — while the v3 COLLECTOR
# deliberately runs w_align=2000 (15x weaker) + tilt seeding to broaden the demo distribution. The
# benchmark was therefore never measuring the configuration that actually generates the datasets,
# which is why it reported 98-100% success on demos containing stem/pinch/side grasps.
#
# `strict` reproduces the historical benchmark EXACTLY (pass nothing -> metric defaults), so the
# existing 200-episode runs stay comparable. `collector_v3` is the honest v3 baseline.
GRASP_PROFILES = {
    # historical benchmark: strict flush-alignment metric, no diversity, pitch seeds at 0
    "strict": {},
    # what collect_demos_synth_v3.py ACTUALLY runs (its argparse defaults, :770-791)
    "collector_v3": dict(w_align=2000.0, diversity_tol=0.3, jitter_deg=20.0,
                         jitter_pos=0.003, pitch_seed_deg=25.0),
    # v4: keep the collector's diversity, but re-enable the anti-pinch peak term and add the
    # geometry priors. Weights are tuned in Iteration 3; these are the starting points.
    "v4": dict(w_align=2000.0, diversity_tol=0.3, jitter_deg=20.0, jitter_pos=0.003,
               pitch_seed_deg=25.0, w_peak=None, w_com=0.0, w_tilt=0.0, w_occ=0.0, area_min=0.0),
    # v4fix: the OPERATING-POINT correction. The executor closes 4.5mm tighter than the width the
    # objective scores (2.5mm base squeeze + 2mm firm), which is ~10x more stress — measured 54.8kPa
    # executed vs 5.4kPa scored, i.e. 1.37x the mushroom's yield on 100% of episodes.
    # execute_offset scores each candidate at the width actually commanded.
    # area_min is NOT optional here: correcting the operating point alone makes the optimizer grasp
    # something too thin to compress (contact area 66 -> 12 mm2, align 0.98 -> 0.19), and w_peak
    # alone does not stop it. Offline, all three together give 21.0kPa executed (0.53x yield) with
    # 41mm2 of flush contact — a 2.6x reduction. THIS PROFILE IS THE ONE TO BENCHMARK NEXT.
    "v4fix": dict(w_align=2000.0, diversity_tol=0.3, jitter_deg=20.0, jitter_pos=0.003,
                  pitch_seed_deg=25.0, w_peak=0.3, area_min=3e-5,
                  execute_offset=0.0045),
}
from gentle_manip.envs.rpc import SimEnvClient   # noqa: E402
from gentle_manip.evaluation import EvalSpec, run_eval  # noqa: E402
from gentle_manip.evaluation.sim_eval_venv import SimEvalVenv  # noqa: E402
from gentle_manip.experiment import Experiment   # noqa: E402


def _rot6d_to_quat_wxyz(r6: np.ndarray) -> np.ndarray:
    """(N,6) 6D rotation (first two rotation-matrix columns, Zhou 2019) -> (N,4) wxyz quat."""
    a1, a2 = r6[:, :3], r6[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2p = a2 - (b1 * a2).sum(1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=-1)                     # columns = basis
    xyzw = Rot.from_matrix(mat).as_quat()
    return np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)


class GraspSynthPolicy:
    """Batched scripted grasp-synthesis expert plugged into the canonical harness.

    reset(): (re)synthesize on the next act — needs the reset obs (home pose + object pose) + the
             current DR'd mesh (from the venv), which are only available once the episode starts.
    act(obs): on the first call of an episode, synthesize best_x PER ENV, then step the shared FSM;
              subsequent calls step the FSM. Absolute-action output (10-dim). The soft firm check
              reads priv_stress (top10, /yield -> Pa) at the grasp->firm boundary.
    """

    def __init__(self, num_envs, action_config, venv, *, synth, maxfevals, yield_pa,
                 object_name, grasp_kw, table_z, seed, extra_close=0.0,
                 objective=None, schedule=SCHEDULE_V3, lift_height=None,
                 standoff=0.05, use_minjerk=False, preshape_factor=0.0):
        self.num_envs = int(num_envs)
        self.action_config = action_config
        self.venv = venv
        self.synth = synth                       # "sdf" | "fem"
        self.maxfevals = int(maxfevals)
        self.extra_close = float(extra_close)    # squeeze TIGHTER than synth width (m); mirrors v3 collector
        self.yield_pa = float(yield_pa)
        self.object_name = object_name
        self.grasp_kw = dict(grasp_kw)           # E, density, mu, accel, n_starts, voxel_div, target_tets, gpu
        # The RESOLVED grasp objective (profile + CLI overrides) forwarded to plan_finger_grasp.
        # Recorded into summary.json so every run self-documents which objective produced it.
        self.objective = dict(objective or {})
        self.schedule = schedule
        self.lift_height = float(gsv3.LIFT_HEIGHT if lift_height is None else lift_height)
        self.standoff = float(standoff)
        self.use_minjerk = bool(use_minjerk)
        self.preshape_factor = float(preshape_factor)
        self._audit = [{} for _ in range(self.num_envs)]      # per-env grasp-quality audit
        self._obj_radius_m = 0.02                             # bbox radius; refined once the FEM builds
        self._r_obj = 0.03                                    # object-point radius (bbox + 5mm)
        self._occ_gt = [{} for _ in range(self.num_envs)]     # rendered-cloud occlusion
        self.traj = None
        self.table_z = float(table_z)
        self._seed_rng = np.random.default_rng(seed)
        self._synthed = False
        self.phase_idx = None
        # FEM cache (rebuilt only when the mesh changes = a scene-DR group boundary)
        self._fem_mesh = None
        self._fem_obj = self._fem_pad = None
        # SDF pool + finger surface samples (reused across episodes)
        self._pool = None
        if synth == "sdf":
            self._pool = ProcessPoolExecutor(max_workers=self.num_envs)
            self._left_pts = gsv3.sample_finger_surface(gsv3.LEFT_FINGER, n=300)
            self._right_pts = gsv3.sample_finger_surface(gsv3.RIGHT_FINGER, n=300)
        self._log_dir = Path(tempfile.mkdtemp(prefix="gm_eval_cmaes_"))

    # ── plumbing ──────────────────────────────────────────────────────────────
    def reset(self):
        self._synthed = False
        # Clear the audit too: it is overwritten per env during synthesis, so without this an env
        # whose synthesis failed (or the SDF arm, which never fills it) would report the PREVIOUS
        # episode's grasp as if it were this one's.
        self._audit = [{} for _ in range(self.num_envs)]
        self._occ_gt = [{} for _ in range(self.num_envs)]

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def _current_mesh(self) -> str:
        """The mesh Genesis is ACTUALLY simulating this scene group — the deformed file exposed via
        scene_params (scale baked in), or the nominal registry mesh (× scale) if no scene DR."""
        sc = (self.venv.scenario_params() or {}).get("scene") or {}
        mp = sc.get("mesh_path")
        if mp and Path(mp).exists():
            return str(mp)
        from gentle_manip.assets.registry import get_object_def
        nom = get_object_def(self.object_name).mesh_path
        scale = float(sc.get("scale", 1.0) or 1.0)
        if abs(scale - 1.0) < 1e-6:
            return str(nom)
        import trimesh
        m = trimesh.load(str(nom), process=False, force="mesh")
        m.apply_scale(scale)
        dst = self._log_dir / f"nom_scaled_{scale:.4f}.obj"
        m.export(str(dst))
        return str(dst)

    def _stress_pa(self, obs) -> np.ndarray:
        """(N,) top10 von-Mises in Pa from priv_stress ([mean, top10] / yield)."""
        return np.asarray(obs["priv_stress"], np.float64)[:, 1] * self.yield_pa

    # ── grasp-quality audit ─────────────────────────────────────────────────────
    # The synthesizer already knows everything needed to COUNT the three reported defects; it was
    # simply thrown away. `episode_metrics()` hands it to the harness so stem/pinch/side grasps
    # become numbers in episodes.csv instead of something you have to eyeball in a video.
    # PROVISIONAL thresholds — calibrate against the Iteration-1 distribution before drawing
    # conclusions from the RATES (the underlying grasp_min_pad_mm2 / grasp_com_lever_mm columns are
    # the ground truth and are always reported raw). First measurement on 10 collector_v3 episodes:
    # worst-pad areas spanned ~8-40 mm2 (mean 25), so 20 mm2 sits mid-distribution and flags ~half
    # — informative as a relative signal, but not yet a calibrated "this is a pinch" boundary.
    STEM_LEVER_FRAC = 0.25      # lever > this * object bbox radius => grasping away from the mass
    PINCH_AREA_MM2 = 20.0       # worst-pad contact area below this => a pinch, not a grasp

    def _audit_row(self, r, obj_pos) -> dict:
        x = r.get("x")
        area_mm2 = (float(r["min_pad_area"]) * 1e6) if r.get("min_pad_area") is not None else None
        lever_mm = (float(r["com_lever"]) * 1e3) if r.get("com_lever") is not None else None
        row = {
            "grasp_tilt_deg": r.get("tilt_deg"),
            "grasp_min_pad_mm2": area_mm2,
            "grasp_com_lever_mm": lever_mm,
            "grasp_width_mm": (float(x[6]) * 1e3) if x is not None else None,
            "grasp_occ_pred": r.get("occ"),
            "grasp_stress_pred_pa": r.get("stress_top10"),
            "grasp_score": r.get("score"),
            "grasp_status": r.get("status"),
        }
        thr = self.STEM_LEVER_FRAC * self._obj_radius_m * 1e3
        row["stem_grasp"] = int(lever_mm is not None and lever_mm > thr)
        row["pinch_grasp"] = int(area_mm2 is not None and area_mm2 < self.PINCH_AREA_MM2)
        return row

    # ── ground-truth occlusion from the RENDERED cloud ──────────────────────────
    # `grasp_occ_pred` is a geometric prediction; this is what the camera actually lost. Only
    # active when the eval obs carries a point cloud (the *_grasp_eval_pcd experiment).
    def _count_object_points(self, obs, j: int, tcp_pos, tcp_quat, grip) -> int:
        """Object points visible to the camera for env j, this step.

        No segmentation exists in the pipeline, so object-ness is recovered in two steps:
          1. keep points within `r_obj` of the TRUE object centre (priv_object_pos), extending the
             precedent in examples/fov_check_always_fail.py — privileged, so it tracks the object
             through the lift rather than assuming it stays put;
          2. SUBTRACT the gripper analytically: reject points within ~5mm of the finger surface at
             the COMMANDED TCP pose. The policy knows that pose exactly, so this costs nothing and
             removes the one confounder the radius test cannot (fingers closing INSIDE the radius
             would otherwise read as extra "object" points and mask the very occlusion we measure).
        """
        pc = np.asarray(obs["point_cloud"][j], np.float64)
        pc = pc[np.any(pc != 0.0, axis=1)]                 # drop the zero padding
        if pc.size == 0:
            return 0
        oc = np.asarray(obs["priv_object_pos"][j], np.float64)
        near = pc[np.linalg.norm(pc - oc, axis=1) < self._r_obj]
        if near.size == 0 or self._fem_pad is None:
            return int(len(near))
        q = np.asarray(tcp_quat, float)
        x = np.concatenate([np.asarray(tcp_pos, float),
                            Rot.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz"),
                            [float(grip)]])
        Lw, Rw = fg.finger_world_pts(x, self._fem_pad)
        fingers = np.vstack([Lw, Rw])[::4]
        d = np.linalg.norm(near[:, None, :] - fingers[None, :, :], axis=2).min(axis=1)
        return int((d > 0.005).sum())

    def episode_metrics(self):
        """Optional harness hook: per-env grasp-quality columns for this episode."""
        rows = [dict(a) for a in self._audit]
        for j, r in enumerate(rows):
            b = self._occ_gt[j]
            if b.get("baseline"):
                r["occ_pcd_baseline_pts"] = b["baseline"]
                for k in ("grasp", "lift"):
                    if b.get(k) is not None:
                        r[f"occ_pcd_{k}"] = max(0.0, 1.0 - b[k] / max(b["baseline"], 1))
        return rows

    # ── synthesis ───────────────────────────────────────────────────────────────
    def _synthesize(self, obs):
        N = self.num_envs
        home_pos = np.asarray(obs["ee_pos"], np.float32)
        home_quat = np.asarray(obs["ee_quat"], np.float32)
        obj_pos = np.asarray(obs["priv_object_pos"], np.float64)
        obj_quat = _rot6d_to_quat_wxyz(np.asarray(obs["priv_object_rot6d"], np.float64))
        mesh = self._current_mesh()
        gk = self.grasp_kw
        best_x = []
        if self.synth == "fem":
            if mesh != self._fem_mesh:
                self._fem_obj, self._fem_pad, meta = fg.build_grasp_fem(
                    mesh, voxel_div=gk["voxel_div"], target_tets=gk["target_tets"], use_gpu=gk["gpu"])
                self._fem_mesh = mesh
                # object bbox radius -> scales the stem-grasp lever threshold to THIS object/DR scale
                ext = self._fem_obj.verts.max(0) - self._fem_obj.verts.min(0)
                self._obj_radius_m = float(np.linalg.norm(ext[:2]) / 2.0)
                self._r_obj = float(np.linalg.norm(ext) / 2.0) + 0.005
                print(f"  [{self.synth}] FEM {meta['tets']} tets ndof={meta['ndof']} gpu={meta['gpu']}"
                      f"  mesh={Path(mesh).name}  r_xy={self._obj_radius_m*1e3:.1f}mm", flush=True)
            for i in range(N):
                seed = int(self._seed_rng.integers(1, 2**31 - 1))
                r = fg.synthesize_grasp(self._fem_obj, self._fem_pad, obj_pos[i], obj_quat[i],
                                        E=gk["E"], density=gk["density"], mu=gk["mu"],
                                        table_z=self.table_z, maxfevals=self.maxfevals,
                                        n_starts=gk["n_starts"], seed=seed, accel=gk["accel"],
                                        **self.objective)
                best_x.append(r["x"])
                self._audit[i] = self._audit_row(r, obj_pos[i])
        else:  # sdf (v2 geometric grasp cost)
            payloads = []
            for i in range(N):
                lb, ub = gsv3._synth_bounds(obj_pos[i])
                seed = int(self._seed_rng.integers(1, 2**31 - 1))
                payloads.append((mesh, obj_pos[i], obj_quat[i], self._left_pts, self._right_pts,
                                 self.maxfevals, lb, ub, str(self._log_dir), seed))
            futs = [self._pool.submit(gsv3._synth_worker, p) for p in payloads]
            best_x = [f.result()[0] for f in futs]
        self._setup_fsm(np.asarray(best_x), home_pos, home_quat)
        self._synthed = True

    def _setup_fsm(self, all_best_x, home_pos, home_quat):
        """Build the shared trajectory engine. Replaces the hand-rolled copy of the collector's
        `_env_target` that used to live here — the duplication (plus reading gsv3's mutable PHASES
        globals) is exactly how the benchmark drifted away from the collector's real behaviour."""
        self.traj = GraspTrajectory(
            self.schedule, all_best_x, home_pos, home_quat,
            lift_height=self.lift_height, extra_close=self.extra_close,
            firm_close=gsv3.FIRM_EXTRA_CLOSE_M, standoff=self.standoff,
            use_minjerk=self.use_minjerk, preshape_factor=self.preshape_factor)
        N = self.num_envs
        self.phase_idx = np.zeros(N, np.int64)
        self.phase_step = np.zeros(N, np.int64)
        self.rest_stress = None

    def act(self, obs):
        if not self._synthed:
            self._synthesize(obs)
        if self.rest_stress is None:
            self.rest_stress = self._stress_pa(obs)          # settled-rest baseline (top10 Pa)
        N = self.num_envs
        cur_pos = np.zeros((N, 3), np.float32)
        cur_quat = np.zeros((N, 4), np.float32)
        cur_grip = np.zeros(N, np.float32)
        for i in range(N):
            if self.phase_idx[i] < self.schedule.n_phases:
                p, q, g = self.traj.target(i, int(self.phase_idx[i]), int(self.phase_step[i]))
            else:                                             # FSM done -> freeze at lift/hold target
                p, q, g = self.traj.frozen_target(i)
            cur_pos[i], cur_quat[i], cur_grip[i] = p, q, g
        # Ground-truth occlusion, sampled AFTER the targets so the finger geometry subtracted is the
        # pose actually commanded THIS step. Using the grasp pose throughout would be wrong during
        # the lift, when the gripper has moved away from it.
        if isinstance(obs, dict) and "point_cloud" in obs and "priv_object_pos" in obs:
            gi, li = self.schedule.index("grasp"), self.schedule.index("lift")
            for j in range(N):
                ph, b = int(self.phase_idx[j]), self._occ_gt[j]
                if not b.get("baseline"):                 # step 0: arm at home, nothing occluded
                    b["baseline"] = self._count_object_points(obs, j, cur_pos[j], cur_quat[j],
                                                             cur_grip[j])
                elif gi >= 0 and ph == gi and b.get("grasp") is None:
                    b["grasp"] = self._count_object_points(obs, j, cur_pos[j], cur_quat[j],
                                                           cur_grip[j])
                elif li >= 0 and ph == li and b.get("lift") is None:
                    b["lift"] = self._count_object_points(obs, j, cur_pos[j], cur_quat[j],
                                                          cur_grip[j])
        action = gsv3._invert_actions_absolute(cur_pos, cur_quat, cur_grip, self.action_config)
        self._advance(obs)
        return action.astype(np.float32)

    def _advance(self, obs):
        N = self.num_envs
        n_phases = self.schedule.n_phases
        active = self.phase_idx < n_phases
        self.phase_step[active] += 1
        durations = np.array([self.schedule.duration(min(int(p), n_phases - 1))
                              for p in self.phase_idx])
        rolled = active & (self.phase_step >= durations)
        grasp_idx = self.schedule.index("grasp")
        leaving_grasp = rolled & (self.phase_idx == grasp_idx)
        if np.any(leaving_grasp):                            # soft: ALWAYS firm base, weak firms more
            cur = self._stress_pa(obs)
            for i in np.where(leaving_grasp)[0]:
                rise = float(cur[i] - self.rest_stress[i])
                if rise < gsv3.FIRM_STRESS_THRESH_PA:        # weak grasp -> extra close
                    self.traj.set_firm_close(i, gsv3.FIRM_EXTRA_CLOSE_M + gsv3.FIRM_WEAK_EXTRA_CLOSE_M)
                # else: base firm close (never skip)
        self.phase_idx[rolled] += 1
        self.phase_step[rolled] = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", choices=["sdf", "fem"], required=True,
                    help="grasp synthesis: sdf = v2 geometric cost, fem = v3 gentleness metric")
    ap.add_argument("--experiment", default="single_lift_mushroom_soft_grasp_eval")
    ap.add_argument("--port", type=int, default=5583)
    ap.add_argument("--num-envs", type=int, default=5, help="MUST match the server")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="SAME across synths -> identical scenarios")
    ap.add_argument("--max-steps", type=int, default=300, help="policy (sim) steps per episode")
    ap.add_argument("--scene-group-size", type=int, default=2,
                    help="rebuild geometry every K batches (matches the learned soft 200-ep eval)")
    ap.add_argument("--record-batches", type=int, default=None,
                    help="None (default) = ALL batches (one clip per episode); 0 disables video")
    ap.add_argument("--maxfevals", type=int, default=1000, help="CMA budget per env (both synths)")
    # grasp-synthesis params (defaults match the v3 collection)
    ap.add_argument("--grasp-E", type=float, default=3e5)
    ap.add_argument("--grasp-density", type=float, default=1000.0)
    ap.add_argument("--grasp-mu", type=float, default=0.7)
    ap.add_argument("--grasp-accel", type=float, default=9.81)
    ap.add_argument("--grasp-n-starts", type=int, default=6)
    ap.add_argument("--grasp-voxel-div", type=int, default=14)
    ap.add_argument("--grasp-target-tets", type=int, default=1500)
    ap.add_argument("--table-z", type=float, default=0.0)
    ap.add_argument("--grasp-cpu", action="store_true", help="FEM on CPU (default GPU)")
    ap.add_argument("--grasp-extra-close", type=float, default=0.0,
                    help="squeeze TIGHTER than the synthesized width by this many meters (e.g. 0.005 = "
                         "5mm), for EVERY grasp — mirrors collect_demos_synth_v3.py. 0 = native synth width.")
    # ── grasp OBJECTIVE (the fix for the benchmark-vs-collector mismatch) ──────
    ap.add_argument("--grasp-profile", choices=sorted(GRASP_PROFILES), default="strict",
                    help="which grasp objective to evaluate. 'strict' (default) = the historical "
                         "benchmark, bit-identical. 'collector_v3' = what collect_demos_synth_v3.py "
                         "ACTUALLY runs (w_align 2000 + tilt seeding + diversity) — the honest v3 "
                         "baseline. 'v4' = collector diversity + the re-enabled peak term and the new "
                         "geometry priors. Individual --grasp-* flags below override the profile.")
    ap.add_argument("--grasp-align", type=float, default=None, help="override w_align")
    ap.add_argument("--grasp-peak", type=float, default=None, help="override w_peak (0.3 = metric default)")
    ap.add_argument("--grasp-com", type=float, default=None, help="override w_com (COM lever arm)")
    ap.add_argument("--grasp-tilt", type=float, default=None, help="override w_tilt (verticality prior)")
    ap.add_argument("--grasp-occ", type=float, default=None, help="override w_occ (camera occlusion)")
    ap.add_argument("--grasp-area-min", type=float, default=None,
                    help="override area_min (m^2 floor on the worst pad's contact area)")
    ap.add_argument("--grasp-execute-offset", type=float, default=None,
                    help="score each candidate at width MINUS this (metres) — the extra squeeze the "
                         "executor applies (base 2.5mm + firm 2mm = 0.0045). 0 scores an operating "
                         "point the robot never visits; see the v4fix profile.")
    ap.add_argument("--grasp-roll-max-deg", type=float, default=None,
                    help="half-width of the roll search band about top-down. 90 (default) admits a "
                         "fully horizontal tool axis, i.e. pure side grasps; try 15-30 to exclude them.")
    ap.add_argument("--grasp-diversity-tol", type=float, default=None)
    ap.add_argument("--grasp-jitter-deg", type=float, default=None)
    ap.add_argument("--grasp-jitter-pos", type=float, default=None)
    ap.add_argument("--grasp-pitch-seed-deg", type=float, default=None)
    # ── trajectory ────────────────────────────────────────────────────────────
    ap.add_argument("--traj", choices=["v3", "v4"], default="v3",
                    help="v3 = single lerp+slerp home->grasp (linear time). v4 = pre-grasp standoff "
                         "(approach_xy -> align -> descend) with minimum-jerk time scaling.")
    ap.add_argument("--standoff", type=float, default=0.05, help="v4 pre-grasp standoff distance (m)")
    ap.add_argument("--preshape-factor", type=float, default=0.0,
                    help="v4: preshape the gripper to this multiple of the grasp width during the "
                         "approach instead of holding it fully open (1.4 ~ human reach). 0 = disabled.")
    ap.add_argument("--no-minjerk", action="store_true",
                    help="use linear time scaling even with --traj v4 (isolates the standoff change)")
    ap.add_argument("--pinch-area-mm2", type=float, default=None,
                    help="worst-pad contact area below which a grasp is flagged pinch_grasp "
                         "(provisional default 20; the raw grasp_min_pad_mm2 column is the truth)")
    ap.add_argument("--stem-lever-frac", type=float, default=None,
                    help="COM lever arm, as a fraction of the object's bbox radius, above which a "
                         "grasp is flagged stem_grasp (provisional default 0.25)")
    args = ap.parse_args()

    # Resolve profile + explicit overrides into ONE objective dict, and record it in summary.json so
    # every run self-documents the objective that produced it (the missing piece that let the
    # benchmark silently diverge from the collector in the first place).
    objective = dict(GRASP_PROFILES[args.grasp_profile])
    for key, val in (("w_align", args.grasp_align), ("w_peak", args.grasp_peak),
                     ("w_com", args.grasp_com), ("w_tilt", args.grasp_tilt),
                     ("w_occ", args.grasp_occ), ("area_min", args.grasp_area_min),
                     ("execute_offset", args.grasp_execute_offset),
                     ("diversity_tol", args.grasp_diversity_tol), ("jitter_deg", args.grasp_jitter_deg),
                     ("jitter_pos", args.grasp_jitter_pos), ("pitch_seed_deg", args.grasp_pitch_seed_deg)):
        if val is not None:
            objective[key] = val
    if args.grasp_roll_max_deg is not None:
        objective["roll_max"] = np.radians(args.grasp_roll_max_deg)

    exp = Experiment.load(args.experiment)
    # object von-Mises yield (priv_stress is normalized by it -> convert back to Pa for the firm check)
    from gentle_manip.assets.registry import get_object_def
    obj_name = exp.task_cfg.get("object_name", "mushroom")
    _mat = get_object_def(obj_name).material
    yield_pa = float(getattr(_mat, "von_mises_yield_stress", None)
                     or (_mat.get("von_mises_yield_stress") if isinstance(_mat, dict) else None) or 4e4)

    client = SimEnvClient(port=args.port)
    # policy_dt = the scene's sim_dt: this venv steps the sim exactly once per policy step, so the
    # EE trace is sampled at the true sim rate and the smoothness metrics are meaningful.
    from gentle_manip.tasks.single_lift import SingleLiftTask as _SLT
    venv = SimEvalVenv(client, args.num_envs, args.max_steps,
                       policy_dt=float(_SLT(exp.task_cfg).scene_spec.sim_dt))
    grasp_kw = dict(E=args.grasp_E, density=args.grasp_density, mu=args.grasp_mu, accel=args.grasp_accel,
                    n_starts=args.grasp_n_starts, voxel_div=args.grasp_voxel_div,
                    target_tets=args.grasp_target_tets, gpu=not args.grasp_cpu)
    # Always pass the camera the sim actually renders from, even when w_occ == 0: the term stays
    # inert in the objective, but the grasp AUDIT then reports real occlusion instead of a
    # placeholder zero. Costs ~0.05 ms/candidate.
    from gentle_manip.tasks.single_lift import SingleLiftTask
    objective.setdefault("cam_pos", tuple(SingleLiftTask(exp.task_cfg).scene_spec.cameras[0].pos))
    schedule = SCHEDULE_V3 if args.traj == "v3" else SCHEDULE_V4
    print(f"[eval_grasp_synth] objective profile={args.grasp_profile} traj={args.traj} -> "
          f"{ {k: v for k, v in objective.items() if k != 'cam_pos'} }", flush=True)
    policy = GraspSynthPolicy(args.num_envs, exp.action_config, venv, synth=args.synth,
                              maxfevals=args.maxfevals, yield_pa=yield_pa, object_name=obj_name,
                              grasp_kw=grasp_kw, table_z=args.table_z, seed=args.seed,
                              extra_close=args.grasp_extra_close,
                              objective=objective, schedule=schedule,
                              standoff=args.standoff, preshape_factor=args.preshape_factor,
                              use_minjerk=(args.traj == "v4" and not args.no_minjerk))
    if args.pinch_area_mm2 is not None:
        policy.PINCH_AREA_MM2 = float(args.pinch_area_mm2)
    if args.stem_lever_frac is not None:
        policy.STEM_LEVER_FRAC = float(args.stem_lever_frac)
    spec = EvalSpec(n_episodes=args.n_episodes, num_envs=args.num_envs, seed=args.seed,
                    max_policy_steps=args.max_steps, scene_group_size=args.scene_group_size)

    out_dir = (_REPO / "logs" / "scripted_policy"
               / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_grasp_synth_{args.synth}")
    run_eval(venv, policy, spec, out_dir, experiment_name=args.experiment, checkpoint=None,
             record_batches=args.record_batches,
             extra_meta={"algorithm": "scripted_policy", "algo_variant": f"grasp_synth_{args.synth}",
                         "maxfevals": args.maxfevals, "grasp_extra_close": args.grasp_extra_close,
                         # the RESOLVED objective + trajectory, so a run is self-documenting
                         "grasp_profile": args.grasp_profile,
                         "grasp_objective": {k: (list(v) if isinstance(v, tuple) else v)
                                             for k, v in objective.items()},
                         "traj": args.traj, "standoff": args.standoff,
                         "minjerk": bool(args.traj == "v4" and not args.no_minjerk),
                         "preshape_factor": args.preshape_factor})
    policy.close()
    venv.close()
    print(f"\n[eval_grasp_synth:{args.synth}] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
