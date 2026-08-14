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

import collect_demos_synth_v3 as gsv3          # noqa: E402 — FSM + SDF/FEM synth helpers + constants
from smgrasp import finger_grasp as fg          # noqa: E402
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
                 object_name, grasp_kw, table_z, seed):
        self.num_envs = int(num_envs)
        self.action_config = action_config
        self.venv = venv
        self.synth = synth                       # "sdf" | "fem"
        self.maxfevals = int(maxfevals)
        self.yield_pa = float(yield_pa)
        self.object_name = object_name
        self.grasp_kw = dict(grasp_kw)           # E, density, mu, accel, n_starts, voxel_div, target_tets, gpu
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
                print(f"  [{self.synth}] FEM {meta['tets']} tets ndof={meta['ndof']} gpu={meta['gpu']}"
                      f"  mesh={Path(mesh).name}", flush=True)
            for i in range(N):
                seed = int(self._seed_rng.integers(1, 2**31 - 1))
                r = fg.synthesize_grasp(self._fem_obj, self._fem_pad, obj_pos[i], obj_quat[i],
                                        E=gk["E"], density=gk["density"], mu=gk["mu"],
                                        table_z=self.table_z, maxfevals=self.maxfevals,
                                        n_starts=gk["n_starts"], seed=seed, accel=gk["accel"])
                best_x.append(r["x"])
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
        N = self.num_envs
        poses = [gsv3._x_to_targets(x, 1) for x in all_best_x]
        self.pos_b = np.concatenate([p[0] for p in poses], 0).astype(np.float32)
        self.quat_b = np.concatenate([p[1] for p in poses], 0).astype(np.float32)
        self.lift_b = self.pos_b.copy(); self.lift_b[:, 2] += gsv3.LIFT_HEIGHT
        self.width_open = np.full(N, 0.08, np.float32)
        self.width_cls = np.array([p[2] - 0.0025 for p in poses], np.float32)
        self.grip_target = self.width_cls.copy()
        # per-env firm close: soft firms EVERY grasp by the base amount, weak grasps more (never skip)
        self.firm_close = np.full(N, gsv3.FIRM_EXTRA_CLOSE_M, np.float32)
        self.home_pos = home_pos.copy()
        self.home_quat = home_quat.copy()

        def _wxyz_to_rot(q):
            return Rot.from_quat([q[1], q[2], q[3], q[0]])
        self.slerps = [Slerp([0., 1.], Rot.concatenate([_wxyz_to_rot(home_quat[i]),
                                                        _wxyz_to_rot(self.quat_b[i])]))
                       for i in range(N)]
        self.phase_idx = np.zeros(N, np.int64)
        self.phase_step = np.zeros(N, np.int64)
        self.rest_stress = None

    # ── FSM per-env target (mirrors collect_demos_synth_v3._env_target) ─────────
    def _env_target(self, i, phase_idx, phase_step):
        name, dur = gsv3.PHASES[phase_idx]
        if name == "approach":
            alpha = (phase_step + 1) / dur
            pos = self.home_pos[i] + alpha * (self.pos_b[i] - self.home_pos[i])
            xyzw = self.slerps[i](alpha).as_quat()
            quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], np.float32)
            grip = self.width_open[i]
        elif name == "settle":
            pos, quat, grip = self.pos_b[i], self.quat_b[i], self.width_open[i]
        elif name == "grasp":
            alpha = (phase_step + 1) / dur
            pos, quat = self.pos_b[i], self.quat_b[i]
            grip = self.width_open[i] + alpha * (self.width_cls[i] - self.width_open[i])
        elif name == "firm":
            pos, quat = self.pos_b[i], self.quat_b[i]
            alpha = (phase_step + 1) / dur
            self.grip_target[i] = max(0.0, self.width_cls[i] - alpha * self.firm_close[i])
            grip = self.grip_target[i]
        elif name == "lift":
            alpha = (phase_step + 1) / dur
            pos = self.pos_b[i] + alpha * (self.lift_b[i] - self.pos_b[i])
            quat, grip = self.quat_b[i], self.grip_target[i]
        else:  # "hold"
            pos, quat, grip = self.lift_b[i], self.quat_b[i], self.grip_target[i]
        return pos, quat, grip

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
            if self.phase_idx[i] < gsv3.N_PHASES:
                p, q, g = self._env_target(i, int(self.phase_idx[i]), int(self.phase_step[i]))
            else:                                             # FSM done -> freeze at lift/hold target
                p, q, g = self.lift_b[i], self.quat_b[i], self.grip_target[i]
            cur_pos[i], cur_quat[i], cur_grip[i] = p, q, g
        action = gsv3._invert_actions_absolute(cur_pos, cur_quat, cur_grip, self.action_config)
        self._advance(obs)
        return action.astype(np.float32)

    def _advance(self, obs):
        N = self.num_envs
        active = self.phase_idx < gsv3.N_PHASES
        self.phase_step[active] += 1
        durations = np.array([gsv3.PHASES[min(int(p), gsv3.N_PHASES - 1)][1] for p in self.phase_idx])
        rolled = active & (self.phase_step >= durations)
        advance = np.ones(N, np.int64)
        leaving_grasp = rolled & (self.phase_idx == gsv3._GRASP_IDX)
        if np.any(leaving_grasp):                            # soft: ALWAYS firm base, weak firms more
            cur = self._stress_pa(obs)
            for i in np.where(leaving_grasp)[0]:
                rise = float(cur[i] - self.rest_stress[i])
                if rise < gsv3.FIRM_STRESS_THRESH_PA:        # weak grasp -> extra close
                    self.firm_close[i] = gsv3.FIRM_EXTRA_CLOSE_M + gsv3.FIRM_WEAK_EXTRA_CLOSE_M
                # else: base firm close (never skip)
        self.phase_idx[rolled] += advance[rolled]
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
    args = ap.parse_args()

    exp = Experiment.load(args.experiment)
    # object von-Mises yield (priv_stress is normalized by it -> convert back to Pa for the firm check)
    from gentle_manip.assets.registry import get_object_def
    obj_name = exp.task_cfg.get("object_name", "mushroom")
    _mat = get_object_def(obj_name).material
    yield_pa = float(getattr(_mat, "von_mises_yield_stress", None)
                     or (_mat.get("von_mises_yield_stress") if isinstance(_mat, dict) else None) or 4e4)

    client = SimEnvClient(port=args.port)
    venv = SimEvalVenv(client, args.num_envs, args.max_steps)
    grasp_kw = dict(E=args.grasp_E, density=args.grasp_density, mu=args.grasp_mu, accel=args.grasp_accel,
                    n_starts=args.grasp_n_starts, voxel_div=args.grasp_voxel_div,
                    target_tets=args.grasp_target_tets, gpu=not args.grasp_cpu)
    policy = GraspSynthPolicy(args.num_envs, exp.action_config, venv, synth=args.synth,
                              maxfevals=args.maxfevals, yield_pa=yield_pa, object_name=obj_name,
                              grasp_kw=grasp_kw, table_z=args.table_z, seed=args.seed)
    spec = EvalSpec(n_episodes=args.n_episodes, num_envs=args.num_envs, seed=args.seed,
                    max_policy_steps=args.max_steps, scene_group_size=args.scene_group_size)

    out_dir = (_REPO / "logs" / "scripted_policy"
               / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_grasp_synth_{args.synth}")
    run_eval(venv, policy, spec, out_dir, experiment_name=args.experiment, checkpoint=None,
             record_batches=args.record_batches,
             extra_meta={"algorithm": "scripted_policy", "algo_variant": f"grasp_synth_{args.synth}",
                         "maxfevals": args.maxfevals})
    policy.close()
    venv.close()
    print(f"\n[eval_grasp_synth:{args.synth}] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
