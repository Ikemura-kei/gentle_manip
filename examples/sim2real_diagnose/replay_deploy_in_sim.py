"""Open-loop sim2real diagnostic: replay real DEPLOYMENT actions in sim and compare.

Loads recorded deployment episodes from `dataset/real_deploy/<run>/shard_*.pkl`,
replays each episode's actions in Genesis sim (open-loop), and produces:

  Per episode:
  - 4×3 grid figure comparing ee_pos xyz, ee_quat wxyz, gripper_width, quat angular
    diff, and point-cloud zmean(t) — same layout as replay_demo_in_sim.py
  - Cloud overlay figure at the grasp moment (real L515 in blue, sim rendered in red)
  - Multi-step point-cloud snapshots (real vs sim, 5 moments per episode)
  - Optional side-by-side rolling mp4 (--video flag)

Config is driven by a single --experiment name (Experiment.load protocol) — the same
source of truth as collection, training and eval.  Task object/physics, action scales
and obs config are all read from the experiment YAML; nothing is hard-coded here.

Usage (envs/sim):
    uv run --project envs/sim python examples/sim2real_diagnose/replay_deploy_in_sim.py \\
        dataset/real_deploy/rigid_sma_apioc2000 \\
        --experiment single_lift_mushroom_rigid \\
        --n-episodes 5

Override sim physics for edge-case experiments:
    --sim-substeps 210 --mpm-grid-density 250   # soft mushroom substeps
    --cam-fov 40                                # non-default collection fov
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]


# ── helpers ───────────────────────────────────────────────────────────────────

def _valid(pc):
    """Drop all-zero rows (zero-padding)."""
    return pc[~np.all(pc == 0, axis=1)]


# Named (elev, azim) camera angles for the multi-step point-cloud snapshot figure — spans
# the workspace box (x:0.2-0.71, y:-0.215-0.215, z:0-0.45) from enough distinct directions
# to catch a real-vs-sim offset that a single fixed angle (the old "iso" default) could hide.
_VIEW_ANGLES = {
    "iso":   (20, -60),    # original default — 3/4 angled view
    "front": (0, -90),     # looking along -Y -> XZ profile (height vs depth)
    "side":  (0, 0),       # looking along -X -> YZ profile (height vs left/right)
    "top":   (89, -90),    # near top-down -> XY footprint
}


def _quat_angular_diff(q1, q2):
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _align_quat_sign(reference, query):
    sign = np.sign(np.sum(reference * query, axis=-1, keepdims=True))
    sign[sign == 0] = 1.0
    return query * sign


def find_settled_spawn(env, backend, target_xy: np.ndarray, default_xy: np.ndarray,
                      max_corrections: int = 5, max_random_tries: int = 6,
                      tol: float = 0.006, rng=None):
    """Search for an object_dxy whose POST-SETTLE resting XY lands within `tol` meters
    of target_xy (not just the naive spawn-there-and-hope offset).

    Rigid objects can be unstable in their "upright" spawn pose and topple/roll during
    the settle phase (see single_lift_mushroom_rigid.yaml's settle_* comment) —
    independent of spawn height (verified: dropping the object with less clearance
    doesn't reduce this). Since we don't need the SETTLE PROCESS to match reality, only
    the FINAL resting position (matching the real point cloud), we instead search for
    whatever spawn offset happens to settle in the right place:
      1. Start from the naive offset (target - default).
      2. If the resting position misses by more than `tol`, correct the offset by the
         observed error (assumes the settle-induced shift is roughly translation-
         invariant near the current candidate — usually converges in 1-3 tries since
         the topple direction/magnitude tends to be consistent).
      3. If that doesn't converge within `max_corrections`, fall back to random
         perturbations around the best candidate found so far (a small black-box
         search) for `max_random_tries` more attempts.
    Every attempt is a real env.reset() (full settle), so this is `env.step`-free but
    NOT free — budget accordingly.

    Returns (obs, offset_used, achieved_drift_m, n_tries).
    """
    rng = rng or np.random.default_rng(0)
    offset = (target_xy - default_xy).astype(np.float32)
    best = {"obs": None, "offset": None, "drift": np.inf}
    n_tries = 0

    def _try(off):
        nonlocal n_tries
        n_tries += 1
        obs = env.reset(object_dxy=off[None, :])
        settled_xy = np.asarray(backend.get_sim_feedback().object_center)[0, :2]
        drift_vec = settled_xy - target_xy
        drift = float(np.linalg.norm(drift_vec))
        if drift < best["drift"]:
            best.update(obs=obs, offset=off.copy(), drift=drift)
        return obs, drift_vec, drift

    for _ in range(max_corrections):
        obs, drift_vec, drift = _try(offset)
        if drift <= tol:
            return obs, offset, drift, n_tries
        offset = offset - drift_vec        # push the spawn point opposite the observed miss

    for _ in range(max_random_tries):
        cand = best["offset"] + rng.uniform(-0.02, 0.02, size=2).astype(np.float32)
        obs, _drift_vec, drift = _try(cand)
        if drift <= tol:
            return obs, cand, drift, n_tries

    return best["obs"], best["offset"], best["drift"], n_tries


def load_shards(deploy_dir: Path) -> list:
    shards = sorted(deploy_dir.glob("shard_*.pkl"))
    if not shards:
        single = deploy_dir / "data.pkl"
        shards = [single] if single.exists() else []
    import pickle
    episodes = []
    for s in shards:
        d = pickle.load(open(s, "rb"))
        episodes.extend(d["episodes"])
    return episodes


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("deploy_dir", type=Path,
                    help="deployment run dir (shard_*.pkl files)")
    ap.add_argument("--experiment", default="single_lift_mushroom_rigid",
                    help="experiment name (configs/experiments/<name>.yaml) — "
                         "single source of truth for task/obs/action/dr")

    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--episodes", default="",
                    help="comma-sep explicit episode indices (overrides --n-episodes)")
    ap.add_argument("--seed", type=int, default=0)

    # Physics overrides (default: read from task_cfg in the experiment YAML).
    ap.add_argument("--sim-substeps", type=int, default=None,
                    help="override substeps (default: task_cfg.sim_substeps)")
    ap.add_argument("--mpm-grid-density", type=float, default=None,
                    help="override MPM grid density (default: task_cfg.mpm_grid_density)")
    ap.add_argument("--cam-fov", type=float, default=None,
                    help="override camera fov (default: task_cfg.cam_fov or 46.0)")

    ap.add_argument("--max-steps", type=int, default=0, help="0 = full episode")
    ap.add_argument("--sim-pc-shift-x", type=float, default=0.0,
                    help="POST-hoc shift (meters) applied to the sim point cloud's x "
                         "coordinate, visualization-only (does not affect the sim rollout, "
                         "ee/quat/gripper tracking, or any other metric) — a quick way to "
                         "test an x-offset-alignment hypothesis between real (L515) and "
                         "sim-rendered clouds without touching calibration/extrinsics.")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <deploy_dir>/sim_replay/<timestamp>/)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--video", action="store_true",
                    help="render side-by-side (real|sim) rolling cloud mp4 per episode")
    ap.add_argument("--video-episodes", type=int, default=2)
    ap.add_argument("--video-fps", type=int, default=15)
    ap.add_argument("--overlay-video-views", default="iso,front,side,top",
                    help="comma-sep camera angles for the rolling real+sim OVERLAY video — "
                         "one mp4 PER view (traj_NN_cloud_overlay_video_<view>.mp4). "
                         "front/side/top are the views normal to the xz/yz/xy planes "
                         "respectively (see _VIEW_ANGLES). Render cost scales with the "
                         "number of views (each is a separate full pass over the episode).")
    ap.add_argument("--pc-views", default="iso,front,side,top",
                     help="comma-sep named camera angles for the multi-step point-cloud "
                          "snapshot figure (traj_NN_pointcloud.png), so you can see the "
                          "real-vs-sim cloud difference from more than one angle. Choices: "
                          "iso,front,side,top (see _VIEW_ANGLES).")
    ap.add_argument("--search-spawn", action="store_true",
                    help="instead of a single naive object_dxy=(cube_xy-default_xy) reset, "
                         "search for a spawn offset whose POST-SETTLE resting XY actually "
                         "lands near cube_xy (see find_settled_spawn) -- use when the object "
                         "topples/rolls away from the intended spot during the settle phase "
                         "regardless of spawn height. Costs extra env.reset() calls/episode.")
    ap.add_argument("--search-tol", type=float, default=0.006,
                    help="max acceptable object_xy_drift (meters) for --search-spawn to "
                         "stop early")
    ap.add_argument("--search-max-corrections", type=int, default=5)
    ap.add_argument("--search-max-random-tries", type=int, default=6)
    args = ap.parse_args()

    view_names = [v.strip() for v in args.pc_views.split(",") if v.strip()]
    unknown = [v for v in view_names if v not in _VIEW_ANGLES]
    if unknown:
        raise ValueError(f"unknown --pc-views entries {unknown}; choices: {list(_VIEW_ANGLES)}")
    n_views = len(view_names)

    overlay_video_views = [v.strip() for v in args.overlay_video_views.split(",") if v.strip()]
    unknown_ov = [v for v in overlay_video_views if v not in _VIEW_ANGLES]
    if unknown_ov:
        raise ValueError(f"unknown --overlay-video-views entries {unknown_ov}; "
                          f"choices: {list(_VIEW_ANGLES)}")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gentle_manip.assets.registry import get_object_def
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.experiment import Experiment
    from gentle_manip.tasks.single_lift import SingleLiftTask

    # ── load experiment (single source of truth) ──────────────────────────────
    exp = Experiment.load(args.experiment)
    obs_cfg    = exp.view_obs("student")   # deployable point-cloud student view
    act_cfg    = exp.action_config
    task_cfg   = dict(exp.task_cfg)        # mutable copy for physics overrides

    # Apply CLI overrides — let caller pin exact physics without touching the YAML.
    if args.sim_substeps is not None:
        task_cfg["sim_substeps"] = args.sim_substeps
    if args.mpm_grid_density is not None:
        task_cfg["mpm_grid_density"] = args.mpm_grid_density
    if args.cam_fov is not None:
        task_cfg["cam_fov"] = args.cam_fov

    task = SingleLiftTask(task_cfg)
    fov  = task.scene_spec.cameras[0].fov
    obj_name = task_cfg.get("object_name", "mushroom")
    obj_type = task_cfg.get("object_type", "rigid")

    default_xy = np.array(get_object_def(obj_name).default_pos[:2], dtype=np.float32)

    print(f"Experiment : {args.experiment}", flush=True)
    print(f"Object     : {obj_name}/{obj_type}  substeps={task_cfg.get('sim_substeps')}"
          f"  cam_fov={fov}", flush=True)
    print(f"Obs        : {obs_cfg}", flush=True)
    print(f"Action     : {act_cfg}", flush=True)

    # ── sim env ────────────────────────────────────────────────────────────────
    backend = SimBackend(task.scene_spec, 1, config={"sim": {"settle_steps": 20}},
                         use_subprocess=False)
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=None, max_episode_steps=10 ** 9)

    # ── load episodes ──────────────────────────────────────────────────────────
    episodes = load_shards(args.deploy_dir)
    print(f"Loaded {len(episodes)} episodes from {args.deploy_dir}", flush=True)

    if args.episodes.strip():
        picks = [int(x) for x in args.episodes.split(",")]
    else:
        rng = np.random.default_rng(args.seed)
        picks = sorted(rng.choice(len(episodes), size=min(args.n_episodes, len(episodes)),
                                   replace=False).tolist())
    print(f"Episodes: {picks}", flush=True)

    # ── output dir ────────────────────────────────────────────────────────────
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = args.deploy_dir / "sim_replay" / ts
    else:
        out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out}", flush=True)

    cam = task.scene_spec.cameras[0]
    run_config = {
        "deploy_dir": str(args.deploy_dir),
        "experiment": args.experiment,
        "episodes": picks,
        "seed": args.seed,
        "object": obj_name,
        "object_type": obj_type,
        "task_cfg": task_cfg,
        "max_steps": args.max_steps,
        "camera": {
            "name": cam.name,
            "fov": float(cam.fov),
            "pos": [float(x) for x in cam.pos],
            "lookat": [float(x) for x in cam.lookat],
            "resolution": list(cam.resolution),
        },
    }
    run_config["sim_pc_shift_x"] = args.sim_pc_shift_x
    (out / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))

    def _sim_pc(sim_obs_t):
        """sim_obs[t]'s point cloud, with the visualization-only x-shift applied (see
        --sim-pc-shift-x). No-op when the shift is 0. Only shifts VALID (non-zero-padded)
        rows — shifting a zero-padding row would turn it into a fake nonzero point and
        break _valid()'s zero-padding filter downstream."""
        pc = sim_obs_t["point_cloud"][0]
        if args.sim_pc_shift_x == 0.0:
            return pc
        pc = pc.copy()
        nz = ~np.all(pc == 0, axis=1)
        pc[nz, 0] += args.sim_pc_shift_x
        return pc

    # ── replay loop ────────────────────────────────────────────────────────────
    summary = []
    videos_made = 0

    for ep_idx in picks:
        ep = episodes[ep_idx]
        actions = np.asarray(ep["actions"], dtype=np.float32)
        T = len(actions) if args.max_steps <= 0 else min(args.max_steps, len(actions))

        obs_ep  = ep["observations"]
        re_ee   = np.asarray(obs_ep["ee_pos"],        np.float32)[:T]
        re_quat = np.asarray(obs_ep["ee_quat"],       np.float32)[:T]
        re_gw   = np.asarray(obs_ep["gripper_width"], np.float32)[:T, 0]
        re_pc   = np.asarray(obs_ep["point_cloud"],   np.float32)[:T]

        # Seed object XY from EE position at the grasp (lowest EE z in real trajectory).
        grasp_t = int(np.argmin(re_ee[:, 2]))
        cube_xy = re_ee[grasp_t, :2]

        if args.search_spawn:
            obs, _offset, obj_xy_drift, n_spawn_tries = find_settled_spawn(
                env, backend, cube_xy, default_xy,
                max_corrections=args.search_max_corrections,
                max_random_tries=args.search_max_random_tries,
                tol=args.search_tol)
        else:
            obs = env.reset(object_dxy=(cube_xy - default_xy)[None, :])
            n_spawn_tries = 1
            # Diagnostic: where the object ACTUALLY settled vs. where we told it to spawn.
            # If the object free-falls + rolls during the post-spawn settle phase (rigid
            # bodies can roll for a while — see single_lift_mushroom_rigid.yaml's settle_*
            # comment), the resting position can drift away from cube_xy before the episode
            # even starts, desyncing sim from the real point cloud it was seeded from.
            obj_center = backend.get_sim_feedback().object_center[0]      # (3,), post-settle
            obj_xy_drift = float(np.linalg.norm(obj_center[:2] - cube_xy))

        sim_obs = [obs]
        for t in range(T - 1):
            sim_obs.append(env.step(actions[t][None, :])[0])

        sim_ee   = np.stack([o["ee_pos"][0]          for o in sim_obs])
        sim_quat = np.stack([o["ee_quat"][0]          for o in sim_obs])
        sim_gw   = np.array([o["gripper_width"][0, 0] for o in sim_obs])

        re_quat_aligned = _align_quat_sign(sim_quat, re_quat)
        quat_ang        = _quat_angular_diff(sim_quat, re_quat)
        quat_elem_err   = np.abs(sim_quat - re_quat_aligned)

        re_zm  = np.array([_valid(re_pc[t])[:, 2].mean()
                            if len(_valid(re_pc[t])) else 0.0 for t in range(T)])
        sim_zm = np.array([_valid(_sim_pc(sim_obs[t]))[:, 2].mean()
                            for t in range(T)])

        ee_err         = np.abs(sim_ee - re_ee).mean(0)
        quat_ang_err   = float(np.rad2deg(quat_ang.mean()))
        quat_elem_mean = quat_elem_err.mean(0)
        gw_err         = float(np.abs(sim_gw - re_gw).mean())
        zoff           = float(np.abs(sim_zm - re_zm).mean())
        summary.append((ep_idx, ee_err, quat_ang_err, quat_elem_mean, gw_err, zoff,
                       obj_xy_drift, n_spawn_tries))

        print(
            f"ep {ep_idx}: T={T}  cube_xy={cube_xy.round(3)}"
            f"  ee_err(mm)={(ee_err*1000).round(1)}"
            f"  quat_ang(deg)={quat_ang_err:.2f}"
            f"  gw_err(mm)={gw_err*1000:.1f}"
            f"  cloud_zoff(mm)={zoff*1000:.1f}"
            f"  obj_xy_drift(mm)={obj_xy_drift*1000:.1f}"
            + (f"  spawn_tries={n_spawn_tries}" if args.search_spawn else ""),
            flush=True,
        )

        ts_ax = np.arange(T)

        # ── 4×3 comparison grid ──────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 12))

        for i, lbl in enumerate("xyz"):
            ax = fig.add_subplot(4, 3, i + 1)
            ax.plot(ts_ax, re_ee[:, i],  label="real", lw=2)
            ax.plot(ts_ax, sim_ee[:, i], "--", label="sim", lw=2)
            ax.set_title(f"ee_pos {lbl} (m)")
            ax.grid(alpha=0.3);  ax.legend(fontsize=8)

        quat_labels = ("w", "x", "y", "z")
        for i in range(3):
            ax = fig.add_subplot(4, 3, i + 4)
            ax.plot(ts_ax, re_quat_aligned[:, i], label="real", lw=2)
            ax.plot(ts_ax, sim_quat[:, i],        "--", label="sim", lw=2)
            ax.set_title(f"ee_quat {quat_labels[i]}")
            ax.grid(alpha=0.3);  ax.legend(fontsize=8)

        ax = fig.add_subplot(4, 3, 7)
        ax.plot(ts_ax, re_quat_aligned[:, 3], label="real", lw=2)
        ax.plot(ts_ax, sim_quat[:, 3],        "--", label="sim", lw=2)
        ax.set_title(f"ee_quat {quat_labels[3]}")
        ax.grid(alpha=0.3);  ax.legend(fontsize=8)

        ax = fig.add_subplot(4, 3, 8)
        ax.plot(ts_ax, re_gw,  label="real", lw=2)
        ax.plot(ts_ax, sim_gw, "--", label="sim", lw=2)
        ax.set_title("gripper_width (m)")
        ax.grid(alpha=0.3);  ax.legend(fontsize=8)

        ax = fig.add_subplot(4, 3, 9)
        ax.plot(ts_ax, np.rad2deg(quat_ang), lw=2, color="tab:purple")
        ax.set_title("quat angular diff (deg)")
        ax.grid(alpha=0.3)

        ax = fig.add_subplot(4, 3, 10)
        ax.plot(ts_ax, re_zm,  label="real", lw=2)
        ax.plot(ts_ax, sim_zm, "--", label="sim", lw=2)
        ax.set_title("point-cloud zmean(t) (m)")
        ax.grid(alpha=0.3);  ax.legend(fontsize=8)

        fig.suptitle(
            f"Deploy ep {ep_idx} [{args.experiment}] (fov={fov}) — "
            f"ee err {(ee_err*1000).round(1)} mm | "
            f"quat ang {quat_ang_err:.2f} deg | "
            f"gw {gw_err*1000:.1f} mm | "
            f"cloud zmean {zoff*1000:.1f} mm"
        )
        fig.tight_layout()
        fpath = out / f"traj_{ep_idx:02d}.png"
        fig.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fpath}", flush=True)

        # ── cloud overlay at grasp (multiple view angles) ──────────────────────
        rp = _valid(re_pc[grasp_t])
        sp = _valid(_sim_pc(sim_obs[grasp_t]))
        figo = plt.figure(figsize=(6 * n_views, 5.5))
        for vi, view_name in enumerate(view_names):
            elev, azim = _VIEW_ANGLES[view_name]
            a3 = figo.add_subplot(1, n_views, vi + 1, projection="3d")
            a3.scatter(rp[:, 0], rp[:, 1], rp[:, 2], s=2, c="tab:blue",
                       alpha=0.4, label=f"real L515 ({len(rp)} pts)")
            a3.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=2, c="tab:red",
                       alpha=0.4, label=f"sim rendered ({len(sp)} pts)")
            a3.set_title(f"[{view_name}]")
            if vi == 0:
                a3.legend(fontsize=8)
            a3.set_xlim(0.2, 0.71); a3.set_ylim(-0.215, 0.215); a3.set_zlim(0, 0.45)
            a3.view_init(elev, azim)
        figo.suptitle(f"Deploy ep {ep_idx} [{args.experiment}] — cloud overlay @ grasp (t={grasp_t})")
        figo.tight_layout()
        opath = out / f"traj_{ep_idx:02d}_cloud_overlay.png"
        figo.savefig(opath, dpi=110, bbox_inches="tight")
        plt.close(figo)
        print(f"  saved {opath}", flush=True)

        # ── multi-step point-cloud snapshots (multiple view angles) ───────────
        # rows = snapshot moments, cols = (real, sim) PER view angle, so each moment
        # can be visually cross-checked from more than one direction — a real-vs-sim
        # offset that's invisible from one fixed angle (the old iso-only default) can
        # show up clearly from another (e.g. a height offset is obvious "front", a
        # lateral offset is obvious "top").
        snaps = sorted(set([0, T // 4, T // 2, 3 * T // 4, T - 1]))
        figp = plt.figure(figsize=(6 * n_views, 3.6 * len(snaps)))
        n_cols = 2 * n_views
        for r, t in enumerate(snaps):
            for vi, view_name in enumerate(view_names):
                elev, azim = _VIEW_ANGLES[view_name]
                for c, (tag, pc) in enumerate([
                        ("real (L515)", re_pc[t]),
                        ("sim (rendered)", _sim_pc(sim_obs[t]))]):
                    v = _valid(pc)
                    col = vi * 2 + c
                    a = figp.add_subplot(len(snaps), n_cols, r * n_cols + col + 1, projection="3d")
                    a.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c=v[:, 2],
                              cmap="viridis", vmin=0.0, vmax=0.45, alpha=0.5)
                    a.set_title(f"{tag}  t={t}  [{view_name}]  ({len(v)} pts)", fontsize=9)
                    a.set_xlim(0.2, 0.71); a.set_ylim(-0.215, 0.215); a.set_zlim(0, 0.45)
                    a.view_init(elev, azim)
        figp.suptitle(
            f"Deploy ep {ep_idx} [{args.experiment}] — point cloud: real L515 vs sim  "
            f"(views: {', '.join(view_names)})")
        figp.tight_layout()
        ppath = out / f"traj_{ep_idx:02d}_pointcloud.png"
        figp.savefig(ppath, dpi=110, bbox_inches="tight")
        plt.close(figp)
        print(f"  saved {ppath}", flush=True)

        # ── optional rolling cloud videos (side-by-side AND overlaid) ──────────
        if args.video and videos_made < args.video_episodes:
            import imageio.v2 as imageio

            # side-by-side: real | sim in separate subplots
            figv = plt.figure(figsize=(12, 5.5))
            axr = figv.add_subplot(1, 2, 1, projection="3d")
            axs = figv.add_subplot(1, 2, 2, projection="3d")
            frames = []
            for t in range(T):
                for ax, tag, pc in [
                        (axr, "real (L515)", re_pc[t]),
                        (axs, "sim (rendered)", _sim_pc(sim_obs[t]))]:
                    ax.clear()
                    v = _valid(pc)
                    ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c=v[:, 2],
                               cmap="viridis", vmin=0.0, vmax=0.45, alpha=0.5)
                    ax.set_xlim(0.2, 0.71); ax.set_ylim(-0.215, 0.215); ax.set_zlim(0, 0.45)
                    ax.view_init(30, -60)
                    ax.set_title(f"{tag}  t={t}  ({len(v)} pts)")
                figv.suptitle(f"Deploy ep {ep_idx} [{args.experiment}] — real vs sim cloud")
                figv.canvas.draw()
                frames.append(np.asarray(figv.canvas.buffer_rgba())[..., :3].copy())
            plt.close(figv)
            vpath = out / f"traj_{ep_idx:02d}_cloud_video.mp4"
            imageio.mimsave(str(vpath), frames, fps=args.video_fps, macro_block_size=1)
            print(f"  saved {vpath} ({len(frames)} frames)", flush=True)

            # overlaid: real (blue) + sim (red) in ONE subplot — the rolling version of
            # the static traj_NN_cloud_overlay.png grasp snapshot, across the WHOLE episode,
            # so a real-vs-sim offset that drifts over time (not just at the grasp instant)
            # is directly visible rather than needing to compare two separate panels. One
            # mp4 PER requested view (front/side/top = normal to the xz/yz/xy planes).
            for view_name in overlay_video_views:
                elev, azim = _VIEW_ANGLES[view_name]
                figov = plt.figure(figsize=(7, 6))
                axov = figov.add_subplot(111, projection="3d")
                oframes = []
                for t in range(T):
                    axov.clear()
                    rp_t = _valid(re_pc[t])
                    sp_t = _valid(_sim_pc(sim_obs[t]))
                    axov.scatter(rp_t[:, 0], rp_t[:, 1], rp_t[:, 2], s=2, c="tab:blue",
                                 alpha=0.4, label=f"real L515 ({len(rp_t)} pts)")
                    axov.scatter(sp_t[:, 0], sp_t[:, 1], sp_t[:, 2], s=2, c="tab:red",
                                 alpha=0.4, label=f"sim rendered ({len(sp_t)} pts)")
                    axov.set_xlim(0.2, 0.71); axov.set_ylim(-0.215, 0.215); axov.set_zlim(0, 0.45)
                    axov.view_init(elev, azim)
                    axov.legend(fontsize=8, loc="upper left")
                    axov.set_title(f"Deploy ep {ep_idx} [{args.experiment}] — cloud overlay  "
                                    f"t={t}/{T - 1}  [{view_name}]")
                    figov.canvas.draw()
                    oframes.append(np.asarray(figov.canvas.buffer_rgba())[..., :3].copy())
                plt.close(figov)
                ovpath = out / f"traj_{ep_idx:02d}_cloud_overlay_video_{view_name}.mp4"
                imageio.mimsave(str(ovpath), oframes, fps=args.video_fps, macro_block_size=1)
                print(f"  saved {ovpath} ({len(oframes)} frames)", flush=True)

            videos_made += 1

    env.close()

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n=== sim-replay summary [{args.experiment}] fov={fov} ===", flush=True)
    print(f"{'ep':>4}  {'ee_err xyz(mm)':>22}  {'quat_ang':>8}  {'gw_err':>7}  {'cloud_z':>7}"
          f"  {'obj_drift':>9}  {'tries':>5}", flush=True)
    for ep_idx, ee_err, quat_ang_err, _, gw_err, zoff, obj_xy_drift, n_spawn_tries in summary:
        print(
            f"{ep_idx:>4}  {str((ee_err*1000).round(1)):>22}"
            f"  {quat_ang_err:>7.2f}°"
            f"  {gw_err*1000:>6.1f}mm"
            f"  {zoff*1000:>6.1f}mm"
            f"  {obj_xy_drift*1000:>7.1f}mm"
            f"  {n_spawn_tries:>5}",
            flush=True,
        )
    print(f"\nAll outputs → {out}", flush=True)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
