"""Generate a PAIRED sim twin of a recorded real demo run by replaying its actions in sim.

For every real episode: place the sim object at the real first-frame TCP xy ("right below
the arm" — the item-1 probe protocol), match the sim home pose to the real first-frame EE
pose (Cartesian home_offset), then replay the recorded delta actions open-loop through the
SAME shared ActionPipeline/PerceptionPipeline (obs + action dicts are taken from the real
recording's own config.yaml — the baked authority — so the sim observations are processed
identically by construction). Output is a demo-schema data.pkl paired STEP-FOR-STEP with
the real one: episode i / step t in both pkls is the same commanded state.

Purpose: real–sim paired training data for encoder feature-consistency regularization
(features of paired real/sim steps should be close), plus the item-1 data-difference
analysis. A match report (per-episode EE/quat/gripper/cloud errors) and per-episode
real|sim figures+videos are written next to the dataset so pairing quality is auditable.

    uv run --project envs/sim python -m gentle_manip.scripts.replay_real_to_sim_paired \
        --real-run dataset/demos/single_lift_cube3_real/26-08-23-oso
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]


def _valid(pc):
    return pc[~np.all(pc == 0, axis=1)]


def _align_quat_sign(reference, query):
    sign = np.sign(np.sum(reference * query, axis=-1, keepdims=True))
    sign[sign == 0] = 1.0
    return query * sign


def _quat_angular_diff_deg(q1, q2):
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def _cloud_nn_dist(a, b):
    """Mean nearest-neighbour distance a->b (m), zero-pad aware."""
    from scipy.spatial import cKDTree
    a, b = _valid(a), _valid(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(cKDTree(b).query(a, k=1)[0].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-run", type=Path, required=True,
                    help="real demo run dir (data.pkl + config.yaml from demos/record.py)")
    ap.add_argument("--task-config", type=Path,
                    default=REPO / "gentle_manip/configs/tasks/single_lift_cube3_rigid.yaml",
                    help="sim task cfg providing the scene (object + calibrated cam_ext twin)")
    ap.add_argument("--task-name", default="single_lift_cube3_rigid")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dataset dir; default dataset/demos/<task-name>/<real leaf>")
    ap.add_argument("--episodes", default="", help="comma-sep episode indices (default: all)")
    ap.add_argument("--object-z", type=float, default=None,
                    help="object centre height (m) at spawn; default = the task's object_spawn_z (no z shift)")
    ap.add_argument("--object-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="where the real object actually was (robot-frame m), same for every episode; "
                         "default = the real first-frame TCP xy (the 'cube right below the arm' protocol)")
    ap.add_argument("--video-stride", type=int, default=2)
    ap.add_argument("--with-rgb", action="store_true",
                    help="also render RGB in sim (paired RGB). Requires the scene spec's camera "
                         "resolution; slower. Default drops images from the obs config.")
    ap.add_argument("--render-only", action="store_true",
                    help="skip the sim replay; re-render figures/videos from the existing "
                         "paired data.pkl in --out (e.g. after changing the video views)")
    args = ap.parse_args()

    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.assets.registry import get_object_def
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.tasks.single_lift import SingleLiftTask

    real = pickle.load(open(args.real_run / "data.pkl", "rb"))
    rec_cfg = yaml.safe_load((args.real_run / "config.yaml").read_text())
    eps = real["episodes"]
    picks = ([int(x) for x in args.episodes.split(",")] if args.episodes.strip()
             else list(range(len(eps))))

    # The recording's own config.yaml is the authority for obs/action processing (the
    # record-time pipeline is baked into the stored clouds) — NOT any training snapshot.
    obs_cfg = ObsConfig.from_dict(rec_cfg["obs"])
    # A PAIRED-RGB real recording carries `images:` in its obs config. The sim env then needs an
    # rgb_shape, and the backend must actually render RGB -- neither is needed for the point-cloud
    # discrepancy this tool exists for. Default: drop images and say so. --with-rgb keeps them and
    # supplies the shape from the scene spec (also gives PAIRED RGB, useful for the VLA sim2real
    # question, at the cost of rendering every frame).
    if obs_cfg.images is not None and not args.with_rgb:
        import dataclasses as _dc
        obs_cfg = _dc.replace(obs_cfg, images=None)
        print("[replay] real obs config has images:; DROPPED for the sim twin (point clouds are "
              "what the pairing is for). Pass --with-rgb to keep them.", flush=True)
    # The recorder stores actions in `record_action` space when that key exists (newer demos:
    # teleop reads deltas via `action`, but RECORDS 7-dim absolute euler). Decoding absolute
    # actions with the delta config integrates ~0.5-magnitude "increments" every step -- the sim
    # arm walks off (seen: 70 mm mean / 293 mm max EE error on the red-cube demo). The cube3-era
    # demos have no `record_action`; for them `action` is both the teleop and the storage space.
    _act_src = "record_action" if rec_cfg.get("record_action") else "action"
    act_cfg = ActionConfig.from_dict(rec_cfg[_act_src])
    print(f"[replay] decoding actions with rec_cfg[{_act_src!r}] "
          f"(mode={rec_cfg[_act_src].get('mode', 'delta')})", flush=True)

    task_dict = yaml.safe_load(args.task_config.read_text())
    task = SingleLiftTask(task_dict)
    # The object shift is applied RELATIVE TO THE SCENE SPAWN (the base particles/pose captured at build):
    # task object_spawn_xy/z when set, else the registry default_pos.
    _def = get_object_def(task.object_name).default_pos
    spawn_xy = np.array(task.object_spawn_xy if getattr(task, "object_spawn_xy", None) is not None else _def[:2], np.float64)
    spawn_z = float(task.object_spawn_z if getattr(task, "object_spawn_z", None) is not None else _def[2])
    print(f"[replay] scene spawn (shift reference): xy {spawn_xy} z {spawn_z:.4f}", flush=True)

    out = args.out or (REPO / "dataset/demos" / args.task_name / args.real_run.name)
    out.mkdir(parents=True, exist_ok=True)
    print(f"real run: {args.real_run}\nout:      {out}\nepisodes: {picks}", flush=True)

    if args.render_only:                      # figures + report + config from the existing paired data.pkl
        paired = pickle.load(open(out / "data.pkl", "rb")); report, placements = [], []
        for ep_idx, sep in zip(picks, paired["episodes"]):
            ep = eps[ep_idx]
            re_ee, re_quat = np.asarray(ep["observations"]["ee_pos"]), np.asarray(ep["observations"]["ee_quat"])
            re_gw, re_pc = np.asarray(ep["observations"]["gripper_width"])[:, 0], np.asarray(ep["observations"]["point_cloud"])
            rec = {k: np.asarray(v) for k, v in sep["observations"].items()}
            row = _match_row(ep_idx, re_ee, re_quat, re_gw, re_pc, rec); report.append(row)
            placements.append({"episode": int(ep_idx), "cube_xy": ([float(v) for v in args.object_xy] if args.object_xy is not None
                                                                   else [float(v) for v in re_ee[0, :2]]), "home_offset": None})
            print(f"ep {ep_idx}: T={row['steps']}  ee_err {row['ee_err_mean_mm']:.1f} mm (max {row['ee_err_max_mm']:.1f})  "
                  f"quat {row['quat_ang_mean_deg']:.2f} deg  gw {row['gw_err_mean_mm']:.1f} mm  cloud_nn {row['cloud_nn_mean_mm']:.1f} mm", flush=True)
            _figures(out, ep_idx, re_ee, re_quat, re_gw, re_pc, rec, args.video_stride)
        _write_meta(out, args, rec_cfg, _act_src, report, placements)
        return

    backend = SimBackend(task.scene_spec, 1, config={"sim": {"settle_steps": 20}},
                         use_subprocess=False, show_viewer=False)
    _rgb_shape = None
    if obs_cfg.images is not None:
        _cam = task.scene_spec.cameras[0]
        _w, _h = _cam.resolution
        _rgb_shape = (int(_h), int(_w))
        print(f"[replay] rgb_shape={_rgb_shape} from scene_spec camera {_cam.name}", flush=True)
    env = PolicyEnv(backend, obs_cfg, act_cfg, task=None, max_episode_steps=10 ** 9,
                    rgb_shape=_rgb_shape)

    # Nominal sim home (no offset) — reference for the per-episode Cartesian home match.
    obs0 = env.reset(object_dxy=[[0.0, 0.0]], home_offset=[[0.0, 0.0, 0.0]],
                     object_euler=[[0.0, 0.0, 0.0]])
    nominal_home = obs0["ee_pos"][0].astype(np.float64)
    print(f"sim nominal home EE: {nominal_home.round(4)}", flush=True)

    sim_episodes, report, placements = [], [], []
    for ep_idx in picks:
        ep = eps[ep_idx]
        actions = np.asarray(ep["actions"], dtype=np.float32)
        T = len(actions)
        re_ee = np.asarray(ep["observations"]["ee_pos"])
        re_quat = np.asarray(ep["observations"]["ee_quat"])
        re_gw = np.asarray(ep["observations"]["gripper_width"])[:, 0]
        re_pc = np.asarray(ep["observations"]["point_cloud"])

        # Object xy: given explicitly, else the item-1 protocol (cube right below the arm = first-frame TCP xy).
        cube_xy = (np.asarray(args.object_xy, np.float64) if args.object_xy is not None
                   else re_ee[0, :2].astype(np.float64))
        home_off = re_ee[0].astype(np.float64) - nominal_home
        placements.append({"episode": int(ep_idx),
                           "cube_xy": [float(v) for v in cube_xy],
                           "home_offset": [float(v) for v in home_off]})

        shift = np.r_[cube_xy - spawn_xy, (args.object_z - spawn_z) if args.object_z is not None else 0.0]
        obs = env.reset(object_dxy=shift[None, :],
                        home_offset=home_off[None, :],
                        object_euler=[[0.0, 0.0, 0.0]])
        _oc = backend.process.read_state()["object_center"][0]
        print(f"  sim object centre after settle: {np.round(_oc, 4)}  (requested xy {np.round(cube_xy, 4)})", flush=True)
        sim_obs = [obs]
        for t in range(T - 1):
            obs, _, _, _ = env.step(actions[t][None, :])
            sim_obs.append(obs)

        # Use the real recording's key list ONLY where the sim obs actually has the key: with
        # images dropped (no --with-rgb) the sim twin has no image_cam_ext, and indexing the real
        # key list blindly raised KeyError. The pairing is per-key; missing keys are named once.
        _keys = [k for k in real["meta"]["obs_keys"] if k in sim_obs[0]]
        _missing = [k for k in real["meta"]["obs_keys"] if k not in sim_obs[0]]
        if _missing and ep_idx == picks[0]:
            print(f"[replay] real-only obs keys not in the sim twin (dropped/unavailable): "
                  f"{_missing}", flush=True)
        rec = {k: np.stack([o[k][0] for o in sim_obs]) for k in _keys}
        sim_episodes.append({"observations": rec,
                             "actions": actions.copy(),
                             "rewards": np.zeros(T, dtype=np.float32)})

        row = _match_row(ep_idx, re_ee, re_quat, re_gw, re_pc, rec)   # pairing-quality metrics
        report.append(row)
        print(f"ep {ep_idx}: T={T}  ee_err {row['ee_err_mean_mm']:.1f} mm "
              f"(max {row['ee_err_max_mm']:.1f})  quat {row['quat_ang_mean_deg']:.2f} deg  "
              f"gw {row['gw_err_mean_mm']:.1f} mm  cloud_nn {row['cloud_nn_mean_mm']:.1f} mm",
              flush=True)

        _figures(out, ep_idx, re_ee, re_quat, re_gw, re_pc, rec, args.video_stride)

    env.close()
    meta = dict(real["meta"])
    meta.update(task=args.task_name, n_episodes=len(sim_episodes),
                created=datetime.now(timezone.utc).isoformat(),
                paired_source=str(args.real_run))
    with open(out / "data.pkl", "wb") as f:
        pickle.dump({"meta": meta, "episodes": sim_episodes}, f)
    _write_meta(out, args, rec_cfg, _act_src, report, placements)


def _match_row(ep_idx, re_ee, re_quat, re_gw, re_pc, rec):
    """Pairing-quality metrics of one episode: real vs sim proprio + cloud nearest-neighbour."""
    T = len(re_ee); sim_ee, sim_quat, sim_gw = rec["ee_pos"], rec["ee_quat"], rec["gripper_width"][:, 0]
    ee_err = np.abs(sim_ee - re_ee)
    qa = _quat_angular_diff_deg(sim_quat, _align_quat_sign(sim_quat, re_quat))
    nn = np.array([_cloud_nn_dist(re_pc[t], rec["point_cloud"][t]) for t in range(0, T, 5)])
    return {"episode": int(ep_idx), "steps": int(T),
            "ee_err_mean_mm": float(ee_err.mean() * 1000),
            "ee_err_max_mm": float(np.linalg.norm(sim_ee - re_ee, axis=1).max() * 1000),
            "quat_ang_mean_deg": float(qa.mean()),
            "gw_err_mean_mm": float(np.abs(sim_gw - re_gw).mean() * 1000),
            "cloud_nn_mean_mm": float(np.nanmean(nn) * 1000),
            "cloud_nn_p95_mm": float(np.nanpercentile(nn, 95) * 1000)}


def _write_meta(out, args, rec_cfg, act_src, report, placements):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    tc = args.task_config.resolve()
    (out / "config.yaml").write_text(yaml.safe_dump({
        "task_name": args.task_name,
        "description": "sim twin of the paired real run: real actions replayed open-loop, home matched to the real "
                       "first frame, object at " + ("--object-xy" if args.object_xy is not None else "the real first-frame TCP xy"),
        "paired_source": str(args.real_run),
        "generator": "gentle_manip/scripts/replay_real_to_sim_paired.py",
        "git_commit": commit,
        "task_config": str(tc.relative_to(REPO)) if tc.is_relative_to(REPO) else str(args.task_config),
        "obs": rec_cfg["obs"], "action": rec_cfg[act_src],
        "placements": placements,
    }, sort_keys=False))
    (out / "match_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print(f"\nsaved {out/'data.pkl'} + config.yaml + match_report.yaml", flush=True)


def _figures(out, ep_idx, re_ee, re_quat, re_gw, re_pc, rec, stride):
    """Per-episode proprio-overlay figure + real|sim rolling cloud video."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    T = len(rec["ee_pos"])
    ts = np.arange(T)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for i, lbl in enumerate("xyz"):
        ax = axes[0][i]
        ax.plot(ts, re_ee[:T, i], label="real", lw=1.5)
        ax.plot(ts, rec["ee_pos"][:, i], "--", label="sim", lw=1.5)
        ax.set_title(f"ee_pos {lbl} (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[1][0].plot(ts, re_gw[:T], label="real", lw=1.5)
    axes[1][0].plot(ts, rec["gripper_width"][:, 0], "--", label="sim", lw=1.5)
    axes[1][0].set_title("gripper_width (m)"); axes[1][0].grid(alpha=0.3)
    axes[1][0].legend(fontsize=8)
    qa = _quat_angular_diff_deg(rec["ee_quat"], _align_quat_sign(rec["ee_quat"], re_quat[:T]))
    axes[1][1].plot(ts, qa, lw=1.5, color="tab:purple")
    axes[1][1].set_title("quat angular diff (deg)"); axes[1][1].grid(alpha=0.3)
    zm_r = np.array([_valid(re_pc[t])[:, 2].mean() if len(_valid(re_pc[t])) else np.nan
                     for t in range(T)])
    zm_s = np.array([_valid(rec["point_cloud"][t])[:, 2].mean() for t in range(T)])
    axes[1][2].plot(ts, zm_r, label="real", lw=1.5)
    axes[1][2].plot(ts, zm_s, "--", label="sim", lw=1.5)
    axes[1][2].set_title("cloud zmean (m)"); axes[1][2].grid(alpha=0.3)
    axes[1][2].legend(fontsize=8)
    fig.suptitle(f"paired replay — episode {ep_idx + 1}")
    fig.tight_layout()
    fig.savefig(out / f"ep_{ep_idx + 1:03d}_match.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # OVERLAY paired video: real (blue) + sim (red) in the SAME axes, 3 views side by side
    # (camera-side, low side, top-down) — the direct visual of the sim2real cloud gap.
    views = [("cam view", 30, -60), ("side view", 8, -150), ("top-down", 78, -90)]
    figv = plt.figure(figsize=(16.5, 5.6))
    axes3d = [figv.add_subplot(1, 3, c + 1, projection="3d") for c in range(3)]
    frames = []
    for t in range(0, T, stride):
        vr = _valid(re_pc[t])
        vs = _valid(rec["point_cloud"][t])
        for ax, (vname, elev, azim) in zip(axes3d, views):
            ax.clear()
            ax.scatter(vr[:, 0], vr[:, 1], vr[:, 2], s=2, c="tab:blue", alpha=0.45,
                       label=f"real ({len(vr)})")
            ax.scatter(vs[:, 0], vs[:, 1], vs[:, 2], s=2, c="tab:red", alpha=0.45,
                       label=f"sim ({len(vs)})")
            ax.set_xlim(0.2, 0.71); ax.set_ylim(-0.215, 0.215); ax.set_zlim(0, 0.45)
            ax.view_init(elev, azim)
            ax.set_title(f"{vname}  t={t}", fontsize=10)
            if ax is axes3d[0]:
                ax.legend(fontsize=8, loc="upper right")
        figv.suptitle(f"paired clouds OVERLAY (real=blue, sim=red) — episode {ep_idx + 1}")
        figv.canvas.draw()
        frames.append(np.asarray(figv.canvas.buffer_rgba())[..., :3].copy())
    plt.close(figv)
    vpath = out / f"ep_{ep_idx + 1:03d}_paired_cloud.mp4"
    imageio.mimsave(str(vpath), frames, fps=max(30 // stride, 1), macro_block_size=1)
    print(f"  saved {vpath}", flush=True)


if __name__ == "__main__":
    main()
