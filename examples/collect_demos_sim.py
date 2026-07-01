"""Collect teleop demonstrations IN SIM (keyboard) with the Genesis viewer.

Config-driven: one YAML (configs/collect/*.yaml) holds every setting and references
the obs / action / dr / augmentation sub-configs by name. The run is fully
reproducible and self-describing — each session writes to

    <out_dir>/<task_name>/<YY-MM-DD-abc>/
        data.pkl              # the recorded (obs, action) episodes (DP3 schema)
        config.yaml           # verbatim copy of the collection config used
        config_resolved.yaml  # the same, with every sub-config inlined

It's the sim counterpart of `python -m gentle_manip.demos.record` (real-only):
same KeyboardTeleop + DemoRecorder, so the episode schema is identical. The
inherent ee_quat noise (obs config) and reset DR (cube + arm-pose jitter) are
active, so the demos carry the robustness variation.

Two windows open: a small pygame window (focus it for keys) and the Genesis
viewer. Needs a display.

    MUJOCO_GL=glfw uv run --project envs/sim python examples/collect_demos_sim.py \
        --config gentle_manip/configs/collect/red_cube_sim.yaml

Controls: W/S A/D Up/Dn move, L/R R/F Q/E rotate, O/P grip, SPACE save,
BACKSPACE discard, ESC quit.
"""
import os

if os.environ.get("MUJOCO_GL") not in {"glfw", "egl", "osmesa"}:
    os.environ["MUJOCO_GL"] = "glfw"

import argparse
import random
import string
from datetime import datetime
from pathlib import Path

import yaml

_PKG = Path(__file__).resolve().parents[1] / "gentle_manip"
_CFG = _PKG / "configs"
_DEFAULT_CONFIG = _CFG / "collect" / "red_cube_sim.yaml"


def _load_named(subdir: str, name) -> dict:
    """Load configs/<subdir>/<name>.yaml as a dict (None/'' -> {})."""
    if not name:
        return {}
    return yaml.safe_load((_CFG / subdir / f"{name}.yaml").read_text()) or {}


def _make_run_dir(base: Path) -> Path:
    """Create <base>/<YY-MM-DD-abc>/ with a unique random suffix."""
    date = datetime.now().strftime("%y-%m-%d")
    for _ in range(10000):
        abc = "".join(random.choices(string.ascii_lowercase, k=3))
        run_dir = base / f"{date}-{abc}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            return run_dir
    raise RuntimeError(f"no free run-dir name under {base}")


def main() -> None:
    p = argparse.ArgumentParser(description="Collect teleop demos in sim (config-driven).")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                   help="collection config (configs/collect/*.yaml)")
    args = p.parse_args()

    cfg_text = args.config.read_text()
    cfg = yaml.safe_load(cfg_text)
    use_exp = bool(cfg.get("experiment"))   # experiment mode = single source of truth

    # Deferred imports so --help is cheap and doesn't build genesis.
    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.demos.record import DemoRecorder
    from gentle_manip.demos.teleop_keyboard import KeyboardTeleop
    from gentle_manip.envs.policy_env import PolicyEnv
    from gentle_manip.envs.sim_backend import SimBackend
    from gentle_manip.perception.augmentation import AugmentationConfig
    from gentle_manip.perception.obs_config import ObsConfig
    from gentle_manip.tasks.single_lift import SingleLiftTask

    if use_exp:
        # Everything (obs=SUPERSET, task WITH reward, action, dr, aug) from the one
        # experiment config, so a demo carries every modality AND the per-step reward.
        from gentle_manip.experiment import Experiment
        exp = Experiment.load(cfg["experiment"])
        obs_config = exp.collection_obs()
        action_config = exp.action_config
        dr_d = exp.dr
        aug = exp.augmentation_config()
        task = SingleLiftTask(exp.task_cfg)          # full reward -> logged per step
        task_name = cfg.get("task_name", exp.name)
        resolved = {**cfg, "experiment_obs_keys": obs_config.obs_keys()}
    else:
        obs_d = _load_named("obs", cfg["obs_config"])
        act_d = _load_named("action", cfg["action_config"])
        dr_d = _load_named("dr", cfg.get("dr"))
        aug_d = _load_named("augmentation", cfg.get("augmentation"))
        obs_config = ObsConfig.from_dict(obs_d)
        action_config = ActionConfig.from_dict(act_d)
        aug = AugmentationConfig.from_dict(aug_d) if aug_d else None
        task = SingleLiftTask({"object_name": cfg["object"], "object_type": cfg["object_type"]})
        task_name = cfg["task_name"]
        resolved = {**cfg,
                    "obs_config": {"_name": cfg["obs_config"], **obs_d},
                    "action_config": {"_name": cfg["action_config"], **act_d},
                    "dr": ({"_name": cfg["dr"], **dr_d} if cfg.get("dr") else None),
                    "augmentation": ({"_name": cfg["augmentation"], **aug_d} if cfg.get("augmentation") else None)}

    # Create the run dir and snapshot the config (verbatim + resolved).
    run_dir = _make_run_dir(Path(cfg["out_dir"]) / task_name)
    (run_dir / "config.yaml").write_text(cfg_text)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    # "keyboard" (human teleop, viewer) or "scripted" (automatic, headless by default).
    mode = cfg.get("input", "keyboard")
    show_viewer = cfg.get("show_viewer", mode == "keyboard")

    backend = SimBackend(
        task.scene_spec, num_envs=1, use_subprocess=False, show_viewer=show_viewer,
        config={"sim": {"settle_steps": cfg["settle_steps"]}, "dr": dr_d},
    )
    # task ON in experiment mode -> the demo's per-step reward is real (for RLPD);
    # legacy mode keeps task=None (reward 0, DP3-only).
    env = PolicyEnv(backend, obs_config, action_config, task=(task if use_exp else None),
                    max_episode_steps=10 ** 9, augmentation=aug)

    if mode == "scripted":
        from gentle_manip.demos.scripted_policy import ScriptedLiftDemonstrator
        sc = cfg.get("scripted", {})
        driver = ScriptedLiftDemonstrator(
            backend, action_config.scales, n_episodes=cfg["n_episodes"], rate_hz=cfg["rate"],
            lift_height=sc.get("lift_height", 0.2), hold_seconds=sc.get("hold_seconds", 2.0),
            approach_height=sc.get("approach_height", 0.12), grasp_z=sc.get("grasp_z", 0.006),
            grasp_gw=sc.get("grasp_gw", 0.030), grasp_firm_steps=sc.get("grasp_firm_steps", 1),
            gripper_close=cfg["gripper_value"], speed_cap=cfg["speed"],
        )
        controls = f"scripted x{cfg['n_episodes']} (auto save/discard/quit)"
    else:
        driver = KeyboardTeleop(move_speed=cfg["speed"], rot_speed=cfg["speed"],
                                gripper_value=cfg["gripper_value"])
        controls = "W/S A/D Up/Dn move, L/R R/F Q/E rotate, O/P grip, SPACE save, BACKSPACE discard, ESC quit."

    # Optional: record the external camera (cam_ext) RGB for the first N episodes as a
    # quality-check video, written to <run_dir>/videos/ep_NNN.mp4.
    frame_fn = video_dir = None
    if cfg.get("record_video"):
        from gentle_manip.robot.xarm7_sim import _np
        cam = next(iter(backend.process.handle.cameras.values()))[0]   # in-process worker
        frame_fn = lambda: _np(cam.render(rgb=True, depth=False)[0])
        video_dir = run_dir

    recorder = DemoRecorder(
        env=env, teleop=driver, keyboard=driver, task_name=task_name,
        out_dir=run_dir, rate_hz=cfg["rate"], dataset_path=run_dir / "data.pkl",
        idle_threshold=cfg["idle_threshold"], keep_trailing_idle=cfg["keep_trailing_idle"],
        max_interior_idle=cfg["max_interior_idle"],
        action_noise_std=cfg.get("action_noise_std", 0.0),
        frame_fn=frame_fn, video_dir=video_dir, video_fps=cfg.get("video_fps", cfg["rate"]),
        video_episodes=cfg.get("video_episodes", 0),
    )
    src = f"experiment={cfg['experiment']}" if use_exp else f"obs={cfg.get('obs_config')}"
    print(f"collecting '{task_name}' in sim ({mode}) -> {run_dir}\n"
          f"  {src}  obs_keys={obs_config.obs_keys()}  reward={'on' if use_exp else 'off'}\n"
          f"  {controls}")
    recorder.run()


if __name__ == "__main__":
    main()
