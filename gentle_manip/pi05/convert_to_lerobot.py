"""Convert our grasp-synthesis demos -> a LeRobot dataset that openpi can train on UNCHANGED.

DESIGN CONSTRAINT (user, 2026-08-30): "no internal code needs to be changed, only config
modification, and maybe mild adaptation for our evaluation env". So this script makes OUR data
look like what an EXISTING openpi config already expects, rather than adding a config/transform
to their tree. Concretely we emit the exact feature names `LeRobotLiberoDataConfig` repacks:

    image | wrist_image | state | actions   (+ LeRobot's own `task` -> the prompt)

so `pi05_libero` can be reused with CLI overrides only (--data.repo_id, batch size, steps, ...).
`LiberoInputs` passes state/actions through WITHOUT any hardcoded dimension, so our 8-dim state
and 7-dim action work as-is; openpi pads them to the model dim. (`LiberoOutputs` DOES slice to 7
at inference -- harmless for us, since our action space is 7-dim. Verified, not assumed.)

ACTIONS -- 7-dim euler absolute, derived, NOT the recorded 10-dim.
The collector always records 10-dim rot6d (`_invert_actions_absolute` hardcodes it); every demo
set on disk is `action_dim: 10`. The 7-dim euler every policy trains on is DERIVED at conversion.
We mirror the generalist's exact recipe (`.agent_tmp/build_3obj_generalist.sh`):
    --derive-action        abs_pose_euler_abs_gripper.yaml   (target; carries euler_frame_offset_deg)
    --derive-source-action abs_pose_abs_gripper.yaml         (source: the recorded commands)
with lookahead=1. The frame offset is NOT optional: without it a top-down grasp's roll sits on the
+/-pi seam, sign-flips between frames, trains to a low loss and decodes ~180 deg wrong (run oppsu).

CAMERAS -- two variants from ONE dataset, because the comparison must hold the data fixed:
    --cameras ext_wrist : image=cam_ext, wrist_image=cam_wrist
    --cameras ext       : image=cam_ext, wrist_image=ZEROS
The zeros are openpi's own documented idiom for a missing camera (they do exactly this for
`right_wrist_0_rgb`). Note honestly what it is: `LiberoInputs` hardcodes the left-wrist mask to
True, so the ext-only variant still spends image tokens on a blank frame rather than masking it
off. That is a fair "no wrist information" ablation, not a free lunch -- state it in the writeup.
The real rig has NO wrist camera, so `ext` is the deployable variant and `ext_wrist` is a sim-only
upper bound.

IMAGES are stored at 224x224 via openpi's OWN `image_tools.resize_with_pad`, i.e. the identical
letterbox their model transform would apply at train time -- so this loses nothing and keeps the
LeRobot dataset ~6x smaller than storing 640x480. Full-resolution frames remain in the source
data.pkl.
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np

# The four phrasings the user specified. Assigned per EPISODE (deterministically, by index) so
# every instruction is paired with a spread of scenes rather than correlated with one region of
# the collection. "20 centimeters" is honest: the success band is an absolute object height of
# 0.175-0.275 m and demos sit at ~0.209 m, i.e. ~19 cm above the ~0.02 m rest height.
# PER-OBJECT, and that matters: this dataset mixes objects, so labelling a tofu episode
# "pick up the mushroom" would teach the model that the instruction is NOISE — the opposite of what
# a language-conditioned policy should learn, and it would silently destroy the only signal that
# makes the task language-conditioned at all.
# SIM phrasings (kept for the sim datasets already built with them).
_PHRASINGS = [
    "pick up the {obj}",
    "pick the {obj} up",
    "lift the {obj}",
    "lift the {obj} up for 20 centimeters",   # ~19 cm rise: band 0.175-0.275 m from ~0.02 m rest
]

# REAL phrasings (user, 2026-09-02) -- used with --phrasings real.
# The last TWO say the literal word "object" instead of naming it. They are IN TRAINING on
# purpose (~1/3 of episodes by round-robin): they teach the policy to act on an UNNAMED object,
# which is what makes a generic-prompt evaluation possible -- ask for "the object" and see whether
# it still grasps, including on an object whose name it was never given.
_PHRASINGS_REAL = [
    "lift the {obj} up gently",
    "pick up the {obj} gently",
    "pick the {obj} up from the table carefully",
    "lift the {obj} up from the table carefully",
    "lift the object up from table gently",
    "pick up the object from table gently",
]
_ACTIVE_PHRASINGS = _PHRASINGS
INSTRUCTIONS = [p.format(obj="mushroom") for p in _PHRASINGS]   # back-compat for single-object use


def instructions_for(object_name: str) -> list:
    """The four phrasings, bound to this object's name."""
    return [p.format(obj=object_name) for p in _ACTIVE_PHRASINGS]


def _load_episodes(src: Path) -> list:
    """Accept a data.pkl, a directory holding one, or a directory of shard_*.pkl.

    Shards matter: they are what exists WHILE a collection is still running (the merge into
    data.pkl happens at the end and deletes them), so this is what makes an early smoke possible.
    """
    if src.is_file():
        paths = [src]
    else:
        paths = sorted(src.glob("data.pkl")) or sorted(src.glob("shard_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no data.pkl or shard_*.pkl under {src}")
    eps = []
    for p in paths:
        eps.extend(pickle.load(open(p, "rb"))["episodes"])
    print(f"loaded {len(eps)} episodes from {[p.name for p in paths]}")
    return eps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, nargs="+",
                    help="one or more demo data.pkl / run dirs. Several sources are merged into ONE "
                         "LeRobot dataset (that is how the mixed-object low-data set is built).")
    ap.add_argument("--object-names", nargs="+", default=None,
                    help="object name per src, in order — drives the language instruction. "
                         "Inferred from each run dir's config.yaml when omitted.")
    ap.add_argument("--repo-id", required=True, help="LeRobot dataset id, e.g. gm/mushroom_pi05_ext")
    ap.add_argument("--cameras", choices=["ext", "ext_wrist"], required=True,
                    help="ext = external only (wrist_image zero-filled); ext_wrist = both")
    ap.add_argument("--max-episodes", type=int, default=None, help="cap, for smoke runs")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--root", type=Path, default=None,
                    help="output root (default: $HF_LEROBOT_HOME)")
    # Resolve against the REPO ROOT, not the cwd: this script is run from third_party/openpi
    # (openpi's uv project), so a repo-relative default would not exist there.
    _REPO = Path(__file__).resolve().parents[2]
    ap.add_argument("--action-config", type=Path,
                    default=_REPO / "gentle_manip/configs/action/abs_pose_euler_abs_gripper.yaml")
    ap.add_argument("--source-action-config", type=Path,
                    default=_REPO / "gentle_manip/configs/action/abs_pose_abs_gripper.yaml")
    ap.add_argument("--lookahead", type=int, default=1)
    ap.add_argument("--phrasings", choices=["sim", "real"], default="sim",
                    help="sim = the 4 original phrasings; real = the 6 gentleness phrasings, the "
                         "last two of which say the literal word 'object' (generic-prompt training)")
    args = ap.parse_args()

    global _ACTIVE_PHRASINGS
    _ACTIVE_PHRASINGS = _PHRASINGS_REAL if args.phrasings == "real" else _PHRASINGS

    import yaml
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    from openpi.shared import image_tools

    from gentle_manip.actions.action_config import ActionConfig
    from gentle_manip.actions.derive import derive_action_set
    from gentle_manip.utils.image_codec import decode_images

    tgt_cfg = ActionConfig.from_dict(yaml.safe_load(args.action_config.read_text()))
    src_cfg = ActionConfig.from_dict(yaml.safe_load(args.source_action_config.read_text()))
    # target action width: 7 for abs_pose_euler_abs_gripper (3 pos + 3 euler + 1 gripper)
    tgt_dim = 7 if getattr(tgt_cfg, "rot_repr", "euler") == "euler" else 10
    assert tgt_cfg.mode == "absolute" and tgt_cfg.rot_repr == "euler", "target must be 7d euler abs"
    assert tuple(tgt_cfg.euler_frame_offset_deg) == (180.0, 0.0, 0.0), (
        "euler_frame_offset_deg must be [180,0,0] -- without it a top-down grasp's roll sits on "
        "the +/-pi seam and decodes ~180 deg wrong (run oppsu)")

    import yaml as _yaml
    srcs = list(args.src)
    names = args.object_names or []
    if len(names) != len(srcs):                       # infer from each run's config snapshot
        names = []
        for sp in srcs:
            d = sp if sp.is_dir() else sp.parent
            cfg = d / "config.yaml"
            obj = "object"
            if cfg.exists():
                c = _yaml.safe_load(cfg.read_text()) or {}
                obj = (c.get("task", {}) or {}).get("object_name") or c.get("object_name") or obj
                if obj == "object":                   # fall back to the experiment/task name
                    t = str(c.get("experiment", "")) + str(c.get("task_name", ""))
                    for cand in ("mushroom", "tofu", "raspberry", "tomato", "strawberry", "banana"):
                        if cand in t: obj = cand; break
            names.append(obj)
    episodes, ep_obj = [], []
    for sp, nm in zip(srcs, names):
        e = _load_episodes(sp)
        if args.max_episodes:
            e = e[: args.max_episodes]
        episodes += e; ep_obj += [nm] * len(e)
        print(f"  {sp}  -> {len(e)} episodes, object={nm!r}")
    print(f"total: {len(episodes)} episodes across {len(srcs)} source(s)")

    out_root = args.root or HF_LEROBOT_HOME
    out_path = Path(out_root) / args.repo_id
    if out_path.exists():
        print(f"removing existing {out_path}")
        shutil.rmtree(out_path)

    S = args.image_size
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, root=args.root, robot_type="xarm7", fps=args.fps,
        features={
            "image":       {"dtype": "image", "shape": (S, S, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (S, S, 3), "names": ["height", "width", "channel"]},
            "state":       {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions":     {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10, image_writer_processes=5,
    )

    n_frames = 0
    for ei, ep in enumerate(episodes):
        obs = decode_images(ep["observations"])
        T = len(ep["actions"])
        # 7-dim euler absolute, derived from the RECORDED 10-dim commands (see module docstring)
        # REAL demos already record 7-dim euler-absolute actions; SIM demos record 10-dim rot6d
        # and must be DERIVED. Deriving an already-7-dim episode indexes [:, 9] of a 7-wide array
        # -> "IndexError: index 9 is out of bounds for axis 1 with size 7". Detect and pass through.
        _raw = np.asarray(ep["actions"], np.float32)
        if _raw.ndim == 2 and _raw.shape[1] == tgt_dim:
            act = _raw                                   # already in the target space
        else:
            act = derive_action_set(ep, tgt_cfg, lookahead=args.lookahead, source_config=src_cfg)
        assert act.shape == (T, 7), f"expected (T,7) actions, got {act.shape}"
        state = np.concatenate([
            np.asarray(obs["ee_pos"], np.float32).reshape(T, 3),
            np.asarray(obs["ee_quat"], np.float32).reshape(T, 4),
            np.asarray(obs["gripper_width"], np.float32).reshape(T, 1),
        ], axis=1)
        ext = np.asarray(obs["image_cam_ext"])
        wrist = np.asarray(obs["image_cam_wrist"]) if args.cameras == "ext_wrist" else None
        phr = instructions_for(ep_obj[ei])
        instruction = phr[ei % len(phr)]
        for t in range(T):
            base = image_tools.resize_with_pad(ext[t], S, S)
            ds.add_frame({
                "image": np.asarray(base, np.uint8),
                "wrist_image": (np.asarray(image_tools.resize_with_pad(wrist[t], S, S), np.uint8)
                                if wrist is not None else np.zeros((S, S, 3), np.uint8)),
                "state": state[t].astype(np.float32),
                "actions": act[t].astype(np.float32),
                "task": instruction,
            })
            n_frames += 1
        ds.save_episode()
        if (ei + 1) % 10 == 0 or ei == len(episodes) - 1:
            print(f"  {ei+1}/{len(episodes)} episodes, {n_frames} frames")

    print(f"\nDONE  {len(episodes)} episodes / {n_frames} frames -> {out_path}")
    print(f"cameras={args.cameras}  (wrist_image is "
          f"{'the real wrist view' if args.cameras=='ext_wrist' else 'ZEROS -- no wrist information'})")
    from collections import Counter
    print("instructions:", sorted({i for n in set(ep_obj) for i in instructions_for(n)}))
    print("episodes per object:", dict(Counter(ep_obj)))


if __name__ == "__main__":
    main()
