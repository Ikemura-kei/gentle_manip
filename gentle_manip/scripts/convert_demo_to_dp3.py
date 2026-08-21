from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# gentle_manip isn't a package in envs/dp3 — put the repo root on the path so `--derive-action`
# can import gentle_manip.actions (this script otherwise has no gentle_manip dependency).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


DEFAULT_AGENT_POS_KEYS = ("ee_pos", "ee_quat", "gripper_width")


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read pickles written by numpy 2.x in a numpy 1.x env (and vice-versa).

    Sim demos are pickled in envs/sim (numpy 2.x, which uses the ``numpy._core``
    module path), but conversion runs in envs/dp3 (numpy 1.x, ``numpy.core``). Map
    the module path so array reconstruction resolves in either numpy.
    """

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def _iter_pickles(inputs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in inputs:
        path = path.expanduser()
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.pkl")))
        else:
            paths.append(path)
    if not paths:
        raise ValueError("no pickle files found")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"input pickle(s) not found: {missing}")
    return paths


def _parse_key_mapping(values: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        src, sep, dst = value.partition(":")
        src = src.strip()
        dst = dst.strip() if sep else src
        if not src or not dst:
            raise ValueError(f"invalid key mapping: {value!r}; expected key or key:output_key")
        mapping[src] = dst
    return mapping


def _as_2d_state_part(value: np.ndarray, key: str, T: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[0] != T:
        raise ValueError(f"observation {key!r} has {arr.shape[0]} steps, expected {T}")
    return arr.reshape(T, -1)


def episode_to_dp3_arrays(
    episode: dict,
    *,
    agent_pos_keys: Sequence[str] = DEFAULT_AGENT_POS_KEYS,
    point_cloud_key: str = "point_cloud",
    image_key_map: dict[str, str] | None = None,
    derive_action_config=None,
    derive_lookahead: int = 1,
) -> dict[str, np.ndarray]:
    """Convert one gentle_manip episode to DP3 zarr arrays.

    DP3 dataset classes in this checkout expect proprioception under the zarr
    key ``state`` and expose it to the policy as ``obs.agent_pos``.

    derive_action_config: if set, DERIVE the action (delta / absolute) from the recorded EE-pose
    trajectory instead of using episode["actions"] — so one delta-teleop collection yields both a
    delta and an absolute DP3 dataset (identical derivation as the DPPO convert_demos path).
    """
    observations = episode["observations"]
    if derive_action_config is not None:
        from gentle_manip.actions.derive import derive_action_set
        actions = derive_action_set(episode, derive_action_config,
                                    lookahead=derive_lookahead).astype(np.float32)
    else:
        actions = np.asarray(episode["actions"], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be rank-2 (T, Da), got {actions.shape}")
    T = actions.shape[0]

    missing = [k for k in agent_pos_keys if k not in observations]
    if missing:
        raise KeyError(f"missing proprioceptive observation key(s): {missing}")
    if point_cloud_key not in observations:
        raise KeyError(f"missing point cloud observation key: {point_cloud_key!r}")

    state_parts = [_as_2d_state_part(observations[k], k, T) for k in agent_pos_keys]
    out = {
        "state": np.concatenate(state_parts, axis=-1).astype(np.float32, copy=False),
        "action": actions,
        "point_cloud": np.asarray(observations[point_cloud_key], dtype=np.float32),
    }
    if out["point_cloud"].shape[0] != T:
        raise ValueError(
            f"point cloud has {out['point_cloud'].shape[0]} steps, expected {T}"
        )

    for src_key, dst_key in (image_key_map or {}).items():
        if src_key not in observations:
            raise KeyError(f"missing image observation key: {src_key!r}")
        image = np.asarray(observations[src_key])
        if image.shape[0] != T:
            raise ValueError(f"image {src_key!r} has {image.shape[0]} steps, expected {T}")
        out[dst_key] = image
    return out


def _load_episodes(paths: Sequence[Path]) -> tuple[list[dict], dict]:
    episodes: list[dict] = []
    metas: list[dict] = []
    for path in paths:
        with open(path, "rb") as f:
            data = _NumpyCompatUnpickler(f).load()
        if not isinstance(data, dict) or "episodes" not in data:
            raise ValueError(f"{path} is not a gentle_manip demo pickle")
        episodes.extend(data["episodes"])
        metas.append(data.get("meta", {}))
    if not episodes:
        raise ValueError("input contains no saved episodes")
    task_names = sorted({m.get("task", "") for m in metas if m.get("task")})
    meta = {
        "source_files": [str(p) for p in paths],
        "source_tasks": task_names,
    }
    return episodes, meta


def _chunk_for(arr: np.ndarray, chunk_length: int) -> tuple[int, ...]:
    return (min(chunk_length, arr.shape[0]),) + arr.shape[1:]


def write_dp3_zarr(
    arrays: dict[str, np.ndarray],
    episode_ends: np.ndarray,
    output: Path,
    *,
    overwrite: bool = False,
    chunk_length: int = 100,
    attrs: dict | None = None,
) -> None:
    try:
        import numcodecs
        import zarr
    except ImportError as exc:
        raise ImportError(
            "Writing DP3 zarr datasets requires zarr and numcodecs. "
            "Install them in the environment used for conversion."
        ) from exc

    output = output.expanduser()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.group(store=zarr.DirectoryStore(str(output)))
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    for key, arr in arrays.items():
        data_group.create_dataset(
            key,
            data=arr,
            chunks=_chunk_for(arr, chunk_length),
            dtype=arr.dtype,
            compressor=compressor,
            overwrite=True,
        )
    meta_group.create_dataset(
        "episode_ends",
        data=episode_ends.astype(np.int64, copy=False),
        chunks=(min(chunk_length, len(episode_ends)),),
        dtype=np.int64,
        compressor=compressor,
        overwrite=True,
    )
    if attrs:
        root.attrs.update(attrs)


def convert_pickles_to_dp3(
    inputs: Sequence[Path],
    output: Path,
    *,
    agent_pos_keys: Sequence[str] = DEFAULT_AGENT_POS_KEYS,
    point_cloud_key: str = "point_cloud",
    image_key_map: dict[str, str] | None = None,
    overwrite: bool = False,
    chunk_length: int = 100,
    derive_action_config=None,
    derive_lookahead: int = 1,
) -> dict[str, tuple[tuple[int, ...], str]]:
    paths = _iter_pickles(inputs)
    episodes, source_meta = _load_episodes(paths)

    per_episode = [
        episode_to_dp3_arrays(
            ep,
            agent_pos_keys=agent_pos_keys,
            point_cloud_key=point_cloud_key,
            image_key_map=image_key_map,
            derive_action_config=derive_action_config,
            derive_lookahead=derive_lookahead,
        )
        for ep in episodes
    ]
    keys = list(per_episode[0].keys())
    for ep_arrays in per_episode:
        if list(ep_arrays.keys()) != keys:
            raise ValueError("all episodes must produce the same output keys")

    episode_lengths = [ep_arrays["action"].shape[0] for ep_arrays in per_episode]
    arrays = {
        key: np.concatenate([ep_arrays[key] for ep_arrays in per_episode], axis=0)
        for key in keys
    }
    episode_ends = np.cumsum(episode_lengths, dtype=np.int64)
    attrs = {
        **source_meta,
        "agent_pos_keys": list(agent_pos_keys),
        "point_cloud_key": point_cloud_key,
        "image_key_map": image_key_map or {},
    }
    write_dp3_zarr(
        arrays,
        episode_ends,
        output,
        overwrite=overwrite,
        chunk_length=chunk_length,
        attrs=attrs,
    )
    summary = {key: (arr.shape, str(arr.dtype)) for key, arr in arrays.items()}
    summary["episode_ends"] = (episode_ends.shape, str(episode_ends.dtype))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert gentle_manip demo pickle(s) to DP3 zarr format."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="demo pickle(s) or directories")
    parser.add_argument("-o", "--output", required=True, type=Path, help="output .zarr path")
    parser.add_argument(
        "--agent-pos-keys",
        nargs="+",
        default=list(DEFAULT_AGENT_POS_KEYS),
        help="observation keys concatenated into DP3 state/agent_pos",
    )
    parser.add_argument("--point-cloud-key", default="point_cloud")
    parser.add_argument(
        "--image-key",
        action="append",
        default=[],
        help="optional observation image key to copy; use key:output_key to rename",
    )
    parser.add_argument("--chunk-length", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--derive-lookahead", type=int, default=1,
                        help="ABSOLUTE --derive-action only: target = pose this many steps "
                             "ahead (default 1; use ~4 -- see actions.derive docstring: K=1 "
                             "stalls a BC absolute policy at a closed-loop fixed point)")
    parser.add_argument("--derive-action", type=Path, default=None,
                        help="DERIVE the action from the recorded EE-pose trajectory using this "
                             "action config (delta / abs_pose_abs_gripper / abs_pose_euler_abs_gripper) "
                             "instead of the stored actions — one delta collection -> both a delta and "
                             "an absolute DP3 dataset (same derivation as the DPPO convert_demos path).")
    args = parser.parse_args()

    derive_cfg = None
    if args.derive_action is not None:
        import yaml
        from gentle_manip.actions.action_config import ActionConfig
        p = args.derive_action if args.derive_action.is_file() else (_REPO / args.derive_action)
        derive_cfg = ActionConfig.from_dict(yaml.safe_load(open(p)))
        print(f"  deriving actions via {p.name} (mode={derive_cfg.mode}, "
              f"rot_repr={getattr(derive_cfg,'rot_repr','-')}, action_dim={derive_cfg.action_dim})")

    summary = convert_pickles_to_dp3(
        args.inputs,
        args.output,
        agent_pos_keys=args.agent_pos_keys,
        point_cloud_key=args.point_cloud_key,
        image_key_map=_parse_key_mapping(args.image_key),
        overwrite=args.overwrite,
        chunk_length=args.chunk_length,
        derive_action_config=derive_cfg,
        derive_lookahead=args.derive_lookahead,
    )
    print(f"wrote {args.output}")
    for key, (shape, dtype) in summary.items():
        print(f"  {key}: shape={shape} dtype={dtype}")


if __name__ == "__main__":
    main()
