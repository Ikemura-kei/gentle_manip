"""Stream gentle_manip tactile demo pickle(s) into a DP3-compatible zarr.

Unlike convert_demo_to_dp3.py's convert_pickles_to_dp3() (which loads every
input pickle fully into memory and then np.concatenate()s per-key across all
episodes before writing), this script processes one input file at a time and,
within it, one episode at a time — appending directly into resizable zarr
arrays. That bounds peak memory to roughly one source pickle's size instead of
~2x the combined dataset size, which matters here: the largest tactile demo
pickle (26-07-07-lzw.pkl) is ~12GB on disk.

GelSight tactile frames are also transformed on the way in, per-episode:
  1. resized from (480, 640, 3) to (tactile_size, tactile_size, 3) via
     torch.nn.functional.interpolate (area mode — matches cv2.INTER_AREA
     behavior for downsampling, avoids adding a new image-lib dependency).
  2. converted to a delta image relative to that episode's own frame 0
     (assumed pre-contact / undeformed gel), stored as int16.
This keeps the zarr small (~1000x smaller than raw 640x480 uint8 pairs) and
keeps dataset.py simple (no resize/delta logic needed at load time).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from gentle_manip.scripts.convert_demo_to_dp3 import (
    DEFAULT_AGENT_POS_KEYS,
    _iter_pickles,
    _NumpyCompatUnpickler,
    episode_to_dp3_arrays,
)

TACTILE_IMAGE_KEY_MAP = {
    "tactile_tactile_left": "tactile_left_raw",
    "tactile_tactile_right": "tactile_right_raw",
}
TACTILE_OUTPUT_KEYS = ("tactile_left_delta", "tactile_right_delta")


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, size, size, 3) uint8, area-downsampled."""
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # T,3,H,W
    t = F.interpolate(t, size=(size, size), mode="area")
    return t.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy()


def _delta_from_first_frame(frames: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, H, W, 3) int16, relative to frame 0."""
    ref = frames[0:1].astype(np.int16)
    return frames.astype(np.int16) - ref


def episode_to_tactile_arrays(episode: dict, *, tactile_size: int) -> dict[str, np.ndarray]:
    """One episode -> {state, action, point_cloud, tactile_left_delta, tactile_right_delta}."""
    arrays = episode_to_dp3_arrays(
        episode,
        agent_pos_keys=DEFAULT_AGENT_POS_KEYS,
        point_cloud_key="point_cloud",
        image_key_map=TACTILE_IMAGE_KEY_MAP,
    )
    out = {
        "state": arrays["state"],
        "action": arrays["action"],
        "point_cloud": arrays["point_cloud"],
    }
    for raw_key, out_key in zip(
        ("tactile_left_raw", "tactile_right_raw"), TACTILE_OUTPUT_KEYS
    ):
        resized = _resize_frames(arrays[raw_key], tactile_size)
        out[out_key] = _delta_from_first_frame(resized)
    return out


def _chunk_for(arr: np.ndarray, chunk_length: int) -> tuple[int, ...]:
    return (min(chunk_length, max(arr.shape[0], 1)),) + arr.shape[1:]


def convert_pickles_to_tactile_zarr(
    inputs: Sequence[Path],
    output: Path,
    *,
    tactile_size: int = 128,
    overwrite: bool = False,
    chunk_length: int = 100,
) -> dict[str, tuple[tuple[int, ...], str]]:
    import numcodecs
    import shutil
    import zarr

    paths = _iter_pickles(inputs)

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

    zarr_arrays: dict[str, "zarr.Array"] = {}
    episode_ends: list[int] = []
    running_total = 0
    source_tasks: set[str] = set()

    for path in paths:
        print(f"[convert] loading {path} ...", flush=True)
        with open(path, "rb") as f:
            file_data = _NumpyCompatUnpickler(f).load()
        if not isinstance(file_data, dict) or "episodes" not in file_data:
            raise ValueError(f"{path} is not a gentle_manip demo pickle")
        task = file_data.get("meta", {}).get("task")
        if task:
            source_tasks.add(task)
        episodes = file_data["episodes"]
        print(f"[convert] {path.name}: {len(episodes)} episodes loaded, converting ...", flush=True)

        for i, episode in enumerate(episodes):
            ep_arrays = episode_to_tactile_arrays(episode, tactile_size=tactile_size)
            T = ep_arrays["action"].shape[0]
            for key, arr in ep_arrays.items():
                if key not in zarr_arrays:
                    zarr_arrays[key] = data_group.create_dataset(
                        key,
                        shape=(0,) + arr.shape[1:],
                        chunks=_chunk_for(arr, chunk_length),
                        dtype=arr.dtype,
                        compressor=compressor,
                    )
                zarr_arrays[key].append(arr, axis=0)
            running_total += T
            episode_ends.append(running_total)
            print(
                f"[convert] {path.name} episode {i+1}/{len(episodes)}: T={T}, "
                f"total_steps={running_total}",
                flush=True,
            )
            del ep_arrays

        del episodes, file_data

    if not episode_ends:
        raise ValueError("input contains no saved episodes")

    meta_group.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        chunks=(min(chunk_length, len(episode_ends)),),
        dtype=np.int64,
        compressor=compressor,
    )
    root.attrs.update(
        {
            "source_files": [str(p) for p in paths],
            "source_tasks": sorted(source_tasks),
            "tactile_size": tactile_size,
            "tactile_delta_reference": "episode frame 0",
            "agent_pos_keys": list(DEFAULT_AGENT_POS_KEYS),
        }
    )

    summary = {key: (arr.shape, str(arr.dtype)) for key, arr in zarr_arrays.items()}
    summary["episode_ends"] = ((len(episode_ends),), "int64")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream gentle_manip tactile demo pickle(s) into a DP3-compatible zarr "
        "(point_cloud, state, action, tactile_{left,right}_delta)."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="demo pickle(s) or directories")
    parser.add_argument("-o", "--output", required=True, type=Path, help="output .zarr path")
    parser.add_argument("--tactile-size", type=int, default=128)
    parser.add_argument("--chunk-length", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = convert_pickles_to_tactile_zarr(
        args.inputs,
        args.output,
        tactile_size=args.tactile_size,
        overwrite=args.overwrite,
        chunk_length=args.chunk_length,
    )
    print(f"wrote {args.output}")
    for key, (shape, dtype) in summary.items():
        print(f"  {key}: shape={shape} dtype={dtype}")


if __name__ == "__main__":
    main()
