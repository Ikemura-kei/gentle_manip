"""Shared env-0 clip recording for eval venvs (dual imageio backend + per-episode numbering)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _even_dims(f: np.ndarray) -> np.ndarray:
    h, w = f.shape[:2]                       # h264 / PyAV need even dimensions
    return f[: h - (h % 2), : w - (w % 2)]


def write_clip(path, frames, fps: int = 30) -> int:
    if not frames:
        return 0
    import imageio.v2 as imageio
    Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
    fr = [_even_dims(np.asarray(f, np.uint8)) for f in frames]
    try:                                     # imageio-ffmpeg backend
        imageio.mimsave(str(path), fr, fps=fps, macro_block_size=1)
    except TypeError:                        # PyAV backend (envs/dppo)
        imageio.mimsave(str(path), fr, fps=fps, codec="libx264")
    print(f"  [eval] saved clip {path} ({len(fr)} frames)", flush=True)
    return len(fr)


class ClipRecorder:
    """Accumulates env-0 frames and writes ONE clip per episode (numbered), like the DPPO
    bridge: <video_path>, then <stem>_ep1<suffix>, ... for later episodes in the same session."""

    def __init__(self):
        self.path = None
        self.frames: list = []
        self.ep = 0

    def start(self, video_path) -> None:
        self.flush()
        self.path, self.ep, self.frames = video_path, 0, []

    def add(self, frame) -> None:
        if self.path is not None and frame is not None:
            self.frames.append(np.asarray(frame, np.uint8))

    def flush(self) -> None:
        if self.path and self.frames:
            p = Path(self.path)
            out = p if self.ep == 0 else p.with_name(f"{p.stem}_ep{self.ep}{p.suffix}")
            write_clip(out, self.frames)
            self.ep += 1
        self.frames = []


class MultiClipRecorder:
    """Per-trajectory recording: one clip PER ENV. Frames arrive as (N,H,W,3) each step; env j
    is written to paths[j] (None -> skip that env). This is what gives 'num videos = num episodes'
    — the harness passes a per-env video_path for every env of every recorded batch. Episodes
    within a session are numbered (<stem>, <stem>_ep1, ...) like ClipRecorder."""

    def __init__(self):
        self.paths = None            # list[str|None]
        self.frames = None           # list[list[frame]]
        self.ep = 0

    @property
    def active(self) -> bool:
        return bool(self.paths) and any(p for p in self.paths)

    def start(self, paths) -> None:
        self.flush()
        self.paths = list(paths) if paths else None
        self.frames = [[] for _ in self.paths] if self.paths else None
        self.ep = 0

    def add(self, frames_nhwc) -> None:
        if not self.active or frames_nhwc is None:
            return
        arr = np.asarray(frames_nhwc, np.uint8)
        if arr.ndim == 3:                        # a single (H,W,3) -> only env 0 available
            arr = arr[None]
        for j, p in enumerate(self.paths):
            if p and j < arr.shape[0]:
                self.frames[j].append(arr[j])

    def flush(self) -> None:
        if self.active:
            for j, p in enumerate(self.paths):
                if p and self.frames[j]:
                    write_clip(Path(p), self.frames[j])
        # Deactivate after writing: record ONLY the episode between this start() and its flush.
        # Critical for the finetune-render case — the DPPO agent calls reset_arg (-> start) only on
        # EVAL iterations, so without deactivating, recording bled into the ~24 following TRAIN
        # iterations (mislabeled _ep1.._ep24, and rendering every train step = big slowdown).
        # Per-episode consumers (harness / SimEvalVenv / eval bridge) call start() again next episode.
        self.paths = None
        self.frames = None
        self.ep = 0
