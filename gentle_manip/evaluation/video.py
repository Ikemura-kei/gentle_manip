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
