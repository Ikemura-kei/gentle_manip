"""Async GelSight Mini reader (real-only) — ported from examples/tactile.py (confirmed working).

Each GelSight Mini is a UVC camera (MJPG 3280x2464 @ ~25 fps native, ~18-20 fps sustained with
two on one USB bus). It runs in its own background thread, reading continuously, downsampling, and
stamping each frame with time.monotonic() into a ring buffer. The RealSense camera is the SYNCED
anchor: RealBackend reads it in the control loop and uses its capture timestamp to fetch the
nearest tactile sample from each sensor (nearest-neighbor alignment on one monotonic clock).

cv2 is imported lazily so the genesis-free / sim side never needs opencv (tactile is real-only).
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Optional, Tuple

import numpy as np


class TactileSensor:
    def __init__(
        self,
        device,                                       # int index or path (prefer /dev/v4l/by-id/...)
        name: str = "tactile",
        output_size: Optional[Tuple[int, int]] = (640, 480),  # (w, h); None = native 8 MP
        crop: Optional[Tuple[int, int, int, int]] = None,     # (x, y, w, h) gel region, pre-resize
        buffer_len: int = 32,                         # ~1.3 s of history at 25 fps
        backend: Optional[int] = None,                # default cv2.CAP_V4L2 on Linux
    ):
        import cv2
        self._cv2 = cv2
        self.name = name
        self.output_size = output_size
        self.crop = crop
        backend = cv2.CAP_V4L2 if backend is None else backend

        self.cap = cv2.VideoCapture(device, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{name}] could not open tactile device {device!r}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # NOTE: deliberately NOT setting CAP_PROP_BUFFERSIZE=1 (would tank the 8 MP-decode FPS);
        # the continuous-read thread keeps latency low instead.

        self._buf = collections.deque(maxlen=buffer_len)   # (timestamp, frame)
        self._lock = threading.Lock()
        self._running = True
        self._frame_count = 0
        self._last_fps_t = time.monotonic()
        self._last_fps_n = 0
        self._fps = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True, name=name)
        self._thread.start()

    def _process(self, frame: np.ndarray) -> np.ndarray:
        if self.crop is not None:
            x, y, w, h = self.crop
            frame = frame[y:y + h, x:x + w]
        if self.output_size is not None:
            frame = self._cv2.resize(frame, self.output_size, interpolation=self._cv2.INTER_AREA)
        return frame

    def _loop(self):
        fail_streak = 0
        while self._running:
            ok, frame = self.cap.read()          # blocks ~40 ms; includes JPEG decode
            t = time.monotonic()                 # stamp as close to capture as we can
            if not ok:
                fail_streak += 1
                if fail_streak % 30 == 0:
                    print(f"[{self.name}] read() failing ({fail_streak} in a row)", flush=True)
                time.sleep(0.001)
                continue
            fail_streak = 0
            frame = self._process(frame)
            with self._lock:
                self._buf.append((t, frame))
                self._frame_count += 1
            if t - self._last_fps_t >= 1.0:
                self._fps = (self._frame_count - self._last_fps_n) / (t - self._last_fps_t)
                self._last_fps_t, self._last_fps_n = t, self._frame_count
            time.sleep(0.001)                    # yield; don't burn CPU

    @property
    def fps(self) -> float:
        return self._fps

    def wait_for_frames(self, min_frames: int = 1, timeout: float = 5.0) -> None:
        """Block until the buffer has >= min_frames (call once at startup)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._buf) >= min_frames:
                    return
            time.sleep(0.005)
        raise RuntimeError(f"[{self.name}] no frames within {timeout}s (check device id / cabling)")

    def nearest(self, t_query: float) -> Tuple[float, np.ndarray]:
        """(timestamp, frame) whose timestamp is closest to t_query. Do not mutate the frame."""
        with self._lock:
            if not self._buf:
                raise RuntimeError(f"[{self.name}] buffer empty; call wait_for_frames() first")
            return min(self._buf, key=lambda tf: abs(tf[0] - t_query))

    def latest(self) -> Tuple[float, np.ndarray]:
        with self._lock:
            return self._buf[-1]

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()
