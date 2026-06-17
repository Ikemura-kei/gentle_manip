from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# NOTE: pyrealsense2 is a hardware-only dependency (the `real` extra). It is
# imported lazily inside start() so this module imports cleanly on any machine
# and tests can run with an injected fake pipeline. Never import genesis here —
# this is the real side of the RawObs boundary.


class RealSenseCamera:
    """Single Intel RealSense device (one physical camera = one instance).

    Produces metric depth + colour aligned to the colour frame, plus the colour
    intrinsics, in the units RawObs expects:
        depth: (H, W) float32, meters
        rgb:   (H, W, 3) uint8
        K:     (3, 3) float32

    Multi-camera setups are just multiple instances keyed by name in the
    RealBackend — there is no special-casing for the current single-camera rig.

    Args:
        name:       logical camera name (e.g. "cam_ext"); must match configs.
        serial:     device serial number (selects the physical device).
        width, height: colour/depth stream resolution.
        depth_min, depth_max: valid depth range (meters); outside → 0 (invalid).
        fps:        stream frame rate.
        _pipeline:  test seam — a pre-built object exposing wait_for_frames();
                    when None, start() builds a real rs.pipeline.
    """

    def __init__(
        self,
        name: str,
        serial: str,
        width: int = 640,
        height: int = 480,
        depth_min: float = 0.1,
        depth_max: float = 0.85,
        fps: int = 30,
        _pipeline: Optional[object] = None,
    ) -> None:
        self.name = name
        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.fps = int(fps)

        self._pipeline = _pipeline      # rs.pipeline (or fake) once started
        self._align = None              # rs.align(color) — depth→color alignment
        self._depth_scale = 1.0e-3      # meters per depth unit; refreshed in start()
        self._started = _pipeline is not None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the stream. Real path lazy-imports pyrealsense2."""
        if self._started:
            return

        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        profile = pipeline.start(config)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self._align = rs.align(rs.stream.color)
        self._pipeline = pipeline
        self._started = True

    def stop(self) -> None:
        if self._pipeline is not None and self._started:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._started = False

    # ── Frame grab ────────────────────────────────────────────────────────────

    def get_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grab one aligned frame.

        Returns:
            depth_m: (H, W) float32 meters; invalid/out-of-range pixels are 0.
            rgb:     (H, W, 3) uint8, RGB order.
            K:       (3, 3) float32 colour intrinsics.
        """
        if not self._started or self._pipeline is None:
            raise RuntimeError(f"RealSenseCamera {self.name!r} not started")

        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_raw = np.asarray(depth_frame.get_data())          # (H, W) uint16
        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        # Zero out points outside the trusted depth range → treated as invalid.
        invalid = (depth_m < self.depth_min) | (depth_m > self.depth_max)
        depth_m[invalid] = 0.0

        bgr = np.asarray(color_frame.get_data())                # (H, W, 3) uint8, BGR
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])             # → RGB

        K = self._intrinsics_matrix(color_frame)
        return depth_m, rgb, K

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _intrinsics_matrix(color_frame) -> np.ndarray:
        """Build a 3×3 intrinsics matrix from the colour frame's profile."""
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        return np.array(
            [[intr.fx, 0.0,     intr.ppx],
             [0.0,     intr.fy, intr.ppy],
             [0.0,     0.0,     1.0]],
            dtype=np.float32,
        )
