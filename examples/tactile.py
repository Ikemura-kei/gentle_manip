"""
Camera-anchored multimodal reading for a manipulation policy.

Design
------
- Each GelSight Mini runs in its own background thread (`TactileSensor`),
  capturing continuously at its native 25 fps and pushing timestamped,
  downsampled frames into a ring buffer. The thread keeps the UVC buffer
  drained, so no stale backlog builds up and the expensive 8 MP JPEG decode
  stays off the control loop's critical path.
- The scene camera is the ANCHOR: it is read synchronously in the control
  loop, which paces the loop. Its capture timestamp is the query key used to
  fetch the nearest tactile sample from each sensor.
- All timestamps use a single `time.monotonic()` clock (same process), which
  is what makes nearest-neighbor alignment across modalities valid.

Key gotchas baked in
--------------------
- Tactiles do NOT set CAP_PROP_BUFFERSIZE=1: the continuous-reading thread
  already keeps latency low, and =1 would collapse the 8 MP decode to ~10 fps.
- The camera DOES set CAP_PROP_BUFFERSIZE=1: its decode is cheap, so this
  keeps the anchor frame fresh without hurting throughput.
- The camera timestamp is taken right after read() so it reflects true(ish)
  capture time, not "now". If it lied, the whole alignment would silently break.
"""

import time
import threading
import collections
from typing import Optional, Tuple

import cv2
import numpy as np


class TactileSensor:
    """
    Async grabber for a single GelSight Mini (UVC camera, MJPG 3280x2464 @ 25 fps).

    Runs a background thread that reads frames continuously, downsamples them,
    stamps them with time.monotonic(), and stores (timestamp, frame) pairs in a
    ring buffer. Query the buffer with `nearest(t)` from your control loop.
    """

    def __init__(
        self,
        device,                          # int index (e.g. 2) or path ("/dev/v4l/by-id/...")
        name: str = "tactile",
        output_size: Optional[Tuple[int, int]] = (640, 480),  # (w, h); None = keep native 8 MP
        crop: Optional[Tuple[int, int, int, int]] = None,     # (x, y, w, h) gel region, pre-resize
        buffer_len: int = 32,            # ~1.3 s of history at 25 fps
        backend: int = cv2.CAP_ANY,      # on Linux, cv2.CAP_V4L2 is often more reliable
    ):
        self.name = name
        self.output_size = output_size
        self.crop = crop

        self.cap = cv2.VideoCapture(device, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{name}] could not open device {device!r}")
        # The Mini only exposes MJPG @ 3280x2464; set it explicitly to document intent.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # NOTE: deliberately NOT setting CAP_PROP_BUFFERSIZE=1 here (would tank FPS).

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
            # INTER_AREA is the correct choice for downscaling.
            frame = cv2.resize(frame, self.output_size, interpolation=cv2.INTER_AREA)
        return frame

    def _loop(self):
        fail_streak = 0
        while self._running:
            ok, frame = self.cap.read()          # blocks ~40 ms; includes JPEG decode
            t = time.monotonic()                 # stamp as close to capture as we can
            if not ok:
                fail_streak += 1
                if fail_streak % 30 == 0:
                    print(f"[{self.name}] read() failing ({fail_streak} in a row)")
                time.sleep(0.001)
                continue
            fail_streak = 0
            frame = self._process(frame)
            with self._lock:
                self._buf.append((t, frame))
                self._frame_count += 1

            # lightweight fps estimate for diagnostics
            if t - self._last_fps_t >= 1.0:
                self._fps = (self._frame_count - self._last_fps_n) / (t - self._last_fps_t)
                self._last_fps_t = t
                self._last_fps_n = self._frame_count

            time.sleep(0.001)  # yield to other threads; don't burn CPU

    @property
    def fps(self) -> float:
        return self._fps

    def wait_for_frames(self, min_frames: int = 1, timeout: float = 5.0):
        """Block until the buffer has at least `min_frames` (call once at startup)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._buf) >= min_frames:
                    return
            time.sleep(0.005)
        raise RuntimeError(f"[{self.name}] no frames within {timeout}s (check device id)")

    def nearest(self, t_query: float) -> Tuple[float, np.ndarray]:
        """
        Return (timestamp, frame) whose timestamp is closest to t_query.
        Do not mutate the returned frame in place (it is the stored copy).
        """
        with self._lock:
            if not self._buf:
                raise RuntimeError(f"[{self.name}] buffer empty; call wait_for_frames() first")
            return min(self._buf, key=lambda tf: abs(tf[0] - t_query))

    def latest(self) -> Tuple[float, np.ndarray]:
        with self._lock:
            return self._buf[-1]

    def release(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()


class SyncCamera:
    """
    Synchronous scene camera used as the loop anchor. buffersize=1 keeps the
    frame fresh; its decode is cheap enough that this does not hurt throughput.
    """

    def __init__(self, device, backend: int = cv2.CAP_ANY,
                 fourcc: Optional[str] = "MJPG"):
        self.cap = "dummy"

    def read(self) -> Tuple[float, np.ndarray]:
        """Blocking read; returns (capture_timestamp, frame)."""
        ok, frame = True, None  # dummy
        t = time.monotonic()
        if not ok:
            raise RuntimeError("[camera] read() failed")
        return t, frame

    def release(self):
        return
"""
Two-tactile FPS test. No camera, no policy — just spin up both GelSight Minis
and watch sustained per-sensor FPS + read health.

Both Minis are on the same USB 2.0 (480 Mbps) bus on this machine, so the number
to care about is the *sustained* rate with BOTH running, not the peak. If FPS
sags below ~25 or you see "read() failing" spam, it's bus/CPU contention.

Run:
    python test_two_tactiles.py
    python test_two_tactiles.py --native            # skip host-side resize
    python test_two_tactiles.py --dev0 /dev/video8 --dev1 /dev/video10

Ctrl-C to stop; prints a per-sensor summary (mean / min FPS) on exit.
"""
import time
import argparse
import statistics

import cv2
import numpy as np

# from camera_tactile import TactileSensor  # <-- rename to your module's filename


def side_by_side(a, b, height=480):
    """Resize both to a common height and hstack — display only, does not
    touch the stored frames or the FPS measurement."""
    def fit(x):
        h, w = x.shape[:2]
        s = height / h
        return cv2.resize(x, (int(w * s), height), interpolation=cv2.INTER_AREA)
    return np.hstack([fit(a), fit(b)])


def main():
    ap = argparse.ArgumentParser()
    # Capture nodes are the LOWER index of each pair (odd node = UVC metadata):
    #   2DYF-163L -> /dev/video8   ,   2DYE-1LB8 -> /dev/video10
    # For stability across replugs, swap these for by-id paths:
    #   ls /dev/v4l/by-id/
    ap.add_argument("--dev0", default="/dev/video8") # left GelSight Mini 
    ap.add_argument("--dev1", default="/dev/video10") # right GelSight Mini
    ap.add_argument("--native", action="store_true",
                    help="keep native 8MP (no host resize) to isolate USB/decode cost")
    ap.add_argument("--secs", type=float, default=0.0,
                    help="auto-stop after N seconds (0 = run until Ctrl-C)")
    ap.add_argument("--no-show", action="store_true",
                    help="headless: FPS numbers only, no imshow window")
    args = ap.parse_args()

    out = None if args.native else (640, 480)

    dummy_cam = SyncCamera(0)  # dummy; just to get the output size
    tac0 = TactileSensor(args.dev0, name="tac0", output_size=out)
    tac1 = TactileSensor(args.dev1, name="tac1", output_size=out)

    print("waiting for first frames...")
    tac0.wait_for_frames()
    tac1.wait_for_frames()

    hist0, hist1 = [], []
    t_start = time.monotonic()
    last_print = t_start
    print("running (Ctrl-C to stop; press q in the window to quit)")
    try:
        while True:
            t, frame = dummy_cam.read()  # anchor the loop; discard the frame
            # Pull the newest frame from each sensor's ring buffer. These come
            # from the capture threads; the display rate here is independent of
            # (and usually faster than) the 25 fps capture rate, so you'll see
            # some frames repeat — that's expected and doesn't affect FPS.
            t0f, f0img = tac0.nearest(t)
            t1f, f1img = tac1.nearest(t)

            if not args.no_show:
                cv2.imshow("tac0 | tac1", side_by_side(f0img, f1img))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)

            now = time.monotonic()
            if now - last_print >= 0.5:
                last_print = now
                f0, f1 = tac0.fps, tac1.fps
                if f0 > 0:
                    hist0.append(f0)
                if f1 > 0:
                    hist1.append(f1)
                skew_ms = 1000 * abs(t0f - t1f)
                print(f"tac0 {f0:5.1f} fps | tac1 {f1:5.1f} fps | "
                      f"latest-skew {skew_ms:4.0f} ms", end="\r")

            if args.secs and (now - t_start) >= args.secs:
                break
    except KeyboardInterrupt:
        pass
    finally:
        tac0.release()
        tac1.release()
        cv2.destroyAllWindows()
        print("\n--- summary ---")
        for name, h in (("tac0", hist0), ("tac1", hist1)):
            if h:
                print(f"{name}: mean {statistics.mean(h):5.1f}  "
                      f"min {min(h):5.1f}  max {max(h):5.1f}  fps  "
                      f"(n_samples={len(h)})")
            else:
                print(f"{name}: no fps samples (device never delivered frames?)")


if __name__ == "__main__":
    main()