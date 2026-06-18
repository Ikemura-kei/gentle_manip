from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

# 3Dconnexion SpaceMouse teleop. pyspacemouse is imported lazily (hardware-only,
# the `real` extra) so this module imports on any machine and tests inject a fake
# device. No genesis/env imports — this just turns device state into a normalized
# action in [-1, 1], i.e. the same raw_action a policy would output.

# Action layout (matches ActionConfig): [dx, dy, dz, droll, dpitch, dyaw, dgripper].
DEFAULT_SCALE = 0.8        # per-axis magnitude on the [-1,1] device reading
DEFAULT_DEADZONE = 0.1     # raw |value| below this → 0 (filters drift/jitter)
DEFAULT_GRIPPER_VALUE = 0.3  # per-step gripper delta when a button is held


class SpaceMouseTeleop:
    """Background-polled SpaceMouse → 7-D normalized action.

    Mirrors the old SpacemouseHandler: a daemon thread continuously reads the
    device into the latest state; get_action() maps that state to the action.

    Args:
        scale:         per-axis multiplier on the device reading.
        deadzone:      raw |reading| below this is zeroed.
        gripper_value: per-step gripper delta while a button is held.
        _device:       test seam — object exposing read() → state with
                       x,y,z,roll,pitch,yaw,buttons; when None, opens the real device.
    """

    def __init__(
        self,
        scale: float = DEFAULT_SCALE,
        deadzone: float = DEFAULT_DEADZONE,
        gripper_value: float = DEFAULT_GRIPPER_VALUE,
        _device: Optional[object] = None,
    ) -> None:
        self.scale = float(scale)
        self.deadzone = float(deadzone)
        self.gripper_value = float(gripper_value)

        self._device = _device
        self._state = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        if _device is not None:
            self._start_thread()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the real device and start polling (no-op if already running)."""
        if self._running:
            return
        if self._device is None:
            import pyspacemouse
            self._device = pyspacemouse.open()
            if not self._device:
                raise RuntimeError("pyspacemouse.open() failed — check the device / udev rule")
        self._start_thread()

    def _start_thread(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        while self._running:
            state = self._device.read()
            if state is not None:
                with self._lock:
                    self._state = state
            time.sleep(0.001)

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    # ── Action mapping ────────────────────────────────────────────────────────

    def get_action(self) -> np.ndarray:
        """Latest command as a (7,) float32 action in [-1, 1].

        X and Y are negated to match the robot base frame (old convention);
        gripper dim is +value (button 0, open) / −value (button 1, close) / 0.
        """
        with self._lock:
            state = self._state
        if state is None:
            return np.zeros(7, dtype=np.float32)

        dz = self._deadzone
        action = np.array([
            -dz(state.x) * self.scale,      # dx  (X negated)
            -dz(state.y) * self.scale,      # dy  (Y negated)
             dz(state.z) * self.scale,      # dz
             dz(state.roll) * self.scale,
             dz(state.pitch) * self.scale,
             dz(state.yaw) * self.scale,
             self._gripper(state),
        ], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)

    def is_idle(self, thresh: float = 1e-3) -> bool:
        """True if the current motion command (excl. gripper) is ~zero."""
        return float(np.linalg.norm(self.get_action()[:6])) < thresh

    # ── Internal ──────────────────────────────────────────────────────────────

    def _deadzone(self, value: float) -> float:
        return float(value) if abs(value) > self.deadzone else 0.0

    def _gripper(self, state) -> float:
        buttons = getattr(state, "buttons", None) or []
        if len(buttons) >= 1 and buttons[0]:
            return self.gripper_value       # open
        if len(buttons) >= 2 and buttons[1]:
            return -self.gripper_value      # close
        return 0.0
