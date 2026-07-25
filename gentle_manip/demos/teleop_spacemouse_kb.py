from __future__ import annotations

from typing import Optional, Set

import numpy as np

from gentle_manip.demos.keyboard_pygame import DISCARD, QUIT, SAVE
from gentle_manip.demos.teleop_spacemouse import (
    DEFAULT_DEADZONE, DEFAULT_SCALE, SpaceMouseTeleop,
)

# Mixed teleop: SpaceMouse controls the 6-DOF pose; keyboard (Z/X) controls the
# gripper. Episode edge events (SPACE/BACKSPACE/ESC) come from the same pygame
# window. A single instance serves as both `teleop` and `keyboard` in record.py.
#
# Control layout:
#   pose:    SpaceMouse (tx/ty/tz/roll/pitch/yaw)
#   gripper: Z = open (+)   X = close (−)
#   episode: SPACE = save   BACKSPACE = discard   ESC = quit

CAPTION = "SpaceMouse: pose  |  Z=open  X=close  |  SPACE=save  BKSP=discard  ESC=quit"

DEFAULT_GRIPPER_VALUE = 0.45


class SpaceMousePygameTeleop:
    """SpaceMouse for pose + pygame keyboard for gripper and episode control.

    Implements both the teleop interface (get_action) and the keyboard interface
    (poll) so record.py can pass one instance as both ``teleop`` and ``keyboard``.

    Args:
        scale:         SpaceMouse per-axis scale (matches SpaceMouseTeleop).
        deadzone:      SpaceMouse deadzone (matches SpaceMouseTeleop).
        gripper_value: per-step gripper delta while Z or X is held.
        _device:       test seam — fake SpaceMouse device for unit tests.
        _pygame:       test seam — fake pygame module for unit tests.
    """

    def __init__(
        self,
        scale: float = DEFAULT_SCALE,
        deadzone: float = DEFAULT_DEADZONE,
        gripper_value: float = DEFAULT_GRIPPER_VALUE,
        _device: Optional[object] = None,
        _pygame: Optional[object] = None,
    ) -> None:
        self.gripper_value = float(gripper_value)
        self._sm = SpaceMouseTeleop(scale=scale, deadzone=deadzone,
                                    gripper_value=0.0, _device=_device)
        self._pg = _pygame
        self._opened = False
        self._gripper_open_key: int = 0   # resolved in open()
        self._gripper_close_key: int = 0
        self._episode_keys: dict = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        if self._opened:
            return
        self._sm.open()
        if self._pg is None:
            import pygame
            self._pg = pygame
        pg = self._pg
        pg.init()
        pg.display.set_mode((640, 60))
        pg.display.set_caption(CAPTION)
        self._gripper_open_key = pg.K_z
        self._gripper_close_key = pg.K_x
        self._episode_keys = {
            pg.K_SPACE: SAVE,
            pg.K_BACKSPACE: DISCARD,
            pg.K_ESCAPE: QUIT,
        }
        self._opened = True

    def close(self) -> None:
        if self._opened:
            self._sm.close()
            self._pg.quit()
            self._opened = False

    # ── Teleop interface ──────────────────────────────────────────────────────

    def get_action(self) -> np.ndarray:
        """7-D action: dims 0-5 from SpaceMouse, dim 6 (gripper) from Z/X keys."""
        sm_action = self._sm.get_action()
        gripper = self._gripper_from_keys()
        action = np.concatenate([sm_action[:6], [gripper]], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)

    def is_idle(self, thresh: float = 1e-3) -> bool:
        return float(np.linalg.norm(self.get_action()[:6])) < thresh

    # ── Keyboard interface ────────────────────────────────────────────────────

    def poll(self) -> Set[str]:
        """Drain pygame events; return episode edge events for this tick."""
        events: Set[str] = set()
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                events.add(QUIT)
            elif event.type == self._pg.KEYDOWN and event.key in self._episode_keys:
                events.add(self._episode_keys[event.key])
        return events

    # ── Internal ──────────────────────────────────────────────────────────────

    def _gripper_from_keys(self) -> float:
        pressed = self._pg.key.get_pressed()
        if pressed[self._gripper_open_key]:
            return self.gripper_value
        if pressed[self._gripper_close_key]:
            return -self.gripper_value
        return 0.0
