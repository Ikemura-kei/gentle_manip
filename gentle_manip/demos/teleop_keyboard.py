from __future__ import annotations

from typing import Optional, Set

import numpy as np

from gentle_manip.demos.keyboard_pygame import DISCARD, QUIT, SAVE, TOGGLE_FILTER

# Keyboard teleop: a drop-in alternative to SpaceMouseTeleop using a pygame
# window. It implements BOTH interfaces the recorder needs — get_action() (motion
# from held keys) and poll() (episode edge events) — so record.py passes one
# instance as both `teleop` and `keyboard` (a single shared pygame context).
#
# Held keys give a constant-magnitude command in [-1, 1] (the same raw_action a
# policy outputs); pressing opposite keys cancels. pygame is imported lazily and
# needs a display on the robot host; tests inject a fake module via _pygame.

# Control layout (also shown in the window caption):
#   translate: W/S = +x/-x   A/D = +y/-y   Up/Down = +z/-z
#   rotate:    Left/Right = +roll/-roll   R/F = +pitch/-pitch   Q/E = +yaw/-yaw
#   gripper:   O = open   P = close
#   episode:   SPACE = save   BACKSPACE = discard   ESC = quit
CAPTION = "trans W/S A/D Up/Dn  rot L/R R/F Q/E  grip O/P  M filter  SPACE save  BKSP discard  ESC quit"

DEFAULT_MOVE_SPEED = 0.5
DEFAULT_ROT_SPEED = 1.0
DEFAULT_GRIPPER_VALUE = 0.05


class KeyboardTeleop:
    """pygame keyboard teleop providing both get_action() and poll()."""

    def __init__(
        self,
        move_speed: float = DEFAULT_MOVE_SPEED,
        rot_speed: float = DEFAULT_ROT_SPEED,
        gripper_value: float = DEFAULT_GRIPPER_VALUE,
        _pygame: Optional[object] = None,
    ) -> None:
        self.move_speed = float(move_speed)
        self.rot_speed = float(rot_speed)
        self.gripper_value = float(gripper_value)
        self._pg = _pygame
        self._opened = False
        self._motion: list = []       # (key, action_index, sign, magnitude)
        self._episode: dict = {}      # key → event string

    # ── Lifecycle (idempotent: record.py opens/closes it as teleop AND keyboard) ─

    def open(self) -> None:
        if self._opened:
            return
        if self._pg is None:
            import pygame
            self._pg = pygame
        pg = self._pg
        pg.init()
        pg.display.set_mode((480, 90))
        pg.display.set_caption(CAPTION)

        m, r, g = self.move_speed, self.rot_speed, self.gripper_value
        self._motion = [
            (pg.K_w, 0, +1, m), (pg.K_s, 0, -1, m),     # x
            (pg.K_a, 1, +1, m), (pg.K_d, 1, -1, m),     # y
            (pg.K_UP, 2, +1, m), (pg.K_DOWN, 2, -1, m),     # z
            (pg.K_LEFT, 3, +1, r), (pg.K_RIGHT, 3, -1, r),   # roll
            (pg.K_r, 4, +1, r), (pg.K_f, 4, -1, r),      # pitch
            (pg.K_q, 5, +1, r), (pg.K_e, 5, -1, r),          # yaw
            (pg.K_o, 6, +1, g), (pg.K_p, 6, -1, g),     # gripper
        ]
        self._episode = {
            pg.K_SPACE: SAVE,
            pg.K_BACKSPACE: DISCARD,
            pg.K_ESCAPE: QUIT,
            pg.K_m: TOGGLE_FILTER,           # live A/B of the ground_residual cloud filter
        }
        self._opened = True

    def close(self) -> None:
        if self._opened:
            self._pg.quit()
            self._opened = False

    # ── Motion (held keys) ────────────────────────────────────────────────────

    def get_action(self) -> np.ndarray:
        """Current command as a (7,) float32 action in [-1, 1] from held keys."""
        pressed = self._pg.key.get_pressed()
        action = np.zeros(7, dtype=np.float32)
        for key, idx, sign, mag in self._motion:
            if pressed[key]:
                action[idx] += sign * mag
        return np.clip(action, -1.0, 1.0)

    # ── Episode events (edge keys) ────────────────────────────────────────────

    def poll(self) -> Set[str]:
        events: Set[str] = set()
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                events.add(QUIT)
            elif event.type == self._pg.KEYDOWN and event.key in self._episode:
                events.add(self._episode[event.key])
        return events
