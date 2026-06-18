from __future__ import annotations

from typing import Optional, Set

# Episode-control keyboard via a small pygame window (matches the old collector).
# pygame is imported lazily (the `real` extra) and needs a display on the robot
# host. poll() returns edge events so a held key fires once.

SAVE = "save"
DISCARD = "discard"
QUIT = "quit"


class PygameKeyboard:
    """Tiny focused window translating key presses into episode events.

        SPACE → "save"      BACKSPACE → "discard"      ESC → "quit"

    Keys match KeyboardTeleop so the two teleop modes share one episode-key
    scheme (and A stays free for keyboard-teleop +y motion).

    Args:
        _pygame: test seam — inject a fake pygame module; when None, imports real pygame.
    """

    def __init__(self, _pygame: Optional[object] = None) -> None:
        self._pg = _pygame
        self._opened = False
        # Map is built in open() once the key constants are available.
        self._keymap: dict = {}

    def open(self) -> None:
        if self._opened:
            return
        if self._pg is None:
            import pygame
            self._pg = pygame
        pg = self._pg
        pg.init()
        pg.display.set_mode((320, 80))
        pg.display.set_caption("teleop: SPACE=save  BACKSPACE=discard  ESC=quit")
        self._keymap = {
            pg.K_SPACE: SAVE,
            pg.K_BACKSPACE: DISCARD,
            pg.K_ESCAPE: QUIT,
        }
        self._opened = True

    def poll(self) -> Set[str]:
        """Drain the event queue; return the set of episode events this tick."""
        events: Set[str] = set()
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                events.add(QUIT)
            elif event.type == self._pg.KEYDOWN and event.key in self._keymap:
                events.add(self._keymap[event.key])
        return events

    def close(self) -> None:
        if self._opened:
            self._pg.quit()
            self._opened = False
