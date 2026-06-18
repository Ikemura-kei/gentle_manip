import types

import numpy as np
import pytest

from gentle_manip.demos.teleop_keyboard import KeyboardTeleop
from gentle_manip.demos.keyboard_pygame import SAVE, DISCARD, QUIT


# ── Fake pygame ───────────────────────────────────────────────────────────────

class _Pressed:
    def __init__(self, held):
        self._held = set(held)
    def __getitem__(self, key):
        return key in self._held


class FakePygame:
    """Minimal pygame stand-in: held keys for get_pressed, scripted events."""

    # unique key/event constants
    (K_w, K_s, K_a, K_d, K_r, K_f, K_LEFT, K_RIGHT, K_UP, K_DOWN,
     K_q, K_e, K_o, K_p, K_SPACE, K_BACKSPACE, K_ESCAPE, KEYDOWN, QUIT) = range(19)

    def __init__(self):
        self._held = set()
        self._events = []
        self.key = types.SimpleNamespace(get_pressed=lambda: _Pressed(self._held))
        self.event = types.SimpleNamespace(get=self._get_events)
        self.display = types.SimpleNamespace(set_mode=lambda *a, **k: None,
                                             set_caption=lambda *a, **k: None)

    def init(self): pass
    def quit(self): pass
    def _get_events(self):
        ev, self._events = self._events, []
        return ev

    # test helpers
    def hold(self, *keys):
        self._held = set(keys)
    def queue_keydown(self, key):
        self._events.append(types.SimpleNamespace(type=self.KEYDOWN, key=key))
    def queue_quit(self):
        self._events.append(types.SimpleNamespace(type=self.QUIT, key=None))


def make_teleop(**kw):
    pg = FakePygame()
    t = KeyboardTeleop(_pygame=pg, **kw)
    t.open()
    return t, pg


# ── Motion mapping ────────────────────────────────────────────────────────────

def test_translation_axes():
    t, pg = make_teleop(move_speed=0.5)
    pg.hold(pg.K_w);    assert t.get_action()[0] == pytest.approx(0.5)
    pg.hold(pg.K_s);    assert t.get_action()[0] == pytest.approx(-0.5)
    pg.hold(pg.K_a);    assert t.get_action()[1] == pytest.approx(0.5)
    pg.hold(pg.K_d);    assert t.get_action()[1] == pytest.approx(-0.5)
    pg.hold(pg.K_UP);   assert t.get_action()[2] == pytest.approx(0.5)
    pg.hold(pg.K_DOWN); assert t.get_action()[2] == pytest.approx(-0.5)


def test_rotation_axes():
    t, pg = make_teleop(rot_speed=0.5)
    pg.hold(pg.K_LEFT);  assert t.get_action()[3] == pytest.approx(0.5)
    pg.hold(pg.K_RIGHT); assert t.get_action()[3] == pytest.approx(-0.5)
    pg.hold(pg.K_r);     assert t.get_action()[4] == pytest.approx(0.5)
    pg.hold(pg.K_f);     assert t.get_action()[4] == pytest.approx(-0.5)
    pg.hold(pg.K_q);     assert t.get_action()[5] == pytest.approx(0.5)
    pg.hold(pg.K_e);     assert t.get_action()[5] == pytest.approx(-0.5)


def test_gripper_keys():
    t, pg = make_teleop(gripper_value=0.3)
    pg.hold(pg.K_o); assert t.get_action()[6] == pytest.approx(0.3)
    pg.hold(pg.K_p); assert t.get_action()[6] == pytest.approx(-0.3)


def test_opposite_keys_cancel():
    t, pg = make_teleop(move_speed=0.5)
    pg.hold(pg.K_w, pg.K_s)
    assert t.get_action()[0] == pytest.approx(0.0)


def test_action_clipped():
    t, pg = make_teleop(move_speed=2.0)   # > 1
    pg.hold(pg.K_w)
    a = t.get_action()
    assert a[0] == pytest.approx(1.0)
    assert a.min() >= -1.0 and a.max() <= 1.0


def test_no_keys_zero_action():
    t, pg = make_teleop()
    assert np.allclose(t.get_action(), 0.0)


# ── Episode events ────────────────────────────────────────────────────────────

def test_poll_episode_events():
    t, pg = make_teleop()
    pg.queue_keydown(pg.K_SPACE)
    assert t.poll() == {SAVE}
    pg.queue_keydown(pg.K_BACKSPACE)
    assert t.poll() == {DISCARD}
    pg.queue_keydown(pg.K_ESCAPE)
    assert t.poll() == {QUIT}
    pg.queue_quit()
    assert t.poll() == {QUIT}


def test_poll_empty_when_no_events():
    t, pg = make_teleop()
    assert t.poll() == set()


def test_open_close_idempotent():
    # record.py opens/closes the same object as both teleop and keyboard.
    t, pg = make_teleop()
    t.open()         # second open → no-op
    t.close()
    t.close()        # second close → no-op
    assert not t._opened
