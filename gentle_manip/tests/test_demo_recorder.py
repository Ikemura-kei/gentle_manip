import pickle
import types

import numpy as np
import pytest

from gentle_manip.demos.record import DemoRecorder
from gentle_manip.demos.keyboard_pygame import SAVE, DISCARD, QUIT
from gentle_manip.demos.teleop_spacemouse import SpaceMouseTeleop


# ── SpaceMouse mapping (fake device, no pyspacemouse) ─────────────────────────

def make_state(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, buttons=(0, 0)):
    return types.SimpleNamespace(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                 buttons=list(buttons))


class FakeDevice:
    def __init__(self, state):
        self._state = state
    def read(self):
        return self._state


def teleop_with_state(state, **kw):
    t = SpaceMouseTeleop(_device=FakeDevice(state), **kw)
    # background thread has read at least once; give it a beat
    for _ in range(1000):
        if t._state is not None:
            break
    import time; time.sleep(0.01)
    return t


def test_xy_negated_z_not():
    t = teleop_with_state(make_state(x=1.0, y=1.0, z=1.0), scale=0.8, deadzone=0.1)
    a = t.get_action()
    t.close()
    assert a[0] == pytest.approx(-0.8)   # X negated
    assert a[1] == pytest.approx(-0.8)   # Y negated
    assert a[2] == pytest.approx(0.8)    # Z not


def test_deadzone_zeros_small_input():
    t = teleop_with_state(make_state(x=0.05, roll=0.05), deadzone=0.1)
    a = t.get_action()
    t.close()
    assert a[0] == 0.0
    assert a[3] == 0.0


def test_gripper_buttons():
    t_open = teleop_with_state(make_state(buttons=(1, 0)), gripper_value=0.3)
    t_close = teleop_with_state(make_state(buttons=(0, 1)), gripper_value=0.3)
    a_open, a_close = t_open.get_action(), t_close.get_action()
    t_open.close(); t_close.close()
    assert a_open[6] == pytest.approx(0.3)
    assert a_close[6] == pytest.approx(-0.3)


def test_action_clipped_to_unit():
    t = teleop_with_state(make_state(x=-1.0), scale=2.0, deadzone=0.1)  # scale>1
    a = t.get_action()
    t.close()
    assert a.min() >= -1.0 and a.max() <= 1.0


def test_no_state_returns_zeros():
    t = SpaceMouseTeleop(_device=None)  # never opened → no polling thread
    assert np.allclose(t.get_action(), 0.0)


# ── Recorder logic (mock env / teleop / keyboard) ─────────────────────────────

class MockEnv:
    """Minimal PolicyEnv stand-in: obs dict with leading num_envs=1 dim."""

    num_envs = 1

    def __init__(self):
        self.reset_calls = 0
        self.closed = False
        self._z = 0.0

    def _obs(self):
        return {
            "ee_pos": np.array([[0.4, 0.0, self._z]], dtype=np.float32),
            "gripper_width": np.array([[0.08]], dtype=np.float32),
        }

    def reset(self, **kw):
        self.reset_calls += 1
        self._z = 0.0
        return self._obs()

    def step(self, action):
        self._z += float(action[0, 2])
        obs = self._obs()
        return obs, np.zeros(1, np.float32), np.zeros(1, bool), [{}]

    def close(self):
        self.closed = True


class FakeTeleop:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.opened = self.closed = False
    def open(self): self.opened = True
    def get_action(self): return self.action.copy()
    def close(self): self.closed = True


class ScriptedKeyboard:
    """Returns a scripted sequence of event sets, one per poll()."""

    def __init__(self, script):
        self.script = list(script)
        self.opened = self.closed = False
    def open(self): self.opened = True
    def poll(self):
        return self.script.pop(0) if self.script else {QUIT}
    def close(self): self.closed = True


def make_recorder(script, tmp_path, action=(0, 0, 0.01, 0, 0, 0, 0)):
    return DemoRecorder(
        env=MockEnv(),
        teleop=FakeTeleop(action),
        keyboard=ScriptedKeyboard(script),
        task_name="unit",
        out_dir=tmp_path,
        rate_hz=0.0,  # no sleeping in tests
    )


def test_three_steps_then_save(tmp_path):
    # 3 plain ticks (step+record), then SAVE, then QUIT.
    rec = make_recorder([set(), set(), set(), {SAVE}, {QUIT}], tmp_path)
    rec.run()
    assert len(rec.episodes) == 1
    ep = rec.episodes[0]
    assert ep["actions"].shape == (3, 7)
    assert ep["observations"]["ee_pos"].shape == (3, 3)   # num_envs dropped
    assert set(ep["observations"]) == {"ee_pos", "gripper_width"}


def test_discard_keeps_nothing(tmp_path):
    rec = make_recorder([set(), set(), {DISCARD}, {QUIT}], tmp_path)
    rec.run()
    assert rec.episodes == []


def test_save_then_reset_then_second_episode(tmp_path):
    rec = make_recorder([set(), {SAVE}, set(), set(), {SAVE}, {QUIT}], tmp_path)
    rec.run()
    assert len(rec.episodes) == 2
    assert rec.episodes[0]["actions"].shape == (1, 7)
    assert rec.episodes[1]["actions"].shape == (2, 7)
    assert rec.env.reset_calls == 3   # initial + after each save


def test_obs_action_alignment(tmp_path):
    # action dz=0.01 each step; ee_pos z should be 0 at first recorded obs,
    # then increase — i.e. obs_t is BEFORE action_t is applied.
    rec = make_recorder([set(), set(), set(), {SAVE}, {QUIT}], tmp_path)
    rec.run()
    z = rec.episodes[0]["observations"]["ee_pos"][:, 2]
    assert z[0] == pytest.approx(0.0)
    assert z[1] == pytest.approx(0.01)
    assert z[2] == pytest.approx(0.02)


def test_quit_drops_in_progress_buffer(tmp_path):
    rec = make_recorder([set(), set(), {QUIT}], tmp_path)
    rec.run()
    assert rec.episodes == []
    assert rec.env.closed and rec.teleop.closed and rec.keyboard.closed


def test_write_roundtrip_schema(tmp_path):
    rec = make_recorder([set(), set(), {SAVE}, {QUIT}], tmp_path)
    rec.run()
    path = rec.write()
    with open(path, "rb") as f:
        data = pickle.load(f)
    assert data["meta"]["task"] == "unit"
    assert data["meta"]["action_dim"] == 7
    assert sorted(data["meta"]["obs_keys"]) == ["ee_pos", "gripper_width"]
    assert len(data["episodes"]) == 1
    assert data["episodes"][0]["actions"].shape == (2, 7)


def test_write_nothing_when_no_episodes(tmp_path):
    rec = make_recorder([{QUIT}], tmp_path)
    rec.run()
    assert rec.write() is None


def test_config_snapshot_written_next_to_dataset(tmp_path):
    import yaml
    cfg = {"task_name": "unit", "setup": {"robot": {"ip": "1.2.3.4"}},
           "obs": {"point_cloud": {"max_points": 1024}}, "control": {"speed": 0.35}}
    rec = DemoRecorder(env=MockEnv(), teleop=FakeTeleop((0, 0, 0.01, 0, 0, 0, 0)),
                       keyboard=ScriptedKeyboard([set(), {SAVE}, {QUIT}]),
                       task_name="unit", out_dir=tmp_path, rate_hz=0.0,
                       collection_config=cfg)
    rec.run()
    path = rec.write()
    sidecar = path.parent / "config.yaml"              # lives in the run dir
    assert sidecar.exists()
    loaded = yaml.safe_load(open(sidecar))
    assert loaded["setup"]["robot"]["ip"] == "1.2.3.4"
    assert loaded["control"]["speed"] == 0.35


def test_no_config_snapshot_when_none(tmp_path):
    rec = make_recorder([set(), {SAVE}, {QUIT}], tmp_path)   # collection_config defaults to None
    rec.run()
    path = rec.write()
    assert not (path.parent / "config.yaml").exists()


def test_episode_flushed_immediately(tmp_path):
    import pickle, re
    # SAVE then QUIT — shard must exist after run() WITHOUT a final write() call.
    # Use shard_size=1 so the first saved episode immediately flushes to disk.
    rec = DemoRecorder(
        env=MockEnv(), teleop=FakeTeleop((0, 0, 0.01, 0, 0, 0, 0)),
        keyboard=ScriptedKeyboard([set(), set(), {SAVE}, {QUIT}]),
        task_name="unit", out_dir=tmp_path, rate_hz=0.0, shard_size=1,
    )
    rec.run()
    assert rec._run_dir_path is not None
    shard = rec._run_dir_path / "shard_0000.pkl"
    assert shard.exists()
    data = pickle.load(open(shard, "rb"))
    assert len(data["episodes"]) == 1
    # run dir name is YY-MM-DD-xyz
    assert re.fullmatch(r"\d{2}-\d{2}-\d{2}-[a-z]{3}", rec._run_dir_path.name)


def test_no_file_until_first_save(tmp_path):
    rec = make_recorder([set(), {QUIT}], tmp_path)   # step but never save
    rec.run()
    assert rec._run_dir_path is None
    assert not any(tmp_path.rglob("*.pkl"))


# ── Idle trimming (leading dropped, trailing capped, interior kept) ───────────

IDLE = np.zeros(7, dtype=np.float32)
MOVE = np.array([0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32)
GRIP = np.array([0, 0, 0, 0, 0, 0, 0.3], dtype=np.float32)


def bare_recorder(tmp_path, **kw):
    return DemoRecorder(env=None, teleop=None, keyboard=None, task_name="t",
                        out_dir=tmp_path, **kw)


def fill(rec, actions):
    rec._obs_buf = [{"ee_pos": np.array([i, 0, 0], np.float32)} for i in range(len(actions))]
    rec._act_buf = [a.copy() for a in actions]


def test_leading_idle_dropped_trailing_kept(tmp_path):
    rec = bare_recorder(tmp_path, keep_trailing_idle=5)
    fill(rec, [IDLE, IDLE, MOVE, MOVE, IDLE])
    n = rec._save_episode()
    assert n == 3                                  # 2 leading idle dropped
    ep = rec.episodes[0]
    assert list(ep["observations"]["ee_pos"][:, 0]) == [2, 3, 4]  # kept frames aligned
    assert ep["actions"].shape == (3, 7)


def test_trailing_idle_capped(tmp_path):
    rec = bare_recorder(tmp_path, keep_trailing_idle=5)
    fill(rec, [MOVE] + [IDLE] * 10)
    n = rec._save_episode()
    assert n == 6                                  # 1 move + 5 trailing idle (capped)


def test_short_interior_idle_preserved(tmp_path):
    rec = bare_recorder(tmp_path, max_interior_idle=3)
    fill(rec, [MOVE, IDLE, IDLE, MOVE])
    n = rec._save_episode()
    assert n == 4                                  # 2 <= 3, untouched


def test_long_interior_idle_capped(tmp_path):
    rec = bare_recorder(tmp_path, max_interior_idle=3)
    fill(rec, [MOVE, IDLE, IDLE, IDLE, IDLE, IDLE, MOVE])   # interior run of 5
    n = rec._save_episode()
    assert n == 5                                  # MOVE + 3 idle + MOVE
    assert list(rec.episodes[0]["observations"]["ee_pos"][:, 0]) == [0, 1, 2, 3, 6]


def test_multiple_interior_runs_each_capped(tmp_path):
    rec = bare_recorder(tmp_path, max_interior_idle=2)
    fill(rec, [MOVE, IDLE, IDLE, IDLE, MOVE, IDLE, IDLE, IDLE, MOVE])
    n = rec._save_episode()
    assert n == 7                                  # MOVE + 2 + MOVE + 2 + MOVE


def test_leading_interior_trailing_together(tmp_path):
    rec = bare_recorder(tmp_path, max_interior_idle=2, keep_trailing_idle=3)
    fill(rec, [IDLE, IDLE, MOVE, IDLE, IDLE, IDLE, IDLE, MOVE, IDLE, IDLE, IDLE, IDLE])
    n = rec._save_episode()
    # leading 2 dropped; interior run of 4 → 2; trailing run of 4 → 3
    assert n == 1 + 2 + 1 + 3                       # MOVE +2 interior + MOVE + 3 trailing


def test_all_idle_saves_nothing(tmp_path):
    rec = bare_recorder(tmp_path, keep_trailing_idle=5)
    fill(rec, [IDLE, IDLE, IDLE])
    assert rec._save_episode() == 0
    assert rec.episodes == []


def test_threshold_zero_disables_trim(tmp_path):
    rec = bare_recorder(tmp_path, idle_threshold=0.0)
    fill(rec, [IDLE, MOVE, IDLE])
    assert rec._save_episode() == 3                # nothing trimmed


def test_gripper_frame_is_not_idle(tmp_path):
    # leading idle dropped, but the gripper-only frame counts as motion and is kept
    rec = bare_recorder(tmp_path, keep_trailing_idle=5)
    fill(rec, [IDLE, GRIP, IDLE])
    n = rec._save_episode()
    assert n == 2
    assert list(rec.episodes[0]["observations"]["ee_pos"][:, 0]) == [1, 2]
