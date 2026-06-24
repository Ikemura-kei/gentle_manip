from pathlib import Path

import numpy as np
import pytest
import yaml

from gentle_manip.domain_randomization import DRConfig, DR_PRESETS, aggressive, mild

_DR_CFG = Path(__file__).resolve().parents[1] / "configs" / "dr"


# ── config ──────────────────────────────────────────────────────────────────────
def test_default_is_noop():
    cfg = DRConfig()
    assert cfg.is_noop() and not cfg.has_reset_dr() and not cfg.has_scene_dr()


def test_from_dict_lists_become_tuples_and_unknowns_ignored():
    cfg = DRConfig.from_dict({"object_pos_xy": 0.03, "object_E": [3e3, 6e3], "bogus": 1})
    assert cfg.object_pos_xy == 0.03
    assert cfg.object_E == (3000.0, 6000.0) and isinstance(cfg.object_E, tuple)
    assert cfg.has_reset_dr() and cfg.has_scene_dr()


def test_from_dict_none_or_empty():
    assert DRConfig.from_dict(None).is_noop()
    assert DRConfig.from_dict({}).is_noop()


# ── sampling ────────────────────────────────────────────────────────────────────
def test_sample_object_dxy_shape_range_and_disabled():
    rng = np.random.default_rng(0)
    assert DRConfig().sample_object_dxy(rng, 8) is None         # disabled -> None
    dxy = DRConfig(object_pos_xy=0.03).sample_object_dxy(rng, 8)
    assert dxy.shape == (8, 2) and dxy.dtype == np.float32
    assert np.all(np.abs(dxy) <= 0.03)


def test_sample_scene_only_set_keys_and_in_range():
    rng = np.random.default_rng(0)
    cfg = DRConfig(object_E=(3e3, 6e3), coup_friction=(3.5, 4.5))
    for _ in range(50):
        s = cfg.sample_scene(rng)
        assert set(s) == {"E", "coup_friction"}                 # only randomized keys
        assert 3e3 <= s["E"] <= 6e3 and 3.5 <= s["coup_friction"] <= 4.5
    assert DRConfig().sample_scene(rng) == {}                   # nothing set


# ── presets + yaml configs ──────────────────────────────────────────────────────
def test_presets():
    assert set(DR_PRESETS) == {"mild", "aggressive"}
    assert mild().has_reset_dr() and mild().has_scene_dr()
    assert aggressive().object_nu is not None


@pytest.mark.parametrize("name", ["mild", "aggressive"])
def test_yaml_configs_load(name):
    cfg = DRConfig.from_dict(yaml.safe_load((_DR_CFG / f"{name}.yaml").read_text()))
    assert not cfg.is_noop()
    # yaml floats like 3.0e3 parse and land in range form
    assert cfg.object_E[0] < cfg.object_E[1]
