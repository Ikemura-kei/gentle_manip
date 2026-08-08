from pathlib import Path

import numpy as np
import pytest
import yaml

from gentle_manip.assets.registry import get_object_def
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


def test_sample_object_euler_shape_range_and_disabled():
    rng = np.random.default_rng(0)
    assert DRConfig().sample_object_euler(rng, 8) is None        # disabled -> None
    assert not DRConfig().has_reset_dr()
    cfg = DRConfig(object_yaw_deg=180, object_pitch_roll_deg=15)
    assert cfg.has_reset_dr()
    e = cfg.sample_object_euler(rng, 8)                          # (roll, pitch, yaw) radians
    assert e.shape == (8, 3) and e.dtype == np.float32
    assert np.all(np.abs(e[:, :2]) <= np.deg2rad(15) + 1e-6)     # pitch & roll within +/-15 deg
    assert np.all(np.abs(e[:, 2]) <= np.deg2rad(180) + 1e-6)     # yaw within +/-180 deg
    # yaw-only / tilt-only each still count as reset DR
    assert DRConfig(object_yaw_deg=90).sample_object_euler(rng, 4).shape == (4, 3)


def test_sample_scene_only_set_keys_and_in_range():
    rng = np.random.default_rng(0)
    cfg = DRConfig(object_E=(3e3, 6e3), coup_friction=(3.5, 4.5))
    for _ in range(50):
        s = cfg.sample_scene(rng)
        assert set(s) == {"E", "coup_friction"}                 # only randomized keys
        assert 3e3 <= s["E"] <= 6e3 and 3.5 <= s["coup_friction"] <= 4.5
    assert DRConfig().sample_scene(rng) == {}                   # nothing set


def test_sample_shape_scale_ranges_and_units():
    rng = np.random.default_rng(0)
    assert DRConfig().sample_shape_scale(rng) == {}             # disabled -> empty
    assert not DRConfig().has_shape_dr()
    cfg = DRConfig(object_scale=(0.8, 1.2), object_bend_deg=(-25, 25), object_taper=(-0.1, 0.1))
    assert cfg.has_scene_dr() and cfg.has_shape_dr()
    for _ in range(50):
        s = cfg.sample_shape_scale(rng)
        assert set(s) == {"scale", "bend", "taper"}
        assert 0.8 <= s["scale"] <= 1.2
        assert abs(s["bend"]) <= np.deg2rad(25) + 1e-6          # deg -> radians
        assert abs(s["taper"]) <= 0.1
    # scale-only counts as scene DR but not shape DR
    assert DRConfig(object_scale=(0.9, 1.1)).has_scene_dr()
    assert not DRConfig(object_scale=(0.9, 1.1)).has_shape_dr()


# ── cross-category (Stage 2) ──────────────────────────────────────────────────
def test_sample_category_disabled_returns_none():
    rng = np.random.default_rng(0)
    assert DRConfig().sample_category(rng) is None
    assert not DRConfig().has_scene_dr()


def test_from_dict_category_pool_stays_strings_not_floats():
    cfg = DRConfig.from_dict({"object_category_pool": ["mushroom", "raspberry"]})
    assert cfg.object_category_pool == ("mushroom", "raspberry")
    assert all(isinstance(x, str) for x in cfg.object_category_pool)
    assert cfg.has_scene_dr()          # a category pool alone counts as scene DR


def test_sample_category_only_draws_from_pool():
    rng = np.random.default_rng(0)
    cfg = DRConfig(object_category_pool=("mushroom", "raspberry"))
    draws = {cfg.sample_category(rng) for _ in range(100)}
    assert draws == {"mushroom", "raspberry"}      # both drawn, nothing else


def test_sample_category_determinism():
    cfg = DRConfig(object_category_pool=("mushroom", "raspberry"))
    rng_a, rng_b = np.random.default_rng(42), np.random.default_rng(42)
    a = [cfg.sample_category(rng_a) for _ in range(20)]
    b = [cfg.sample_category(rng_b) for _ in range(20)]
    assert a == b                                   # same seed -> identical draw sequence


def test_sample_category_weights_skew_distribution():
    rng = np.random.default_rng(0)
    cfg = DRConfig(object_category_pool=("mushroom", "raspberry"),
                   object_category_weights=(0.95, 0.05))
    draws = [cfg.sample_category(rng) for _ in range(500)]
    assert draws.count("mushroom") > draws.count("raspberry") * 5


def test_sample_scene_category_fallback_is_relative_to_category_nominal():
    rng = np.random.default_rng(0)
    cfg = DRConfig(object_category_pool=("mushroom", "raspberry"))   # no absolute material fields
    mushroom, raspberry = get_object_def("mushroom"), get_object_def("raspberry")
    for _ in range(50):
        m = cfg.sample_scene(rng, category=mushroom)
        lo, hi = mushroom.material_dr_mult["E"]
        assert lo * mushroom.material.youngs_modulus <= m["E"] <= hi * mushroom.material.youngs_modulus
        r = cfg.sample_scene(rng, category=raspberry)
        lo, hi = raspberry.material_dr_mult["yield"]
        nominal = raspberry.material.von_mises_yield_stress
        assert lo * nominal <= r["yield"] <= hi * nominal
    # coup_friction has no per-category fallback by design
    assert "coup_friction" not in cfg.sample_scene(rng, category=mushroom)


def test_sample_scene_absolute_dr_overrides_category_fallback():
    rng = np.random.default_rng(0)
    cfg = DRConfig(object_E=(1.0, 2.0))     # explicit absolute range, well outside any category's Pa scale
    mushroom = get_object_def("mushroom")
    for _ in range(20):
        s = cfg.sample_scene(rng, category=mushroom)
        assert 1.0 <= s["E"] <= 2.0         # absolute field wins over category_dr_mult


def test_sample_shape_scale_category_fallback_uses_category_ranges():
    rng = np.random.default_rng(0)
    cfg = DRConfig(object_category_pool=("mushroom", "raspberry"))
    raspberry = get_object_def("raspberry")
    for _ in range(50):
        s = cfg.sample_shape_scale(rng, category=raspberry)
        assert set(s) == {"scale", "bend", "twist", "taper", "rbf", "axis_scale", "axis_scale_ax"}
        lo, hi = raspberry.shape_dr_ranges["rbf"]
        assert lo <= s["rbf"] <= hi
    # a category with no rbf range (mushroom) never produces an "rbf" key
    mushroom = get_object_def("mushroom")
    assert "rbf" not in cfg.sample_shape_scale(rng, category=mushroom)


def test_sample_shape_scale_no_category_is_unchanged():
    rng = np.random.default_rng(0)
    cfg = DRConfig(object_scale=(0.8, 1.2), object_bend_deg=(-25, 25))
    assert cfg.sample_shape_scale(rng, category=None) == cfg.sample_shape_scale(np.random.default_rng(0), category=None)


# ── presets + yaml configs ──────────────────────────────────────────────────────
def test_presets():
    assert set(DR_PRESETS) == {"mild", "aggressive", "cross_category_food"}
    assert mild().has_reset_dr() and mild().has_scene_dr()
    assert aggressive().object_nu is not None


def test_cross_category_food_preset():
    from gentle_manip.domain_randomization import cross_category_food

    cfg = cross_category_food()
    assert cfg.object_category_pool == (
        "mushroom", "raspberry", "apple", "pear", "grape", "kiwi", "cherry",
        "blueberry", "egg", "avocado")
    assert cfg.has_scene_dr() and cfg.has_reset_dr()
    # absolute material/shape fields deliberately unset -> per-category fallback applies
    assert cfg.object_E is None and cfg.object_bend_deg is None
    cfg2 = cross_category_food(categories=("mushroom",), weights=(1.0,))
    rng = np.random.default_rng(0)
    assert all(cfg2.sample_category(rng) == "mushroom" for _ in range(5))


@pytest.mark.parametrize("name", ["mild", "aggressive"])
def test_yaml_configs_load(name):
    cfg = DRConfig.from_dict(yaml.safe_load((_DR_CFG / f"{name}.yaml").read_text()))
    assert not cfg.is_noop()
    # yaml floats like 3.0e3 parse and land in range form
    assert cfg.object_E[0] < cfg.object_E[1]


def test_cross_category_food_yaml_loads():
    cfg = DRConfig.from_dict(yaml.safe_load((_DR_CFG / "cross_category_food.yaml").read_text()))
    assert not cfg.is_noop()
    assert len(cfg.object_category_pool) == 10
    assert "mushroom" in cfg.object_category_pool and "avocado" in cfg.object_category_pool
    assert cfg.object_E is None    # per-category fallback, not an absolute yaml range
