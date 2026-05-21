import pytest

from gentle_manip.scenes.scene_spec import (
    CameraEntry,
    FixtureEntry,
    ObjectEntry,
    SceneSpec,
)


# ── ObjectEntry ───────────────────────────────────────────────────────────────

def test_object_entry_defaults():
    obj = ObjectEntry(name="tofu")
    assert obj.object_type == "soft"
    assert obj.count == 1
    assert obj.youngs_modulus is None
    assert obj.poisson_ratio is None
    assert obj.density is None
    assert obj.scale == 1.0


def test_object_entry_full_override():
    obj = ObjectEntry(
        name="tofu",
        object_type="soft",
        count=2,
        youngs_modulus=500.0,
        poisson_ratio=0.4,
        density=1200.0,
        scale=0.8,
    )
    obj.validate()


def test_object_entry_rigid():
    ObjectEntry(name="spam", object_type="rigid").validate()


def test_object_entry_bad_type():
    with pytest.raises(ValueError, match="object_type"):
        ObjectEntry(name="tofu", object_type="fluid").validate()


def test_object_entry_bad_count():
    with pytest.raises(ValueError, match="count"):
        ObjectEntry(name="tofu", count=0).validate()


def test_object_entry_bad_youngs_modulus():
    with pytest.raises(ValueError, match="youngs_modulus"):
        ObjectEntry(name="tofu", youngs_modulus=-1.0).validate()


def test_object_entry_poisson_ratio_at_boundary():
    with pytest.raises(ValueError, match="poisson_ratio"):
        ObjectEntry(name="tofu", poisson_ratio=0.5).validate()  # must be < 0.5
    with pytest.raises(ValueError, match="poisson_ratio"):
        ObjectEntry(name="tofu", poisson_ratio=0.0).validate()  # must be > 0


def test_object_entry_bad_density():
    with pytest.raises(ValueError, match="density"):
        ObjectEntry(name="tofu", density=0.0).validate()


def test_object_entry_bad_scale():
    with pytest.raises(ValueError, match="scale"):
        ObjectEntry(name="tofu", scale=-0.5).validate()


# ── FixtureEntry ──────────────────────────────────────────────────────────────

def test_fixture_entry_all_valid_types():
    for ft in ("table", "platform", "chopping_board", "bin"):
        FixtureEntry(fixture_type=ft).validate()


def test_fixture_entry_bad_type():
    with pytest.raises(ValueError, match="fixture_type"):
        FixtureEntry(fixture_type="wall").validate()


def test_fixture_entry_with_params():
    FixtureEntry(
        fixture_type="platform",
        pose=(0.4, 0.1, 0.0),
        params={"height": 0.08},
    ).validate()


# ── CameraEntry ───────────────────────────────────────────────────────────────

def test_camera_entry_defaults():
    CameraEntry(name="cam_ext").validate()


def test_camera_entry_wrist():
    CameraEntry(
        name="cam_wrist",
        pos=(0.3, 0.0, 0.4),
        lookat=(0.3, 0.0, 0.0),
        fov=60.0,
        resolution=(640, 480),
    ).validate()


def test_camera_entry_empty_name():
    with pytest.raises(ValueError, match="name"):
        CameraEntry(name="").validate()


def test_camera_entry_bad_fov():
    with pytest.raises(ValueError, match="fov"):
        CameraEntry(name="cam_ext", fov=0.0).validate()
    with pytest.raises(ValueError, match="fov"):
        CameraEntry(name="cam_ext", fov=180.0).validate()


def test_camera_entry_bad_resolution():
    with pytest.raises(ValueError, match="resolution"):
        CameraEntry(name="cam_ext", resolution=(0, 480)).validate()


# ── SceneSpec ─────────────────────────────────────────────────────────────────

def make_valid_scene() -> SceneSpec:
    return SceneSpec(
        objects=[ObjectEntry(name="tofu", youngs_modulus=500.0,
                             poisson_ratio=0.4, density=1200.0)],
        fixtures=[FixtureEntry(fixture_type="table", pose=(0.3, 0.0, 0.0))],
        cameras=[
            CameraEntry(name="cam_wrist", pos=(0.3, -0.6, 0.5), lookat=(0.3, 0.0, 0.1)),
            CameraEntry(name="cam_ext",   pos=(0.6,  0.0, 0.5), lookat=(0.3, 0.0, 0.1)),
        ],
    )


def test_scene_spec_valid():
    make_valid_scene().validate()


def test_scene_spec_empty_lists_valid():
    SceneSpec().validate()


def test_scene_spec_bad_sim_dt():
    spec = make_valid_scene()
    spec.sim_dt = 0.0
    with pytest.raises(ValueError, match="sim_dt"):
        spec.validate()


def test_scene_spec_bad_sim_substeps():
    spec = make_valid_scene()
    spec.sim_substeps = 0
    with pytest.raises(ValueError, match="sim_substeps"):
        spec.validate()


def test_scene_spec_bad_mpm_bounds():
    spec = make_valid_scene()
    spec.mpm_bounds = ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0))  # lo.x > hi.x
    with pytest.raises(ValueError, match="mpm_bounds"):
        spec.validate()


def test_scene_spec_duplicate_camera_names():
    spec = make_valid_scene()
    spec.cameras.append(CameraEntry(name="cam_ext"))  # duplicate
    with pytest.raises(ValueError, match="unique"):
        spec.validate()


def test_scene_spec_multiple_objects():
    spec = SceneSpec(
        objects=[
            ObjectEntry(name="tofu",  object_type="soft",  count=2),
            ObjectEntry(name="spam",  object_type="rigid", count=1),
            ObjectEntry(name="waffle", object_type="soft", youngs_modulus=300.0,
                        poisson_ratio=0.3, density=900.0),
        ],
        fixtures=[
            FixtureEntry(fixture_type="table",    pose=(0.3, 0.0, 0.0)),
            FixtureEntry(fixture_type="bin",      pose=(0.5, 0.3, 0.0),
                         params={"size": (0.15, 0.15, 0.1)}),
            FixtureEntry(fixture_type="platform", pose=(0.4, -0.1, 0.0),
                         params={"height": 0.05}),
        ],
        cameras=[CameraEntry(name="cam_ext")],
    )
    spec.validate()
