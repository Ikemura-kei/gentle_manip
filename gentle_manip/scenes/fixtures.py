"""Fixture builders: SceneSpec FixtureEntry -> Genesis geometry.

The scene's ground plane already serves as the table top (robot base sits on the
table at the world origin), so a "table" fixture adds nothing. The raised
fixtures (platform / chopping_board / bin) are fixed rigid boxes. Genesis import
is local to keep this a sim-only module.
"""
from __future__ import annotations

from typing import Iterable, List

import genesis as gs


def add_fixtures(scene, fixtures: Iterable) -> List:
    """Add fixture geometry to ``scene``; return the created entities (table -> none)."""
    built: List = []
    for f in fixtures:
        if f.fixture_type == "table":
            continue  # ground plane is the table surface
        elif f.fixture_type in ("platform", "chopping_board"):
            h = float(f.params.get("height", 0.05))
            size = tuple(f.params.get("size", (0.15, 0.15, h)))
            built.append(
                scene.add_entity(
                    gs.morphs.Box(size=size, pos=tuple(f.pose), fixed=True),
                    material=gs.materials.Rigid(),
                )
            )
        elif f.fixture_type == "backdrop":
            # ENV-LEAKAGE OCCLUDER (2026-08-30). Soft/MPM scenes must use the per-env bound-camera
            # path with env_separate_rigid=False (the rasterizer cannot separate MPM geometry per
            # env), so at ENV_SPACING=2.5 m every env's cam_ext sees its NEIGHBOURS on the horizon
            # — and they MOVE, so they are dynamic distractors in the RGB observation. The point
            # cloud was never affected (it is cropped), which is why this only surfaced with RGB.
            #
            # A wall is preferable to raising ENV_SPACING (neighbours shrink but stay in frame) and
            # far cheaper than num_envs=1 (~8x slower): it also makes the scene more real-lab-like,
            # so it REDUCES the sim2real gap rather than merely hiding a sim artifact.
            #
            # Placement must sit OUTSIDE the point-cloud crop (crop_max x 0.71, |y| 0.215) so that
            # every existing point-cloud experiment is bit-identical — verified by the caller.
            size = tuple(f.params.get("size", (0.02, 3.0, 1.5)))
            # BLACK, not the default light surface: the XArm is white, so a bright wall gives no
            # contrast and blows out cam_ext (the first version did exactly that). Dark background
            # + white arm + lit object is also closer to the real lab rig.
            col = tuple(f.params.get("color", (0.02, 0.02, 0.02, 1.0)))
            built.append(
                scene.add_entity(
                    gs.morphs.Box(size=size, pos=tuple(f.pose), fixed=True),
                    material=gs.materials.Rigid(),
                    surface=gs.surfaces.Default(color=col),
                )
            )
        elif f.fixture_type == "bin":
            # TODO: real open-top bin; a solid box placeholder for now.
            size = tuple(f.params.get("size", (0.2, 0.2, 0.08)))
            built.append(
                scene.add_entity(
                    gs.morphs.Box(size=size, pos=tuple(f.pose), fixed=True),
                    material=gs.materials.Rigid(),
                )
            )
    return built
