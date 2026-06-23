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
