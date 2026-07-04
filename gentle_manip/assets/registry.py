"""Object registry: name -> ObjectDef (geometry + default material).

OBJECT_MAP is the single source of truth for what each object name spawns. Objects
are either a primitive MPM/rigid box (mesh_path=None) or a scanned mesh (mesh_path
set -> Genesis loads gs.morphs.Mesh). Scanned meshes live in assets/objects/ and are
stored in METERS, so no scale conversion is needed at the call site.

SceneBuilder reads OBJECT_MAP to turn a SceneSpec ObjectEntry (which may override
E/nu/rho/scale/pose) into Genesis morph + material calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from gentle_manip.assets.materials import MATERIALS, Material

_OBJ_DIR = Path(__file__).resolve().parent / "objects"


@dataclass(frozen=True)
class ObjectDef:
    name: str
    material: Material
    object_type: str = "soft"                       # "soft" (MPM) | "rigid"
    size: Tuple[float, float, float] = (0.04, 0.04, 0.04)   # primitive box extents (m)
    default_pos: Tuple[float, float, float] = (0.50, 0.0, 0.03)  # resting pose (m)
    mesh_path: Optional[str] = None                 # reserved; None => primitive box


# The validated baseline matches examples/gs_sim_backend_dev.py: a 4 cm tofu cube
# resting at (0.50, 0, 0.03), within reach and the MPM bounds.
OBJECT_MAP: dict[str, ObjectDef] = {
    "tofu":    ObjectDef("tofu", MATERIALS["tofu"]),
    "gelatin": ObjectDef("gelatin", MATERIALS["gelatin"]),
    "sponge":  ObjectDef("sponge", MATERIALS["sponge"]),
    # 3 cm cube matching the real red cube the DP3 policy was trained on; rests on
    # the table (half-extent 0.015 -> center z ~0.015 after settling).
    "red_cube": ObjectDef("red_cube", MATERIALS["red_cube"],
                          size=(0.03, 0.03, 0.03), default_pos=(0.47, 0.011, 0.02)),
    # Gripper-width calibration cubes (use as rigid): known sizes spread across x so
    # all four fit in one scene; spawn z = half-extent so they rest on the table.
    "cal_cube_6": ObjectDef("cal_cube_6", MATERIALS["red_cube"], size=(0.06, 0.06, 0.06), default_pos=(0.34, 0.0, 0.035)),
    "cal_cube_5": ObjectDef("cal_cube_5", MATERIALS["red_cube"], size=(0.05, 0.05, 0.05), default_pos=(0.42, 0.0, 0.030)),
    "cal_cube_4": ObjectDef("cal_cube_4", MATERIALS["red_cube"], size=(0.04, 0.04, 0.04), default_pos=(0.50, 0.0, 0.025)),
    "cal_cube_3": ObjectDef("cal_cube_3", MATERIALS["red_cube"], size=(0.03, 0.03, 0.03), default_pos=(0.58, 0.0, 0.020)),
    # Real scanned edible mushroom (assets/objects/mushroom.obj, in meters: ~3.3 x 3.2
    # x 3.5 cm). Soft MPM body; rests on the table (mesh bottom at z=-0.0148, so
    # default_pos z=0.016 puts it just above the surface). size is informational only
    # (ignored for meshes). Material is the soft-end "Config C" (see materials.py).
    "mushroom": ObjectDef("mushroom", MATERIALS["mushroom"], object_type="soft",
                          size=(0.033, 0.032, 0.035), default_pos=(0.47, 0.0, 0.016),
                          mesh_path=str(_OBJ_DIR / "mushroom.obj")),
}


def get_object_def(name: str) -> ObjectDef:
    if name not in OBJECT_MAP:
        raise KeyError(f"Unknown object {name!r}; known objects: {sorted(OBJECT_MAP)}")
    return OBJECT_MAP[name]
