"""Object registry: name -> ObjectDef (geometry + default material).

OBJECT_MAP is the single source of truth for what each object name spawns. For
the MVP every object is a primitive MPM box (no meshes yet — assets/meshes/objects
is empty); a mesh_path field is reserved for when real scanned meshes are added.

SceneBuilder reads OBJECT_MAP to turn a SceneSpec ObjectEntry (which may override
E/nu/rho/scale/pose) into Genesis morph + material calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from gentle_manip.assets.materials import MATERIALS, Material


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
}


def get_object_def(name: str) -> ObjectDef:
    if name not in OBJECT_MAP:
        raise KeyError(f"Unknown object {name!r}; known objects: {sorted(OBJECT_MAP)}")
    return OBJECT_MAP[name]
