from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ObjectEntry:
    name: str                                       # key into OBJECT_MAP, e.g. "tofu", "waffle"
    object_type: str = "soft"                       # "soft" | "rigid"
    count: int = 1                                  # number of instances to spawn
    youngs_modulus: Optional[float] = None          # override default E (Pa)
    poisson_ratio: Optional[float] = None           # override default ν, in (0, 0.5)
    density: Optional[float] = None                 # override default ρ (kg/m³)
    pose_range: Optional[Dict[str, Tuple[float, float]]] = None  # {"x": (lo, hi), ...}
    scale: float = 1.0
    mesh_path: Optional[str] = None                 # per-scene mesh override (e.g. a shape-DR
                                                    # deformed .obj); None => registry default
    spawn_xy: Optional[Tuple[float, float]] = None  # override the registry default_pos x/y — the
                                                    # NOMINAL spawn. Iteration-2 recenters it on the
                                                    # real workspace centre so DR stays symmetric.
    spawn_z: Optional[float] = None                 # override the registry default_pos z (spawn
                                                    # height, m). None => registry default. Used to
                                                    # clear the MPM domain's inward safety padding
                                                    # at coarse grid_density (bigger cells => bigger
                                                    # padding => higher padded floor).

    def validate(self) -> None:
        if self.object_type not in ("soft", "rigid"):
            raise ValueError(
                f"ObjectEntry.object_type must be 'soft' or 'rigid', got {self.object_type!r}"
            )
        if self.count < 1:
            raise ValueError(f"ObjectEntry.count must be >= 1, got {self.count}")
        if self.youngs_modulus is not None and self.youngs_modulus <= 0:
            raise ValueError(f"ObjectEntry.youngs_modulus must be > 0, got {self.youngs_modulus}")
        if self.poisson_ratio is not None and not (0 < self.poisson_ratio < 0.5):
            raise ValueError(
                f"ObjectEntry.poisson_ratio must be in (0, 0.5), got {self.poisson_ratio}"
            )
        if self.density is not None and self.density <= 0:
            raise ValueError(f"ObjectEntry.density must be > 0, got {self.density}")
        if self.scale <= 0:
            raise ValueError(f"ObjectEntry.scale must be > 0, got {self.scale}")


@dataclass
class FixtureEntry:
    fixture_type: str                               # "table" | "platform" | "chopping_board" | "bin"
    pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    params: Dict[str, Any] = field(default_factory=dict)

    _VALID_TYPES = frozenset(("table", "platform", "chopping_board", "bin", "backdrop"))

    def validate(self) -> None:
        if self.fixture_type not in self._VALID_TYPES:
            raise ValueError(
                f"FixtureEntry.fixture_type must be one of {self._VALID_TYPES}, "
                f"got {self.fixture_type!r}"
            )
        if len(self.pose) != 3:
            raise ValueError(f"FixtureEntry.pose must be (x, y, z), got {self.pose}")


@dataclass
class CameraEntry:
    name: str                                       # must match real_lab.yaml and ObsConfig
    pos: Tuple[float, float, float] = (0.5, -0.5, 0.5)
    lookat: Tuple[float, float, float] = (0.3, 0.0, 0.1)
    fov: float = 40.0                               # degrees (Genesis fov is VERTICAL)
    resolution: Tuple[int, int] = (640, 480)        # (width, height)
    # World "up" for the image, i.e. MINUS the camera's y (image-down) axis. pos+lookat alone
    # leave the ROLL about the optical axis unconstrained, and Genesis's default (0,0,1) is ~3 deg
    # off the real D435i mounting. Setting this reproduces a calibrated world_T_cam EXACTLY:
    #   pos = T[:3,3]   lookat = pos + d*T[:3,2]   up = -T[:3,1]
    # (verified: cross(forward, up) reproduces T[:3,0] to 1e-6). None = Genesis default (0,0,1).
    up: Optional[Tuple[float, float, float]] = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("CameraEntry.name must not be empty")
        if len(self.pos) != 3:
            raise ValueError(f"CameraEntry.pos must be (x, y, z), got {self.pos}")
        if len(self.lookat) != 3:
            raise ValueError(f"CameraEntry.lookat must be (x, y, z), got {self.lookat}")
        if not (0 < self.fov < 180):
            raise ValueError(f"CameraEntry.fov must be in (0, 180), got {self.fov}")
        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise ValueError(f"CameraEntry.resolution must be positive, got {self.resolution}")
        if self.up is not None:
            if len(self.up) != 3:
                raise ValueError(f"CameraEntry.up must be (x, y, z), got {self.up}")
            if float(np.linalg.norm(self.up)) < 1e-9:
                raise ValueError("CameraEntry.up must not be a zero vector")


@dataclass
class SceneSpec:
    """
    Declarative description of a Genesis simulation scene.
    No Genesis imports — this is pure data.
    SceneBuilder translates this into Genesis API calls.
    """
    objects:  List[ObjectEntry]  = field(default_factory=list)
    fixtures: List[FixtureEntry] = field(default_factory=list)
    cameras:  List[CameraEntry]  = field(default_factory=list)

    # Genesis simulation parameters
    sim_dt:           float = 4e-3
    sim_substeps:     int   = 6
    plane_friction:   float = 1.0
    mpm_bounds:       Tuple[Tuple[float, float, float],
                            Tuple[float, float, float]] = (
                          (0.05, -0.26, -0.03),
                          (0.60,  0.26,  0.35),
                      )
    mpm_grid_density: float = 200.0
    # RGBA 0-1 for the gripper's finger + knuckle links, or None (URDF default, unchanged).
    # Visualisation only: the point cloud carries no colour. The real gripper's fingers are
    # black, so this is what makes sim RGB renders match the hardware in figures.
    finger_color: Optional[Tuple[float, float, float, float]] = None

    def validate(self) -> None:
        for obj in self.objects:
            obj.validate()
        for fix in self.fixtures:
            fix.validate()
        for cam in self.cameras:
            cam.validate()

        if self.sim_dt <= 0:
            raise ValueError(f"sim_dt must be > 0, got {self.sim_dt}")
        if self.sim_substeps < 1:
            raise ValueError(f"sim_substeps must be >= 1, got {self.sim_substeps}")
        if self.plane_friction < 0:
            raise ValueError(f"plane_friction must be >= 0, got {self.plane_friction}")
        if self.mpm_grid_density <= 0:
            raise ValueError(f"mpm_grid_density must be > 0, got {self.mpm_grid_density}")

        lo, hi = self.mpm_bounds
        if any(lo[i] >= hi[i] for i in range(3)):
            raise ValueError(f"mpm_bounds lo must be < hi, got lo={lo}, hi={hi}")

        cam_names = [c.name for c in self.cameras]
        if len(cam_names) != len(set(cam_names)):
            raise ValueError(f"Camera names must be unique, got {cam_names}")
