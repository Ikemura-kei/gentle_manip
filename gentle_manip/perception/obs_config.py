from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PointCloudConfig:
    cameras: List[str]                      # e.g. ["cam_wrist", "cam_ext"]
    crop_min: Tuple[float, float, float]    # workspace bounding box min (meters)
    crop_max: Tuple[float, float, float]    # workspace bounding box max (meters)
    max_points: int = 2048                  # subsample target after merge + crop

    # Optional cloud-quality filters, applied (shared sim+real) AFTER crop and
    # BEFORE subsample, so the freed budget is reallocated to what's kept. Both
    # default off (None) — enable per obs config.
    #   outlier_voxel_size: density outlier removal — drop points whose voxel holds
    #     < outlier_min_neighbors valid points (removes L515 flying-pixel/edge noise).
    #   focus_z_lo / focus_r_ee: object focus — keep only points that are low
    #     (z < focus_z_lo) OR near the EE (within focus_r_ee), dropping the arm body.
    outlier_voxel_size: Optional[float] = None
    outlier_min_neighbors: int = 2
    focus_z_lo: Optional[float] = None
    focus_r_ee: float = 0.13


@dataclass
class VoxelConfig:
    cameras: List[str]
    voxel_size: float                       # meters per voxel edge
    crop_min: Tuple[float, float, float]
    crop_max: Tuple[float, float, float]


@dataclass
class ImageConfig:
    cameras: List[str]                      # which RGB streams to pass through


@dataclass
class TactileConfig:
    sensors: List[str]                      # e.g. ["tactile_left", "tactile_right"]
    # GelSight Mini images are passed through as-is (no processing in pipeline)


@dataclass
class ObsConfig:
    """
    Declares which modalities PerceptionPipeline includes in the output obs dict.

    Rules:
      - ee_pos, ee_quat, gripper_width are always included.
      - All other modalities are opt-in via the fields below.
      - tactile is real-only; set to None for any sim experiment.
      - point_cloud and voxel are mutually exclusive (use one or the other).

    Loaded from configs/obs/*.yaml via ObsConfig.from_dict().
    """
    include_joint_pos: bool = False
    include_joint_vel: bool = False

    # Small inherent quaternion noise, applied in PerceptionPipeline to ee_quat and
    # renormalized — SHARED across sim and real (unlike the sim-only PolicyEnv
    # augmentation). Keeps ee_quat from ever being an exact constant so a policy
    # never overfits one clean quaternion. 0.0 disables it.
    quat_noise_std: float = 0.0

    point_cloud: Optional[PointCloudConfig] = None
    voxel: Optional[VoxelConfig] = None
    images: Optional[ImageConfig] = None
    tactile: Optional[TactileConfig] = None

    def validate(self) -> None:
        if self.point_cloud is not None and self.voxel is not None:
            raise ValueError("point_cloud and voxel are mutually exclusive in ObsConfig")

    @classmethod
    def from_dict(cls, d: dict) -> ObsConfig:
        """Build from a plain dict (e.g. loaded via yaml.safe_load)."""
        pc = None
        if "point_cloud" in d:
            pc_d = d["point_cloud"]
            out_d = pc_d.get("outlier_removal") or {}
            foc_d = pc_d.get("object_focus") or {}
            pc = PointCloudConfig(
                cameras=pc_d["cameras"],
                crop_min=tuple(pc_d["crop_min"]),
                crop_max=tuple(pc_d["crop_max"]),
                max_points=pc_d.get("max_points", 2048),
                outlier_voxel_size=out_d.get("voxel_size"),
                outlier_min_neighbors=out_d.get("min_neighbors", 2),
                focus_z_lo=foc_d.get("z_lo"),
                focus_r_ee=foc_d.get("r_ee", 0.13),
            )

        voxel = None
        if "voxel" in d:
            v_d = d["voxel"]
            voxel = VoxelConfig(
                cameras=v_d["cameras"],
                voxel_size=v_d["voxel_size"],
                crop_min=tuple(v_d["crop_min"]),
                crop_max=tuple(v_d["crop_max"]),
            )

        images = None
        if "images" in d:
            images = ImageConfig(cameras=d["images"]["cameras"])

        tactile = None
        if "tactile" in d:
            tactile = TactileConfig(sensors=d["tactile"]["sensors"])

        cfg = cls(
            include_joint_pos=d.get("include_joint_pos", False),
            include_joint_vel=d.get("include_joint_vel", False),
            quat_noise_std=d.get("quat_noise_std", 0.0),
            point_cloud=pc,
            voxel=voxel,
            images=images,
            tactile=tactile,
        )
        cfg.validate()
        return cfg
