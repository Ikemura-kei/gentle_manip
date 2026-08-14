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

    # Depth range (camera-frame z, meters). Points outside [depth_min, depth_max] are
    # discarded before backprojection. depth_max=1.0 is enough for our workspace
    # (farthest in-crop point ≤0.79m); it also eliminates env-leaking from parallel
    # Genesis envs (neighbour at ~2.8m) and floor hits at large camera angles.
    depth_min: float = 0.01
    depth_max: float = 3.0

    # Speed optimisation: randomly pre-select this many pixel positions from the depth
    # image BEFORE backprojection. Reduces the intermediate tensor from (N, H*W, 3) to
    # (N, pixel_sample_n, 3) — ~20x smaller for a 640×480 image. The final random
    # subsample to max_points is unaffected in quality (same distribution). The same
    # pixel subset is shared across all envs. None = disabled (process all pixels,
    # identical to prior behaviour). Recommended for training: 8 * max_points so after
    # crop (~30% of pixels survive) there are still ~2.5× the max_points target.
    pixel_sample_n: Optional[int] = None


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
class PrivilegedConfig:
    """SIM-ONLY privileged observations for a state-based RL *teacher* (distilled to a
    deployable point-cloud student later). Computed by PolicyEnv from SimFeedback —
    NOT the shared PerceptionPipeline — so real deployment can never produce them.
    Requires a task (sim). The student/real obs config must NOT set privileged."""
    object_pos: bool = False      # SimFeedback.object_center (num_envs, 3) — true object position
    object_quat: bool = False     # object orientation (num_envs, 4) wxyz — rigid only
    object_rot6d: bool = False    # object orientation as 6D (Zhou et al. 2019): first two columns
                                  # of rotation matrix concatenated → (num_envs, 6). Continuous,
                                  # singularity-free, better than quat for wide rotation ranges.
    object_vel: bool = False      # finite-diff object velocity (num_envs, 3)
    object_dr_params: bool = False  # episode-constant DR vector (num_envs, 2): [scale, bend_deg]
    stress: bool = False          # normalized von Mises [mean/yield, top10/yield] (num_envs, 2)
    contact_force: bool = False   # rigid-body surrogate for stress (num_envs,) — sum of contact
                                  # force magnitudes between the robot (gripper) and the object,
                                  # Newtons. Rigid bodies have no von Mises stress; this is the
                                  # analogous "how hard is the grip" scalar. Requires a rigid task.


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

    # Represent the EE orientation in the FINAL obs as a continuous 6D rotation (Zhou et al. 2019 —
    # first two rotation-matrix columns) INSTEAD of the wxyz quaternion. Derived in PerceptionPipeline
    # from the exact same (noise-augmented + sign-canonicalized) ee_quat, so sim/real are identical and
    # the quat augmentation flows through; only which one lands in the obs changes. NN-friendlier:
    # continuous + singularity-free + sign-invariant (rot6d(q)==rot6d(-q)). When True the obs key is
    # `ee_rot6d` (6,) and `ee_quat` is dropped.
    ee_rot6d: bool = False

    point_cloud: Optional[PointCloudConfig] = None
    voxel: Optional[VoxelConfig] = None
    images: Optional[ImageConfig] = None
    tactile: Optional[TactileConfig] = None
    privileged: Optional[PrivilegedConfig] = None   # sim teacher only (see PrivilegedConfig)

    #: optional modalities that a view can enable/disable (state = ee_pos/ee_quat/
    #: gripper_width is ALWAYS present and never masked).
    MODALITIES = ("joint_pos", "joint_vel", "point_cloud", "voxel", "images",
                  "tactile", "privileged")

    def needs_cameras(self) -> bool:
        """True if any modality requires camera rendering. Lets the sim backend skip
        the (expensive) per-env depth render when a state-based obs config doesn't use
        it — observation-only, so it never changes the physics/dynamics."""
        return self.point_cloud is not None or self.voxel is not None or self.images is not None

    def subview(self, modalities) -> "ObsConfig":
        """Return a copy with ONLY the named optional modalities kept — a *view* of this
        (superset) config. State (ee_pos/ee_quat/gripper_width + quat_noise_std) is always
        kept. All processing PARAMS are inherited unchanged, so a view can never drift
        from the superset the demos were recorded with; a view just selects which
        already-defined modalities the policy sees. Enables "same demo, different obs"
        (record the superset once; train teacher/student/ablations off subviews).
        """
        m = set(modalities)
        unknown = m - set(self.MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities {sorted(unknown)}; known: {self.MODALITIES}")
        return ObsConfig(
            include_joint_pos=self.include_joint_pos and "joint_pos" in m,
            include_joint_vel=self.include_joint_vel and "joint_vel" in m,
            quat_noise_std=self.quat_noise_std,
            ee_rot6d=self.ee_rot6d,
            point_cloud=self.point_cloud if "point_cloud" in m else None,
            voxel=self.voxel if "voxel" in m else None,
            images=self.images if "images" in m else None,
            tactile=self.tactile if "tactile" in m else None,
            privileged=self.privileged if "privileged" in m else None,
        )

    def obs_keys(self) -> list:
        """The obs-dict keys this config produces (matches PolicyEnv output). Used to
        subset a recorded superset demo down to a view."""
        keys = ["ee_pos", "ee_rot6d" if self.ee_rot6d else "ee_quat", "gripper_width"]
        if self.include_joint_pos:
            keys.append("joint_pos")
        if self.include_joint_vel:
            keys.append("joint_vel")
        if self.point_cloud is not None:
            keys.append("point_cloud")
        if self.voxel is not None:
            keys.append("voxel_grid")
        if self.images is not None:
            keys += [f"image_{c}" for c in self.images.cameras]
        if self.tactile is not None:
            keys += [f"tactile_{s}" for s in self.tactile.sensors]
        if self.privileged is not None:
            if self.privileged.object_pos:
                keys.append("priv_object_pos")
            if self.privileged.object_quat:
                keys.append("priv_object_quat")
            if self.privileged.object_rot6d:
                keys.append("priv_object_rot6d")
            if self.privileged.object_vel:
                keys.append("priv_object_vel")
            if self.privileged.object_dr_params:
                keys.append("priv_object_dr_params")
            if self.privileged.stress:
                keys.append("priv_stress")
            if self.privileged.contact_force:
                keys.append("priv_contact_force")
        return keys

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
                depth_min=float(pc_d.get("depth_min", 0.01)),
                depth_max=float(pc_d.get("depth_max", 3.0)),
                pixel_sample_n=pc_d.get("pixel_sample_n"),
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

        privileged = None
        if "privileged" in d:
            pv = d["privileged"] or {}
            privileged = PrivilegedConfig(
                object_pos=bool(pv.get("object_pos", False)),
                object_quat=bool(pv.get("object_quat", False)),
                object_rot6d=bool(pv.get("object_rot6d", False)),
                object_vel=bool(pv.get("object_vel", False)),
                object_dr_params=bool(pv.get("object_dr_params", False)),
                stress=bool(pv.get("stress", False)),
                contact_force=bool(pv.get("contact_force", False)),
            )

        cfg = cls(
            include_joint_pos=d.get("include_joint_pos", False),
            include_joint_vel=d.get("include_joint_vel", False),
            quat_noise_std=d.get("quat_noise_std", 0.0),
            ee_rot6d=d.get("ee_rot6d", False),
            point_cloud=pc,
            voxel=voxel,
            images=images,
            tactile=tactile,
            privileged=privileged,
        )
        cfg.validate()
        return cfg
