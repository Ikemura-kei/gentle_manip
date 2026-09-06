"""Stochastic observation augmentation — shared by sim and real-imitation.

Lives in the perception layer so the *same* transform is applied wherever the obs
dict is built; it never changes shapes (only values), so the obs space is
unaffected. All knobs default to 0 / False (no-op). Used two ways:
  - sim policy training: make the policy robust to the sim2real gap.
  - deploy-on-sim: noise the clean sim point cloud toward the L515 distribution so a
    real-trained policy sees a familiar input.

Batched: every field carries a leading num_envs dim. The RNG advances each call, so
consecutive steps get independent noise.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np


@dataclass
class AugmentationConfig:
    # point cloud
    pc_jitter_std: float = 0.0     # per-point Gaussian (m)
    pc_dropout: float = 0.0        # fraction of points replaced by duplicates (mimic missing returns)
    pc_offset_std: float = 0.0     # per-cloud rigid offset (m)
    # ── D435i depth-noise model (2026-09-03) ─────────────────────────────────────────────
    # `pc_jitter_std` is ISOTROPIC, which no depth camera produces. A stereo camera's error is
    # ALONG THE VIEWING RAY and grows with the SQUARE of range (disparity->depth inversion):
    #     sigma_axial(d) = pc_axial_coeff * d^2
    # MEASURED on our D435i (temporal std over 12 frames of a static board, by distance):
    #     0.45 m -> 0.373 mm | 0.55 m -> 0.553 | 0.675 m -> 0.943 | 0.85 m -> 1.891
    # which fits pc_axial_coeff ~= 2.0e-3 m/m^2 (effective subpixel ~0.06, i.e. this camera is
    # BETTER than the usual 0.1 rule of thumb). Ray direction needs the camera origin, so
    # pc_axial_cam_pos must be the world-fixed cam_ext position; zero disables the term.
    # Lateral (in-image) error is the pixel footprint d/f quantised: sigma_lat = coeff_lat * d,
    # ~0.24 mm at 0.5 m — an order below axial, so it defaults off.
    pc_axial_coeff: float = 0.0            # m per m^2, along the camera ray
    pc_lateral_coeff: float = 0.0          # m per m, perpendicular to the ray
    pc_axial_cam_pos: tuple = (0.0, 0.0, 0.0)
    # ── Leaked table residue (2026-09-06) ────────────────────────────────────────────────
    # The real D435i cloud carries small flat clusters just above the board that the voxel outlier
    # filter keeps (compact crumbs / board residue). Measured on a real deploy (tzdhk_2000 ep0, at the
    # 1024-point scale, after the deploy filter's own definition): 1-12 points per cluster (median 3),
    # 1-4 mm thick, 4-13 mm across, ALL between 19 and 29 mm height, 1-3 clusters in 34 % of frames,
    # anywhere on the board 3-15 cm from the gripper. 5 such points shifted the policy's gripper
    # command 4-7 mm toward open. The real pipeline now removes most of them (ground_residual), so
    # this augmentation covers what that filter cannot: the policy must be indifferent to them.
    # Sim-only; points are REPLACED (cloud size unchanged). Sizes are in points of the stored cloud.
    pc_residue_prob: float = 0.0           # per-cloud probability of adding residue; 0 = off
    pc_residue_clusters: tuple = (1, 3)    # clusters per affected cloud (inclusive)
    pc_residue_points: tuple = (1, 8)      # points per cluster (inclusive); measured median 3, max 12 —
                                           # neighbouring clusters merge, which supplies the 9-12+ tail
    pc_residue_z: tuple = (0.019, 0.032)   # height band (m) — generous cap over the measured 29 mm max
    pc_residue_extent: tuple = (0.004, 0.015)   # cluster footprint across (m)
    pc_residue_thickness: float = 0.003    # vertical spread (m)
    pc_residue_xy_min: tuple = (0.22, -0.20)    # placement region (m) — the board inside the crop
    pc_residue_xy_max: tuple = (0.50, 0.20)
    # Patch dropout: with pc_patch_prob per cloud, every point within a random sphere (radius drawn in
    # pc_patch_radius, centred on a random valid point) is replaced by duplicates of other points —
    # an occlusion / missing-return blob, as opposed to pc_dropout's scattered misses. Sim-only.
    pc_patch_prob: float = 0.0
    pc_patch_radius: tuple = (0.005, 0.01)   # max 1 cm (2026-09-06): a 3 cm patch could swallow a small object
    pc_patch_z_min: float = 0.10             # patch centre AND removed points only above this height (arm body),
                                             # never the object on the board or a low-held object
    # low-dim state
    ee_pos_std: float = 0.0        # m
    ee_quat_std: float = 0.0       # additive quat noise (renormalized)
    gripper_std: float = 0.0       # m
    joint_std: float = 0.0         # rad (joint_pos / joint_vel)
    # representation
    quat_sign_flip: bool = False   # randomly negate ee_quat (q == -q double cover)
    # Cleaning (not noise): snap ee_quat elements within eps of {-1,0,1} to those
    # exact values, then renormalize — makes sim's quaternion as clean as the real
    # demos (which are axis-aligned and noise-free). Only valid when the orientation
    # stays near axis-aligned (e.g. gripper pointing straight down).
    quat_snap: bool = False
    quat_snap_eps: float = 0.05
    seed: int = 0

    def is_noop(self) -> bool:
        return not (self.pc_jitter_std or self.pc_dropout or self.pc_offset_std
                    or self.pc_axial_coeff or self.pc_lateral_coeff or self.pc_residue_prob or self.pc_patch_prob
                    or self.ee_pos_std or self.ee_quat_std or self.gripper_std
                    or self.joint_std or self.quat_sign_flip or self.quat_snap)

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentationConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class ObsAugmentor:
    """Apply an AugmentationConfig to a batched obs dict in place; returns the dict."""

    def __init__(self, cfg: AugmentationConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.last_residue_hit = None      # (N,) bool after each call: envs whose cloud got residue clusters

    def __call__(self, obs: dict) -> dict:
        c = self.cfg
        self.last_residue_hit = None
        if "point_cloud" in obs and (c.pc_jitter_std or c.pc_dropout or c.pc_offset_std
                                     or c.pc_axial_coeff or c.pc_lateral_coeff or c.pc_residue_prob
                                     or c.pc_patch_prob):
            obs["point_cloud"] = self._point_cloud(obs["point_cloud"])
        if "ee_pos" in obs and c.ee_pos_std:
            obs["ee_pos"] = (obs["ee_pos"] + self._n(obs["ee_pos"].shape, c.ee_pos_std)).astype(np.float32)
        if "ee_quat" in obs and (c.ee_quat_std or c.quat_sign_flip or c.quat_snap):
            obs["ee_quat"] = self._quat(obs["ee_quat"])
        if "gripper_width" in obs and c.gripper_std:
            gw = obs["gripper_width"] + self._n(obs["gripper_width"].shape, c.gripper_std)
            obs["gripper_width"] = np.maximum(gw, 0.0).astype(np.float32)
        for k in ("joint_pos", "joint_vel"):
            if k in obs and c.joint_std:
                obs[k] = (obs[k] + self._n(obs[k].shape, c.joint_std)).astype(np.float32)
        return obs

    # ── helpers ───────────────────────────────────────────────────────────────
    def _n(self, shape, std) -> np.ndarray:
        return self.rng.normal(0.0, std, shape).astype(np.float32)

    def _point_cloud(self, pc: np.ndarray) -> np.ndarray:
        c = self.cfg
        pc = pc.astype(np.float32).copy()                       # (N, P, 3)
        N, P, _ = pc.shape
        if c.pc_dropout > 0:
            k = int(P * c.pc_dropout)
            if k > 0:
                for i in range(N):  # replace k points with duplicates of random kept points
                    drop = self.rng.choice(P, size=k, replace=False)
                    pc[i, drop] = pc[i, self.rng.integers(0, P, size=k)]
        if c.pc_axial_coeff > 0 or c.pc_lateral_coeff > 0:
            # Ray-aligned, range-dependent noise — the physical stereo model.
            cam = np.asarray(c.pc_axial_cam_pos, np.float32).reshape(1, 1, 3)
            v = pc - cam                                        # camera -> point
            d = np.linalg.norm(v, axis=-1, keepdims=True)       # (N, P, 1) range
            u = v / np.maximum(d, 1e-6)                         # unit ray
            if c.pc_axial_coeff > 0:                            # sigma ~ coeff * d^2, along u
                pc += u * (self.rng.normal(0.0, 1.0, d.shape).astype(np.float32)
                           * (c.pc_axial_coeff * d * d))
            if c.pc_lateral_coeff > 0:                          # sigma ~ coeff * d, perp to u
                g = self.rng.normal(0.0, 1.0, pc.shape).astype(np.float32)
                g -= u * np.sum(g * u, axis=-1, keepdims=True)  # project out the radial part
                pc += g * (c.pc_lateral_coeff * d)
        if c.pc_jitter_std > 0:
            pc += self._n(pc.shape, c.pc_jitter_std)
        if c.pc_offset_std > 0:
            pc += self._n((N, 1, 3), c.pc_offset_std)
        if c.pc_patch_prob > 0:
            pc = self._patch_dropout(pc)
        if c.pc_residue_prob > 0:
            pc = self._residue(pc)
        return pc

    def _patch_dropout(self, pc: np.ndarray) -> np.ndarray:
        c, r = self.cfg, self.rng
        N, P, _ = pc.shape
        for i in range(N):
            if r.random() >= c.pc_patch_prob:
                continue
            valid = np.nonzero(np.any(pc[i] != 0, axis=1))[0]
            high = valid[pc[i, valid, 2] > c.pc_patch_z_min]           # candidates: above z_min only
            if high.size < 2:
                continue
            centre = pc[i, r.choice(high)]
            hit = high[np.linalg.norm(pc[i, high] - centre, axis=1) < r.uniform(*c.pc_patch_radius)]
            keep = np.setdiff1d(valid, hit)
            if hit.size and keep.size:
                pc[i, hit] = pc[i, r.choice(keep, hit.size)]
        return pc

    def _residue(self, pc: np.ndarray) -> np.ndarray:
        """Leaked-table-residue clusters (see AugmentationConfig): flat, tiny, low; replace points."""
        c, r = self.cfg, self.rng
        N, P, _ = pc.shape
        hit = np.zeros(N, dtype=bool)
        for i in range(N):
            if r.random() >= c.pc_residue_prob:
                continue
            hit[i] = True
            for _ in range(int(r.integers(c.pc_residue_clusters[0], c.pc_residue_clusters[1] + 1))):
                k = int(r.integers(c.pc_residue_points[0], c.pc_residue_points[1] + 1))
                center = np.array([r.uniform(c.pc_residue_xy_min[0], c.pc_residue_xy_max[0]),
                                   r.uniform(c.pc_residue_xy_min[1], c.pc_residue_xy_max[1]),
                                   r.uniform(*c.pc_residue_z)], np.float32)
                ext = r.uniform(*c.pc_residue_extent)
                q = center + np.stack([r.uniform(-ext / 2, ext / 2, k), r.uniform(-ext / 2, ext / 2, k),
                                       r.uniform(-c.pc_residue_thickness / 2, c.pc_residue_thickness / 2, k)], 1)
                q[:, 2] = np.clip(q[:, 2], c.pc_residue_z[0], None)
                pc[i, r.choice(P, size=k, replace=False)] = q.astype(np.float32)
        self.last_residue_hit = hit
        return pc

    def _quat(self, q: np.ndarray) -> np.ndarray:
        c = self.cfg
        q = q.astype(np.float32).copy()                         # (N, 4) wxyz
        if c.quat_snap:                                         # clean toward axis-aligned
            snapped = np.round(q)                              # nearest of {-1, 0, 1}
            near = np.abs(q - snapped) < c.quat_snap_eps
            q = np.where(near, snapped, q).astype(np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
        if c.ee_quat_std > 0:
            q = q + self._n(q.shape, c.ee_quat_std)
            q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
        if c.quat_sign_flip:
            flip = self.rng.random(q.shape[0]) < 0.5            # q and -q are the same orientation
            q[flip] = -q[flip]
        return q


def build_augmentor(cfg: Optional[AugmentationConfig]) -> Optional[ObsAugmentor]:
    """Return an ObsAugmentor, or None if there's nothing to do."""
    if cfg is None or cfg.is_noop():
        return None
    return ObsAugmentor(cfg)
