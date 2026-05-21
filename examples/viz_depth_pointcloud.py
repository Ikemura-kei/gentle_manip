"""
Visualization of depth_to_pointcloud with synthetic known geometry.

Scene (top-down camera, identity extrinsics):
  - Background plane at 1.0m
  - Raised box (centre 1/3 of image) at 0.7m  → should appear as elevated slab
  - Two corner pillars at 0.5m                 → should appear tallest

Run from repo root:
    python examples/viz_depth_pointcloud.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from gentle_manip.perception.depth_to_pointcloud import depth_to_pointcloud


# ── Camera (identity: at origin, looking straight down +Z) ───────────────────
H, W = 240, 320
fx = fy = 280.0
cx, cy = W / 2.0, H / 2.0

K = np.array([[fx,  0., cx],
              [ 0., fy, cy],
              [ 0.,  0., 1.]], dtype=np.float32)

# Camera at origin looking in +Z — world frame == camera frame
T = np.eye(4, dtype=np.float32)


# ── Synthetic depth image ─────────────────────────────────────────────────────
# Depths represent distance from camera along Z.
# Smaller depth = closer to camera = appears "taller" in world.

depth = np.full((H, W), 1.0, dtype=np.float32)   # background plane at 1.0m

# Raised box: centre third of image, 0.3m closer than background
h1, h2 = H // 3, 2 * H // 3
w1, w2 = W // 3, 2 * W // 3
depth[h1:h2, w1:w2] = 0.7

# Left pillar: top-left corner, 0.5m from camera
depth[:H // 5, :W // 6] = 0.5

# Right pillar: top-right corner, 0.5m from camera
depth[:H // 5, 5 * W // 6:] = 0.5

# Add a little noise to the background so it looks natural
rng = np.random.default_rng(0)
depth += rng.uniform(-0.005, 0.005, depth.shape).astype(np.float32)


# ── Unproject ─────────────────────────────────────────────────────────────────
depth_batch = depth[np.newaxis]   # (1, H, W)

pts_vec,  valid_vec  = depth_to_pointcloud(depth_batch, K, T,
                                           depth_min=0.01, vectorized=True)
pts_loop, valid_loop = depth_to_pointcloud(depth_batch, K, T,
                                           depth_min=0.01, vectorized=False)

pts = pts_vec[0][valid_vec[0]]     # (N, 3) valid points, world frame

print(f"Valid points: {pts.shape[0]} / {H * W}")
print(f"Vectorized == loop: {np.allclose(pts_vec, pts_loop, atol=1e-5)}")
print(f"World-frame bounding box:")
print(f"  X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] m")
print(f"  Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] m")
print(f"  Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}] m")

# Expected: pillars at z≈0.5, box at z≈0.7, background at z≈1.0
z_vals = np.unique(np.round(pts[:, 2], 1))
print(f"Distinct depth layers (rounded): {z_vals}")


# ── Visualise ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))
fig.suptitle("depth_to_pointcloud — synthetic scene (identity camera)", fontsize=13)

# 1. Depth image
ax1 = fig.add_subplot(1, 3, 1)
im = ax1.imshow(depth, cmap="plasma_r", origin="upper",
                vmin=depth.min(), vmax=depth.max())
plt.colorbar(im, ax=ax1, label="depth (m)")
ax1.set_title("Depth image\n(dark = closer)")
ax1.set_xlabel("u (px)")
ax1.set_ylabel("v (px)")

# Annotate regions
ax1.text(W // 2, (h1 + h2) // 2, "box\n0.7m",
         ha="center", va="center", color="white", fontsize=8)
ax1.text(W // 12, H // 10, "pillar\n0.5m",
         ha="center", va="center", color="white", fontsize=7)
ax1.text(11 * W // 12, H // 10, "pillar\n0.5m",
         ha="center", va="center", color="white", fontsize=7)

# 2. Validity mask
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(valid_vec[0].reshape(H, W), cmap="gray", origin="upper")
ax2.set_title(f"Valid pixel mask\n({pts.shape[0]} / {H*W} valid)")
ax2.set_xlabel("u (px)")
ax2.set_ylabel("v (px)")

# 3. 3D point cloud coloured by Z
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
stride = 3
p = pts[::stride]
sc = ax3.scatter(p[:, 0], p[:, 1], p[:, 2],
                 c=p[:, 2], cmap="plasma_r",
                 s=0.8, alpha=0.7,
                 vmin=pts[:, 2].min(), vmax=pts[:, 2].max())
plt.colorbar(sc, ax=ax3, label="Z (m)", shrink=0.6)
ax3.set_title("Point cloud (world frame)\ncoloured by depth")
ax3.set_xlabel("X (m)")
ax3.set_ylabel("Y (m)")
ax3.set_zlabel("Z (m)")
ax3.view_init(elev=30, azim=-60)

plt.tight_layout()
out = Path("examples/depth_pointcloud_viz.png")
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")
plt.show()
