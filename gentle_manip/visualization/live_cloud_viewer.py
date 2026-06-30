"""Non-blocking Open3D window that displays an externally-supplied point cloud,
updated each step. Unlike point_cloud_viewer (which owns the camera and shows the
RAW cloud for crop tuning), this is fed the already-PROCESSED cloud — so during demo
collection you watch the exact cloud the policy will see (crop + filters applied),
without the collect-then-visualize round trip.

Runs in the deploy env (open3d from the `real` extra). Imports open3d lazily.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class LiveCloudViewer:
    def __init__(self, crop_min: Optional[Sequence[float]] = None,
                 crop_max: Optional[Sequence[float]] = None,
                 title: str = "processed point cloud (live)",
                 z_range: tuple = (0.0, 0.45)) -> None:
        import open3d as o3d

        self._o3d = o3d
        try:                                   # matplotlib >= 3.7
            from matplotlib import colormaps
            self._cmap = colormaps["viridis"]
        except Exception:                      # older matplotlib
            import matplotlib.cm as cm
            self._cmap = cm.get_cmap("viridis")
        self._zlo, self._zhi = z_range
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(title, width=960, height=720)
        self.pcd = o3d.geometry.PointCloud()
        self._added = False
        # Reference geometry: base-frame axes + (optional) workspace crop box.
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        if crop_min is not None and crop_max is not None:
            box = o3d.geometry.AxisAlignedBoundingBox(
                np.asarray(crop_min, float), np.asarray(crop_max, float))
            box.color = (1.0, 0.0, 0.0)
            self.vis.add_geometry(box)
        self.alive = True

    def update(self, points: np.ndarray) -> bool:
        """Show ``points`` (M, 3); zero-padding rows are dropped, colour by height.
        Returns False once the window is closed (caller may then drop the viewer)."""
        if not self.alive:
            return False
        o3d = self._o3d
        pts = np.asarray(points)
        pts = pts[np.any(pts != 0.0, axis=1)]
        if len(pts):
            zn = np.clip((pts[:, 2] - self._zlo) / (self._zhi - self._zlo + 1e-9), 0.0, 1.0)
            self.pcd.points = o3d.utility.Vector3dVector(pts)
            self.pcd.colors = o3d.utility.Vector3dVector(self._cmap(zn)[:, :3])
            if not self._added:
                self.vis.add_geometry(self.pcd)   # add once it has points (sets the view)
                self._added = True
            else:
                self.vis.update_geometry(self.pcd)
        self.alive = self.vis.poll_events()
        self.vis.update_renderer()
        return self.alive

    def close(self) -> None:
        try:
            self.vis.destroy_window()
        except Exception:
            pass
        self.alive = False
