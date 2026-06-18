from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Interactive episode player for a recorded demo pickle. Plays each episode's
# point cloud as a video, with the EE pose drawn as a moving coordinate frame and
# the gripper as a line whose length = gripper width. Keyboard:
#
#   SPACE  play / pause
#   F / D  step one frame forward / back (when paused)
#   N / B  next / previous episode
#   R      restart current episode
#   Q/ESC  quit
#
# Runs in the 3.11 deploy env (needs open3d + a display):
#   uv run --directory deploy python -m gentle_manip.visualization.episode_player <pickle>


def _R_from_quat_wxyz(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _nonzero(pc):
    return pc[np.any(pc != 0.0, axis=1)]


def _height_colors(pts):
    if len(pts) == 0:
        return pts
    z = pts[:, 2]
    lo, hi = z.min(), z.max()
    t = (z - lo) / (hi - lo + 1e-9)
    # simple viridis-ish ramp without importing matplotlib
    return np.stack([t, np.abs(1 - 2 * t), 1 - t], axis=1)


class EpisodePlayer:
    def __init__(self, episodes, meta):
        self.episodes = episodes
        self.rate = float(meta.get("rate_hz", 20.0))
        self.has_pc = "point_cloud" in episodes[0]["observations"]
        self.ep = 0
        self.frame = 0
        self.playing = True
        self._ee_prev_T = np.eye(4)

        import open3d as o3d
        self.o3d = o3d
        self.pcd = o3d.geometry.PointCloud()
        self.ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self.gripper = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.zeros((2, 3))),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        self.gripper.colors = o3d.utility.Vector3dVector([[1.0, 0.2, 0.2]])

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def T(self):
        return self.episodes[self.ep]["actions"].shape[0]

    def _obs(self, key):
        return self.episodes[self.ep]["observations"][key]

    def _refresh(self, vis):
        f = self.frame
        ee = self._obs("ee_pos")[f]
        quat = self._obs("ee_quat")[f]
        width = float(self._obs("gripper_width")[f].reshape(-1)[0])

        if self.has_pc:
            pts = _nonzero(self._obs("point_cloud")[f])
            self.pcd.points = self.o3d.utility.Vector3dVector(pts)
            self.pcd.colors = self.o3d.utility.Vector3dVector(_height_colors(pts))
            vis.update_geometry(self.pcd)

        # EE coordinate frame → absolute pose via the tracked previous transform
        T = np.eye(4)
        R = _R_from_quat_wxyz(quat)
        T[:3, :3] = R
        T[:3, 3] = ee
        self.ee_frame.transform(T @ np.linalg.inv(self._ee_prev_T))
        self._ee_prev_T = T
        vis.update_geometry(self.ee_frame)

        # Gripper: a segment of length=width centred at EE, along the EE local x-axis
        a = 0.5 * width * R[:, 0]
        self.gripper.points = self.o3d.utility.Vector3dVector(np.array([ee + a, ee - a]))
        vis.update_geometry(self.gripper)

        print(f"\rep {self.ep + 1}/{len(self.episodes)}  frame {f + 1}/{self.T}  "
              f"gripper {width * 1000:5.1f} mm  {'▶' if self.playing else '⏸'}   ",
              end="", flush=True)

    def _load_episode(self, vis, idx):
        self.ep = idx % len(self.episodes)
        self.frame = 0
        self._refresh(vis)
        vis.reset_view_point(True)

    # ── key callbacks (return False: no extra redraw needed beyond our updates) ─

    def _toggle(self, vis): self.playing = not self.playing; return False
    def _step_fwd(self, vis): self.playing = False; self.frame = min(self.frame + 1, self.T - 1); self._refresh(vis); return False
    def _step_back(self, vis): self.playing = False; self.frame = max(self.frame - 1, 0); self._refresh(vis); return False
    def _next_ep(self, vis): self._load_episode(vis, self.ep + 1); return False
    def _prev_ep(self, vis): self._load_episode(vis, self.ep - 1); return False
    def _restart(self, vis): self.frame = 0; self._refresh(vis); return False

    def _animate(self, vis):
        if self.playing:
            self.frame = (self.frame + 1) % self.T
            self._refresh(vis)
        return False

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        o3d = self.o3d
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("episode player  —  SPACE play/pause  F/D step  N/B episode  R restart  Q quit")
        if self.has_pc:
            vis.add_geometry(self.pcd)
        vis.add_geometry(self.ee_frame)
        vis.add_geometry(self.gripper)
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))  # world origin

        vis.register_key_callback(ord(" "), self._toggle)
        vis.register_key_callback(ord("F"), self._step_fwd)
        vis.register_key_callback(ord("D"), self._step_back)
        vis.register_key_callback(ord("N"), self._next_ep)
        vis.register_key_callback(ord("B"), self._prev_ep)
        vis.register_key_callback(ord("R"), self._restart)
        vis.register_animation_callback(self._animate)

        self._load_episode(vis, 0)
        vis.run()
        vis.destroy_window()
        print()


def main():
    p = argparse.ArgumentParser(description="Interactive point-cloud episode player")
    p.add_argument("pickle", type=Path)
    args = p.parse_args()
    data = pickle.load(open(args.pickle, "rb"))
    print(f"meta: {data['meta']}")
    print("keys: SPACE play/pause  F/D step  N/B next/prev episode  R restart  Q quit")
    EpisodePlayer(data["episodes"], data["meta"]).run()


if __name__ == "__main__":
    main()
