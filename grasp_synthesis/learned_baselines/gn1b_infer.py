"""GraspNet-1Billion (Fang et al., CVPR 2020) inference CLI for the E1 learned baseline.

Runs the ORIGINAL graspnet-baseline network + pretrained realsense checkpoint on a point
cloud, with the demo.py defaults (20000 points, NMS, sort, model-free collision detection
at their demo settings). Zero algorithmic edits — this file is pure I/O glue: reads an
(N,3) float32 .npy cloud in CAMERA frame (z forward), prints one GRASP_POSE line per
surviving grasp (their frame: rotation column 0 = approach, translation = grasp centre,
depth = closing-region depth along approach).

Run with the dedicated venv: third_party/gn1b_venv/bin/python gn1b_infer.py <cloud.npy>
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "third_party", "graspnet-baseline")
sys.path.append(os.path.join(ROOT, "models"))
sys.path.append(os.path.join(ROOT, "utils"))
from graspnet import GraspNet, pred_decode          # noqa: E402
from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from graspnetAPI import GraspGroup                  # noqa: E402

CKPT = os.path.join(ROOT, "weights", "checkpoint-rs.tar")
NUM_POINT = 20000


def main(npy_path, seed=0):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    cloud = np.load(npy_path).astype(np.float32)
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                   is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    ckpt = torch.load(CKPT, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    if len(cloud) >= NUM_POINT:
        idxs = np.random.choice(len(cloud), NUM_POINT, replace=False)
    else:
        idxs = np.concatenate([np.arange(len(cloud)),
                               np.random.choice(len(cloud), NUM_POINT - len(cloud), replace=True)])
    sampled = torch.from_numpy(cloud[idxs][np.newaxis]).to(device)
    with torch.no_grad():
        end_points = net({"point_clouds": sampled})
        preds = pred_decode(end_points)
    gg = GraspGroup(preds[0].detach().cpu().numpy())

    # demo.py defaults: collision detection on the full cloud, NMS, sort
    mfc = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
    gg = gg[~mfc.detect(gg, approach_dist=0.05, collision_thresh=0.01)]
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]

    for i in range(len(gg)):
        g = gg[i]
        R = g.rotation_matrix.reshape(-1)
        t = g.translation
        print("GRASP_POSE %d score %.4f pos %.6f %.6f %.6f rot %s width %.6f depth %.6f"
              % (i, g.score, t[0], t[1], t[2],
                 " ".join("%.6f" % v for v in R), g.width, g.depth))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 0)
