"""Contact-GraspNet (Sundermeyer et al., ICRA 2021) inference CLI for the E1 learned baseline.

Runs the ORIGINAL NVlabs code + pretrained checkpoint (scene_test_2048_bs3_hor_sigma_001,
their default) on a point cloud — their inference.py minus file-format handling and
visualization; no algorithmic edits. Reads an (N,3) float32 .npy cloud in CAMERA frame
(z forward), predicts on the full cloud (pc_segments = whole cloud so filter_grasps can
assign contacts), prints one GRASP_POSE line per grasp: score, translation, 3x3 rotation
(flattened row-major, CGN panda convention: z = approach, y = closing... columns verified
adapter-side), opening width.

Run with: third_party/cgn_venv/bin/python cgn_infer.py <cloud.npy>
"""
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "third_party", "contact_graspnet")
# inner module dir FIRST so `import contact_graspnet` resolves to the model FILE
# contact_graspnet/contact_graspnet.py, not the same-named package directory at BASE.
sys.path.insert(0, os.path.join(BASE, "contact_graspnet"))
sys.path.append(BASE)
import config_utils                                   # noqa: E402
from contact_grasp_estimator import GraspEstimator    # noqa: E402

CKPT_DIR = os.path.join(BASE, "checkpoints", "scene_test_2048_bs3_hor_sigma_001")


def main(npy_path, seg_path=None, seed=0):
    np.random.seed(int(seed))
    tf.set_random_seed(int(seed))
    from data import regularize_pc_point_count       # their own resampling helper
    pc_full = np.load(npy_path).astype(np.float32)
    pc_full = regularize_pc_point_count(pc_full, 20000)   # fixed-size input; the raw kernels
    pc_seg = np.load(seg_path).astype(np.float32) if seg_path else pc_full  # crash on small clouds
    global_config = config_utils.load_config(CKPT_DIR, batch_size=1, arg_configs=[])
    estimator = GraspEstimator(global_config)
    estimator.build_network()
    saver = tf.train.Saver(save_relative_paths=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    estimator.load_weights(sess, saver, CKPT_DIR, mode="test")

    pred_grasps_cam, scores, contact_pts, gripper_openings = estimator.predict_scene_grasps(
        sess, pc_full, pc_segments={1: pc_seg}, local_regions=seg_path is not None,
        filter_grasps=True, forward_passes=1)
    n_raw = sum(len(v) for v in pred_grasps_cam.values())
    print("RAW_GRASPS %d" % n_raw, file=sys.stderr)

    for seg_id, grasps in pred_grasps_cam.items():
        sc = scores[seg_id]
        w = np.asarray(gripper_openings.get(seg_id, np.full(len(grasps), 0.08))).reshape(-1)
        order = np.argsort(-np.asarray(sc))
        for rank, k in enumerate(order[:50]):
            T = grasps[k]                      # (4,4) cam frame
            R = T[:3, :3].reshape(-1)
            t = T[:3, 3]
            print("GRASP_POSE %d score %.4f pos %.6f %.6f %.6f rot %s width %.6f"
                  % (rank, float(sc[k]), t[0], t[1], t[2],
                     " ".join("%.6f" % v for v in R), float(w[k])))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3] if len(sys.argv) > 3 else 0)
