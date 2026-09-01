# External grasp-planner baselines — setup recipes (2026-09-01)

Reproduction recipes for the three external planners used in the E1 comparison
(`docs/paper/synthesis_experiments.md` §4). The clones, venvs, and weights are NOT committed
(`.gitignore`); local patches live in `patches/*.patch`. All adapters:
`grasp_synthesis/baseline_synth.py` (`gpd_planner`, `cgn_planner`, `gn1b_planner`) via
`collect_demos_baseline.py --baseline {gpd,cgn,gn1b}`.

Shared gotchas on the lab box (cost hours; read first):
- **Conda hijacks builds**: always `CC=/usr/bin/gcc CXX=/usr/bin/g++`, keep miniconda out of
  PATH for cmake/nvcc (CUDA 12.1 rejects conda's gcc-15; conda libffi poisons links).
- **nvcc**: `/usr/local/cuda-12.1/bin/nvcc` matches torch cu121; always pass
  `-gencode=arch=compute_89,code=sm_89` (RTX 4090) — without it kernels can hit
  CUDA_ERROR_ILLEGAL_ADDRESS.
- **Subprocess env** (`baseline_synth._run_learned`): PATH must lead with
  `/usr/local/cuda-12.1/bin` (else TF finds the system's ptxas 10.1 → XLA miscompiles →
  EMPTY predictions) and `LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64` (our compiled TF ops
  link cuda-12.1's libcudart; mixing with TF's bundled 12.2 copy → illegal address/abort).

## GPD (ten Pas et al., IJRR 2017) — WORKING

```
git clone https://github.com/atenpas/gpd third_party/gpd && cd third_party/gpd
git apply ../../grasp_synthesis/learned_baselines/patches/gpd.patch   # const comparator,
#   GRASP_POSE stdout lines, plot_*=0 + absolute paths in cfg/eigen_params.cfg
mkdir build && cd build
env -u LD_LIBRARY_PATH -u LIBRARY_PATH PATH=/usr/local/bin:/usr/bin:/bin \
  CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. -DPCL_DIR=/usr/lib/x86_64-linux-gnu/cmake/pcl
make gpd_detect_grasps -j8      # NOTE: target is gpd_detect_grasps; `make detect_grasps` no-ops
# if libffi link errors: sed miniconda3 tokens out of CMakeFiles/*/link.txt and append
#   /lib/x86_64-linux-gnu/libffi.so.7 after the -o target
```
Config: `cfg/gm_gpd.cfg` + `cfg/gm_hand_xarm.cfg` (XArm-approximate hand, approach filter,
headless). Adapter feeds the full privileged dense surface cloud.

## GraspNet-1Billion (Fang et al., CVPR 2020) — WORKING (the primary learned baseline)

```
git clone https://github.com/graspnet/graspnet-baseline third_party/graspnet-baseline
git clone https://github.com/graspnet/graspnetAPI third_party/graspnetAPI
uv venv third_party/gn1b_venv --python 3.10
uv pip install --python third_party/gn1b_venv/bin/python "torch==2.5.1+cu121" \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python third_party/gn1b_venv/bin/python "numpy<2" scipy pillow tqdm gdown \
  open3d scikit-learn transforms3d grasp_nms autolab_core
uv pip install --python third_party/gn1b_venv/bin/python -e third_party/graspnetAPI
cd third_party/graspnet-baseline
export CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=8.9 CC=/usr/bin/gcc CXX=/usr/bin/g++
export PATH=/usr/local/cuda-12.1/bin:/usr/local/bin:/usr/bin:/bin
(cd pointnet2 && ../../gn1b_venv/bin/python setup.py install)
(cd knn && ../../gn1b_venv/bin/python setup.py install)
mkdir weights && gn1b_venv/bin/python -m gdown 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk \
  -O weights/checkpoint-rs.tar     # realsense checkpoint
```
CLI: `gn1b_infer.py` (demo.py defaults: 20000 pts, collision detection, NMS, top-50).

## Contact-GraspNet (Sundermeyer et al., ICRA 2021) — INTEGRATED BUT UNSTABLE on this stack

```
git clone https://github.com/NVlabs/contact_graspnet third_party/contact_graspnet
cd third_party/contact_graspnet
git apply ../../grasp_synthesis/learned_baselines/patches/contact_graspnet.patch
#   (OkStatus rename for TF>=2.10, yaml SafeLoader, c++17 in compile script)
uv venv ../cgn_venv --python 3.11
uv pip install --python ../cgn_venv/bin/python "tensorflow[and-cuda]==2.15.1" "numpy<2" \
  pyyaml opencv-python-headless scipy pillow tqdm trimesh pyrender
# checkpoints (274 MB, Google Drive folder):
../gn1b_venv/bin/python -m gdown --folder 1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl -O checkpoints/
mv checkpoints/contact_graspnet_models/* checkpoints/
# compile tf ops: rm -f pointnet2/tf_ops/*/*.o pointnet2/tf_ops/*/*.so first (repo ships
# stale TF2.2-ABI .so files), then per compile_pointnet_tfops.sh with c++17, the venv's
# TF_CFLAGS/TF_LFLAGS, nvcc -ccbin /usr/bin/gcc -gencode sm_89.
```
⚠ **Measured instability (RTX 4090 + CUDA 12 + TF 2.15):** identical seeded input yields
0 grasps / N grasps / SIGABRT(CUDA_ERROR_ILLEGAL_ADDRESS) across runs — the 2019-era
pointnet2 TF ops are unreliable on this stack (`-G` debug build did not help; TF 2.20 has an
incompatible custom-op ABI). The adapter retries ×3 with shifted seeds; still ~high
synthesis-failure rate. **Proper fix (post-deadline): their pinned TF 2.5 / CUDA 11 conda
env or container.** The E1 probe run quantifies the usable-yield honestly.

## Adapter semantics (all learned planners; see baseline_synth.py comments)

- Input: single-view cloud (front-facing surface points + table disc) from steep virtual
  cameras (77°, 90° elevation) in OpenCV camera frame. The classic ~54° tabletop view was
  measured to yield ZERO executable proposals (side-ish approaches put 45 mm fingers through
  the table on 3–4 cm objects) — itself a finding.
- Pose: planner hand frame → our 7-DOF TCP (GPD/gn1b columns [approach, closing, axis];
  CGN [closing, axis, approach]); planner origin → pad mid-plane centre via each planner's
  depth semantics; table clearance by backing off along −approach with slice re-measurement.
- Width: learned planners emit PRE-SHAPE openings (close-until-force execution model);
  faithful width-command equivalent = local object cross-section at the final slice − 2 mm
  (`_local_xsec`). Same yield-crossing closure can be swapped in via `--baseline-width v41`.
- First candidate (score order) passing the same geometric validity ladder wins.
