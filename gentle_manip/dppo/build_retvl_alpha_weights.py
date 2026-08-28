"""ReTVL v2: per-timestep alpha (Eq 10) as SAMPLING weights instead of hard chunk
deletion -- fixes the temporal-fragmentation bug found in build_retvl_weighted_dataset.py
(cutting episodes into separate spans broke cond_steps=8 history continuity right at
the decision boundary, likely inflating SR via survivorship rather than genuinely
fixing regrasp).

This script does NOT modify any episode content. It (1) writes the 150-direct +
50-retry source episodes out UNMODIFIED as one flat 200-episode pkl (same file
convert_demos.py will consume normally), and (2) writes a PARALLEL "alpha" pkl with
the identical 200 episodes in the identical order, where each episode's only
observation is its own per-timestep alpha value. Both get run through
convert_demos.py's convert() with the SAME seed, so np.random.default_rng(seed)
.permutation(n) produces an IDENTICAL train/val split and episode ordering for both
-- guaranteeing the resulting alpha_train.npz is perfectly index-aligned with the
real train.npz's `states`/`actions` arrays (both share the same flat per-step
indexing scheme, verified against StitchedSequenceDataset.make_indices: index i's
`start` is a GLOBAL flat-array offset into `states`, so alpha_flat[start] is the
weight for that exact policy training sample).

Usage (envs/dppo):
  uv run --project envs/dppo python -m gentle_manip.dppo.build_retvl_alpha_weights \
    --direct-pkl dataset/demos/single_lift_banana_soft/26-08-15-zet/data.pkl \
    --retry-pkl dataset/demos_retry_v2/single_lift_banana_soft/26-08-25-qkc/data.pkl \
    --value-model logs/retvl/banana/value_model.pt \
    --out-dir logs/retvl/banana/v2
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from gentle_manip.dppo.retvl_value import RetryValueNet
from gentle_manip.dppo.train_retvl_value import episode_state, load_episodes
from gentle_manip.dppo.build_retvl_weighted_dataset import compute_alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct-pkl", required=True)
    ap.add_argument("--retry-pkl", required=True)
    ap.add_argument("--value-model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.value_model, map_location=device, weights_only=False)
    model = RetryValueNet(
        state_dim=ckpt["state_dim"],
        pointnet={"in_channels": 3, "use_layernorm": True, "final_norm": "layernorm"},
        pc_cond_steps=ckpt["pc_cond_steps"], visual_feature_dim=256, mlp_dims=(512, 512),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mu, sigma, delta_a = ckpt["mu"], ckpt["sigma"], ckpt["delta_a"]
    print(f"[retvl-alpha] loaded value model: mu={mu:.5f} sigma={sigma:.5f} "
         f"delta_a={delta_a}", flush=True)

    direct_eps = load_episodes(args.direct_pkl)
    retry_eps = load_episodes(args.retry_pkl)
    all_eps = list(direct_eps) + list(retry_eps)  # FIXED order -- both pkls below must match
    print(f"[retvl-alpha] {len(direct_eps)} direct + {len(retry_eps)} retry = "
         f"{len(all_eps)} source episodes (unmodified, full length)", flush=True)

    raw_episodes = []
    alpha_episodes = []
    for i, e in enumerate(all_eps):
        state = episode_state(e["observations"])
        pc = np.asarray(e["observations"]["point_cloud"], dtype=np.float32)
        T = state.shape[0]
        alphas = compute_alpha(model, state, pc, mu, sigma, delta_a,
                               ckpt["pc_cond_steps"], args.n_points, device, rng)
        raw_episodes.append(e)  # untouched, full episode
        alpha_episodes.append({
            "observations": {"alpha": alphas.reshape(-1, 1).astype(np.float32)},
            "actions": np.asarray(e["actions"]),   # unused, present only to satisfy convert()'s schema
            "rewards": np.asarray(e["rewards"]),
        })
        if i % 25 == 0:
            print(f"[retvl-alpha] ep {i}/{len(all_eps)}: T={T} "
                 f"alpha mean={alphas.mean():.3f}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "raw_200.pkl", "wb") as f:
        pickle.dump({"meta": {"task": "single_lift_banana_soft", "n_episodes": len(raw_episodes)},
                    "episodes": raw_episodes}, f)
    with open(out_dir / "alpha_200.pkl", "wb") as f:
        pickle.dump({"meta": {"task": "single_lift_banana_soft_alpha", "n_episodes": len(alpha_episodes)},
                    "episodes": alpha_episodes}, f)
    print(f"RAW_PKL={out_dir / 'raw_200.pkl'}")
    print(f"ALPHA_PKL={out_dir / 'alpha_200.pkl'}")


if __name__ == "__main__":
    main()
