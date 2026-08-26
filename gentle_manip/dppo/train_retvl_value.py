"""Trains the ReTVL value function (arXiv 2606.24633, Eq 2 + Eq 8 + Eq 9) on the banana
150-direct + 50-regrasp demo set, using algorithmic retry-keypoint labels
(retvl_retry_labeling.py) in place of the paper's human annotations. Saves the trained
value net + offline r_t statistics (mu, sigma) needed for Eq 10's BC-weighting formula.

Usage (envs/sim -- needs numpy/torch, matches the sim env's torch install):
  uv run --project envs/sim python -m gentle_manip.dppo.train_retvl_value \
    --direct-pkl dataset/demos/single_lift_banana_soft/26-08-15-zet/data.pkl \
    --retry-pkl dataset/demos_retry_v2/single_lift_banana_soft/26-08-25-qkc/data.pkl \
    --out logs/retvl/banana/value_model.pt
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gentle_manip.dppo.retvl_value import RetryValueNet, progress_to_bin, N_BINS
from gentle_manip.dppo.retvl_retry_labeling import retry_keypoints

# Table 4 windows (paper: 5Hz frame indices [r-12,r-2]/[r-1,r+1]/[r+2,r+12]; scaled to our
# ~30Hz recording rate by the paper's own stated 6x factor: 30/5=6).
DELTA_PRE = 72
DELTA_NEAR = 6
DELTA_POST = 72
T_PREF = 0.1     # preference loss temperature
TAU_W = 6.0      # soft window weight decay
LAMBDA_ABS = 1.0
LAMBDA_PREF = 3.0


def load_episodes(path: str) -> list[dict]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["episodes"]


def episode_state(obs: dict) -> np.ndarray:
    """(T, state_dim) -- ee_pos(3)+ee_quat(4)+gripper_width(1) = 8, matches PROPRIO_VIEW."""
    return np.concatenate([
        np.asarray(obs["ee_pos"]), np.asarray(obs["ee_quat"]),
        np.asarray(obs["gripper_width"]).reshape(-1, 1),
    ], axis=-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct-pkl", required=True)
    ap.add_argument("--retry-pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--pc-cond-steps", type=int, default=1)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    direct_eps = load_episodes(args.direct_pkl)
    retry_eps = load_episodes(args.retry_pkl)
    print(f"[retvl-value] {len(direct_eps)} direct + {len(retry_eps)} retry episodes")

    # Build a flat list of (state_TxD, pointcloud_TxNx3, T, keypoints[list[int]]) per episode.
    episodes = []
    for e in direct_eps:
        st = episode_state(e["observations"])
        pc = np.asarray(e["observations"]["point_cloud"], dtype=np.float32)
        episodes.append({"state": st, "pc": pc, "T": len(st), "keypoints": []})
    n_retry_labeled = 0
    for e in retry_eps:
        st = episode_state(e["observations"])
        pc = np.asarray(e["observations"]["point_cloud"], dtype=np.float32)
        kps = retry_keypoints(e["observations"]["gripper_width"])
        if kps:
            n_retry_labeled += 1
        episodes.append({"state": st, "pc": pc, "T": len(st), "keypoints": kps})
    print(f"[retvl-value] {n_retry_labeled}/{len(retry_eps)} retry episodes labeled "
         f"with >=1 retry keypoint")

    state_dim = episodes[0]["state"].shape[-1]
    model = RetryValueNet(
        state_dim=state_dim,
        pointnet={"in_channels": 3, "use_layernorm": True, "final_norm": "layernorm"},
        pc_cond_steps=args.pc_cond_steps, visual_feature_dim=256, mlp_dims=(512, 512),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def sample_pc(ep, t):
        # (pc_cond_steps, N, 3), padded at the start of the episode by repeating frame 0.
        idxs = [max(0, t - k) for k in reversed(range(args.pc_cond_steps))]
        frames = ep["pc"][idxs]  # (pc_cond_steps, N_raw, 3)
        n = frames.shape[1]
        if n >= args.n_points:
            sel = rng.choice(n, args.n_points, replace=False)
        else:
            sel = rng.choice(n, args.n_points, replace=True)
        return frames[:, sel]

    def global_batch(bs):
        states, pcs, targets = [], [], []
        for _ in range(bs):
            ep = episodes[rng.integers(len(episodes))]
            t = int(rng.integers(ep["T"]))
            states.append(ep["state"][t])
            pcs.append(sample_pc(ep, t))
            targets.append(t / max(ep["T"] - 1, 1))
        return (torch.from_numpy(np.stack(states)).to(device),
               torch.from_numpy(np.stack(pcs)).float().to(device),
               torch.tensor(targets, dtype=torch.float32, device=device))

    retry_episodes = [ep for ep in episodes if ep["keypoints"]]

    def pref_batch(bs):
        s_pos, pc_pos, s_neg, pc_neg, weights = [], [], [], [], []
        for _ in range(bs):
            ep = retry_episodes[rng.integers(len(retry_episodes))]
            r = ep["keypoints"][rng.integers(len(ep["keypoints"]))]
            pre_lo, pre_hi = max(0, r - DELTA_PRE), max(0, r - DELTA_NEAR)
            post_lo = min(ep["T"] - 1, r + DELTA_NEAR)
            post_hi = min(ep["T"] - 1, r + DELTA_POST)
            if pre_hi <= pre_lo or post_hi <= post_lo:
                continue
            t_neg = int(rng.integers(pre_lo, pre_hi))
            t_pos = int(rng.integers(post_lo, post_hi))
            s_pos.append(ep["state"][t_pos]); pc_pos.append(sample_pc(ep, t_pos))
            s_neg.append(ep["state"][t_neg]); pc_neg.append(sample_pc(ep, t_neg))
            weights.append(np.exp(-abs(t_pos - r) / TAU_W))
        if not s_pos:
            return None
        return (torch.from_numpy(np.stack(s_pos)).to(device),
               torch.from_numpy(np.stack(pc_pos)).float().to(device),
               torch.from_numpy(np.stack(s_neg)).to(device),
               torch.from_numpy(np.stack(pc_neg)).float().to(device),
               torch.tensor(weights, dtype=torch.float32, device=device))

    print(f"[retvl-value] training on {device}, {len(retry_episodes)} retry episodes "
         f"with keypoints available for preference pairs")
    for epoch in range(1, args.epochs + 1):
        model.train()
        s, pc, v_star = global_batch(args.batch_size)
        logits = model.logits(s, pc)
        bins = progress_to_bin(v_star)
        loss_abs = F.cross_entropy(logits, bins)

        pref = pref_batch(args.batch_size)
        if pref is not None:
            s_pos, pc_pos, s_neg, pc_neg, w = pref
            v_pos = model.value(s_pos, pc_pos)
            v_neg = model.value(s_neg, pc_neg)
            loss_pref = -(w * F.logsigmoid((v_pos - v_neg) / T_PREF)).mean()
        else:
            loss_pref = torch.tensor(0.0, device=device)

        loss = LAMBDA_ABS * loss_abs + LAMBDA_PREF * loss_pref
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"[retvl-value] epoch {epoch}/{args.epochs} "
                 f"loss={loss.item():.4f} abs={loss_abs.item():.4f} "
                 f"pref={loss_pref.item():.4f}", flush=True)

    # Offline r_t statistics for Eq 10 (mu, sigma of value-improvement over the FULL dataset)
    model.eval()
    delta_a = 4  # our horizon_steps, matches the paper's chunk-size choice for Delta_a
    r_vals = []
    with torch.no_grad():
        for ep in episodes:
            T = ep["T"]
            ts = list(range(0, T - delta_a, max(1, (T - delta_a) // 40 or 1)))
            for t in ts:
                s_t = torch.from_numpy(ep["state"][t:t + 1]).to(device)
                pc_t = torch.from_numpy(sample_pc(ep, t)[None]).float().to(device)
                s_t2 = torch.from_numpy(ep["state"][t + delta_a:t + delta_a + 1]).to(device)
                pc_t2 = torch.from_numpy(sample_pc(ep, t + delta_a)[None]).float().to(device)
                r = (model.value(s_t2, pc_t2) - model.value(s_t, pc_t)).item()
                r_vals.append(r)
    mu, sigma = float(np.mean(r_vals)), float(np.std(r_vals) + 1e-6)
    print(f"[retvl-value] r_t offline stats: mu={mu:.5f} sigma={sigma:.5f} "
         f"(n={len(r_vals)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "state_dim": state_dim, "pc_cond_steps": args.pc_cond_steps,
        "n_points": args.n_points, "mu": mu, "sigma": sigma, "delta_a": delta_a,
    }, out_path)
    print(f"[retvl-value] saved to {out_path}")


if __name__ == "__main__":
    main()
