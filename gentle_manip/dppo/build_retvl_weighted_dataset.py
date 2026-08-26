"""Builds the ReTVL-weighted training set (arXiv 2606.24633, Eq 10) from the trained value
network: computes per-chunk alpha_t = clip((r_t-(mu-2*sigma))/(4*sigma+eps), 0, 1) for every
horizon_steps-sized chunk in the 150-direct + 50-retry episode pool, then Bernoulli-keeps
each chunk with probability alpha_t (a standard importance-weighted-resampling equivalent
to per-sample loss reweighting -- chosen over patching DPPO's third_party training loop
directly, to keep this experiment additive/low-risk). Surviving chunks are stitched back
into contiguous runs and written out as a new demo run dir in the SAME episode-shard schema
convert_demos.py expects, so the existing run_fragile25_specialist.py pipeline can train on
it unmodified.

Usage (envs/dppo, needs the trained value net + torch):
  uv run --project envs/dppo python -m gentle_manip.dppo.build_retvl_weighted_dataset \
    --direct-pkl dataset/demos/single_lift_banana_soft/26-08-15-zet/data.pkl \
    --retry-pkl dataset/demos_retry_v2/single_lift_banana_soft/26-08-25-qkc/data.pkl \
    --value-model logs/retvl/banana/value_model.pt \
    --out-dir dataset/demos/single_lift_banana_soft
"""
from __future__ import annotations

import argparse
import datetime
import pickle
import random
import string
from pathlib import Path

import numpy as np
import torch
import yaml

from gentle_manip.dppo.retvl_value import RetryValueNet
from gentle_manip.dppo.train_retvl_value import episode_state, load_episodes

HORIZON_STEPS = 4  # matches the specialist policy's horizon_steps / our Delta_a
MIN_SPAN = 4        # drop stitched-together runs shorter than this (too short to train on)


def _rel_or_abs(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def _make_run_dir(out_dir: Path, task_name: str) -> Path:
    base = out_dir / task_name
    base.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now().strftime("%y-%m-%d")
    for _ in range(10000):
        sfx = "".join(random.choices(string.ascii_lowercase, k=3)) + "-retvl"
        cand = base / f"{date}-{sfx}"
        if not cand.exists():
            cand.mkdir()
            return cand
    raise RuntimeError("could not create run dir")


def compute_alpha(model: RetryValueNet, state: np.ndarray, pc: np.ndarray, mu: float,
                  sigma: float, delta_a: int, pc_cond_steps: int, n_points: int,
                  device: str, rng: np.random.Generator) -> np.ndarray:
    """alpha_t for t in [0, T-delta_a) -- Eq 10, per-timestep (chunk-start) weight."""
    T = state.shape[0]
    alphas = np.zeros(T, dtype=np.float32)
    with torch.no_grad():
        for t in range(0, T - delta_a):
            def cloud(tt):
                idxs = [max(0, tt - k) for k in reversed(range(pc_cond_steps))]
                frames = pc[idxs]
                n = frames.shape[1]
                sel = rng.choice(n, n_points, replace=n < n_points)
                return frames[:, sel]

            s_t = torch.from_numpy(state[t:t + 1]).to(device)
            pc_t = torch.from_numpy(cloud(t)[None]).float().to(device)
            s_t2 = torch.from_numpy(state[t + delta_a:t + delta_a + 1]).to(device)
            pc_t2 = torch.from_numpy(cloud(t + delta_a)[None]).float().to(device)
            r_t = (model.value(s_t2, pc_t2) - model.value(s_t, pc_t)).item()
            alphas[t] = float(np.clip((r_t - (mu - 2 * sigma)) / (4 * sigma + 1e-6), 0.0, 1.0))
    alphas[T - delta_a:] = alphas[max(0, T - delta_a - 1)]  # tail: hold last value
    return alphas


def stitch_kept_spans(episode: dict, kept: np.ndarray) -> list[dict]:
    """kept: (T,) bool, one entry per chunk-start t (kept[t]=True means chunk [t,t+H) survives).
    Merges contiguous kept chunk-starts into spans and slices the episode's
    observations/actions/rewards accordingly. Returns a list of new episode dicts (a single
    original episode may split into multiple spans if the middle got dropped)."""
    T = len(kept)
    spans = []
    t = 0
    while t < T:
        if not kept[t]:
            t += 1
            continue
        start = t
        while t < T and kept[t]:
            t += 1
        end = min(t + HORIZON_STEPS, len(episode["actions"]))  # include the last chunk's tail
        if end - start >= MIN_SPAN:
            spans.append((start, end))
    obs = episode["observations"]
    out = []
    for start, end in spans:
        new_obs = {k: np.asarray(v)[start:end] for k, v in obs.items()}
        out.append({
            "observations": new_obs,
            "actions": np.asarray(episode["actions"])[start:end],
            "rewards": np.asarray(episode["rewards"])[start:end],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct-pkl", required=True)
    ap.add_argument("--retry-pkl", required=True)
    ap.add_argument("--value-model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-name", default="single_lift_banana_soft")
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth-window", type=int, default=25,
                    help="moving-average window over alpha_t before thresholding -- "
                         "independent per-chunk Bernoulli keeps fragment every trajectory "
                         "into ~1/(1-p) length runs (p~0.5 -> ~6-step spans, destroying "
                         "cond_steps=8 history context), so we smooth first for spatially "
                         "coherent (long, contiguous) keep/drop decisions instead.")
    ap.add_argument("--keep-threshold", type=float, default=0.3,
                    help="deterministic keep cutoff on the SMOOTHED alpha signal.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.value_model, map_location=device)
    model = RetryValueNet(
        state_dim=ckpt["state_dim"],
        pointnet={"in_channels": 3, "use_layernorm": True, "final_norm": "layernorm"},
        pc_cond_steps=ckpt["pc_cond_steps"], visual_feature_dim=256, mlp_dims=(512, 512),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mu, sigma, delta_a = ckpt["mu"], ckpt["sigma"], ckpt["delta_a"]
    print(f"[retvl-weighted] loaded value model: mu={mu:.5f} sigma={sigma:.5f} "
         f"delta_a={delta_a}", flush=True)

    direct_eps = load_episodes(args.direct_pkl)
    retry_eps = load_episodes(args.retry_pkl)
    print(f"[retvl-weighted] {len(direct_eps)} direct + {len(retry_eps)} retry source episodes",
         flush=True)

    kept_episodes = []
    n_chunks_total, n_chunks_kept = 0, 0
    for src_name, eps in (("direct", direct_eps), ("retry", retry_eps)):
        for i, e in enumerate(eps):
            state = episode_state(e["observations"])
            pc = np.asarray(e["observations"]["point_cloud"], dtype=np.float32)
            T = state.shape[0]
            if T <= delta_a:
                continue
            alphas = compute_alpha(model, state, pc, mu, sigma, delta_a,
                                   ckpt["pc_cond_steps"], args.n_points, device, rng)
            w = min(args.smooth_window, T)
            kernel = np.ones(w) / w
            smoothed = np.convolve(alphas, kernel, mode="same")
            keep = smoothed > args.keep_threshold
            n_chunks_total += T
            n_chunks_kept += int(keep.sum())
            spans = stitch_kept_spans(e, keep)
            kept_episodes.extend(spans)
            if i % 25 == 0:
                print(f"[retvl-weighted] {src_name} ep {i}: T={T} kept_frac="
                     f"{keep.mean():.2f} -> {len(spans)} span(s)", flush=True)

    print(f"[retvl-weighted] TOTAL: {n_chunks_kept}/{n_chunks_total} chunk-starts kept "
         f"({n_chunks_kept/max(n_chunks_total,1):.1%}), {len(kept_episodes)} output spans "
         f"from {len(direct_eps)+len(retry_eps)} source episodes", flush=True)

    out_run = _make_run_dir(Path(args.out_dir), args.task_name)
    meta = {
        "task": args.task_name,
        "obs_keys": sorted(kept_episodes[0]["observations"].keys()) if kept_episodes else [],
        "action_dim": int(kept_episodes[0]["actions"].shape[-1]) if kept_episodes else 0,
        "rate_hz": 30.0,
        "n_episodes": len(kept_episodes),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(out_run / "data.pkl", "wb") as f:
        pickle.dump({"meta": meta, "episodes": kept_episodes}, f)

    cfg = {
        "task_name": args.task_name,
        "description": (f"ReTVL-weighted (arXiv 2606.24633): {len(direct_eps)} direct-grasp + "
                        f"{len(retry_eps)} regrasp source episodes, Eq10 alpha_t-weighted "
                        f"Bernoulli chunk resampling ({HORIZON_STEPS}-step chunks) using a "
                        f"value net trained with Eq2 global progress CE + Eq8 retry-keypoint "
                        f"preference loss (algorithmic gripper-width keypoint labeling in "
                        f"place of the paper's human annotation). {n_chunks_kept}/"
                        f"{n_chunks_total} chunk-starts kept ({n_chunks_kept/max(n_chunks_total,1):.1%})."),
        "source": "build_retvl_weighted_dataset",
        "direct_run": _rel_or_abs(Path(args.direct_pkl).parent),
        "retry_run": _rel_or_abs(Path(args.retry_pkl).parent),
        "value_model": str(Path(args.value_model)),
        "n_source_episodes": len(direct_eps) + len(retry_eps),
        "n_output_spans": len(kept_episodes),
        "chunk_keep_fraction": n_chunks_kept / max(n_chunks_total, 1),
        "created": meta["created"],
    }
    with open(out_run / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"OUT_RUN={out_run}")


if __name__ == "__main__":
    main()
