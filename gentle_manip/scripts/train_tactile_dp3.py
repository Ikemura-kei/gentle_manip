"""Train TactileDiffusionPolicy (gentle_manip/baselines/tactile_dp3) on a
converted zarr dataset (see convert_tactile_demo_to_zarr.py).

Plain argparse + YAML config + manual training loop — no Hydra, per this repo's
convention (see CLAUDE.md). Evaluation is held-out validation loss only (no live
sim/real rollout), logged to wandb alongside the training loss.

Usage:
    uv run --project envs/dp3 python -m gentle_manip.scripts.train_tactile_dp3 \\
        --config gentle_manip/configs/tactile_dp3/cube.yaml
    uv run --project envs/dp3 python -m gentle_manip.scripts.train_tactile_dp3 \\
        --config gentle_manip/configs/tactile_dp3/mushroom.yaml \\
        --init-from dataset/tactile_dp3/checkpoints/cube/best.ckpt
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parents[2]
_DP3 = _REPO / "third_party" / "DP3" / "3D-Diffusion-Policy"
for p in (str(_REPO), str(_DP3)):
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel

from gentle_manip.baselines.tactile_dp3.augmentation import apply_batch_augmentation
from gentle_manip.baselines.tactile_dp3.dataset import TactileDP3Dataset
from gentle_manip.baselines.tactile_dp3.model import TactileDiffusionPolicy


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_policy(cfg: dict, device: torch.device) -> TactileDiffusionPolicy:
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=cfg.get("num_train_timesteps", 100),
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )
    observation_space = {
        "agent_pos": (cfg["state_dim"],),
        "point_cloud": (cfg["point_cloud_points"], 3),
        "tactile_left": (cfg["tactile_image_size"], cfg["tactile_image_size"], 3),
        "tactile_right": (cfg["tactile_image_size"], cfg["tactile_image_size"], 3),
    }
    policy = TactileDiffusionPolicy(
        action_dim=cfg["action_dim"],
        horizon=cfg["horizon"],
        n_action_steps=cfg["n_action_steps"],
        n_obs_steps=cfg["n_obs_steps"],
        noise_scheduler=noise_scheduler,
        observation_space=observation_space,
        num_inference_steps=cfg.get("num_inference_steps", 10),
        diffusion_step_embed_dim=cfg.get("diffusion_step_embed_dim", 128),
        down_dims=tuple(cfg.get("down_dims", [128, 256, 384])),
        encoder_output_dim=cfg.get("encoder_output_dim", 128),
        state_mlp_size=tuple(cfg.get("state_mlp_size", [64, 64])),
        tactile_out_channels=cfg.get("tactile_out_channels", 32),
        dropout=cfg.get("dropout", 0.0),
    )
    return policy


def build_datasets(cfg: dict):
    train_set = TactileDP3Dataset(
        zarr_path=cfg["zarr_path"],
        horizon=cfg["horizon"],
        pad_before=cfg["n_obs_steps"] - 1,
        pad_after=cfg["n_action_steps"] - 1,
        seed=cfg.get("seed", 42),
        val_ratio=cfg.get("val_ratio", 0.1),
    )
    val_set = train_set.get_validation_dataset()
    return train_set, val_set


@torch.no_grad()
def evaluate(model, val_loader, device: torch.device) -> float:
    """Works for either the raw policy or ema.averaged_model — restores whatever
    training-mode state the model had rather than assuming .train() afterward,
    since ema.averaged_model must always stay in eval mode (see EMAModel.__init__)."""
    was_training = model.training
    model.eval()
    losses = []
    for batch in val_loader:
        batch = dict_apply(batch, lambda x: x.to(device))
        loss, _ = model.compute_loss(batch)
        losses.append(loss.item())
    model.train(was_training)
    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(
    path: Path, policy, ema, normalizer, cfg: dict, epoch: int, val_loss_ema: float, val_loss_raw: float
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "ema_model": ema.averaged_model.state_dict(),
            "normalizer": normalizer.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "val_loss": val_loss_ema,  # primary metric — matches what predict_action should use
            "val_loss_raw": val_loss_raw,
        },
        path,
    )


def log_experiment(log_path: Path, record: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """--set key=value (dotted key for one level of nesting, e.g. augmentation.enabled=true).
    Value is parsed with yaml.safe_load so ints/floats/bools/lists all work without quoting."""
    for kv in overrides:
        key, sep, val = kv.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {kv!r}")
        parsed = yaml.safe_load(val)
        if "." in key:
            k1, k2 = key.split(".", 1)
            cfg.setdefault(k1, {})
            cfg[k1][k2] = parsed
        else:
            cfg[key] = parsed
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--init-from", type=Path, default=None, help="warm-start from a checkpoint's model+normalizer")
    parser.add_argument("--max-epochs", type=int, default=None, help="override cfg epochs (smoke tests)")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="truncate each epoch (smoke tests)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--set", action="append", default=[], dest="overrides",
        help="override a config value, e.g. --set dropout=0.25 --set augmentation.enabled=true",
    )
    parser.add_argument(
        "--tag", type=str, default="default",
        help="experiment tag: suffixes checkpoint_dir and wandb run name, keys the experiment log",
    )
    parser.add_argument(
        "--exp-log", type=Path, default=Path("dataset/tactile_dp3/experiments.jsonl"),
        help="JSONL file each run appends its final result to",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    if args.max_epochs is not None:
        cfg["epochs"] = args.max_epochs

    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    train_set, val_set = build_datasets(cfg)
    print(f"[train] train episodes={train_set.train_mask.sum()} steps={len(train_set)}  "
          f"val episodes={val_set.train_mask.sum()} steps={len(val_set)}", flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.get("num_workers", 4) > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=cfg.get("num_workers", 4) > 0,
    )

    policy = build_policy(cfg, device)
    normalizer = train_set.get_normalizer()
    policy.set_normalizer(normalizer)

    if args.init_from is not None:
        print(f"[train] warm-starting from {args.init_from}", flush=True)
        payload = torch.load(args.init_from, map_location=device)
        policy.load_state_dict(payload["model"])

    # LinearNormalizer.load_state_dict rebuilds params_dict from scratch (see
    # DictOfTensorMixin._load_from_state_dict), which bypasses any earlier
    # .to(device) — move the whole policy (incl. normalizer) to device only
    # after set_normalizer/init_from have both run.
    policy = policy.to(device)

    ema = EMAModel(model=copy.deepcopy(policy), power=cfg.get("ema_power", 0.75))

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.get("lr", 1.0e-4),
        betas=tuple(cfg.get("betas", [0.95, 0.999])),
        eps=1.0e-8,
        weight_decay=cfg.get("weight_decay", 1.0e-6),
    )
    steps_per_epoch = args.max_steps_per_epoch or len(train_loader)
    num_training_steps = steps_per_epoch * cfg["epochs"]
    lr_scheduler = get_scheduler(
        cfg.get("lr_scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=cfg.get("lr_warmup_steps", 500),
        num_training_steps=num_training_steps,
    )

    run_name = f"{cfg['name']}-{args.tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.get("wandb_project", "gentle-manip-tactile-dp3"),
            group=cfg["name"],
            name=run_name,
            config={**cfg, "tag": args.tag},
        )

    ckpt_dir = Path(cfg["checkpoint_dir"]) / args.tag
    best_val_loss = float("inf")
    best_val_loss_raw = float("inf")
    best_epoch = -1
    global_step = 0
    run_start = time.time()

    aug_cfg = cfg.get("augmentation") or {}
    aug_enabled = bool(aug_cfg.get("enabled", False))

    for epoch in range(cfg["epochs"]):
        epoch_losses = []
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            if args.max_steps_per_epoch is not None and i >= args.max_steps_per_epoch:
                break
            batch = dict_apply(batch, lambda x: x.to(device))
            if aug_enabled:
                batch["obs"] = apply_batch_augmentation(batch["obs"], aug_cfg)
            loss, loss_dict = policy.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.step(policy)

            epoch_losses.append(loss.item())
            global_step += 1
            if use_wandb:
                wandb.log(
                    {"train/loss": loss.item(), "train/lr": lr_scheduler.get_last_lr()[0], "epoch": epoch},
                    step=global_step,
                )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        dt = time.time() - t0
        msg = f"[train] epoch {epoch}: train_loss={train_loss:.5f} ({dt:.1f}s)"

        val_every = cfg.get("val_every", 1)
        did_val = (epoch + 1) % val_every == 0 or epoch == cfg["epochs"] - 1
        val_loss_ema, val_loss_raw = None, None
        if did_val:
            # EMA is the primary metric — matches DP3's own train.py (evaluates/rolls
            # out with ema_model, not the raw noisy training weights) and is what
            # predict_action should actually be judged on for a small, overfit-prone
            # dataset like this one. Raw is logged alongside for diagnostics only.
            val_loss_ema = evaluate(ema.averaged_model, val_loader, device)
            val_loss_raw = evaluate(policy, val_loader, device)
            msg += f" val_loss_ema={val_loss_ema:.5f} val_loss_raw={val_loss_raw:.5f}"
            if use_wandb:
                wandb.log(
                    {"val/loss_ema": val_loss_ema, "val/loss_raw": val_loss_raw, "epoch": epoch},
                    step=global_step,
                )

        print(msg, flush=True)

        save_checkpoint(ckpt_dir / "last.ckpt", policy, ema, normalizer, cfg, epoch, val_loss_ema, val_loss_raw)
        if val_loss_ema is not None and val_loss_ema < best_val_loss:
            best_val_loss = val_loss_ema
            best_val_loss_raw = val_loss_raw
            best_epoch = epoch
            save_checkpoint(ckpt_dir / "best.ckpt", policy, ema, normalizer, cfg, epoch, val_loss_ema, val_loss_raw)
            print(f"[train] new best val_loss_ema={val_loss_ema:.5f} -> {ckpt_dir / 'best.ckpt'}", flush=True)

    duration = time.time() - run_start
    print(
        f"[train] done. best val_loss_ema={best_val_loss:.5f} (epoch {best_epoch})  "
        f"checkpoints in {ckpt_dir}  ({duration/60:.1f} min)",
        flush=True,
    )
    log_experiment(
        args.exp_log,
        {
            "task": cfg["name"],
            "tag": args.tag,
            "config_path": str(args.config),
            "overrides": args.overrides,
            "epochs": cfg["epochs"],
            "best_val_loss_ema": best_val_loss,
            "best_val_loss_raw": best_val_loss_raw,
            "best_epoch": best_epoch,
            "final_train_loss": train_loss,
            "duration_min": duration / 60,
            "checkpoint_dir": str(ckpt_dir),
            "wandb_url": wandb.run.url if use_wandb else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
