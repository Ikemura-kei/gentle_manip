"""Arm F runner: THEIR WorkSpace/trainer, THEIR policy+head+optimizer+schedule, our cloud branch.

Their tree is untouched except the two-line `lambda` keyword fix in gap/gap.py (their file does
not parse as published). Everything else here is injection:
  * config       — assembled from THEIR yamls, so every proven hyperparameter comes from their files
  * dataset      — dataset.GMCloudPhaseDataset  (our npz in their batch format)
  * policy       — policies.bc.CloudBCPolicy    (their BCPolicy, cloud-shaped compute_loss)
  * encoder      — policies.visual_encoder.CloudFeatureExtractor
Their WorkSpace.train() runs verbatim: their loss, their GAP gradient rule, their AdamW, their
CosineAnnealingLR, their snapshot policy.

TWO ENVIRONMENT STUBS (neither touches training math):
  * torchvision   — their visual_encoder imports resnet18 at module level. Arm F never builds an
                    ImgEncoder/DepthEncoder, so the stub is never called. Stubbed rather than
                    installed because installing torchvision into the shared aarch64 venv risks
                    pulling a different torch build while three other jobs are training in it.
  * SummaryWriter — tensorboard is not in the venv; the shim writes the same scalars to stdout and
                    a CSV. Same reason: no install into a venv that running jobs depend on.
"""
import os
import sys
import types
import csv
import argparse
from pathlib import Path

REPO = Path("/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip")
GAP = REPO / "third_party/GAP"

ap = argparse.ArgumentParser()
ap.add_argument("--data-env", default="single_lift_generalist_3obj")
ap.add_argument("--phase-json", default=str(REPO / ".agent_tmp/gap_phase_single_lift_generalist_3obj.json"))
ap.add_argument("--out", required=True, help="run directory (their WorkSpace writes snapshots here)")
ap.add_argument("--lambda-gap", type=float, default=None, help="override cfg.lambda; 0 = GAP OFF control")
ap.add_argument("--demo-num", type=int, default=None, help="default: every trajectory in train.npz")
ap.add_argument("--epoch", type=int, default=None)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--num-workers", type=int, default=4)
ap.add_argument("--down-dims", default=None,
                help="comma-separated ConditionalUnet1D down_dims, e.g. 16,32,64. THEIR kwarg, so "
                     "this is config-only. Used for the CAPACITY-MATCHED control (arm F3): "
                     "[16,32,64] -> head 1.83M, total 10.65M = 1.12x arm F's 9.49M, vs the "
                     "default [256,512,1024] which is 6.9x.")
ap.add_argument("--head", default="dmm", choices=["dmm", "diffunet"],
                help="THEIR head configs. dmm = DeterministicMLP (their GAP default); "
                     "diffunet = ConditionalUnet1D diffusion head, also shipped by them.")
args = ap.parse_args()

# ---------------------------------------------------------------- environment stubs (see docstring)
# ORDER MATTERS: import diffusers FIRST. It touches torchvision on import, so if our FAKE
# torchvision is already in sys.modules the import fails and (previously, silently) fell back to a
# diffusers stub — which then raised mid-run for --head diffunet. Real diffusers imports fine with
# torchvision genuinely absent (verified job 1765722).
_DIFFUSERS_OK = False
try:
    import diffusers  # noqa: F401
    _DIFFUSERS_OK = True
    print(f"[armF] real diffusers {diffusers.__version__}", flush=True)
except Exception as _e:                       # LOUD: never silently degrade to a stub
    print(f"[armF] diffusers unavailable ({type(_e).__name__}: {_e})", flush=True)

if "torchvision" not in sys.modules:
    try:
        import torchvision  # noqa: F401
    except Exception:
        tv = types.ModuleType("torchvision")
        tvm = types.ModuleType("torchvision.models")
        tvt = types.ModuleType("torchvision.transforms")

        def _resnet18_unavailable(*a, **k):
            raise RuntimeError("resnet18 stub called — arm F must not build the RGB/depth branch")

        tvm.resnet18 = _resnet18_unavailable
        tv.models, tv.transforms = tvm, tvt
        # A bare ModuleType has __spec__ = None, and diffusers probes torchvision via
        # importlib.util.find_spec, which RAISES ValueError("torchvision.__spec__ is None").
        # Give each stub a real ModuleSpec: find_spec then resolves, and the subsequent
        # importlib.metadata version lookup fails naturally (no dist-info), so diffusers
        # correctly concludes torchvision is unavailable instead of crashing.
        import importlib.machinery as _im
        for _m, _n in ((tv, "torchvision"), (tvm, "torchvision.models"), (tvt, "torchvision.transforms")):
            _m.__spec__ = _im.ModuleSpec(_n, loader=None)
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.models"] = tvm
        sys.modules["torchvision.transforms"] = tvt
        print("[armF] torchvision STUBBED (unused: no ResNet branch in this arm)", flush=True)

# their head.py imports diffusers schedulers at module level. Arm F's head is DeterministicMLP
# (verified diffusers-free: DDPMScheduler is used only inside ConditionalUnet1D and
# TransformerForDiffusion, neither of which this arm builds). Stubbed for the same reason as
# torchvision: no installs into the venv three running jobs depend on. Every stub RAISES if
# called, so a silent fake scheduler is impossible.
if args.head == "diffunet" and not _DIFFUSERS_OK:
    raise SystemExit("[armF] --head diffunet needs REAL diffusers, and it failed to import "
                     "(see the reason above). Refusing to run with a stub.")
if not _DIFFUSERS_OK:
    def _unavailable(name):
        def _f(*a, **k):
            raise RuntimeError(f"diffusers.{name} stub called — arm F must use DeterministicMLP")
        return _f

    for _name, _attrs in (
        ("diffusers", []),
        ("diffusers.schedulers", []),
        ("diffusers.schedulers.scheduling_ddpm", ["DDPMScheduler"]),
        ("diffusers.schedulers.scheduling_ddim", ["DDIMScheduler"]),
        ("diffusers.training_utils", ["EMAModel"]),
    ):
        _m = types.ModuleType(_name)
        for _a in _attrs:
            setattr(_m, _a, _unavailable(_a))
        sys.modules[_name] = _m
    sys.modules["diffusers"].schedulers = sys.modules["diffusers.schedulers"]
    sys.modules["diffusers"].training_utils = sys.modules["diffusers.training_utils"]
    print("[armF] diffusers STUBBED (unused: DeterministicMLP head)", flush=True)

import torch

_SCALARS = []
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
except Exception:
    class SummaryWriter:                                    # noqa: N801 - shim name must match
        def __init__(self, log_dir=None, **kw):
            self.log_dir = log_dir

        def add_scalar(self, tag, value, step):
            _SCALARS.append((int(step), tag, float(value)))
            print(f"[armF] epoch {int(step)}: {tag} = {float(value):.6f}", flush=True)

        def close(self):
            pass

    tb = types.ModuleType("torch.utils.tensorboard")
    tb.SummaryWriter = SummaryWriter
    sys.modules["torch.utils.tensorboard"] = tb
    print("[armF] SummaryWriter SHIMMED (scalars -> stdout + scalars.csv)", flush=True)

# ---------------------------------------------------------------- their modules
sys.path.insert(0, str(GAP))            # costdirection (ruptures model="direction")
sys.path.insert(0, str(GAP / "gap"))    # their modules import each other by bare name
sys.path.insert(0, str(REPO / "gentle_manip/dppo/gap_arch"))

from omegaconf import OmegaConf
import dataset as gap_dataset
import policies.bc
import policies.visual_encoder
import gap as gap_trainer                                   # their WorkSpace

from gm_cloud_adapter import CloudBCPolicy, CloudFeatureExtractor, GMCloudPhaseDataset

# inject ours under names their eval() calls can resolve
gap_dataset.GMCloudPhaseDataset = GMCloudPhaseDataset
policies.bc.CloudBCPolicy = CloudBCPolicy
policies.visual_encoder.CloudFeatureExtractor = CloudFeatureExtractor

# ---------------------------------------------------------------- config, assembled from THEIR yamls
C = GAP / "gap/cfgs"


def _load(rel):
    c = OmegaConf.load(C / rel)
    c.pop("defaults", None)
    return c


base = _load("gap.yaml")                       # lambda 0.3, horizon 9, n_obs_steps 5, bs 128, lr 3e-4,
base.pop("hydra", None)                        #   hidden_dim 512, epoch 101 — all THEIRS
policy = _load("policy/bc_policy.yaml")
policy.encoder = _load("policy/encoder/feature.yaml")
policy.head = _load(f"policy/head/{args.head}.yaml")   # dmm = their GAP default; diffunet =
                                                       # their diffusion head, same interface

import numpy as np
data_dir = REPO / "dataset/dppo" / args.data_env
n_traj = int(len(np.load(data_dir / "train.npz")["traj_lengths"]))

cfg = OmegaConf.merge(base, OmegaConf.create({
    "policy": policy,
    "optimizer": _load("optimizer/adam.yaml"),
    "task": {
        "name": args.data_env,
        "data_path": str(data_dir / "train.npz"),
        "pro_dim": 8,                          # our proprio: ee_pos(3) + ee_quat(4) + gripper(1)
        "action_shape": 7,                     # abs_pose_euler_abs_gripper
        "demo_num": args.demo_num or n_traj,
    },
    "dataset": {
        "_target_": "dataset.GMCloudPhaseDataset",
        "kwargs": {
            "data_path": str(data_dir / "train.npz"),
            # LITERALS, not "${batch_size}"-style interpolations: their WorkSpace does
            # copy.deepcopy(cfg.dataset) before instantiating, which can orphan a child node from
            # the root it interpolates against. These are read straight from their gap.yaml above.
            "batch_size": int(base.batch_size),
            "demo_range": [],
            "history": int(base.n_obs_steps),
            "horizon": int(base.horizon),
            "phase_json": args.phase_json,
        },
    },
    "raw": False, "depth": False, "save_video": False,
}))
cfg.policy._target_ = "policies.bc.CloudBCPolicy"
cfg.policy.encoder._target_ = "policies.visual_encoder.CloudFeatureExtractor"
cfg.image, cfg.proprio = True, True
if args.down_dims is not None:
    assert args.head == "diffunet", "--down-dims only applies to the diffunet head"
    cfg.policy.head.network_kwargs.down_dims = [int(v) for v in args.down_dims.split(",")]
    print(f"[armF]   down_dims = {list(cfg.policy.head.network_kwargs.down_dims)} "
          f"(capacity control)", flush=True)
if args.lambda_gap is not None:
    cfg["lambda"] = args.lambda_gap
if args.epoch is not None:
    cfg.epoch = args.epoch
if args.seed is not None:
    cfg.seed = args.seed

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
os.chdir(out)                                  # their WorkSpace writes to Path.cwd()
OmegaConf.save(cfg, out / "resolved_config.yaml", resolve=True)

print("[armF] ==== proven params, sourced from their yamls ====", flush=True)
for k in ("lambda", "horizon", "n_obs_steps", "batch_size", "lr", "hidden_dim", "epoch", "seed"):
    print(f"[armF]   {k} = {cfg[k]}", flush=True)
print(f"[armF]   head = {cfg.policy.head._target_}", flush=True)
print(f"[armF]   optimizer = {cfg.optimizer._target_} {cfg.optimizer.network_kwargs}", flush=True)
print(f"[armF]   demo_num = {cfg.task.demo_num} / {n_traj} trajectories", flush=True)

# ---------------------------------------------------------------- build + report the damped set
np.random.seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
gap_dataset.__dict__.setdefault("GMCloudPhaseDataset", GMCloudPhaseDataset)

_orig_get_dl = gap_dataset.Dataset.get_dataloader
gap_dataset.Dataset.get_dataloader = (
    lambda self, num_workers=args.num_workers, shuffle=True: _orig_get_dl(self, num_workers, shuffle))

w = gap_trainer.WorkSpace(cfg)

damped = [n for n, _ in w.policy.encoder.named_parameters() if "pro" in n]
total = [n for n, _ in w.policy.encoder.named_parameters()]
nd = sum(p.numel() for n, p in w.policy.encoder.named_parameters() if "pro" in n)
nt = sum(p.numel() for _, p in w.policy.encoder.named_parameters())
print(f"[armF] GAP damps {len(damped)}/{len(total)} encoder tensors "
      f"({nd:,}/{nt:,} params = {100*nd/max(nt,1):.1f}%) via their `'pro' in name` filter", flush=True)
branches = {}
for n in damped:
    branches[n.split(".")[0]] = branches.get(n.split(".")[0], 0) + 1
print(f"[armF]   damped by branch: {branches}", flush=True)
print(f"[armF]   NOTE: 'projection' contains 'pro', so their filter also damps the VISUAL "
      f"branch's projection head — reproduced here deliberately.", flush=True)

# instrument the LR schedule from OUTSIDE (their train() steps it PER BATCH with T_max=epoch)
_lrs = []
_orig_sched_step = w.lr_scheduler.step
def _sched_step(*a, **k):
    _lrs.append(w.optimizer.param_groups[0]["lr"])
    return _orig_sched_step(*a, **k)
w.lr_scheduler.step = _sched_step

try:
    w.train()
finally:
    with open(out / "scalars.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["epoch", "tag", "value"]); wr.writerows(_SCALARS)
    if _lrs:
        import numpy as _np
        a = _np.array(_lrs)
        print(f"[armF] lr over {len(a)} scheduler steps: first {a[0]:.3e} min {a.min():.3e} "
              f"max {a.max():.3e} last {a[-1]:.3e}", flush=True)
        _np.save(out / "lr_trace.npy", a)
    print(f"[armF] done -> {out}", flush=True)
