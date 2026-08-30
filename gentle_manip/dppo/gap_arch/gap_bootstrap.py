"""Make third_party/GAP importable, with the two unused deps stubbed. See run_gap_arch.py."""
import sys
import types
from pathlib import Path

REPO = Path("/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip")
GAP = REPO / "third_party/GAP"


def bootstrap(verbose=True):
    try:
        import torchvision  # noqa: F401
    except Exception:
        tv, tvm, tvt = (types.ModuleType(n) for n in
                        ("torchvision", "torchvision.models", "torchvision.transforms"))

        def _resnet18(*a, **k):
            raise RuntimeError("resnet18 stub called — this arm builds no ResNet branch")

        tvm.resnet18 = _resnet18
        tv.models, tv.transforms = tvm, tvt
        import importlib.machinery as _im
        for _m, _n in ((tv, "torchvision"), (tvm, "torchvision.models"), (tvt, "torchvision.transforms")):
            _m.__spec__ = _im.ModuleSpec(_n, loader=None)   # diffusers probes find_spec
        sys.modules.update({"torchvision": tv, "torchvision.models": tvm,
                            "torchvision.transforms": tvt})
        if verbose:
            print("[armF] torchvision STUBBED", flush=True)

    try:
        import diffusers  # noqa: F401
    except Exception:
        def _un(name):
            def _f(*a, **k):
                raise RuntimeError(f"diffusers.{name} stub called — this arm uses DeterministicMLP")
            return _f

        for _n, _attrs in (("diffusers", []), ("diffusers.schedulers", []),
                           ("diffusers.schedulers.scheduling_ddpm", ["DDPMScheduler"]),
                           ("diffusers.schedulers.scheduling_ddim", ["DDIMScheduler"]),
                           ("diffusers.training_utils", ["EMAModel"])):
            _m = types.ModuleType(_n)
            for _a in _attrs:
                setattr(_m, _a, _un(_a))
            sys.modules[_n] = _m
        sys.modules["diffusers"].schedulers = sys.modules["diffusers.schedulers"]
        sys.modules["diffusers"].training_utils = sys.modules["diffusers.training_utils"]
        if verbose:
            print("[armF] diffusers STUBBED", flush=True)

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except Exception:
        class SummaryWriter:                       # noqa: N801
            def __init__(self, log_dir=None, **kw):
                self.log_dir = log_dir

            def add_scalar(self, tag, value, step):
                print(f"[armF] {int(step)}: {tag} = {float(value):.6f}", flush=True)

            def close(self):
                pass

        tb = types.ModuleType("torch.utils.tensorboard")
        tb.SummaryWriter = SummaryWriter
        sys.modules["torch.utils.tensorboard"] = tb

    for p in (str(GAP), str(GAP / "gap"), str(REPO / "gentle_manip/dppo/gap_arch")):
        if p not in sys.path:
            sys.path.insert(0, p)
