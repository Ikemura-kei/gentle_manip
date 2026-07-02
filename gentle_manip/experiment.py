"""Experiment config = the single source of truth for a task across data collection,
training, inference and deploy.

One YAML (configs/experiments/<name>.yaml) composes: task + action + dr + augmentation +
a SUPERSET obs config, plus named VIEWS (subsets of the superset). Every stage loads the
same file:
  - collection uses collection_obs() (the superset — records EVERY modality once);
  - training/online use view_obs("teacher"|"student"|...) — a subview carrying only that
    view's modalities, so the online env has no residual pathways (e.g. teacher = no
    point-cloud -> no render), while all views share the SAME modality params (they
    subview the superset), so they can't drift from the recorded demos.

Genesis-free (only ObsConfig/ActionConfig), so it loads in every env (sim 3.12, serl
3.10, deploy 3.8). Overrides for testing: pass a modified dict / swap the view.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from gentle_manip.perception.obs_config import ObsConfig
from gentle_manip.actions.action_config import ActionConfig

_CFG = Path(__file__).resolve().parent / "configs"


def _load(sub: str, name: str) -> dict:
    return yaml.safe_load((_CFG / sub / f"{name}.yaml").read_text())


class Experiment:
    def __init__(self, path: Path) -> None:
        d = yaml.safe_load(Path(path).read_text())
        self.name = Path(path).stem
        self.path = Path(path)
        self._raw = dict(d)                                     # for run-dir config snapshot
        self.task_cfg = _load("tasks", d["task"])
        self.action_config = ActionConfig.from_dict(_load("action", d["action"]))
        self.dr = _load("dr", d["dr"]) if d.get("dr") else {}
        self.augmentation = d.get("augmentation")               # config name (applied sim-only)
        self.superset_obs = ObsConfig.from_dict(_load("obs", d["obs"]))
        self._views = d.get("views", {})
        self.rl = d.get("rl", {})                               # optional SAC/RLPD hyperparams

    @classmethod
    def load(cls, name_or_path) -> "Experiment":
        p = Path(name_or_path)
        if not p.is_file():
            p = _CFG / "experiments" / f"{name_or_path}.yaml"
        return cls(p)

    # ── obs ───────────────────────────────────────────────────────────────────
    def collection_obs(self) -> ObsConfig:
        """Superset obs — record EVERY modality (demo collection only)."""
        return self.superset_obs

    def views(self) -> list:
        return list(self._views)

    def view_obs(self, view: str) -> ObsConfig:
        """A view's obs config: the superset subset to that view's modalities (params
        inherited, so it matches the recorded demos exactly). Use this for online."""
        if view not in self._views:
            raise KeyError(f"unknown view {view!r}; have {self.views()}")
        return self.superset_obs.subview(self._views[view])

    def augmentation_config(self):
        if not self.augmentation:
            return None
        from gentle_manip.perception.augmentation import AugmentationConfig
        return AugmentationConfig.from_dict(_load("augmentation", self.augmentation))


def subset_demo(data: dict, obs_config: ObsConfig) -> dict:
    """Filter a recorded (superset) demo dict down to the obs keys a view uses —
    'same demonstration, different obs'. Keeps actions / rewards / meta; drops the obs
    keys the view doesn't need. Raises if the demo is missing a required key."""
    keys = obs_config.obs_keys()
    episodes = []
    for ep in data["episodes"]:
        obs = ep["observations"]
        missing = [k for k in keys if k not in obs]
        if missing:
            raise KeyError(f"demo missing view keys {missing}; recorded keys: {sorted(obs)}")
        ep2 = dict(ep)
        ep2["observations"] = {k: obs[k] for k in keys}
        episodes.append(ep2)
    return {**{k: v for k, v in data.items() if k != "episodes"}, "episodes": episodes}
