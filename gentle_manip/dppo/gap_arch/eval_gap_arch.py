"""Arm F on the CANONICAL harness — same EvalSpec, same venv, same metrics as every other arm.

Hard requirement #1: one shared eval harness. This adds only the per-algorithm adapters the
harness's Protocols ask for (a Policy), and reuses DPPO's EvalAgent for venv construction, so F's
success / stress / width numbers are directly comparable to C'/D'/E and to lulkx.

NORMALIZATION (the B1/B17 bug class), stated precisely — it is guarded, not impossible:
  * IN THIS FILE nothing is rescaled. The venv hands back the same normalized obs dict our DPPO
    arms consume, and F trained on npz `states`/`actions` already normalized to exactly [-1,1].
  * BUT the venv itself uses `normalization_path` to normalize obs and DEnormalize actions, and
    dppo_eval.sbatch's mismatch guard keys off the checkpoint's .hydra/config.yaml, which arm F
    has no equivalent of — so that guard SILENTLY SKIPS for us. GapArchEvalAgent.__init__
    therefore re-asserts it from the dataset name stored in the snapshot's own cfg.

THE ONE THING THAT MUST BE CHECKED, AND IS ASSERTED BELOW: their policy consumes n_obs_steps=5
frames at ONE-ENV-STEP spacing. The venv stacks `cond_steps` frames at exactly that spacing, so
F's eval config MUST set `cond_steps: 5`. Buffering across act() calls instead would space the
history act_steps(=4) env steps apart — a silent train/eval mismatch. The asserts make it loud.
"""
import atexit
import os
from pathlib import Path

import numpy as np
import torch

from gentle_manip.dppo.gap_arch.gap_bootstrap import bootstrap

bootstrap()

import policies.bc                                   # noqa: E402
import policies.visual_encoder                       # noqa: E402
from gm_cloud_adapter import CloudBCPolicy, CloudFeatureExtractor   # noqa: E402

policies.bc.CloudBCPolicy = CloudBCPolicy
policies.visual_encoder.CloudFeatureExtractor = CloudFeatureExtractor

from agent.eval.eval_agent import EvalAgent          # noqa: E402
from gentle_manip.evaluation import EvalSpec, run_eval   # noqa: E402


class GapArchModel(torch.nn.Module):
    """Rebuild their trained BCPolicy from the snapshot's OWN saved cfg, then load its weights.

    Using the cfg stored inside the snapshot (rather than re-deriving one) means the architecture
    can never drift from what was trained — the mistake that made a whole margin sweep meaningless
    when a checkpoint met the wrong normalization.
    """

    def __init__(self, snapshot, device="cuda"):
        super().__init__()
        blob = torch.load(snapshot, map_location=device, weights_only=False)
        cfg = blob["cfg"]
        self.policy = CloudBCPolicy(cfg.policy).to(device)
        self.policy.load_state_dict(blob["policy"], strict=True)   # strict: no silent partial load
        self.policy.eval()
        self.device = device
        self.n_obs = int(cfg.n_obs_steps)
        self.horizon = int(cfg.horizon)
        self.data_env = str(cfg.task.name)      # the dataset this policy was TRAINED on
        print(f"[armF-eval] loaded {snapshot} | n_obs_steps={self.n_obs} horizon={self.horizon} "
              f"| {sum(p.numel() for p in self.policy.parameters()):,} params", flush=True)


class GapArchPolicy:
    """Harness Policy: venv obs dict -> action chunk in the normalized action space.

    Also writes the WIDTH DUMP that `.agent_tmp/decompose_width.py` consumes, so arm F can be
    scored on this phase's TARGET METRIC (commanded width vs object size), not just success/stress.
    The campaign's slope does NOT come from episodes.csv — it is regressed on `width_cmd_mm` /
    `ee_z_m` from these dumps — so without this, F would be uncomparable on the one number the
    phase is about. The conversion below is COPIED LINE-FOR-LINE from
    eval_agent._DiffusionPolicy._flush_dump: normalized -> derive space (action_min/max) -> mm
    (x88). Re-deriving it independently is exactly the B10/B17 reference-frame mistake.
    """

    def __init__(self, model, act_steps):
        self.m = model
        self.act_steps = int(act_steps)
        self._dump_tag = os.environ.get("GM_WIDTH_DUMP")
        self._dump_buf, self._dump_batch, self._dump_mm, self._dump_z = [], 0, None, None
        # GM_OBS_DUMP: closed-loop state+action capture, for diagnosing behaviour the offline
        # teacher-forced checks cannot see (e.g. an approach offset that only appears once the
        # policy's own trajectory leaves the demo manifold). Mirrors eval_agent's obs dump.
        self._obs_tag = os.environ.get("GM_OBS_DUMP")
        self._obs_cloud = bool(os.environ.get("GM_OBS_DUMP_CLOUD"))
        self._obs_buf = {"state": [], "action": [], "point_cloud": []}
        if self._obs_tag:
            atexit.register(self._flush_obs)
            print(f"[armF-eval] OBS DUMP active -> .agent_tmp/{self._obs_tag}_obs_b*.npz "
                  f"(cloud={'yes' if self._obs_cloud else 'no'})", flush=True)
        if self._dump_tag:
            nzp = os.environ.get("GM_WIDTH_NORM")
            assert nzp, "GM_WIDTH_DUMP needs GM_WIDTH_NORM=<dataset>/normalization.npz"
            nzd = np.load(nzp)
            self._dump_mm = (float(nzd["action_min"][-1]), float(nzd["action_max"][-1]))
            self._dump_z = (float(nzd["obs_min"][2]), float(nzd["obs_max"][2]))
            atexit.register(self._flush_dump)          # else the FINAL batch is never written
            print(f"[armF-eval] width DUMP active -> .agent_tmp/{self._dump_tag}_widthcmd_b*.npz",
                  flush=True)

    def _flush_dump(self):
        if not self._dump_tag or not self._dump_buf:
            return
        out = (Path("/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip")
               / ".agent_tmp" / f"{self._dump_tag}_widthcmd_b{self._dump_batch}.npz")
        a_lo, a_hi = self._dump_mm
        buf = np.asarray(self._dump_buf)                                     # (T, n_env, 2)
        u = (buf[..., 0] + 1) / 2 * (a_hi - a_lo + 1e-6) + a_lo              # derive space
        z_lo, z_hi = self._dump_z
        np.savez(out, width_cmd_mm=(u + 1) / 2 * 88.0,                       # (T, n_env), mm
                 ee_z_m=(buf[..., 1] + 1) / 2 * (z_hi - z_lo + 1e-6) + z_lo)  # (T, n_env), m
        self._dump_buf = []
        self._dump_batch += 1

    def _flush_obs(self):
        if not self._obs_tag or not self._obs_buf["state"]:
            return
        out = (Path("/nobackup/proj/disk/softenable-codesign26/personal/ikemura/gentle_manip")
               / ".agent_tmp" / f"{self._obs_tag}_obs_b{self._dump_batch}.npz")
        pay = {"state_norm": np.asarray(self._obs_buf["state"], np.float32),
               "action_norm": np.asarray(self._obs_buf["action"], np.float32),
               # PAIRING KEY. Matching a dump to episodes.csv by (file index, env index) is an
               # ASSUMPTION about flush order, and it was WRONG once: ee@t0 vs home_dx gave
               # r=-0.22 where it must be ~+1.0 (identical sds => right values, wrong order),
               # which invalidated a whole conclusion. Record the batch counter explicitly and
               # ALWAYS validate a pairing against a known-correct column before using it.
               "dump_batch": np.int64(self._dump_batch)}
        nzp = os.environ.get("GM_WIDTH_NORM")
        if nzp:
            nz = np.load(nzp)
            lo, hi = nz["obs_min"], nz["obs_max"]
            # DENORMALIZED alongside the raw arrays so a consumer cannot pick the wrong decode
            pay["state_phys"] = ((pay["state_norm"] + 1) / 2 * (hi - lo + 1e-6) + lo).astype(np.float32)
            pay["obs_min"], pay["obs_max"] = lo, hi
            pay["action_min"], pay["action_max"] = nz["action_min"], nz["action_max"]
        if self._obs_cloud and self._obs_buf["point_cloud"]:
            pay["point_cloud"] = np.asarray(self._obs_buf["point_cloud"], np.float32)
        np.savez_compressed(out, **pay)
        self._obs_buf = {"state": [], "action": [], "point_cloud": []}

    def reset(self):
        self._flush_obs()         # same batch boundary as the width dump
        self._flush_dump()        # batch boundary, as in eval_agent.reset()
                                  # otherwise stateless: the venv supplies the obs history

    def act(self, obs):
        with torch.no_grad():
            cloud = torch.from_numpy(np.asarray(obs["point_cloud"])).float().to(self.m.device)
            state = torch.from_numpy(np.asarray(obs["state"])).float().to(self.m.device)
            assert cloud.ndim == 4 and state.ndim == 3, \
                f"expected (B,T,N,3)/(B,T,D), got {tuple(cloud.shape)}/{tuple(state.shape)}"
            assert cloud.shape[1] == self.m.n_obs and state.shape[1] == self.m.n_obs, (
                f"obs history is {state.shape[1]} steps but the policy was trained on "
                f"{self.m.n_obs} — set cond_steps={self.m.n_obs} in the eval config")
            # THEIR BCPolicy.get_action hard-assumes a single env
            # (`if img.shape[0] != 1: img = img.unsqueeze(0)`), which would reshape this 5-env
            # batch into nonsense. These are the two lines that wrapper wraps, unchanged.
            hidden = self.m.policy.encoder(cloud, state, None, None)      # (B,T,E)
            traj = self.m.policy.head.get_action(hidden)                  # (B,horizon,A)
            assert traj.shape[1] >= self.act_steps, \
                f"policy horizon {traj.shape[1]} < act_steps {self.act_steps}"
            traj = traj.cpu().numpy()[:, :self.act_steps]
            if self._dump_tag:
                z = np.asarray(obs["state"])[:, -1, 2]           # last cond step, ee_z (normalized)
                for k in range(self.act_steps):
                    self._dump_buf.append(np.stack([traj[:, k, -1].copy(), z], axis=-1))  # (n_env,2)
            if self._obs_tag:
                self._obs_buf["state"].append(np.asarray(obs["state"])[:, -1].copy())
                self._obs_buf["action"].append(traj.copy())
                if self._obs_cloud:
                    self._obs_buf["point_cloud"].append(
                        np.asarray(obs["point_cloud"])[:, -1].astype(np.float32).copy())
            return traj


class GapArchEvalAgent(EvalAgent):
    """DPPO's EvalAgent for venv construction; the canonical run_eval for the protocol."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        # NORMALIZATION GUARD. dppo_eval.sbatch derives the expected dataset from the checkpoint's
        # .hydra/config.yaml, which arm F has no equivalent of — so its guard silently SKIPS for us
        # while the venv still uses normalization_path to (de)normalize. Evaluating against the
        # wrong normalization corrupts action denormalization and still reports a plausible success
        # rate; that cost a full margin sweep on 2026-08-26. Enforce it here instead.
        want = self.model.data_env
        got = Path(str(cfg.normalization_path)).parent.name
        assert got == want, (
            f"NORMALIZATION MISMATCH — refusing to run. This policy was trained on '{want}' but "
            f"normalization_path points at '{got}'. Fix: "
            f"NORM=$REPO/dataset/dppo/{want}/normalization.npz")
        print(f"[armF-eval] normalization OK: {want}", flush=True)

    def run(self):
        spec = EvalSpec(
            n_episodes=int(self.cfg.get("n_episodes", 100)),
            num_envs=self.n_envs,
            seed=int(self.cfg.get("seed", 0)),
            max_policy_steps=int(self.cfg.env.max_episode_steps) // self.act_steps,
            scene_group_size=int(self.cfg.get("scene_group_size", 0)),
        )
        run_eval(
            self.venv, GapArchPolicy(self.model, self.act_steps), spec, self.logdir,
            experiment_name=self.cfg.get("experiment"),
            checkpoint=str(self.cfg.model.snapshot),
            record_batches=self.cfg.get("record_batches", None),
        )
