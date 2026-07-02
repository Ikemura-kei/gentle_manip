"""Generic HIL-SERL SAC/RLPD trainer — task-agnostic, driven by one experiment config.

Trains a sample-efficient state-based privileged TEACHER (later distilled to a
point-cloud student). Adapted from hil-serl's examples/train_rlpd.py, stripped of the
real-robot bits (reward classifier, spacemouse intervention, franka wrappers, learned
gripper): sim gives dense reward, and RLPD demos replace human interventions.

Everything comes from configs/experiments/<name>.yaml via Experiment + --view, so a new
task = a new experiment YAML, no new code. The genesis sim runs in envs/sim (3.12) behind
scripts/serl_sim_server.py; this (envs/serl, 3.10, jax) reaches it via SimGymEnv over rpc.

HIL-SERL is pixel-first; for pure state we use the standard idiom: obs = {"state": flat},
image_keys=("state",) (so the agent's pack/unpack check passes), and a StateEncoder that
returns obs["state"] into MLP actor/critic (encoder-free networks).

    # genesis teacher server (envs/sim, 3.12):
    uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server --experiment mushroom_lift --view teacher --port 5566
    # learner + actor (envs/serl, 3.10):
    uv run --project envs/serl python -m gentle_manip.serl.train_serl --experiment mushroom_lift --view teacher --learner --demo-path demos_serl/mushroom.pkl
    uv run --project envs/serl python -m gentle_manip.serl.train_serl --experiment mushroom_lift --view teacher --actor --port 5566
"""
from __future__ import annotations

import os
# The genesis sim server, this learner, and the actor all share one GPU; jax otherwise
# pre-grabs ~75% each -> OOM. Allocate on demand (the state MLPs are tiny). Set before jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import argparse
import copy
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict

# gentle_manip is genesis-free here (Experiment + rpc); add repo root to sys.path.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from gentle_manip.experiment import Experiment
from gentle_manip.serl.gym_env import SimGymEnv


# ── obs: flatten the state dict to {"state": vec} (SERL's expected structure) ──────
class SerlStateWrapper(gym.ObservationWrapper):
    """Concatenate the state obs keys into a single flat vector under "state" (fixed key
    order), matching how demos are flattened. No "images" (pure-state teacher)."""

    def __init__(self, env, keys):
        super().__init__(env)
        self.keys = list(keys)
        dim = int(sum(int(np.prod(env.observation_space[k].shape)) for k in self.keys))
        self.observation_space = Dict({"state": Box(-np.inf, np.inf, (dim,), np.float32)})

    def observation(self, obs):
        return {"state": flatten_state(obs, self.keys)}


def flatten_state(obs: dict, keys) -> np.ndarray:
    return np.concatenate([np.asarray(obs[k], np.float32).reshape(-1) for k in keys]).astype(np.float32)


# ── state-based SAC agent (mirror make_sac_pixel_agent, encoder-free) ──────────────
def make_sac_state_agent(seed, sample_obs, sample_action,
                         critic_hidden_dims=(64, 128, 128, 128, 256),
                         actor_hidden_dims=(64, 128, 128, 128, 256),
                         critic_ensemble_size=10, critic_subsample_size=2,
                         discount=0.97, target_entropy=None, critic_lr=3e-4,
                         actor_lr=3e-4, clip_grad_norm=1.0):
    # Defaults are the proven config from the prior codesign-dfom serl_control project: deeper
    # nets, REDQ critic ensemble (10, subsample 2 -> strong overestimation control),
    # backup_entropy True. The ensemble + net depth are what kept training stable there.
    import jax
    import flax.linen as nn
    from functools import partial
    from serl_launcher.agents.continuous.sac import SACAgent
    from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
    from serl_launcher.networks.mlp import MLP
    from serl_launcher.networks.lagrange import GeqLagrangeMultiplier

    class StateEncoder(nn.Module):
        @nn.compact
        def __call__(self, observations, train=False, stop_gradient=False):
            return observations["state"]

    critic_kwargs = dict(hidden_dims=list(critic_hidden_dims), activations=nn.tanh, use_layer_norm=True)
    actor_kwargs = dict(hidden_dims=list(actor_hidden_dims), activations=nn.tanh, use_layer_norm=True)
    critic_backbone = ensemblize(partial(MLP, **critic_kwargs), critic_ensemble_size)(name="critic_ensemble")
    critic_def = partial(Critic, encoder=StateEncoder(), network=critic_backbone)(name="critic")
    policy_def = Policy(
        encoder=StateEncoder(), network=MLP(**actor_kwargs), action_dim=sample_action.shape[-1],
        tanh_squash_distribution=True, std_parameterization="exp", std_min=1e-5, std_max=5,
        name="actor",
    )
    temperature_def = GeqLagrangeMultiplier(
        init_value=1e-2, constraint_shape=(), constraint_type="geq", name="temperature")
    return SACAgent.create(
        jax.random.PRNGKey(seed), sample_obs, sample_action,
        actor_def=policy_def, critic_def=critic_def, temperature_def=temperature_def,
        discount=discount, backup_entropy=True,   # True in the proven serl_control config
        critic_ensemble_size=critic_ensemble_size, critic_subsample_size=critic_subsample_size,
        target_entropy=target_entropy, image_keys=("state",),   # pass the pack/unpack check
        # clip_grad_norm bounds the Q-target feedback loop (the state teacher otherwise
        # lets critic_loss explode to ~1e14 under high UTD); see utd_ratio gate below.
        actor_optimizer_kwargs={"learning_rate": actor_lr, "clip_grad_norm": clip_grad_norm},
        critic_optimizer_kwargs={"learning_rate": critic_lr, "clip_grad_norm": clip_grad_norm},
        temperature_optimizer_kwargs={"learning_rate": actor_lr},
    )


# ── loops (simplified from train_rlpd) ─────────────────────────────────────────────
def _flatten_batch(obs: dict, keys) -> np.ndarray:
    """Batched obs dict {k: (N, ...)} -> (N, state_dim) in fixed key order."""
    return np.concatenate(
        [np.asarray(obs[k], np.float32).reshape(np.asarray(obs[k]).shape[0], -1) for k in keys], axis=1)


def actor_loop(agent, data_store, client, keys, cfg, sampling_rng, ip, port_agentlace, num_envs, action_dim):
    """VECTORIZED actor: steps num_envs parallel genesis envs (over the rpc SimEnvClient),
    inserts one transition per env each step, resets ALL envs synchronously at the horizon
    (matching the proven codesign-dfom loop). More + more diverse data than a single env."""
    import jax
    from agentlace.trainer import TrainerClient
    from serl_launcher.utils.launcher import make_trainer_config
    import tqdm

    tclient = TrainerClient("actor_env", ip, make_trainer_config(port_agentlace, port_agentlace + 1),
                            data_stores={"actor_env": data_store}, wait_for_server=True, timeout_ms=3000)
    tclient.recv_network_callback(lambda params: agent.replace(state=agent.state.replace(params=params)))

    N, horizon = num_envs, cfg["max_episode_steps"]
    obs = client.reset()                                 # dict of (N, ...)
    running_return = np.zeros(N, dtype=np.float64)
    ep_success = np.zeros(N, dtype=bool)
    t = env_steps = 0
    for step in tqdm.tqdm(range(cfg["max_steps"]), desc="actor"):
        state = _flatten_batch(obs, keys)                # (N, state_dim)
        if env_steps < cfg["random_steps"]:              # uniform exploration to seed the buffer
            actions = np.random.uniform(-1.0, 1.0, size=(N, action_dim)).astype(np.float32)
        else:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = np.asarray(jax.device_get(agent.sample_actions(
                observations=jax.device_put({"state": state}), seed=key, argmax=False))).reshape(N, action_dim)
        next_obs, reward, _done, info = client.step(actions.astype(np.float32))
        reward = np.asarray(reward, np.float32).reshape(N)
        success = np.array([bool(i.get("success", False)) for i in info], dtype=bool)
        running_return += reward
        ep_success |= success
        t += 1
        env_steps += N
        truncated = t >= horizon
        next_state = _flatten_batch(next_obs, keys)
        for j in range(N):                               # one transition per env
            data_store.insert(dict(
                observations={"state": state[j]}, actions=actions[j],
                next_observations={"state": next_state[j]}, rewards=float(reward[j]),
                masks=1.0 - float(success[j]),           # bootstrap unless a true (success) terminal
                dones=bool(success[j] or truncated)))
        obs = next_obs
        if truncated:                                    # synchronous reset of ALL envs
            tclient.request("send-stats", {"environment": {
                "return": float(running_return.mean()), "succeed": float(ep_success.mean())}})
            tclient.update()
            running_return[:] = 0.0
            ep_success[:] = False
            t = 0
            obs = client.reset()


def learner_loop(agent, replay_buffer, demo_buffer, cfg, sampling_rng, wandb_logger=None, ckpt_dir=None):
    import jax
    import tqdm
    from agentlace.trainer import TrainerServer
    from serl_launcher.utils.launcher import make_trainer_config
    from serl_launcher.utils.train_utils import concat_batches
    from flax.training import checkpoints as flax_ckpt
    ckpt_period = cfg.get("ckpt_period", 10_000)

    # Capture the actor's per-episode return/success (sent via client.request("send-stats"))
    # so we can watch ACTUAL task performance, not just losses. Kept as "latest" and emitted
    # with the periodic log below (stepwise-constant between episodes — fine for a curve).
    latest_env_stats: dict = {}

    def _request_cb(t, p):
        if t == "send-stats" and isinstance(p, dict):
            latest_env_stats.update(p.get("environment", {}) or {})
        return {}

    server = TrainerServer(make_trainer_config(cfg["port_agentlace"], cfg["port_agentlace"] + 1),
                           request_callback=_request_cb)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(total=cfg["training_starts"], desc="filling replay buffer")
    while len(replay_buffer) < cfg["training_starts"]:
        pbar.n = len(replay_buffer); pbar.refresh(); time.sleep(1)
    pbar.close()
    server.publish_network(agent.state.params)

    # State (non-image) buffer: no pack_obs_and_next_obs (that's a memory-efficient
    # image-buffer feature; the plain ReplayBuffer.sample() rejects it).
    half = cfg["batch_size"] // 2
    replay_it = replay_buffer.get_iterator(sample_args={"batch_size": half})
    demo_it = demo_buffer.get_iterator(sample_args={"batch_size": half})
    critic_only = frozenset({"critic"})
    all_nets = frozenset({"critic", "actor", "temperature"})

    # Learner runs FREELY at full speed — NO actor throttle. This matches the proven
    # prior-project loop (codesign-dfom serl_runner). The earlier learner-waits-for-actor
    # gate (added pre-REDQ to stop divergence) throttled the learner to ~1 grad step per
    # online transition (≈actor rate), doing ~10x too few updates -> stuck learning. With the
    # REDQ critic ensemble bounding Q, free-running is both stable AND fast (like the reference).
    pbar = tqdm.tqdm(range(cfg["max_steps"]), desc="learner")
    for step in pbar:
        for _ in range(cfg["cta_ratio"] - 1):        # critic_actor_ratio-1 extra critic updates
            batch = concat_batches(next(replay_it), next(demo_it), axis=0)
            agent, _ = agent.update(batch, networks_to_update=critic_only)
        batch = concat_batches(next(replay_it), next(demo_it), axis=0)
        agent, info = agent.update(batch, networks_to_update=all_nets)
        if step % cfg["steps_per_update"] == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)
        if ckpt_dir is not None and step > 0 and step % ckpt_period == 0:
            flax_ckpt.save_checkpoint(str(ckpt_dir), agent.state, step=step, keep=5, overwrite=True)
        if step % cfg["log_period"] == 0:
            flat = {f"{k}/{kk}": float(vv) for k, v in info.items() if isinstance(v, dict)
                    for kk, vv in v.items()}
            flat.update({k: float(v) for k, v in info.items() if not isinstance(v, dict)})
            flat.update({f"environment/{k}": float(v) for k, v in latest_env_stats.items()})
            ret = latest_env_stats.get("return")
            print(f"[learner step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in flat.items()
                                                       if "loss" in k)
                  + (f" | return={ret:.3f} succeed={latest_env_stats.get('succeed', 0):.2f}"
                     if ret is not None else ""), flush=True)
            if wandb_logger is not None:
                wandb_logger.log(flat, step=step)


def main():
    ap = argparse.ArgumentParser(description="Generic SERL SAC/RLPD trainer (experiment-driven)")
    ap.add_argument("--experiment", default="mushroom_lift")
    ap.add_argument("--view", default="teacher")
    ap.add_argument("--learner", action="store_true")
    ap.add_argument("--actor", action="store_true")
    ap.add_argument("--ip", default="localhost", help="learner IP (actor connects here)")
    ap.add_argument("--port", type=int, default=5566, help="genesis rpc server port (actor)")
    ap.add_argument("--port-agentlace", type=int, default=5588, help="learner<->actor port")
    ap.add_argument("--demo-path", action="append", default=[], help="RLPD demo pickle(s) (learner)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true", help="log learner metrics to wandb")
    ap.add_argument("--run-name", default=None, help="run id (also the wandb name + logs/serl/<task>/<run> dir); "
                                                     "pass the SAME to server + learner to share a dir")
    args = ap.parse_args()
    assert args.learner ^ args.actor, "pass exactly one of --learner / --actor"

    import jax
    from serl_launcher.data.data_store import ReplayBufferDataStore

    exp = Experiment.load(args.experiment)
    keys = exp.view_obs(args.view).obs_keys()
    # Defaults = the proven prior-project config: utd_ratio 1, critic_actor_ratio (cta_ratio) 4,
    # steps_per_update 30, replay 600k, REDQ ensemble 10/subsample 2 (in make_sac_state_agent).
    rl = {"max_steps": 2_000_000, "random_steps": 300, "training_starts": 1000,
          "batch_size": 256, "discount": 0.97, "cta_ratio": 4, "steps_per_update": 30,
          "log_period": 50, "utd_ratio": 1, "critic_lr": 3e-4, "clip_grad_norm": 1.0,
          "max_episode_steps": 150, "replay_capacity": 600_000,
          "critic_ensemble_size": 10, "critic_subsample_size": 2,
          "port_agentlace": args.port_agentlace, **exp.rl}
    if args.max_steps:
        rl["max_steps"] = args.max_steps

    action_dim = len(exp.action_config.scales)   # delta-pose (6) + gripper (1) = 7

    if args.actor:
        from agentlace.data.data_store import QueuedDataStore
        from gentle_manip.envs.rpc import SimEnvClient
        # Connect to the (possibly vectorized) genesis sim server and infer num_envs + state_dim
        # from the reset obs, so the actor matches whatever --num-envs the server was launched with.
        client = SimEnvClient(port=args.port)
        obs0 = client.reset()
        num_envs = int(np.asarray(obs0[keys[0]]).shape[0])
        state_dim = int(_flatten_batch(obs0, keys).shape[1])
        sample_obs = {"state": np.zeros(state_dim, np.float32)}
        sample_action = np.zeros(action_dim, np.float32)
        agent = make_sac_state_agent(args.seed, sample_obs, sample_action, discount=rl["discount"],
                                     critic_lr=rl["critic_lr"], clip_grad_norm=rl["clip_grad_norm"],
                                     critic_ensemble_size=rl["critic_ensemble_size"],
                                     critic_subsample_size=rl["critic_subsample_size"])
        # Actor streams transitions to the learner -> a QueuedDataStore (implements
        # get_latest_data); the learner side is the ReplayBufferDataStore.
        data_store = QueuedDataStore(rl["replay_capacity"])
        _, sampling_rng = jax.random.split(jax.random.PRNGKey(args.seed))
        print(f"[actor] exp={exp.name} view={args.view} num_envs={num_envs} state_dim={state_dim} "
              f"action_dim={action_dim}", flush=True)
        actor_loop(agent, data_store, client, keys, rl, sampling_rng, args.ip, rl["port_agentlace"],
                   num_envs, action_dim)
    else:
        # Learner: no env (spaces built from the demos + config).
        assert args.demo_path, "learner needs --demo-path (RLPD demos)"
        obs_space = Dict({"state": Box(-np.inf, np.inf, (_demo_state_dim(args.demo_path[0], keys),), np.float32)})
        act_space = Box(-1.0, 1.0, (action_dim,), np.float32)
        sample_obs = obs_space.sample(); sample_action = act_space.sample()
        agent = make_sac_state_agent(args.seed, sample_obs, sample_action, discount=rl["discount"],
                                     critic_lr=rl["critic_lr"], clip_grad_norm=rl["clip_grad_norm"],
                                     critic_ensemble_size=rl["critic_ensemble_size"],
                                     critic_subsample_size=rl["critic_subsample_size"])
        replay_buffer = ReplayBufferDataStore(obs_space, act_space, rl["replay_capacity"])
        demo_buffer = ReplayBufferDataStore(obs_space, act_space, rl["replay_capacity"])
        n = _load_demos_into(demo_buffer, args.demo_path, keys)
        print(f"[learner] exp={exp.name} view={args.view} demos={n} state_dim={sample_obs['state'].shape}",
              flush=True)

        # Per-run output dir: logs/serl/<task>/<run_name>/{config,videos,checkpoints}.
        from gentle_manip.utils.run_paths import make_run_name, run_dir, snapshot_experiment, write_run_meta
        run_name = args.run_name or make_run_name(exp.name)
        rdir = run_dir("serl", exp.name, run_name)
        snapshot_experiment(exp, rdir)
        write_run_meta(rdir, algo="serl", view=args.view, demos=n, demo_paths=args.demo_path, rl=rl)
        print(f"[learner] run dir: {rdir}", flush=True)

        wandb_logger = None
        if args.wandb:
            from serl_launcher.common.wandb import WandBLogger
            wcfg = WandBLogger.get_default_config()
            wcfg.project = "gentle-manip-serl"
            wcfg.exp_descriptor = exp.name
            # experiment_id = f"{exp_descriptor}_{unique_identifier}" -> make it == run_name.
            wcfg.unique_identifier = (run_name[len(exp.name) + 1:]
                                      if run_name.startswith(exp.name + "_") else run_name)
            wcfg.tag = exp.name                                  # constructor reads config.tag
            wandb_logger = WandBLogger(wandb_config=wcfg, variant={"rl": rl, "view": args.view},
                                       wandb_output_dir=str(rdir), debug=False)
        _, sampling_rng = jax.random.split(jax.random.PRNGKey(args.seed))
        learner_loop(agent, replay_buffer, demo_buffer, rl, sampling_rng, wandb_logger=wandb_logger,
                     ckpt_dir=rdir / "checkpoints")


def _flatten_transition(tr: dict, keys) -> dict:
    """Convert a raw-dict-obs transition (from convert_demos) to SERL's {"state": vec}."""
    tr = dict(tr)
    tr["observations"] = {"state": flatten_state(tr["observations"], keys)}
    tr["next_observations"] = {"state": flatten_state(tr["next_observations"], keys)}
    return tr


def _load_demos_into(buffer, paths, keys) -> int:
    n = 0
    for p in paths:
        for tr in pickle.load(open(p, "rb")):
            buffer.insert(_flatten_transition(tr, keys))
            n += 1
    return n


def _demo_state_dim(path, keys) -> int:
    tr = pickle.load(open(path, "rb"))[0]
    return int(flatten_state(tr["observations"], keys).shape[0])


if __name__ == "__main__":
    main()
