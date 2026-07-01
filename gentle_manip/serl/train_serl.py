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
def make_sac_state_agent(seed, sample_obs, sample_action, hidden_dims=(256, 256),
                         discount=0.97, target_entropy=None):
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

    net_kwargs = dict(hidden_dims=list(hidden_dims), activations=nn.tanh, use_layer_norm=True)
    critic_backbone = ensemblize(partial(MLP, **net_kwargs), 2)(name="critic_ensemble")
    critic_def = partial(Critic, encoder=StateEncoder(), network=critic_backbone)(name="critic")
    policy_def = Policy(
        encoder=StateEncoder(), network=MLP(**net_kwargs), action_dim=sample_action.shape[-1],
        tanh_squash_distribution=True, std_parameterization="exp", std_min=1e-5, std_max=5,
        name="actor",
    )
    temperature_def = GeqLagrangeMultiplier(
        init_value=1e-2, constraint_shape=(), constraint_type="geq", name="temperature")
    return SACAgent.create(
        jax.random.PRNGKey(seed), sample_obs, sample_action,
        actor_def=policy_def, critic_def=critic_def, temperature_def=temperature_def,
        discount=discount, backup_entropy=False, critic_ensemble_size=2,
        target_entropy=target_entropy, image_keys=("state",),   # pass the pack/unpack check
    )


# ── loops (simplified from train_rlpd) ─────────────────────────────────────────────
def actor_loop(agent, data_store, env, cfg, sampling_rng, ip, port_agentlace):
    import jax
    from agentlace.trainer import TrainerClient
    from serl_launcher.utils.launcher import make_trainer_config
    import tqdm

    client = TrainerClient("actor_env", ip, make_trainer_config(port_agentlace, port_agentlace + 1),
                           data_stores={"actor_env": data_store}, wait_for_server=True, timeout_ms=3000)
    client.recv_network_callback(lambda params: agent.replace(state=agent.state.replace(params=params)))

    obs, _ = env.reset()
    running_return = 0.0
    for step in tqdm.tqdm(range(cfg["max_steps"]), desc="actor"):
        if step < cfg["random_steps"]:
            actions = env.action_space.sample()
        else:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = np.asarray(jax.device_get(
                agent.sample_actions(observations=jax.device_put(obs), seed=key, argmax=False)))
        next_obs, reward, terminated, truncated, info = env.step(actions)
        running_return += reward
        data_store.insert(dict(observations=obs, actions=actions, next_observations=next_obs,
                               rewards=reward, masks=1.0 - float(terminated), dones=terminated or truncated))
        obs = next_obs
        if terminated or truncated:
            client.request("send-stats", {"environment": {"return": running_return,
                                                          "succeed": float(info.get("succeed", 0.0))}})
            running_return = 0.0
            client.update()
            obs, _ = env.reset()


def learner_loop(agent, replay_buffer, demo_buffer, cfg, sampling_rng):
    import jax
    import tqdm
    from agentlace.trainer import TrainerServer
    from serl_launcher.utils.launcher import make_trainer_config
    from serl_launcher.utils.train_utils import concat_batches

    server = TrainerServer(make_trainer_config(cfg["port_agentlace"], cfg["port_agentlace"] + 1),
                           request_callback=lambda t, p: {})
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(total=cfg["training_starts"], desc="filling replay buffer")
    while len(replay_buffer) < cfg["training_starts"]:
        pbar.n = len(replay_buffer); pbar.refresh(); time.sleep(1)
    pbar.close()
    server.publish_network(agent.state.params)

    half = cfg["batch_size"] // 2
    replay_it = replay_buffer.get_iterator(sample_args={"batch_size": half, "pack_obs_and_next_obs": True})
    demo_it = demo_buffer.get_iterator(sample_args={"batch_size": half, "pack_obs_and_next_obs": True})
    critic_only = frozenset({"critic"})
    all_nets = frozenset({"critic", "actor", "temperature"})

    for step in tqdm.tqdm(range(cfg["max_steps"]), desc="learner"):
        for _ in range(cfg["cta_ratio"] - 1):        # extra critic updates (high UTD)
            batch = concat_batches(next(replay_it), next(demo_it), axis=0)
            agent, _ = agent.update(batch, networks_to_update=critic_only)
        batch = concat_batches(next(replay_it), next(demo_it), axis=0)
        agent, info = agent.update(batch, networks_to_update=all_nets)
        if step % cfg["steps_per_update"] == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)


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
    args = ap.parse_args()
    assert args.learner ^ args.actor, "pass exactly one of --learner / --actor"

    import jax
    from serl_launcher.data.data_store import ReplayBufferDataStore

    exp = Experiment.load(args.experiment)
    keys = exp.view_obs(args.view).obs_keys()
    rl = {"max_steps": 100_000, "random_steps": 300, "training_starts": 1000,
          "batch_size": 256, "discount": 0.97, "cta_ratio": 2, "steps_per_update": 10,
          "replay_capacity": 200_000, "port_agentlace": args.port_agentlace, **exp.rl}
    if args.max_steps:
        rl["max_steps"] = args.max_steps

    action_dim = len(exp.action_config.scales)   # delta-pose (6) + gripper (1) = 7

    if args.actor:
        env = SerlStateWrapper(SimGymEnv(port=args.port), keys=keys)
        sample_obs = env.observation_space.sample()
        sample_action = env.action_space.sample()
        agent = make_sac_state_agent(args.seed, sample_obs, sample_action, discount=rl["discount"])
        data_store = ReplayBufferDataStore(env.observation_space, env.action_space, rl["replay_capacity"])
        _, sampling_rng = jax.random.split(jax.random.PRNGKey(args.seed))
        print(f"[actor] exp={exp.name} view={args.view} state_dim={sample_obs['state'].shape} "
              f"action_dim={sample_action.shape}", flush=True)
        actor_loop(agent, data_store, env, rl, sampling_rng, args.ip, rl["port_agentlace"])
    else:
        # Learner: no env (spaces built from the demos + config).
        assert args.demo_path, "learner needs --demo-path (RLPD demos)"
        obs_space = Dict({"state": Box(-np.inf, np.inf, (_demo_state_dim(args.demo_path[0], keys),), np.float32)})
        act_space = Box(-1.0, 1.0, (action_dim,), np.float32)
        sample_obs = obs_space.sample(); sample_action = act_space.sample()
        agent = make_sac_state_agent(args.seed, sample_obs, sample_action, discount=rl["discount"])
        replay_buffer = ReplayBufferDataStore(obs_space, act_space, rl["replay_capacity"])
        demo_buffer = ReplayBufferDataStore(obs_space, act_space, rl["replay_capacity"])
        n = _load_demos_into(demo_buffer, args.demo_path, keys)
        print(f"[learner] exp={exp.name} view={args.view} demos={n} state_dim={sample_obs['state'].shape}",
              flush=True)
        _, sampling_rng = jax.random.split(jax.random.PRNGKey(args.seed))
        learner_loop(agent, replay_buffer, demo_buffer, rl, sampling_rng)


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
