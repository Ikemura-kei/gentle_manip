# HIL-SERL (SAC/RLPD) integration — mushroom state-based teacher

Train a sample-efficient SAC/RLPD **state-based privileged teacher** for the mushroom
lift in sim, later distilled to a point-cloud student. HIL-SERL is JAX/Python-3.10;
genesis is 3.12 — they can't share an interpreter, so the sim is reached over the
pure-python `gentle_manip.envs.rpc` socket (same idea as the sim↔DP3 eval bridge).

## Architecture
```
 SERL learner (envs/serl, 3.10, jax)            SERL actor (envs/serl, 3.10, jax)
   SAC/RLPD update, 50/50 demo/online     <--agentlace-->   collect transitions
   replay + demo buffers                                    |
                                                            | SimGymEnv (gym_env.py)
                                                            |   single-env gymnasium,
                                                            v   owns episode horizon
                                              gentle_manip.envs.rpc  (socket)
                                                            |
                                          serl_sim_server (envs/sim, 3.12, genesis)
                                            PolicyEnv(mushroom_lift, state_privileged,
                                            num_envs=1, render off, no auto-reset)
```

## Pieces (all present & verified)
| file | env | role |
|---|---|---|
| `envs/serl/pyproject.toml` | — | py3.10 + jax 0.4.35 (GPU) + serl_launcher + agentlace |
| `gentle_manip/scripts/serl_sim_server.py` | envs/sim (3.12) | serves the mushroom teacher PolicyEnv over rpc |
| `gentle_manip/serl/gym_env.py` `SimGymEnv` | envs/serl (3.10) | single-env gymnasium adapter over the rpc client |
| `gentle_manip/serl/convert_demos.py` | envs/serl (3.10) | our demo episodes → SERL RLPD transition pickle |

**Verified:** jax sees the GPU (`CudaDevice(id=0)`); `SACAgent`/`SERLObsWrapper`/replay
buffer/`agentlace`/`SimGymEnv` all import; and the **bridge end-to-end** — the SERL side
resets/steps the genesis teacher, getting the state+privileged obs
(`ee_pos, ee_quat, gripper_width, priv_object_pos, priv_object_vel, priv_stress`) and the
live shaped reward.

## Env gotcha
`jax[cuda12]==0.4.35` pulls cuda-12.9 wheels whose `nvidia.cuda_nvcc` is a PEP-420
namespace package (`__file__` is None), which crashes jax's nvcc probe. Fixed by pinning
`nvidia-cuda-nvcc-cu12==12.6.85` (has a real `__init__.py`) in the pyproject.

## How to run (once the training entry below exists)
```bash
# 1. genesis teacher server (3.12) — one per actor
uv run --project envs/sim python -m gentle_manip.scripts.serl_sim_server --port 5566
# 2. learner (3.10)   3. actor (3.10, connects to server on 5566)
uv run --project envs/serl python <train_entry> --learner --demo_path demos_serl/mushroom.pkl
uv run --project envs/serl python <train_entry> --actor  --ip localhost
```

## Remaining work (the training entry)
1. **`train_serl_mushroom.py`** — adapt `third_party/hil-serl/examples/train_rlpd.py`:
   - drop the real-robot bits (reward classifier, SpacemouseIntervention, franka wrappers);
   - env = `SERLObsWrapper(SimGymEnv(...))` (+ `ChunkingWrapper` if desired) — flattens the
     dict obs to the `state` key;
   - build a **state-based** SAC agent with `SACAgent.create(...)` + MLP actor/critic
     (`serl_launcher.networks`) — no image encoders (`image_keys=None`); HIL-SERL only
     ships pixel factories, so this is the one new bit of wiring;
   - reuse the actor/learner loops + `MemoryEfficientReplayBufferDataStore` + demo loading.
2. **Mushroom demos WITH reward** for RLPD: collect through the teacher env (the env's
   reward is available each step) — e.g. drive the scripted lift policy through `SimGymEnv`
   and dump episodes with a `rewards` array, then `convert_demos.py` → the SERL pickle.
   (Our existing demos are red-cube + point-cloud obs + no reward, so not reusable as-is.)
