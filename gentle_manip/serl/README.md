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

## Training (`train_serl.py`) — generic, experiment-driven
`gentle_manip/serl/train_serl.py` is the one task-agnostic SAC/RLPD trainer (adapted from
`train_rlpd.py`, stripped of reward-classifier / spacemouse / franka / learned-gripper).
Everything comes from `Experiment` + `--view`; new task = new experiment YAML, no new code.

The one new piece — a **state-based SAC agent** (HIL-SERL ships only pixel factories) —
uses the standard idiom for pure state: obs `{"state": flat}`, `image_keys=("state",)`
(passes the agent's pack/unpack check), and a `StateEncoder` returning `obs["state"]` into
encoder-free MLP actor/critic. **Verified in envs/serl (no genesis needed):** the agent
constructs, `sample_actions` returns a (7,) action, and `agent.update` runs gradient steps.

Run (genesis teacher server in envs/sim, learner+actor in envs/serl):
```bash
uv run --project envs/sim  python -m gentle_manip.scripts.serl_sim_server --experiment mushroom_lift --view teacher --port 5566
uv run --project envs/serl python -m gentle_manip.serl.train_serl --experiment mushroom_lift --view teacher --learner --demo-path demos_serl/mushroom.pkl
uv run --project envs/serl python -m gentle_manip.serl.train_serl --experiment mushroom_lift --view teacher --actor  --port 5566
```

## Remaining
1. **Mushroom demos WITH reward** for RLPD: collect through the teacher env (env reward is
   available each step) — drive the scripted lift via `SimGymEnv` and dump episodes with a
   `rewards` array, then `convert_demos.py` → the SERL pickle. (Existing demos are red-cube
   + point-cloud + no reward, not reusable.)
2. **End-to-end run:** actor/learner loops are adapted from train_rlpd but only unit-level
   verified (agent build + train step); the full server↔actor↔learner run needs the demos
   above — first shakedown will likely need small tweaks (agentlace ports, iterator shapes).
