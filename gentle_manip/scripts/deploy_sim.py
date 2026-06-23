"""Run a real-trained DP3 policy on the Genesis sim (cross-version via RPC).

DP3 needs Python 3.8 (envs/dp3); Genesis needs 3.12 (envs/sim) — they can't share an
interpreter. So the DP3 policy runs here in 3.8 and drives a remote
PolicyEnv(SimBackend) in a 3.12 sim server over a localhost socket
(gentle_manip.envs.rpc). It auto-spawns the sim server and reuses the exact deploy
loop from deploy_real.py — "watch the real policy fly the sim". Doubles as a
sim2real-gap visualizer: the policy sees sim-rendered clouds instead of real L515.

    uv run --project envs/dp3 python gentle_manip/scripts/deploy_sim.py \
        --ckpt third_party/DP3/3D-Diffusion-Policy/data/outputs/<run>/checkpoints/latest.ckpt
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_DP3 = _REPO / "third_party" / "DP3" / "3D-Diffusion-Policy"
for p in (str(_REPO), str(_DP3)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gentle_manip.envs.rpc import SimEnvClient                                   # noqa: E402
from gentle_manip.scripts.deploy_real import DP3PolicyAdapter, run_deploy_loop   # noqa: E402


def _spawn_sim_server(port: int, object_name: str, no_viewer: bool) -> subprocess.Popen:
    cmd = ["uv", "run", "--project", "envs/sim", "python", "-m",
           "gentle_manip.scripts.sim_server", "--port", str(port), "--object", object_name]
    if no_viewer:
        cmd.append("--no-viewer")
    print(f"launching sim server (Genesis build takes ~2 min): {' '.join(cmd)}", file=sys.stderr)
    return subprocess.Popen(cmd, cwd=str(_REPO), start_new_session=True)   # own group → killable


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy a DP3 policy on the Genesis sim via RPC")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--port", type=int, default=5560)
    p.add_argument("--object", default="red_cube")
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--rate", type=float, default=30.0, help="control rate (Hz)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-viewer", action="store_true", help="run the sim server headless")
    p.add_argument("--no-spawn", action="store_true",
                   help="connect to an already-running sim server instead of spawning one")
    args = p.parse_args()

    server = None if args.no_spawn else _spawn_sim_server(args.port, args.object, args.no_viewer)
    try:
        env = SimEnvClient(port=args.port)               # retries until the server binds (~2 min)
        policy = DP3PolicyAdapter(str(args.ckpt), device=args.device)
        run_deploy_loop(env, policy, args.max_steps, args.rate)   # closes env (→ server exits)
    finally:
        if server is not None:
            try:
                server.wait(timeout=15)                  # client close() should end it cleanly
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server.pid), signal.SIGTERM)
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(server.pid), signal.SIGKILL)


if __name__ == "__main__":
    main()
