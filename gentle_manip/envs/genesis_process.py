"""Subprocess isolation for Genesis (the GPU-memory-leak fix).

Genesis leaks GPU memory across scene rebuilds, so the only reliable way to
reclaim it is to kill the owning process. GenesisProcess runs a GenesisWorker in
a child process and talks to it over multiprocessing Queues; ``restart()`` kills
and respawns the child (e.g. when a material change needs a fresh scene), and the
OS reclaims all GPU memory.

The child re-imports genesis from scratch, so a 'spawn' context is required —
'fork' inherits a CUDA context and crashes. Commands carry plain numpy / picklable
payloads; results are the worker's numpy state dicts.

IMPORTANT (spawn): the child re-imports the launching __main__ module, so any
script that constructs a SimBackend/GenesisProcess MUST guard its entry point
with ``if __name__ == "__main__":``. Without it, each spawned child re-runs the
script top-to-bottom and spawns more children — an unbounded fork bomb that hangs.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any, Optional

import numpy as np

from gentle_manip.scenes.scene_spec import SceneSpec


def _worker_loop(cmd_q: "mp.Queue", res_q: "mp.Queue", kwargs: dict) -> None:
    """Child entry point: build the worker, then serve reset/step/stop commands."""
    try:
        from gentle_manip.envs.genesis_worker import GenesisWorker  # imports genesis here
        worker = GenesisWorker(**kwargs)
        res_q.put(("ready", None))
    except Exception:
        res_q.put(("error", traceback.format_exc()))
        return

    while True:
        cmd, payload = cmd_q.get()
        if cmd == "stop":
            worker.close()
            res_q.put(("stopped", None))
            return
        try:
            if cmd == "reset":
                result = worker.reset(**payload)
            elif cmd == "step":
                result = worker.step(*payload)
            elif cmd == "render":                     # env-0 (H,W,3) or all envs (N,H,W,3)
                result = worker.render_rgb(bool(payload.get("all_envs", False)) if payload else False)
            else:
                raise ValueError(f"unknown command {cmd!r}")
            res_q.put(("ok", result))
        except Exception:
            res_q.put(("error", traceback.format_exc()))


class GenesisProcess:
    def __init__(self, spec: SceneSpec, num_envs: int, **worker_kwargs: Any) -> None:
        self.num_envs = int(num_envs)
        self._kwargs = dict(spec=spec, num_envs=num_envs, **worker_kwargs)
        self._ctx = mp.get_context("spawn")
        self._proc: Optional[mp.process.BaseProcess] = None
        self._cmd_q: Optional["mp.Queue"] = None
        self._res_q: Optional["mp.Queue"] = None

    # ── lifecycle ───────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Spawn the child and block until its scene is built (genesis init + build
        can take a couple of minutes)."""
        if self._proc is not None:
            raise RuntimeError("GenesisProcess already started")
        self._cmd_q = self._ctx.Queue()
        self._res_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker_loop, args=(self._cmd_q, self._res_q, self._kwargs), daemon=True
        )
        self._proc.start()
        status, payload = self._res_q.get()
        if status != "ready":
            self.stop()
            raise RuntimeError(f"GenesisWorker init failed in subprocess:\n{payload}")

    def stop(self) -> None:
        """Stop the child; kill it if it does not exit promptly (reclaims GPU mem)."""
        if self._proc is None:
            return
        try:
            if self._proc.is_alive():
                self._cmd_q.put(("stop", None))
                self._proc.join(timeout=15)
        except Exception:
            pass
        finally:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)
            self._proc = None
            self._cmd_q = self._res_q = None

    def restart(self, new_spec: Optional[SceneSpec] = None, **worker_kwarg_updates: Any) -> None:
        """Kill + respawn — the only way to change global material params (E/nu/rho).

        new_spec swaps the scene (e.g. randomized object material on the ObjectEntry);
        worker_kwarg_updates overrides build kwargs like coup_friction (None is ignored).
        """
        if new_spec is not None:
            self._kwargs["spec"] = new_spec
        for k, v in worker_kwarg_updates.items():
            if v is not None:
                self._kwargs[k] = v
        self.stop()
        self.start()

    # ── commands ────────────────────────────────────────────────────────────────
    def reset(self, object_dxy: Optional[np.ndarray] = None,
              home_offset: Optional[np.ndarray] = None,
              object_euler: Optional[np.ndarray] = None,
              perturb: Optional[dict] = None) -> dict:
        return self._call("reset", {"object_dxy": object_dxy, "home_offset": home_offset,
                                    "object_euler": object_euler, "perturb": perturb})

    def step(self, target_pos: np.ndarray, target_quat: np.ndarray, target_gripper: np.ndarray) -> dict:
        return self._call("step", (target_pos, target_quat, target_gripper))

    def render(self, all_envs: bool = False):
        """RGB from the child — env-0 (H,W,3) or all envs (N,H,W,3); None if no camera.
        For behaviour clips / per-trajectory eval video."""
        return self._call("render", {"all_envs": all_envs})

    def _call(self, cmd: str, payload: Any) -> dict:
        if self._proc is None:
            raise RuntimeError("GenesisProcess not started")
        self._cmd_q.put((cmd, payload))
        status, result = self._res_q.get()
        if status == "error":
            raise RuntimeError(f"GenesisWorker {cmd!r} failed in subprocess:\n{result}")
        return result
