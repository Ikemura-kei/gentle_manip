"""Tiny socket RPC to bridge the py3.12 sim and the py3.8 DP3 policy.

Genesis needs Python 3.12 and the DP3 stack needs 3.8, so they can't share an
interpreter — this carries obs/action across a localhost socket instead. The wire
format is a length-prefixed frame: a JSON header (command / scalars + per-array
dtype+shape) followed by raw array bytes reconstructed with np.frombuffer. That is
**numpy-version-safe** (sim is on numpy 2.x, dp3 on numpy 1.x), unlike pickle.

This module imports neither genesis nor torch, so it loads in both envs. The sim
process calls ``serve_env(env, ...)``; the policy process drives ``SimEnvClient``,
which mimics the PolicyEnv methods the deploy loop uses (reset/step/close).
"""
from __future__ import annotations

import json
import socket
import struct
import time
from typing import Any, Dict, List, Tuple

import numpy as np

_U32 = struct.Struct(">I")


# ── framed numpy-safe messages ────────────────────────────────────────────────
def _recvall(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        buf += chunk
    return bytes(buf)


def send_msg(conn: socket.socket, header: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    """Send (header dict, {name: ndarray}). header must be JSON-serializable."""
    specs, blobs = [], []
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        specs.append({"name": name, "dtype": a.dtype.str, "shape": list(a.shape)})
        blobs.append(a.tobytes())
    meta = json.dumps({"header": header, "arrays": specs}).encode("utf-8")
    payload = b"".join([_U32.pack(len(meta)), meta, *blobs])
    conn.sendall(_U32.pack(len(payload)) + payload)


def recv_msg(conn: socket.socket) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    payload = _recvall(conn, _U32.unpack(_recvall(conn, 4))[0])
    mlen = _U32.unpack(payload[:4])[0]
    meta = json.loads(payload[4:4 + mlen].decode("utf-8"))
    off = 4 + mlen
    arrays: Dict[str, np.ndarray] = {}
    for spec in meta["arrays"]:
        dt = np.dtype(spec["dtype"])
        shape = tuple(spec["shape"])
        n = dt.itemsize * (int(np.prod(shape)) if shape else 1)
        arrays[spec["name"]] = np.frombuffer(payload[off:off + n], dtype=dt).reshape(shape).copy()
        off += n
    return meta["header"], arrays


# ── server: drive any PolicyEnv-like object over the socket ────────────────────
def serve_env(env, host: str = "127.0.0.1", port: int = 5555, ready_msg: str = "SIM_SERVER_READY",
              frame_fn=None, video_dir=None, video_episodes: int = 0, video_every: int = 0) -> None:
    """Serve reset/step/close requests for ``env`` until the client disconnects.

    ``env`` must expose reset()->obs dict and step(action)->(obs, reward, done, info),
    matching PolicyEnv. Prints ``ready_msg`` once the port is bound (so a launcher
    can wait for it).

    If frame_fn is given, an mp4 of frame_fn() (an (H,W,3) uint8) is written into
    ``video_dir`` for selected episodes (reset = boundary):
      - ``video_episodes`` > 0: the FIRST N episodes (offline eval visualisation).
      - ``video_every``    > 0: EVERY Nth episode, indefinitely — for periodic
        behaviour clips during a long training run (watch the policy improve).
    The two can combine; an episode records if either rule selects it.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(ready_msg, flush=True)

    vframes: list = []
    vep = [0]   # episodes started (reset count)

    def _recording(ep: int) -> bool:
        return (video_episodes > 0 and ep <= video_episodes) or \
               (video_every > 0 and ep % video_every == 0)

    def flush_video():
        if vframes and video_dir is not None:
            import imageio.v2 as imageio
            from pathlib import Path
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(video_dir) / f"ep_{vep[0]:04d}.mp4")
            imageio.mimsave(out, vframes, fps=30, macro_block_size=1)
            print(f"  saved clip {out} ({len(vframes)} frames)", flush=True)
        vframes.clear()

    # Accept clients repeatedly: an RL actor may crash/restart mid-run, and a mere
    # disconnect must NOT kill the (expensive) genesis env. Only an explicit "close"
    # command shuts the server down.
    shutdown = False
    try:
        while not shutdown:
            conn, _ = srv.accept()
            try:
                while True:
                    header, arrays = recv_msg(conn)
                    cmd = header.get("cmd")
                    if cmd == "close":
                        shutdown = True
                        break
                    elif cmd == "reseed":
                        if hasattr(env, "reseed"):
                            env.reseed(int(header["seed"]))
                        send_msg(conn, {"ok": True}, {})
                    elif cmd == "set_auto_scene_dr":
                        # Freeze/thaw the training server's periodic AUTO scene-DR relaunch so it
                        # can't fire mid-eval (the harness drives its own deterministic per-group
                        # rebuild). No-op if the env has no such control (e.g. scene_dr_every=0).
                        if hasattr(env, "set_auto_scene_dr"):
                            env.set_auto_scene_dr(bool(header["enabled"]))
                        send_msg(conn, {"ok": True}, {})
                    elif cmd == "reset":
                        flush_video()              # save the episode just finished
                        vep[0] += 1
                        send_msg(conn, _scenario_header(env), _as_arrays(env.reset()))
                    elif cmd == "randomize_scene":
                        # Rebuild the scene (material+size+shape) — the eval harness's deterministic
                        # per-group scene DR (reseed then randomize_scene => reproducible geometry).
                        flush_video()
                        vep[0] += 1
                        obs = env.randomize_scene() if hasattr(env, "randomize_scene") else env.reset()
                        send_msg(conn, _scenario_header(env), _as_arrays(obs))
                    elif cmd == "step":
                        obs, reward, done, info = env.step(arrays["action"])
                        if frame_fn is not None and _recording(vep[0]):
                            vframes.append(np.asarray(frame_fn(), dtype=np.uint8))
                        resp = {
                            "ok": True,
                            "reward": [float(x) for x in np.asarray(reward).ravel()],
                            "done": bool(np.asarray(done).all()),
                            "success": [bool(i.get("success", False)) for i in info],
                        }
                        if info and "obj_z" in info[0]:            # object height (diagnostic; any task)
                            resp["obj_z"] = [float(i["obj_z"]) for i in info]
                        if info and "stress_max" in info[0]:      # soft body: per-env von-Mises
                            for key in ("stress_max", "stress_mean", "stress_top10", "stress_top20"):
                                if key in info[0]:
                                    resp[key] = [float(i[key]) for i in info]
                        send_msg(conn, resp, _as_arrays(obs))
                    elif cmd == "render":
                        # On-demand RGB for a client that writes its own video (DPPO eval/finetune
                        # bridge, SimEvalVenv). all_envs=False -> env-0 (H,W,3); all_envs=True ->
                        # all envs (N,H,W,3) for per-trajectory eval video. frame_fn=backend.render_rgb.
                        if frame_fn is not None:
                            fr = frame_fn(bool(header.get("all_envs", False)))
                            send_msg(conn, {"ok": True}, {"frame": np.asarray(fr, dtype=np.uint8)})
                        else:
                            send_msg(conn, {"ok": False, "error": "server has no frame camera (start with --render-rgb)"}, {})
                    else:
                        send_msg(conn, {"ok": False, "error": f"unknown cmd {cmd!r}"}, {})
            except (ConnectionError, OSError):
                print("  client disconnected; waiting for a new one", flush=True)
            finally:
                flush_video()
                conn.close()
    finally:
        srv.close()
        env.close()


def _as_arrays(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in obs.items()}


def _scenario_header(env) -> Dict[str, Any]:
    """reset/randomize_scene response header: per-env reset DR + material + size/shape scene
    params, for the eval audit (SimEnvClient.last_scenario -> episodes.csv)."""
    hdr: Dict[str, Any] = {"ok": True}
    be = getattr(env, "backend", None)
    dr = getattr(be, "_last_reset_dr", None)
    if dr is not None:
        hdr["dr"] = {k: (None if v is None else np.asarray(v).tolist()) for k, v in dr.items()}
    for key, attr in (("material", "material_params"), ("scene", "scene_params")):
        if be is not None and hasattr(be, attr):
            try:
                hdr[key] = getattr(be, attr)()
            except Exception:
                pass
    return hdr


# ── client: a PolicyEnv stand-in that forwards to the sim server ───────────────
class SimEnvClient:
    """Drop-in for PolicyEnv on the policy side: reset()/step()/close() over RPC."""

    num_envs = 1

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, connect_timeout: float = 240.0) -> None:
        self.conn = self._connect(host, port, connect_timeout)
        self.last_scenario = None      # {"dr":..., "material":...} from the most recent reset

    @staticmethod
    def _connect(host: str, port: int, timeout: float) -> socket.socket:
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                c = socket.create_connection((host, port), timeout=10.0)
                c.settimeout(None)                     # blocking data ops: a scene rebuild
                                                       # (randomize_scene) can take ~90s to reply
                c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return c
            except OSError as e:                       # server not up yet — retry
                last = e
                time.sleep(0.5)
        raise ConnectionError(f"could not connect to sim server at {host}:{port}: {last}")

    def reseed(self, seed: int) -> None:
        send_msg(self.conn, {"cmd": "reseed", "seed": int(seed)}, {})
        recv_msg(self.conn)

    def set_auto_scene_dr(self, enabled: bool) -> None:
        """Freeze (enabled=False) / restore the server's periodic auto scene-DR relaunch. Wrap a
        fixed-seed eval in set_auto_scene_dr(False)…(True) so the training server's every-N-resets
        rebuild can't fire mid-eval and corrupt the deterministic per-group geometry."""
        send_msg(self.conn, {"cmd": "set_auto_scene_dr", "enabled": bool(enabled)}, {})
        recv_msg(self.conn)

    def reset(self) -> Dict[str, np.ndarray]:
        send_msg(self.conn, {"cmd": "reset"}, {})
        header, obs = recv_msg(self.conn)
        # per-env randomization applied this reset (object dxy/euler, arm home) + material —
        # stashed for the eval harness (last_scenario), obs return unchanged.
        self.last_scenario = {"dr": header.get("dr"), "material": header.get("material"),
                              "scene": header.get("scene")}
        return obs

    def randomize_scene(self) -> Dict[str, np.ndarray]:
        """Rebuild the sim's scene (material + size + shape). Deterministic if reseed() was called
        first (eval per-group scene DR). Returns the fresh obs; updates last_scenario."""
        send_msg(self.conn, {"cmd": "randomize_scene"}, {})
        header, obs = recv_msg(self.conn)
        self.last_scenario = {"dr": header.get("dr"), "material": header.get("material"),
                              "scene": header.get("scene")}
        return obs

    def step(self, action: np.ndarray):
        send_msg(self.conn, {"cmd": "step"}, {"action": np.asarray(action, dtype=np.float32)})
        header, obs = recv_msg(self.conn)
        reward = np.asarray(header.get("reward", [0.0]), dtype=np.float32)
        done = np.asarray([header.get("done", False)], dtype=bool)
        info = [{"success": s} for s in header.get("success", [False])]
        if header.get("obj_z") is not None:                                  # diagnostic (any task)
            for k, d in enumerate(info):
                d["obj_z"] = float(header["obj_z"][k])
        if header.get("stress_max") is not None:                            # soft body only
            for key in ("stress_max", "stress_mean", "stress_top10", "stress_top20"):
                vals = header.get(key)
                if vals is not None:
                    for k, d in enumerate(info):
                        d[key] = float(vals[k])
        return obs, reward, done, info

    def render(self, all_envs: bool = False):
        """Request RGB from the server, or None if unavailable. all_envs=False -> env-0 (H,W,3);
        all_envs=True -> all envs (N,H,W,3) for per-trajectory eval video.
        The server must be started with a frame camera (serl_sim_server --render-rgb)."""
        send_msg(self.conn, {"cmd": "render", "all_envs": all_envs}, {})
        header, arrays = recv_msg(self.conn)
        return arrays.get("frame") if header.get("ok") else None

    def close(self) -> None:
        try:
            send_msg(self.conn, {"cmd": "close"}, {})
        except OSError:
            pass
        finally:
            self.conn.close()
