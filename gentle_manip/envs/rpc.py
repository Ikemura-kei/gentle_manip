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
def serve_env(env, host: str = "127.0.0.1", port: int = 5555, ready_msg: str = "SIM_SERVER_READY") -> None:
    """Serve reset/step/close requests for ``env`` until the client disconnects.

    ``env`` must expose reset()->obs dict and step(action)->(obs, reward, done, info),
    matching PolicyEnv. Prints ``ready_msg`` once the port is bound (so a launcher
    can wait for it).
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(ready_msg, flush=True)
    conn, _ = srv.accept()
    try:
        while True:
            header, arrays = recv_msg(conn)
            cmd = header.get("cmd")
            if cmd == "close":
                break
            elif cmd == "reset":
                send_msg(conn, {"ok": True}, _as_arrays(env.reset()))
            elif cmd == "step":
                obs, reward, done, info = env.step(arrays["action"])
                send_msg(conn, {
                    "ok": True,
                    "reward": [float(x) for x in np.asarray(reward).ravel()],
                    "done": bool(np.asarray(done).all()),
                    "success": [bool(i.get("success", False)) for i in info],
                }, _as_arrays(obs))
            else:
                send_msg(conn, {"ok": False, "error": f"unknown cmd {cmd!r}"}, {})
    except (ConnectionError, OSError):
        pass
    finally:
        conn.close()
        srv.close()
        env.close()


def _as_arrays(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in obs.items()}


# ── client: a PolicyEnv stand-in that forwards to the sim server ───────────────
class SimEnvClient:
    """Drop-in for PolicyEnv on the policy side: reset()/step()/close() over RPC."""

    num_envs = 1

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, connect_timeout: float = 240.0) -> None:
        self.conn = self._connect(host, port, connect_timeout)

    @staticmethod
    def _connect(host: str, port: int, timeout: float) -> socket.socket:
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                c = socket.create_connection((host, port), timeout=10.0)
                c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return c
            except OSError as e:                       # server not up yet — retry
                last = e
                time.sleep(0.5)
        raise ConnectionError(f"could not connect to sim server at {host}:{port}: {last}")

    def reset(self) -> Dict[str, np.ndarray]:
        send_msg(self.conn, {"cmd": "reset"}, {})
        _, obs = recv_msg(self.conn)
        return obs

    def step(self, action: np.ndarray):
        send_msg(self.conn, {"cmd": "step"}, {"action": np.asarray(action, dtype=np.float32)})
        header, obs = recv_msg(self.conn)
        reward = np.asarray(header.get("reward", [0.0]), dtype=np.float32)
        done = np.asarray([header.get("done", False)], dtype=bool)
        info = [{"success": s} for s in header.get("success", [False])]
        return obs, reward, done, info

    def close(self) -> None:
        try:
            send_msg(self.conn, {"cmd": "close"}, {})
        except OSError:
            pass
        finally:
            self.conn.close()
