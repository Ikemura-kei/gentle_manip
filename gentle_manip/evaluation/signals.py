"""Per-episode COMMAND-vs-STATE signal plots for sim evals (2026-09-06, user request).

The harness buffers, per policy step and env, the pre-step proprio (ee_pos, ee_quat if present,
gripper width) and the action chunk the policy emitted. Here the chunk is decoded to physical
targets (derive-space [-1,1] -> ActionPipeline of the experiment's action config), laid out at
sub-step time, and overlaid on the state, which is sampled once per chunk. Optional per-step
flags (e.g. whether the obs augmentation injected residue) are drawn as a binary trace.

Written to <eval>/signals/epNNN.png + epNNN.npz. Genesis-free; numpy + matplotlib only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def decode_chunks(act_chunks: np.ndarray, action_config):
    """(T, A*K) or (T, K, A) derive-space chunks -> pos (T*K,3), euler xyz deg (T*K,3), grip (T*K,)."""
    from scipy.spatial.transform import Rotation as R
    from gentle_manip.actions.pipeline import ActionPipeline
    a = np.asarray(act_chunks, np.float32)
    A = action_config.action_dim
    a = a.reshape(a.shape[0], -1, A).reshape(-1, A)                      # (T*K, A)
    phys = ActionPipeline(action_config).process(a)                     # (T*K, 8) pos, quat wxyz, grip
    eul = R.from_quat(phys[:, [4, 5, 6, 3]]).as_euler("xyz", degrees=True)
    return phys[:, :3], eul, phys[:, 7], a.shape[0] // max(act_chunks.shape[0], 1)


def plot_plan_vs_executed(out_png: Path, *, act_chunks, act_chunks_full, action_config,
                          act_steps: int, title: str, demo_width_m: float | None = None) -> None:
    """Gripper width the policy PLANNED vs what it EXECUTED, on an absolute-step axis.

    With horizon H executing K < H steps, only h=0..K-1 of each chunk ever runs and the rest is
    replaced by the next decision. Drawing each chunk from its own decision point shows whether a
    policy plans the right closing width and then drifts off it (the plans dip below the executed
    trace) or never plans it at all (plans and execution coincide).

    Measured on G3/cherry 2026-09-09: successes execute within 1.1 mm of plan, failures fall 4.3 mm
    short while planning the SAME width -- i.e. the width was predicted correctly and not realized.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = np.asarray(act_chunks_full)
    if full.size == 0:
        return
    K = int(act_steps)
    _, _, ex_w, _ = decode_chunks(np.asarray(act_chunks), action_config)      # (T*K,) metres
    ex_mm = np.asarray(ex_w, float) * 1000.0
    T, H = full.shape[0], full.shape[1]
    _, _, fu_w, _ = decode_chunks(full, action_config)                        # (T*H,) metres
    fu_mm = np.asarray(fu_w, float).reshape(T, H) * 1000.0

    fig, ax = plt.subplots(figsize=(11, 3.6))
    for t in range(T):                                    # every chunk's full plan, from its own step
        ax.plot(np.arange(t * K, t * K + H), fu_mm[t], color="#d62728", lw=0.6, alpha=0.30)
    ax.plot(np.arange(ex_mm.size), ex_mm, color="#1f77b4", lw=2.0, label="executed", zorder=5)
    ax.plot([], [], color="#d62728", lw=0.9, alpha=0.8, label=f"each chunk's {H}-step plan")
    for b in range(0, ex_mm.size + K, K):                 # chunk breaks: plan replaced here
        ax.axvline(b, color="0.8", lw=0.4, ls=":", zorder=0)
    if demo_width_m:
        ax.axhline(demo_width_m * 1000.0, color="green", lw=1.0, ls="--", alpha=0.7,
                   label=f"demo {demo_width_m * 1000:.0f} mm")
    # headline number: how much looser the execution ended up than the tightest realizable plan
    reach = [fu_mm[t, h] for t in range(T) for h in range(H) if t * K + h < ex_mm.size]
    if reach:
        short = float(ex_mm.min() - min(reach))
        ax.set_title(f"{title}   |   plan min {min(reach):.1f} mm, exec min {ex_mm.min():.1f} mm, "
                     f"shortfall {short:+.1f} mm", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel("absolute action step (dotted = chunk break)"); ax.set_ylabel("gripper width (mm)")
    ax.grid(alpha=0.2); ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def plot_episode(out_png: Path, *, ee, grip, quat, act_chunks, action_config, act_steps: int,
                 dt: float, title: str, flags: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R
    ee = np.asarray(ee, float); grip = np.asarray(grip, float).reshape(-1)
    T = len(ee)
    cpos, ceul, cw, K = decode_chunks(np.asarray(act_chunks), action_config)
    K = act_steps
    t_state = np.arange(T) * K * dt
    t_cmd = np.arange(len(cpos)) * dt
    unwrap = lambda e: np.degrees(np.unwrap(np.radians(e), axis=0))
    ceul = unwrap(ceul)
    seul = unwrap(R.from_quat(np.asarray(quat)[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)) if quat is not None else None
    nrow = 3 + (1 if flags else 0)
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 2.7 * nrow), sharex=True)
    for i, lab in enumerate("xyz"):
        ax = axes[0, i]; ax.plot(t_cmd, cpos[:, i] * 1e3, "C1", lw=1, label="command (chunk)"); ax.plot(t_state, ee[:, i] * 1e3, "C0.-", lw=1, ms=3, label="state (per policy step)")
        ax.set_title(f"pos {lab} [mm]"); ax.grid(alpha=.3)
        ax = axes[1, i]; ax.plot(t_cmd, ceul[:, i], "C1", lw=1)
        if seul is not None: ax.plot(t_state, seul[:, i], "C0.-", lw=1, ms=3)
        ax.set_title(f"euler {['roll','pitch','yaw'][i]} [deg]"); ax.grid(alpha=.3)
    ax = axes[2, 0]; ax.plot(t_cmd, cw * 1e3, "C1", lw=1, label="command"); ax.plot(t_state, grip * 1e3, "C0.-", lw=1, ms=3, label="state"); ax.set_title("gripper width [mm]"); ax.grid(alpha=.3); ax.legend(loc="best", fontsize=8)
    err = np.linalg.norm(cpos[::K][:T] - ee[:len(cpos[::K][:T])], axis=1) * 1e3
    ax = axes[2, 1]; ax.plot(t_state[:len(err)], err, "k", lw=1); ax.set_title("|first cmd of chunk - state| [mm]"); ax.grid(alpha=.3)
    ax = axes[2, 2]; ax.plot(t_state[1:], np.linalg.norm(np.diff(ee, axis=0), axis=1) * 1e3 / (K * dt), "C0", lw=1); ax.set_title("EE speed [mm/s]"); ax.grid(alpha=.3)
    if flags:
        for i, (name, v) in enumerate(list(flags.items())[:3]):
            ax = axes[3, i]; v = np.asarray(v, float).reshape(-1); ax.step(t_state[:len(v)], v, "C3", where="post", lw=1.2)
            ax.set_ylim(-0.1, max(1.1, v.max() + 0.1) if v.size else 1.1); ax.set_title(f"{name} (sub-steps flagged per policy-step chunk)"); ax.grid(alpha=.3)
        for i in range(len(flags), 3): axes[3, i].axis("off")
    for ax in axes[-1]: ax.set_xlabel("t [s]")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_png, dpi=100); plt.close(fig)
