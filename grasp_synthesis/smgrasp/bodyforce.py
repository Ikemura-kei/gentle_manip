"""M2 — wrench → body-force field map P (grasp_synthesis/CLAUDE.md §3.2).

Represent an external wrench w = (w_f, w_τ) ∈ ℝ⁶ as a body-force field linear in x:
    g(x) = g0 + G x           (12 DOF: g0 ∈ ℝ³, G = ∇g ∈ ℝ^{3×3})

With the origin at the COM (∫ x dx = 0):
    net force  ∫ g dx = |Ω| g0                      ⇒  g0 = w_f / |Ω|
    net torque ∫ x × g dx = T · vec(G) = w_τ        (the g0 part vanishes since ∫ x dx = 0)

T (3×9) is built from the second moment S: column (c,k) of T is S[:,k] × e_c, i.e.
T[a, 3c+k] = ε_{abc} S_{bk}. G is under-determined (9 unknowns, 3 constraints); solve the
min-norm problem  min ∫ ‖G x‖² dx = vec(G)ᵀ (I₃ ⊗ S) vec(G)  s.t.  T vec(G) = w_τ:

    vec(G) = M⁻¹ Tᵀ (T M⁻¹ Tᵀ)⁻¹ w_τ,   M = I₃ ⊗ S      (vec is ROW-major: vec[3c+k]=G[c,k])

P : ℝ⁶ → ℝ¹² stacks (g0, vec(G)) = P w and is precomputed once per object.

NOTE on vec convention: this module uses row-major vec(G) throughout (T and M derived in the
same convention), so the min-norm solve is internally consistent; the CLAUDE.md's "M = S⊗I₃"
is the column-major statement of the same map. The M2 invariant test (integrate g and x×g,
recover w) is convention-independent and is what pins correctness.
"""
from __future__ import annotations

import numpy as np

# Levi-Civita symbol ε_{abc}
_EPS = np.zeros((3, 3, 3))
_EPS[0, 1, 2] = _EPS[1, 2, 0] = _EPS[2, 0, 1] = 1.0
_EPS[0, 2, 1] = _EPS[2, 1, 0] = _EPS[1, 0, 2] = -1.0


def torque_map(S: np.ndarray) -> np.ndarray:
    """T (3, 9): net torque of the linear field G x is T @ vec(G) (row-major vec(G))."""
    return np.einsum("abc,bk->ack", _EPS, S).reshape(3, 9)


def body_force_map(volume: float, S: np.ndarray) -> np.ndarray:
    """Precompute P (12, 6): (g0, vec(G)) = P @ w, w = (w_f(3), w_τ(3))."""
    S = np.asarray(S, float).reshape(3, 3)
    T = torque_map(S)                                     # (3, 9)
    M = np.kron(np.eye(3), S)                             # (9, 9) = I₃ ⊗ S, SPD
    Minv_Tt = np.linalg.solve(M, T.T)                     # M⁻¹ Tᵀ  (9, 3)
    grad_from_tau = Minv_Tt @ np.linalg.inv(T @ Minv_Tt)  # (9, 3)  vec(G) = (...) w_τ

    P = np.zeros((12, 6))
    P[0:3, 0:3] = np.eye(3) / volume                      # g0 = w_f / |Ω|
    P[3:12, 3:6] = grad_from_tau                          # vec(G) from w_τ
    return P


def eval_body_force(P: np.ndarray, w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate g(x) = g0 + G x for wrench w at points x (…, 3) -> (…, 3)."""
    coef = P @ np.asarray(w, float).reshape(6)            # (12,)
    g0, G = coef[:3], coef[3:].reshape(3, 3)
    x = np.asarray(x, float)
    return g0 + x @ G.T                                   # (…, 3)
