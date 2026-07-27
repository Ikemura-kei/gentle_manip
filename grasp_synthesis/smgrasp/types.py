"""Core datatypes for the stress-minimization grasp metric (Q_SM).

See grasp_synthesis/CLAUDE.md §2. Everything is in the object's COM frame (the mesh is
recentered so ∫_Ω x dx = 0 before any moment/body-force formula is applied).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ContactSet:
    """A set of point contacts on the object surface (object COM frame)."""
    points: np.ndarray            # (N, 3) contact points, object COM frame
    normals: np.ndarray           # (N, 3) unit, pointing INTO the material (into the object)
    mu: float                     # Coulomb friction coefficient

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        self.normals = np.asarray(self.normals, dtype=float).reshape(-1, 3)
        n = np.linalg.norm(self.normals, axis=1, keepdims=True)
        if np.any(n < 1e-12):
            raise ValueError("ContactSet.normals contains a zero-length normal")
        self.normals = self.normals / n
        if self.points.shape[0] != self.normals.shape[0]:
            raise ValueError("points and normals must have the same length")

    @property
    def n_contacts(self) -> int:
        return self.points.shape[0]


@dataclass
class MetricConfig:
    """Knobs for the metric. Units are normalized: E = 1, σ_max = 1 (only ν is physical)."""
    nu: float = 0.33                          # Poisson ratio (default copper, as in the paper)
    W: Optional[np.ndarray] = None            # 6x6 SPD wrench-space metric (None -> identity)
    n_dirs: int = 64                          # initial sampled directions on S^5
    eps: float = 1e-3                         # outer-loop convergence tolerance
    mask_contact_elems: bool = True           # exclude elements directly under contacts (§6.2)

    def wrench_metric(self) -> np.ndarray:
        return np.eye(6) if self.W is None else np.asarray(self.W, dtype=float).reshape(6, 6)


@dataclass
class ElasticObject:
    """Everything precomputed once per object (M1 geometry; M4 fills A, B)."""
    verts: np.ndarray                         # (n, 3) tet-mesh nodes, RECENTERED to COM
    tets: np.ndarray                          # (m, 4) int tetra vertex indices
    volume: float                             # |Ω| = ∫_Ω dx
    com: np.ndarray                           # (3,) COM in the ORIGINAL mesh frame (pre-recenter)
    second_moment: np.ndarray                 # (3, 3) S = ∫_Ω x xᵀ dx (about COM, SPD)
    nu: float = 0.33
    # Affine stress bases (filled by stressmap.py at M4):
    A: Optional[np.ndarray] = None            # (M, 6) per-element stress-vs-wrench (Voigt-stacked)
    B: Optional[np.ndarray] = None            # (M, 3N) per-element stress-vs-contact-force
    elem_centroids: Optional[np.ndarray] = None   # (m, 3) tet centroids (recentered frame)
