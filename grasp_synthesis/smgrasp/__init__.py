"""smgrasp — simulation-agnostic stress-minimization grasp metric (Q_SM).

Public API (built up milestone by milestone — see grasp_synthesis/CLAUDE.md §7):
  M1  build_elastic_object   geometry precompute (volume, COM, second moment)
  M4  build_elastic_object   also fills the affine stress maps A, B
  M6  q_sm                    scalar metric
  M10 sample_contacts         finger mesh + transform -> ContactSet
"""
from .types import ContactSet, ElasticObject, MetricConfig
from .bodyforce import body_force_map, eval_body_force, torque_map
from .geometry import (
    build_elastic_object,
    geometry_moments,
    load_mesh,
    tet_quadrature,
    tetrahedralize,
)
from .metric import q_sm, support_point, wrench_map

__all__ = [
    "ContactSet",
    "ElasticObject",
    "MetricConfig",
    "build_elastic_object",
    "geometry_moments",
    "load_mesh",
    "tetrahedralize",
    "q_sm",
    "support_point",
    "wrench_map",
]
