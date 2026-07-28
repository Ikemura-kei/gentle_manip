"""M9 test — the CMA-ES Q_SM planner improves on a random grasp (kept lean: coarse mesh, few evals)."""
import numpy as np
import trimesh

from smgrasp.geometry import build_elastic_object
from smgrasp.metric import q_sm
from smgrasp.planner import grasp_contacts, plan_grasp


def test_planner_finds_positive_qsm():
    obj = build_elastic_object(trimesh.creation.box(extents=[1, 1, 1]), switches="pq1.4a0.05")
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    res = plan_grasp(obj, mesh, maxfevals=24, n_dirs=8, pad_half=0.25, mu=0.6, sigma=0.35, seed=1)
    assert res["contacts"] is not None
    assert res["q_sm"] > 0                                     # found a force-closure grasp
    # the reported best equals a fresh evaluation of the returned pose (consistency)
    q = q_sm(obj, res["contacts"], n_dirs=8)
    assert q == __import__("pytest").approx(res["q_sm"], abs=0.05)
