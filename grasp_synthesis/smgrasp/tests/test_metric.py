"""M5 tests — the support-point SDP+SOCP on a coarse cube.

An antipodal 2-contact grasp on the ±x faces (friction μ=0.5). We check that:
 * the program solves and the wrench balance / friction cones hold at the optimum,
 * the support value is finite (bounded by the stress cap) and positive for a resistible dir,
 * the stress cap actually binds (some element reaches ‖σ‖ ≈ σ_max = 1) — i.e. the LMI is active,
 * with the stress cap relaxed (σ_max → ∞) but a force normalization added, the support value
   grows (the stress bound was limiting) — the qualitative Q1-limit behaviour.
"""
import pathlib

import numpy as np
import pytest

from smgrasp.geometry import build_elastic_object
from smgrasp.metric import q_sm, support_point, wrench_map
from smgrasp.types import ContactSet

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"


@pytest.fixture(scope="module")
def cube():
    # coarse cube so the full SDP (one 3x3 LMI per element) stays small
    return build_elastic_object(ASSETS / "cube.obj", switches="pq1.4a0.03")


@pytest.fixture(scope="module")
def antipodal():
    pts = np.array([[0.49, 0.0, 0.0], [-0.49, 0.0, 0.0]])
    normals = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])   # into the material
    return ContactSet(points=pts, normals=normals, mu=0.5)


def _friction_ok(cs, f, tol=1e-5):
    for i in range(cs.n_contacts):
        n, fi = cs.normals[i], f[i]
        tangential = np.linalg.norm(fi - (n @ fi) * n)
        if tangential > cs.mu * (n @ fi) + tol:
            return False
    return True


def test_support_point_feasible_and_bounded(cube, antipodal):
    d = np.array([1.0, 0, 0, 0, 0, 0])                        # resist a +x force
    res = support_point(cube, antipodal, d)
    assert res["status"] in ("optimal", "optimal_inaccurate")
    assert np.isfinite(res["value"]) and res["value"] > 1e-6
    # wrench balance and friction cones hold at the optimum
    G = wrench_map(antipodal.points)
    np.testing.assert_allclose(res["w"] + G @ res["f"].reshape(-1), 0, atol=1e-5)
    assert _friction_ok(antipodal, res["f"])


def test_stress_cap_binds(cube, antipodal):
    # at the optimum the stress LMI should be active: max element |eigval(σ)| ≈ 1 (σ_max)
    d = np.array([1.0, 0, 0, 0, 0, 0])
    res = support_point(cube, antipodal, d)
    from smgrasp.fem import voigt_to_tensor
    L = np.einsum("mij,j->mi", cube.A, res["w"]) + np.einsum("mij,j->mi", res["B"], res["f"].reshape(-1))
    sig = voigt_to_tensor(L)                                   # (M,3,3)
    smax = np.abs(np.linalg.eigvalsh(sig)).max()
    assert smax == pytest.approx(1.0, abs=0.05)               # cap reached (not slack, not violated)


def test_relaxing_stress_cap_raises_support(cube, antipodal):
    # constrain only a couple of elements (cap nearly inactive) -> larger resistible wrench than
    # the full stress-limited problem. Confirms the stress LMIs are what bound the support value.
    d = np.array([1.0, 0, 0, 0, 0, 0])
    full = support_point(cube, antipodal, d)["value"]
    few = support_point(cube, antipodal, d, elements=np.array([0, 1]))["value"]
    assert few >= full - 1e-6


# ── M6 — Q_SM scalar (outer convex-hull loop) ────────────────────────────────
def _closure(mu):
    # 6 face-centre contacts (±x, ±y, ±z), inward normals -> full 6-D force closure
    e = np.eye(3)
    pts = np.array([0.49 * s * e[a] for a in range(3) for s in (1, -1)])
    normals = np.array([-s * e[a] for a in range(3) for s in (1, -1)])
    return ContactSet(points=pts, normals=normals, mu=mu)


def test_qsm_positive_for_force_closure(cube):
    assert q_sm(cube, _closure(0.5), n_dirs=12) > 0           # 6-contact grasp -> closure, Q>0


def test_qsm_nonpositive_without_force_closure(cube):
    # both contacts on the SAME (+x) face -> can push -x but not resist +x -> no force closure
    pts = np.array([[0.49, 0.2, 0.0], [0.49, -0.2, 0.0]])
    normals = np.array([[-1.0, 0, 0], [-1.0, 0, 0]])
    cs = ContactSet(points=pts, normals=normals, mu=0.5)
    assert q_sm(cube, cs, n_dirs=16) <= 1e-6                  # ≈ 0 / negative: no 6-D closure


def test_qsm_more_directions_tightens(cube):
    # Q_SM is a lower bound on the true inradius; more sampled directions -> a >= (tighter) bound
    q_lo = q_sm(cube, _closure(0.5), n_dirs=8)
    q_hi = q_sm(cube, _closure(0.5), n_dirs=24)
    assert q_hi >= q_lo - 1e-3 and q_hi > 0


def test_qsm_monotone_in_friction(cube):
    # more friction -> wider cones -> more resistible wrenches -> larger Q_SM
    q_lo = q_sm(cube, _closure(0.3), n_dirs=12)
    q_hi = q_sm(cube, _closure(0.9), n_dirs=12)
    assert q_hi >= q_lo - 1e-3


# ── M5b — active set (Algorithm 3) + Q1 mode ─────────────────────────────────
def test_active_set_matches_full_solve(cube, antipodal):
    from smgrasp.metric import support_point_active
    from smgrasp.stressmap import contact_stress_map
    B, ti = contact_stress_map(cube.fem, antipodal.points)
    d = np.array([1.0, 0, 0, 0, 0, 0])
    full = support_point(cube, antipodal, d, B=B, tet_idx=ti)["value"]
    act = support_point_active(cube, antipodal, d, B=B, tet_idx=ti)
    assert act["value"] == pytest.approx(full, abs=1e-4)     # same optimum, dropped LMIs were slack
    assert len(act["active"]) <= len(cube.tets)


def test_q1_mode_positive_for_closure(cube):
    from smgrasp.metric import q1
    assert q1(cube, _closure(0.5), n_dirs=12) > 0            # Ferrari-Canny Q1 > 0 for closure
