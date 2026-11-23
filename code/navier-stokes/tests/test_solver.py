import os

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from ns_amg.fem import Config, Discretization, simulate
from ns_amg.linear import AMGSolver
from ns_amg.pipeline import convergence

BACKEND = os.environ.get("NS_TEST_BACKEND", "pyamg")


def test_projection_preserves_wall_values_and_removes_weak_divergence():
    d = Discretization(Config(cells=4))
    np.testing.assert_allclose(d.schur @ np.ones(d.pressure.N), 0, atol=1e-13)
    assert np.linalg.eigvalsh(d.pressure_matrix.toarray()).min() > 0
    rng = np.random.default_rng(5)
    u, v, p = d.project(rng.normal(size=len(d.free)), rng.normal(size=len(d.free)),
                         lambda rhs: spsolve(d.pressure_matrix, rhs))
    assert np.linalg.norm(d.bx @ u + d.by @ v) < 1e-12
    assert np.max(np.abs(d.expand(u)[d.boundary])) == 0
    assert abs(d.pressure_weights @ p) < 1e-12


@pytest.mark.parametrize("theta", [0.1, 0.5, 0.9])
def test_amg_matches_direct_solution_and_reuses_setup(theta):
    matrix = Discretization(Config(cells=4)).pressure_matrix
    rhs = np.random.default_rng(0).normal(size=matrix.shape[0])
    solver = AMGSolver(matrix, theta, BACKEND)
    try:
        expected = spsolve(matrix, rhs)
        np.testing.assert_allclose(solver.solve(rhs), expected, atol=1e-7, rtol=1e-7)
        setup_seconds = solver.setup_seconds
        np.testing.assert_allclose(solver.solve(2 * rhs), 2 * expected, atol=1e-7, rtol=1e-7)
        assert solver.setup_seconds == setup_seconds
        assert solver.summary()["solves"] == 2
    finally:
        solver.close()


def test_manufactured_navier_stokes_refines(tmp_path):
    rows = convergence(tmp_path / "convergence", BACKEND)
    assert rows[-1]["velocity_l2_error"] < 0.001
    assert rows[-1]["pressure_l2_error"] < 0.006
    assert all(row["weak_divergence"] < 1e-8 for row in rows)


def test_selector_is_used_once_per_operator():
    calls = []
    def select(matrix, role, horizon):
        calls.append((role, horizon))
        return 0.3 if role == "pressure" else 0.7
    result, _, _ = simulate(Config(cells=4, steps=3), BACKEND, select)
    assert calls == [("velocity", 6), ("pressure", 4)]
    assert result["linear"]["velocity"]["theta"] == 0.7
    assert result["linear"]["pressure"]["theta"] == 0.3
    assert result["linear"]["velocity"]["solves"] == 6
    assert result["linear"]["pressure"]["solves"] == 4


def test_unstable_explicit_step_is_rejected():
    with pytest.raises(ValueError, match="CFL"):
        simulate(Config(cells=4, dt=0.1), BACKEND)


@pytest.mark.skipif(BACKEND != "petsc-cuda", reason="requires the cluster CUDA runtime")
def test_cuda_flow_agrees_with_cpu_petsc():
    config = Config(cells=8, steps=3)
    _, reference, _ = simulate(config, "petsc")
    report, actual, _ = simulate(config, "petsc-cuda")
    for role in ("velocity", "pressure"):
        assert report["linear"][role]["matrix_type"] == "seqaijcusparse"
        assert report["linear"][role]["vector_type"] == "seqcuda"
    for name in ("u", "v", "p"):
        np.testing.assert_allclose(actual[name], reference[name], atol=1e-6, rtol=1e-6)
