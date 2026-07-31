"""Unit tests for qml_observer.advanced.geometry.qfim."""

import numpy as np
import pytest

from qml_observer.advanced.geometry.qfim import estimate_qfim


def single_qubit_ry_state(theta: np.ndarray) -> np.ndarray:
    """|psi(theta)> = RY(theta[0]) |0>, a 1-parameter analytic reference.

    QFIM for a single-qubit RY rotation is known analytically:
    F = [[1]] (in units where the generator is Y/2, F = 1 everywhere,
    independent of theta) -- see Meyer et al. / standard single-qubit
    Fubini-Study references. Used as a ground-truth check on the
    finite-difference estimator.
    """
    t = theta[0]
    return np.array([np.cos(t / 2), np.sin(t / 2)], dtype=complex)


def two_independent_ry_state(theta: np.ndarray) -> np.ndarray:
    """Two independent qubits, each RY-rotated by its own parameter.

    Product state -> QFIM should be diagonal (no cross terms), each
    diagonal entry ~1, matching the single-qubit case per parameter.
    """
    t0, t1 = theta[0], theta[1]
    q0 = np.array([np.cos(t0 / 2), np.sin(t0 / 2)])
    q1 = np.array([np.cos(t1 / 2), np.sin(t1 / 2)])
    return np.kron(q0, q1).astype(complex)


def redundant_state(theta: np.ndarray) -> np.ndarray:
    """Two parameters that only ever appear as their sum: theta0 + theta1.

    The QFIM should be singular here (rank 1, not 2): moving along
    theta0 - theta1 (holding the sum fixed) does not change the state
    at all.
    """
    t = theta[0] + theta[1]
    return np.array([np.cos(t / 2), np.sin(t / 2)], dtype=complex)


class TestEstimateQfim:
    def test_single_parameter_matches_analytic_value(self):
        fim = estimate_qfim(single_qubit_ry_state, [0.7])
        assert fim.shape == (1, 1)
        assert fim[0, 0] == pytest.approx(1.0, abs=1e-3)

    def test_result_is_symmetric(self):
        fim = estimate_qfim(two_independent_ry_state, [0.3, 1.1])
        assert fim == pytest.approx(fim.T)

    def test_independent_parameters_give_diagonal_qfim(self):
        fim = estimate_qfim(two_independent_ry_state, [0.4, 0.9])
        assert fim[0, 1] == pytest.approx(0.0, abs=1e-3)
        assert fim[1, 0] == pytest.approx(0.0, abs=1e-3)
        assert fim[0, 0] == pytest.approx(1.0, abs=1e-3)
        assert fim[1, 1] == pytest.approx(1.0, abs=1e-3)

    def test_redundant_parameters_give_singular_qfim(self):
        fim = estimate_qfim(redundant_state, [0.2, 0.5])
        eigenvalues = np.linalg.eigvalsh(fim)
        # One eigenvalue near zero (the theta0 - theta1 null direction).
        assert eigenvalues.min() == pytest.approx(0.0, abs=1e-2)
        assert eigenvalues.max() > 0.5

    def test_qfim_is_positive_semidefinite(self):
        fim = estimate_qfim(two_independent_ry_state, [0.1, 0.2])
        eigenvalues = np.linalg.eigvalsh(fim)
        assert np.all(eigenvalues >= -1e-8)

    def test_empty_parameters_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            estimate_qfim(single_qubit_ry_state, [])

    def test_degenerate_state_raises(self):
        with pytest.raises(ValueError, match="finite, positive norm"):
            estimate_qfim(lambda theta: np.zeros(2, dtype=complex), [0.5])

    def test_nan_state_raises(self):
        with pytest.raises(ValueError, match="finite, positive norm"):
            estimate_qfim(lambda theta: np.array([np.nan, 0.0], dtype=complex), [0.5])
