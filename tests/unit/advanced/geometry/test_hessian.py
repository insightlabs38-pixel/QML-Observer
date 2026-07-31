"""Unit tests for qml_observer.advanced.geometry.hessian."""

import numpy as np
import pytest

from qml_observer.advanced.geometry.hessian import estimate_hessian_vector_product


def quadratic_loss(theta: np.ndarray) -> float:
    """f(theta) = 0.5 * theta^T A theta, A = diag([2, 4, 6]).

    Hessian is exactly A everywhere; Hv = A @ v is the ground truth for
    any v and any theta (a quadratic's Hessian is constant).
    """
    a = np.array([2.0, 4.0, 6.0])
    return float(0.5 * np.sum(a * theta**2))


def coupled_quadratic_loss(theta: np.ndarray) -> float:
    """f(theta) = theta0^2 + theta1^2 + theta0*theta1.

    Hessian = [[2, 1], [1, 2]] everywhere.
    """
    return float(theta[0] ** 2 + theta[1] ** 2 + theta[0] * theta[1])


class TestEstimateHessianVectorProduct:
    def test_matches_analytic_hvp_diagonal_case(self):
        theta = np.array([0.3, -0.5, 1.2])
        v = np.array([1.0, 0.0, 0.0])
        hv = estimate_hessian_vector_product(quadratic_loss, theta, v)
        expected = np.array([2.0, 0.0, 0.0])  # A @ v
        assert hv == pytest.approx(expected, abs=1e-3)

    def test_matches_analytic_hvp_coupled_case(self):
        theta = np.array([0.1, 0.2])
        v = np.array([1.0, 1.0])
        hv = estimate_hessian_vector_product(coupled_quadratic_loss, theta, v)
        expected = np.array([[2.0, 1.0], [1.0, 2.0]]) @ v
        assert hv == pytest.approx(expected, abs=1e-3)

    def test_result_independent_of_theta_for_quadratic(self):
        v = np.array([0.0, 1.0, 0.0])
        hv_a = estimate_hessian_vector_product(quadratic_loss, [0.0, 0.0, 0.0], v)
        hv_b = estimate_hessian_vector_product(quadratic_loss, [5.0, -3.0, 2.0], v)
        assert hv_a == pytest.approx(hv_b, abs=1e-3)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="same shape"):
            estimate_hessian_vector_product(quadratic_loss, [0.1, 0.2, 0.3], [1.0, 0.0])

    def test_empty_parameters_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            estimate_hessian_vector_product(quadratic_loss, [], [])

    def test_non_positive_eps_raises(self):
        with pytest.raises(ValueError, match="eps"):
            estimate_hessian_vector_product(
                quadratic_loss, [0.1, 0.2, 0.3], [1.0, 0.0, 0.0], eps=0.0
            )
        with pytest.raises(ValueError, match="grad_eps"):
            estimate_hessian_vector_product(
                quadratic_loss, [0.1, 0.2, 0.3], [1.0, 0.0, 0.0], grad_eps=-1.0
            )
