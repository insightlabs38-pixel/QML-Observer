"""Unit tests for qml_observer.statistics.gradients."""

import numpy as np
import pytest

from qml_observer.statistics.gradients import (
    gradient_norm,
    gradient_percentiles,
    gradient_variance,
    mean_absolute_gradient,
)


class TestGradientNorm:
    def test_l2_norm_matches_numpy(self):
        grads = np.array([3.0, 4.0])
        assert gradient_norm(grads) == pytest.approx(5.0)

    def test_l1_norm(self):
        grads = np.array([1.0, -2.0, 3.0])
        assert gradient_norm(grads, ord=1) == pytest.approx(6.0)

    def test_multi_dimensional_is_flattened(self):
        grads = np.array([[3.0, 4.0]])
        assert gradient_norm(grads) == pytest.approx(5.0)

    def test_list_input_is_coerced(self):
        assert gradient_norm([3.0, 4.0]) == pytest.approx(5.0)

    def test_empty_array_raises_clear_error(self):
        with pytest.raises(ValueError, match="empty gradient array"):
            gradient_norm(np.array([]))

    def test_nan_propagates_without_raising(self):
        result = gradient_norm(np.array([float("nan"), 1.0]))
        assert math_isnan(result)

    def test_inf_propagates(self):
        result = gradient_norm(np.array([float("inf"), 1.0]))
        assert result == float("inf")


class TestMeanAbsoluteGradient:
    def test_matches_numpy(self):
        grads = np.array([1.0, -2.0, 3.0])
        assert mean_absolute_gradient(grads) == pytest.approx(2.0)

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty gradient array"):
            mean_absolute_gradient(np.array([]))


class TestGradientVariance:
    def test_matches_numpy_population_variance(self):
        grads = np.array([1.0, 2.0, 3.0, 4.0])
        assert gradient_variance(grads) == pytest.approx(float(np.var(grads)))

    def test_single_value_has_zero_variance(self):
        assert gradient_variance(np.array([5.0])) == pytest.approx(0.0)

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty gradient array"):
            gradient_variance(np.array([]))

    def test_nan_propagates(self):
        result = gradient_variance(np.array([float("nan"), 1.0]))
        assert math_isnan(result)


class TestGradientPercentiles:
    def test_default_percentiles_returned(self):
        grads = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gradient_percentiles(grads)
        assert set(result.keys()) == {1, 10, 50, 90, 99}
        assert result[50] == pytest.approx(3.0)

    def test_custom_percentiles(self):
        grads = np.arange(1, 101, dtype=float)
        result = gradient_percentiles(grads, percentiles=(25, 75))
        assert result[25] == pytest.approx(float(np.percentile(grads, 25)))
        assert result[75] == pytest.approx(float(np.percentile(grads, 75)))

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty gradient array"):
            gradient_percentiles(np.array([]))

    def test_empty_percentiles_raises(self):
        with pytest.raises(ValueError, match="at least one percentile"):
            gradient_percentiles(np.array([1.0, 2.0]), percentiles=())

    def test_out_of_range_percentile_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            gradient_percentiles(np.array([1.0, 2.0]), percentiles=(150,))

        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            gradient_percentiles(np.array([1.0, 2.0]), percentiles=(-1,))


def math_isnan(value: float) -> bool:
    return value != value
