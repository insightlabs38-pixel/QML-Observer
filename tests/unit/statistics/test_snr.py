"""Unit tests for qml_observer.statistics.snr."""

import math

import pytest

from qml_observer.statistics.snr import estimate_gradient_snr, estimate_measurement_uncertainty


class TestEstimateGradientSnr:
    def test_basic_ratio(self):
        assert estimate_gradient_snr(mean_gradient=1.0, gradient_std=0.5) == pytest.approx(2.0)

    def test_sign_of_mean_is_discarded(self):
        assert estimate_gradient_snr(-1.0, 0.5) == pytest.approx(2.0)

    def test_zero_std_zero_mean_is_zero_not_inf(self):
        """A fully degenerate all-zero estimate is 'no signal', not 'perfect SNR'."""
        assert estimate_gradient_snr(0.0, 0.0) == 0.0

    def test_zero_std_nonzero_mean_is_inf(self):
        assert estimate_gradient_snr(1.0, 0.0) == float("inf")

    def test_negative_std_raises(self):
        with pytest.raises(ValueError, match="gradient_std"):
            estimate_gradient_snr(1.0, -0.1)

    def test_nan_std_yields_nan(self):
        assert math.isnan(estimate_gradient_snr(1.0, float("nan")))

    def test_nan_mean_yields_nan(self):
        assert math.isnan(estimate_gradient_snr(float("nan"), 0.5))

    def test_non_numeric_raises_type_error(self):
        with pytest.raises(TypeError):
            estimate_gradient_snr("1.0", 0.5)
        with pytest.raises(TypeError):
            estimate_gradient_snr(1.0, "0.5")

    def test_bool_raises_type_error(self):
        with pytest.raises(TypeError):
            estimate_gradient_snr(True, 0.5)


class TestEstimateMeasurementUncertainty:
    def test_basic_formula(self):
        # sqrt(4.0 / 100) = 0.2
        assert estimate_measurement_uncertainty(4.0, 100) == pytest.approx(0.2)

    def test_more_shots_reduces_uncertainty(self):
        low_shots = estimate_measurement_uncertainty(1.0, 10)
        high_shots = estimate_measurement_uncertainty(1.0, 1000)
        assert high_shots < low_shots

    def test_zero_variance_gives_zero_uncertainty(self):
        assert estimate_measurement_uncertainty(0.0, 50) == 0.0

    def test_shots_must_be_positive(self):
        with pytest.raises(ValueError, match="shots"):
            estimate_measurement_uncertainty(1.0, 0)
        with pytest.raises(ValueError, match="shots"):
            estimate_measurement_uncertainty(1.0, -5)

    def test_shots_must_be_int(self):
        with pytest.raises(TypeError):
            estimate_measurement_uncertainty(1.0, 10.5)
        with pytest.raises(TypeError):
            estimate_measurement_uncertainty(1.0, True)

    def test_negative_variance_raises(self):
        with pytest.raises(ValueError, match="expectation_variance"):
            estimate_measurement_uncertainty(-1.0, 10)

    def test_nan_variance_yields_nan(self):
        assert math.isnan(estimate_measurement_uncertainty(float("nan"), 10))

    def test_inf_variance_yields_inf(self):
        assert estimate_measurement_uncertainty(float("inf"), 10) == float("inf")

    def test_non_numeric_variance_raises_type_error(self):
        with pytest.raises(TypeError):
            estimate_measurement_uncertainty("1.0", 10)
