"""Unit tests for qml_observer.advanced.scaling."""

import math

import numpy as np
import pytest

from qml_observer.advanced.scaling import (
    ScalingAnalysisResult,
    ScalingAnalyzer,
    ScalingObservation,
    scaling_observation_from_run_summary,
)


def exponential_runs(n_qubits_values, a=0.5, b=1.0, depth=10):
    """Synthetic runs with *exact* exponential decay: variance = b*exp(-a*n)."""
    return [
        ScalingObservation(
            n_qubits=n, gradient_variance=b * math.exp(-a * n), depth=depth, label=f"n{n}"
        )
        for n in n_qubits_values
    ]


class TestScalingObservation:
    def test_valid_construction(self):
        obs = ScalingObservation(n_qubits=4, gradient_variance=0.01, depth=3)
        assert obs.n_qubits == 4
        assert obs.depth == 3

    def test_zero_variance_is_legal(self):
        obs = ScalingObservation(n_qubits=4, gradient_variance=0.0)
        assert obs.gradient_variance == 0.0

    def test_nan_variance_is_legal(self):
        obs = ScalingObservation(n_qubits=4, gradient_variance=float("nan"))
        assert math.isnan(obs.gradient_variance)

    def test_negative_variance_raises(self):
        with pytest.raises(ValueError, match="gradient_variance"):
            ScalingObservation(n_qubits=4, gradient_variance=-0.1)

    def test_non_positive_n_qubits_raises(self):
        with pytest.raises(ValueError, match="n_qubits"):
            ScalingObservation(n_qubits=0, gradient_variance=0.1)

    def test_negative_depth_raises(self):
        with pytest.raises(ValueError, match="depth"):
            ScalingObservation(n_qubits=4, gradient_variance=0.1, depth=-1)


class TestAnalyzeQubitScaling:
    def test_perfect_exponential_decay_is_detected(self):
        runs = exponential_runs(range(4, 13))
        result = ScalingAnalyzer().analyze_qubit_scaling(runs)
        assert isinstance(result, ScalingAnalysisResult)
        assert result.x_label == "n_qubits"
        assert result.slope == pytest.approx(-0.5, abs=1e-6)
        assert result.r_squared == pytest.approx(1.0, abs=1e-6)
        assert result.consistent_with_exponential_decay is True

    def test_flat_variance_is_not_consistent_with_decay(self):
        runs = [
            ScalingObservation(n_qubits=n, gradient_variance=0.5, label=f"n{n}")
            for n in range(4, 13)
        ]
        result = ScalingAnalyzer().analyze_qubit_scaling(runs)
        assert result.slope == pytest.approx(0.0, abs=1e-8)
        assert result.consistent_with_exponential_decay is False

    def test_growing_variance_is_not_consistent_with_decay(self):
        runs = exponential_runs(range(4, 13), a=-0.3)  # variance grows with n
        result = ScalingAnalyzer().analyze_qubit_scaling(runs)
        assert result.slope > 0
        assert result.consistent_with_exponential_decay is False

    def test_fewer_than_three_points_has_nan_r_squared(self):
        runs = exponential_runs([4, 8])
        result = ScalingAnalyzer().analyze_qubit_scaling(runs)
        assert result.n_points == 2
        assert math.isnan(result.r_squared)
        # Still flags a negative-slope result as "consistent" (2 points
        # always fit exactly; r_squared just isn't a meaningful gate yet).
        assert result.consistent_with_exponential_decay is True

    def test_single_run_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ScalingAnalyzer().analyze_qubit_scaling(exponential_runs([4]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ScalingAnalyzer().analyze_qubit_scaling([])

    def test_duplicate_qubit_counts_only_raises(self):
        runs = [
            ScalingObservation(n_qubits=4, gradient_variance=0.1),
            ScalingObservation(n_qubits=4, gradient_variance=0.2),
        ]
        with pytest.raises(ValueError, match="distinct"):
            ScalingAnalyzer().analyze_qubit_scaling(runs)

    def test_aggregate_mean_averages_repeated_qubit_counts(self):
        runs = [
            ScalingObservation(n_qubits=4, gradient_variance=0.1, label="a"),
            ScalingObservation(n_qubits=4, gradient_variance=0.3, label="b"),
            ScalingObservation(n_qubits=8, gradient_variance=0.01, label="c"),
        ]
        result = ScalingAnalyzer(aggregate="mean").analyze_qubit_scaling(runs)
        assert result.n_points == 2
        assert result.gradient_variance[0] == pytest.approx(0.2)  # mean(0.1, 0.3)

    def test_aggregate_none_keeps_every_point(self):
        runs = [
            ScalingObservation(n_qubits=4, gradient_variance=0.1),
            ScalingObservation(n_qubits=4, gradient_variance=0.3),
            ScalingObservation(n_qubits=8, gradient_variance=0.01),
        ]
        result = ScalingAnalyzer(aggregate="none").analyze_qubit_scaling(runs)
        assert result.n_points == 3

    def test_zero_variance_does_not_raise_or_produce_inf(self):
        runs = [
            ScalingObservation(n_qubits=4, gradient_variance=0.5),
            ScalingObservation(n_qubits=8, gradient_variance=0.0),
            ScalingObservation(n_qubits=12, gradient_variance=0.0),
        ]
        result = ScalingAnalyzer().analyze_qubit_scaling(runs)
        assert np.all(np.isfinite(result.log_variance))

    def test_x_values_and_labels_are_sorted_ascending(self):
        runs = [
            ScalingObservation(n_qubits=8, gradient_variance=0.01, label="big"),
            ScalingObservation(n_qubits=4, gradient_variance=0.5, label="small"),
        ]
        result = ScalingAnalyzer().analyze_qubit_scaling(runs)
        assert list(result.x_values) == [4.0, 8.0]
        assert result.labels == ["small", "big"]


class TestAnalyzeDepthScaling:
    def test_perfect_exponential_decay_is_detected(self):
        runs = [
            ScalingObservation(n_qubits=6, depth=d, gradient_variance=math.exp(-0.2 * d))
            for d in range(2, 20, 2)
        ]
        result = ScalingAnalyzer().analyze_depth_scaling(runs)
        assert result.x_label == "depth"
        assert result.slope == pytest.approx(-0.2, abs=1e-6)
        assert result.consistent_with_exponential_decay is True

    def test_missing_depth_raises(self):
        runs = [
            ScalingObservation(n_qubits=4, gradient_variance=0.1, depth=2),
            ScalingObservation(n_qubits=4, gradient_variance=0.2, depth=None),
        ]
        with pytest.raises(ValueError, match="depth"):
            ScalingAnalyzer().analyze_depth_scaling(runs)


class TestScalingAnalyzerConstruction:
    def test_invalid_r_squared_threshold_raises(self):
        with pytest.raises(ValueError, match="r_squared_threshold"):
            ScalingAnalyzer(r_squared_threshold=1.5)

    def test_non_positive_variance_floor_raises(self):
        with pytest.raises(ValueError, match="variance_floor"):
            ScalingAnalyzer(variance_floor=0.0)

    def test_invalid_aggregate_raises(self):
        with pytest.raises(ValueError, match="aggregate"):
            ScalingAnalyzer(aggregate="median")


class TestScalingObservationFromRunSummary:
    def _summary(self, n_qubits=5, depth=4, variance=0.02, run_id="run-1"):
        return {
            "run_id": run_id,
            "circuit": {"n_qubits": n_qubits, "depth": depth, "n_parameters": 12},
            "gradient": {"variance": variance, "norm_l2": 0.1},
        }

    def test_builds_observation_from_summary(self):
        obs = scaling_observation_from_run_summary(self._summary())
        assert obs.n_qubits == 5
        assert obs.depth == 4
        assert obs.gradient_variance == pytest.approx(0.02)
        assert obs.n_parameters == 12
        assert obs.label == "run-1"

    def test_explicit_label_overrides_run_id(self):
        obs = scaling_observation_from_run_summary(self._summary(), label="custom")
        assert obs.label == "custom"

    def test_missing_circuit_raises(self):
        summary = self._summary()
        summary["circuit"] = None
        with pytest.raises(ValueError, match="n_qubits"):
            scaling_observation_from_run_summary(summary)

    def test_missing_n_qubits_raises(self):
        summary = self._summary()
        summary["circuit"]["n_qubits"] = None
        with pytest.raises(ValueError, match="n_qubits"):
            scaling_observation_from_run_summary(summary)

    def test_missing_gradient_raises(self):
        summary = self._summary()
        summary["gradient"] = None
        with pytest.raises(ValueError, match="variance"):
            scaling_observation_from_run_summary(summary)

    def test_missing_variance_raises(self):
        summary = self._summary()
        summary["gradient"]["variance"] = None
        with pytest.raises(ValueError, match="variance"):
            scaling_observation_from_run_summary(summary)
