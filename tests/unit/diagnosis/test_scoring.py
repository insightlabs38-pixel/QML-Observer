"""Unit tests for qml_observer.diagnosis.scoring.combine_detector_results."""

import pytest

from qml_observer.detectors.base import DetectorResult
from qml_observer.diagnosis.scoring import combine_detector_results
from qml_observer.schemas.diagnosis import IssueType


def _result(name, triggered, confidence, evidence=None, recommendations=None):
    return DetectorResult(
        detector_name=name,
        triggered=triggered,
        confidence=confidence,
        evidence=evidence or [],
        recommendations=recommendations or [],
    )


class TestEmptyResults:
    def test_no_results_is_insufficient_evidence(self):
        result = combine_detector_results([])
        assert result.issue == IssueType.INSUFFICIENT_EVIDENCE
        assert result.confidence == 0.0


class TestNoTrigger:
    def test_all_no_data_is_insufficient_evidence(self):
        results = [_result("barren_plateau", False, 0.0), _result("stagnation", False, 0.0)]
        result = combine_detector_results(results)
        assert result.issue == IssueType.INSUFFICIENT_EVIDENCE

    def test_some_evidence_but_no_trigger_is_healthy(self):
        results = [
            _result("barren_plateau", False, 0.2, evidence=["some evidence"]),
            _result("stagnation", False, 0.1, evidence=["some evidence"]),
        ]
        result = combine_detector_results(results)
        assert result.issue == IssueType.HEALTHY
        # healthy confidence reflects the closest near-miss (1 - 0.2)
        assert result.confidence == pytest.approx(0.8)


class TestSingleDetectorTrigger:
    def test_single_trigger_confidence_passthrough(self):
        """A single contributing detector's confidence is unchanged by noisy-OR."""
        results = [_result("barren_plateau", True, 0.73, evidence=["e"], recommendations=["r"])]
        result = combine_detector_results(results)
        assert result.issue == IssueType.POSSIBLE_BARREN_PLATEAU
        assert result.confidence == pytest.approx(0.73)
        assert result.recommendations == ["r"]

    def test_unrecognized_detector_name_does_not_steer_diagnosis(self):
        results = [_result("some_future_detector", True, 0.9)]
        result = combine_detector_results(results)
        assert result.issue == IssueType.INSUFFICIENT_EVIDENCE
        assert result.confidence == 0.0


class TestWeightedNoisyOr:
    def test_agreement_between_two_detectors_on_same_issue_increases_confidence(self):
        # Two independent "barren_plateau"-mapped detectors, each 0.5 confidence.
        results = [
            _result("barren_plateau", True, 0.5),
            _result("barren_plateau", True, 0.5),
        ]
        result = combine_detector_results(results)
        # noisy-OR: 1 - (1-0.5)*(1-0.5) = 0.75
        assert result.confidence == pytest.approx(0.75)

    def test_zero_weight_silences_a_detector(self):
        results = [_result("barren_plateau", True, 0.9)]
        result = combine_detector_results(results, weights={"barren_plateau": 0.0})
        assert result.confidence == 0.0
        assert result.issue == IssueType.POSSIBLE_BARREN_PLATEAU

    def test_custom_weight_scales_contribution(self):
        results = [_result("barren_plateau", True, 0.8)]
        result = combine_detector_results(results, weights={"barren_plateau": 0.5})
        assert result.confidence == pytest.approx(0.4)


class TestConvergedPriority:
    def test_converged_wins_even_with_lower_confidence(self):
        results = [
            _result("barren_plateau", True, 0.95),
            _result("convergence", True, 0.6),
        ]
        result = combine_detector_results(results)
        assert result.issue == IssueType.CONVERGED
        assert result.confidence == pytest.approx(0.6)
        assert result.severity == "info"


class TestNoiseDominatedPriority:
    """Milestone 9, Issue #67: a low-SNR reading must pull the headline
    diagnosis toward NOISE_DOMINATED and away from POSSIBLE_BARREN_PLATEAU
    -- a plateau and a noisy-but-healthy run must never be conflated.
    """

    def test_noise_result_maps_to_noise_dominated(self):
        results = [_result("noise", True, 0.8, evidence=["e"], recommendations=["r"])]
        result = combine_detector_results(results)
        assert result.issue == IssueType.NOISE_DOMINATED
        assert result.confidence == pytest.approx(0.8)

    def test_noise_wins_over_barren_plateau_even_with_lower_confidence(self):
        results = [
            _result("barren_plateau", True, 0.95),
            _result("noise", True, 0.4),
        ]
        result = combine_detector_results(results)
        assert result.issue == IssueType.NOISE_DOMINATED
        assert result.confidence == pytest.approx(0.4)

    def test_converged_still_wins_over_noise(self):
        results = [
            _result("noise", True, 0.9),
            _result("convergence", True, 0.5),
        ]
        result = combine_detector_results(results)
        assert result.issue == IssueType.CONVERGED
        assert result.confidence == pytest.approx(0.5)

    def test_barren_plateau_alone_is_unaffected(self):
        """Sanity check: without a noise signal, plateau detection is untouched."""
        results = [_result("barren_plateau", True, 0.9)]
        result = combine_detector_results(results)
        assert result.issue == IssueType.POSSIBLE_BARREN_PLATEAU


class TestSeverity:
    def test_high_confidence_non_converged_is_critical(self):
        results = [_result("barren_plateau", True, 0.9)]
        result = combine_detector_results(results)
        assert result.severity == "critical"

    def test_low_confidence_non_converged_is_warning(self):
        results = [_result("stagnation", True, 0.5)]
        result = combine_detector_results(results)
        assert result.severity == "warning"

    def test_converged_is_always_info(self):
        results = [_result("convergence", True, 0.99)]
        result = combine_detector_results(results)
        assert result.severity == "info"


class TestEvidenceAttribution:
    def test_evidence_is_prefixed_with_detector_name(self):
        results = [_result("barren_plateau", True, 0.9, evidence=["gradient collapsed"])]
        result = combine_detector_results(results)
        assert result.evidence == ["[barren_plateau] gradient collapsed"]

    def test_recommendations_deduplicated_preserving_order(self):
        results = [
            _result("barren_plateau", True, 0.9, recommendations=["stop and inspect"]),
            _result("stagnation", True, 0.9, recommendations=["stop and inspect", "check lr"]),
        ]
        result = combine_detector_results(results)
        assert result.recommendations == ["stop and inspect", "check lr"]
