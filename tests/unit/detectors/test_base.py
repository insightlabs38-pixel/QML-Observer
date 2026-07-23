"""Unit tests for qml_observer.detectors.base."""

import pytest

from qml_observer.detectors.base import BaseDetector, DetectorResult


class TestDetectorResult:
    def test_valid_construction(self):
        r = DetectorResult(
            detector_name="dummy",
            triggered=True,
            confidence=0.5,
            evidence=["a"],
            recommendations=["b"],
        )
        assert r.detector_name == "dummy"
        assert r.triggered is True
        assert r.confidence == 0.5

    def test_empty_detector_name_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            DetectorResult("", True, 0.5, [], [])

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            DetectorResult("dummy", True, 1.5, [], [])
        with pytest.raises(ValueError):
            DetectorResult("dummy", True, -0.1, [], [])

    def test_non_bool_triggered_raises(self):
        with pytest.raises(TypeError):
            DetectorResult("dummy", "yes", 0.5, [], [])

    def test_non_string_evidence_raises(self):
        with pytest.raises(TypeError):
            DetectorResult("dummy", True, 0.5, [1, 2], [])


class TestBaseDetectorIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseDetector()  # type: ignore[abstract]

    def test_subclass_must_implement_all_methods(self):
        class Incomplete(BaseDetector):
            def update(self, event, state):
                pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self):
        class Complete(BaseDetector):
            name = "complete"

            def update(self, event, state):
                pass

            def diagnose(self):
                return DetectorResult("complete", False, 0.0, [], [])

            def reset(self):
                pass

        d = Complete()
        d.update(None, None)
        assert d.diagnose().detector_name == "complete"
        d.reset()
