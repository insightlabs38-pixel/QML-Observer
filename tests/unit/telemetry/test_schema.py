"""Unit tests for qml_observer.telemetry.schema."""

from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.telemetry.schema import (
    TelemetryRecord,
    bucket_qubit_count,
    build_telemetry_record,
    extract_detector_thresholds,
)


class TestBucketQubitCount:
    def test_none_stays_none(self):
        assert bucket_qubit_count(None) is None

    def test_small_count(self):
        assert bucket_qubit_count(3) == "1-5"

    def test_boundary_is_inclusive(self):
        assert bucket_qubit_count(5) == "1-5"
        assert bucket_qubit_count(6) == "6-10"

    def test_large_count_bucketed(self):
        assert bucket_qubit_count(37) == "21-50"

    def test_very_large_count(self):
        assert bucket_qubit_count(500) == "101+"

    def test_never_returns_exact_count_as_string(self):
        # The whole point: the exact number should never appear verbatim.
        result = bucket_qubit_count(42)
        assert "42" not in result


class TestExtractDetectorThresholds:
    def test_extracts_known_threshold_attrs(self):
        detector = BarrenPlateauDetector(
            gradient_threshold=1e-8, variance_threshold=1e-16, patience=50
        )
        thresholds = extract_detector_thresholds(detector)
        assert thresholds["BarrenPlateauDetector.gradient_threshold"] == 1e-8
        assert thresholds["BarrenPlateauDetector.variance_threshold"] == 1e-16
        assert thresholds["BarrenPlateauDetector.patience"] == 50.0

    def test_strips_leading_underscore_from_private_attrs(self):
        # Detectors store constructor args as `self._gradient_threshold`
        # etc; the exported key should be the clean, no-underscore name.
        detector = BarrenPlateauDetector(gradient_threshold=1e-8)
        thresholds = extract_detector_thresholds(detector)
        assert "BarrenPlateauDetector.gradient_threshold" in thresholds
        assert not any(key.split(".", 1)[1].startswith("_") for key in thresholds)

    def test_ignores_non_numeric_attrs(self):
        class Fake:
            def __init__(self):
                self.gradient_threshold = 1e-8
                self.name = "not-a-number"
                self.enabled = True  # bool must be excluded despite isinstance(bool, int)

        thresholds = extract_detector_thresholds(Fake())
        assert "Fake.gradient_threshold" in thresholds
        assert "Fake.name" not in thresholds
        assert "Fake.enabled" not in thresholds


class TestBuildTelemetryRecord:
    def test_basic_record(self):
        record = build_telemetry_record(
            detector_names=["BarrenPlateauDetector", "StagnationDetector"],
            thresholds={"BarrenPlateauDetector.gradient_threshold": 5e-6},
            issue="possible_barren_plateau",
            confidence=0.91,
            framework="pennylane",
            n_qubits=12,
            detection_latency_steps=240,
        )
        assert isinstance(record, TelemetryRecord)
        assert record.schema_version == "1"
        assert record.detector_names == ["BarrenPlateauDetector", "StagnationDetector"]
        assert record.qubit_bucket == "11-20"
        assert record.framework == "pennylane"
        assert record.detection_latency_steps == 240

    def test_detector_names_are_sorted(self):
        record = build_telemetry_record(
            detector_names=["StagnationDetector", "BarrenPlateauDetector"],
            thresholds={},
            issue="healthy",
            confidence=1.0,
        )
        assert record.detector_names == ["BarrenPlateauDetector", "StagnationDetector"]

    def test_to_dict_never_contains_forbidden_keys(self):
        record = build_telemetry_record(
            detector_names=["BarrenPlateauDetector"],
            thresholds={},
            issue="healthy",
            confidence=1.0,
        )
        forbidden = {
            "gradients",
            "loss",
            "parameters",
            "run_id",
            "file_path",
            "hostname",
            "ansatz",
        }
        assert forbidden.isdisjoint(record.to_dict().keys())

    def test_package_version_matches_installed_package(self):
        from qml_observer import __version__

        record = build_telemetry_record(
            detector_names=[], thresholds={}, issue="healthy", confidence=1.0
        )
        assert record.package_version == __version__
