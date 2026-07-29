"""Unit tests for qml_observer.reporting.jsonl (Milestone 7, Issue #48)."""

import numpy as np
import pytest

from qml_observer.reporting.jsonl import (
    JSONL_SCHEMA_VERSION,
    JSONLWriter,
    circuit_metadata_to_dict,
    diagnosis_record,
    diagnosis_result_to_dict,
    event_record,
    gradient_snapshot_to_dict,
    optimizer_metadata_to_dict,
    read_jsonl,
    summary_record,
    training_event_to_dict,
)
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType
from qml_observer.schemas.gradient import summarize_gradient
from qml_observer.schemas.optimizer import OptimizerMetadata
from qml_observer.schemas.training import TrainingEvent
from qml_observer.statistics.confidence import attach_gradient_norm_ci


def _event(step=0, loss=1.0):
    return TrainingEvent(run_id="run-1", step=step, loss=loss, timestamp=100.0, wall_time=0.5)


def _diagnosis(issue=IssueType.HEALTHY, degraded=False):
    return DiagnosisResult(
        issue=issue,
        confidence=0.8,
        severity="info",
        evidence=["e1"],
        recommendations=["r1"],
        degraded=degraded,
        degraded_reason="boom" if degraded else None,
    )


class TestScalarSerialization:
    def test_training_event_to_dict(self):
        d = training_event_to_dict(_event(step=3, loss=0.5))
        assert d == {
            "run_id": "run-1",
            "step": 3,
            "loss": 0.5,
            "epoch": None,
            "timestamp": 100.0,
            "wall_time": 0.5,
        }

    def test_diagnosis_result_to_dict_uses_enum_value(self):
        d = diagnosis_result_to_dict(_diagnosis())
        assert d["issue"] == "healthy"
        assert d["degraded"] is False
        assert d["degraded_reason"] is None

    def test_circuit_metadata_to_dict_none(self):
        assert circuit_metadata_to_dict(None) is None

    def test_circuit_metadata_to_dict(self):
        circuit = CircuitMetadata(
            n_qubits=4, depth=3, n_parameters=8, n_gates=10, n_entangling_gates=2
        )
        d = circuit_metadata_to_dict(circuit)
        assert d["n_qubits"] == 4
        assert d["n_entangling_gates"] == 2

    def test_optimizer_metadata_to_dict_none(self):
        assert optimizer_metadata_to_dict(None) is None

    def test_optimizer_metadata_to_dict(self):
        optimizer = OptimizerMetadata(name="Adam", learning_rate=0.01, gradient_method="adjoint")
        d = optimizer_metadata_to_dict(optimizer)
        assert d == {"name": "Adam", "learning_rate": 0.01, "gradient_method": "adjoint"}

    def test_gradient_snapshot_to_dict_none(self):
        assert gradient_snapshot_to_dict(None) is None

    def test_gradient_snapshot_to_dict_excludes_values_by_default(self):
        snap = summarize_gradient(np.array([1.0, -2.0, 3.0]))
        d = gradient_snapshot_to_dict(snap)
        assert "values" not in d
        assert d["norm_l2"] == pytest.approx(snap.norm_l2)

    def test_gradient_snapshot_to_dict_includes_values_when_requested(self):
        snap = summarize_gradient(np.array([1.0, -2.0, 3.0]))
        d = gradient_snapshot_to_dict(snap, include_values=True)
        assert d["values"] == [1.0, -2.0, 3.0]

    def test_gradient_snapshot_to_dict_includes_ci_fields(self):
        """Issue #69/#108: CI fields must round-trip through JSONL, same as snr/uncertainty."""
        snap = attach_gradient_norm_ci(summarize_gradient(np.array([1.0, -2.0, 3.0])), shots=100)
        d = gradient_snapshot_to_dict(snap)
        assert d["ci_lower"] == pytest.approx(snap.ci_lower)
        assert d["ci_upper"] == pytest.approx(snap.ci_upper)
        assert d["ci_level"] == snap.ci_level
        assert d["ci_method"] == "shot-noise-analytic"

    def test_gradient_snapshot_to_dict_ci_fields_default_to_none(self):
        snap = summarize_gradient(np.array([1.0, -2.0, 3.0]))
        d = gradient_snapshot_to_dict(snap)
        assert d["ci_lower"] is None
        assert d["ci_upper"] is None
        assert d["ci_level"] is None
        assert d["ci_method"] is None


class TestRecordBuilders:
    def test_event_record_minimal(self):
        record = event_record(_event())
        assert record["type"] == "event"
        assert record["run_id"] == "run-1"
        assert record["schema_version"] == JSONL_SCHEMA_VERSION

    def test_event_record_with_extras(self):
        circuit = CircuitMetadata(n_qubits=2)
        optimizer = OptimizerMetadata(name="SPSA")
        record = event_record(_event(), circuit=circuit, optimizer=optimizer, shots=1000)
        assert record["circuit"]["n_qubits"] == 2
        assert record["optimizer"]["name"] == "SPSA"
        assert record["shots"] == 1000

    def test_diagnosis_record_includes_step(self):
        record = diagnosis_record(_diagnosis(), step=5)
        assert record["type"] == "diagnosis"
        assert record["step"] == 5
        assert record["issue"] == "healthy"
        assert record["schema_version"] == JSONL_SCHEMA_VERSION

    def test_diagnosis_record_step_optional(self):
        record = diagnosis_record(_diagnosis())
        assert "step" not in record

    def test_summary_record_wraps_type(self):
        record = summary_record({"run_id": "run-1", "steps": 10})
        assert record["type"] == "summary"
        assert record["run_id"] == "run-1"
        assert record["schema_version"] == JSONL_SCHEMA_VERSION


class TestJSONLWriterReader:
    def test_write_then_read_round_trip(self, tmp_path):
        path = tmp_path / "run.jsonl"
        with JSONLWriter(path) as writer:
            writer.write(event_record(_event(step=0)))
            writer.write(event_record(_event(step=1)))
            writer.write(diagnosis_record(_diagnosis(), step=1))

        records = list(read_jsonl(path))
        assert len(records) == 3
        assert records[0]["step"] == 0
        assert records[2]["type"] == "diagnosis"

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "run.jsonl"
        writer = JSONLWriter(path)
        writer.write(event_record(_event()))
        writer.close()
        assert path.exists()

    def test_append_mode_accumulates_across_writers(self, tmp_path):
        path = tmp_path / "run.jsonl"
        with JSONLWriter(path) as writer:
            writer.write(event_record(_event(step=0)))
        with JSONLWriter(path) as writer:
            writer.write(event_record(_event(step=1)))

        records = list(read_jsonl(path))
        assert len(records) == 2

    def test_ndarray_default_serialization(self, tmp_path):
        path = tmp_path / "run.jsonl"
        snap = summarize_gradient(np.array([1.0, 2.0]))
        with JSONLWriter(path) as writer:
            writer.write(event_record(_event(), gradient=snap, include_gradient_values=True))

        records = list(read_jsonl(path))
        assert records[0]["gradient"]["values"] == [1.0, 2.0]

    def test_double_close_is_safe(self, tmp_path):
        path = tmp_path / "run.jsonl"
        writer = JSONLWriter(path)
        writer.close()
        writer.close()  # must not raise

    def test_read_jsonl_skips_blank_lines(self, tmp_path):
        path = tmp_path / "run.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
        records = list(read_jsonl(path))
        assert records == [{"a": 1}, {"a": 2}]

    def test_read_jsonl_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(read_jsonl(tmp_path / "missing.jsonl"))
