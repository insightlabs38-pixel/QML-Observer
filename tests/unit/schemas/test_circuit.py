"""Unit tests for qml_observer.schemas.circuit.CircuitMetadata."""

import pytest

from qml_observer.schemas.circuit import CircuitMetadata


class TestConstruction:
    def test_all_defaults_are_none(self):
        meta = CircuitMetadata()
        assert meta.n_qubits is None
        assert meta.depth is None
        assert meta.n_parameters is None
        assert meta.n_gates is None
        assert meta.n_entangling_gates is None
        assert meta.ansatz_name is None
        assert meta.initialization is None

    def test_full_construction(self):
        meta = CircuitMetadata(
            n_qubits=8,
            depth=20,
            n_parameters=160,
            n_gates=200,
            n_entangling_gates=56,
            ansatz_name="StronglyEntanglingLayers",
            initialization="random_uniform",
        )
        assert meta.n_qubits == 8
        assert meta.ansatz_name == "StronglyEntanglingLayers"

    def test_entangling_gates_equal_to_total_gates_is_allowed(self):
        CircuitMetadata(n_gates=10, n_entangling_gates=10)


class TestValidation:
    @pytest.mark.parametrize(
        "field", ["n_qubits", "depth", "n_parameters", "n_gates", "n_entangling_gates"]
    )
    def test_negative_int_fields_raise(self, field):
        with pytest.raises(ValueError, match=field):
            CircuitMetadata(**{field: -1})

    def test_entangling_exceeding_total_gates_raises(self):
        with pytest.raises(ValueError, match="n_entangling_gates"):
            CircuitMetadata(n_gates=5, n_entangling_gates=10)

    def test_empty_ansatz_name_raises(self):
        with pytest.raises(ValueError, match="ansatz_name"):
            CircuitMetadata(ansatz_name="")

    def test_empty_initialization_raises(self):
        with pytest.raises(ValueError, match="initialization"):
            CircuitMetadata(initialization="   ")

    def test_non_int_field_raises(self):
        with pytest.raises(TypeError):
            CircuitMetadata(n_qubits=4.5)  # type: ignore[arg-type]
