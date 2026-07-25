"""Unit tests for qml_observer.adapters.qiskit.adapter.QiskitAdapter.

Milestone 8, Issues #58-#60. Skipped entirely if the optional `qiskit`
dependency isn't installed (`pip install qml-observer[qiskit]`).
"""

import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit.circuit.library import efficient_su2  # noqa: E402

from qml_observer.adapters.qiskit.adapter import QiskitAdapter  # noqa: E402
from qml_observer.core.monitor import QMLMonitor  # noqa: E402
from qml_observer.schemas.circuit import CircuitMetadata  # noqa: E402
from qml_observer.schemas.diagnosis import DiagnosisResult  # noqa: E402
from qml_observer.schemas.optimizer import OptimizerMetadata  # noqa: E402


def _ansatz(n_qubits=3, reps=1):
    return efficient_su2(n_qubits, reps=reps)


class _FakeSettingsOptimizer:
    """Stand-in for a qiskit_algorithms/qiskit-machine-learning `Optimizer`
    exposing a `.settings` dict, without requiring `qiskit_algorithms` as a
    test dependency."""

    def __init__(self, settings):
        self.settings = settings


def _fake_optimizer(name, **settings):
    """Build a fake optimizer whose *class name* is `name` (since
    `normalize_optimizer_metadata` keys off `type(optimizer).__name__`)."""
    cls = type(name, (_FakeSettingsOptimizer,), {})
    return cls(settings)


class TestConstruction:
    def test_wraps_monitor(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        assert adapter.monitor is monitor
        assert adapter.attached is False

    def test_rejects_non_monitor(self):
        with pytest.raises(TypeError):
            QiskitAdapter("not-a-monitor")

    def test_attaches_circuit_at_construction(self):
        adapter = QiskitAdapter(QMLMonitor(), _ansatz())
        assert adapter.attached is True


class TestAttachDetach:
    def test_attach_returns_self(self):
        adapter = QiskitAdapter(QMLMonitor())
        circuit = _ansatz()
        assert adapter.attach(circuit) is adapter
        assert adapter.attached is True

    def test_attach_rejects_unrecognized_object(self):
        adapter = QiskitAdapter(QMLMonitor())
        with pytest.raises(TypeError):
            adapter.attach(object())

    def test_attach_accepts_object_exposing_circuit_attribute(self):
        class FakeTrainer:
            def __init__(self, circuit):
                self.circuit = circuit

        adapter = QiskitAdapter(QMLMonitor())
        adapter.attach(FakeTrainer(_ansatz()))
        assert adapter.attached is True

    def test_attach_accepts_object_exposing_ansatz_attribute(self):
        class FakeTrainer:
            def __init__(self, ansatz):
                self.ansatz = ansatz

        adapter = QiskitAdapter(QMLMonitor())
        adapter.attach(FakeTrainer(_ansatz()))
        assert adapter.attached is True

    def test_detach_clears_circuit(self):
        adapter = QiskitAdapter(QMLMonitor(), _ansatz())
        adapter.detach()
        assert adapter.attached is False


class TestRecordStep:
    def test_record_step_returns_diagnosis(self):
        adapter = QiskitAdapter(QMLMonitor())
        diagnosis = adapter.record_step(0, loss=1.0)
        assert isinstance(diagnosis, DiagnosisResult)

    def test_record_step_forwards_loss_and_gradients(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        adapter.record_step(0, loss=0.5, gradients=[0.1, 0.2])

        obs = monitor.state.latest_observation
        assert obs.training_event.loss == 0.5
        assert obs.gradient is not None

    def test_record_step_without_attached_circuit_has_no_circuit_metadata(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        adapter.record_step(0, loss=1.0, parameters=[0.1, 0.2, 0.3])
        assert monitor.state.latest_observation.circuit is None

    def test_record_multiple_steps_advances_monitor(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor, _ansatz())
        adapter.record_step(0, loss=1.0)
        adapter.record_step(1, loss=0.5)
        assert monitor.state.step_count == 2

    def test_fail_open_preserved_through_adapter(self):
        adapter = QiskitAdapter(QMLMonitor())
        diagnosis = adapter.record_step(0, loss=1.0, gradients=[])
        assert diagnosis.degraded is True

    def test_default_shots_used_when_not_specified(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor, shots=4096)
        adapter.record_step(0, loss=0.5)
        assert monitor.state.latest_observation.shots == 4096

    def test_explicit_shots_override_default(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor, shots=4096)
        adapter.record_step(0, loss=0.5, shots=100)
        assert monitor.state.latest_observation.shots == 100


class TestRecordGradient:
    """`record_gradient()` caches a gradient for the next recorded step."""

    def test_cached_gradient_consumed_by_next_record_step(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        adapter.record_gradient([0.01, 0.02])
        adapter.record_step(0, loss=0.5)
        assert monitor.state.latest_observation.gradient is not None

    def test_cached_gradient_cleared_after_use(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        adapter.record_gradient([0.01, 0.02])
        adapter.record_step(0, loss=0.5)
        adapter.record_step(1, loss=0.4)
        assert monitor.state.latest_observation.gradient is None

    def test_explicit_gradient_argument_overrides_cache(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        adapter.record_gradient([0.01, 0.02])
        adapter.record_step(0, loss=0.5, gradients=[9.0, 9.0, 9.0])
        assert monitor.state.latest_observation.gradient.values.tolist() == [9.0, 9.0, 9.0]


class TestCallbackIntegration:
    """Issue #59: normalizing across the Qiskit callback shapes in the wild."""

    def test_scipy_style_single_argument(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        diagnosis = adapter.callback([0.1, 0.2])
        assert isinstance(diagnosis, DiagnosisResult)
        assert monitor.state.latest_observation.training_event.loss is None

    def test_vqc_style_weights_and_loss(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        adapter.callback([0.1, 0.2], 0.75)
        assert monitor.state.latest_observation.training_event.loss == 0.75

    def test_blueprint_style_iteration_parameters_loss(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        adapter.callback(7, [0.1, 0.2], 0.5)
        assert monitor.state.latest_observation.training_event.step == 7

    def test_spsa_style_five_arguments(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        adapter.callback(3, [0.1, 0.2], 0.6, 0.01, True)
        assert monitor.state.latest_observation.training_event.loss == 0.6

    def test_iteration_auto_increments_across_two_arg_calls(self):
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        adapter.callback([0.1], 0.5)
        adapter.callback([0.2], 0.4)
        assert monitor.state.step_count == 2
        assert monitor.state.latest_observation.training_event.step == 1

    def test_unsupported_argument_count_raises(self):
        adapter = QiskitAdapter(QMLMonitor())
        with pytest.raises(TypeError, match="unsupported number"):
            adapter.callback(1, 2, 3, 4)

    def test_callback_usable_as_bare_function_reference(self):
        """Must work when passed directly as `callback=adapter.callback`,
        i.e. as a bound method reference, not just called inline."""
        monitor = QMLMonitor()
        adapter = QiskitAdapter(monitor)
        cb = adapter.callback
        cb([0.1, 0.2], 0.9)
        assert monitor.state.latest_observation.training_event.loss == 0.9


class TestExtractCircuitMetadata:
    def test_extracts_qubits_gates_params_and_entangling_count(self):
        adapter = QiskitAdapter(QMLMonitor())
        circuit = _ansatz(n_qubits=3, reps=1)

        meta = adapter.extract_circuit_metadata(circuit)

        assert isinstance(meta, CircuitMetadata)
        assert meta.n_qubits == 3
        assert meta.n_parameters == circuit.num_parameters
        assert meta.n_gates == circuit.size()
        assert meta.n_entangling_gates is not None
        assert meta.n_entangling_gates > 0

    def test_extracts_depth(self):
        adapter = QiskitAdapter(QMLMonitor())
        circuit = _ansatz(n_qubits=3, reps=2)

        meta = adapter.extract_circuit_metadata(circuit)

        assert meta.depth is not None
        assert meta.depth > 0

    def test_passes_through_ansatz_name_and_initialization(self):
        adapter = QiskitAdapter(QMLMonitor())
        circuit = _ansatz()

        meta = adapter.extract_circuit_metadata(
            circuit, ansatz_name="EfficientSU2", initialization="random_uniform"
        )

        assert meta.ansatz_name == "EfficientSU2"
        assert meta.initialization == "random_uniform"

    def test_handles_malformed_circuit_gracefully(self):
        """An object missing most attributes should degrade to an (almost)
        empty CircuitMetadata rather than raising."""
        adapter = QiskitAdapter(QMLMonitor())

        class FakeCircuit:
            pass

        meta = adapter.extract_circuit_metadata(FakeCircuit())
        assert isinstance(meta, CircuitMetadata)
        assert meta.n_qubits is None
        assert meta.n_gates is None

    def test_record_step_populates_circuit_metadata_when_attached(self):
        monitor = QMLMonitor()
        circuit = _ansatz(n_qubits=2, reps=1)
        adapter = QiskitAdapter(monitor, circuit)

        adapter.record_step(0, loss=0.5)

        circuit_meta = monitor.state.latest_observation.circuit
        assert circuit_meta is not None
        assert circuit_meta.n_qubits == 2


class TestNormalizeOptimizerMetadata:
    """Issue #60."""

    def test_returns_none_when_nothing_known(self):
        assert QiskitAdapter.normalize_optimizer_metadata() is None

    def test_explicit_name_only(self):
        meta = QiskitAdapter.normalize_optimizer_metadata(name="COBYLA")
        assert isinstance(meta, OptimizerMetadata)
        assert meta.name == "COBYLA"
        assert meta.learning_rate is None

    def test_infers_name_from_optimizer_object(self):
        optimizer = _fake_optimizer("SPSA", learning_rate=0.02, perturbation=0.01)
        meta = QiskitAdapter.normalize_optimizer_metadata(optimizer)
        assert meta.name == "SPSA"

    def test_infers_learning_rate_key_variants(self):
        spsa = _fake_optimizer("SPSA", learning_rate=0.02)
        adam = _fake_optimizer("ADAM", lr=0.05)
        cobyla = _fake_optimizer("COBYLA", rhobeg=1.0)

        assert QiskitAdapter.normalize_optimizer_metadata(spsa).learning_rate == 0.02
        assert QiskitAdapter.normalize_optimizer_metadata(adam).learning_rate == 0.05
        assert QiskitAdapter.normalize_optimizer_metadata(cobyla).learning_rate is None

    def test_infers_gradient_method_for_known_optimizers(self):
        spsa = _fake_optimizer("SPSA", learning_rate=0.02)
        cobyla = _fake_optimizer("COBYLA")

        assert (
            QiskitAdapter.normalize_optimizer_metadata(spsa).gradient_method == "spsa-approximation"
        )
        assert QiskitAdapter.normalize_optimizer_metadata(cobyla).gradient_method == "gradient-free"

    def test_unrecognized_optimizer_name_leaves_gradient_method_none(self):
        mystery = _fake_optimizer("SomeFutureOptimizer", learning_rate=0.1)
        meta = QiskitAdapter.normalize_optimizer_metadata(mystery)
        assert meta.gradient_method is None

    def test_explicit_kwargs_take_precedence_over_optimizer_object(self):
        spsa = _fake_optimizer("SPSA", learning_rate=0.02)
        meta = QiskitAdapter.normalize_optimizer_metadata(
            spsa, name="CustomSPSA", learning_rate=0.5, gradient_method="custom-method"
        )
        assert meta.name == "CustomSPSA"
        assert meta.learning_rate == 0.5
        assert meta.gradient_method == "custom-method"

    def test_boolean_settings_value_not_mistaken_for_learning_rate(self):
        """`bool` is a subclass of `int` in Python; a stray boolean flag
        under a learning-rate-like key must not be coerced into a rate."""
        weird = _fake_optimizer("Weird", learning_rate=True)
        meta = QiskitAdapter.normalize_optimizer_metadata(weird)
        assert meta.learning_rate is None

    def test_optimizer_object_without_settings_attribute_leaves_rate_none(self):
        """An optimizer-like object with no `.settings` dict at all (e.g. a
        raw `scipy.optimize.minimize`-compatible `Minimizer` callable) must
        not raise -- just fall back to name-only metadata."""

        class BareMinimizer:
            pass

        meta = QiskitAdapter.normalize_optimizer_metadata(BareMinimizer())
        assert meta.name == "BareMinimizer"
        assert meta.learning_rate is None

    def test_record_step_uses_adapter_level_optimizer_config(self):
        monitor = QMLMonitor()
        spsa = _fake_optimizer("SPSA", learning_rate=0.03)
        adapter = QiskitAdapter(monitor, optimizer=spsa)

        adapter.record_step(0, loss=0.5)

        optimizer_meta = monitor.state.latest_observation.optimizer
        assert optimizer_meta.name == "SPSA"
        assert optimizer_meta.learning_rate == 0.03
        assert optimizer_meta.gradient_method == "spsa-approximation"


class TestImportGuard:
    def test_clear_error_when_qiskit_missing(self, monkeypatch):
        import qml_observer.adapters.qiskit.adapter as adapter_module

        monkeypatch.setattr(adapter_module, "qiskit", None)
        monkeypatch.setattr(adapter_module, "_IMPORT_ERROR", ImportError("boom"))
        with pytest.raises(ImportError, match="qiskit"):
            adapter_module.QiskitAdapter(QMLMonitor())
