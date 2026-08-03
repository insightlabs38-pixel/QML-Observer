"""Unit tests for qml_observer.adapters.autograd.AutogradAdapter.

Milestone 14, Issue #100 ("Generic autograd adapter"). Uses only numpy
(no torch/jax dependency) plus a minimal fake tensor class to exercise the
`.detach()`/`.cpu()`/`.numpy()` duck-typing path without an optional
dependency.
"""

import numpy as np
import pytest

from qml_observer.adapters.autograd import AutogradAdapter, to_numpy
from qml_observer.core.monitor import QMLMonitor
from qml_observer.schemas.diagnosis import DiagnosisResult


class _FakeTensor:
    """Minimal torch.Tensor-like duck type: .detach().cpu().numpy()."""

    def __init__(self, array, detach_calls=None):
        self._array = np.asarray(array)
        self._detach_calls = detach_calls if detach_calls is not None else []

    def detach(self):
        self._detach_calls.append("detach")
        return self

    def cpu(self):
        self._detach_calls.append("cpu")
        return self

    def numpy(self):
        self._detach_calls.append("numpy")
        return self._array


class TestToNumpy:
    def test_none_passthrough(self):
        assert to_numpy(None) is None

    def test_ndarray_passthrough(self):
        arr = np.array([1.0, 2.0])
        assert to_numpy(arr) is arr

    def test_tensor_like_duck_typing(self):
        calls = []
        tensor = _FakeTensor([1.0, 2.0, 3.0], detach_calls=calls)
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        assert calls == ["detach", "cpu", "numpy"]

    def test_plain_list_via_asarray_fallback(self):
        result = to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_scalar_via_asarray_fallback(self):
        result = to_numpy(3.5)
        assert isinstance(result, np.ndarray)
        assert float(result) == 3.5


class TestConstruction:
    def test_wraps_monitor(self):
        monitor = QMLMonitor()
        adapter = AutogradAdapter(monitor)
        assert adapter.monitor is monitor

    def test_rejects_non_monitor(self):
        with pytest.raises(TypeError):
            AutogradAdapter("not-a-monitor")


class TestRecordStep:
    def test_returns_diagnosis(self):
        adapter = AutogradAdapter(QMLMonitor())
        diagnosis = adapter.record_step(0, loss=1.0)
        assert isinstance(diagnosis, DiagnosisResult)

    def test_converts_tensor_like_loss_and_gradients(self):
        monitor = QMLMonitor()
        adapter = AutogradAdapter(monitor)
        loss = _FakeTensor(0.75)
        gradients = _FakeTensor([0.1, -0.2, 0.3])

        adapter.record_step(0, loss=loss, gradients=gradients)

        obs = monitor.state.latest_observation
        assert obs.training_event.loss == pytest.approx(0.75)
        assert obs.gradient is not None
        np.testing.assert_allclose(obs.gradient.values, [0.1, -0.2, 0.3])

    def test_infers_n_parameters_from_parameters_array(self):
        monitor = QMLMonitor()
        adapter = AutogradAdapter(monitor)
        adapter.record_step(0, loss=1.0, parameters=np.zeros(12))
        obs = monitor.state.latest_observation
        assert obs.circuit is not None
        assert obs.circuit.n_parameters == 12

    def test_optimizer_metadata_populated_when_given(self):
        monitor = QMLMonitor()
        adapter = AutogradAdapter(monitor, optimizer_name="Adam", learning_rate=0.01)
        adapter.record_step(0, loss=1.0, gradient_method="backprop")
        obs = monitor.state.latest_observation
        assert obs.optimizer is not None
        assert obs.optimizer.name == "Adam"
        assert obs.optimizer.learning_rate == 0.01
        assert obs.optimizer.gradient_method == "backprop"

    def test_no_optimizer_metadata_when_nothing_given(self):
        monitor = QMLMonitor()
        adapter = AutogradAdapter(monitor)
        adapter.record_step(0, loss=1.0)
        obs = monitor.state.latest_observation
        assert obs.optimizer is None

    def test_fail_open_preserved_through_adapter(self):
        adapter = AutogradAdapter(QMLMonitor())
        diagnosis = adapter.record_step(0, loss=1.0, gradients=np.array([]))
        assert diagnosis.degraded is True

    def test_multi_element_loss_degrades_to_none_scalar(self):
        monitor = QMLMonitor()
        adapter = AutogradAdapter(monitor)
        diagnosis = adapter.record_step(0, loss=np.array([1.0, 2.0]))
        assert diagnosis is not None
        obs = monitor.state.latest_observation
        assert obs.training_event.loss is None
