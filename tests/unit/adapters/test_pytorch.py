"""Unit tests for qml_observer.adapters.pytorch.adapter.PyTorchAdapter.

Milestone 14, Issue #98 ("PyTorch hybrid-workflow integration"). Skipped
entirely if the optional `torch` dependency isn't installed (`pip install
qml-observer[torch]`).
"""

import pytest

torch = pytest.importorskip("torch")

from qml_observer.adapters.pytorch.adapter import PyTorchAdapter  # noqa: E402
from qml_observer.core.monitor import QMLMonitor  # noqa: E402
from qml_observer.schemas.diagnosis import DiagnosisResult  # noqa: E402


def _tiny_model():
    return torch.nn.Linear(3, 1)


class TestConstruction:
    def test_wraps_monitor(self):
        monitor = QMLMonitor()
        adapter = PyTorchAdapter(monitor)
        assert adapter.monitor is monitor
        assert adapter.attached is False

    def test_rejects_non_monitor(self):
        with pytest.raises(TypeError):
            PyTorchAdapter("not-a-monitor")

    def test_attaches_module_at_construction(self):
        model = _tiny_model()
        adapter = PyTorchAdapter(QMLMonitor(), module=model)
        assert adapter.attached is True

    def test_attaches_optimizer_at_construction(self):
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        adapter = PyTorchAdapter(QMLMonitor(), optimizer=optimizer)
        assert adapter.attached is True


class TestAttachDetach:
    def test_attach_returns_self(self):
        adapter = PyTorchAdapter(QMLMonitor())
        assert adapter.attach(module=_tiny_model()) is adapter

    def test_attach_rejects_non_module(self):
        adapter = PyTorchAdapter(QMLMonitor())
        with pytest.raises(TypeError):
            adapter.attach(module=object())

    def test_attach_rejects_non_optimizer(self):
        adapter = PyTorchAdapter(QMLMonitor())
        with pytest.raises(TypeError):
            adapter.attach(optimizer=object())

    def test_detach_clears_module_and_optimizer(self):
        model = _tiny_model()
        adapter = PyTorchAdapter(QMLMonitor(), module=model)
        adapter.detach()
        assert adapter.attached is False


class TestRecordStep:
    def test_record_step_returns_diagnosis(self):
        adapter = PyTorchAdapter(QMLMonitor())
        diagnosis = adapter.record_step(0, loss=1.0)
        assert isinstance(diagnosis, DiagnosisResult)

    def test_auto_collects_gradients_from_attached_module(self):
        torch.manual_seed(0)
        model = _tiny_model()
        monitor = QMLMonitor()
        adapter = PyTorchAdapter(monitor, module=model)

        x = torch.randn(4, 3)
        y = torch.randn(4, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()

        adapter.record_step(0, loss)

        obs = monitor.state.latest_observation
        assert obs.gradient is not None
        n_params = sum(p.numel() for p in model.parameters())
        assert obs.gradient.values.shape == (n_params,)
        assert obs.circuit is not None
        assert obs.circuit.n_parameters == n_params

    def test_no_gradients_when_backward_not_called(self):
        model = _tiny_model()
        monitor = QMLMonitor()
        adapter = PyTorchAdapter(monitor, module=model)
        adapter.record_step(0, loss=1.0)
        obs = monitor.state.latest_observation
        assert obs.gradient is None

    def test_optimizer_metadata_from_attached_optimizer(self):
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        monitor = QMLMonitor()
        adapter = PyTorchAdapter(monitor, module=model, optimizer=optimizer)
        adapter.record_step(0, loss=1.0)
        obs = monitor.state.latest_observation
        assert obs.optimizer is not None
        assert obs.optimizer.name == "Adam"
        assert obs.optimizer.learning_rate == pytest.approx(0.03)
        assert obs.optimizer.gradient_method == "backprop"

    def test_explicit_gradients_override_module_collection(self):
        import numpy as np

        model = _tiny_model()
        monitor = QMLMonitor()
        adapter = PyTorchAdapter(monitor, module=model)
        adapter.record_step(0, loss=1.0, gradients=np.array([0.5, 0.5]))
        obs = monitor.state.latest_observation
        assert obs.gradient.values.shape == (2,)

    def test_fail_open_preserved_through_adapter(self):
        import numpy as np

        adapter = PyTorchAdapter(QMLMonitor())
        diagnosis = adapter.record_step(0, loss=1.0, gradients=np.array([]))
        assert diagnosis.degraded is True

    def test_loss_tensor_with_grad_converted_to_float(self):
        model = _tiny_model()
        monitor = QMLMonitor()
        adapter = PyTorchAdapter(monitor, module=model)
        x = torch.randn(4, 3)
        y = torch.randn(4, 1)
        loss = torch.nn.functional.mse_loss(model(x), y)
        adapter.record_step(0, loss)
        obs = monitor.state.latest_observation
        assert isinstance(obs.training_event.loss, float)
