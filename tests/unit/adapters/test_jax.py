"""Unit tests for qml_observer.adapters.jax.adapter.JAXAdapter.

Milestone 14, Issue #99 ("JAX hybrid-workflow integration"). Skipped
entirely if the optional `jax` dependency isn't installed (`pip install
qml-observer[jax]`).
"""

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from qml_observer.adapters.jax.adapter import JAXAdapter  # noqa: E402
from qml_observer.core.monitor import QMLMonitor  # noqa: E402
from qml_observer.schemas.diagnosis import DiagnosisResult  # noqa: E402


def _pytree_params():
    return {"w": jnp.ones((2, 3)), "b": jnp.zeros((3,))}


class TestConstruction:
    def test_wraps_monitor(self):
        monitor = QMLMonitor()
        adapter = JAXAdapter(monitor)
        assert adapter.monitor is monitor
        assert adapter.attached is False

    def test_rejects_non_monitor(self):
        with pytest.raises(TypeError):
            JAXAdapter("not-a-monitor")

    def test_attaches_params_at_construction(self):
        adapter = JAXAdapter(QMLMonitor(), _pytree_params())
        assert adapter.attached is True


class TestAttachDetach:
    def test_attach_returns_self(self):
        adapter = JAXAdapter(QMLMonitor())
        assert adapter.attach(_pytree_params()) is adapter

    def test_detach_clears_template(self):
        adapter = JAXAdapter(QMLMonitor(), _pytree_params())
        adapter.detach()
        assert adapter.attached is False


class TestRecordStep:
    def test_record_step_returns_diagnosis(self):
        adapter = JAXAdapter(QMLMonitor())
        diagnosis = adapter.record_step(0, loss=1.0)
        assert isinstance(diagnosis, DiagnosisResult)

    def test_flattens_pytree_gradients(self):
        monitor = QMLMonitor()
        adapter = JAXAdapter(monitor)
        params = _pytree_params()
        grads = {"w": jnp.full((2, 3), 0.1), "b": jnp.full((3,), 0.2)}

        adapter.record_step(0, loss=0.5, gradients=grads, parameters=params)

        obs = monitor.state.latest_observation
        assert obs.gradient is not None
        assert obs.gradient.values.shape == (9,)  # 2*3 + 3
        assert obs.circuit is not None
        assert obs.circuit.n_parameters == 9

    def test_parameter_count_falls_back_to_attached_template(self):
        monitor = QMLMonitor()
        adapter = JAXAdapter(monitor, _pytree_params())
        adapter.record_step(0, loss=0.5, gradients=jnp.array([0.1, 0.2]))
        obs = monitor.state.latest_observation
        assert obs.circuit is not None
        assert obs.circuit.n_parameters == 9

    def test_single_array_gradients_still_work(self):
        monitor = QMLMonitor()
        adapter = JAXAdapter(monitor)
        adapter.record_step(0, loss=0.5, gradients=jnp.array([0.1, 0.2, 0.3]))
        obs = monitor.state.latest_observation
        assert obs.gradient is not None
        assert obs.gradient.values.shape == (3,)

    def test_optimizer_metadata_requires_explicit_values(self):
        monitor = QMLMonitor()
        adapter = JAXAdapter(monitor, optimizer_name="Adam", learning_rate=0.02)
        adapter.record_step(0, loss=0.5)
        obs = monitor.state.latest_observation
        assert obs.optimizer is not None
        assert obs.optimizer.name == "Adam"
        assert obs.optimizer.learning_rate == pytest.approx(0.02)
        assert obs.optimizer.gradient_method == "autodiff"

    def test_loss_scalar_array_converted_to_float(self):
        monitor = QMLMonitor()
        adapter = JAXAdapter(monitor)
        adapter.record_step(0, loss=jnp.array(0.75))
        obs = monitor.state.latest_observation
        assert isinstance(obs.training_event.loss, float)
        assert obs.training_event.loss == pytest.approx(0.75)

    def test_fail_open_preserved_through_adapter(self):
        adapter = JAXAdapter(QMLMonitor())
        diagnosis = adapter.record_step(0, loss=1.0, gradients=jnp.array([]))
        assert diagnosis.degraded is True
