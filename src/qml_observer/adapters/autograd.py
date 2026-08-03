"""AutogradAdapter: a framework-neutral adapter for hybrid classical-autodiff QML loops.

Milestone 14 (`future_milestones_plan.md`), Issue #100 ("Generic autograd
adapter").

This sits between `GenericAdapter` (Milestone 2) and the fully
framework-specific adapters (`PennyLaneAdapter`/`QiskitAdapter`, and now
`PyTorchAdapter`/`JAXAdapter`, Issues #98/#99). `GenericAdapter` performs
*no* conversion at all -- it assumes the caller already has plain
`float`/`numpy.ndarray` values. Many hybrid quantum-classical workflows
instead compute loss/gradients with a classical autodiff library (PyTorch,
JAX, or something else entirely -- a custom research autodiff stack, a
less common framework, etc.), whose tensors/arrays don't feed directly
into `summarize_gradient()`'s `np.asarray(..., dtype=float)` call (a
`torch.Tensor` with `requires_grad=True` raises there, for example).

`AutogradAdapter` is directly usable on its own for exactly that case: it
duck-types its way through `.detach()`/`.cpu()`/`.numpy()`-style methods,
`__array__`, or a plain `np.asarray()` fallback, so *any* autodiff
framework's tensors can be forwarded to `QMLMonitor.update()` without a
framework-specific adapter existing for it. `PyTorchAdapter` and
`JAXAdapter` are thin subclasses that add framework-aware ergonomics
(auto-collecting gradients from an attached `torch.nn.Module`, pytree-aware
parameter counting for JAX, etc.) on top of this same conversion logic --
neither reimplements it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qml_observer.core.monitor import QMLMonitor
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult
from qml_observer.schemas.optimizer import OptimizerMetadata


def to_numpy(value: Any) -> Any:
    """Best-effort conversion of a tensor/array from any autodiff framework to plain numpy.

    Tries, in order:
      1. `None`/already-`numpy.ndarray` passthrough.
      2. Torch-`Tensor`-style duck typing: `.detach()` (drops the autograd
         graph, since a tensor with `requires_grad=True` cannot be handed
         to `np.asarray` directly), then `.cpu()` if present (moves off an
         accelerator device), then `.numpy()`.
      3. `np.asarray(value)`, which already handles anything exposing
         `__array__` (JAX arrays, plain lists, scalars, etc.).

    Any failure at a given step falls through to the next, and a total
    failure returns `value` unchanged rather than raising -- conversion
    errors surface later as ordinary, already-handled fail-open failures
    inside `QMLMonitor.update()` (addendum §1) rather than here.
    """
    if value is None or isinstance(value, np.ndarray):
        return value

    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            detached = detach()
            cpu = getattr(detached, "cpu", None)
            detached = cpu() if callable(cpu) else detached
            as_numpy = getattr(detached, "numpy", None)
            if callable(as_numpy):
                return as_numpy()
        except Exception:
            pass

    try:
        return np.asarray(value)
    except Exception:
        return value


class AutogradAdapter:
    """Framework-neutral adapter for a classical-autodiff-driven hybrid QML loop.

    Example (a hand-rolled autodiff stack, or any framework without its
    own dedicated adapter):
        >>> from qml_observer import QMLMonitor
        >>> from qml_observer.adapters.autograd import AutogradAdapter
        >>>
        >>> monitor = QMLMonitor()
        >>> adapter = AutogradAdapter(monitor, optimizer_name="Adam", learning_rate=0.01)
        >>> for step in range(200):
        ...     loss, grad_tensor, param_tensor = my_autodiff_training_step()
        ...     diagnosis = adapter.record_step(step, loss, grad_tensor, param_tensor)
    """

    #: Overridden by subclasses (`"pytorch"`, `"jax"`) purely as a
    #: human-readable label; not consumed by `QMLMonitor` itself.
    framework_name: str = "autograd"

    def __init__(
        self,
        monitor: QMLMonitor,
        *,
        optimizer_name: str | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Create an adapter wrapping `monitor`.

        Args:
            monitor: The `QMLMonitor` to forward recorded steps to.
            optimizer_name: Name of the classical optimizer driving
                training, if known (e.g. `"Adam"`). Populates
                `OptimizerMetadata.name`.
            learning_rate: Learning rate of the classical optimizer, if
                known. Populates `OptimizerMetadata.learning_rate`.

        Raises:
            TypeError: If `monitor` is not a `QMLMonitor` instance.
        """
        if not isinstance(monitor, QMLMonitor):
            raise TypeError(f"monitor must be a QMLMonitor, got {type(monitor)!r}")
        self.monitor = monitor
        self._optimizer_name = optimizer_name
        self._learning_rate = learning_rate
        self._n_parameters: int | None = None

    def record_step(
        self,
        step: int,
        loss: Any | None = None,
        gradients: Any | None = None,
        parameters: Any | None = None,
        *,
        shots: int | None = None,
        gradient_method: str | None = None,
        ansatz_name: str | None = None,
        initialization: str | None = None,
    ) -> DiagnosisResult:
        """Record one training step, converting `loss`/`gradients`/`parameters` first.

        Args:
            step: Monotonically increasing step index.
            loss: Observed loss value, if available. May be a scalar
                tensor from any autodiff framework (converted via
                `to_numpy` and reduced to a plain `float`), a Python
                number, or `None`.
            gradients: Raw gradient tensor/array, if available.
                Converted via `to_numpy` before being forwarded to
                `QMLMonitor.update()`, which summarizes it via
                `summarize_gradient`.
            parameters: Raw parameter tensor/array, if available.
                Converted the same way. Its size (if it can be
                determined) is used to populate `CircuitMetadata.n_parameters`
                when the subclass hasn't already supplied a parameter
                count some other way (e.g. `PyTorchAdapter` counting an
                attached module's parameters directly).
            shots: Shot count for this step, if using shot-based
                execution somewhere in the hybrid pipeline.
            gradient_method: Name of the gradient computation method
                (e.g. `"backprop"`, `"autodiff"`), if the caller/subclass
                wants to record it.
            ansatz_name: Name of the ansatz in use, if known. Not
                auto-detected (this adapter has no circuit introspection
                of its own).
            initialization: Name of the parameter initialization
                strategy, if known. Not auto-detected.

        Returns:
            The `DiagnosisResult` for this step, exactly as returned by
            `QMLMonitor.update()`.
        """
        loss_value = self._to_scalar(loss)
        grad_array = to_numpy(gradients) if gradients is not None else None
        param_array = to_numpy(parameters) if parameters is not None else None

        n_parameters = self._n_parameters
        if n_parameters is None and param_array is not None:
            n_parameters = self._safe_size(param_array)

        circuit_meta: CircuitMetadata | None = None
        if n_parameters is not None or ansatz_name is not None or initialization is not None:
            circuit_meta = CircuitMetadata(
                n_parameters=n_parameters,
                ansatz_name=ansatz_name,
                initialization=initialization,
            )

        optimizer_meta = self._build_optimizer_metadata(gradient_method)

        return self.monitor.update(
            step=step,
            loss=loss_value,
            gradients=grad_array,
            parameters=param_array,
            circuit=circuit_meta,
            optimizer=optimizer_meta,
            shots=shots,
        )

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _safe_size(array: Any) -> int | None:
        try:
            return int(np.asarray(array).size)
        except Exception:
            return None

    @classmethod
    def _to_scalar(cls, value: Any) -> float | None:
        """Reduce a scalar-valued loss (possibly a 0-d/1-element tensor) to a `float`."""
        if value is None:
            return None
        converted = to_numpy(value)
        try:
            array = np.asarray(converted, dtype=float)
        except Exception:
            return None
        if array.size == 0:
            return None
        if array.size == 1:
            return float(array.reshape(-1)[0])
        # Not actually scalar-valued -- an adapter/user error, but per the
        # fail-open policy this degrades to "unknown" here rather than
        # raising; `QMLMonitor.update()` still gets a usable step.
        return None

    def _build_optimizer_metadata(self, gradient_method: str | None) -> OptimizerMetadata | None:
        if self._optimizer_name is None and gradient_method is None and self._learning_rate is None:
            return None
        return OptimizerMetadata(
            name=self._optimizer_name or "unknown",
            learning_rate=self._learning_rate,
            gradient_method=gradient_method,
        )
