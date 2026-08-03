"""PyTorchAdapter: observability for PyTorch-driven hybrid quantum-classical training.

Milestone 14 (`future_milestones_plan.md`), Issue #98 ("PyTorch
hybrid-workflow integration").

Targets the common hybrid pattern where a quantum circuit is wrapped as a
`torch.nn.Module` (e.g. via `qml.qnn.TorchLayer` or a hand-rolled
`torch.autograd.Function`) and trained inside an ordinary PyTorch loop with
a `torch.optim.Optimizer`. Per the blueprint's core architectural rule,
this adapter *observes* an already-run `loss.backward()` step -- it never
calls `.backward()` itself or reimplements the optimizer step. Its value
over `AutogradAdapter`/`GenericAdapter` is auto-collecting gradients and
parameters from an attached module (so the caller doesn't have to flatten
`p.grad` for every parameter by hand) and reading optimizer name/learning
rate directly off an attached `torch.optim.Optimizer`.
"""

from __future__ import annotations

from typing import Any

from qml_observer.adapters.autograd import AutogradAdapter
from qml_observer.core.monitor import QMLMonitor
from qml_observer.schemas.diagnosis import DiagnosisResult
from qml_observer.schemas.optimizer import OptimizerMetadata

try:
    import torch
except ImportError as _exc:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = _exc
else:
    _IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorchAdapter requires the optional 'torch' dependency. "
            "Install it with `pip install qml-observer[torch]` or "
            "`pip install torch>=2.0`."
        ) from _IMPORT_ERROR


class PyTorchAdapter(AutogradAdapter):
    """Adapter observing a PyTorch module's (and optionally optimizer's) training loop.

    Example:
        >>> import torch
        >>> from qml_observer import QMLMonitor
        >>> from qml_observer.adapters.pytorch.adapter import PyTorchAdapter
        >>>
        >>> model = build_hybrid_qnn()          # e.g. wraps a qml.qnn.TorchLayer
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>>
        >>> monitor = QMLMonitor()
        >>> adapter = PyTorchAdapter(monitor, module=model, optimizer=optimizer)
        >>> for step in range(200):
        ...     optimizer.zero_grad()
        ...     loss = loss_fn(model(x), y)
        ...     loss.backward()
        ...     diagnosis = adapter.record_step(step, loss)
        ...     optimizer.step()
        ...     if monitor.should_stop():
        ...         break
    """

    framework_name = "pytorch"

    def __init__(
        self,
        monitor: QMLMonitor,
        module: Any | None = None,
        optimizer: Any | None = None,
        *,
        optimizer_name: str | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Create an adapter wrapping `monitor`.

        Args:
            monitor: The `QMLMonitor` to forward recorded steps to.
            module: Optional `torch.nn.Module` to `attach()` immediately,
                so `record_step()` can auto-collect gradients/parameters
                from it.
            optimizer: Optional `torch.optim.Optimizer` to `attach()`
                immediately, so `record_step()` can auto-populate
                `OptimizerMetadata.name`/`learning_rate` from it.
            optimizer_name: Explicit optimizer name, used if no
                `optimizer` is attached (or as an override).
            learning_rate: Explicit learning rate, same override
                semantics as `optimizer_name`.

        Raises:
            ImportError: If the `torch` package is not installed.
            TypeError: If `monitor` is not a `QMLMonitor` instance, or
                `module`/`optimizer` are given but aren't the expected
                torch types.
        """
        _require_torch()
        super().__init__(monitor, optimizer_name=optimizer_name, learning_rate=learning_rate)
        self._module: Any | None = None
        self._optimizer: Any | None = None
        if module is not None or optimizer is not None:
            self.attach(module=module, optimizer=optimizer)

    # -- attach/detach lifecycle --------------------------------------------

    def attach(self, module: Any | None = None, optimizer: Any | None = None) -> PyTorchAdapter:
        """Attach a `torch.nn.Module` and/or `torch.optim.Optimizer`.

        Either or both may be given; each replaces whatever was
        previously attached of that kind.

        Returns:
            `self`, to allow `PyTorchAdapter(monitor).attach(module=model)`.

        Raises:
            TypeError: If `module` isn't a `torch.nn.Module`, or
                `optimizer` isn't a `torch.optim.Optimizer`.
        """
        if module is not None:
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"module must be a torch.nn.Module, got {type(module)!r}")
            self._module = module
        if optimizer is not None:
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError(
                    f"optimizer must be a torch.optim.Optimizer, got {type(optimizer)!r}"
                )
            self._optimizer = optimizer
        return self

    def detach(self) -> None:
        """Detach the current module and optimizer.

        `record_step()` still works afterward, but without automatic
        gradient/parameter collection or optimizer metadata until
        `attach()` is called again.
        """
        self._module = None
        self._optimizer = None

    @property
    def attached(self) -> bool:
        """Whether a module and/or optimizer is currently attached."""
        return self._module is not None or self._optimizer is not None

    # -- recording -----------------------------------------------------------

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
        """Record one PyTorch training step.

        If `gradients`/`parameters` aren't given explicitly and a module
        is attached, they're auto-collected from `module.parameters()`
        (flattening and concatenating each parameter's `.grad`/data into
        one 1-D array) -- the caller only needs to call `loss.backward()`
        beforehand, exactly like the usage example above.

        Args:
            step: Monotonically increasing step index.
            loss: Observed loss for this step -- typically the scalar
                `torch.Tensor` returned by the loss function, still
                carrying its autograd graph; converted to a plain
                `float` here.
            gradients: Explicit gradient tensor/array, overriding
                auto-collection from an attached module.
            parameters: Explicit parameter tensor/array, overriding
                auto-collection from an attached module.
            shots: Shot count for this step, if the wrapped quantum layer
                uses shot-based execution.
            gradient_method: Name of the gradient computation method.
                Defaults to `"backprop"` (PyTorch's own gradient
                computation) if not given explicitly.
            ansatz_name: Name of the ansatz in use, if known.
            initialization: Name of the parameter initialization
                strategy, if known.

        Returns:
            The `DiagnosisResult` for this step.
        """
        if gradients is None and self._module is not None:
            gradients = self._collect_gradients()
        if parameters is None and self._module is not None:
            parameters = self._collect_parameters()

        if self._n_parameters is None and self._module is not None:
            module = self._module
            self._n_parameters = self._safe(lambda: sum(p.numel() for p in module.parameters()))

        return super().record_step(
            step,
            loss=loss,
            gradients=gradients,
            parameters=parameters,
            shots=shots,
            gradient_method=gradient_method or "backprop",
            ansatz_name=ansatz_name,
            initialization=initialization,
        )

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _safe(fn: Any) -> Any:
        try:
            return fn()
        except Exception:
            return None

    def _collect_gradients(self) -> Any | None:
        """Concatenate every attached parameter's `.grad` into one flat array.

        Parameters with no gradient yet (e.g. unused branches, or before
        the first `backward()`) are skipped rather than raising. Returns
        `None` (rather than an empty array) if no gradients are available
        at all, so `QMLMonitor.update()` treats this step as having no
        gradient observation, consistent with `summarize_gradient()`
        rejecting empty arrays.
        """
        return self._safe(self._collect_gradients_unsafe)

    def _collect_gradients_unsafe(self) -> Any | None:
        assert self._module is not None
        grads = [
            p.grad.detach().reshape(-1) for p in self._module.parameters() if p.grad is not None
        ]
        if not grads:
            return None
        return torch.cat(grads).cpu().numpy()

    def _collect_parameters(self) -> Any | None:
        return self._safe(self._collect_parameters_unsafe)

    def _collect_parameters_unsafe(self) -> Any | None:
        assert self._module is not None
        params = [p.detach().reshape(-1) for p in self._module.parameters()]
        if not params:
            return None
        return torch.cat(params).cpu().numpy()

    def _build_optimizer_metadata(self, gradient_method: str | None) -> OptimizerMetadata | None:
        name = self._optimizer_name
        lr = self._learning_rate
        if self._optimizer is not None:
            name = name or type(self._optimizer).__name__
            if lr is None:
                lr = self._safe(lambda: self._optimizer.param_groups[0].get("lr"))
        if name is None and lr is None and gradient_method is None:
            return None
        return OptimizerMetadata(
            name=name or "unknown", learning_rate=lr, gradient_method=gradient_method
        )
