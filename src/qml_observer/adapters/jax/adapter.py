"""JAXAdapter: observability for JAX-driven hybrid quantum-classical training.

Milestone 14 (`future_milestones_plan.md`), Issue #99 ("JAX hybrid-workflow
integration").

JAX training loops are typically written in an explicit, functional style:
parameters and gradients are pytrees (nested dicts/lists/tuples of
arrays, e.g. as produced by `jax.grad`), not attributes of a stateful
module/optimizer object the way PyTorch's are. This adapter's value over
`AutogradAdapter`/`GenericAdapter` is therefore pytree-awareness: it
flattens a parameter/gradient pytree into the single 1-D array
`QMLMonitor.update()` expects (via `jax.tree_util.tree_leaves`), and
counts total parameters across the whole pytree rather than requiring the
caller to do that arithmetic by hand.

Unlike `PyTorchAdapter`, no optimizer object is introspected here: JAX
optimizer state (e.g. from `optax`) is itself an opaque pytree with no
standard place to read a name or learning rate from, so
`optimizer_name`/`learning_rate` must be supplied explicitly if wanted --
the same pattern `PennyLaneAdapter` uses for the same reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qml_observer.adapters.autograd import AutogradAdapter, to_numpy
from qml_observer.core.monitor import QMLMonitor
from qml_observer.schemas.diagnosis import DiagnosisResult

try:
    import jax
except ImportError as _exc:  # pragma: no cover - exercised only without jax installed
    jax = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = _exc
else:
    _IMPORT_ERROR = None


def _require_jax() -> None:
    if jax is None:
        raise ImportError(
            "JAXAdapter requires the optional 'jax' dependency. "
            "Install it with `pip install qml-observer[jax]` or "
            "`pip install jax>=0.4`."
        ) from _IMPORT_ERROR


class JAXAdapter(AutogradAdapter):
    """Adapter observing a JAX pytree-based hybrid QML training loop.

    Example:
        >>> import jax
        >>> from qml_observer import QMLMonitor
        >>> from qml_observer.adapters.jax.adapter import JAXAdapter
        >>>
        >>> params = init_hybrid_params()          # a pytree of jax arrays
        >>> monitor = QMLMonitor()
        >>> adapter = JAXAdapter(monitor, params, optimizer_name="Adam", learning_rate=0.01)
        >>>
        >>> for step in range(200):
        ...     loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        ...     diagnosis = adapter.record_step(step, loss, grads, params)
        ...     params = optax_update(params, grads)
        ...     if monitor.should_stop():
        ...         break
    """

    framework_name = "jax"

    def __init__(
        self,
        monitor: QMLMonitor,
        params: Any | None = None,
        *,
        optimizer_name: str | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Create an adapter wrapping `monitor`.

        Args:
            monitor: The `QMLMonitor` to forward recorded steps to.
            params: Optional parameter pytree to `attach()` immediately,
                used as a fallback parameter-count source for steps where
                `parameters` isn't passed to `record_step()` directly
                (e.g. because only `gradients` changes each step).
            optimizer_name: Optimizer name, since JAX/optax optimizer
                state carries no standard name/learning-rate field to
                introspect (see module docstring).
            learning_rate: Learning rate, same reasoning as
                `optimizer_name`.

        Raises:
            ImportError: If the `jax` package is not installed.
            TypeError: If `monitor` is not a `QMLMonitor` instance.
        """
        _require_jax()
        super().__init__(monitor, optimizer_name=optimizer_name, learning_rate=learning_rate)
        self._params_template: Any | None = None
        if params is not None:
            self.attach(params)

    # -- attach/detach lifecycle --------------------------------------------

    def attach(self, params: Any) -> JAXAdapter:
        """Attach a parameter pytree, used as a fallback for parameter counting.

        Returns:
            `self`, to allow `JAXAdapter(monitor).attach(params)`.
        """
        self._params_template = params
        return self

    def detach(self) -> None:
        """Detach the current parameter pytree template."""
        self._params_template = None

    @property
    def attached(self) -> bool:
        """Whether a parameter pytree is currently attached."""
        return self._params_template is not None

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
        """Record one JAX training step.

        `gradients`/`parameters` may each be a single array or an
        arbitrary pytree (dict/list/tuple nesting of arrays, as produced
        by `jax.grad`/`jax.value_and_grad`); both are flattened into one
        1-D array via `jax.tree_util.tree_leaves` before being forwarded.

        Args:
            step: Monotonically increasing step index.
            loss: Observed loss for this step -- a JAX scalar array or
                plain float.
            gradients: Gradient pytree/array for this step, if available.
            parameters: Parameter pytree/array for this step, if
                available. If omitted, the pytree passed to `attach()` /
                the constructor is used only to determine parameter
                *count* (not flattened gradients/parameters themselves).
            shots: Shot count for this step, if the wrapped circuit uses
                shot-based execution.
            gradient_method: Name of the gradient computation method.
                Defaults to `"autodiff"` if not given explicitly.
            ansatz_name: Name of the ansatz in use, if known.
            initialization: Name of the parameter initialization
                strategy, if known.

        Returns:
            The `DiagnosisResult` for this step.
        """
        pytree_for_count = parameters if parameters is not None else self._params_template
        if pytree_for_count is not None:
            counted = self._count_params(pytree_for_count)
            if counted is not None:
                self._n_parameters = counted

        flat_gradients = self._flatten_pytree(gradients) if gradients is not None else None
        flat_parameters = self._flatten_pytree(parameters) if parameters is not None else None

        return super().record_step(
            step,
            loss=loss,
            gradients=flat_gradients,
            parameters=flat_parameters,
            shots=shots,
            gradient_method=gradient_method or "autodiff",
            ansatz_name=ansatz_name,
            initialization=initialization,
        )

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _count_params(pytree: Any) -> int | None:
        try:
            leaves = jax.tree_util.tree_leaves(pytree)
            return int(sum(np.asarray(leaf).size for leaf in leaves))
        except Exception:
            return None

    @staticmethod
    def _flatten_pytree(pytree: Any) -> Any | None:
        """Flatten a pytree of arrays into one 1-D numpy array.

        Falls back to `to_numpy()` on the object as a whole if it isn't a
        pytree `jax.tree_util` can walk (e.g. it's already a single
        array), so a plain array argument still works exactly like
        `AutogradAdapter`'s own conversion.
        """
        try:
            leaves = jax.tree_util.tree_leaves(pytree)
        except Exception:
            return to_numpy(pytree)
        if not leaves:
            return None
        try:
            flat = [np.asarray(leaf).ravel() for leaf in leaves]
            return np.concatenate(flat)
        except Exception:
            return to_numpy(pytree)
