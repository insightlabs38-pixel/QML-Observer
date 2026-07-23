"""GenericAdapter: the framework-neutral, lowest-level integration path.

Milestone 2, Issue #17 ("Add generic adapter").

Per blueprint Volume VIII, this is the adapter that every other adapter
(PennyLane, Qiskit, Milestones 6/8) effectively sits in front of: those
adapters translate framework-specific objects into the same plain
argument shapes accepted here and then call through to `QMLMonitor.update()`.
`GenericAdapter` exposes that shape directly, so a user with a fully custom
research training loop can instrument it in a couple of lines without
waiting on any framework-specific integration.

Scope note: unlike the (future) PennyLane/Qiskit adapters, `GenericAdapter`
does not implement the `attach`/`detach` lifecycle described for
`adapters/base.py` (Volume VIII) -- that hook exists to attach to a
framework's own training object (a QNode, an `OptimizerResult`, etc.), and
there is no such object to attach to in the generic case. This matches the
blueprint's own sketch of `GenericAdapter`, which only defines
`__init__(monitor)` and `record(...)`.
"""

from __future__ import annotations

from typing import Any

from qml_observer.core.monitor import QMLMonitor
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult
from qml_observer.schemas.optimizer import OptimizerMetadata


class GenericAdapter:
    """Framework-neutral adapter: a thin, explicit pass-through to `QMLMonitor.update()`.

    This adapter adds no behavior of its own beyond argument forwarding and
    a constructor type check -- it exists purely as a stable, documented
    integration surface for manual/custom training loops, distinct from
    calling `monitor.update()` directly only in that it gives adopters a
    named "adapter" object matching the project's adapter-layer vocabulary
    (useful once multiple adapters, e.g. PennyLane's, are also in play).

    Example:
        >>> from qml_observer import QMLMonitor
        >>> from qml_observer.adapters.generic import GenericAdapter
        >>> monitor = QMLMonitor()
        >>> adapter = GenericAdapter(monitor)
        >>> for step in range(100):
        ...     loss, gradients = my_training_step()
        ...     diagnosis = adapter.record(step, loss=loss, gradients=gradients)
    """

    def __init__(self, monitor: QMLMonitor) -> None:
        """Wrap an existing `QMLMonitor` for manual step recording.

        Raises:
            TypeError: If `monitor` is not a `QMLMonitor` instance.
        """
        if not isinstance(monitor, QMLMonitor):
            raise TypeError(f"monitor must be a QMLMonitor, got {type(monitor)!r}")
        self.monitor = monitor

    def record(
        self,
        step: int,
        loss: float | None = None,
        gradients: Any | None = None,
        parameters: Any | None = None,
        circuit: CircuitMetadata | None = None,
        optimizer: OptimizerMetadata | None = None,
        shots: int | None = None,
    ) -> DiagnosisResult:
        """Record one step of an arbitrary training loop.

        A direct pass-through to `self.monitor.update()` -- see that
        method's docstring for full argument semantics, including the
        fail-open guarantee (step-processing errors never raise here;
        only misuse, like calling `record()` after the run has
        `finish()`-ed, does).
        """
        return self.monitor.update(
            step=step,
            loss=loss,
            gradients=gradients,
            parameters=parameters,
            circuit=circuit,
            optimizer=optimizer,
            shots=shots,
        )
