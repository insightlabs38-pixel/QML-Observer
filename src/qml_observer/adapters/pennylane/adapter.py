"""PennyLaneAdapter: the first real framework integration.

Milestone 6 (Volume IX): Issue #41 ("Implement PennyLane adapter"),
Issue #42 ("Support parameter-shift metadata"), Issue #43 ("Support
adjoint differentiation"), Issue #44 ("Support finite shots"), Issue #45
("Extract circuit metadata").

Per the blueprint's core architectural rule, this adapter *observes*
PennyLane training, it does not reimplement PennyLane's gradient
machinery. The user's training loop still computes `loss`/`gradients`
however it likes (autograd, Torch, JAX interfaces; parameter-shift,
adjoint, finite-diff, backprop, or any other `diff_method`); this adapter
is responsible only for:

  1. Forwarding already-computed `loss`/`gradients`/`parameters` to
     `QMLMonitor.update()` (like `GenericAdapter`, but framework-aware).
  2. Filling in `CircuitMetadata` and `OptimizerMetadata` automatically
     from the attached `QNode`, so the user doesn't have to build those
     by hand every step.

Because PennyLane's internal APIs for tape construction have changed
across releases (the project targets `pennylane>=0.35`), every piece of
metadata extraction here is best-effort: if a given PennyLane version
doesn't expose something the way we expect, that field is simply left
`None` rather than raising -- consistent with `CircuitMetadata`'s own
"every field is optional" design and this project's fail-open policy
(addendum §1). `QMLMonitor.update()` additionally wraps the whole step in
its own fail-open try/except, so even a totally unexpected PennyLane
internal change degrades gracefully instead of crashing the training
loop.
"""

from __future__ import annotations

import logging
from typing import Any

from qml_observer.core.monitor import QMLMonitor
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult
from qml_observer.schemas.optimizer import OptimizerMetadata

_logger = logging.getLogger("qml_observer")

try:
    import pennylane as qml
except ImportError as _exc:  # pragma: no cover - exercised only without pennylane installed
    qml = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = _exc
else:
    _IMPORT_ERROR = None


def _require_pennylane() -> None:
    if qml is None:
        raise ImportError(
            "PennyLaneAdapter requires the optional 'pennylane' dependency. "
            "Install it with `pip install qml-observer[pennylane]` or "
            "`pip install pennylane>=0.35`."
        ) from _IMPORT_ERROR


class PennyLaneAdapter:
    """Adapter observing a PennyLane `QNode`'s training loop.

    Example:
        >>> import pennylane as qml
        >>> from qml_observer import QMLMonitor
        >>> from qml_observer.adapters.pennylane.adapter import PennyLaneAdapter
        >>>
        >>> dev = qml.device("default.qubit", wires=4)
        >>> @qml.qnode(dev, diff_method="parameter-shift")
        ... def circuit(params):
        ...     qml.StronglyEntanglingLayers(params, wires=range(4))
        ...     return qml.expval(qml.PauliZ(0))
        >>>
        >>> monitor = QMLMonitor()
        >>> adapter = PennyLaneAdapter(monitor, optimizer_name="Adam", learning_rate=0.05)
        >>> adapter.attach(circuit)
        >>> for step in range(200):
        ...     loss = circuit(params)
        ...     grads = qml.grad(circuit)(params)
        ...     diagnosis = adapter.record_step(step, loss, grads, parameters=params)
    """

    def __init__(
        self,
        monitor: QMLMonitor,
        qnode: Any | None = None,
        *,
        optimizer_name: str | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Create an adapter wrapping `monitor`.

        Args:
            monitor: The `QMLMonitor` to forward recorded steps to.
            qnode: Optional PennyLane `QNode` to `attach()` immediately.
            optimizer_name: Name of the classical optimizer driving
                training (e.g. "Adam"), if known. Used to populate
                `OptimizerMetadata.name`; defaults to `"unknown"` in the
                resulting metadata if omitted but other optimizer
                information (learning rate or gradient method) is
                available.
            learning_rate: Learning rate of the classical optimizer, if
                known. Populates `OptimizerMetadata.learning_rate`.

        Raises:
            ImportError: If the `pennylane` package is not installed.
            TypeError: If `monitor` is not a `QMLMonitor` instance.
        """
        _require_pennylane()
        if not isinstance(monitor, QMLMonitor):
            raise TypeError(f"monitor must be a QMLMonitor, got {type(monitor)!r}")

        self.monitor = monitor
        self._optimizer_name = optimizer_name
        self._learning_rate = learning_rate
        self._qnode: Any | None = None
        if qnode is not None:
            self.attach(qnode)

    # -- attach/detach lifecycle (blueprint Volume IX) --------------------

    def attach(self, training_object: Any) -> PennyLaneAdapter:
        """Attach a PennyLane `QNode` as the source of circuit/gradient metadata.

        Once attached, `record_step()` will automatically populate
        `CircuitMetadata` (by constructing the tape for the given
        `parameters`) and `OptimizerMetadata.gradient_method` (from the
        QNode's configured `diff_method`) on every call.

        Args:
            training_object: A PennyLane `QNode` (or any object exposing
                the same `.device` and `.diff_method` attributes).

        Returns:
            `self`, to allow `PennyLaneAdapter(monitor).attach(circuit)`.

        Raises:
            TypeError: If `training_object` doesn't look like a QNode.
        """
        if not hasattr(training_object, "device") or not hasattr(training_object, "diff_method"):
            raise TypeError(
                "attach() expects a PennyLane QNode (or QNode-like object "
                "exposing `.device` and `.diff_method`), got "
                f"{type(training_object)!r}"
            )
        self._qnode = training_object
        return self

    def detach(self) -> None:
        """Detach the current QNode. `record_step()` still works, but without
        automatic circuit/gradient-method metadata until `attach()` is
        called again."""
        self._qnode = None

    @property
    def attached(self) -> bool:
        """Whether a QNode is currently attached."""
        return self._qnode is not None

    # -- recording ---------------------------------------------------------

    def record_step(
        self,
        step: int,
        loss: float | None = None,
        gradients: Any | None = None,
        parameters: Any | None = None,
        *,
        shots: int | None = None,
        ansatz_name: str | None = None,
        initialization: str | None = None,
    ) -> DiagnosisResult:
        """Record one PennyLane training step.

        Args:
            step: Monotonically increasing step index.
            loss: Observed loss value for this step, if available.
            gradients: Raw gradient array (e.g. from `qml.grad(circuit)(params)`),
                if available. Forwarded as-is to `QMLMonitor.update()`,
                which summarizes it via `summarize_gradient`.
            parameters: The parameter values the circuit was evaluated at
                this step. If a QNode is attached, this is used to
                construct the tape for circuit-metadata extraction
                (Issue #45) and, when `shots` is not given explicitly, to
                read the shot count actually used for this call
                (Issue #44).
            shots: Explicit shot count for this step, overriding whatever
                would otherwise be inferred from the tape/device. Useful
                when the caller already knows the value or the tape
                can't be reconstructed from `parameters` alone.
            ansatz_name: Name of the ansatz in use (e.g.
                "StronglyEntanglingLayers"), if the caller wants to record
                it. Not auto-detected: a generic `QuantumScript` doesn't
                generically expose an "ansatz name" the way it exposes
                gate/wire/parameter counts.
            initialization: Name of the parameter initialization strategy
                in use, if the caller wants to record it. Not
                auto-detected, for the same reason as `ansatz_name`.

        Returns:
            The `DiagnosisResult` for this step, exactly as returned by
            `QMLMonitor.update()` (including its fail-open `degraded`
            behavior).
        """
        tape = self._construct_tape(parameters) if parameters is not None else None

        circuit_meta: CircuitMetadata | None = None
        if tape is not None:
            circuit_meta = self.extract_circuit_metadata(
                tape, ansatz_name=ansatz_name, initialization=initialization
            )
            if shots is None:
                shots = self._shots_from_tape(tape)

        if shots is None:
            shots = self._resolve_device_shots()

        gradient_method = self._resolve_gradient_method()
        optimizer_meta = self._build_optimizer_metadata(gradient_method)

        return self.monitor.update(
            step=step,
            loss=loss,
            gradients=gradients,
            parameters=parameters,
            circuit=circuit_meta,
            optimizer=optimizer_meta,
            shots=shots,
        )

    # -- circuit metadata extraction (Issue #45) --------------------------

    def extract_circuit_metadata(
        self,
        tape: Any,
        *,
        ansatz_name: str | None = None,
        initialization: str | None = None,
    ) -> CircuitMetadata:
        """Build a `CircuitMetadata` from a PennyLane tape (`QuantumScript`).

        Every field is extracted defensively: if the installed PennyLane
        version doesn't expose a given piece of information the way this
        method expects, that field is left `None` rather than raising,
        per `CircuitMetadata`'s own "every field is optional" contract.

        Args:
            tape: A PennyLane `QuantumScript`/tape, e.g. as returned by
                `qml.workflow.construct_tape(qnode)(*params)`.
            ansatz_name: Passed straight through to the resulting
                `CircuitMetadata.ansatz_name` (see `record_step` docstring
                for why this isn't auto-detected).
            initialization: Passed straight through to
                `CircuitMetadata.initialization` (same reasoning).

        Returns:
            A `CircuitMetadata` populated with whatever fields could be
            extracted from `tape`.
        """
        wires = getattr(tape, "wires", None)
        n_qubits = len(wires) if wires is not None else None

        operations = list(getattr(tape, "operations", None) or [])
        n_gates = len(operations) if operations else None
        n_entangling_gates = None
        if operations:
            try:
                n_entangling_gates = sum(1 for op in operations if len(op.wires) > 1)
            except Exception:
                n_entangling_gates = None

        n_parameters = self._safe(lambda: len(tape.get_parameters()))
        if n_parameters is None:
            n_parameters = self._safe(lambda: int(tape.num_params))

        depth = self._safe(lambda: tape.graph.get_depth())

        return CircuitMetadata(
            n_qubits=n_qubits,
            depth=depth,
            n_parameters=n_parameters,
            n_gates=n_gates,
            n_entangling_gates=n_entangling_gates,
            ansatz_name=ansatz_name,
            initialization=initialization,
        )

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _safe(fn: Any) -> Any:
        """Run `fn()`, returning `None` on any exception (best-effort extraction)."""
        try:
            return fn()
        except Exception:
            return None

    def _construct_tape(self, parameters: Any) -> Any | None:
        """Best-effort tape construction for the attached QNode at `parameters`.

        Positional args: a `tuple` is unpacked as multiple QNode arguments
        (for multi-argument circuits); anything else (array, list, scalar)
        is treated as the QNode's single argument.
        """
        if self._qnode is None or qml is None:
            return None
        args = parameters if isinstance(parameters, tuple) else (parameters,)
        try:
            return qml.workflow.construct_tape(self._qnode)(*args)
        except Exception:
            pass
        # Fall back for older PennyLane releases without `qml.workflow.construct_tape`.
        try:
            self._qnode.construct(args, {})
            return getattr(self._qnode, "tape", None) or getattr(self._qnode, "qtape", None)
        except Exception:
            _logger.debug(
                "qml_observer: could not construct a tape from the attached QNode; "
                "circuit metadata will be omitted for this step.",
                exc_info=True,
            )
            return None

    def _resolve_gradient_method(self) -> str | None:
        if self._qnode is None:
            return None
        method = getattr(self._qnode, "diff_method", None)
        return method if isinstance(method, str) else None

    def _resolve_device_shots(self) -> int | None:
        if self._qnode is None:
            return None
        device = getattr(self._qnode, "device", None)
        return self._total_shots(getattr(device, "shots", None))

    def _shots_from_tape(self, tape: Any) -> int | None:
        return self._total_shots(getattr(tape, "shots", None))

    @staticmethod
    def _total_shots(shots_obj: Any) -> int | None:
        total = getattr(shots_obj, "total_shots", None)
        if isinstance(total, int) and not isinstance(total, bool):
            return total
        return None

    def _build_optimizer_metadata(self, gradient_method: str | None) -> OptimizerMetadata | None:
        if self._optimizer_name is None and gradient_method is None and self._learning_rate is None:
            return None
        return OptimizerMetadata(
            name=self._optimizer_name or "unknown",
            learning_rate=self._learning_rate,
            gradient_method=gradient_method,
        )
