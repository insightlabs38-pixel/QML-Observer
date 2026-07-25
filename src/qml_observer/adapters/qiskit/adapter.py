"""QiskitAdapter: the second real framework integration.

Milestone 8 (Volume X): Issue #58 ("Implement Qiskit adapter"), Issue #59
("Implement callback integration"), Issue #60 ("Normalize optimizer
metadata").

Per the blueprint's core architectural rule (same as `PennyLaneAdapter`,
Milestone 6), this adapter *observes* Qiskit training -- it does not
reimplement Qiskit's optimization or gradient machinery. The user's
training loop (a hand-written `QuantumCircuit` + `Estimator`/`Sampler`
primitive loop, a `qiskit_algorithms`-style optimizer, or a
`qiskit-machine-learning` trainer such as `VQC`/`NeuralNetworkClassifier`)
still does the actual optimization; this adapter is responsible only for:

  1. Forwarding already-computed `loss`/`gradients`/`parameters` to
     `QMLMonitor.update()` (like `GenericAdapter`, but Qiskit-aware).
  2. Filling in `CircuitMetadata` automatically from an attached
     `QuantumCircuit` (or a trainer object exposing one), and
     `OptimizerMetadata` from an attached optimizer object, so the user
     doesn't have to build those by hand every step.
  3. Normalizing the handful of shapes Qiskit optimizer/trainer callbacks
     actually use in practice into a single call into the monitor.

Blueprint note (Volume X): "Because Qiskit APIs vary across components,
isolate version-specific logic inside this adapter." That variance shows
up in two places this module handles explicitly:

  - **Callback signatures.** `qiskit_machine_learning` trainers
    (`VQC`, `NeuralNetworkClassifier`, `NeuralNetworkRegressor`, ...) call
    back as `callback(weights, obj_func_eval)`; `qiskit_algorithms`-style
    optimizers with native callback support (e.g. `SPSA`) call back as
    `callback(nfev, params, fval, stepsize, accepted)`; plain
    `scipy.optimize.minimize`-style callbacks (used by some
    gradient-based optimizers) call back as `callback(xk)` only, with no
    loss value at all. `callback()` below detects and normalizes all of
    these by argument count, alongside the blueprint's own sketched
    3-argument `callback(iteration, parameters, loss)` form for manual
    use.
  - **Optimizer metadata.** Different optimizer classes expose their
    configuration under different attribute names (`learning_rate` for
    `SPSA`, `lr` for `ADAM`, nothing at all for gradient-free optimizers
    like `COBYLA`). `normalize_optimizer_metadata()` handles this
    best-effort, via each optimizer's own `.settings` dict rather than
    hardcoding attribute access, and never raises on an unrecognized
    optimizer type.

Every piece of metadata extraction here is best-effort, consistent with
`CircuitMetadata`/`OptimizerMetadata`'s "every field but the essentials is
optional" design and this project's fail-open policy (addendum §1): if a
given Qiskit/qiskit-machine-learning version doesn't expose something the
way we expect, that field is simply left `None` rather than raising.
`QMLMonitor.update()` additionally wraps the whole step in its own
fail-open try/except, so even a totally unexpected internal change
degrades gracefully instead of crashing the training loop.
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
    import qiskit
except ImportError as _exc:  # pragma: no cover - exercised only without qiskit installed
    qiskit = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = _exc
else:
    _IMPORT_ERROR = None


def _require_qiskit() -> None:
    if qiskit is None:
        raise ImportError(
            "QiskitAdapter requires the optional 'qiskit' dependency. "
            "Install it with `pip install qml-observer[qiskit]` or "
            "`pip install qiskit>=1.0`."
        ) from _IMPORT_ERROR


#: Best-effort mapping from optimizer class name to a `gradient_method`
#: label for `normalize_optimizer_metadata()`. Deliberately small and
#: conservative: an unrecognized optimizer name simply leaves
#: `gradient_method=None` rather than guessing.
_KNOWN_GRADIENT_METHODS: dict[str, str] = {
    "SPSA": "spsa-approximation",
    "QNSPSA": "spsa-approximation",
    "COBYLA": "gradient-free",
    "NELDER_MEAD": "gradient-free",
    "NelderMead": "gradient-free",
    "POWELL": "gradient-free",
    "GradientDescent": "finite-difference",
    "ADAM": "finite-difference",
    "AQGD": "finite-difference",
}

#: Candidate attribute names for a numeric learning rate / step size across
#: `Optimizer.settings` dicts of different Qiskit optimizer classes (Issue
#: #60): `SPSA` uses "learning_rate", `ADAM` uses "lr", gradient-free
#: optimizers (e.g. `COBYLA`) expose neither.
_LEARNING_RATE_KEYS: tuple[str, ...] = ("learning_rate", "lr", "eta", "step_size")


class QiskitAdapter:
    """Adapter observing a Qiskit `QuantumCircuit`/trainer's training loop.

    Example (manual loop with a `QuantumCircuit` + `Estimator`):
        >>> from qiskit.circuit.library import efficient_su2
        >>> from qml_observer import QMLMonitor
        >>> from qml_observer.adapters.qiskit.adapter import QiskitAdapter
        >>>
        >>> ansatz = efficient_su2(4, reps=2)
        >>> monitor = QMLMonitor()
        >>> adapter = QiskitAdapter(monitor, ansatz, optimizer_name="COBYLA")
        >>> for step in range(200):
        ...     loss, gradients = my_energy_and_gradient(params)
        ...     diagnosis = adapter.record_step(step, loss, gradients, parameters=params)

    Example (callback integration with a `qiskit_algorithms` optimizer):
        >>> from qiskit_algorithms.optimizers import SPSA
        >>> adapter = QiskitAdapter(monitor, ansatz, optimizer=SPSA(maxiter=200))
        >>> result = SPSA(maxiter=200).minimize(cost_fn, x0, jac=grad_fn)
        >>> # or, wired directly into the optimizer's own callback hook:
        >>> opt = SPSA(maxiter=200, callback=adapter.callback)

    Example (callback integration with a `qiskit-machine-learning` trainer):
        >>> from qiskit_machine_learning.algorithms.classifiers import VQC
        >>> adapter = QiskitAdapter(monitor)
        >>> vqc = VQC(feature_map=fm, ansatz=ansatz, optimizer=COBYLA(maxiter=100),
        ...           callback=adapter.callback)
        >>> adapter.attach(vqc)
        >>> vqc.fit(X, y)
    """

    def __init__(
        self,
        monitor: QMLMonitor,
        circuit: Any | None = None,
        *,
        optimizer: Any | None = None,
        optimizer_name: str | None = None,
        learning_rate: float | None = None,
        gradient_method: str | None = None,
        shots: int | None = None,
    ) -> None:
        """Create an adapter wrapping `monitor`.

        Args:
            monitor: The `QMLMonitor` to forward recorded steps to.
            circuit: Optional `QuantumCircuit` (or trainer object exposing
                one, e.g. a `VQC`/`NeuralNetworkClassifier`) to `attach()`
                immediately.
            optimizer: Optional live optimizer object (e.g. a
                `qiskit_algorithms.optimizers.Optimizer` instance) used to
                populate `OptimizerMetadata` via `normalize_optimizer_metadata()`.
            optimizer_name: Explicit optimizer name, overriding whatever
                would otherwise be inferred from `optimizer`. Useful when
                no live optimizer object is available (e.g. a raw
                `scipy.optimize.minimize` callable).
            learning_rate: Explicit learning rate, overriding whatever
                would otherwise be inferred from `optimizer.settings`.
            gradient_method: Explicit gradient computation method (e.g.
                "parameter-shift", "spsa-approximation"), overriding the
                best-effort inference in `normalize_optimizer_metadata()`.
            shots: Default shot count to report when a step doesn't supply
                one explicitly (e.g. a fixed `Estimator`/`Sampler` shot
                budget configured once for the whole run).

        Raises:
            ImportError: If the `qiskit` package is not installed.
            TypeError: If `monitor` is not a `QMLMonitor` instance.
        """
        _require_qiskit()
        if not isinstance(monitor, QMLMonitor):
            raise TypeError(f"monitor must be a QMLMonitor, got {type(monitor)!r}")

        self.monitor = monitor
        self._optimizer = optimizer
        self._optimizer_name = optimizer_name
        self._learning_rate = learning_rate
        self._gradient_method = gradient_method
        self._default_shots = shots
        self._circuit: Any | None = None
        self._pending_gradients: Any | None = None
        self._callback_iteration = 0
        if circuit is not None:
            self.attach(circuit)

    # -- attach/detach lifecycle (blueprint Volume X) ----------------------

    def attach(self, training_object: Any) -> QiskitAdapter:
        """Attach a `QuantumCircuit` (or trainer exposing one) as the source
        of circuit metadata.

        Once attached, `record_step()`/`callback()` will automatically
        populate `CircuitMetadata` from the circuit on every call.

        Args:
            training_object: A `qiskit.circuit.QuantumCircuit`, or a
                trainer-like object exposing one via a `.circuit` attribute
                (e.g. `VQC`/`NeuralNetworkClassifier`/`NeuralNetworkRegressor`,
                or their underlying `NeuralNetwork`) or a `.ansatz`
                attribute.

        Returns:
            `self`, to allow `QiskitAdapter(monitor).attach(circuit)`.

        Raises:
            TypeError: If `training_object` isn't a `QuantumCircuit` and
                doesn't expose one via `.circuit`/`.ansatz`.
        """
        circuit = self._resolve_circuit(training_object)
        if circuit is None:
            raise TypeError(
                "attach() expects a QuantumCircuit, or a trainer-like object "
                "exposing one via `.circuit` or `.ansatz` (e.g. VQC, "
                "NeuralNetworkClassifier), got "
                f"{type(training_object)!r}"
            )
        self._circuit = circuit
        return self

    def detach(self) -> None:
        """Detach the current circuit. `record_step()`/`callback()` still
        work, but without automatic circuit metadata until `attach()` is
        called again."""
        self._circuit = None

    @property
    def attached(self) -> bool:
        """Whether a circuit is currently attached."""
        return self._circuit is not None

    def _resolve_circuit(self, training_object: Any) -> Any | None:
        if qiskit is not None and isinstance(training_object, qiskit.circuit.QuantumCircuit):
            return training_object
        circuit = getattr(training_object, "circuit", None)
        if circuit is not None and qiskit is not None:
            if isinstance(circuit, qiskit.circuit.QuantumCircuit):
                return circuit
        ansatz = getattr(training_object, "ansatz", None)
        if ansatz is not None and qiskit is not None:
            if isinstance(ansatz, qiskit.circuit.QuantumCircuit):
                return ansatz
        return None

    # -- recording -----------------------------------------------------------

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
        """Record one Qiskit training step.

        Args:
            step: Monotonically increasing step index.
            loss: Observed loss/energy value for this step, if available.
            gradients: Raw gradient array, if available. If omitted and a
                gradient was previously cached via `record_gradient()`,
                the cached value is used instead (and then cleared).
                Forwarded as-is to `QMLMonitor.update()`.
            parameters: The parameter values the circuit was evaluated at
                this step. Stored as-is; not used for circuit-metadata
                extraction (unlike PennyLane's tape, a bound `QuantumCircuit`
                doesn't need re-construction from `parameters` -- an
                attached circuit's *structure* is static regardless of the
                parameter values it's bound to).
            shots: Explicit shot count for this step, overriding the
                adapter's default (from `__init__`).
            ansatz_name: Name of the ansatz in use, if the caller wants to
                record it. Not auto-detected -- a generic `QuantumCircuit`
                doesn't generically expose an "ansatz name" beyond its
                gate/qubit/parameter counts (same reasoning as
                `PennyLaneAdapter.record_step`).
            initialization: Name of the parameter initialization strategy
                in use, if the caller wants to record it. Not auto-detected,
                for the same reason as `ansatz_name`.

        Returns:
            The `DiagnosisResult` for this step, exactly as returned by
            `QMLMonitor.update()` (including its fail-open `degraded`
            behavior).
        """
        if gradients is None and self._pending_gradients is not None:
            gradients = self._pending_gradients
        self._pending_gradients = None

        circuit_meta: CircuitMetadata | None = None
        if self._circuit is not None:
            circuit_meta = self.extract_circuit_metadata(
                self._circuit, ansatz_name=ansatz_name, initialization=initialization
            )

        resolved_shots = shots if shots is not None else self._default_shots
        optimizer_meta = self.normalize_optimizer_metadata(
            self._optimizer,
            name=self._optimizer_name,
            learning_rate=self._learning_rate,
            gradient_method=self._gradient_method,
        )

        return self.monitor.update(
            step=step,
            loss=loss,
            gradients=gradients,
            parameters=parameters,
            circuit=circuit_meta,
            optimizer=optimizer_meta,
            shots=resolved_shots,
        )

    def record_gradient(self, gradients: Any) -> None:
        """Cache a gradient array to be attached to the *next* recorded step.

        Most Qiskit optimizer/trainer callbacks (see `callback()`) do not
        carry gradient information themselves -- Qiskit's own gradient
        machinery (e.g. a `qiskit.primitives.BaseEstimatorGradient`) is
        typically invoked separately from the optimizer's callback hook.
        Call this right after computing a gradient (e.g. from a custom
        gradient wrapper around the cost function) and it will be
        consumed by the next `record_step()`/`callback()` call, then
        cleared -- whichever happens first.

        Args:
            gradients: Raw gradient array-like for the upcoming step.
        """
        self._pending_gradients = gradients

    # -- callback integration (Issue #59) ------------------------------------

    def callback(self, *args: Any) -> DiagnosisResult:
        """Callback compatible with common Qiskit optimizer/trainer signatures.

        Pass this method directly as the `callback=` argument to a
        `qiskit-machine-learning` trainer (`VQC`, `NeuralNetworkClassifier`,
        `NeuralNetworkRegressor`, ...) or a `qiskit_algorithms`-style
        optimizer that supports one (e.g. `SPSA`), or call it manually.
        The argument shape is detected by position count and normalized
        into a single `record_step()` call -- see the module docstring for
        why this variance exists. Supported shapes:

          - `callback(parameters)` -- plain `scipy.optimize.minimize`-style
            (no loss reported; iteration is auto-incremented).
          - `callback(weights, obj_func_eval)` -- `qiskit-machine-learning`
            trainer style (iteration is auto-incremented).
          - `callback(iteration, parameters, loss)` -- the blueprint's own
            manual/generic form (Volume X).
          - `callback(nfev, params, fval, stepsize, accepted)` --
            `qiskit_algorithms` `SPSA`-style (iteration is auto-incremented
            from the callback call count, not `nfev`, since `nfev` counts
            function evaluations rather than optimizer steps).

        Returns:
            The `DiagnosisResult` for this step, exactly as `record_step()`
            returns.

        Raises:
            TypeError: If `args` doesn't match any recognized shape.
        """
        iteration, parameters, loss = self._normalize_callback_args(args)
        return self.record_step(iteration, loss, parameters=parameters)

    def _normalize_callback_args(self, args: tuple[Any, ...]) -> tuple[int, Any, float | None]:
        n = len(args)
        if n == 1:
            (parameters,) = args
            return self._next_iteration(), parameters, None
        if n == 2:
            parameters, loss = args
            return self._next_iteration(), parameters, self._as_float(loss)
        if n == 3:
            iteration, parameters, loss = args
            return int(iteration), parameters, self._as_float(loss)
        if n == 5:
            _nfev, parameters, loss, _stepsize, _accepted = args
            return self._next_iteration(), parameters, self._as_float(loss)
        raise TypeError(
            f"callback() got an unsupported number of positional arguments ({n}); "
            "expected 1 (scipy-style `xk`), 2 (qiskit-machine-learning-style "
            "`weights, obj_func_eval`), 3 (blueprint-style `iteration, "
            "parameters, loss`), or 5 (qiskit_algorithms SPSA-style `nfev, "
            "params, fval, stepsize, accepted`)."
        )

    def _next_iteration(self) -> int:
        iteration = self._callback_iteration
        self._callback_iteration += 1
        return iteration

    @staticmethod
    def _as_float(value: Any) -> float | None:
        return None if value is None else float(value)

    # -- circuit metadata extraction (part of Issue #58) ---------------------

    def extract_circuit_metadata(
        self,
        circuit: Any,
        *,
        ansatz_name: str | None = None,
        initialization: str | None = None,
    ) -> CircuitMetadata:
        """Build a `CircuitMetadata` from a Qiskit `QuantumCircuit`.

        Every field is extracted defensively: if the installed Qiskit
        version doesn't expose a given piece of information the way this
        method expects, that field is left `None` rather than raising, per
        `CircuitMetadata`'s own "every field is optional" contract.

        Args:
            circuit: A `qiskit.circuit.QuantumCircuit` (e.g. an ansatz, or
                a full trainer circuit including a feature map).
            ansatz_name: Passed straight through to the resulting
                `CircuitMetadata.ansatz_name` (see `record_step` docstring
                for why this isn't auto-detected).
            initialization: Passed straight through to
                `CircuitMetadata.initialization` (same reasoning).

        Returns:
            A `CircuitMetadata` populated with whatever fields could be
            extracted from `circuit`.
        """
        n_qubits = self._safe(lambda: circuit.num_qubits)
        depth = self._safe(lambda: circuit.depth())
        n_parameters = self._safe(lambda: circuit.num_parameters)
        n_gates = self._safe(lambda: circuit.size())

        n_entangling_gates = self._safe(
            lambda: sum(1 for instr in circuit.data if len(instr.qubits) > 1)
        )

        return CircuitMetadata(
            n_qubits=n_qubits,
            depth=depth,
            n_parameters=n_parameters,
            n_gates=n_gates,
            n_entangling_gates=n_entangling_gates,
            ansatz_name=ansatz_name,
            initialization=initialization,
        )

    # -- optimizer metadata normalization (Issue #60) ------------------------

    @classmethod
    def normalize_optimizer_metadata(
        cls,
        optimizer: Any | None = None,
        *,
        name: str | None = None,
        learning_rate: float | None = None,
        gradient_method: str | None = None,
    ) -> OptimizerMetadata | None:
        """Normalize a Qiskit optimizer object into `OptimizerMetadata`.

        Qiskit optimizer objects vary widely in how they expose their own
        configuration: `qiskit_algorithms`/`qiskit-machine-learning`
        `Optimizer` subclasses expose a `.settings` dict, but with
        inconsistent keys across implementations (e.g. `SPSA` uses
        `"learning_rate"`, `ADAM` uses `"lr"`); gradient-free optimizers
        (e.g. `COBYLA`) expose no learning rate at all; and a raw
        `scipy.optimize.minimize`-compatible `Minimizer` callable exposes
        no configuration whatsoever. This method is deliberately
        best-effort and never raises on an unrecognized `optimizer`: any
        explicitly-passed `name`/`learning_rate`/`gradient_method` always
        takes precedence over whatever would otherwise be inferred from
        `optimizer.settings`.

        Args:
            optimizer: A live optimizer object (e.g. a
                `qiskit_algorithms.optimizers.Optimizer` instance), or
                `None` if only the explicit keyword arguments should be
                used.
            name: Explicit optimizer name, overriding `type(optimizer).__name__`.
            learning_rate: Explicit learning rate, overriding whatever
                would otherwise be read from `optimizer.settings`.
            gradient_method: Explicit gradient computation method,
                overriding the best-effort inference from the optimizer's
                class name (see `_KNOWN_GRADIENT_METHODS`).

        Returns:
            An `OptimizerMetadata`, or `None` if no optimizer information
            (object or explicit keyword arguments) was given at all --
            matching `PennyLaneAdapter`'s convention of omitting optimizer
            metadata entirely rather than reporting an all-`"unknown"`
            placeholder when nothing is actually known.
        """
        resolved_name = name
        resolved_lr = learning_rate
        resolved_grad_method = gradient_method

        if optimizer is not None:
            if resolved_name is None:
                resolved_name = type(optimizer).__name__
            if resolved_lr is None:
                resolved_lr = cls._extract_learning_rate(optimizer)
            if resolved_grad_method is None:
                resolved_grad_method = _KNOWN_GRADIENT_METHODS.get(resolved_name)

        if resolved_name is None and resolved_lr is None and resolved_grad_method is None:
            return None

        return OptimizerMetadata(
            name=resolved_name or "unknown",
            learning_rate=resolved_lr,
            gradient_method=resolved_grad_method,
        )

    @staticmethod
    def _extract_learning_rate(optimizer: Any) -> float | None:
        settings = getattr(optimizer, "settings", None)
        if not isinstance(settings, dict):
            return None
        for key in _LEARNING_RATE_KEYS:
            value = settings.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
        return None

    @staticmethod
    def _safe(fn: Any) -> Any:
        """Run `fn()`, returning `None` on any exception (best-effort extraction)."""
        try:
            return fn()
        except Exception:
            return None
