"""QMLMonitor: the central public monitoring API.

Milestone 2 (Volume III): Issue #11 ("Implement QMLMonitor"), Issue #13
("Implement update lifecycle"), Issue #15 ("Implement monitor
reset/finalization"). Issue #12 (rolling state) is in `core/state.py` and
Issue #14 (run IDs) is in `core/run.py`; both are used here.

Scope note: `detectors` and `policy` are accepted per the blueprint's
constructor signature (Volume III) for forward compatibility, but the
statistics engine (Milestone 3), detector implementations, and the
`DiagnosisEngine` (Milestone 4) do not exist yet. Until then, `update()` and
`finish()` always return an `INSUFFICIENT_EVIDENCE` placeholder diagnosis via
`_default_diagnosis()` -- there is simply no detection logic to run yet. This
keeps the milestone 2 goal ("a manually instrumented QML training loop
works") honest: this class proves the event/lifecycle plumbing, not
detection quality. `_evaluate()` is the single seam Milestone 4 will replace
with real `DiagnosisEngine` delegation, without touching lifecycle code.

Failure semantics (addendum §1, "Fail-open with transparency"): any
exception raised while processing a step's data (schema validation,
gradient summarization, future detector/statistics calls) is caught inside
`update()`/`finish()`, logged at `warning` level with a full traceback, and
converted into a `degraded=True` diagnosis. The exception is never
propagated into the caller's training loop.

Thread-safety (addendum, Concurrency): `QMLMonitor` is **not** thread-safe.
See `core/state.py` for the full rationale; use one monitor per
process/rank for multi-process training.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from qml_observer.core.events import StepObservation
from qml_observer.core.run import generate_run_id, validate_run_id
from qml_observer.core.state import RunState
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType
from qml_observer.schemas.gradient import summarize_gradient
from qml_observer.schemas.optimizer import OptimizerMetadata
from qml_observer.schemas.training import TrainingEvent

_logger = logging.getLogger("qml_observer")

#: Supported `ActionPolicy` modes (blueprint Volume XI). The full policy
#: engine ships in Milestone 5; this class only validates the name for now
#: and applies the conservative subset described in `should_stop()`.
_VALID_POLICIES = frozenset({"log", "warn", "pause", "stop", "adaptive"})

_F = TypeVar("_F", bound=Callable[..., Any])


class QMLMonitor:
    """Central entry point for observing a variational QML training run.

    Not thread-safe: use one `QMLMonitor` per process/rank (see module
    docstring).

    Example (manual usage):
        >>> monitor = QMLMonitor()
        >>> for step in range(100):
        ...     diagnosis = monitor.update(step=step, loss=0.5)
        ...     if monitor.should_stop():
        ...         break
        >>> final = monitor.finish()

    Example (context manager):
        >>> with QMLMonitor() as monitor:
        ...     for step in range(100):
        ...         monitor.update(step=step, loss=0.5)
    """

    def __init__(
        self,
        detectors: list[Any] | None = None,
        policy: str = "warn",
        window_size: int = 100,
        run_id: str | None = None,
        reporter: Any | None = None,
        planned_steps: int | None = None,
    ) -> None:
        """Create a monitor for a new run.

        Args:
            detectors: Reserved for the Milestone 4 diagnosis engine.
                Accepted now for forward API compatibility; has no effect
                yet (see module docstring scope note).
            policy: Action policy mode. One of "log", "warn", "pause",
                "stop", "adaptive". The full `ActionPolicy` engine ships in
                Milestone 5; today this only affects `should_stop()`'s
                conservative built-in behavior.
            window_size: Maximum number of recent steps retained in the
                rolling window. Must be a positive int.
            run_id: Identifier for this run. Auto-generated via
                `generate_run_id()` if omitted.
            reporter: Optional object implementing the future
                `RunReporter` duck-type (`record_event`, `record_diagnosis`,
                `finalize`). If provided, `update()`/`finish()` will call it
                best-effort; failures in the reporter itself are logged,
                never raised (same fail-open policy as everything else).
            planned_steps: Optional total steps this run is expected to
                take, stored for the future compute-saved estimate
                (Milestone 7).

        Raises:
            ValueError: If `policy` is not a recognized mode, `window_size`
                is not a positive int, or `planned_steps` is negative.
        """
        if policy not in _VALID_POLICIES:
            raise ValueError(f"policy must be one of {sorted(_VALID_POLICIES)}, got {policy!r}")
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
            raise ValueError(f"window_size must be a positive int, got {window_size!r}")
        if planned_steps is not None:
            if not isinstance(planned_steps, int) or isinstance(planned_steps, bool):
                raise ValueError(f"planned_steps must be an int or None, got {planned_steps!r}")
            if planned_steps < 0:
                raise ValueError(f"planned_steps must be >= 0, got {planned_steps}")

        resolved_run_id = validate_run_id(run_id) if run_id is not None else generate_run_id()

        self._detectors = list(detectors) if detectors else []
        self._policy = policy
        self._window_size = window_size
        self._planned_steps = planned_steps
        self._reporter = reporter
        self._last_perf_time: float | None = None

        self._state = RunState(
            run_id=resolved_run_id,
            window_size=window_size,
            planned_steps=planned_steps,
        )

    # -- identity / read-only views -----------------------------------

    @property
    def run_id(self) -> str:
        """The identifier of the current run."""
        return self._state.run_id

    @property
    def policy(self) -> str:
        """The configured action policy mode."""
        return self._policy

    @property
    def state(self) -> RunState:
        """Read access to the underlying rolling state (advanced/testing use)."""
        return self._state

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Mark the run as started and begin wall-time tracking.

        Idempotent: calling `start()` again while already started is a
        no-op. Raises if the run has already been `finish()`-ed; call
        `reset()` first to begin a new run.
        """
        if self._state.finished:
            raise RuntimeError(
                f"Cannot start() run {self.run_id!r}: it has already finished. "
                "Call reset() to begin a new run."
            )
        if self._state.started:
            return
        self._state.started = True
        self._state.start_time = time.time()
        self._last_perf_time = time.perf_counter()

    def update(
        self,
        *,
        step: int,
        loss: float | None = None,
        gradients: Any | None = None,
        parameters: Any | None = None,
        circuit: CircuitMetadata | None = None,
        optimizer: OptimizerMetadata | None = None,
        shots: int | None = None,
    ) -> DiagnosisResult:
        """Record one training step and return the current diagnosis.

        Auto-starts the run (calling `start()`) if it hasn't been started
        yet, so the zero-config path (just calling `update()` in a loop)
        works without explicit lifecycle management.

        Per the fail-open policy (addendum §1), any exception raised while
        processing this step (schema validation, gradient summarization,
        etc.) is caught, logged, and converted into a `degraded=True`
        diagnosis -- it is never re-raised into the training loop.

        Args:
            step: Monotonically increasing step index.
            loss: Observed loss value, if available.
            gradients: Raw gradient array-like, if available. Summarized
                via `summarize_gradient` before storage.
            parameters: Raw parameter vector/snapshot, if available. Stored
                as-is (no schema exists for this yet).
            circuit: Circuit metadata for this step, if available.
            optimizer: Optimizer metadata for this step, if available.
            shots: Shot count for this step, if using shot-based execution.

        Returns:
            The `DiagnosisResult` for this step (currently always an
            `INSUFFICIENT_EVIDENCE` placeholder or a degraded result -- see
            module docstring scope note).

        Raises:
            RuntimeError: If called after `finish()` without an
                intervening `reset()`. This is a programmer-error misuse
                case, not a mid-training data issue, so it is not covered
                by the fail-open policy.
        """
        if self._state.finished:
            raise RuntimeError(
                f"Cannot update() run {self.run_id!r}: it has already finished. "
                "Call reset() to begin a new run."
            )
        if not self._state.started:
            self.start()

        try:
            diagnosis = self._process_update(
                step=step,
                loss=loss,
                gradients=gradients,
                parameters=parameters,
                circuit=circuit,
                optimizer=optimizer,
                shots=shots,
            )
        except Exception as exc:  # fail-open: never propagate into the training loop
            diagnosis = self._degrade(exc, step=step)

        self._state.latest_diagnosis = diagnosis
        return diagnosis

    def _process_update(
        self,
        *,
        step: int,
        loss: float | None,
        gradients: Any | None,
        parameters: Any | None,
        circuit: CircuitMetadata | None,
        optimizer: OptimizerMetadata | None,
        shots: int | None,
    ) -> DiagnosisResult:
        now_perf = time.perf_counter()
        wall_time = (
            None if self._last_perf_time is None else max(0.0, now_perf - self._last_perf_time)
        )
        self._last_perf_time = now_perf

        event = TrainingEvent(
            run_id=self.run_id,
            step=step,
            loss=loss,
            timestamp=time.time(),
            wall_time=wall_time,
        )
        gradient_snapshot = summarize_gradient(gradients) if gradients is not None else None

        observation = StepObservation(
            training_event=event,
            gradient=gradient_snapshot,
            circuit=circuit,
            optimizer=optimizer,
            shots=shots,
            parameters=parameters,
        )
        self._state.record(observation)

        if self._reporter is not None:
            self._reporter.record_event(event)

        return self._evaluate()

    def finish(self) -> DiagnosisResult:
        """Finalize the run and return the final diagnosis.

        Idempotent: calling `finish()` again after the run has already
        finished simply returns the previously computed final diagnosis
        without recomputing anything or calling the reporter again.

        Raises:
            RuntimeError: If the run was never started (no `start()` or
                `update()` call).
        """
        if not self._state.started:
            raise RuntimeError(
                f"Cannot finish() run {self.run_id!r}: it was never started. "
                "Call start() or update() first."
            )
        if self._state.finished:
            assert self._state.latest_diagnosis is not None  # set before finished=True below
            return self._state.latest_diagnosis

        try:
            diagnosis = self._evaluate()
        except Exception as exc:  # fail-open, same as update()
            diagnosis = self._degrade(exc, step=self._state.step_count)

        self._state.latest_diagnosis = diagnosis
        self._state.finished = True
        self._state.end_time = time.time()

        if self._reporter is not None:
            try:
                self._reporter.record_diagnosis(diagnosis)
                self._reporter.finalize()
            except Exception:
                _logger.warning(
                    "qml_observer: reporter failed while finishing run_id=%s",
                    self.run_id,
                    exc_info=True,
                )

        return diagnosis

    def reset(self, *, run_id: str | None = None) -> None:
        """Clear all recorded state and prepare the monitor for a new run.

        Args:
            run_id: Identifier for the new run. Auto-generated if omitted.
                Reusing the previous run's ID is allowed but discouraged,
                since it can conflate separate runs' logs/reports.
        """
        new_run_id = validate_run_id(run_id) if run_id is not None else generate_run_id()
        self._state.run_id = new_run_id
        self._state.reset()
        self._last_perf_time = None

    # -- diagnosis access -------------------------------------------------

    def should_stop(self) -> bool:
        """Whether the configured policy currently recommends stopping.

        Conservative by design (addendum §1): a `degraded` diagnosis never
        triggers a stop recommendation unless `policy="adaptive"` was
        explicitly chosen. The full `ActionPolicy` engine (Milestone 5)
        will replace this with richer per-severity logic.
        """
        diagnosis = self._state.latest_diagnosis
        if diagnosis is None:
            return False
        if diagnosis.degraded and self._policy != "adaptive":
            return False
        return self._policy == "stop" and diagnosis.severity == "critical"

    def latest_diagnosis(self) -> DiagnosisResult | None:
        """The most recent `DiagnosisResult`, or None if no step has run yet."""
        return self._state.latest_diagnosis

    def _evaluate(self) -> DiagnosisResult:
        """Produce a diagnosis from current state.

        Milestone 4 seam: this will delegate to
        `DiagnosisEngine(self._detectors).evaluate(...)` once the
        statistics engine and detectors exist. For now it always returns
        the placeholder diagnosis.
        """
        return self._default_diagnosis()

    def _default_diagnosis(self) -> DiagnosisResult:
        return DiagnosisResult(
            issue=IssueType.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            severity="info",
            evidence=[
                f"{self._state.step_count} step(s) recorded; no detectors are "
                "wired up yet (Milestone 4)."
            ],
            recommendations=[
                "Attach detectors via QMLMonitor(detectors=...) once the diagnosis engine ships."
            ],
        )

    def _degrade(self, exc: Exception, *, step: int) -> DiagnosisResult:
        _logger.warning(
            "qml_observer: failed to process step for run_id=%s step=%s (%s); "
            "training continues uninterrupted (fail-open policy).",
            self.run_id,
            step,
            type(exc).__name__,
            exc_info=True,
        )
        return DiagnosisResult(
            issue=IssueType.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            severity="warning",
            evidence=[],
            recommendations=[
                "Check the qml_observer logs for the underlying error; this "
                "step's data could not be processed."
            ],
            degraded=True,
            degraded_reason=f"{type(exc).__name__}: {exc}",
        )

    # -- context manager / decorator --------------------------------------

    def __enter__(self) -> QMLMonitor:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        if not self._state.finished:
            self.finish()
        return False  # never suppress exceptions from the caller's code

    def watch(self, func: _F) -> _F:
        """Decorator wrapping `func` in this monitor's context manager.

        A convenience wrapper, not a magic training-loop introspector (per
        the blueprint): it just calls `start()` before `func` runs and
        `finish()` after, regardless of how `func` itself calls `update()`.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]
