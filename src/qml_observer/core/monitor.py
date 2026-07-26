"""QMLMonitor: the central public monitoring API.

Milestone 2 (Volume III): Issue #11 ("Implement QMLMonitor"), Issue #13
("Implement update lifecycle"), Issue #15 ("Implement monitor
reset/finalization"). Issue #12 (rolling state) is in `core/state.py` and
Issue #14 (run IDs) is in `core/run.py`; both are used here.

Milestone 4 update: `detectors` is now wired to a real `DiagnosisEngine`
(Issue #29). When no detectors are configured, `update()`/`finish()` still
return the `INSUFFICIENT_EVIDENCE` placeholder via `_default_diagnosis()`
exactly as in Milestone 2/3 -- an empty monitor proves the event/lifecycle
plumbing only. `_evaluate()` remains the single seam: it now delegates to
`DiagnosisEngine.evaluate()` whenever `detectors` is non-empty, without any
other change to lifecycle code.

Milestone 5 update (Issues #38-#39, "Add warn mode" / "Add stop mode"):
`policy` is now backed by a real `ActionPolicy` (`actions/policies.py`).
Each `update()`/`finish()` call runs the policy against that step's
diagnosis, so `"warn"` genuinely emits terminal alerts and `"stop"`
genuinely arms a `StopAction` -- this replaces the placeholder
`should_stop()` logic used before Milestone 5 shipped the action layer.
`should_stop()` itself stays a pure, side-effect-free recomputation from
`state.latest_diagnosis` (via `ActionPolicy.select_action()`), not a
read of prior side-effect state, so it keeps working whether the caller
drives diagnoses through `update()` or sets `state.latest_diagnosis`
directly (as many existing unit tests do).

Failure semantics (addendum §1, "Fail-open with transparency"): any
exception raised while processing a step's data (schema validation,
gradient summarization, detector/statistics calls) is caught inside
`update()`/`finish()`, logged at `warning` level with a full traceback, and
converted into a `degraded=True` diagnosis. The exception is never
propagated into the caller's training loop. This same guarantee now also
covers the action layer (Issue #40, "Test action safety"): even though
every built-in `Action` already catches its own internal errors (see
`actions/base.py`), a pathological custom `Action`/`ActionPolicy` that
raises anyway is still caught here and logged, never propagated.

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

from qml_observer.actions.base import ActionResult
from qml_observer.actions.policies import VALID_MODES, ActionPolicy, StopAction
from qml_observer.core.events import StepObservation
from qml_observer.core.run import generate_run_id, validate_run_id
from qml_observer.core.state import RunState
from qml_observer.detectors.base import BaseDetector
from qml_observer.diagnosis.engine import DiagnosisEngine
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType
from qml_observer.schemas.gradient import summarize_gradient
from qml_observer.schemas.optimizer import OptimizerMetadata
from qml_observer.schemas.training import TrainingEvent

_logger = logging.getLogger("qml_observer")

#: Supported `ActionPolicy` modes (blueprint Volume XI), re-exported from
#: `actions.policies` so this module has a single source of truth for
#: valid policy strings.
_VALID_POLICIES = VALID_MODES

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
        detectors: list[BaseDetector] | None = None,
        policy: str = "warn",
        window_size: int = 100,
        run_id: str | None = None,
        reporter: Any | None = None,
        planned_steps: int | None = None,
        action_policy: ActionPolicy | None = None,
        telemetry_collector: Any | None = None,
        telemetry_framework: str | None = None,
    ) -> None:
        """Create a monitor for a new run.

        Args:
            detectors: `BaseDetector` instances to run each step (e.g.
                `BarrenPlateauDetector`, `StagnationDetector`,
                `ConvergenceDetector`). If omitted or empty, `update()`/
                `finish()` return the `INSUFFICIENT_EVIDENCE` placeholder
                diagnosis, exactly as before Milestone 4.
            policy: Action policy mode. One of "log", "warn", "pause",
                "stop", "adaptive" (see `actions.policies.ActionPolicy`).
                Ignored if `action_policy` is given (the monitor's
                `.policy` then reflects `action_policy.mode` instead).
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
            action_policy: A pre-configured `ActionPolicy` to use instead
                of building one from `policy`. Use this for advanced
                configuration not expressible via the `policy` string alone
                (e.g. `mode="adaptive"` with `allow_stop_on_degraded=True`,
                or injecting custom/test `Action` instances).
            telemetry_collector: Optional `qml_observer.telemetry.TelemetryCollector`.
                Fully opt-in (addendum §5): even when provided, nothing is
                collected or sent unless the user has separately enabled
                telemetry via `qml_observer.telemetry.enable()` or
                `qml-observer telemetry enable`. If omitted (the default),
                no telemetry code runs at all.
            telemetry_framework: Optional framework label (e.g.
                `"pennylane"`, `"qiskit"`) included in the anonymized
                telemetry record, if telemetry is enabled.

        Raises:
            ValueError: If `policy` is not a recognized mode, `window_size`
                is not a positive int, or `planned_steps` is negative.
            TypeError: If any element of `detectors` is not a `BaseDetector`,
                or `action_policy` is given and is not an `ActionPolicy`.
        """
        if action_policy is not None:
            if not isinstance(action_policy, ActionPolicy):
                raise TypeError(
                    f"action_policy must be an ActionPolicy, got {type(action_policy)!r}"
                )
            policy = action_policy.mode
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
        for i, d in enumerate(self._detectors):
            if not isinstance(d, BaseDetector):
                raise TypeError(f"detectors[{i}] must be a BaseDetector, got {type(d)!r}")
        self._diagnosis_engine = DiagnosisEngine(self._detectors) if self._detectors else None
        self._policy = policy
        self._action_policy = (
            action_policy if action_policy is not None else ActionPolicy(mode=policy)
        )
        self._last_action_result: ActionResult | None = None
        self._window_size = window_size
        self._planned_steps = planned_steps
        self._reporter = reporter
        self._telemetry_collector = telemetry_collector
        self._telemetry_framework = telemetry_framework
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
    def action_policy(self) -> ActionPolicy:
        """The `ActionPolicy` backing this monitor's `policy` mode.

        Exposed for advanced use: e.g. inspecting
        `monitor.action_policy.stop_action.last_diagnosis`, or checking
        `monitor.action_policy.allow_stop_on_degraded`.
        """
        return self._action_policy

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
        self._last_action_result = self._run_action_policy(diagnosis)
        return diagnosis

    def _run_action_policy(self, diagnosis: DiagnosisResult) -> ActionResult | None:
        """Run `self._action_policy` against `diagnosis`, fail-open.

        Every built-in `Action` already catches its own internal errors
        (`actions/base.py`), so this is a defensive second layer: if a
        custom `Action`/`ActionPolicy` still raises, the failure is logged
        and swallowed here rather than propagated into the caller's
        training loop (Issue #40, "Test action safety"), exactly like
        `_degrade()` does for detector/statistics failures.
        """
        try:
            return self._action_policy.execute(diagnosis)
        except Exception:
            _logger.warning(
                "qml_observer: action policy failed for run_id=%s step=%s; "
                "training continues uninterrupted (fail-open policy).",
                self.run_id,
                self._state.step_count,
                exc_info=True,
            )
            return None

    def latest_action_result(self) -> ActionResult | None:
        """The `ActionResult` from the most recent `update()`/`finish()` call.

        `None` before any step has run, or if the action policy itself
        failed on the last call (see `_run_action_policy`).
        """
        return self._last_action_result

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
        self._last_action_result = self._run_action_policy(diagnosis)

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

        if self._telemetry_collector is not None:
            try:
                self._maybe_emit_telemetry(diagnosis)
            except Exception:
                # Fail-open (addendum §1): telemetry must never affect the
                # training loop or its diagnosis, even if misconfigured.
                _logger.warning(
                    "qml_observer: telemetry submission failed for run_id=%s",
                    self.run_id,
                    exc_info=True,
                )

        return diagnosis

    def _maybe_emit_telemetry(self, diagnosis: DiagnosisResult) -> None:
        """Build and hand off an anonymized telemetry record, if enabled.

        A complete no-op unless the user has separately opted in via
        `qml_observer.telemetry.enable()` (checked inside the collector
        itself) -- this method only ever *offers* a record, it never
        forces transmission. See `qml_observer/telemetry/` and
        `docs/development/telemetry.md`.
        """
        from qml_observer.telemetry.schema import (
            build_telemetry_record,
            extract_detector_thresholds,
        )

        thresholds: dict[str, float] = {}
        for detector in self._detectors:
            thresholds.update(extract_detector_thresholds(detector))

        latest = self._state.latest_observation
        n_qubits = latest.circuit.n_qubits if latest is not None and latest.circuit else None
        never_diagnosed = diagnosis.issue in (IssueType.HEALTHY, IssueType.INSUFFICIENT_EVIDENCE)

        record = build_telemetry_record(
            detector_names=[type(d).__name__ for d in self._detectors],
            thresholds=thresholds,
            issue=diagnosis.issue.value,
            confidence=diagnosis.confidence,
            framework=self._telemetry_framework,
            n_qubits=n_qubits,
            detection_latency_steps=(None if never_diagnosed else self._state.step_count),
        )
        assert self._telemetry_collector is not None  # only called when set, see finish()
        self._telemetry_collector.maybe_collect(record)

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
        self._last_action_result = None
        if self._diagnosis_engine is not None:
            self._diagnosis_engine.reset()
        self._action_policy.reset()

    # -- diagnosis access -------------------------------------------------

    def should_stop(self) -> bool:
        """Whether the configured policy currently recommends stopping.

        A pure, side-effect-free recomputation of
        `self._action_policy.select_action(state.latest_diagnosis)`: it
        does not depend on whether `update()`/`finish()` (and therefore
        the action policy's own `execute()`) has actually run for the
        current `state.latest_diagnosis` -- so it gives a consistent
        answer whether diagnoses are produced via `update()` or set
        directly on `state.latest_diagnosis` (e.g. in tests). Conservative
        by design (addendum §1, enforced inside `ActionPolicy`): a
        `degraded` diagnosis never recommends a stop unless the monitor's
        `action_policy` was explicitly configured with
        `mode="adaptive", allow_stop_on_degraded=True`.
        """
        diagnosis = self._state.latest_diagnosis
        if diagnosis is None:
            return False
        return isinstance(self._action_policy.select_action(diagnosis), StopAction)

    def latest_diagnosis(self) -> DiagnosisResult | None:
        """The most recent `DiagnosisResult`, or None if no step has run yet."""
        return self._state.latest_diagnosis

    def _evaluate(self) -> DiagnosisResult:
        """Produce a diagnosis from current state.

        Delegates to `DiagnosisEngine.evaluate()` when one or more
        detectors are configured; otherwise returns the
        `INSUFFICIENT_EVIDENCE` placeholder (no detection logic to run).
        """
        if self._diagnosis_engine is None:
            return self._default_diagnosis()
        observation = self._state.latest_observation
        assert observation is not None  # _evaluate is only called after a record()
        return self._diagnosis_engine.evaluate(observation, self._state)

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

        Note: like reusing a plain `with monitor:` block, calling the
        decorated function a second time without an intervening
        `monitor.reset()` raises `RuntimeError`, since the run has already
        finished. Call `reset()` between invocations if you need to watch
        the same function multiple times.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]
