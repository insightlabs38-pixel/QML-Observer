"""Recovery engine interfaces: strategies, recommendations, and outcomes.

Milestone 13 (blueprint Volume XIV), Issue #90 ("Recovery strategy
interface"). Sequenced *after* Issue #90b (`actions/pause.py`) per
`future_milestones_plan.md`: recovery strategies exist to be proposed
*after* a run has been paused, not as a replacement for pausing.

This module intentionally mirrors two existing seams in the codebase
rather than inventing new vocabulary:

- `RecoveryStrategy` <-> `detectors.base.BaseDetector`: an individually
  pluggable, uniformly-driven unit (`detectors/base.py`'s docstring
  explains the same pattern for detectors; third-party recovery
  strategies are intended to plug in the same way third-party detectors
  will in Milestone 14, Issue #103).
- `RecoveryPlanner`/`RecoveryExecutor` <-> `DiagnosisEngine`/`ActionPolicy`:
  planning ("what could help?") stays separate from execution ("actually
  do it"), exactly as diagnosis ("what's wrong?") stays separate from
  action selection ("what do we do about it?") elsewhere in the project
  (blueprint's second architectural rule).

Core safety stance (blueprint Volume XIV, "Do not implement automatic
recovery until the detection system is validated" + plan.md §2's
non-invasive core principle): `RecoveryExecutor.apply()` can never reach
into a caller's real training loop and mutate its optimizer state
directly -- qml_observer does not own that loop. It can only invoke an
explicit, documented hook on a `training_state` object the caller
supplies, if that object chooses to expose one (see `RecoveryExecutor`'s
docstring). If no matching hook exists, `apply()` reports the
recommendation as *not* applied and explains what the caller should do
manually -- this is the same non-invasive contract `StopAction`/
`PauseAction` already use for their own "the caller must act" behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qml_observer.schemas._validation import (
    check_non_empty_str,
    check_non_negative_int,
    check_range,
    check_str_list,
    check_type,
)
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult
from qml_observer.schemas.gradient import GradientSnapshot
from qml_observer.schemas.optimizer import OptimizerMetadata


@dataclass
class RecoveryContext:
    """Contextual information a `RecoveryStrategy` needs beyond the diagnosis itself.

    Deliberately mirrors the fields already threaded through
    `core.events.StepObservation` (circuit/optimizer/shots/gradient)
    rather than inventing a parallel shape, so a `RecoveryContext` can be
    built directly from a `RunState`'s latest observation.

    Attributes:
        run_id: Identifier of the run being considered for recovery.
        step: The step index at which recovery is being considered
            (typically the step a `PauseAction` captured).
        circuit: Circuit metadata for the run, if known -- e.g. used by
            `ParameterReinitializationStrategy` to reason about the
            current `initialization` strategy, or a future
            depth-reduction strategy to reason about `depth`.
        optimizer: Optimizer metadata for the run, if known -- e.g. used
            by `LearningRateAdjustmentStrategy` to read the current
            `learning_rate`.
        shots: Shot count in use, if known -- used by
            `ShotBudgetAdjustmentStrategy`.
        gradient: The most recent `GradientSnapshot`, if known -- used by
            strategies that need to reason about gradient magnitude/SNR
            (e.g. `ShotBudgetAdjustmentStrategy`).
        planned_steps: Optional total planned steps for the run, carried
            over for strategies/executors that want it (e.g. deciding
            whether there's enough budget left to be worth recovering).
    """

    run_id: str
    step: int
    circuit: CircuitMetadata | None = None
    optimizer: OptimizerMetadata | None = None
    shots: int | None = None
    gradient: GradientSnapshot | None = None
    planned_steps: int | None = None

    def __post_init__(self) -> None:
        check_non_empty_str(self.run_id, "run_id")
        check_non_negative_int(self.step, "step")
        if self.circuit is not None:
            check_type(self.circuit, CircuitMetadata, "circuit")
        if self.optimizer is not None:
            check_type(self.optimizer, OptimizerMetadata, "optimizer")
        if self.shots is not None:
            check_non_negative_int(self.shots, "shots")
        if self.gradient is not None:
            check_type(self.gradient, GradientSnapshot, "gradient")
        if self.planned_steps is not None:
            check_non_negative_int(self.planned_steps, "planned_steps")


@dataclass
class RecoveryRecommendation:
    """A single ranked, concrete recovery recommendation from one strategy.

    This is the "candidate intervention" the blueprint's `RecoveryPlanner.
    recommend()` returns a list of (Volume XIV) -- distinct from
    `actions.base.ActionResult`, which describes the outcome of an
    already-selected `Action`, not a ranked candidate awaiting a decision.

    Attributes:
        strategy_name: The `RecoveryStrategy.name` that produced this
            recommendation (e.g. `"parameter_reinitialization"`).
        description: Human-readable summary suitable for a report/CLI/
            dashboard (e.g. "Reinitialize parameters using a
            reduced-domain strategy").
        priority: Ranking score in `[0, 1]` -- higher means more strongly
            recommended for the observed diagnosis. `RecoveryPlanner`
            sorts by this field, descending. Not the same axis as
            `DiagnosisResult.confidence` (which is confidence in the
            *diagnosis*, not in this particular *remedy*), though a
            strategy will typically derive one from the other.
        parameters: Concrete suggested parameter values (e.g.
            `{"learning_rate": 0.001}`), consumed by `RecoveryExecutor`
            when applying this recommendation via `hook_name`. May be
            empty for a recommendation that is guidance-only (no
            automatable hook exists yet).
        rationale: Human-readable evidence/reasoning strings backing this
            specific recommendation (distinct from `DiagnosisResult.
            evidence`, which backs the diagnosis, not the remedy).
        hook_name: Name of the method `RecoveryExecutor.apply()` will look
            for on a caller-supplied `training_state` object (e.g.
            `"reinitialize_parameters"`). `None` for a recommendation with
            no automatable hook (manual action only).
    """

    strategy_name: str
    description: str
    priority: float
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)
    hook_name: str | None = None

    def __post_init__(self) -> None:
        check_non_empty_str(self.strategy_name, "strategy_name")
        check_non_empty_str(self.description, "description")
        check_range(self.priority, 0.0, 1.0, "priority")
        check_type(self.parameters, dict, "parameters")
        check_str_list(self.rationale, "rationale")
        if self.hook_name is not None:
            check_non_empty_str(self.hook_name, "hook_name")


@dataclass
class RecoveryOutcome:
    """The result of `RecoveryExecutor.apply()` for one `RecoveryRecommendation`.

    Mirrors `actions.base.ActionResult`'s `executed`/`message` shape
    deliberately, for the same reason: callers/tests/reporting can treat
    "did this actually happen" uniformly across the action and recovery
    layers.

    Attributes:
        strategy_name: The `RecoveryRecommendation.strategy_name` this
            outcome is for.
        applied: Whether `training_state` actually exposed and accepted
            the hook this recommendation named. `False` covers both "no
            such hook exists" (manual action required) and "the hook
            itself raised" (fail-open, see `RecoveryExecutor.apply`).
        message: Human-readable explanation of what happened (or why it
            was not applied / what the caller should do manually).
    """

    strategy_name: str
    applied: bool
    message: str

    def __post_init__(self) -> None:
        check_non_empty_str(self.strategy_name, "strategy_name")
        check_type(self.applied, bool, "applied")
        check_type(self.message, str, "message")


class RecoveryStrategy(ABC):
    """Abstract interface every concrete recovery strategy must implement.

    Stateless by contract (unlike `BaseDetector`): a strategy's
    `propose()` is a pure function of `(diagnosis, context)`, since
    recovery is only ever considered after a pause/stop, not incorporated
    incrementally step-by-step the way detection is.
    """

    #: Stable identifier for this strategy, used in
    #: `RecoveryRecommendation.strategy_name`. Concrete subclasses must
    #: override this.
    name: str = "base"

    @abstractmethod
    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        """Whether this strategy is meaningful for `diagnosis.issue`.

        A cheap, side-effect-free filter `RecoveryPlanner` uses before
        even calling `propose()` -- e.g. a shot-budget strategy has
        nothing useful to say about `POSSIBLE_BARREN_PLATEAU` on an
        analytic (shots-free) run.
        """
        raise NotImplementedError

    @abstractmethod
    def propose(
        self, diagnosis: DiagnosisResult, context: RecoveryContext
    ) -> RecoveryRecommendation | None:
        """Propose a concrete recommendation for `diagnosis`, or `None`.

        Only ever called when `applies_to(diagnosis)` is `True`, but a
        strategy may still legitimately return `None` (e.g. it applies to
        this issue type in general but the available `context` is too
        sparse to propose concrete parameters).

        Should never raise for a well-formed `(diagnosis, context)` pair;
        a strategy that cannot safely reason about the given context
        should return `None` rather than propagating an exception. As a
        second line of defense consistent with the project's fail-open
        philosophy (addendum §1), `RecoveryPlanner.recommend()` also
        catches and logs any exception a strategy raises anyway, so one
        broken strategy can never prevent the others from being
        considered.
        """
        raise NotImplementedError
