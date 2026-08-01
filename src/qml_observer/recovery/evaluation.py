"""RecoveryEvaluator: decide whether an applied recovery attempt helped.

Milestone 13, Issue #96 ("Recovery evaluation"). Implements plan.md §22's
recommended recovery-engine behavior: "test a small number of changes,
resume only if health metrics improve."

Scope note: qml_observer does not run the training loop, so it cannot
itself decide *when* "after" has been observed long enough to judge --
that stays the caller's responsibility (typically: resume for some number
of steps after `RecoveryExecutor.apply()`, then call `monitor.
latest_diagnosis()` again and pass both diagnoses here). `RecoveryEvaluator`
only implements the comparison itself: given a "before" and "after"
`DiagnosisResult` for the same run, was training's health better,
unchanged, or worse afterward -- and, per plan.md §22, should the change
be kept or rolled back.
"""

from __future__ import annotations

from dataclasses import dataclass

from qml_observer.integrations.payloads import SEVERITY_RANK
from qml_observer.schemas._validation import check_non_empty_str, check_range, check_type
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType

#: Issue types considered a "good" training state -- reaching one of
#: these from anything else is unambiguous improvement, regardless of
#: severity/confidence comparisons.
_GOOD_ISSUES = frozenset({IssueType.HEALTHY, IssueType.CONVERGED})

#: Default minimum confidence drop (same issue, same severity) required
#: to call it "improved" rather than "unchanged, within noise".
_DEFAULT_CONFIDENCE_IMPROVEMENT_THRESHOLD = 0.1


@dataclass
class RecoveryEvaluationResult:
    """The outcome of comparing a "before" and "after" `DiagnosisResult`.

    Attributes:
        strategy_name: Identifies which recovery attempt this evaluation
            is for (typically `RecoveryRecommendation.strategy_name` /
            `RecoveryOutcome.strategy_name`).
        before: The diagnosis observed before the recovery attempt (e.g.
            the one that triggered a `PauseAction`).
        after: The diagnosis observed after resuming and running for some
            further steps.
        improved: Whether `after` represents better training health than
            `before`, per `RecoveryEvaluator`'s comparison rules. Only
            meaningful when `conclusive` is `True`.
        conclusive: Whether a reliable judgment could be made at all.
            `False` when either diagnosis is `degraded=True` (addendum
            §1: a degraded diagnosis is not trustworthy evidence, so it
            cannot be used to judge whether a recovery attempt helped).
            When `False`, `improved` is always `False` and must not be
            interpreted as "did not improve" -- it means "cannot tell".
        summary: Human-readable explanation of the judgment, suitable for
            a report/CLI/audit log.
    """

    strategy_name: str
    before: DiagnosisResult
    after: DiagnosisResult
    improved: bool
    conclusive: bool
    summary: str

    def __post_init__(self) -> None:
        check_non_empty_str(self.strategy_name, "strategy_name")
        check_type(self.before, DiagnosisResult, "before")
        check_type(self.after, DiagnosisResult, "after")
        check_type(self.improved, bool, "improved")
        check_type(self.conclusive, bool, "conclusive")
        check_type(self.summary, str, "summary")


class RecoveryEvaluator:
    """Compares before/after diagnoses to judge a recovery attempt.

    Example:
        >>> evaluator = RecoveryEvaluator()
        >>> result = evaluator.evaluate("parameter_reinitialization", before, after)
        >>> if evaluator.should_keep(result):
        ...     print("keep the change:", result.summary)
        ... else:
        ...     print("roll back:", result.summary)
    """

    def __init__(
        self, confidence_improvement_threshold: float = _DEFAULT_CONFIDENCE_IMPROVEMENT_THRESHOLD
    ) -> None:
        """Configure the evaluator.

        Args:
            confidence_improvement_threshold: Minimum drop in
                `DiagnosisResult.confidence` (when `before`/`after` report
                the same issue at the same severity) required to call the
                change "improved" rather than "unchanged, within noise".

        Raises:
            ValueError: If `confidence_improvement_threshold` is not in
                `[0, 1]`.
        """
        check_range(confidence_improvement_threshold, 0.0, 1.0, "confidence_improvement_threshold")
        self._confidence_improvement_threshold = confidence_improvement_threshold

    def evaluate(
        self, strategy_name: str, before: DiagnosisResult, after: DiagnosisResult
    ) -> RecoveryEvaluationResult:
        """Compare `before` and `after`, judging whether health improved.

        Never raises for well-formed `DiagnosisResult` inputs.

        Args:
            strategy_name: Identifies the recovery attempt being judged.
            before: Diagnosis observed before the recovery attempt.
            after: Diagnosis observed after resuming.

        Returns:
            A `RecoveryEvaluationResult`. See its docstring for the
            `conclusive`/`improved` semantics.
        """
        if before.degraded or after.degraded:
            return RecoveryEvaluationResult(
                strategy_name=strategy_name,
                before=before,
                after=after,
                improved=False,
                conclusive=False,
                summary=(
                    "Inconclusive: "
                    + (
                        "the 'before' diagnosis was degraded"
                        if before.degraded
                        else "the 'after' diagnosis was degraded"
                    )
                    + "; a degraded diagnosis is not reliable evidence for judging recovery."
                ),
            )

        before_good = before.issue in _GOOD_ISSUES
        after_good = after.issue in _GOOD_ISSUES

        if after_good and not before_good:
            return self._result(
                strategy_name,
                before,
                after,
                improved=True,
                summary=f"Improved: training reached {after.issue.value!r} after recovery.",
            )

        if before_good and not after_good:
            return self._result(
                strategy_name,
                before,
                after,
                improved=False,
                summary=(
                    f"Regressed: training was {before.issue.value!r} before recovery but is "
                    f"now {after.issue.value!r}."
                ),
            )

        if before.issue == after.issue:
            return self._evaluate_same_issue(strategy_name, before, after)

        return self._evaluate_different_issue(strategy_name, before, after)

    def _evaluate_same_issue(
        self, strategy_name: str, before: DiagnosisResult, after: DiagnosisResult
    ) -> RecoveryEvaluationResult:
        before_rank = SEVERITY_RANK[before.severity]
        after_rank = SEVERITY_RANK[after.severity]

        if after_rank < before_rank:
            return self._result(
                strategy_name,
                before,
                after,
                improved=True,
                summary=(
                    f"Improved: {after.issue.value!r} severity dropped from "
                    f"{before.severity!r} to {after.severity!r}."
                ),
            )
        if after_rank > before_rank:
            return self._result(
                strategy_name,
                before,
                after,
                improved=False,
                summary=(
                    f"Worsened: {after.issue.value!r} severity rose from "
                    f"{before.severity!r} to {after.severity!r}."
                ),
            )

        confidence_drop = before.confidence - after.confidence
        if confidence_drop >= self._confidence_improvement_threshold:
            return self._result(
                strategy_name,
                before,
                after,
                improved=True,
                summary=(
                    f"Improved: same issue ({after.issue.value!r}) and severity, but "
                    f"confidence dropped from {before.confidence:.2f} to {after.confidence:.2f}."
                ),
            )
        return self._result(
            strategy_name,
            before,
            after,
            improved=False,
            summary=(
                f"Unchanged: {after.issue.value!r} persists at the same severity and "
                "confidence, within the configured threshold."
            ),
        )

    def _evaluate_different_issue(
        self, strategy_name: str, before: DiagnosisResult, after: DiagnosisResult
    ) -> RecoveryEvaluationResult:
        before_rank = SEVERITY_RANK[before.severity]
        after_rank = SEVERITY_RANK[after.severity]
        improved = after_rank < before_rank
        verdict = (
            "severity decreased, treated as improvement"
            if improved
            else ("severity did not decrease, treated as no improvement")
        )
        return self._result(
            strategy_name,
            before,
            after,
            improved=improved,
            summary=(
                f"Issue changed from {before.issue.value!r} to {after.issue.value!r} "
                f"(severity {before.severity!r} -> {after.severity!r}); {verdict}."
            ),
        )

    @staticmethod
    def _result(
        strategy_name: str,
        before: DiagnosisResult,
        after: DiagnosisResult,
        *,
        improved: bool,
        summary: str,
    ) -> RecoveryEvaluationResult:
        return RecoveryEvaluationResult(
            strategy_name=strategy_name,
            before=before,
            after=after,
            improved=improved,
            conclusive=True,
            summary=summary,
        )

    def should_keep(self, result: RecoveryEvaluationResult) -> bool:
        """Whether the recovery attempt should be kept (vs. rolled back).

        Conservative by construction: an inconclusive result (`conclusive
        =False`) is never kept -- the same fail-open-but-cautious posture
        used everywhere else a degraded diagnosis is involved (addendum
        §1). Only a conclusive, `improved=True` result should be kept.
        """
        return result.conclusive and result.improved
