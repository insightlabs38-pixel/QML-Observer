"""DiagnosisResult schema.

The final output of the diagnosis engine (Milestone 4 / Volume VII):
a single, explainable verdict combining evidence from all detectors,
as distinct from any individual detector's raw `DetectorResult`
(Volume V) — see the blueprint's second architectural rule: detection
and diagnosis are separate concerns.

Bug fix (addendum §6): the blueprint's original draft of this module
contained an invalid `from typing import list` import — `list` is not
exported from `typing`. Python 3.9+ (and this project's 3.12+ target)
supports the builtin generic `list[str]` directly, so no import is
needed for that at all.
"""

from dataclasses import dataclass
from enum import StrEnum

from qml_observer.schemas._validation import check_range, check_str_list, check_type

#: Controlled vocabulary for `DiagnosisResult.severity`. Kept as a plain
#: str field (not its own enum) for MVP simplicity per the blueprint, but
#: validated against this fixed set so severities stay meaningful and
#: comparable across the CLI/report/dashboard consumers.
SEVERITY_LEVELS = frozenset({"info", "warning", "critical"})


class IssueType(StrEnum):
    """The set of training states the diagnosis engine can report."""

    HEALTHY = "healthy"
    CONVERGED = "converged"
    POSSIBLE_BARREN_PLATEAU = "possible_barren_plateau"
    STAGNATION = "stagnation"
    NOISE_DOMINATED = "noise_dominated"
    UNSTABLE = "unstable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class DiagnosisResult:
    """The diagnosis engine's verdict for a training run at a given step.

    Attributes:
        issue: The suspected issue type. Note `POSSIBLE_BARREN_PLATEAU` is
            named deliberately — the engine reports a probable plateau-like
            failure mode, never a definitive proof (blueprint Volume VI-1).
        confidence: Confidence score in [0, 1] for `issue`.
        severity: Human-readable severity label (e.g. "info", "warning",
            "critical"). Kept as a plain string for MVP simplicity; see
            Issue #9 for where format/value validation is added.
        evidence: Human-readable evidence strings supporting `issue`
            (e.g. "gradient norm 2.4e-9 for 240 consecutive steps").
        recommendations: Human-readable suggested next steps.
        degraded: True if this diagnosis was produced while one or more
            detectors/statistics functions failed mid-run (addendum §1,
            fail-open policy). When True, callers (CLI/report/dashboard)
            must visibly flag the run rather than presenting it as a
            fully trustworthy diagnosis, and `ActionPolicy` must not
            auto-escalate to `stop` based on it except in
            `mode="adaptive"` with explicit opt-in.
        degraded_reason: Human-readable explanation of what failed, when
            `degraded` is True (e.g. the offending detector name and
            exception message). None when `degraded` is False.
    """

    issue: IssueType
    confidence: float
    severity: str
    evidence: list[str]
    recommendations: list[str]
    degraded: bool = False
    degraded_reason: str | None = None

    def __post_init__(self) -> None:
        check_type(self.issue, IssueType, "issue")
        check_range(self.confidence, 0.0, 1.0, "confidence")
        check_type(self.severity, str, "severity")
        if self.severity not in SEVERITY_LEVELS:
            raise ValueError(
                f"severity must be one of {sorted(SEVERITY_LEVELS)}, got {self.severity!r}"
            )
        check_str_list(self.evidence, "evidence")
        check_str_list(self.recommendations, "recommendations")
        check_type(self.degraded, bool, "degraded")
        if self.degraded_reason is not None:
            check_type(self.degraded_reason, str, "degraded_reason")
        if self.degraded and self.degraded_reason is None:
            raise ValueError("degraded_reason is required when degraded=True")
        if not self.degraded and self.degraded_reason is not None:
            raise ValueError("degraded_reason must be None when degraded=False")
