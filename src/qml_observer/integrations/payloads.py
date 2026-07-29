"""Structured alert payloads for outbound integrations.

Milestone 10, Issue #71 ("Structured alert payloads") and Issue #73
("Alert severity levels").

`AlertPayload` is the framework-agnostic shape every outbound channel
(webhook, Slack formatter, and any future channel) is built from. Per
plan.md §24, an alert payload should carry: run ID, severity, suspected
issue, confidence, current metrics, evidence snapshot, and recommended
action. This module builds that payload from a `DiagnosisResult` plus
whatever run-level context the caller can supply (`QMLMonitor` doesn't
hand its run context to `Action.execute()` -- see `webhook.py`'s module
docstring for how `WebhookAction` fills that gap without changing the
`Action` interface).

Issue #73 note: severity is *not* a second vocabulary invented for
alerting. Every payload's `severity` field is exactly
`DiagnosisResult.severity`, drawn from the existing
`qml_observer.schemas.diagnosis.SEVERITY_LEVELS` set. `SEVERITY_RANK`
below only adds an *ordering* over that same fixed vocabulary (needed to
implement a `min_severity` threshold), not a new set of values.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from qml_observer.schemas.diagnosis import SEVERITY_LEVELS, DiagnosisResult

#: Ordering over the existing `SEVERITY_LEVELS` vocabulary (Issue #73).
#: Higher rank = more severe. Used by `WebhookAction(min_severity=...)`
#: and any other consumer that needs "at least this severe" filtering
#: without inventing a parallel severity type.
SEVERITY_RANK: dict[str, int] = {"info": 0, "warning": 1, "critical": 2}

assert set(SEVERITY_RANK) == SEVERITY_LEVELS, (
    "SEVERITY_RANK must stay in sync with qml_observer.schemas.diagnosis.SEVERITY_LEVELS"
)


@dataclass
class AlertPayload:
    """Framework-agnostic content of a single outbound alert.

    Attributes:
        run_id: Identifier of the run this alert is for, or `None` if the
            caller (e.g. a bare `WebhookAction` used without a monitor)
            didn't supply one.
        issue: The suspected issue type, as a plain string (`IssueType`
            value) so this dataclass has no dependency on json-encoding
            an `Enum` itself.
        severity: One of `qml_observer.schemas.diagnosis.SEVERITY_LEVELS`
            -- see module docstring; never a channel-specific severity.
        confidence: Confidence score in `[0, 1]` for `issue`.
        current_metrics: Best-effort snapshot of the metrics that led to
            this alert (e.g. `{"step": 4200, "loss": 0.031,
            "gradient_norm": 2.4e-9}`). Empty dict if the caller supplied
            no metrics provider -- this is optional context, not a
            required field, since not every `WebhookAction` is wired to a
            live `QMLMonitor`.
        evidence: Snapshot of `DiagnosisResult.evidence` at alert time.
        recommendations: Snapshot of `DiagnosisResult.recommendations`,
            i.e. the "recommended action" plan.md §24 asks for.
        degraded: Mirrors `DiagnosisResult.degraded` -- an outbound
            channel must be able to show the same degraded-mode warning
            the CLI/report/dashboard are required to (addendum §1).
        timestamp: Unix timestamp the payload was built, for channels
            that want to display or log alert timing.
        redacted: Whether `evidence`/`current_metrics` were stripped
            before this payload left the process (Issue #75b, via
            `redact_payload()`/`WebhookAction(redact_evidence=True)`).
            Surfaced explicitly rather than silently emptying those
            fields, so a receiving service (or a human reading the raw
            payload) can tell "no evidence was collected" apart from
            "evidence was deliberately withheld".
    """

    severity: str
    issue: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    current_metrics: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    degraded: bool = False
    timestamp: float = field(default_factory=time.time)
    redacted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable dict of this payload."""
        return {
            "run_id": self.run_id,
            "severity": self.severity,
            "issue": self.issue,
            "confidence": self.confidence,
            "current_metrics": dict(self.current_metrics),
            "evidence": list(self.evidence),
            "recommendations": list(self.recommendations),
            "degraded": self.degraded,
            "timestamp": self.timestamp,
            "redacted": self.redacted,
        }


def build_alert_payload(
    diagnosis: DiagnosisResult,
    *,
    run_id: str | None = None,
    current_metrics: dict[str, Any] | None = None,
) -> AlertPayload:
    """Build an `AlertPayload` from a `DiagnosisResult` plus optional context.

    Args:
        diagnosis: The diagnosis this alert is reporting on.
        run_id: Run identifier to attach, if the caller has one available
            (e.g. `monitor.run_id`).
        current_metrics: Optional snapshot of current training metrics
            (step, loss, gradient statistics, etc). Never required:
            `DiagnosisResult` alone is always enough to build a payload.

    Returns:
        A populated `AlertPayload`. Never raises for a well-formed
        `DiagnosisResult` (the same fail-open expectation as every other
        `Action`-adjacent helper in this codebase).
    """
    return AlertPayload(
        severity=diagnosis.severity,
        issue=diagnosis.issue.value,
        confidence=diagnosis.confidence,
        evidence=list(diagnosis.evidence),
        recommendations=list(diagnosis.recommendations),
        current_metrics=dict(current_metrics) if current_metrics else {},
        run_id=run_id,
        degraded=diagnosis.degraded,
    )


def redact_payload(payload: AlertPayload) -> AlertPayload:
    """Return a copy of `payload` with raw evidence/metrics stripped.

    Milestone 10, Issue #75b. `evidence` (which quotes specific gradient
    norms, thresholds, and step counts) and `current_metrics` (loss,
    gradient statistics, etc.) can reveal details about a proprietary
    circuit/ansatz to a third-party service the webhook posts to (e.g.
    Slack) -- see `docs/development/data_handling.md`. `recommendations`
    are generic, static advice strings (not derived from the run's data)
    and are left intact; `severity`/`issue`/`confidence`/`run_id`/
    `degraded` are summary-level, not raw evidence, and are also left
    intact.

    The result has `redacted=True` set explicitly rather than silently
    presenting empty `evidence`/`current_metrics` as "nothing was
    observed" -- see `AlertPayload.redacted`'s docstring.
    """
    return AlertPayload(
        severity=payload.severity,
        issue=payload.issue,
        confidence=payload.confidence,
        evidence=[],
        recommendations=list(payload.recommendations),
        current_metrics={},
        run_id=payload.run_id,
        degraded=payload.degraded,
        timestamp=payload.timestamp,
        redacted=True,
    )
