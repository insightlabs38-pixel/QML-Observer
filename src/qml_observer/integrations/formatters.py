"""Webhook payload formatters.

Milestone 10, Issue #72 ("Slack-compatible webhook example/formatter").

A `WebhookAction` formatter is any `Callable[[AlertPayload], dict]`: it
takes the channel-agnostic `AlertPayload` (Issue #71) and returns the
JSON body actually POSTed. `default_formatter` posts the payload as-is
(`AlertPayload.to_dict()`); `slack_formatter` reshapes it into Slack's
incoming-webhook JSON shape.

Per the Issue #72 scope note in `future_milestones_plan.md`: this is
"Slack's incoming-webhook JSON shape as one built-in payload formatter,
not a Slack-specific dependency" -- there is no `slack_sdk` import
anywhere here, just a plain dict matching what Slack's incoming webhooks
expect (`text` + `attachments`), which is also compatible with several
other chat-ops tools that accept the same shape.
"""

from __future__ import annotations

from typing import Any

from qml_observer.integrations.payloads import AlertPayload

#: Slack "attachment" side-bar colors per severity. `SEVERITY_LEVELS`
#: (schemas.diagnosis) is the source of truth for which severities exist;
#: this only maps each of those existing values to a display color.
_SLACK_COLOR_FOR_SEVERITY: dict[str, str] = {
    "info": "#2EB67D",  # good / green
    "warning": "#ECB22E",  # warning / yellow
    "critical": "#E01E5A",  # danger / red
}

_HEADLINE_FOR_ISSUE: dict[str, str] = {
    "possible_barren_plateau": "Possible barren plateau detected",
    "stagnation": "Training appears stagnant",
    "noise_dominated": "Gradient signal appears noise-dominated",
    "unstable": "Training appears numerically unstable",
    "converged": "Training has converged",
    "healthy": "Training is healthy",
    "insufficient_evidence": "Insufficient evidence for a diagnosis",
}


def default_formatter(payload: AlertPayload) -> dict[str, Any]:
    """Return `payload` as a plain JSON-serializable dict, unchanged.

    This is `WebhookAction`'s default formatter: a generic consumer (a
    user's own alerting backend, a custom dashboard ingestion endpoint,
    etc.) almost always wants the raw structured payload rather than a
    chat-message-shaped body.
    """
    return payload.to_dict()


def slack_formatter(payload: AlertPayload) -> dict[str, Any]:
    """Format `payload` as a Slack incoming-webhook message body.

    Produces `{"text": ..., "attachments": [...]}`, matching what Slack's
    (and several Slack-compatible) incoming webhooks expect: a short
    fallback `text` plus a colored `attachments` block with the run,
    confidence, evidence, and recommended next steps as fields.
    """
    headline = _HEADLINE_FOR_ISSUE.get(payload.issue, f"Status: {payload.issue}")
    degraded_prefix = ":warning: DIAGNOSIS DEGRADED -- " if payload.degraded else ""
    text = f"{degraded_prefix}*qml-observer alert:* {headline}"

    fields: list[dict[str, Any]] = [
        {"title": "Severity", "value": payload.severity, "short": True},
        {"title": "Confidence", "value": f"{payload.confidence:.0%}", "short": True},
    ]
    if payload.run_id:
        fields.append({"title": "Run ID", "value": payload.run_id, "short": True})
    if payload.redacted:
        # Issue #75b: make the withholding visible rather than presenting
        # empty evidence/metrics as "nothing was observed".
        fields.append(
            {
                "title": "Evidence / metrics",
                "value": "(redacted -- raw evidence and metrics withheld from this channel)",
                "short": False,
            }
        )
    if payload.current_metrics:
        metrics_str = ", ".join(f"{k}={v}" for k, v in payload.current_metrics.items())
        fields.append({"title": "Current metrics", "value": metrics_str, "short": False})
    if payload.evidence:
        evidence_str = "\n".join(f"- {e}" for e in payload.evidence)
        fields.append({"title": "Evidence", "value": evidence_str, "short": False})
    if payload.recommendations:
        fields.append(
            {
                "title": "Recommended action",
                "value": "\n".join(f"- {r}" for r in payload.recommendations),
                "short": False,
            }
        )

    return {
        "text": text,
        "attachments": [
            {
                "color": _SLACK_COLOR_FOR_SEVERITY.get(payload.severity, "#999999"),
                "fields": fields,
                "ts": payload.timestamp,
            }
        ],
    }
