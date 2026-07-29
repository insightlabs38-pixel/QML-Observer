"""qml_observer.integrations: outbound alert delivery (Milestone 10).

Milestone 10 (`future_milestones_plan.md`) extends the action layer
(`qml_observer.actions`, Milestone 5) with a real delivery mechanism for
level-2 ("warn") diagnoses beyond the terminal/logger output `AlertAction`
already provides (plan.md §7, §24): a generic webhook (Issue #70) carrying
a structured, framework-agnostic payload (Issue #71), with a built-in
Slack-compatible formatter (Issue #72) as one example consumer of that
payload. Severity is not reinvented here -- everything reuses
`DiagnosisResult.severity`/`SEVERITY_LEVELS` (Issue #73). Repeated,
unchanged alerts firing on every single `update()` call while a condition
persists are suppressed by default (Issue #74), optionally replaced by a
periodic re-send once a configured cooldown elapses (Issue #75). Raw
evidence/metrics can be stripped before delivery (Issue #75b), and
obviously-internal webhook targets are refused by default as a minimal
SSRF safeguard (Issue #75c).

Public re-exports for `qml_observer.integrations.*`.
"""

from __future__ import annotations

from qml_observer.integrations.formatters import default_formatter, slack_formatter
from qml_observer.integrations.payloads import (
    SEVERITY_RANK,
    AlertPayload,
    build_alert_payload,
    redact_payload,
)
from qml_observer.integrations.security import UnsafeWebhookURLError, check_webhook_url
from qml_observer.integrations.webhook import WebhookAction, WebhookDeliveryError

__all__ = [
    "AlertPayload",
    "build_alert_payload",
    "redact_payload",
    "SEVERITY_RANK",
    "default_formatter",
    "slack_formatter",
    "WebhookAction",
    "WebhookDeliveryError",
    "check_webhook_url",
    "UnsafeWebhookURLError",
]
