"""WebhookAction: deliver a structured alert to a configured HTTP endpoint.

Milestone 10, Issues #70 ("Generic webhook integration"), #74 ("Alert
deduplication"), #75 ("Alert cooldowns"), #75b (`redact_evidence`), and
#75c ("Threat-model webhook URLs").

This is the delivery mechanism the intervention model's level-2 "warn"
step (plan.md §7) gains beyond the terminal/logger `AlertAction`
(Milestone 5): "Emit terminal warning, dashboard alert, or *webhook
notification*." No third-party HTTP client is added as a dependency --
`urllib.request` (stdlib) is sufficient for a single JSON POST and keeps
the project's "no heavy frontend / low overhead" default install as light
as it is today (plan.md §16, addendum's performance rules).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from qml_observer.actions.base import Action, ActionResult
from qml_observer.integrations.formatters import default_formatter
from qml_observer.integrations.payloads import (
    SEVERITY_RANK,
    AlertPayload,
    build_alert_payload,
    redact_payload,
)
from qml_observer.integrations.security import check_webhook_url
from qml_observer.schemas.diagnosis import DiagnosisResult

_logger = logging.getLogger("qml_observer.integrations.webhook")

#: `(issue, severity, degraded)` -- the key alert deduplication/cooldown
#: (Issues #74/#75) key off of. Two alerts with the same key are
#: considered "the same kind" of alert for the same run.
_DedupKey = tuple[str, str, bool]


class WebhookDeliveryError(Exception):
    """Uniform wrapper around any transport failure while POSTing an alert.

    Never escapes `WebhookAction.execute()` -- it exists only so
    `execute()` has one consistent exception type/message to catch and
    report through `ActionResult`, regardless of whether the underlying
    failure was `urllib.error.HTTPError`, `URLError`, a timeout, or
    something else.
    """


class WebhookAction(Action):
    """POSTs a structured `AlertPayload` (Issue #71) to a webhook URL.

    Never raises *during delivery*: a network failure, timeout, non-2xx
    response, or a misbehaving formatter/provider is caught and reported
    via `ActionResult(executed=False, ...)`, consistent with the
    fail-open philosophy every other `Action` in this codebase follows
    (addendum §1) -- an unreachable or misconfigured webhook endpoint
    must never interrupt the caller's training loop. Construction *can*
    raise (see `UnsafeWebhookURLError` below) -- that is a configuration
    error, not a mid-training data issue, so it is deliberately not
    covered by the fail-open policy (the same distinction `QMLMonitor`
    draws for its own `RuntimeError` on misuse).

    Run context (`Action.execute()` only receives a bare `DiagnosisResult`
    -- see `actions/base.py` -- which has no run-identity or live-metrics
    fields of its own; that context lives one layer up, on
    `QMLMonitor`/`RunState`, see `core/monitor.py`): rather than changing
    the shared `Action` interface for this one action, `WebhookAction`
    accepts optional `run_id_provider`/`metrics_provider` callables the
    caller can wire up once, e.g.::

        webhook = WebhookAction(
            "https://example.com/hooks/qml-observer",
            run_id_provider=lambda: monitor.run_id,
            metrics_provider=lambda: (
                {"step": obs.training_event.step, "loss": obs.training_event.loss}
                if (obs := monitor.state.latest_observation) is not None
                else None
            ),
        )

    Both default to `None`, in which case `AlertPayload.run_id`/
    `current_metrics` are simply omitted -- a bare `WebhookAction` with
    just a URL is always usable on its own.

    **Deduplication and cooldowns (Issues #74, #75)**: by default,
    `execute()` suppresses an alert identical in `(issue, severity,
    degraded)` to the last alert this instance actually delivered --
    otherwise a persistent condition (e.g. a barren plateau lasting 500
    steps under `policy="warn"`) would fire one HTTP request per
    `update()` call for the entire duration. Any *change* in issue,
    severity, or degraded-ness always re-fires immediately.

    That suppression is permanent by default (an unchanged condition
    never re-fires once delivered). Set `cooldown_seconds` to instead
    allow a periodic re-send of the *same* alert once that many seconds
    have elapsed since it was last delivered -- e.g. `cooldown_seconds=
    300` re-notifies at most once every 5 minutes for an ongoing,
    unresolved issue, rather than staying silent indefinitely.
    `cooldown_seconds=None` (the default) keeps the original Issue #74
    behavior: suppress forever until the alert kind changes.

    **Redaction (Issue #75b)**: `redact_evidence=True` strips
    `evidence`/`current_metrics` from the payload before it is formatted
    and sent (see `qml_observer.integrations.payloads.redact_payload`),
    so raw circuit/gradient details aren't posted to a third-party
    service (e.g. Slack). `AlertPayload.redacted=True` is still sent, so
    the receiving side can tell "withheld" apart from "nothing to
    report".

    **URL safety (Issue #75c)**: construction refuses a URL that isn't
    `http(s)`, or that looks like it targets `localhost`/a loopback,
    link-local, or private-range address, unless `allow_internal_targets
    =True` is passed explicitly. This is a minimal, DNS-free safeguard
    against SSRF if a webhook URL is ever sourced from an untrusted
    caller -- see `qml_observer.integrations.security` for exactly what
    it does and does not cover.
    """

    name = "webhook"

    def __init__(
        self,
        url: str,
        *,
        formatter: Callable[[AlertPayload], dict[str, Any]] | None = None,
        min_severity: str = "warning",
        timeout: float = 5.0,
        headers: dict[str, str] | None = None,
        run_id_provider: Callable[[], str | None] | None = None,
        metrics_provider: Callable[[], dict[str, Any] | None] | None = None,
        deduplicate: bool = True,
        cooldown_seconds: float | None = None,
        redact_evidence: bool = False,
        allow_internal_targets: bool = False,
    ) -> None:
        """Create a `WebhookAction`.

        Args:
            url: Endpoint to POST the alert payload to. Must be `http`
                or `https`, and must not look like an internal/loopback
                target unless `allow_internal_targets=True` (Issue #75c).
            formatter: `Callable[[AlertPayload], dict]` building the JSON
                body actually sent. Defaults to `default_formatter` (the
                raw structured payload as-is); pass `slack_formatter`
                (Issue #72) for a Slack-compatible body, or any custom
                callable for another chat-ops/monitoring tool.
            min_severity: Minimum `DiagnosisResult.severity` (ordered via
                `SEVERITY_RANK`, Issue #73) that triggers delivery.
                Diagnoses below this threshold are skipped
                (`executed=False`), mirroring `AlertAction` skipping
                `"info"`-severity results.
            timeout: Socket timeout, in seconds, for the HTTP request.
            headers: Extra HTTP headers to send (e.g. an auth token).
                `Content-Type: application/json` is always set and is not
                overridable via this argument.
            run_id_provider: Optional zero-arg callable returning the
                current run ID (or `None`), invoked fresh on every
                `execute()` call.
            metrics_provider: Optional zero-arg callable returning a
                current-metrics dict (or `None`), invoked fresh on every
                `execute()` call.
            deduplicate: Whether to suppress repeat identical alerts
                (Issue #74). Set `False` to attempt delivery on every
                qualifying step regardless of repetition (in which case
                `cooldown_seconds` is ignored, since there is no
                "last delivered alert" state to rate-limit against).
            cooldown_seconds: If set, and `deduplicate=True`, a repeat of
                the last-delivered alert kind is allowed through again
                once this many seconds have elapsed (Issue #75), instead
                of being suppressed forever. `None` keeps the original
                Issue #74 behavior (permanent suppression until the
                alert kind changes).
            redact_evidence: If `True`, strip `evidence`/`current_metrics`
                from the payload before formatting/sending (Issue #75b).
            allow_internal_targets: If `True`, skip the Issue #75c
                obviously-internal-host check for `url` (e.g. for local
                development against a local receiver).

        Raises:
            ValueError: If `url` is empty/blank, or `min_severity` is not
                one of `SEVERITY_RANK`'s keys.
            UnsafeWebhookURLError: If `url` isn't `http(s)`, or looks
                like an internal/loopback/link-local target and
                `allow_internal_targets` is `False`.
        """
        if not url or not url.strip():
            raise ValueError("url must be a non-empty string")
        if min_severity not in SEVERITY_RANK:
            raise ValueError(
                f"min_severity must be one of {sorted(SEVERITY_RANK)}, got {min_severity!r}"
            )
        check_webhook_url(url, allow_internal_targets=allow_internal_targets)

        self._url = url
        self._formatter = formatter if formatter is not None else default_formatter
        self._min_severity = min_severity
        self._timeout = timeout
        self._headers = dict(headers) if headers else {}
        self._run_id_provider = run_id_provider
        self._metrics_provider = metrics_provider
        self._deduplicate = deduplicate
        self._cooldown_seconds = cooldown_seconds
        self._redact_evidence = redact_evidence
        self._last_fired_key: _DedupKey | None = None
        self._last_fired_monotonic: float | None = None

    @property
    def url(self) -> str:
        """The configured webhook endpoint."""
        return self._url

    @property
    def min_severity(self) -> str:
        """The minimum severity (per `SEVERITY_RANK`) that triggers delivery."""
        return self._min_severity

    def execute(self, diagnosis: DiagnosisResult) -> ActionResult:
        """POST a structured alert for `diagnosis`, unless skipped or deduplicated."""
        if SEVERITY_RANK[diagnosis.severity] < SEVERITY_RANK[self._min_severity]:
            return ActionResult(
                action_name=self.name,
                executed=False,
                message=(
                    f"Skipped: severity {diagnosis.severity!r} is below "
                    f"min_severity {self._min_severity!r}."
                ),
            )

        key: _DedupKey = (diagnosis.issue.value, diagnosis.severity, diagnosis.degraded)
        skip_reason = self._dedup_skip_reason(key)
        if skip_reason is not None:
            return ActionResult(action_name=self.name, executed=False, message=skip_reason)

        run_id = self._safe_call(self._run_id_provider, "run_id_provider")
        metrics = self._safe_call(self._metrics_provider, "metrics_provider")
        payload = build_alert_payload(diagnosis, run_id=run_id, current_metrics=metrics)
        if self._redact_evidence:
            payload = redact_payload(payload)

        try:
            body = self._formatter(payload)
        except Exception as exc:
            return ActionResult(
                action_name=self.name,
                executed=False,
                message=f"WebhookAction formatter failed: {type(exc).__name__}: {exc}",
            )

        try:
            self._post(body)
        except WebhookDeliveryError as exc:
            _logger.warning("qml_observer webhook delivery failed: %s", exc)
            return ActionResult(
                action_name=self.name,
                executed=False,
                message=f"WebhookAction delivery failed: {exc}",
            )

        self._last_fired_key = key
        self._last_fired_monotonic = time.monotonic()
        return ActionResult(
            action_name=self.name,
            executed=True,
            message=f"Delivered webhook alert for {diagnosis.issue.value} to {self._url}.",
        )

    def _dedup_skip_reason(self, key: _DedupKey) -> str | None:
        """Return a skip message if `key` should be suppressed, else `None`.

        Implements Issue #74 (permanent suppression of an unchanged alert
        kind) and Issue #75 (an optional cooldown that instead allows a
        periodic re-send of that same alert kind).
        """
        if not self._deduplicate or key != self._last_fired_key:
            return None
        if self._cooldown_seconds is None:
            return (
                "Skipped: duplicate alert suppressed "
                "(same issue/severity/degraded as the last alert delivered)."
            )
        elapsed = time.monotonic() - (self._last_fired_monotonic or 0.0)
        remaining = self._cooldown_seconds - elapsed
        if remaining > 0:
            return (
                "Skipped: duplicate alert suppressed "
                f"({remaining:.1f}s remaining in {self._cooldown_seconds:.1f}s cooldown)."
            )
        return None  # cooldown elapsed: allow this periodic re-send through

    @staticmethod
    def _safe_call(provider: Callable[[], Any] | None, provider_name: str) -> Any | None:
        if provider is None:
            return None
        try:
            return provider()
        except Exception:
            _logger.warning(
                "qml_observer webhook: %s raised; continuing without it.",
                provider_name,
                exc_info=True,
            )
            return None

    def _post(self, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", **self._headers}
        request = urllib.request.Request(self._url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                status = getattr(response, "status", 200)
                if not (200 <= status < 300):
                    raise WebhookDeliveryError(f"unexpected HTTP status {status}")
        except urllib.error.HTTPError as exc:
            raise WebhookDeliveryError(f"HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise WebhookDeliveryError(f"connection error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise WebhookDeliveryError(f"timed out after {self._timeout}s") from exc
        except WebhookDeliveryError:
            raise
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise WebhookDeliveryError(f"{type(exc).__name__}: {exc}") from exc

    def reset(self) -> None:
        """Clear deduplication/cooldown memory (Issues #74, #75), e.g. when
        starting a new run."""
        self._last_fired_key = None
        self._last_fired_monotonic = None
