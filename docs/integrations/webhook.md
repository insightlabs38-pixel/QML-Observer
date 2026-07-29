# Webhook alerting

**Requires:** nothing extra -- `WebhookAction` uses only `urllib.request`
(Python stdlib); no `requests`/`httpx`/`slack_sdk` dependency is added.

Milestone 10 adds a delivery mechanism for diagnoses beyond `AlertAction`'s
terminal/logger output (`docs/architecture/actions.md`): `WebhookAction`
(Issue #70) POSTs a structured, framework-agnostic alert payload
(Issue #71) to a configured URL.

## Quickstart

```python
from qml_observer import QMLMonitor
from qml_observer.integrations import WebhookAction

monitor = QMLMonitor(detectors=[...], policy="warn")
webhook = WebhookAction("https://hooks.example.com/qml-observer")

for step in range(n_steps):
    diagnosis = monitor.update(step=step, loss=loss, gradients=gradients)
    result = webhook.execute(diagnosis)
    # result.executed is False if the diagnosis was below min_severity,
    # a duplicate of the last alert, or delivery failed -- see below.
```

`WebhookAction` is a regular `Action` (`execute(diagnosis) -> ActionResult`,
see `docs/architecture/actions.md`), so it composes with anything already
built for that interface -- it's just not one of the actions
`ActionPolicy` selects automatically, since it needs a URL to be
constructed. Call it explicitly wherever you already call `monitor.
update()`, or from a custom `Action` that wraps it.

## Structured payload (Issue #71)

Every alert is built from `AlertPayload`
(`qml_observer.integrations.payloads`):

| Field               | Source                                            |
|---------------------|----------------------------------------------------|
| `run_id`            | `run_id_provider()`, if configured (see below)     |
| `severity`          | `DiagnosisResult.severity` (unchanged, see below)  |
| `issue`             | `DiagnosisResult.issue.value`                      |
| `confidence`        | `DiagnosisResult.confidence`                       |
| `current_metrics`   | `metrics_provider()`, if configured (see below)    |
| `evidence`          | `DiagnosisResult.evidence`                         |
| `recommendations`   | `DiagnosisResult.recommendations`                  |
| `degraded`          | `DiagnosisResult.degraded` (addendum §1)           |
| `timestamp`         | wall-clock time the payload was built              |

`Action.execute()` only receives a `DiagnosisResult` -- it has no run
identity or live-metrics fields of its own (those live one layer up, on
`QMLMonitor`/`RunState`). Rather than changing the shared `Action`
interface for this one action, `WebhookAction` accepts optional
zero-arg callables, invoked fresh on every `execute()`:

```python
webhook = WebhookAction(
    url,
    run_id_provider=lambda: monitor.run_id,
    metrics_provider=lambda: (
        {"step": obs.training_event.step, "loss": obs.training_event.loss}
        if (obs := monitor.state.latest_observation) is not None
        else None
    ),
)
```

Both default to `None`; a bare `WebhookAction(url)` is always usable on
its own, just without `run_id`/`current_metrics` populated.

## Formatters, including Slack (Issue #72)

By default, the JSON body POSTed is `AlertPayload.to_dict()` unchanged
(`default_formatter`) -- suited to a custom alerting backend or ingestion
endpoint. Pass `slack_formatter` for Slack's incoming-webhook JSON shape
(`{"text": ..., "attachments": [...]}`, colored by severity), which is
also compatible with several other chat-ops tools that accept the same
shape -- no `slack_sdk` dependency involved:

```python
from qml_observer.integrations import WebhookAction, slack_formatter

webhook = WebhookAction(slack_incoming_webhook_url, formatter=slack_formatter)
```

Any `Callable[[AlertPayload], dict]` works as a `formatter`; write your
own for another target.

## Severity (Issue #73)

`WebhookAction` does not invent a second severity vocabulary. Every
payload's `severity` is exactly `DiagnosisResult.severity`, one of
`qml_observer.schemas.diagnosis.SEVERITY_LEVELS`. `min_severity`
(default `"warning"`) filters using `SEVERITY_RANK`
(`qml_observer.integrations.payloads`), which only orders that same fixed
set (`info < warning < critical`) -- it does not add new severity values.

## Deduplication and cooldowns (Issues #74, #75)

Under `policy="warn"`, an unresolved condition (e.g. a barren plateau
persisting for hundreds of steps) produces a new, `"warning"`/`"critical"`
`DiagnosisResult` on every single `update()` call. `AlertAction`'s
terminal output already fires every time this happens; a webhook that did
the same would flood the receiving endpoint for the entire duration of a
long-running issue. By default, `WebhookAction` only delivers when
`(issue, severity, degraded)` differs from the last alert it actually
sent -- a genuine change re-fires immediately, but an unchanged, ongoing
condition fires once and then stays silent (Issue #74).

Set `cooldown_seconds` to relax that into a periodic re-send instead of
permanent silence -- e.g. `cooldown_seconds=300` re-notifies at most once
every 5 minutes for a still-unresolved issue (Issue #75):

```python
webhook = WebhookAction(url, cooldown_seconds=300)
```

`deduplicate=False` disables suppression entirely (every qualifying step
attempts delivery, and `cooldown_seconds` has no effect since there is no
"last delivered alert" state to rate-limit against); `webhook.reset()`
clears the dedup/cooldown memory, e.g. when starting a new run with the
same `WebhookAction` instance.

## Redacting evidence (Issue #75b)

`redact_evidence=True` strips `evidence` and `current_metrics` from the
payload before it's formatted and sent, so raw gradient values,
thresholds, and step counts aren't posted to a third-party service:

```python
webhook = WebhookAction(url, redact_evidence=True)
```

The payload still carries `redacted=True` explicitly (rather than
silently sending empty `evidence`/`current_metrics`), so a receiving
service -- or a human reading the raw JSON -- can tell "withheld" apart
from "nothing was observed". `slack_formatter` renders this as a
"(redacted -- raw evidence and metrics withheld from this channel)" note
in place of the evidence/metrics fields.

## Webhook URL safety (Issue #75c)

`WebhookAction` refuses to construct against a URL that isn't `http`/
`https`, or that looks like it targets `localhost`, a loopback address,
a link-local address, or a private IP range (`10.0.0.0/8`,
`192.168.0.0/16`, cloud metadata endpoints like `169.254.169.254`, etc.),
unless you pass `allow_internal_targets=True`:

```python
WebhookAction("http://localhost:9000/hook")  # raises UnsafeWebhookURLError
WebhookAction("http://localhost:9000/hook", allow_internal_targets=True)  # OK
```

This is a **minimal, DNS-free safeguard**, primarily relevant if
`WebhookAction` is ever constructed from a URL your own code didn't
choose (e.g. a shared/multi-tenant service that accepts a "webhook URL"
from a caller) -- the classic SSRF shape. It is a literal-IP/hostname
check only: it does not resolve DNS (so a public-looking hostname whose
DNS response later points internally isn't caught) and it does not
re-validate HTTP redirect targets. See `qml_observer.integrations.
security` and `SECURITY.md` for the exact boundary. For a single-user
script configuring its own alert destination, this mostly just means
remembering to pass `allow_internal_targets=True` when pointing a
webhook at a local receiver, as `examples/generic/webhook_alerting.py`
does.

## Failure handling

Like every `Action` in this project (addendum §1's fail-open philosophy),
`WebhookAction.execute()` never raises. A DNS/connection failure, timeout,
non-2xx HTTP status, or a broken custom `formatter`/`run_id_provider`/
`metrics_provider` is caught and reported via
`ActionResult(executed=False, message=...)`; a broken or unreachable
webhook endpoint must never interrupt your training loop. (Construction
-- i.e. `WebhookAction(...)` itself -- *can* raise `ValueError`/
`UnsafeWebhookURLError` for a malformed/unsafe configuration; that's a
programmer-error case, the same distinction `QMLMonitor` draws for
calling `update()` after `finish()`.)

See `examples/generic/webhook_alerting.py` for a complete, runnable
end-to-end example (spins up a local HTTP receiver, no external service
needed).
