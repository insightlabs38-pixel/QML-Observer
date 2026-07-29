# Actions

`qml_observer.actions` implements the blueprint's four-level intervention
model (plan.md §7): log-only, warn, pause (not yet distinct from warn --
see below), and stop, plus a `"adaptive"` mode for opting into stopping on
a `degraded` diagnosis.

- **`Action`** (`actions/base.py`) -- shared interface: `execute(diagnosis)
  -> ActionResult`. Every built-in action catches its own internal errors,
  so a broken custom `Action` still can't crash a training loop (the
  monitor also catches at a higher level as a second line of defense).
- **`LogAction`** -- always executes, never raises; records every
  diagnosis regardless of severity.
- **`AlertAction`** -- terminal + logger warning for any non-`"info"`
  severity diagnosis.
- **`StopAction`** -- records a stop request via a `.triggered` flag for
  the caller's own training loop to check (`monitor.should_stop()`); never
  reaches into the loop directly, preserving the non-invasive core
  principle.
- **`PauseAction`** -- not yet implemented (Milestone 13); `"pause"` mode
  currently behaves identically to `"warn"`, a deliberate conservative
  choice rather than a silent no-op.

**`ActionPolicy`** (`actions/policies.py`) selects which `Action` to run
for a given `DiagnosisResult` and mode. It enforces the addendum §1
degraded-diagnosis safety rule: a `degraded=True` diagnosis never selects
`StopAction` unless `mode="adaptive"` **and** the caller explicitly passed
`allow_stop_on_degraded=True`.

## Webhook alerting (Milestone 10)

`AlertAction`'s terminal/logger output is one channel; `qml_observer.
integrations.WebhookAction` (Milestone 10, Issue #70) is another,
delivering the same diagnosis as a structured HTTP POST instead. It is
not selected by `ActionPolicy` automatically -- it's used the same way a
user wires up any other external integration, alongside `monitor.
update()`:

```python
from qml_observer.integrations import WebhookAction, slack_formatter

webhook = WebhookAction(
    "https://hooks.example.com/qml-observer",
    formatter=slack_formatter,  # Issue #72; omit for the raw payload
    min_severity="warning",  # Issue #73, via SEVERITY_RANK
    run_id_provider=lambda: monitor.run_id,
)

for step in range(n_steps):
    diagnosis = monitor.update(...)
    webhook.execute(diagnosis)
```

See `qml_observer.integrations` module docs and
`examples/generic/webhook_alerting.py` for the full picture, including:

- **Structured payload** (Issue #71) -- `AlertPayload`: run ID, severity,
  issue, confidence, current metrics, evidence, recommendations,
  `degraded`.
- **Severity gating** (Issue #73) -- reuses `DiagnosisResult.severity`
  directly; `SEVERITY_RANK` only adds an ordering over that same
  vocabulary for `min_severity` filtering.
- **Deduplication and cooldowns** (Issues #74, #75) -- a persistent,
  unchanged condition (the common case under `policy="warn"`) fires the
  webhook once and then stays silent by default, not once per `update()`
  call; a change in issue/severity/degraded re-fires immediately.
  `cooldown_seconds` optionally relaxes "silent forever" into "re-notify
  at most every N seconds" for a still-unresolved condition.
- **Redaction** (Issue #75b) -- `redact_evidence=True` strips raw
  evidence/metrics from the payload before it's sent to a third-party
  service, while keeping the fact that it was withheld visible
  (`AlertPayload.redacted`).
- **URL safety** (Issue #75c) -- construction refuses obviously
  internal-looking targets (`localhost`, loopback/link-local/private
  ranges) by default, a minimal SSRF safeguard; `allow_internal_targets=
  True` opts out.
- **Fail-open** -- exactly like every other `Action`, a network failure,
  timeout, or non-2xx response is caught and reported via
  `ActionResult(executed=False, ...)`, never raised into the caller's loop.
