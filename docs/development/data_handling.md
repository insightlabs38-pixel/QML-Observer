# Data handling & privacy

## What QML Observer writes, and where

JSONL logs (`RunReporter`/`JSONLWriter`, see `docs/architecture/events.md`)
and CLI reports may contain proprietary circuit/ansatz metadata and loss
curves. **All of this is written locally only, by default.** No training
data, metrics, or metadata leaves the machine running `qml-observer`
unless one of the following is explicitly configured by the user:

- A webhook alert channel (`qml_observer.integrations.WebhookAction`,
  Milestone 10). Alert payloads support a `redact_evidence=True` option
  so raw evidence strings/metrics can be withheld from third-party
  services (e.g. Slack) -- see `docs/integrations/webhook.md`.
- A future opt-in telemetry collector (see below).

## Telemetry

QML Observer ships an **opt-in, disabled-by-default** anonymized
telemetry system (addendum §5). Nothing is collected or transmitted
unless you explicitly run `qml-observer telemetry enable` or call
`qml_observer.telemetry.enable()`. Even then, no data leaves your machine
unless a telemetry endpoint is explicitly configured (there is no bundled
backend). See `docs/development/telemetry.md` for the exact schema and
what is/is not collected.

## Third-party detector plugins

A community detector plugin API shipped in Milestone 14
(`qml_observer.detectors.plugins`, Issue #103). Plugin detectors,
discovered via the `qml_observer.detectors` entry-point group, run
in-process with no sandboxing -- a malicious or buggy plugin would have
full code execution in your training process. This is an accepted
tradeoff for a research tool. `list_detector_plugins()` inspects what's
registered without importing/executing anything;
`discover_detector_plugins()`/`load_detector_plugins()` do execute each
plugin's code. See `SECURITY.md` and `docs/development/plugin_api.md`.
