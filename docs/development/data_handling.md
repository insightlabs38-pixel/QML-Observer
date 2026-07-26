# Data handling & privacy

## What QML Observer writes, and where

JSONL logs (`RunReporter`/`JSONLWriter`, see `docs/architecture/events.md`)
and CLI reports may contain proprietary circuit/ansatz metadata and loss
curves. **All of this is written locally only, by default.** No training
data, metrics, or metadata leaves the machine running `qml-observer`
unless one of the following is explicitly configured by the user:

- A webhook alert channel (planned for Milestone 10 -- not yet shipped in
  `0.1.0`). When it ships, alert payloads will support a `redact_evidence`
  option so raw evidence strings/metrics can be withheld from third-party
  services (e.g. Slack).
- A future opt-in telemetry collector (see below).

## Telemetry

QML Observer ships an **opt-in, disabled-by-default** anonymized
telemetry system (addendum §5). Nothing is collected or transmitted
unless you explicitly run `qml-observer telemetry enable` or call
`qml_observer.telemetry.enable()`. Even then, no data leaves your machine
unless a telemetry endpoint is explicitly configured (there is no bundled
backend). See `docs/development/telemetry.md` for the exact schema and
what is/is not collected.

## Third-party detector plugins (future)

A community detector plugin API is planned for Milestone 14. Plugin
detectors will run in-process with no sandboxing -- a malicious plugin
would have full code execution in your training process. This is an
accepted tradeoff for a research tool, but it will be documented
explicitly (not silently assumed) once the plugin API ships.
