# Telemetry schema

QML Observer includes an **opt-in, anonymized** telemetry system to help
improve default detector calibration community-wide (addendum §5). This
page is the exact, auditable schema of what is sent if -- and only if --
you explicitly enable it.

## Default: off

Telemetry is **disabled by default**. Enable it explicitly via:

```bash
qml-observer telemetry enable    # persists consent to
                                  #   ~/.config/qml-observer/telemetry.json
qml-observer telemetry status
qml-observer telemetry disable
```

or from Python:

```python
from qml_observer import telemetry

telemetry.enable()
collector = telemetry.TelemetryCollector()
monitor = QMLMonitor(
    detectors=[...],
    telemetry_collector=collector,
    telemetry_framework="pennylane",
)
```

Consent is checked fresh on every emission (`consent.is_enabled()`), so
running `qml-observer telemetry disable` takes effect immediately, even
mid-process. In non-interactive environments (CI, piped input, no TTY),
`prompt_for_consent()` never auto-enables telemetry -- it always defaults
to disabled unless a human has explicitly opted in beforehand.

## What is collected (only if enabled)

One `TelemetryRecord` per finished `QMLMonitor` run:

| Field                      | Type             | Description |
|-----------------------------|------------------|-------------|
| `schema_version`             | `str`            | Version of this schema (currently `"1"`). |
| `package_version`            | `str`            | `qml_observer.__version__`. |
| `detector_names`             | `list[str]`      | Class names of the detectors that ran (e.g. `"BarrenPlateauDetector"`). |
| `thresholds`                 | `dict[str, float]` | Each detector's numeric threshold/patience constructor values, keyed `"<DetectorName>.<attr>"` (e.g. `"BarrenPlateauDetector.gradient_threshold"`). |
| `issue`                      | `str`            | The final diagnosis's `IssueType` value (e.g. `"possible_barren_plateau"`). |
| `confidence`                 | `float`          | The final diagnosis's confidence score. |
| `framework`                  | `str \| None`     | Optional label supplied via `telemetry_framework=` (e.g. `"pennylane"`, `"qiskit"`). |
| `qubit_bucket`               | `str \| None`     | A coarse range (`"1-5"`, `"6-10"`, `"11-20"`, `"21-50"`, `"51-100"`, `"101+"`) -- never the exact qubit count. |
| `detection_latency_steps`    | `int \| None`     | Total steps recorded when a non-healthy/non-`INSUFFICIENT_EVIDENCE` diagnosis was reached; `None` for healthy/inconclusive runs. |

## What is never collected

Raw gradients, loss values, circuit structure or ansatz source, parameter
values, run IDs tied to identifiable projects, file paths, or hostnames.
There is no field in `TelemetryRecord` that carries any of these, by
construction -- see `src/qml_observer/telemetry/schema.py`.

## Where it goes

This release ships **no bundled telemetry backend**. If telemetry is
enabled but no endpoint is configured, records are written locally as
JSON Lines to `~/.cache/qml-observer/telemetry_queue.jsonl` (or
`$XDG_CACHE_HOME/qml-observer/telemetry_queue.jsonl`) and never leave the
machine. Setting the `QML_OBSERVER_TELEMETRY_ENDPOINT` environment
variable (or passing `TelemetryCollector(endpoint=...)`) will POST the
JSON record to that URL instead. No such endpoint is configured by
default, and none is bundled with `qml-observer` itself.

## Source

- `src/qml_observer/telemetry/schema.py` -- `TelemetryRecord`, bucketing, threshold extraction.
- `src/qml_observer/telemetry/consent.py` -- opt-in state, persisted at `~/.config/qml-observer/telemetry.json`.
- `src/qml_observer/telemetry/collector.py` -- `TelemetryCollector` (local queue or configured endpoint).
- `tests/unit/telemetry/` -- schema, consent, and collector tests.
