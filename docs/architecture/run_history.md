# Run history & experiment management

Milestone 14 (`future_milestones_plan.md`), Issue #103b ("Run comparison /
experiment management", plan.md §25). Implements run comparison, A/B
testing of ansatzes, saved run history, and CSV/JSON export at the level
plan.md actually scopes them -- a local, append-only ledger plus simple
comparison/export -- rather than a competing experiment-tracking UI (for
that, see [`integrations/experiment_trackers.md`](../integrations/experiment_trackers.md)'s
`MLflowTracker`/`WandbTracker` instead).

## How this differs from per-step JSONL logging

`RunReporter`'s JSONL log (`reporting/jsonl.py`, Milestone 7) records
*one run's* own events/diagnoses/summary in detail. `RunHistory`
(`qml_observer.reporting.history`) is a separate, append-only ledger
recording exactly one compact summary line per **completed** run, so it
can answer "how do my last 10 runs compare" -- a question the per-run
JSONL log alone can't, since each run's log is its own file.

## Recording runs into history

`HistoryReporter` wraps a `RunReporter` internally (reusing its summary-
building logic rather than duplicating it) and appends to a `RunHistory`
ledger once the run finishes:

```python
from qml_observer import QMLMonitor
from qml_observer.reporting.history import RunHistory, HistoryReporter

history = RunHistory("experiments/history.jsonl")
reporter = HistoryReporter(history, tags={"ansatz": "hardware_efficient"})

monitor = QMLMonitor(reporter=reporter, planned_steps=1000)
for step in range(1000):
    monitor.update(step=step, loss=loss)
monitor.finish()
```

`tags` is arbitrary user metadata (ansatz name, framework, experiment
notes, ...) -- exactly what makes A/B testing of ansatzes possible: tag
each run with which ansatz it used, then filter/compare by that tag
later.

Appending to `history` is wrapped in the same fail-open policy as the
rest of the reporting layer (addendum §1): a full disk or unwritable
ledger path is logged as a warning, never raised into your training loop.

## Comparing and exporting runs

```python
from qml_observer.reporting.history import RunHistory, compare_runs, format_comparison_table

history = RunHistory("experiments/history.jsonl")

# Every run tagged with a given ansatz (A/B testing use case):
hea_runs = history.filter_by_tag("ansatz", "hardware_efficient")

# A flat list of row dicts, for programmatic use:
rows = compare_runs(hea_runs)

# The same data as a simple text table, e.g. for printing:
print(format_comparison_table(hea_runs))

# CSV/JSON export of the entire ledger:
history.export_csv("history.csv")
history.export_json("history.json")
```

`RunHistory.get(run_id)` looks up a single run by ID (the most recently
recorded match, if a `run_id` was ever reused).

## From the CLI

```bash
qml-observer history list experiments/history.jsonl
qml-observer history list experiments/history.jsonl --tag ansatz=hardware_efficient
qml-observer history compare experiments/history.jsonl --run-id run-1 --run-id run-2
qml-observer history export experiments/history.jsonl --format csv out.csv
qml-observer history export experiments/history.jsonl --format json out.json
```

## What's stored

A `RunRecord` per completed run: `run_id`, `framework`, `steps`,
`duration`, `final_diagnosis`, `confidence`, `severity`, `degraded`,
`estimated_compute_saved`, `tags`, and `recorded_at` (when the record was
appended). This deliberately mirrors `RunReporter.finalize()`'s own
summary shape rather than the fuller per-run detail (evidence,
recommendations, loss-curve summary) -- that fuller detail already lives
in the per-run JSONL log; `RunHistory` stays a compact,
many-rows-at-a-glance ledger.

Records are written with a `schema_version` field
(`HISTORY_SCHEMA_VERSION`), following the same documented-and-versioned
convention as `reporting/jsonl.py`'s `JSONL_SCHEMA_VERSION` (Issue #108):
a future field change to `RunRecord` bumps this constant and gets a
`CHANGELOG.md` note, rather than silently changing what an older ledger's
rows mean.

## Relationship to experiment trackers (Issue #101)

`RunHistory` is a local-only, dependency-free ledger -- it doesn't
require or replace MLflow/W&B. If you want both a local comparison ledger
*and* an existing tracker, use qml_observer's fan-out reporter pattern
(see [`integrations/experiment_trackers.md`](../integrations/experiment_trackers.md#using-both-a-tracker-and-runreporterjsonl-logging))
with a `HistoryReporter` and a `MLflowTracker`/`WandbTracker` together.
