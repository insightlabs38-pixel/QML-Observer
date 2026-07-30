# Dashboard (Milestone 11)

`qml_observer.dashboard` (optional `qml-observer[dashboard]` extra:
`fastapi` + `uvicorn`) is a read-only visual layer on top of data the
project already produces elsewhere -- it introduces no new place metrics
are computed. Per the addendum's stated ordering, this ships only after
the detection layer (Milestones 4-9) is trustworthy; the dashboard's job
is to *show* diagnoses, not influence them.

## Architecture decision (Issue #76)

- **Backend:** a minimal FastAPI app (`dashboard/app.py`) serving JSON.
- **Frontend:** a small static bundle (`dashboard/static/`) -- plain
  HTML/CSS/JS, no build step, no framework, and a ~100-line vendored
  canvas line-chart helper instead of a CDN-hosted charting library. The
  dashboard renders with zero internet access, matching the project's
  "nothing leaves your machine by default" posture
  (`docs/development/data_handling.md`).
- **Data:** read through a `DashboardDataSource` interface
  (`dashboard/data_source.py`), not a new metrics pipeline. Three
  built-in sources:

  | Source | Loss chart | Gradient chart | Diagnosis / compute panels |
  |---|---|---|---|
  | `MonitorDataSource(monitor)` | yes | yes (from `monitor.state.window`) | yes |
  | `ReporterDataSource(reporter)` | yes | **always empty** -- see below | yes |
  | `JSONLDataSource(path)` | yes | only if gradient was logged manually | yes, `planned_steps` always `None` -- see below |

  `MonitorDataSource` is the richest source because it reads the live
  `QMLMonitor.state.window` directly, which holds full `StepObservation`s
  (loss *and* gradient). A `RunReporter` fed only through `QMLMonitor`'s
  automatic hook never receives gradient detail in the first place (see
  `reporting/reporter.py`), so `ReporterDataSource.gradient_series()` is
  documented to always return empty rather than silently guessing. A
  `JSONLDataSource` only has gradient detail if the caller logged a full
  `event_record(event, gradient=...)` themselves.

  Similarly, `reporting.jsonl.summary_record`/`RunReporter._build_summary`
  never persisted a `planned_steps` field of their own (only the
  already-computed `estimated_compute_saved`), so
  `JSONLDataSource.compute_usage().planned_steps` is always `None` even
  though `estimated_compute_saved` itself is correct for whatever
  `planned_steps` was configured at record time. Use `MonitorDataSource`/
  `ReporterDataSource` in-process if displaying the configured value
  matters.

- **Ship as an optional extra:** importing `qml_observer.dashboard`
  itself never requires `fastapi`/`uvicorn` (core install stays light);
  `create_app`/`run_dashboard` raise a clear `ImportError` with install
  instructions the first time they're actually called without the extra
  installed.

## Routes (Issues #77-#82)

| Route | Issue | Returns |
|---|---|---|
| `GET /api/status` | -- | `{ok, version, run_id}` |
| `GET /api/loss` | #77 | `{steps: [...], loss: [...]}` |
| `GET /api/gradient` | #78 | `{steps, norm_l2, variance, snr, available}` |
| `GET /api/diagnosis` | #79 | Current `DiagnosisResult` as JSON, including `degraded`/`degraded_reason` |
| `GET /api/compute` | #80 | Planned vs. actual steps, mean wall time/step, `estimated_compute_saved` (addendum §11 formula) |
| `GET /api/history` | #81 | `{directory, entries: [...]}` -- other finalized runs, if `history_dir` was configured |
| `GET /api/export.json` | #82 | Full dashboard payload for the current run, as a JSON download |
| `GET /api/export.csv` | #82 | Per-step loss/gradient series for the current run, as a CSV download |
| `GET /` | -- | The static frontend shell |

The frontend (`static/app.js`) polls all five every 1.5s and re-renders.
`/api/diagnosis` surfaces the addendum §1 fail-open contract directly: the
UI shows a visible "⚠ DIAGNOSIS DEGRADED" banner whenever `degraded` is
true, the same posture as `qml-observer report`'s CLI output -- the
dashboard must not be the one place that silently hides a degraded run.

## Running it

```python
from qml_observer.core.monitor import QMLMonitor
from qml_observer.dashboard import MonitorDataSource, run_dashboard

monitor = QMLMonitor(planned_steps=10_000)
# ... attach monitor to a training loop in another thread/process ...

run_dashboard(MonitorDataSource(monitor, framework="pennylane"))
# -> http://127.0.0.1:8765
```

Or, post-hoc, against a finished run's JSONL log:

```python
from qml_observer.dashboard import JSONLDataSource, run_dashboard

run_dashboard(JSONLDataSource("run.jsonl"))
```

## Run history (Issue #81)

`create_app(source, history_dir=...)` (or `run_dashboard(..., history_dir=...)`)
accepts an optional directory of *other*, already-finalized runs' JSONL
logs, scanned by `dashboard/history.py::discover_run_history`. Only runs
whose log contains a `"summary"` record (i.e. `monitor.finish()` was
actually called) are listed -- a run still in progress belongs in the
live single-run view instead, not the history table. `history_dir` is
opt-in and `None` by default; the dashboard never guesses a directory to
scan on its own. A malformed or half-written log file is skipped (with a
stderr warning) rather than breaking the history view for the rest of
the directory, matching the addendum §1 fail-open posture extended to
this read path.

```python
from qml_observer.dashboard import MonitorDataSource, run_dashboard

run_dashboard(
    MonitorDataSource(monitor),
    history_dir="runs/",  # e.g. where past JSONL logs are stored
)
```

## Data export (Issue #82)

`GET /api/export.json` and `GET /api/export.csv` (also reachable directly
as `dashboard.export.export_json(source)`/`export_csv(source)`) download
the currently-viewed run's dashboard data:

- **JSON:** the same `run_id`/`loss`/`gradient`/`diagnosis`/`compute`
  shape the individual `/api/*` routes already return, bundled into one
  document.
- **CSV:** a per-step table (`step, loss, gradient_norm_l2,
  gradient_variance, gradient_snr`). Missing values are left blank, not
  `0` -- a spreadsheet must not mistake "no data from this source" for a
  real zero gradient. Diagnosis/compute-usage fields aren't step-indexed,
  so they're deliberately left out of the CSV; use the JSON export for
  those.

## Security posture (Issue #82b)

`run_dashboard()` binds to `127.0.0.1` by default and **refuses** (raises
`ValueError`) to bind anywhere else unless the caller explicitly passes
`allow_non_loopback=True`, at which point it still prints a warning to
stderr every time. This mirrors the Milestone 10 webhook's
`allow_internal_targets` refuse-by-default pattern
(`qml_observer.integrations.security`) rather than only warning: the
dashboard has no authentication, so binding to a non-loopback interface
is a real, immediate exposure of run data (potentially proprietary
circuit/loss metadata, per `docs/development/data_handling.md`), not
merely a risky default worth a warning after the fact. See `SECURITY.md`
for the corresponding scope note.

## Not yet built

- Interactive run comparison/A-B views across the history table (plan.md
  §25's fuller "experiment management" scope, tracked separately as
  Milestone 14's Issue #103b) -- today's history table is read-only and
  side-by-side comparison is out of scope for Milestone 11.
