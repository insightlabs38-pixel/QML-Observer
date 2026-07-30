"""Export the currently-viewed run's dashboard data (Milestone 11, Issue #82).

Distinct from `reporting.export.export_summary_json` (which writes a
`build_run_summary`-shaped file to disk from Python code): this module
builds the *dashboard's* view of a run -- the same loss/gradient/
diagnosis/compute data the `/api/*` routes already serve -- as either a
single JSON document or a per-step CSV table, for a browser download via
`GET /api/export.json` / `GET /api/export.csv` (`dashboard/app.py`).
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from qml_observer.dashboard.data_source import DashboardDataSource


def build_export_payload(source: DashboardDataSource) -> dict[str, Any]:
    """Bundle everything the dashboard currently shows for `source` into
    one plain-dict payload, shared by both the JSON and CSV exporters."""
    return {
        "run_id": source.run_id(),
        "loss": source.loss_series().to_dict(),
        "gradient": source.gradient_series().to_dict(),
        "diagnosis": source.diagnosis(),
        "compute": source.compute_usage().to_dict(),
    }


def export_json(source: DashboardDataSource) -> str:
    """Render `build_export_payload(source)` as formatted JSON text."""
    return json.dumps(build_export_payload(source), indent=2, sort_keys=True)


def export_csv(source: DashboardDataSource) -> str:
    """Render the per-step loss/gradient series as a single CSV table.

    One row per step present in either series (loss and gradient steps
    are not guaranteed to line up -- e.g. a `ReporterDataSource` always
    has an empty gradient series, addendum-style documented in
    `data_source.py`), with missing values left blank rather than `0`, so
    a spreadsheet doesn't mistake "no data" for a real zero. Diagnosis and
    compute-usage summary fields are not step-indexed, so they're
    intentionally left out of the CSV; use `export_json` for those.
    """
    loss = source.loss_series()
    gradient = source.gradient_series()

    gradient_by_step: dict[int, tuple[Any, Any, Any]] = {
        step: (norm, var, snr)
        for step, norm, var, snr in zip(
            gradient.steps, gradient.norm_l2, gradient.variance, gradient.snr, strict=True
        )
    }
    loss_by_step: dict[int, Any] = dict(zip(loss.steps, loss.loss, strict=True))

    all_steps = sorted(set(loss.steps) | set(gradient.steps))

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["step", "loss", "gradient_norm_l2", "gradient_variance", "gradient_snr"])
    for step in all_steps:
        norm, var, snr = gradient_by_step.get(step, (None, None, None))
        writer.writerow(
            [
                step,
                _blank_if_none(loss_by_step.get(step)),
                _blank_if_none(norm),
                _blank_if_none(var),
                _blank_if_none(snr),
            ]
        )
    return buffer.getvalue()


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value
