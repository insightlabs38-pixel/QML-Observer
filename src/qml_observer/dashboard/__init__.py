"""QML Observer dashboard (Milestone 11).

Optional extra: `pip install 'qml-observer[dashboard]'`.

Public API:
    `create_app(source, history_dir=None)` -- build the FastAPI app from
        any `DashboardDataSource` (see `data_source.py`), optionally with
        a run-history panel (Issue #81, `history.py`).
    `run_dashboard(source, ...)` -- serve it (blocking) via `uvicorn`,
        bound to `127.0.0.1` by default (refuses other hosts unless
        `allow_non_loopback=True`, Issue #82b).
    `MonitorDataSource`, `ReporterDataSource`, `JSONLDataSource` -- the
        three built-in data sources.
    `discover_run_history(directory)` -- scan a directory of finalized
        runs' JSONL logs (Issue #81).
    `export_json(source)`, `export_csv(source)` -- render the currently-
        viewed run's dashboard data for download (Issue #82); also
        reachable via the app's `/api/export.json`/`/api/export.csv`
        routes.

Importing `qml_observer.dashboard` itself never requires `fastapi`/
`uvicorn` to be installed (so `import qml_observer` stays light); the
`ImportError` with install instructions is only raised lazily, the first
time `create_app`/`run_dashboard` is actually called.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qml_observer.dashboard.data_source import (
    ComputeUsage,
    DashboardDataSource,
    GradientSeries,
    JSONLDataSource,
    LossSeries,
    MonitorDataSource,
    ReporterDataSource,
)
from qml_observer.dashboard.export import export_csv, export_json
from qml_observer.dashboard.history import RunHistoryEntry, discover_run_history

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = [
    "ComputeUsage",
    "DashboardDataSource",
    "GradientSeries",
    "JSONLDataSource",
    "LossSeries",
    "MonitorDataSource",
    "ReporterDataSource",
    "RunHistoryEntry",
    "create_app",
    "discover_run_history",
    "export_csv",
    "export_json",
    "run_dashboard",
]


def create_app(source: DashboardDataSource, *, history_dir: str | Path | None = None) -> FastAPI:
    """Lazily-imported wrapper around `dashboard.app.create_app` -- see
    that module for the real implementation and docstring."""
    from qml_observer.dashboard.app import create_app as _create_app

    return _create_app(source, history_dir=history_dir)


def run_dashboard(source: DashboardDataSource, **kwargs: Any) -> None:
    """Lazily-imported wrapper around `dashboard.server.run_dashboard` --
    see that module for the real implementation and docstring."""
    from qml_observer.dashboard.server import run_dashboard as _run_dashboard

    _run_dashboard(source, **kwargs)
