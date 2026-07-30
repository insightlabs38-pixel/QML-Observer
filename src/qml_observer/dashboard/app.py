"""Dashboard FastAPI app (Issue #76 scaffold; routes for Issues #77-#82).

Architecture decision (Issue #76): a minimal FastAPI backend serving JSON
read from a `DashboardDataSource` (`dashboard/data_source.py`), plus a
small static frontend (`dashboard/static/`) that polls those JSON
endpoints and renders plain `<canvas>` charts with a tiny vendored
drawing helper -- no CDN dependency, no new frontend build toolchain, and
nothing that requires internet access to render (consistent with the
project's "nothing leaves your machine by default" posture,
`docs/development/data_handling.md`). This is *not* a new
framework-specific SDK: it reads exactly the data `RunReporter`/
`QMLMonitor`/JSONL logs already produce (see that module's docstring).

Shipped as an optional `qml-observer[dashboard]` extra (`fastapi` +
`uvicorn`) so the core install stays as light as it is today -- importing
this module without those installed raises a clear `ImportError` with
installation instructions, rather than a cryptic one.

Milestone 11 is explicitly sequenced *after* the detection layer is
trustworthy (addendum's stated ordering, `docs/roadmap.md`), and the
dashboard here is deliberately read-only: it has no route that mutates
monitor/run state (no way to trigger stop/pause/recovery from the UI).

`history_dir` (Issue #81, `dashboard/history.py`) is optional and
separate from `source`: `source` is always the single "currently viewed"
run, while `history_dir` (if given) points at a directory of *other*,
already-finalized runs' JSONL logs to list alongside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, Response
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - exercised via import-error test
    raise ImportError(
        "The QML Observer dashboard requires the optional 'dashboard' extra. "
        "Install it with: pip install 'qml-observer[dashboard]'"
    ) from exc

from qml_observer import __version__
from qml_observer.dashboard.data_source import DashboardDataSource
from qml_observer.dashboard.export import export_csv, export_json
from qml_observer.dashboard.history import discover_run_history

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(source: DashboardDataSource, *, history_dir: str | Path | None = None) -> FastAPI:
    """Build the dashboard FastAPI app reading from `source`.

    Args:
        source: Any `DashboardDataSource` (`MonitorDataSource`,
            `ReporterDataSource`, or `JSONLDataSource`) to read run data
            from. The app holds a reference and re-reads it on every
            request -- it never caches a snapshot, so the served data is
            always as fresh as `source` currently is.
        history_dir: Optional directory of other, already-finalized runs'
            JSONL logs to list via `GET /api/history` (Issue #81). `None`
            (the default) means no history is shown -- this is an
            explicit opt-in, not an assumed default directory, since
            scanning an arbitrary directory the caller didn't name would
            be surprising.

    Returns:
        A configured `FastAPI` app. Serve it with
        `dashboard.server.run_dashboard(source)`, or with any ASGI server
        of the caller's choosing (e.g. `uvicorn.run(app)`).
    """
    app = FastAPI(
        title="QML Observer Dashboard",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        """Lightweight liveness/info endpoint."""
        return {"ok": True, "version": __version__, "run_id": source.run_id()}

    @app.get("/api/loss")
    def loss() -> JSONResponse:
        """Loss-over-step series for the live loss chart (Issue #77)."""
        return JSONResponse(source.loss_series().to_dict())

    @app.get("/api/gradient")
    def gradient() -> JSONResponse:
        """Gradient-statistics series for the gradient chart (Issue #78).

        `available: false` (empty series) is a normal, expected response
        when the underlying source has no per-step gradient detail (e.g.
        a `ReporterDataSource`) -- not an error.
        """
        series = source.gradient_series()
        payload = series.to_dict()
        payload["available"] = series.available
        return JSONResponse(payload)

    @app.get("/api/diagnosis")
    def diagnosis() -> JSONResponse:
        """Current diagnosis for the diagnosis panel (Issue #79).

        Surfaces `degraded`/`degraded_reason` verbatim from
        `DiagnosisResult` (addendum §1) -- the frontend renders the
        "DIAGNOSIS DEGRADED" flag whenever `degraded` is true, matching
        the CLI's `qml-observer report` behavior rather than silently
        presenting a possibly-incomplete diagnosis as trustworthy.
        """
        result = source.diagnosis()
        return JSONResponse(result if result is not None else {})

    @app.get("/api/compute")
    def compute() -> JSONResponse:
        """Compute-usage panel data (Issue #80): planned vs. actual steps
        and the addendum §11 compute-saved estimate."""
        return JSONResponse(source.compute_usage().to_dict())

    @app.get("/api/history")
    def history() -> JSONResponse:
        """Run-history table across other finalized runs (Issue #81).

        Empty `entries` with `directory: null` (not an error) when
        `history_dir` wasn't configured for this app -- see `create_app`.
        """
        if history_dir is None:
            return JSONResponse({"directory": None, "entries": []})
        entries = discover_run_history(history_dir)
        return JSONResponse(
            {"directory": str(history_dir), "entries": [e.to_dict() for e in entries]}
        )

    @app.get("/api/export.json")
    def export_json_route() -> Response:
        """Download the currently-viewed run's full dashboard data as one
        JSON document (Issue #82)."""
        run_id = source.run_id() or "run"
        return Response(
            content=export_json(source),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
        )

    @app.get("/api/export.csv")
    def export_csv_route() -> Response:
        """Download the currently-viewed run's per-step loss/gradient
        series as a CSV table (Issue #82). Diagnosis and compute-usage
        fields aren't step-indexed, so use `/api/export.json` for those --
        see `dashboard/export.py`."""
        run_id = source.run_id() or "run"
        return Response(
            content=export_csv(source),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
        )

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))

    return app
