"""Milestone 14, Issue #101: MLflowTracker example.

Attaches `MLflowTracker` as `QMLMonitor`'s reporter so every step's loss
and the final diagnosis land in an MLflow run, alongside (not instead of)
whatever else your training script already logs to MLflow.

Run with:
    python examples/generic/mlflow_tracker_demo.py

Requires: pip install qml-observer[mlflow]

Uses a local sqlite-backed tracking store under ./mlruns.db purely so
this example has no external server to stand up -- point `tracking_uri`
at a real MLflow server the same way in a production setup.
"""

from __future__ import annotations

import math

import mlflow

from qml_observer import QMLMonitor
from qml_observer.adapters.generic import GenericAdapter
from qml_observer.integrations.trackers.mlflow_tracker import MLflowTracker

N_STEPS = 30


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow_demo.db")

    with mlflow.start_run(run_name="qml-observer-demo") as run:
        monitor = QMLMonitor(policy="log", reporter=MLflowTracker())
        adapter = GenericAdapter(monitor)

        print(f"MLflow run ID: {run.info.run_id}\n")
        for step in range(N_STEPS):
            loss = 1.0 / (step + 1) + 0.01 * math.sin(step)
            diagnosis = adapter.record(step, loss=loss)
            print(f"step={step:>2}  loss={loss: .4f}  issue={diagnosis.issue.value}")

        final = monitor.finish()
        print(f"\nFinal diagnosis: {final.issue.value} (confidence={final.confidence:.2f})")
        print("View this run with: mlflow ui --backend-store-uri sqlite:///mlflow_demo.db")


if __name__ == "__main__":
    main()
