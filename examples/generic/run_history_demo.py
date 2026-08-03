"""Milestone 14, Issue #103b: run comparison / experiment management.

Simulates an A/B test between two ansatzes (a plan.md §25 use case
explicitly): runs several short training runs tagged by which ansatz they
used, then compares them via `RunHistory`.

Run with:
    python examples/generic/run_history_demo.py

Afterward, try the same comparison from the CLI:
    qml-observer history list /tmp/qml_observer_history_demo.jsonl
    qml-observer history compare /tmp/qml_observer_history_demo.jsonl \
        --tag ansatz=hardware_efficient
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from qml_observer import QMLMonitor
from qml_observer.adapters.generic import GenericAdapter
from qml_observer.reporting.history import HistoryReporter, RunHistory, format_comparison_table

N_STEPS = 40
HISTORY_PATH = Path(tempfile.gettempdir()) / "qml_observer_history_demo.jsonl"


def _simulated_loss(step: int, ansatz: str) -> float:
    """A stand-in for real training: "hardware_efficient" converges faster."""
    if ansatz == "hardware_efficient":
        return math.exp(-0.15 * step) + 0.01
    return math.exp(-0.05 * step) + 0.05


def run_one(history: RunHistory, run_id: str, ansatz: str) -> None:
    reporter = HistoryReporter(
        history, tags={"ansatz": ansatz}, framework="pennylane", planned_steps=N_STEPS
    )
    monitor = QMLMonitor(policy="log", planned_steps=N_STEPS, reporter=reporter, run_id=run_id)
    adapter = GenericAdapter(monitor)
    for step in range(N_STEPS):
        adapter.record(step, loss=_simulated_loss(step, ansatz))
    monitor.finish()


def main() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    history = RunHistory(HISTORY_PATH)

    print(f"Recording runs to {HISTORY_PATH}\n")
    for i in range(3):
        run_one(history, run_id=f"hea-{i}", ansatz="hardware_efficient")
    for i in range(3):
        run_one(history, run_id=f"strong-ent-{i}", ansatz="strongly_entangling")

    print("All runs:")
    print(format_comparison_table(history.load_all()))

    print("\nOnly hardware_efficient runs:")
    hea_runs = history.filter_by_tag("ansatz", "hardware_efficient")
    print(format_comparison_table(hea_runs))

    csv_path = HISTORY_PATH.with_suffix(".csv")
    history.export_csv(csv_path)
    print(f"\nExported full ledger to {csv_path}")


if __name__ == "__main__":
    main()
