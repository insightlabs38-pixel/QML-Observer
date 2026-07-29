"""Milestone 15, Issue #105 -- basic per-step overhead / memory benchmark.

Pulled forward into the Milestone 9 timeframe per
`future_milestones_plan.md`'s "Gaps & recommendations" #5: "I'd actually
pull a basic version of Issue #105 forward before Milestone 9, not leave
all of it for the final hardening patch -- if there's a real per-step
overhead problem, better to find it while the codebase is still this
size."

This is deliberately the *basic* version plan.md §26 and the addendum's
testing strategy call for: per-step wall-clock overhead of
`QMLMonitor.update()` (bare vs. the full Milestone 9 detector set, with
and without JSONL logging) and peak memory over a few thousand steps.
It is NOT the full Milestone 15 pass -- a 100k+-step soak test and a
CI-gated regression-tracking job remain for that later milestone (see
`docs/roadmap.md`).

The project's whole pitch is "saves compute, doesn't consume it"
(plan.md §26/§31); this is the first artifact that puts a number behind
that claim rather than asserting it.

Run with:
    python benchmarks/run_overhead_benchmark.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

# LogAction (Issue #34) writes through the "qml_observer.actions" logger;
# with no handler configured, Python's logging module falls back to a
# lastResort handler that prints WARNING+ records to stderr. That's
# reasonable default behavior for a real deployment, but floods this
# benchmark's output with one line per step -- silence it here so the
# benchmark's own report is what's visible, not the underlying (correct,
# unrelated) diagnosis stream this fixture happens to produce.
logging.getLogger("qml_observer.actions").setLevel(logging.CRITICAL + 1)

# Reuse the same synthetic-fixture generator the calibration benchmarks
# use, so "a moderate-length, realistic-shaped run" means the same thing
# across this module and `run_benchmarks.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from fixtures.synthetic_runs import healthy_learning_run  # noqa: E402

from qml_observer import QMLMonitor  # noqa: E402
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector  # noqa: E402
from qml_observer.detectors.base import BaseDetector  # noqa: E402
from qml_observer.detectors.convergence import ConvergenceDetector  # noqa: E402
from qml_observer.detectors.noise import NoiseDetector  # noqa: E402
from qml_observer.detectors.stagnation import StagnationDetector  # noqa: E402
from qml_observer.reporting.jsonl import JSONLWriter, event_record  # noqa: E402
from qml_observer.schemas.training import TrainingEvent  # noqa: E402

DEFAULT_N_STEPS = 2000
DEFAULT_N_WARMUP = 50
DEFAULT_PATIENCE = 15


def _full_detector_set(patience: int = DEFAULT_PATIENCE) -> list[BaseDetector]:
    """The same canonical four-detector set `benchmarks/run_benchmarks.py`
    uses (Issue #69b), so this overhead number reflects what a real
    `QMLMonitor` deployment actually runs, not a stripped-down stand-in.
    """
    return [
        BarrenPlateauDetector(patience=patience),
        StagnationDetector(patience=patience),
        ConvergenceDetector(patience=patience),
        NoiseDetector(patience=patience),
    ]


@dataclass
class OverheadResult:
    """Per-scenario overhead measurement."""

    label: str
    n_steps: int
    total_seconds: float
    mean_us_per_step: float
    median_us_per_step: float
    p95_us_per_step: float
    peak_memory_kb: float
    steps_per_second: float


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct))
    return sorted_values[idx]


def _bench_one(
    label: str,
    monitor: QMLMonitor,
    steps: list[dict[str, object]],
    n_warmup: int,
    log_writer: JSONLWriter | None = None,
) -> OverheadResult:
    """Time `n_steps` calls to `monitor.update()`, excluding a warm-up prefix.

    Warm-up steps are excluded from timing because the first `patience`
    or so steps of any detector involve rolling-window fill behavior
    (`RollingWindow.append` before it's at `maxlen`) that isn't
    representative of steady-state per-step cost.
    """
    warmup_steps = steps[:n_warmup]
    timed_steps = steps[n_warmup:]

    for step in warmup_steps:
        monitor.update(**step)

    gc.collect()
    tracemalloc.start()
    per_step_seconds: list[float] = []
    start = time.perf_counter()
    for i, step in enumerate(timed_steps):
        t0 = time.perf_counter()
        monitor.update(**step)
        if log_writer is not None:
            event = TrainingEvent(run_id="overhead-benchmark", step=i, loss=step.get("loss"))
            log_writer.write(event_record(event))
        per_step_seconds.append(time.perf_counter() - t0)
    total = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    per_step_seconds.sort()
    n = len(per_step_seconds)
    return OverheadResult(
        label=label,
        n_steps=n,
        total_seconds=total,
        mean_us_per_step=1e6 * sum(per_step_seconds) / n,
        median_us_per_step=1e6 * _percentile(per_step_seconds, 0.5),
        p95_us_per_step=1e6 * _percentile(per_step_seconds, 0.95),
        peak_memory_kb=peak / 1024,
        steps_per_second=n / total if total > 0 else float("inf"),
    )


def run_overhead_benchmark(
    n_steps: int = DEFAULT_N_STEPS,
    n_warmup: int = DEFAULT_N_WARMUP,
    patience: int = DEFAULT_PATIENCE,
) -> dict[str, OverheadResult]:
    """Basic per-step overhead/memory benchmark (Issue #105, basic version).

    Three scenarios, all fed the *same* underlying fixture data (so
    differences reflect monitor/detector/logging overhead, not scenario
    variance):

    - `no_detectors`: `QMLMonitor` with an empty detector list -- the
      floor cost of `update()`'s own bookkeeping (state recording,
      `RunState`, event construction) with no detection work at all.
    - `full_detector_set`: the canonical four-detector set (Issue #69b),
      i.e. what a real deployment following this project's own
      documentation would run.
    - `full_detector_set_with_jsonl_logging`: the same detector set, with
      every step also written to a `JSONLWriter` (Issue #48) -- isolating
      logging overhead from detection overhead, per plan.md §26's
      "batch logging where possible" performance rule (this establishes
      whether the *current*, unbatched, flush-every-write logging
      strategy already costs enough to be worth revisiting).
    """
    results: dict[str, OverheadResult] = {}

    steps = healthy_learning_run(n_steps=n_steps + n_warmup, seed=0)

    results["no_detectors"] = _bench_one(
        "no_detectors", QMLMonitor(detectors=[], policy="log"), steps, n_warmup
    )
    results["full_detector_set"] = _bench_one(
        "full_detector_set",
        QMLMonitor(detectors=_full_detector_set(patience), policy="log"),
        steps,
        n_warmup,
    )

    with tempfile.TemporaryDirectory() as tmp:
        writer = JSONLWriter(Path(tmp) / "overhead_benchmark_run.jsonl")
        try:
            results["full_detector_set_with_jsonl_logging"] = _bench_one(
                "full_detector_set_with_jsonl_logging",
                QMLMonitor(detectors=_full_detector_set(patience), policy="log"),
                steps,
                n_warmup,
                log_writer=writer,
            )
        finally:
            writer.close()

    return results


def _print_report(results: dict[str, OverheadResult]) -> None:
    print("Per-step overhead benchmark (Milestone 15, Issue #105 -- basic version)")
    print("=" * 74)
    print(
        f"  {'scenario':>36s}  {'mean us':>9s}  {'median us':>10s}  "
        f"{'p95 us':>8s}  {'steps/s':>10s}  {'peak KB':>9s}"
    )
    for r in results.values():
        print(
            f"  {r.label:>36s}  {r.mean_us_per_step:>9.1f}  {r.median_us_per_step:>10.1f}  "
            f"{r.p95_us_per_step:>8.1f}  {r.steps_per_second:>10.0f}  {r.peak_memory_kb:>9.1f}"
        )
    print()
    print(
        "No hard performance target is set for v0.1 -- this establishes a "
        "reproducible baseline (same convention as Issue #54's detection-"
        "latency benchmark), not a pass/fail gate. A 100k+-step soak test "
        "and CI-gated regression tracking remain for the full Milestone 15 "
        "pass."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--n-warmup", type=int, default=DEFAULT_N_WARMUP)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write results as JSON.",
    )
    args = parser.parse_args(argv)

    results = run_overhead_benchmark(args.n_steps, args.n_warmup, args.patience)
    _print_report(results)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump({k: asdict(v) for k, v in results.items()}, fh, indent=2)
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
