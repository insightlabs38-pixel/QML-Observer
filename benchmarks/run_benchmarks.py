"""QML Observer benchmark suite.

Milestone 7 (Volume XIX/plan.md §15), Issues #52-#55:

  #52 - Create benchmark notebook       -> benchmarks/qml_observer_benchmark.ipynb
        (this module is the runnable, non-notebook harness that notebook
        imports and calls, so results are identical in both places).
  #53 - Benchmark healthy convergence   -> `run_false_positive_benchmark()`
        against the `healthy_learning` and `convergence` fixtures.
  #54 - Benchmark artificial plateau    -> `run_detection_latency_benchmark()`
        against the `artificial_plateau` fixture.
  #55 - Benchmark normal convergence vs. plateau -> `run_full_benchmark()`,
        which runs both of the above plus `noise_dominated` (false-positive
        check) and prints/saves the combined comparison used to satisfy
        addendum §3's calibration acceptance criteria.

This intentionally reuses `tests/fixtures/synthetic_runs.py` (Milestone 4,
Issue #32) rather than a second, separate set of circuits: those fixtures
were built framework-agnostic for exactly this purpose (see that module's
docstring). The "depth scaling case" from plan.md §15 is out of scope here
per that same docstring -- it needs `ScalingAnalyzer` (Milestone 12) and is
not part of the MVP acceptance criteria.

Addendum §3 acceptance targets checked here:
  - False-positive rate on healthy/convergence fixtures: target < 5%.
  - Detection latency on artificial-plateau fixture: reported (median,
    p95 steps-to-detection), no hard target for v0.1 (baseline only).

Run with:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --seeds 50 \
        --json benchmarks/results/calibration_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: `LogAction` (Milestone 5) logs every diagnosis through the stdlib
#: `logging` module. That's the right default for a real training run, but
#: it drowns out this benchmark's own console report across hundreds of
#: seeded runs, so raise the threshold for just this module's logger.
logging.getLogger("qml_observer").setLevel(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from fixtures.synthetic_runs import (  # noqa: E402
    artificial_plateau_run,
    convergence_run,
    healthy_learning_run,
    noise_dominated_run,
)

from qml_observer import QMLMonitor  # noqa: E402
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector  # noqa: E402
from qml_observer.detectors.base import BaseDetector  # noqa: E402
from qml_observer.detectors.convergence import ConvergenceDetector  # noqa: E402
from qml_observer.detectors.stagnation import StagnationDetector  # noqa: E402
from qml_observer.schemas.diagnosis import IssueType  # noqa: E402

#: Detector defaults under evaluation. These are the addendum §3
#: "placeholder, not final" thresholds from each detector's own default
#: constructor arguments -- calibration means running this benchmark and
#: deciding whether to change them, not hand-tuning them here.
DEFAULT_PATIENCE = 15

#: A detector factory takes the shared `patience` window and returns a
#: fresh list of detector instances (detectors are stateful, so every
#: seeded run needs its own). Threaded explicitly through every function
#: below (rather than reassigned as a module global) so concurrent/nested
#: calls -- e.g. a notebook cell re-running a sweep -- can never observe
#: or clobber each other's detector configuration.
DetectorFactory = Callable[[int], list[BaseDetector]]


def _default_detectors(patience: int = DEFAULT_PATIENCE) -> list[BaseDetector]:
    return [
        BarrenPlateauDetector(patience=patience),
        StagnationDetector(patience=patience),
        ConvergenceDetector(patience=patience),
    ]


@dataclass
class ScenarioRunResult:
    """Outcome of feeding one seeded scenario run through a fresh monitor."""

    scenario: str
    seed: int
    flagged_plateau: bool
    flagged_stagnation: bool
    first_plateau_step: int | None
    final_issue: str
    final_confidence: float


def _run_one(
    scenario_name: str,
    generator: Callable[..., list[dict[str, object]]],
    seed: int,
    patience: int,
    detector_factory: DetectorFactory,
) -> ScenarioRunResult:
    monitor = QMLMonitor(detectors=detector_factory(patience), policy="log")
    steps = generator(seed=seed)

    first_plateau_step: int | None = None
    flagged_plateau = False
    flagged_stagnation = False
    diagnosis = None
    for i, step in enumerate(steps):
        diagnosis = monitor.update(**step)
        if diagnosis.issue == IssueType.POSSIBLE_BARREN_PLATEAU and not flagged_plateau:
            flagged_plateau = True
            first_plateau_step = i
        if diagnosis.issue == IssueType.STAGNATION:
            flagged_stagnation = True

    assert diagnosis is not None
    return ScenarioRunResult(
        scenario=scenario_name,
        seed=seed,
        flagged_plateau=flagged_plateau,
        flagged_stagnation=flagged_stagnation,
        first_plateau_step=first_plateau_step,
        final_issue=diagnosis.issue.value,
        final_confidence=diagnosis.confidence,
    )


def run_false_positive_benchmark(
    n_seeds: int = 30,
    patience: int = DEFAULT_PATIENCE,
    detector_factory: DetectorFactory = _default_detectors,
) -> dict[str, object]:
    """Issue #53: healthy/convergence fixtures must not be flagged as a plateau.

    Also runs `noise_dominated` (Issue #55's broader comparison) since it is
    the other "must not false-positive" fixture from plan.md §15.
    """
    results: dict[str, list[ScenarioRunResult]] = {}
    for name, generator in (
        ("healthy_learning", healthy_learning_run),
        ("convergence", convergence_run),
        ("noise_dominated", noise_dominated_run),
    ):
        results[name] = [
            _run_one(name, generator, seed, patience, detector_factory) for seed in range(n_seeds)
        ]

    summary = {}
    for name, runs in results.items():
        n_false_positive = sum(1 for r in runs if r.flagged_plateau)
        summary[name] = {
            "n_seeds": n_seeds,
            "n_false_positive": n_false_positive,
            "false_positive_rate": n_false_positive / n_seeds,
            "meets_target_lt_5pct": (n_false_positive / n_seeds) < 0.05,
        }
    return summary


def run_detection_latency_benchmark(
    n_seeds: int = 30,
    patience: int = DEFAULT_PATIENCE,
    detector_factory: DetectorFactory = _default_detectors,
) -> dict[str, object]:
    """Issue #54: steps-to-detection on the artificial-plateau fixture."""
    runs = [
        _run_one("artificial_plateau", artificial_plateau_run, seed, patience, detector_factory)
        for seed in range(n_seeds)
    ]
    detected = [r for r in runs if r.flagged_plateau]
    latencies = [r.first_plateau_step for r in detected if r.first_plateau_step is not None]

    summary: dict[str, object] = {
        "n_seeds": n_seeds,
        "n_detected": len(detected),
        "detection_rate": len(detected) / n_seeds,
    }
    if latencies:
        sorted_lat = sorted(latencies)
        p95_idx = min(len(sorted_lat) - 1, int(round(0.95 * (len(sorted_lat) - 1))))
        summary["median_steps_to_detection"] = statistics.median(sorted_lat)
        summary["p95_steps_to_detection"] = sorted_lat[p95_idx]
        summary["min_steps_to_detection"] = sorted_lat[0]
        summary["max_steps_to_detection"] = sorted_lat[-1]
    else:
        summary["median_steps_to_detection"] = None
        summary["p95_steps_to_detection"] = None
    return summary


def run_full_benchmark(
    n_seeds: int = 30,
    patience: int = DEFAULT_PATIENCE,
    detector_factory: DetectorFactory = _default_detectors,
) -> dict[str, object]:
    """Issue #55: the combined convergence-vs-plateau comparison.

    This is the single entry point the CLI/notebook call; it bundles both
    of the above so a single JSON artifact captures the full addendum §3
    calibration picture for a given detector configuration.
    """
    return {
        "config": {"n_seeds": n_seeds, "patience": patience},
        "false_positive": run_false_positive_benchmark(n_seeds, patience, detector_factory),
        "detection_latency": run_detection_latency_benchmark(n_seeds, patience, detector_factory),
    }


def run_calibration_sweep(
    candidate_gradient_thresholds: list[float],
    n_seeds: int = 30,
    patience: int = DEFAULT_PATIENCE,
) -> list[dict[str, object]]:
    """Addendum §3: sweep `BarrenPlateauDetector.gradient_threshold` and report,
    for each candidate, the false-positive rate on healthy/convergence/noise
    fixtures and the detection rate + latency on the artificial-plateau
    fixture -- the empirical basis for choosing a non-placeholder default.

    Each candidate builds its own `detector_factory` closure and passes it
    straight through to `run_false_positive_benchmark`/
    `run_detection_latency_benchmark` -- no shared/global state is mutated,
    so this is safe to call repeatedly or from concurrent notebook cells.
    """
    sweep_results = []
    for threshold in candidate_gradient_thresholds:

        def make_detectors(patience: int, threshold: float = threshold) -> list[BaseDetector]:
            return [
                BarrenPlateauDetector(patience=patience, gradient_threshold=threshold),
                StagnationDetector(patience=patience),
                ConvergenceDetector(patience=patience),
            ]

        fp = run_false_positive_benchmark(n_seeds, patience, make_detectors)
        latency = run_detection_latency_benchmark(n_seeds, patience, make_detectors)

        sweep_results.append(
            {
                "gradient_threshold": threshold,
                "false_positive": fp,
                "detection_latency": latency,
            }
        )
    return sweep_results


def _print_report(results: dict[str, object]) -> None:
    print("QML Observer — Calibration Benchmark")
    print("=" * 60)
    cfg = results["config"]
    print(f"seeds={cfg['n_seeds']}  patience={cfg['patience']}")
    print()
    print("False-positive rates (target: < 5%)")
    print("-" * 60)
    for name, stats in results["false_positive"].items():
        flag = "OK" if stats["meets_target_lt_5pct"] else "FAIL"
        print(
            f"  {name:20s} {stats['false_positive_rate']:6.1%} "
            f"({stats['n_false_positive']}/{stats['n_seeds']})  [{flag}]"
        )
    print()
    print("Artificial-plateau detection latency (steps to first flag)")
    print("-" * 60)
    lat = results["detection_latency"]
    print(f"  detection rate: {lat['detection_rate']:.1%} ({lat['n_detected']}/{lat['n_seeds']})")
    print(f"  median steps-to-detection: {lat['median_steps_to_detection']}")
    print(f"  p95 steps-to-detection:    {lat['p95_steps_to_detection']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the QML Observer calibration benchmark suite."
    )
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeded runs per scenario.")
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE, help="Detector patience window."
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help=(
            "Optional path to write results as JSON "
            "(e.g. benchmarks/results/calibration_results.json)."
        ),
    )
    args = parser.parse_args(argv)

    results = run_full_benchmark(n_seeds=args.seeds, patience=args.patience)
    _print_report(results)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"\nWrote results to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
