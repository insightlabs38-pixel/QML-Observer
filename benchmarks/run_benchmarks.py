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
    finite_shots_healthy_run,
    finite_shots_plateau_run,
    healthy_learning_run,
    noise_dominated_run,
)

from qml_observer import QMLMonitor  # noqa: E402
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector  # noqa: E402
from qml_observer.detectors.base import BaseDetector  # noqa: E402
from qml_observer.detectors.convergence import ConvergenceDetector  # noqa: E402
from qml_observer.detectors.noise import NoiseDetector  # noqa: E402
from qml_observer.detectors.stagnation import StagnationDetector  # noqa: E402
from qml_observer.schemas.diagnosis import IssueType  # noqa: E402

#: Detector defaults under evaluation. These are the addendum §3
#: "placeholder, not final" thresholds from each detector's own default
#: constructor arguments -- calibration means running this benchmark and
#: deciding whether to change them, not hand-tuning them here.
DEFAULT_PATIENCE = 15

#: Placeholder default per addendum §3, same status as every other MVP
#: threshold -- to be tuned against the sweep below.
DEFAULT_SNR_THRESHOLD = 1.0

#: Shot budgets swept by `run_noise_benchmark` (Issue #68's "varying shot
#: budgets"), from severely shot-starved to effectively analytic.
DEFAULT_SHOT_BUDGETS = (1, 5, 20, 100, 1000)

#: A detector factory takes the shared `patience` window and returns a
#: fresh list of detector instances (detectors are stateful, so every
#: seeded run needs its own). Threaded explicitly through every function
#: below (rather than reassigned as a module global) so concurrent/nested
#: calls -- e.g. a notebook cell re-running a sweep -- can never observe
#: or clobber each other's detector configuration.
DetectorFactory = Callable[[int], list[BaseDetector]]


def _default_detectors(
    patience: int = DEFAULT_PATIENCE, snr_threshold: float = DEFAULT_SNR_THRESHOLD
) -> list[BaseDetector]:
    """The canonical detector set used across every benchmark in this module.

    Issue #69b: `NoiseDetector` is included here (not just in the
    Milestone 9 finite-shots benchmark) so the Milestone 7 calibration
    sweep (`run_false_positive_benchmark`/`run_detection_latency_benchmark`
    /`run_calibration_sweep`) actually exercises the full Milestone 9
    detector set as an input to the diagnosis engine's scoring, per the
    addendum §3 concern that "adding a new signal to the same
    deterministic scoring function can shift false-positive rates on the
    existing fixtures, not just add a new issue type." See
    `run_reconciliation_check()` below for the empirical answer (spoiler:
    it doesn't shift them, for a specific, checked reason -- but that is
    now verified, not assumed).
    """
    return [
        BarrenPlateauDetector(patience=patience),
        StagnationDetector(patience=patience),
        ConvergenceDetector(patience=patience),
        NoiseDetector(patience=patience, snr_threshold=snr_threshold),
    ]


@dataclass
class ScenarioRunResult:
    """Outcome of feeding one seeded scenario run through a fresh monitor."""

    scenario: str
    seed: int
    flagged_plateau: bool
    flagged_stagnation: bool
    flagged_noise: bool
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
    flagged_noise = False
    diagnosis = None
    for i, step in enumerate(steps):
        diagnosis = monitor.update(**step)
        if diagnosis.issue == IssueType.POSSIBLE_BARREN_PLATEAU and not flagged_plateau:
            flagged_plateau = True
            first_plateau_step = i
        if diagnosis.issue == IssueType.STAGNATION:
            flagged_stagnation = True
        if diagnosis.issue == IssueType.NOISE_DOMINATED:
            flagged_noise = True

    assert diagnosis is not None
    return ScenarioRunResult(
        scenario=scenario_name,
        seed=seed,
        flagged_plateau=flagged_plateau,
        flagged_stagnation=flagged_stagnation,
        flagged_noise=flagged_noise,
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
        n_flagged_noise = sum(1 for r in runs if r.flagged_noise)
        summary[name] = {
            "n_seeds": n_seeds,
            "n_false_positive": n_false_positive,
            "false_positive_rate": n_false_positive / n_seeds,
            "meets_target_lt_5pct": (n_false_positive / n_seeds) < 0.05,
            # Issue #69b: reported alongside the plateau false-positive
            # rate so any shift caused by adding NoiseDetector to the
            # scoring inputs is visible here too, not just assumed absent.
            "n_flagged_noise_dominated": n_flagged_noise,
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
    include_noise_benchmark: bool = True,
    shot_budgets: tuple[int, ...] = DEFAULT_SHOT_BUDGETS,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    include_reconciliation_check: bool = True,
) -> dict[str, object]:
    """Issue #55: the combined convergence-vs-plateau comparison.

    This is the single entry point the CLI/notebook call; it bundles both
    of the above so a single JSON artifact captures the full addendum §3
    calibration picture for a given detector configuration. Also includes
    the Milestone 9 (Issue #68) finite-shots noise benchmark by default,
    since it uses the same fixture-sweep machinery and belongs in the same
    reproducible artifact, and the Issue #69b reconciliation check by
    default (whether adding `NoiseDetector` shifted any Milestone 7
    number).
    """
    results: dict[str, object] = {
        "config": {"n_seeds": n_seeds, "patience": patience},
        "false_positive": run_false_positive_benchmark(n_seeds, patience, detector_factory),
        "detection_latency": run_detection_latency_benchmark(n_seeds, patience, detector_factory),
    }
    if include_noise_benchmark:
        results["noise_shot_budget"] = run_noise_benchmark(
            shot_budgets, n_seeds, patience, snr_threshold
        )
    if include_reconciliation_check:
        results["reconciliation_check"] = run_reconciliation_check(n_seeds, patience)
    return results


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
                NoiseDetector(patience=patience, snr_threshold=DEFAULT_SNR_THRESHOLD),
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


@dataclass
class ShotBudgetRunResult:
    """Outcome of feeding one seeded finite-shots scenario through a fresh monitor."""

    scenario: str
    shots: int
    seed: int
    flagged_plateau: bool
    flagged_noise: bool
    final_issue: str
    final_confidence: float


def _run_one_with_shots(
    scenario_name: str,
    generator: Callable[..., list[dict[str, object]]],
    shots: int,
    seed: int,
    patience: int,
    snr_threshold: float,
) -> ShotBudgetRunResult:
    monitor = QMLMonitor(detectors=_default_detectors(patience, snr_threshold), policy="log")
    steps = generator(seed=seed, shots=shots)

    flagged_plateau = False
    flagged_noise = False
    diagnosis = None
    for step in steps:
        diagnosis = monitor.update(**step)
        if diagnosis.issue == IssueType.POSSIBLE_BARREN_PLATEAU:
            flagged_plateau = True
        if diagnosis.issue == IssueType.NOISE_DOMINATED:
            flagged_noise = True

    assert diagnosis is not None
    return ShotBudgetRunResult(
        scenario=scenario_name,
        shots=shots,
        seed=seed,
        flagged_plateau=flagged_plateau,
        flagged_noise=flagged_noise,
        final_issue=diagnosis.issue.value,
        final_confidence=diagnosis.confidence,
    )


def run_noise_benchmark(
    shot_budgets: tuple[int, ...] = DEFAULT_SHOT_BUDGETS,
    n_seeds: int = 30,
    patience: int = DEFAULT_PATIENCE,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
) -> dict[str, object]:
    """Issue #68: sweep shot budgets across the finite-shots fixtures.

    For each shot budget, runs both `finite_shots_healthy_run` (must not
    be flagged as a plateau; ideally not flagged as noise-dominated once
    shots are ample) and `finite_shots_plateau_run` (must still be
    detected as a plateau, never *only* explained away as noise) for
    `n_seeds` seeds, with the full Milestone 9 detector set
    (`BarrenPlateauDetector`, `StagnationDetector`, `ConvergenceDetector`,
    `NoiseDetector`) attached together, exactly as `QMLMonitor` would run
    them.
    """
    results: dict[str, object] = {}
    for shots in shot_budgets:
        healthy_runs = [
            _run_one_with_shots(
                "finite_shots_healthy",
                finite_shots_healthy_run,
                shots,
                seed,
                patience,
                snr_threshold,
            )
            for seed in range(n_seeds)
        ]
        plateau_runs = [
            _run_one_with_shots(
                "finite_shots_plateau",
                finite_shots_plateau_run,
                shots,
                seed,
                patience,
                snr_threshold,
            )
            for seed in range(n_seeds)
        ]

        n_healthy_fp_plateau = sum(1 for r in healthy_runs if r.flagged_plateau)
        n_healthy_flagged_noise = sum(1 for r in healthy_runs if r.flagged_noise)
        n_plateau_detected = sum(1 for r in plateau_runs if r.flagged_plateau)
        n_plateau_conflated_as_noise_only = sum(
            1 for r in plateau_runs if r.flagged_noise and not r.flagged_plateau
        )

        results[str(shots)] = {
            "shots": shots,
            "n_seeds": n_seeds,
            "healthy": {
                "false_positive_plateau_rate": n_healthy_fp_plateau / n_seeds,
                "flagged_noise_dominated_rate": n_healthy_flagged_noise / n_seeds,
            },
            "plateau": {
                "detection_rate": n_plateau_detected / n_seeds,
                "conflated_as_noise_only_rate": n_plateau_conflated_as_noise_only / n_seeds,
            },
        }
    return results


def _detectors_without_noise(patience: int = DEFAULT_PATIENCE) -> list[BaseDetector]:
    """The pre-Milestone-9, three-detector baseline.

    Kept only for `run_reconciliation_check()` below (Issue #69b) -- not
    used as a default anywhere else in this module.
    """
    return [
        BarrenPlateauDetector(patience=patience),
        StagnationDetector(patience=patience),
        ConvergenceDetector(patience=patience),
    ]


def run_reconciliation_check(
    n_seeds: int = 30, patience: int = DEFAULT_PATIENCE
) -> dict[str, object]:
    """Issue #69b: does adding `NoiseDetector` shift the Milestone 7 numbers?

    Addendum §3 flags this concern directly: "adding a new signal to the
    same deterministic scoring function can shift false-positive rates on
    the existing fixtures, not just add a new issue type." This runs the
    Milestone 7 false-positive and detection-latency benchmarks twice --
    once with the pre-Milestone-9 three-detector set
    (`_detectors_without_noise`), once with the current four-detector set
    that includes `NoiseDetector` (`_default_detectors`) -- against the
    exact same fixtures and seeds, and reports whether the numbers are
    identical. This is a checked fact, not an assumption: none of
    `healthy_learning_run`/`convergence_run`/`artificial_plateau_run`/
    `noise_dominated_run` attach a `shots` field, and `NoiseDetector`
    abstains entirely on steps without shot-count information
    (`detectors/noise.py`), so the expectation is that these numbers are
    unchanged -- but that expectation is exactly the kind of thing this
    project's own third technical rule (blueprint's closing section) says
    must be checked, not assumed.
    """
    before_fp = run_false_positive_benchmark(n_seeds, patience, _detectors_without_noise)
    after_fp = run_false_positive_benchmark(n_seeds, patience, _default_detectors)
    before_latency = run_detection_latency_benchmark(n_seeds, patience, _detectors_without_noise)
    after_latency = run_detection_latency_benchmark(n_seeds, patience, _default_detectors)

    fp_rate_unchanged = {
        name: before_fp[name]["false_positive_rate"] == after_fp[name]["false_positive_rate"]
        for name in before_fp
    }
    latency_unchanged = (
        before_latency["detection_rate"] == after_latency["detection_rate"]
        and before_latency["median_steps_to_detection"]
        == after_latency["median_steps_to_detection"]
        and before_latency["p95_steps_to_detection"] == after_latency["p95_steps_to_detection"]
    )

    return {
        "n_seeds": n_seeds,
        "before_noise_detector": {
            "false_positive": before_fp,
            "detection_latency": before_latency,
        },
        "after_noise_detector": {
            "false_positive": after_fp,
            "detection_latency": after_latency,
        },
        "false_positive_rate_unchanged_per_fixture": fp_rate_unchanged,
        "detection_latency_unchanged": latency_unchanged,
        "all_milestone_7_numbers_unchanged": all(fp_rate_unchanged.values()) and latency_unchanged,
    }


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

    if "noise_shot_budget" in results:
        print()
        print("Finite-shots noise benchmark (Milestone 9, Issue #68)")
        print("-" * 60)
        print(
            f"  {'shots':>8s}  {'healthy FP(plateau)':>20s}  "
            f"{'healthy->noise':>14s}  {'plateau detect':>14s}  {'plateau->noise only':>19s}"
        )
        for stats in results["noise_shot_budget"].values():
            print(
                f"  {stats['shots']:>8d}  "
                f"{stats['healthy']['false_positive_plateau_rate']:>19.1%}  "
                f"{stats['healthy']['flagged_noise_dominated_rate']:>13.1%}  "
                f"{stats['plateau']['detection_rate']:>13.1%}  "
                f"{stats['plateau']['conflated_as_noise_only_rate']:>18.1%}"
            )

    if "reconciliation_check" in results:
        rc = results["reconciliation_check"]
        print()
        print("Reconciliation check (Milestone 9, Issue #69b)")
        print("-" * 60)
        print("  Does adding NoiseDetector shift Milestone 7 numbers on the")
        print("  original (non-finite-shots) fixtures?")
        for name, unchanged in rc["false_positive_rate_unchanged_per_fixture"].items():
            mark = "unchanged" if unchanged else "CHANGED"
            print(f"    {name:>20s} false-positive rate: {mark}")
        mark = "unchanged" if rc["detection_latency_unchanged"] else "CHANGED"
        print(f"    {'artificial_plateau':>20s} detection latency:  {mark}")
        print(f"  All Milestone 7 numbers unchanged: {rc['all_milestone_7_numbers_unchanged']}")


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
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=DEFAULT_SNR_THRESHOLD,
        help="NoiseDetector SNR threshold used by the finite-shots noise benchmark.",
    )
    parser.add_argument(
        "--shot-budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_SHOT_BUDGETS),
        help="Shot budgets to sweep in the finite-shots noise benchmark.",
    )
    parser.add_argument(
        "--no-noise-benchmark",
        action="store_true",
        help="Skip the Milestone 9 finite-shots noise benchmark (Issue #68).",
    )
    parser.add_argument(
        "--no-reconciliation-check",
        action="store_true",
        help="Skip the Issue #69b before/after-NoiseDetector reconciliation check.",
    )
    args = parser.parse_args(argv)

    results = run_full_benchmark(
        n_seeds=args.seeds,
        patience=args.patience,
        include_noise_benchmark=not args.no_noise_benchmark,
        shot_budgets=tuple(args.shot_budgets),
        snr_threshold=args.snr_threshold,
        include_reconciliation_check=not args.no_reconciliation_check,
    )
    _print_report(results)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"\nWrote results to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
