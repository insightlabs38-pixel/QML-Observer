"""Qubit- and depth-scaling analysis of gradient variance.

Milestone 12 (blueprint Volume XIII, "Scaling analysis"), Issues #88-#89,
completing Milestone 12 alongside the geometry diagnostics in
`qml_observer.advanced.geometry` (Issues #83-#87).

## Mathematical description

Barren-plateau theory (McClean et al. 2018 and the substantial body of
follow-up work) predicts that, for a broad class of expressive/deep
ansätze, the *variance* of a cost-function gradient component shrinks
exponentially with the number of qubits `n`:

    Var[d C / d theta_i] ~ b * exp(-a * n),   a > 0

Taking logs turns this into a linear relationship:

    log(Var[...]) ~ log(b) - a * n

`ScalingAnalyzer.analyze_qubit_scaling` fits exactly this line -- ordinary
least-squares regression of `log(gradient_variance)` against `n_qubits`
across a set of runs -- and reports whether the fit is consistent with
the theory's signature: a **negative** slope (variance shrinking as
qubit count grows) with a reasonably **tight** fit (`r_squared` above a
threshold). The same regression, substituting circuit `depth` for
`n_qubits`, is `analyze_depth_scaling` -- some barren-plateau mechanisms
(e.g. sufficiently deep random circuits approaching a 2-design) predict
depth-driven decay too, not only qubit-count-driven decay.

## What this is *not*

Per the blueprint's second and third architectural rules (detection/
diagnosis separation; falsifiability), `ScalingAnalyzer` is a research
utility that reports a regression fit, never a verdict like "this ansatz
has a barren plateau." A tight negative-slope fit is *consistent with*
exponential-decay barren-plateau behavior, not proof of it (a small,
non-representative set of runs can produce a tight negative-slope fit by
chance, or by a different mechanism, e.g. compounding measurement/shot
noise rather than a true expressivity effect) -- see Known Limitations in
`docs/research/geometry.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from qml_observer.schemas._validation import check_non_negative_int, check_type


@dataclass
class ScalingObservation:
    """One run's structural size and observed gradient-variance summary.

    A deliberately minimal, framework-agnostic record -- just enough for
    `ScalingAnalyzer` to regress against -- rather than requiring a full
    `RunState`/`DiagnosisResult`. Build these directly, or via
    `scaling_observation_from_run_summary` from an existing
    `reporting.summary.build_run_summary()` dict.

    Attributes:
        n_qubits: Number of qubits in the circuit this observation came
            from. Must be `>= 1`.
        depth: Circuit depth. Optional -- only required by
            `analyze_depth_scaling`, not `analyze_qubit_scaling`.
        gradient_variance: The observed gradient variance for this run
            (e.g. `GradientSnapshot.variance`, or a variance computed
            across several gradient components/steps by the caller).
            Must be `>= 0`; per addendum §7's numerical-edge-case
            convention, exactly `0.0` (a fully collapsed gradient) is a
            legitimate, meaningful observation, not rejected -- see
            `variance_floor` on the analyzer methods for how `log(0)` is
            handled.
        n_parameters: Optional -- the circuit's parameter count, kept
            alongside for context/reporting only; not used in the
            regression itself.
        label: Optional free-text identifier for this run (e.g. a run ID
            or ansatz name), carried through to
            `ScalingAnalysisResult.labels` for traceability, not used in
            the regression itself.
    """

    n_qubits: int
    gradient_variance: float
    depth: int | None = None
    n_parameters: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        check_type(self.n_qubits, int, "n_qubits")
        if isinstance(self.n_qubits, bool) or self.n_qubits < 1:
            raise ValueError(f"n_qubits must be an int >= 1, got {self.n_qubits!r}")
        check_non_negative_int(self.depth, "depth")
        check_non_negative_int(self.n_parameters, "n_parameters")
        check_type(self.gradient_variance, (int, float), "gradient_variance")
        if isinstance(self.gradient_variance, bool):
            raise TypeError("gradient_variance must be a number, got bool")
        if self.gradient_variance < 0 and not np.isnan(self.gradient_variance):
            raise ValueError(
                f"gradient_variance must be >= 0 (or NaN), got {self.gradient_variance}"
            )


@dataclass
class ScalingAnalysisResult:
    """Result of an ordinary-least-squares log-variance-vs-size regression.

    Attributes:
        x_label: Which structural variable was regressed against --
            `"n_qubits"` or `"depth"`.
        x_values: The structural-size values used, in the order fit
            (ascending).
        gradient_variance: The corresponding raw (non-log) gradient
            variances, same order as `x_values`.
        log_variance: `log(clip(gradient_variance, variance_floor,
            None))` -- the values actually regressed against `x_values`.
        slope: Fitted slope of `log_variance` vs. `x_values`. Negative
            means variance shrinks as the structural variable grows.
        intercept: Fitted intercept (`log(b)` in the module docstring's
            notation).
        r_squared: Coefficient of determination of the fit, in `[0, 1]`
            (`1.0` for a perfect fit, `0.0` for a fit no better than the
            mean). `nan` if fewer than 3 points are supplied (see
            `ScalingAnalyzer`'s minimum-points requirement -- 2 points
            always fit a line exactly, making `r_squared` uninformative;
            reported as `nan` rather than a misleading `1.0`).
        n_points: Number of `(x, variance)` points used after collapsing
            duplicate `x` values (see `ScalingAnalyzer.__init__`'s
            `aggregate` parameter).
        consistent_with_exponential_decay: `True` iff `slope < 0` and (
            `n_points < 3` or `r_squared >= r_squared_threshold`) -- see
            the class-level caveat on this field's meaning in
            `ScalingAnalyzer`'s docstring; it is a **pattern-consistency
            flag**, not a claim of barren-plateau proof.
        labels: The `ScalingObservation.label` values, same order as
            `x_values`, for traceability back to source runs (`None`
            entries where a run had no label).
    """

    x_label: str
    x_values: np.ndarray
    gradient_variance: np.ndarray
    log_variance: np.ndarray
    slope: float
    intercept: float
    r_squared: float
    n_points: int
    consistent_with_exponential_decay: bool
    labels: list[str | None]


def scaling_observation_from_run_summary(
    summary: dict[str, Any], label: str | None = None
) -> ScalingObservation:
    """Build a `ScalingObservation` from a `reporting.summary.build_run_summary()` dict.

    Convenience bridge for the common case of running several full
    `QMLMonitor` sessions (one per qubit count / depth) and feeding their
    summaries straight into `ScalingAnalyzer`, without hand-extracting
    `circuit`/`gradient` fields each time.

    Args:
        summary: A dict as returned by `build_run_summary`, i.e. with
            `summary["circuit"]["n_qubits"]`/`["depth"]` and
            `summary["gradient"]["variance"]` keys present (as
            `circuit_metadata_to_dict`/`gradient_snapshot_to_dict`
            produce).
        label: Optional label to attach (e.g. `summary["run_id"]`, passed
            explicitly since the caller may prefer a different label than
            the raw run ID).

    Returns:
        A `ScalingObservation` built from the summary's circuit/gradient
        detail.

    Raises:
        ValueError: If `summary["circuit"]` or `summary["gradient"]` is
            missing/`None`, or if `n_qubits`/`variance` within them is
            `None` -- all indicating the run summary lacks the structural
            or gradient detail this analysis needs (e.g. it came from a
            run with no circuit metadata attached).
    """
    circuit = summary.get("circuit")
    gradient = summary.get("gradient")
    if not circuit or circuit.get("n_qubits") is None:
        raise ValueError(
            "summary['circuit']['n_qubits'] is required but missing/None -- "
            "was CircuitMetadata attached to this run?"
        )
    if not gradient or gradient.get("variance") is None:
        raise ValueError(
            "summary['gradient']['variance'] is required but missing/None -- "
            "was a GradientSnapshot recorded for this run?"
        )
    return ScalingObservation(
        n_qubits=circuit["n_qubits"],
        depth=circuit.get("depth"),
        gradient_variance=gradient["variance"],
        n_parameters=circuit.get("n_parameters"),
        label=label if label is not None else summary.get("run_id"),
    )


def _ols_log_linear_fit(
    x: np.ndarray, variance: np.ndarray, variance_floor: float
) -> tuple[np.ndarray, float, float, float]:
    log_variance = np.log(np.clip(variance, variance_floor, None))
    slope, intercept = np.polyfit(x, log_variance, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((log_variance - predicted) ** 2))
    ss_tot = float(np.sum((log_variance - np.mean(log_variance)) ** 2))
    if ss_tot > 0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        # Every log-variance value is identical: a perfect fit iff the
        # (necessarily flat) line matches exactly, which it always does
        # for an OLS fit against a constant target (ss_res is also 0).
        r_squared = 1.0
    return log_variance, float(slope), float(intercept), r_squared


class ScalingAnalyzer:
    """Fits gradient-variance-vs-size regressions across a set of runs.

    Args:
        r_squared_threshold: Minimum `r_squared` (for fits with `>= 3`
            points) for `consistent_with_exponential_decay` to be `True`
            alongside a negative slope. Default `0.7` is a placeholder,
            not empirically calibrated against a benchmark corpus of
            known-plateauing vs. known-healthy ansatz families -- see
            Known Limitations in `docs/research/geometry.md`. Unlike
            Milestone 7's detector thresholds, no such corpus yet exists
            for this analyzer to be calibrated against.
        variance_floor: Value `gradient_variance` is clipped to (from
            below) before taking `log`, so an exactly-zero variance
            (a fully collapsed gradient -- a legitimate, meaningful
            observation, not a caller error, per addendum §7) contributes
            a large-magnitude but finite log-variance point rather than
            raising or silently producing `-inf`/`nan`.
        aggregate: How to combine multiple observations that share the
            same `x` value (e.g. several seeds at the same qubit count).
            `"mean"` (default) averages their `gradient_variance` before
            fitting -- the standard choice for reducing seed-to-seed
            noise in a scaling trend; `"none"` keeps every point
            (statistically double-counts repeated `x` values in the
            regression, but preserves the full raw scatter for the
            caller's own downstream plotting).
    """

    #: See the comment at its use site in `_analyze` for why this exists.
    _SLOPE_EPSILON = 1e-9

    def __init__(
        self,
        r_squared_threshold: float = 0.7,
        variance_floor: float = 1e-300,
        aggregate: str = "mean",
    ) -> None:
        if not (0.0 <= r_squared_threshold <= 1.0):
            raise ValueError(f"r_squared_threshold must be in [0, 1], got {r_squared_threshold}")
        if variance_floor <= 0:
            raise ValueError(f"variance_floor must be > 0, got {variance_floor}")
        if aggregate not in ("mean", "none"):
            raise ValueError(f"aggregate must be 'mean' or 'none', got {aggregate!r}")
        self.r_squared_threshold = r_squared_threshold
        self.variance_floor = variance_floor
        self.aggregate = aggregate

    def _analyze(self, runs: Sequence[ScalingObservation], x_label: str) -> ScalingAnalysisResult:
        if len(runs) < 2:
            raise ValueError(
                f"at least 2 runs with distinct {x_label} values are required, got {len(runs)}"
            )

        def get_x(run: ScalingObservation) -> int:
            value = getattr(run, x_label)
            if value is None:
                raise ValueError(
                    f"run {run.label!r} has no '{x_label}' value; "
                    f"{x_label} is required for this analysis"
                )
            return value

        points: list[tuple[int, float, str | None]] = [
            (get_x(run), run.gradient_variance, run.label) for run in runs
        ]

        if self.aggregate == "mean":
            by_x: dict[int, list[tuple[float, str | None]]] = {}
            for x_val, variance, label in points:
                by_x.setdefault(x_val, []).append((variance, label))
            aggregated = sorted(by_x.items())
            x_values = np.array([x for x, _ in aggregated], dtype=float)
            gradient_variance = np.array(
                [np.mean([v for v, _ in group]) for _, group in aggregated]
            )
            labels: list[str | None] = [
                "+".join(sorted(str(lbl) for _, lbl in group if lbl is not None)) or None
                for _, group in aggregated
            ]
        else:
            points.sort(key=lambda p: p[0])
            x_values = np.array([p[0] for p in points], dtype=float)
            gradient_variance = np.array([p[1] for p in points], dtype=float)
            labels = [p[2] for p in points]

        n_points = x_values.size
        if len(set(x_values.tolist())) < 2:
            raise ValueError(
                f"at least 2 distinct {x_label} values are required for a regression, "
                f"got only {set(x_values.tolist())}"
            )

        log_variance, slope, intercept, r_squared = _ols_log_linear_fit(
            x_values, gradient_variance, self.variance_floor
        )
        # A slope of exactly (or numerically indistinguishable from) zero
        # must never register as "decay": floating-point OLS solvers can
        # return a tiny nonzero value (e.g. ~1e-18) of *either* sign for
        # perfectly (or near-perfectly) flat input, which would otherwise
        # flip a genuinely flat trend to "consistent" purely from rounding
        # noise. `_SLOPE_EPSILON` is many orders of magnitude below any
        # real barren-plateau-scale log-variance slope (typically
        # O(0.01)-O(1) per unit of n_qubits/depth).
        is_meaningfully_negative = slope < -self._SLOPE_EPSILON
        if n_points < 3:
            reported_r_squared = float("nan")
            consistent = is_meaningfully_negative
        else:
            reported_r_squared = r_squared
            consistent = is_meaningfully_negative and r_squared >= self.r_squared_threshold

        return ScalingAnalysisResult(
            x_label=x_label,
            x_values=x_values,
            gradient_variance=gradient_variance,
            log_variance=log_variance,
            slope=slope,
            intercept=intercept,
            r_squared=reported_r_squared,
            n_points=n_points,
            consistent_with_exponential_decay=consistent,
            labels=labels,
        )

    def analyze_qubit_scaling(self, runs: Sequence[ScalingObservation]) -> ScalingAnalysisResult:
        """Regress `log(gradient_variance)` against `n_qubits` across `runs`.

        Args:
            runs: At least 2 `ScalingObservation`s spanning at least 2
                distinct `n_qubits` values. For a meaningful result,
                `depth` (and ansatz structure generally) should be held
                reasonably fixed across `runs` -- this method does not
                verify or control for that; it is the caller's
                responsibility (see module docstring).

        Returns:
            A `ScalingAnalysisResult` with `x_label="n_qubits"`.

        Raises:
            ValueError: If fewer than 2 runs, fewer than 2 distinct
                `n_qubits` values, or if empty.
        """
        return self._analyze(runs, "n_qubits")

    def analyze_depth_scaling(self, runs: Sequence[ScalingObservation]) -> ScalingAnalysisResult:
        """Regress `log(gradient_variance)` against `depth` across `runs`.

        Args:
            runs: At least 2 `ScalingObservation`s with `depth` set,
                spanning at least 2 distinct `depth` values. For a
                meaningful result, `n_qubits` should be held reasonably
                fixed across `runs` -- not verified/controlled for here,
                same caveat as `analyze_qubit_scaling`.

        Returns:
            A `ScalingAnalysisResult` with `x_label="depth"`.

        Raises:
            ValueError: If fewer than 2 runs, fewer than 2 distinct
                `depth` values, if empty, or if any run has `depth=None`.
        """
        return self._analyze(runs, "depth")
