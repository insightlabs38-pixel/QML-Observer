"""NoiseDetector.

Milestone 9 (Volume VI-4), Issue #66.

Distinguishes a gradient estimate that is *too noisy to trust yet* from
one that is genuinely small (`BarrenPlateauDetector`'s concern). The
blueprint sketches an MVP shape (`snr_threshold`, `patience`); this
implementation additionally consumes the `shots` field that has flowed
through `TrainingEvent`/`StepObservation` since Milestone 2, per the
blueprint's note that "later versions should incorporate shot-based
uncertainty."

Why this only ever looks at shot-based uncertainty
----------------------------------------------------
A naive "signal-to-noise" reading taken purely from a single step's
gradient array (e.g. `mean_abs / sqrt(variance)` across parameters) does
not actually separate noise from collapse for this project's gradient
model: every per-step gradient here is generated as an independent,
roughly zero-mean vector across parameters, so that ratio sits at
approximately the same value (~0.8, i.e. `sqrt(2/pi)`) whether the
gradient is large and healthy or collapsed to `1e-6`. It reflects the
*shape* of a zero-mean distribution, not whether the estimate is
trustworthy.

What actually determines trustworthiness is the shot budget: the same
observed gradient magnitude is either "clearly above the shot-noise
floor" or "statistically indistinguishable from shot noise" depending on
how many measurements it was estimated from. `NoiseDetector` therefore
computes, per step:

    uncertainty = estimate_measurement_uncertainty(grad.variance, shots)
    snr         = estimate_gradient_snr(grad.mean_abs, uncertainty)

treating the gradient's own per-parameter variance as a proxy for the
per-shot measurement variance feeding `estimate_measurement_uncertainty`
(the closest thing to a real per-shot variance available at the
`GradientSnapshot` level; see `schemas/gradient.py`). `mean_abs` (a
*per-parameter* magnitude) is deliberately used as the numerator rather
than `norm_l2` (the aggregate over all parameters): `norm_l2` scales with
`sqrt(n_parameters)` while `uncertainty` is a per-parameter quantity, so
comparing them directly would make the ratio grow with parameter count
alone, independent of shot budget -- exactly the kind of shape artifact
this detector exists to avoid. Comparing two per-parameter quantities
keeps the ratio meaningful (and roughly `sqrt(shots)`-scaled) regardless
of how many parameters the circuit has.

A genuinely collapsed gradient (`BarrenPlateauDetector`'s territory) has
both a small mean magnitude *and* a small variance, so `uncertainty`
shrinks right along with `mean_abs` and the ratio stays informative
regardless of shot count -- this detector correctly stays quiet in that
case. A gradient whose magnitude is small only *relative to how few shots
estimated it* is exactly the ambiguous case this detector exists to flag.

Steps with `shots is None` (analytic/infinite-shots execution) or
`shots <= 0` carry no shot-noise information at all: this detector
abstains entirely on those steps (they neither extend nor reset its
persistence streak) rather than guessing. This also means the detector
never fires at all on a purely analytic run -- which is the correct
behavior, since "shot noise" has no meaning there; a collapsed analytic
gradient is unambiguously `BarrenPlateauDetector`'s finding, not this
detector's.
"""

from __future__ import annotations

import math

from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.statistics.rolling import RollingWindow
from qml_observer.statistics.snr import estimate_gradient_snr, estimate_measurement_uncertainty


class NoiseDetector(BaseDetector):
    """Detects sustained, shot-noise-dominated (statistically unreliable) gradients."""

    name = "noise"

    def __init__(self, snr_threshold: float = 1.0, patience: int = 50):
        """Configure the detector.

        Args:
            snr_threshold: SNR at or below which a step's gradient
                estimate is considered statistically unreliable ("could
                easily be shot noise"). Placeholder default per addendum
                §3 -- to be tuned against the Milestone 9 benchmark
                fixtures (Issue #68), same as every other MVP threshold.
            patience: Number of consecutive shots-bearing steps the
                low-SNR condition must persist for before this detector
                triggers.

        Raises:
            ValueError: If `snr_threshold <= 0` or `patience < 1`.
        """
        if snr_threshold <= 0:
            raise ValueError(f"snr_threshold must be > 0, got {snr_threshold}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self._snr_threshold = snr_threshold
        self._patience = patience

        self._snr_values = RollingWindow(maxlen=patience)
        self._uncertainties = RollingWindow(maxlen=patience)
        self._consecutive_low_snr = 0
        self._n_shot_steps = 0

    def update(self, event: StepObservation, state: RunState) -> None:
        grad = event.gradient
        shots = event.shots

        if grad is None or shots is None or shots <= 0:
            # No shot-noise information this step (analytic execution, or
            # no gradient recorded at all): abstain, don't touch the
            # streak either way.
            return

        self._n_shot_steps += 1
        uncertainty = estimate_measurement_uncertainty(grad.variance, shots)
        snr = estimate_gradient_snr(grad.mean_abs, uncertainty)

        self._uncertainties.append(uncertainty if math.isfinite(uncertainty) else 0.0)
        self._snr_values.append(snr if math.isfinite(snr) else float(self._snr_threshold + 1))

        is_low_snr = math.isfinite(snr) and snr <= self._snr_threshold
        self._consecutive_low_snr = self._consecutive_low_snr + 1 if is_low_snr else 0

    def diagnose(self) -> DetectorResult:
        if len(self._snr_values) == 0:
            return DetectorResult(
                detector_name=self.name,
                triggered=False,
                confidence=0.0,
                evidence=[],
                recommendations=[],
            )

        latest_snr = self._snr_values.values()[-1]
        latest_uncertainty = self._uncertainties.values()[-1]
        persistence_ratio = min(self._consecutive_low_snr / self._patience, 1.0)

        evidence = [
            f"Latest gradient SNR: {latest_snr:.3f} (threshold {self._snr_threshold:.3f}).",
            f"Latest shot-noise uncertainty estimate: {latest_uncertainty:.3e}.",
            f"Low-SNR condition has persisted for {self._consecutive_low_snr} "
            f"consecutive shot-bearing step(s) (patience {self._patience}).",
            f"Observed over {self._n_shot_steps} step(s) with shot-count information.",
        ]

        triggered = self._consecutive_low_snr >= self._patience
        if triggered:
            confidence = min(1.0, 0.6 + 0.4 * persistence_ratio)
            recommendations = [
                "Gradient estimates appear dominated by shot noise rather than a "
                "genuine training signal. Consider increasing the shot budget, "
                "switching to analytic/adjoint differentiation if available, or "
                "averaging gradients over more measurements before concluding "
                "the gradient has truly collapsed (see BarrenPlateauDetector for "
                "that distinct question)."
            ]
        else:
            confidence = round(0.5 * persistence_ratio, 4)
            recommendations = []

        return DetectorResult(
            detector_name=self.name,
            triggered=triggered,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
        )

    def reset(self) -> None:
        self._snr_values.reset()
        self._uncertainties.reset()
        self._consecutive_low_snr = 0
        self._n_shot_steps = 0
