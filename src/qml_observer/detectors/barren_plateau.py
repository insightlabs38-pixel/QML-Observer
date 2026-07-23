"""BarrenPlateauDetector.

Milestone 4 (Volume VI-1), Issue #26.

Algorithm, straight from the blueprint:

    IF gradient is small
    AND gradient variance is small
    AND loss is not improving
    AND condition persists
    THEN
        increase plateau confidence

Important (blueprint Volume VI-1 / plan.md §13): this reports a
*possible* barren plateau, never a definitive proof, and "a small
gradient alone should never be enough to" trigger -- loss stagnation
evidence is always required too, distinguishing this detector from a
naive single-metric threshold check. When no loss data is available at
all to confirm stagnation, this detector never triggers, regardless of
how small or persistent the gradient collapse is; it instead reports a
bounded, sub-triggering confidence so the `DiagnosisEngine` can still see
the trend without conflating "confirmed" with "grounds for suspicion".
"""

from __future__ import annotations

import math

from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.statistics.loss import relative_loss_improvement
from qml_observer.statistics.rolling import RollingWindow


class BarrenPlateauDetector(BaseDetector):
    """Detects sustained gradient collapse coincident with loss stagnation."""

    name = "barren_plateau"

    def __init__(
        self,
        gradient_threshold: float = 1e-8,
        variance_threshold: float | None = None,
        loss_improvement_threshold: float = 1e-6,
        patience: int = 100,
    ):
        """Configure the detector.

        Args:
            gradient_threshold: Gradient L2-norm at or below which a step
                is considered "small". Placeholder default per addendum
                §3 -- intended to be empirically calibrated against the
                benchmark suite, not treated as a final value.
            variance_threshold: Gradient variance at or below which a
                step is considered "small variance" (flat gradient
                distribution, not just a small mean). If `None` (the
                default), it is derived as `gradient_threshold ** 2`: a
                gradient distribution whose variance is at or below the
                square of the norm threshold is consistent with a
                collapsed/flat landscape at that same scale.
            loss_improvement_threshold: Relative loss-improvement
                magnitude below which the loss is considered stagnant
                (passed to `relative_loss_improvement`).
            patience: Number of consecutive steps the small-gradient
                condition must hold before this detector can trigger.

        Raises:
            ValueError: If any threshold is not positive (loss
                improvement threshold may be zero) or `patience < 1`.
        """
        if gradient_threshold <= 0:
            raise ValueError(f"gradient_threshold must be > 0, got {gradient_threshold}")
        if variance_threshold is not None and variance_threshold <= 0:
            raise ValueError(f"variance_threshold must be > 0, got {variance_threshold}")
        if loss_improvement_threshold < 0:
            raise ValueError(
                f"loss_improvement_threshold must be >= 0, got {loss_improvement_threshold}"
            )
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self._gradient_threshold = gradient_threshold
        self._variance_threshold = (
            variance_threshold if variance_threshold is not None else gradient_threshold**2
        )
        self._loss_improvement_threshold = loss_improvement_threshold
        self._patience = patience

        self._gradient_norms = RollingWindow(maxlen=patience)
        self._gradient_variances = RollingWindow(maxlen=patience)
        self._losses = RollingWindow(maxlen=patience)
        self._consecutive_small_gradient = 0

    def update(self, event: StepObservation, state: RunState) -> None:
        grad = event.gradient
        if grad is not None:
            self._gradient_norms.append(grad.norm_l2)
            self._gradient_variances.append(grad.variance)

            is_small = (
                math.isfinite(grad.norm_l2)
                and grad.norm_l2 <= self._gradient_threshold
                and math.isfinite(grad.variance)
                and grad.variance <= self._variance_threshold
            )
            self._consecutive_small_gradient = (
                self._consecutive_small_gradient + 1 if is_small else 0
            )

        loss = event.training_event.loss
        if loss is not None:
            self._losses.append(loss)

    def diagnose(self) -> DetectorResult:
        evidence: list[str] = []
        recommendations: list[str] = []

        if len(self._gradient_norms) == 0:
            return DetectorResult(
                detector_name=self.name,
                triggered=False,
                confidence=0.0,
                evidence=[],
                recommendations=[],
            )

        latest_norm = self._gradient_norms.values()[-1]
        latest_variance = self._gradient_variances.values()[-1]
        persistence_ratio = min(self._consecutive_small_gradient / self._patience, 1.0)

        evidence.append(
            f"Latest gradient norm: {latest_norm:.3e} (threshold {self._gradient_threshold:.1e})."
        )
        evidence.append(
            f"Latest gradient variance: {latest_variance:.3e} "
            f"(threshold {self._variance_threshold:.1e})."
        )
        evidence.append(
            f"Small-gradient condition has persisted for "
            f"{self._consecutive_small_gradient} consecutive step(s) (patience {self._patience})."
        )

        if len(self._losses) < 2:
            evidence.append(
                "Insufficient loss history to confirm stagnation; a barren plateau cannot be "
                "confirmed from gradient behavior alone."
            )
            # Bounded below the triggering threshold: gradient collapse without
            # loss confirmation is suspicious but never sufficient on its own.
            confidence = round(0.5 * persistence_ratio, 4)
            return DetectorResult(
                detector_name=self.name,
                triggered=False,
                confidence=confidence,
                evidence=evidence,
                recommendations=[],
            )

        loss_values = self._losses.values()
        improvement = relative_loss_improvement(loss_values)
        loss_stagnant = math.isfinite(improvement) and abs(improvement) < (
            self._loss_improvement_threshold
        )
        evidence.append(
            f"Relative loss improvement over window: {improvement:.3e} "
            f"(stagnation threshold {self._loss_improvement_threshold:.1e})."
        )

        gradient_persisted = self._consecutive_small_gradient >= self._patience
        triggered = gradient_persisted and loss_stagnant

        if triggered:
            # How far below threshold the gradient sits, saturating at 1.0.
            magnitude_ratio = min(1.0, self._gradient_threshold / max(latest_norm, 1e-300))
            confidence = min(1.0, 0.6 + 0.2 * persistence_ratio + 0.2 * magnitude_ratio)
            recommendations.append(
                "Consider stopping the run; inspect circuit initialization, ansatz "
                "expressivity, and qubit/depth scaling before restarting."
            )
        else:
            confidence = round(0.5 * persistence_ratio * (1.0 if loss_stagnant else 0.5), 4)

        return DetectorResult(
            detector_name=self.name,
            triggered=triggered,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
        )

    def reset(self) -> None:
        self._gradient_norms.reset()
        self._gradient_variances.reset()
        self._losses.reset()
        self._consecutive_small_gradient = 0
