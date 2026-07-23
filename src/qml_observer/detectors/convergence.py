"""ConvergenceDetector.

Milestone 4 (Volume VI-3), Issue #28.

Distinguishes *good* convergence from *bad* gradient collapse -- the
blueprint calls this distinction "essential". Both scenarios can look
identical if you only look at the gradient: it shrinks toward zero. The
difference is the loss:

- Good convergence: gradient shrinks *and* the loss has settled at a low
  (good) absolute value -- the model has actually reached a good optimum.
- Barren plateau: gradient shrinks while the loss remains poor/unimproved
  -- see `BarrenPlateauDetector`, which checks loss *stagnation* (no
  improvement), not loss *magnitude*.

This is why `ConvergenceDetector` checks an absolute `loss_threshold`
(the loss has gotten good), whereas `BarrenPlateauDetector` checks
*relative improvement* (the loss isn't moving, regardless of whether it
is good or bad). The two detectors are intentionally looking at different
questions about the same raw metrics.
"""

from __future__ import annotations

import math

from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.statistics.rolling import RollingWindow


class ConvergenceDetector(BaseDetector):
    """Detects sustained, low-loss, small-gradient convergence to a good optimum."""

    name = "convergence"

    def __init__(
        self,
        loss_threshold: float = 1e-3,
        gradient_threshold: float = 1e-4,
        patience: int = 50,
    ):
        """Configure the detector.

        Args:
            loss_threshold: Absolute loss value at or below which the
                run is considered to have reached a "good" optimum.
                Unlike `BarrenPlateauDetector.loss_improvement_threshold`,
                this is a magnitude, not a relative-improvement figure.
                Placeholder default per addendum §3.
            gradient_threshold: Gradient L2-norm at or below which a step
                is considered "small", consistent with having settled
                near a local optimum.
            patience: Number of consecutive steps both conditions must
                hold before this detector triggers.

        Raises:
            ValueError: If `loss_threshold < 0`, `gradient_threshold <= 0`,
                or `patience < 1`.
        """
        if loss_threshold < 0:
            raise ValueError(f"loss_threshold must be >= 0, got {loss_threshold}")
        if gradient_threshold <= 0:
            raise ValueError(f"gradient_threshold must be > 0, got {gradient_threshold}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self._loss_threshold = loss_threshold
        self._gradient_threshold = gradient_threshold
        self._patience = patience

        self._losses = RollingWindow(maxlen=patience)
        self._gradient_norms = RollingWindow(maxlen=patience)
        self._consecutive_converged = 0

    def update(self, event: StepObservation, state: RunState) -> None:
        loss = event.training_event.loss
        grad = event.gradient

        if loss is not None:
            self._losses.append(loss)
        if grad is not None:
            self._gradient_norms.append(grad.norm_l2)

        step_converged = (
            loss is not None
            and math.isfinite(loss)
            and loss <= self._loss_threshold
            and grad is not None
            and math.isfinite(grad.norm_l2)
            and grad.norm_l2 <= self._gradient_threshold
        )
        self._consecutive_converged = self._consecutive_converged + 1 if step_converged else 0

    def diagnose(self) -> DetectorResult:
        if len(self._losses) == 0 or len(self._gradient_norms) == 0:
            return DetectorResult(
                detector_name=self.name,
                triggered=False,
                confidence=0.0,
                evidence=[],
                recommendations=[],
            )

        latest_loss = self._losses.values()[-1]
        latest_norm = self._gradient_norms.values()[-1]
        persistence_ratio = min(self._consecutive_converged / self._patience, 1.0)

        evidence = [
            f"Latest loss: {latest_loss:.3e} (convergence threshold {self._loss_threshold:.1e}).",
            f"Latest gradient norm: {latest_norm:.3e} (threshold {self._gradient_threshold:.1e}).",
            f"Converged condition has persisted for {self._consecutive_converged} "
            f"consecutive step(s) (patience {self._patience}).",
        ]

        triggered = self._consecutive_converged >= self._patience
        if triggered:
            confidence = min(1.0, 0.7 + 0.3 * persistence_ratio)
            recommendations = [
                "Training appears to have converged to a good optimum; "
                "safe to stop and evaluate the final model."
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
        self._losses.reset()
        self._gradient_norms.reset()
        self._consecutive_converged = 0
