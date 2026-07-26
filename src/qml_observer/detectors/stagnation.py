"""StagnationDetector.

Milestone 4 (Volume VI-2), Issue #27.

Detects an "effectively frozen" optimizer: loss not changing, parameters
not changing, and/or a zero learning rate -- as distinct from
`BarrenPlateauDetector`, which specifically requires a *collapsed
gradient* coincident with loss stagnation. Stagnation can occur without a
collapsed gradient at all (e.g. `learning_rate=0`, or an optimizer stuck
on a numerical issue unrelated to gradient magnitude), so this detector
never inspects gradient values -- only loss, parameters, and optimizer
metadata.
"""

from __future__ import annotations

import math

import numpy as np

from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.statistics.loss import relative_loss_improvement
from qml_observer.statistics.rolling import RollingWindow

#: Parameter-movement magnitude at or below which two consecutive
#: parameter snapshots are considered "unchanged". Not exposed as a
#: constructor argument (unlike the blueprint's other thresholds) because
#: it operates on parameter-vector L2 distance, a different scale from
#: `loss_threshold`; kept as an internal, documented constant instead of
#: a placeholder public parameter with no principled default.
_PARAMETER_FROZEN_EPSILON = 1e-12


class StagnationDetector(BaseDetector):
    """Detects a training run whose optimizer has effectively stopped moving."""

    name = "stagnation"

    def __init__(self, loss_threshold: float = 1e-6, patience: int = 100):
        """Configure the detector.

        Args:
            loss_threshold: Relative loss-improvement magnitude below
                which the loss is considered stagnant over the current
                window (passed to `relative_loss_improvement`).
                Placeholder default per addendum §3.
            patience: Size of the rolling window (in steps) over which
                loss stagnation and parameter movement are assessed.

        Raises:
            ValueError: If `loss_threshold < 0` or `patience < 1`.
        """
        if loss_threshold < 0:
            raise ValueError(f"loss_threshold must be >= 0, got {loss_threshold}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self._loss_threshold = loss_threshold
        self._patience = patience

        self._losses = RollingWindow(maxlen=patience)
        self._param_deltas = RollingWindow(maxlen=patience)
        self._last_parameters: np.ndarray | None = None
        self._latest_learning_rate: float | None = None

    def update(self, event: StepObservation, state: RunState) -> None:
        loss = event.training_event.loss
        if loss is not None:
            self._losses.append(loss)

        if event.optimizer is not None:
            self._latest_learning_rate = event.optimizer.learning_rate

        if event.parameters is not None:
            try:
                current = np.asarray(event.parameters, dtype=float)
            except (TypeError, ValueError):
                current = None

            if current is not None:
                if (
                    self._last_parameters is not None
                    and current.shape == self._last_parameters.shape
                ):
                    delta = float(np.linalg.norm(current - self._last_parameters))
                    self._param_deltas.append(delta)
                self._last_parameters = current

    def diagnose(self) -> DetectorResult:
        evidence: list[str] = []

        have_loss = len(self._losses) >= 2
        have_params = len(self._param_deltas) > 0
        frozen_optimizer = self._latest_learning_rate == 0.0

        if not have_loss and not have_params and not frozen_optimizer:
            return DetectorResult(
                detector_name=self.name,
                triggered=False,
                confidence=0.0,
                evidence=[],
                recommendations=[],
            )

        window_full = len(self._losses) >= self._patience if have_loss else False
        loss_stagnant = False
        slope_confirms_stagnant = False
        if have_loss:
            improvement = relative_loss_improvement(self._losses.values())
            loss_stagnant = math.isfinite(improvement) and abs(improvement) < self._loss_threshold
            evidence.append(
                f"Relative loss improvement over window: {improvement:.3e} "
                f"(stagnation threshold {self._loss_threshold:.1e})."
            )
            # `relative_loss_improvement` compares only the window's first
            # and last value, which a single noisy sample at either end can
            # distort. That's an acceptable trade-off when parameter data
            # independently confirms "frozen", but on its own it is not
            # robust enough to trigger a diagnosis -- e.g. `noise_dominated`
            # fixture runs occasionally showed an 8% false-trigger rate
            # under this endpoint check alone (Milestone 7 calibration,
            # `docs/research/validation.md`). The least-squares `slope()`
            # uses every point in the window, so it is far less sensitive
            # to any single noisy endpoint; requiring it to *also* imply a
            # comparably small fractional change confirms the endpoint
            # reading reflects the whole window's trend, not one outlier.
            slope = self._losses.slope()
            mean_loss = self._losses.mean()
            if slope is not None and math.isfinite(slope) and mean_loss:
                implied_change = abs(slope) * (len(self._losses) - 1) / abs(mean_loss)
                slope_confirms_stagnant = implied_change < self._loss_threshold
        else:
            evidence.append("Insufficient loss history to assess stagnation.")

        params_frozen = False
        if have_params:
            max_delta = max(self._param_deltas.values())
            params_frozen = max_delta <= _PARAMETER_FROZEN_EPSILON
            evidence.append(
                f"Largest parameter-vector movement over window: {max_delta:.3e} "
                f"(frozen threshold {_PARAMETER_FROZEN_EPSILON:.1e})."
            )
        else:
            evidence.append(
                "No parameter data recorded this window; stagnation is being "
                "assessed from loss alone, confirmed against both the "
                "window's endpoint change and its overall least-squares "
                "trend (pass `parameters=` to `monitor.update()` for a "
                "stronger, three-signal confirmation)."
            )

        if frozen_optimizer:
            evidence.append("Optimizer learning_rate is 0.0 (effectively frozen).")

        # A window-length's worth of persistence is required for the loss/
        # parameter signals (mirrors other detectors' patience semantics via
        # window sizing); a directly-observed zero learning rate is treated
        # as sufficient evidence on its own since it is a configuration fact,
        # not a noisy trend that needs to "persist" to be believed.
        #
        # `parameters` is an optional argument to `monitor.update()` -- most
        # integrations (see `adapters.generic`/quickstart examples) only
        # ever pass `loss`/`gradients`. Requiring *both* loss stagnation
        # *and* confirmed-frozen parameters would mean a run that genuinely
        # never improves is silently reported as healthy whenever the
        # caller doesn't happen to also track parameters -- exactly the
        # kind of caller-configuration-dependent false negative the
        # blueprint's "loss not changing / parameters not changing /
        # optimizer effectively frozen" list treats as independent signals,
        # not a single combined one. So: if parameter movement was actually
        # observed and found *not* frozen, that positively contradicts
        # "stagnant" and blocks the trigger; if parameters were never
        # provided, loss stagnation is sufficient *only* once also
        # confirmed by the slope check above, to avoid a single noisy
        # endpoint sample false-triggering on an otherwise-healthy noisy
        # run (see `noise_dominated` case above).
        loss_signal = window_full and loss_stagnant and (have_params or slope_confirms_stagnant)
        param_signal_or_absent = params_frozen or not have_params
        triggered = frozen_optimizer or (loss_signal and param_signal_or_absent)

        persistence_ratio = min(len(self._losses) / self._patience, 1.0) if have_loss else 0.0

        if triggered:
            confidence = 1.0 if frozen_optimizer else min(1.0, 0.7 + 0.3 * persistence_ratio)
            recommendations = [
                "Optimizer appears frozen; check learning-rate schedule, "
                "optimizer state, and whether gradients are reaching the optimizer at all."
            ]
        else:
            signals_present = sum([loss_stagnant, params_frozen])
            confidence = round(0.3 * signals_present * persistence_ratio, 4)
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
        self._param_deltas.reset()
        self._last_parameters = None
        self._latest_learning_rate = None
